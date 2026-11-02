import json
from types import SimpleNamespace

import pytest

from eval.judge import JudgeConfig, evaluate_judge


def _valid_response(quote=None):
    tailored = quote or "Python developer"
    tailored_dimension = {
        "score": 4,
        "explanation": "Clear and relevant.",
        "quotations": [{"source": "tailored_resume", "quote": tailored}],
    }
    return json.dumps({
        "relevance_to_jd": {
            **tailored_dimension,
            "quotations": [
                {"source": "job_description", "quote": "Need Python"},
                {"source": "tailored_resume", "quote": tailored},
            ],
        },
        "prioritisation_of_supported_experience": {
            **tailored_dimension,
            "quotations": [
                {"source": "original_resume", "quote": "Python developer"},
                {"source": "tailored_resume", "quote": tailored},
            ],
        },
        "clarity": tailored_dimension,
        "concision": tailored_dimension,
        "organisation_and_readability": tailored_dimension,
    })


class FakeClient:
    def __init__(self, responses, digest="digest-a"):
        self.responses = list(responses)
        self.digest = digest
        self.calls = 0

    def list(self):
        return SimpleNamespace(models=[
            SimpleNamespace(model="gemma-test", digest=self.digest)
        ])

    def show(self, model):
        return SimpleNamespace(modelinfo={
            "general.architecture": "gemma",
            "general.parameter_count": 3_000_000_000,
        })

    def chat(self, **kwargs):
        self.calls += 1
        content = self.responses.pop(0)
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            prompt_eval_count=10,
            eval_count=20,
        )


def _config(**changes):
    values = {
        "model": "gemma-test",
        "provider": "ollama",
        "temperature": 0,
        "seed": 42,
        "num_ctx": 1024,
        "think": False,
        "reasoning_effort": "medium",
        "max_completion_tokens": 2048,
        "min_request_interval_seconds": 0,
        "max_retries": 1,
        "prompt_version": "test-v1",
        "schema_version": "1",
    }
    values.update(changes)
    return JudgeConfig(**values)


def test_judge_validates_scores_quotes_and_calculates_mean(tmp_path):
    client = FakeClient([_valid_response()])
    record = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), client,
    )
    assert record["status"] == "scored"
    assert record["result"]["overall_mean"] == 4
    assert client.calls == 1


def test_judge_retries_invalid_quote_then_scores(tmp_path):
    client = FakeClient([_valid_response("invented"), _valid_response()])
    record = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), client,
    )
    assert record["status"] == "scored"
    assert len(record["attempts"]) == 2
    assert record["attempts"][0]["status"] == "invalid"


def test_judge_bounded_failures_are_unscored_not_zero(tmp_path):
    client = FakeClient(["{}", "{}"])
    record = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), client,
    )
    assert record["status"] == "unscored"
    assert record["result"] is None
    assert len(record["attempts"]) == 2


def test_judge_cache_and_input_model_invalidation(tmp_path):
    first_client = FakeClient([_valid_response()])
    first = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), first_client,
    )
    cached_client = FakeClient([])
    cached = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), cached_client,
    )
    assert cached["cache_key"] == first["cache_key"]
    assert cached_client.calls == 0

    changed_input = evaluate_judge(
        "Python and SQL developer", "Need Python", "Python developer",
        tmp_path, _config(), FakeClient([_valid_response()]),
    )
    changed_model = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _config(), FakeClient([_valid_response()], digest="digest-b"),
    )
    assert changed_input["cache_key"] != first["cache_key"]
    assert changed_model["cache_key"] != first["cache_key"]


class FakeGroqCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            id="chatcmpl-test",
            model="openai/gpt-oss-120b",
            system_fingerprint="fp-test",
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
            usage=SimpleNamespace(
                prompt_tokens=111,
                completion_tokens=222,
                total_tokens=333,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=123),
            ),
        )


class FakeGroqClient:
    def __init__(self, responses):
        self.completions = FakeGroqCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _groq_config(**changes):
    values = {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "temperature": 0,
        "seed": 42,
        "num_ctx": 16384,
        "think": False,
        "reasoning_effort": "medium",
        "max_completion_tokens": 2048,
        "min_request_interval_seconds": 0,
        "max_retries": 0,
        "prompt_version": "test-groq-v1",
        "schema_version": "1",
    }
    values.update(changes)
    return JudgeConfig(**values)


def test_groq_uses_strict_schema_and_records_exact_usage(tmp_path):
    client = FakeGroqClient([_valid_response()])
    record = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _groq_config(), client,
    )
    request = client.completions.calls[0]
    assert request["model"] == "openai/gpt-oss-120b"
    assert request["reasoning_effort"] == "medium"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in request
    assert record["attempts"][0]["response_id"] == "chatcmpl-test"
    assert record["attempts"][0]["returned_model"] == "openai/gpt-oss-120b"
    assert record["attempts"][0]["prompt_tokens"] == 111
    assert record["attempts"][0]["output_tokens"] == 222
    assert record["attempts"][0]["reasoning_tokens"] == 123
    assert record["attempts"][0]["total_tokens"] == 333


def test_groq_failure_redacts_api_key(tmp_path, monkeypatch):
    fake_key = "gsk_" + "test_secret_value"
    monkeypatch.setenv("GROQ_API_KEY", fake_key)
    client = FakeGroqClient([RuntimeError(f"rejected {fake_key}")])
    record = evaluate_judge(
        "Python developer", "Need Python", "Python developer",
        tmp_path, _groq_config(), client,
    )
    assert record["status"] == "unscored"
    serialized = json.dumps(record)
    assert fake_key not in serialized
    assert "[REDACTED]" in serialized


def test_groq_requires_environment_key_without_injected_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY_MISSING"):
        evaluate_judge(
            "Python developer", "Need Python", "Python developer",
            tmp_path, _groq_config(),
        )
