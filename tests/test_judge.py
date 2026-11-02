import json
from types import SimpleNamespace

from eval.judge import JudgeConfig, evaluate_judge


def _valid_response(quote="Python"):
    dimension = {
        "score": 4,
        "explanation": "Clear and relevant.",
        "quotations": [{"source": "original_resume", "quote": quote}],
    }
    return json.dumps({
        "relevance_to_jd": dimension,
        "prioritisation_of_supported_experience": dimension,
        "clarity": dimension,
        "concision": dimension,
        "organisation_and_readability": dimension,
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
        "temperature": 0,
        "seed": 42,
        "num_ctx": 1024,
        "think": False,
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
