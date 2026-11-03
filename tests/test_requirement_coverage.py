import json
from types import SimpleNamespace

from eval.requirement_coverage import (
    RequirementCoverageConfig,
    evaluate_requirement_coverage,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)))


def _config(**changes):
    values = {
        "provider": "ollama",
        "model": "coverage-test",
        "max_retries": 0,
        "num_ctx": 1024,
    }
    values.update(changes)
    return RequirementCoverageConfig(**values)


MODEL_INFO = {"provider": "ollama", "tag": "coverage-test", "digest": "digest-a"}


def _support(*decisions):
    return json.dumps({"decisions": list(decisions)})


def _coverage(*decisions):
    return json.dumps({"decisions": list(decisions)})


def test_exact_and_synonym_hits_are_scored_without_a_model(tmp_path):
    result = evaluate_requirement_coverage(
        "Built dashboards with PostgreSQL.",
        "Created reporting backed by PostgreSQL.",
        [{"keyword": "SQL", "synonyms": ["PostgreSQL", "MySQL"]}],
        tmp_path,
        _config(),
        model_info=MODEL_INFO,
    )

    assert result["status"] == "scored"
    assert result["coverage"] == 1
    assert result["supportable_requirement_count"] == 1
    assert result["requirements"][0]["support_method"] == "lexical"
    assert result["requirements"][0]["coverage_method"] == "lexical"


def test_semantic_paraphrase_can_support_and_cover_requirement(tmp_path):
    client = FakeClient([
        _support({
            "requirement_id": "req-001",
            "supported": True,
            "source_quotes": ["Automated cloud infrastructure deployments."],
            "rationale": "This is infrastructure automation.",
        }),
        _coverage({
            "requirement_id": "req-001",
            "covered": True,
            "output_quotes": ["Automated infrastructure rollout in the cloud."],
            "rationale": "The output faithfully paraphrases the evidence.",
        }),
    ])
    result = evaluate_requirement_coverage(
        "Automated cloud infrastructure deployments.",
        "Automated infrastructure rollout in the cloud.",
        [{"keyword": "infrastructure as code", "synonyms": []}],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["coverage"] == 1
    assert result["requirements"][0]["support_method"] == "semantic_judge"
    assert result["requirements"][0]["coverage_method"] == "semantic_judge"
    assert len(client.calls) == 2


def test_unsupported_requirement_is_excluded_from_denominator(tmp_path):
    client = FakeClient([_support({
        "requirement_id": "req-002",
        "supported": False,
        "source_quotes": [],
        "rationale": "No Kubernetes evidence.",
    })])
    result = evaluate_requirement_coverage(
        "Used Python for ETL.",
        "Python ETL developer.",
        [
            {"keyword": "Python", "synonyms": []},
            {"keyword": "Kubernetes", "synonyms": ["K8s"]},
        ],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["supportable_requirement_count"] == 1
    assert result["covered_requirement_count"] == 1
    assert result["coverage"] == 1
    assert result["requirements"][1]["coverage_method"] == "not_applicable"


def test_omitted_supported_requirement_lowers_coverage(tmp_path):
    client = FakeClient([_coverage({
        "requirement_id": "req-001",
        "covered": False,
        "output_quotes": [],
        "rationale": "No database work is expressed.",
    })])
    result = evaluate_requirement_coverage(
        "Built SQL reporting pipelines.",
        "Built weekly reporting workflows.",
        [{"keyword": "SQL", "synonyms": []}],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["coverage"] == 0
    assert result["uncovered_supportable_requirements"] == ["SQL"]


def test_negated_term_is_not_treated_as_lexical_support(tmp_path):
    client = FakeClient([_support({
        "requirement_id": "req-001",
        "supported": False,
        "source_quotes": [],
        "rationale": "The resume explicitly denies AWS experience.",
    })])
    result = evaluate_requirement_coverage(
        "No AWS experience.",
        "General software developer.",
        [{"keyword": "AWS", "synonyms": []}],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["status"] == "no_supportable_requirements"
    assert result["coverage"] is None


def test_zero_supportable_denominator_is_none_not_zero(tmp_path):
    client = FakeClient([_support({
        "requirement_id": "req-001",
        "supported": False,
        "source_quotes": [],
        "rationale": "No evidence.",
    })])
    result = evaluate_requirement_coverage(
        "Customer service experience.",
        "Customer service experience.",
        [{"keyword": "Rust", "synonyms": []}],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["status"] == "no_supportable_requirements"
    assert result["coverage"] is None
    assert result["supportable_requirement_count"] == 0


def test_invalid_semantic_quote_is_unscored(tmp_path):
    client = FakeClient([_support({
        "requirement_id": "req-001",
        "supported": True,
        "source_quotes": ["invented evidence"],
        "rationale": "Claimed support.",
    })])
    result = evaluate_requirement_coverage(
        "Worked on deployment automation.",
        "Worked on deployment automation.",
        [{"keyword": "Terraform", "synonyms": []}],
        tmp_path,
        _config(),
        client=client,
        model_info=MODEL_INFO,
    )

    assert result["status"] == "unscored"
    assert result["coverage"] is None
    assert "unverifiable quotation" in result["failure_reason"]


def test_supportability_cache_is_reused_across_workflow_outputs(tmp_path):
    first_client = FakeClient([
        _support({
            "requirement_id": "req-001",
            "supported": True,
            "source_quotes": ["Automated cloud deployments."],
            "rationale": "Relevant automation evidence.",
        }),
        _coverage({
            "requirement_id": "req-001",
            "covered": False,
            "output_quotes": [],
            "rationale": "Not expressed.",
        }),
    ])
    args = (
        "Automated cloud deployments.",
        "General engineer.",
        [{"keyword": "infrastructure automation", "synonyms": []}],
        tmp_path,
        _config(),
    )
    evaluate_requirement_coverage(*args, client=first_client, model_info=MODEL_INFO)

    second_client = FakeClient([_coverage({
        "requirement_id": "req-001",
        "covered": True,
        "output_quotes": ["Automated infrastructure provisioning."],
        "rationale": "Expressed semantically.",
    })])
    second = evaluate_requirement_coverage(
        args[0],
        "Automated infrastructure provisioning.",
        args[2],
        tmp_path,
        _config(),
        client=second_client,
        model_info=MODEL_INFO,
    )

    assert second["coverage"] == 1
    assert len(first_client.calls) == 2
    assert len(second_client.calls) == 1



def _strict_schema_violations(schema, path="$"):
    """Hosted strict JSON mode requires every property to be listed in `required`."""
    problems = []
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            missing = set(props) - set(schema.get("required", []))
            if missing:
                problems.append(f"{path}: {sorted(missing)}")
        for key, value in schema.items():
            problems.extend(_strict_schema_violations(value, f"{path}/{key}"))
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            problems.extend(_strict_schema_violations(value, f"{path}[{i}]"))
    return problems


def test_hosted_schemas_list_every_property_as_required():
    from eval.requirement_coverage import CoverageResponse, SupportResponse

    for model in (SupportResponse, CoverageResponse):
        assert _strict_schema_violations(model.model_json_schema()) == []
