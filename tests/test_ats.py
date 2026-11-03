from eval.ats import (
    CallableATSProvider,
    OfflineATSProvider,
    comparable_ats_records,
    evaluate_ats,
)


ORIGINAL = """# Ada Example
## Experience
- Built Python services.
## Education
- BSc Computing
## Skills
- Python
"""

TAILORED = """# Ada Example
## Experience
- Built Python services.
## Education
- BSc Computing
## Skills
- Python
"""

KEYWORDS = [
    {"keyword": "Python", "synonyms": []},
    {"keyword": "Kubernetes", "synonyms": ["K8s"]},
]


def test_offline_proxy_is_deterministic_and_versioned():
    provider = OfflineATSProvider()
    first = evaluate_ats(provider, ORIGINAL, "Python role", TAILORED, KEYWORDS)
    second = evaluate_ats(provider, ORIGINAL, "Python role", TAILORED, KEYWORDS)

    assert first["ats_status"] == "scored"
    assert first["provider"] == "offline_reference"
    assert first["score_kind"] == "ats_proxy_score"
    assert first["ats_cache_key"] == second["ats_cache_key"]
    assert first["ats_response_hash"] == second["ats_response_hash"]
    assert first["ats_proxy_score"] == second["ats_proxy_score"]
    assert first["details"]["visual_layout_assessed"] is False


def test_unsupported_keyword_stuffing_does_not_raise_supported_component():
    provider = OfflineATSProvider()
    base = evaluate_ats(provider, ORIGINAL, "Role", TAILORED, KEYWORDS)
    stuffed = evaluate_ats(
        provider, ORIGINAL, "Role", TAILORED + "\nKubernetes Kubernetes", KEYWORDS
    )

    assert stuffed["ats_components"]["keyword_coverage"] > base["ats_components"]["keyword_coverage"]
    assert (
        stuffed["ats_components"]["supported_requirement_coverage"]
        == base["ats_components"]["supported_requirement_coverage"]
    )
    assert stuffed["ats_proxy_score"] == base["ats_proxy_score"]


def test_provider_failure_and_invalid_response_remain_unscored():
    def fail(**_kwargs):
        raise TimeoutError("provider timed out api_key=do-not-log")

    def invalid(**_kwargs):
        return {"ats_vendor_score": 4.2}

    timed_out = evaluate_ats(
        CallableATSProvider("fake", "v1", fail), ORIGINAL, "Role", TAILORED
    )
    malformed = evaluate_ats(
        CallableATSProvider("fake", "v1", invalid), ORIGINAL, "Role", TAILORED
    )

    assert timed_out["ats_status"] == "unscored"
    assert timed_out["ats_vendor_score"] is None
    assert "TimeoutError" in timed_out["ats_failure_reason"]
    assert "do-not-log" not in timed_out["ats_failure_reason"]
    assert malformed["ats_status"] == "unscored"
    assert "between 0 and 1" in malformed["ats_failure_reason"]


def test_comparison_requires_same_provider_contract():
    scorer = lambda **_kwargs: {"ats_vendor_score": 0.8}
    first = evaluate_ats(
        CallableATSProvider("fake", "v1", scorer), ORIGINAL, "Role", TAILORED
    )
    same = evaluate_ats(
        CallableATSProvider("fake", "v1", scorer), ORIGINAL, "Role", TAILORED
    )
    changed = evaluate_ats(
        CallableATSProvider("fake", "v2", scorer), ORIGINAL, "Role", TAILORED
    )

    assert comparable_ats_records(first, same)
    assert not comparable_ats_records(first, changed)


def test_empty_resume_is_unscored_not_zero():
    result = evaluate_ats(OfflineATSProvider(), ORIGINAL, "Role", "", KEYWORDS)

    assert result["ats_status"] == "unscored"
    assert result["ats_proxy_score"] is None
