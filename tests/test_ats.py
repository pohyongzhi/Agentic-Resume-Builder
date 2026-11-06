import pytest

from eval.ats import (
    CallableATSProvider,
    OfflineATSProvider,
    ats_identity,
    comparable_ats_records,
    evaluate_ats,
    offline_proxy_config_hash,
    summarise_ats,
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


def test_generation_failure_is_failed_and_not_scored():
    result = evaluate_ats(
        OfflineATSProvider(), ORIGINAL, "Role", None, KEYWORDS, generation_error="timeout"
    )

    assert result["ats_status"] == "failed"
    assert result["ats_proxy_score"] is None
    assert result["ats_failure_reason"].startswith("generation_error")


def test_identity_hash_is_stable_and_covers_scoring_rules():
    first = ats_identity(OfflineATSProvider())
    second = ats_identity(OfflineATSProvider())
    changed = ats_identity(OfflineATSProvider(version="offline-ats-proxy-test"))

    assert first == second
    assert first["provider_config_hash"] == offline_proxy_config_hash()
    assert first["ats_identity_hash"] != changed["ats_identity_hash"]
    record = evaluate_ats(OfflineATSProvider(), ORIGINAL, "Role", TAILORED, KEYWORDS)
    assert record["ats_identity_hash"] == first["ats_identity_hash"]


def test_bold_headings_are_recognised():
    tailored = "**EXPERIENCE**\n- Python\n**Education:**\n- BSc\n__Skills__\n- Python\n"
    result = evaluate_ats(OfflineATSProvider(), ORIGINAL, "Role", tailored, KEYWORDS)

    assert result["ats_components"]["section_completeness"] == 1.0


def test_summary_denominators_exclude_failures_from_mean():
    provider = OfflineATSProvider()
    records = [
        {"workflow": "w1", **evaluate_ats(provider, ORIGINAL, "Role", TAILORED, KEYWORDS)},
        {"workflow": "w1", **evaluate_ats(provider, ORIGINAL, "Role", "", KEYWORDS)},
        {"workflow": "w2", **evaluate_ats(
            provider, ORIGINAL, "Role", None, KEYWORDS, generation_error="boom"
        )},
    ]

    summary = summarise_ats(records)

    overall = summary["overall"]
    assert (overall["eligible"], overall["scored"], overall["unscored"], overall["failed"]) == (
        3, 1, 1, 1
    )
    assert overall["mean_score"] == records[0]["ats_proxy_score"]
    w2 = next(group for group in summary["groups"] if group["workflow"] == "w2")
    assert w2["scored"] == 0
    assert w2["mean_score"] is None
    assert summary["identity"]["provider_version"] == "offline-ats-proxy-v1"


def test_summary_rejects_mixed_scoring_contracts():
    scorer = lambda **_kwargs: {"ats_vendor_score": 0.5}
    records = [
        evaluate_ats(CallableATSProvider("fake", "v1", scorer), ORIGINAL, "Role", TAILORED),
        evaluate_ats(OfflineATSProvider(), ORIGINAL, "Role", TAILORED),
    ]

    with pytest.raises(ValueError, match="more than one scoring contract"):
        summarise_ats(records)
