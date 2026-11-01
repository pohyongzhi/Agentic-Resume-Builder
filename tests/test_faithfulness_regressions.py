import json
from pathlib import Path

from eval import metrics


FIXTURES = Path(__file__).parent / "fixtures" / "faithfulness_regressions.json"


def _cases():
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_regression_fixture_covers_required_failure_modes():
    cases = {case["id"]: case for case in _cases()}
    assert set(cases) == {
        "faithful_formatting_and_paraphrase",
        "mixed_supported_and_invented_clause",
        "altered_protected_fields",
        "fabrication_bait_from_missing_keyword",
    }
    assert cases["mixed_supported_and_invented_clause"]["expected_unsupported_claims"]
    assert set(cases["altered_protected_fields"]["expected_protected_violation_types"]) == {
        "contact_phone",
        "employment_title",
        "employment_date",
    }
    assert cases["fabrication_bait_from_missing_keyword"]["expected_bait_terms"] == [
        "Python",
        "AWS",
    ]


def test_claim_extraction_ignores_protected_field_headers():
    case = next(
        case for case in _cases() if case["id"] == "faithful_formatting_and_paraphrase"
    )
    assert metrics.extract_claims(case["tailored_resume"]) == case["expected_semantic_claims"]


def test_normalization_ignores_formatting_dashes_and_fictional_markers():
    source = "**Example Atlas Solutions (fictional)** (Jan 2025 - Present)"
    output = "**Example Atlas Solutions** (Jan 2025 – Present)"
    assert metrics.normalize_evidence_text(source) == metrics.normalize_evidence_text(output)


def test_short_factual_bullets_are_not_silently_dropped():
    assert metrics.extract_claims("## Experience\n- Built REST APIs.\n") == ["Built REST APIs."]
    assert metrics.extract_claims("## Experience\n- Processed 12345678 records.\n") == [
        "Processed 12345678 records."
    ]


def test_source_evidence_units_preserve_structure_and_split_skills():
    case = next(
        case for case in _cases() if case["id"] == "faithful_formatting_and_paraphrase"
    )
    units = metrics.extract_evidence_units(case["original_resume"])
    assert [unit["evidence_id"] for unit in units] == [
        f"evidence-{index:04d}" for index in range(1, len(units) + 1)
    ]
    assert any(unit["kind"] == "contact" for unit in units)
    assert any(unit["kind"] == "protected_header" for unit in units)
    assert any("improving team turnaround by 35%" in unit["text"] for unit in units)

    skills = metrics.extract_evidence_units("## Skills\nSQL, Excel; Power BI\n")
    assert [(unit["kind"], unit["text"]) for unit in skills] == [
        ("skill", "SQL"),
        ("skill", "Excel"),
        ("skill", "Power BI"),
    ]


def test_retrieval_finds_the_supported_source_bullet_and_penalizes_number_mismatch():
    case = next(
        case for case in _cases() if case["id"] == "faithful_formatting_and_paraphrase"
    )
    units = metrics.extract_evidence_units(case["original_resume"])
    claim = case["expected_semantic_claims"][0]
    matches = metrics.retrieve_evidence_units(claim, units)
    assert "improving team turnaround by 35%" in matches[0]["text"]
    assert all(0 <= match["retrieval_score"] <= 1 for match in matches)

    mismatched = metrics.retrieve_evidence_units(
        claim.replace("35%", "95%"), units,
    )
    assert mismatched[0]["retrieval_score"] < matches[0]["retrieval_score"]
