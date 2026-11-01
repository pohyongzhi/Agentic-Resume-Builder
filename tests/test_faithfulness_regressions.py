import json
from pathlib import Path

import pytest

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


@pytest.mark.xfail(
    strict=True,
    reason="F2 must stop treating protected-field Markdown headers as semantic claims",
)
def test_claim_extraction_ignores_protected_field_headers():
    case = next(
        case for case in _cases() if case["id"] == "faithful_formatting_and_paraphrase"
    )
    assert metrics.extract_claims(case["tailored_resume"]) == case["expected_semantic_claims"]
