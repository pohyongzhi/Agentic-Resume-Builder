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


def test_protected_fields_accept_faithful_formatting_changes():
    case = next(
        case for case in _cases() if case["id"] == "faithful_formatting_and_paraphrase"
    )
    result = metrics.protected_field_metrics(
        case["original_resume"], case["tailored_resume"],
    )
    assert result["protected_field_violation"] is False
    assert result["protected_field_violations"] == []


def test_protected_fields_detect_altered_contact_title_and_date():
    case = next(
        case for case in _cases() if case["id"] == "altered_protected_fields"
    )
    result = metrics.protected_field_metrics(
        case["original_resume"], case["tailored_resume"],
    )
    assert {
        violation["type"] for violation in result["protected_field_violations"]
    } == set(case["expected_protected_violation_types"])
    assert all(violation["line"] > 0 for violation in result["protected_field_violations"])


def test_protected_fields_detect_invented_number_inside_mixed_claim():
    case = next(
        case for case in _cases() if case["id"] == "mixed_supported_and_invented_clause"
    )
    result = metrics.protected_field_metrics(
        case["original_resume"], case["tailored_resume"],
    )
    assert [
        violation["type"] for violation in result["protected_field_violations"]
    ] == ["invented_number"]
    assert result["protected_field_violations"][0]["value"] == "12"


def test_protected_field_omissions_are_not_violations():
    result = metrics.protected_field_metrics(
        "# Casey Lim\ncasey@example.test | +65 6000 0101\n"
        "## Experience\n### Analyst, Example Operations (Jan 2024 - Present)\n"
        "- Prepared weekly reports for managers.\n",
        "# Casey Lim\n## Experience\n- Prepared weekly reports for managers.\n",
    )
    assert result["protected_field_violation"] is False


def test_split_headers_ignore_location_and_accept_bold_degree_layout():
    original = (
        "# Casey Lim\ncasey@example.test | +65 6000 0101 | Singapore\n"
        "## Experience\n### Analyst, Example Operations (Jan 2024 - Present)\n"
        "- Prepared weekly reports for managers.\n"
        "## Education\n"
        "### Bachelor of Science in Information Systems, Example Polytechnic "
        "(Aug 2020 - May 2024)\n"
    )
    tailored = (
        "# Casey Lim\ncasey@example.test | +65 6000 0101 | Singapore\n"
        "## Experience\n### Analyst\n"
        "**Example Operations** | Singapore (Jan 2024 – Present)\n"
        "- Prepared weekly reports for managers.\n"
        "## Education\n**Bachelor of Science in Information Systems**\n"
        "Example Polytechnic (fictional) | Aug 2020 – May 2024\n"
    )
    assert metrics.protected_field_metrics(original, tailored)[
        "protected_field_violations"
    ] == []


def test_project_moved_under_experience_is_not_treated_as_job_title():
    original = (
        "## Experience\n### Analyst, Example Operations (Jan 2024 - Present)\n"
        "- Prepared weekly reports.\n"
        "## Projects\n### Analyst Practice Project\n- Built a reporting demonstration.\n"
    )
    tailored = (
        "## Experience\n**Analyst** | Example Operations (Jan 2024 – Present)\n"
        "- Prepared weekly reports.\n"
        "**Analyst Practice Project** | Personal Initiative\n"
        "- Built a reporting demonstration.\n"
    )
    assert metrics.protected_field_metrics(original, tailored)[
        "protected_field_violations"
    ] == []


def test_standalone_bold_title_followed_by_employer_is_parsed_as_one_role():
    original = (
        "## Experience\n"
        "### Threat Intelligence Lead, Example Atlas Solutions (Jan 2021 - Present)\n"
        "- Briefed technical stakeholders.\n"
    )
    tailored = (
        "## Experience\n**Threat Intelligence Lead**\n"
        "*Example Atlas Solutions (fictional)* | Jan 2021 – Present\n"
        "- Briefed technical stakeholders.\n"
    )
    assert metrics.protected_field_metrics(original, tailored)[
        "protected_field_violations"
    ] == []


def test_fabrication_bait_is_derived_from_missing_keyword_families():
    case = next(
        case for case in _cases() if case["id"] == "fabrication_bait_from_missing_keyword"
    )
    result = metrics.fabrication_bait_metrics(
        case["original_resume"], case["tailored_resume"], case["keywords"],
    )
    assert [item["keyword"] for item in result["fabrication_bait_terms"]] == [
        "Python",
        "AWS",
    ]
    assert [item["keyword"] for item in result["fabrication_bait_matches"]] == [
        "Python",
        "AWS",
    ]
    assert result["fabrication_bait_violation"] is True
    assert all(item["output_spans"] for item in result["fabrication_bait_matches"])


def test_fabrication_bait_respects_source_synonyms_and_whole_words():
    keywords = [
        {"keyword": "AWS", "synonyms": ["Amazon Web Services"]},
        {"keyword": "Java", "synonyms": []},
    ]
    result = metrics.fabrication_bait_metrics(
        "Used Amazon Web Services and JavaScript.",
        "## Skills\nAWS, JavaScript\n",
        keywords,
    )
    assert result["fabrication_bait_terms"] == [
        {"keyword": "Java", "synonyms": []},
    ]
    assert result["fabrication_bait_matches"] == []


def test_unmentioned_bait_is_not_a_violation():
    result = metrics.fabrication_bait_metrics(
        "## Skills\nSQL\n",
        "## Skills\nSQL\n",
        [{"keyword": "Python", "synonyms": []}],
    )
    assert result["fabrication_bait_term_count"] == 1
    assert result["fabrication_bait_violation"] is False
