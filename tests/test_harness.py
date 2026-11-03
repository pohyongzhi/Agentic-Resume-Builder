import os

import pytest

from common.harness import PathNotAllowed, ToolHarness, safe_output_path, to_markdown, validate_resume


# ---------- validate_resume ----------

def test_valid_resume_passes(candidate, good_resume):
    assert validate_resume(good_resume, candidate) == []


def test_invented_employer_is_caught(candidate, good_resume):
    good_resume["experience"][0]["employer"] = "Google"
    errors = validate_resume(good_resume, candidate)
    assert any("not in the original resume" in e for e in errors)


def test_changed_date_is_caught(candidate, good_resume):
    good_resume["experience"][0]["start"] = "Jan 2024"
    errors = validate_resume(good_resume, candidate)
    assert any("start date 'Jan 2024' should be 'May 2025'" in e for e in errors)


def test_invented_skill_is_caught(candidate, good_resume):
    good_resume["skills"].append("Airflow")
    errors = validate_resume(good_resume, candidate)
    assert errors == ["skill 'Airflow' is not in the original resume"]


def test_invented_degree_is_caught(candidate, good_resume):
    good_resume["education"][0]["degree"] = "Master of Science (Computer Science)"
    assert any("education" in e for e in validate_resume(good_resume, candidate))


def test_missing_required_fields(candidate):
    errors = validate_resume({"name": "Alex Tan"}, candidate)
    assert "contact needs an email or phone number" in errors
    assert "at least one experience or education entry is required" in errors


def test_schema_error_reported(candidate):
    errors = validate_resume({"contact": {}}, candidate)
    assert any(e.startswith("schema: name") for e in errors)


def test_entry_missed_by_parser_accepted_from_raw_text(candidate, good_resume):
    candidate["education"] = []
    assert validate_resume(good_resume, candidate) == []


def test_entry_missed_by_parser_with_wrong_date_rejected(candidate, good_resume):
    candidate["education"] = []
    good_resume["education"][0]["end"] = "2024"
    assert any("education" in e for e in validate_resume(good_resume, candidate))


def test_parser_mistake_caught_by_raw_text(candidate, good_resume):
    # parser wrongly invented an employer; it is in candidate_data but not in the original text
    candidate["experience"].append({"employer": "Acme Corp", "title": "Analyst", "start": "2023", "end": "2024"})
    good_resume["experience"].append({"employer": "Acme Corp", "title": "Analyst", "start": "2023", "end": "2024"})
    errors = validate_resume(good_resume, candidate)
    assert any("does not appear in the original resume text" in e for e in errors)


# ---------- render / paths ----------

def test_template_has_entry_headings(good_resume):
    md = to_markdown(good_resume)
    assert md.startswith("# Alex Tan\n")
    assert "### Business Analyst Intern, Example Logistics Pte Ltd (May 2025 - Aug 2025)" in md
    assert "## Skills\nSQL, Power BI, Excel, Python" in md


@pytest.mark.parametrize("name", ["../escape.md", "../../etc/passwd", "/tmp/abs.md", "sub/../../x.md"])
def test_path_traversal_refused(tmp_path, name):
    with pytest.raises(PathNotAllowed):
        safe_output_path(tmp_path / "out", name)


def test_original_resume_cannot_be_overwritten(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    original = out / "resume.md"
    original.write_text("original")
    with pytest.raises(PathNotAllowed):
        safe_output_path(out, "resume.md", resume_path=str(original))
    assert original.read_text() == "original"


def test_existing_file_not_overwritten(tmp_path):
    (tmp_path / "taken.md").write_text("x")
    with pytest.raises(PathNotAllowed):
        safe_output_path(tmp_path, "taken.md")


# ---------- harness rules ----------

def make_harness(candidate, tmp_path, max_calls=6):
    return ToolHarness(candidate, resume_path=None, output_dir=str(tmp_path), max_calls=max_calls, run_id="t")


def test_render_requires_validation(candidate, tmp_path):
    h = make_harness(candidate, tmp_path)
    ok, msg, stop = h.execute("render_resume", {})
    assert not ok and not stop and "requires" in msg
    assert os.listdir(tmp_path) == []


def test_render_refused_after_failed_revalidation(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path)
    assert h.execute("validate_resume", {"resume": good_resume})[0]
    bad = dict(good_resume, skills=["Airflow"])
    h.execute("validate_resume", {"resume": bad})
    ok, msg, _ = h.execute("render_resume", {})
    assert not ok and os.listdir(tmp_path) == []


def test_validate_then_render_saves_inside_output_dir(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path)
    ok, msg, stop = h.execute("validate_resume", {"resume": good_resume})
    assert ok and '"valid": true' in msg and not stop
    ok, msg, stop = h.execute("render_resume", {})
    assert ok and stop
    out = h.outcome()
    assert os.path.dirname(out["output_path"]) == str(tmp_path.resolve())
    assert "### Business Analyst Intern" in out["resume_md"]


def test_missing_name_and_contact_filled_from_candidate(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path)
    draft = {k: v for k, v in good_resume.items() if k not in ("name", "contact")}
    ok, msg, _ = h.execute("validate_resume", {"resume": draft})
    assert ok and '"valid": true' in msg
    assert h.execute("render_resume", {})[0]
    assert h.outcome()["resume_md"].startswith("# Alex Tan\nalex.tan@example.com")


def test_blank_structured_contact_filled_from_candidate(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path)
    draft = dict(good_resume, contact={"email": "", "phone": "", "location": "", "links": []})
    ok, msg, _ = h.execute("validate_resume", {"resume": draft})
    assert ok and '"valid": true' in msg
    assert h.draft["contact"]["email"] == candidate["contact"]["email"]


def test_json_string_argument_accepted(candidate, good_resume, tmp_path):
    import json
    h = make_harness(candidate, tmp_path)
    ok, msg, _ = h.execute("validate_resume", {"resume": json.dumps(good_resume)})
    assert ok and '"valid": true' in msg


def test_disallowed_tool_rejected_and_counted(candidate, tmp_path):
    h = make_harness(candidate, tmp_path, max_calls=2)
    ok, msg, stop = h.execute("write_file", {"path": "/etc/passwd"})
    assert not ok and "not allowed" in msg and h.calls == 1


def test_budget_exceeded(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path, max_calls=2)
    bad = dict(good_resume, skills=["Airflow"])
    h.execute("validate_resume", {"resume": bad})
    h.execute("validate_resume", {"resume": bad})
    ok, msg, stop = h.execute("validate_resume", {"resume": good_resume})
    assert not ok and stop and msg.startswith("TOOL_BUDGET_EXCEEDED")
    out = h.outcome()
    assert out["hit_budget"] and out["errors"][0].startswith("TOOL_BUDGET_EXCEEDED")
    assert "Airflow" in out["errors"][0]


def test_exact_budget_without_render_is_exceeded(candidate, good_resume, tmp_path):
    h = make_harness(candidate, tmp_path, max_calls=1)
    bad = dict(good_resume, skills=["Airflow"])
    h.execute("validate_resume", {"resume": bad})
    out = h.outcome()
    assert out["hit_budget"]
    assert out["errors"][0].startswith("TOOL_BUDGET_EXCEEDED")
