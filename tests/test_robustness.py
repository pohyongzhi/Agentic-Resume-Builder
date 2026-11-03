"""Robustness v2 variants, expected-behaviour logic, denominators and stage wiring.

Deterministic and offline: no network, no Ollama, fake workflows and a fake NLI scorer.
"""
import json
import re
import socket
from collections import Counter
from pathlib import Path

import pytest

import config
import evaluate
from common.data import load_pairs
from eval import reporting
from eval import robustness as rb
from eval.judge import JudgeConfig

PAIRS = load_pairs(split="test")
JD_POOL = rb.build_jd_pool(PAIRS)
PAIR = PAIRS[0]
NON_ORIGINAL = [name for name in rb.VARIANTS if name != rb.ORIGINAL]


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network access is not allowed in robustness tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


# --- Variant set ------------------------------------------------------------------------------


def test_variant_families_and_selection():
    assert set(rb.VARIANTS) == {
        "original", "typos", "pdf_extraction", "web_clutter", "formatting",
        "reordered_sections", "plain_text", "heading_aliases", "mismatched_jd", "degenerate_jd",
    }
    assert {rb.variant_scope(name) for name in ("mismatched_jd", "degenerate_jd")} == {
        "out_of_scope"
    }
    assert rb.select_variants() == rb.VARIANTS
    assert rb.select_variants(families=["noisy"]) == (
        "original", "typos", "pdf_extraction", "web_clutter",
    )
    # Out-of-scope runs never need the original, so it costs nothing extra.
    assert rb.select_variants(families=["out_of_scope"]) == ("mismatched_jd", "degenerate_jd")
    assert rb.select_variants(["plain_text"], ["out_of_scope"]) == (
        "original", "plain_text", "mismatched_jd", "degenerate_jd",
    )
    with pytest.raises(ValueError):
        rb.select_variants(["unknown"])


@pytest.mark.parametrize("variant", rb.VARIANTS)
def test_every_variant_preserves_content_on_every_dataset_pair(variant):
    for pair in PAIRS:
        spec = rb.build_variant(pair, variant, 42, JD_POOL)  # raises on any problem
        assert rb.variant_content_problems(pair, spec, JD_POOL) == []
        if variant != rb.ORIGINAL:
            assert (spec["resume"], spec["jd"]) != (pair["resume"], pair["jd"]), pair["pair_id"]


@pytest.mark.parametrize("variant", NON_ORIGINAL)
def test_variants_are_pure_functions_of_pair_and_seed(variant):
    first = rb.build_variant(PAIR, variant, 42, JD_POOL)
    again = rb.build_variant(dict(PAIR), variant, 42, rb.build_jd_pool(list(reversed(PAIRS))))
    assert first == again


def test_seed_changes_seeded_noise():
    for variant in ("typos", "pdf_extraction", "web_clutter"):
        assert rb.build_variant(PAIR, variant, 42) != rb.build_variant(PAIR, variant, 7)


# --- Noisy variants ---------------------------------------------------------------------------


def test_typos_are_low_rate_and_never_touch_protected_or_keyword_tokens():
    keyword_tokens = rb._keyword_tokens(PAIR["keywords"])
    for pair in PAIRS:
        spec = rb.build_variant(pair, "typos", 42, JD_POOL)
        protected = rb._protected_tokens(pair["resume"]) | rb._keyword_tokens(pair["keywords"])
        resume_changes = spec["metadata"]["resume_typos"]
        jd_changes = spec["metadata"]["jd_typos"]
        assert resume_changes and jd_changes
        jd_protected = rb._keyword_tokens(pair["keywords"]) | set(
            re.findall(r"[a-z0-9]+", rb._jd_title(pair["jd"]).lower())
        )
        for changes, guarded in ((resume_changes, protected), (jd_changes, jd_protected)):
            for change in changes:
                assert change["before"].islower() and change["before"] not in guarded
                assert change["after"] != change["before"]
        assert len(resume_changes) <= max(3, 0.1 * len(pair["resume"].split()))
        # Protected fields parse identically from the noisy input.
        assert rb._protected_values(pair["resume"]) == rb._protected_values(spec["resume"])
        assert rb._keyword_hits(pair["resume"], pair["keywords"]) == rb._keyword_hits(
            spec["resume"], pair["keywords"]
        )
        # Header, contact, date and skill lines are byte-identical.
        claim_lines = rb._resume_typo_lines(pair["resume"])
        for number, (old, new) in enumerate(
            zip(pair["resume"].split("\n"), spec["resume"].split("\n")), 1
        ):
            if number not in claim_lines:
                assert old == new
    assert "javascript" in keyword_tokens


def test_typo_check_rejects_a_changed_employer_or_number():
    spec = rb.build_variant(PAIR, "typos", 42)
    employer = spec["resume"].replace("Example Atlas Solutions", "Exmaple Atlas Solutions")
    number = spec["resume"].replace("19%", "91%")
    for resume in (employer, number):
        assert rb.variant_content_problems(PAIR, {**spec, "resume": resume})


def test_pdf_extraction_artefacts_and_repair():
    spec = rb.build_variant(PAIR, "pdf_extraction", 42)
    text = spec["resume"]
    assert spec["resume_suffix"] == ".txt"
    assert "Page 1 of 2" in text and "Page 2 of 2" in text
    assert "Aisha Tan | Curriculum Vitae" in text
    assert "##" not in text and "\n- " not in text
    assert any(len(line) <= 72 for line in text.split("\n"))
    assert max(len(line.rstrip()) for line in text.split("\n")) <= 80
    assert re.search(r"[a-z]-[ \t]*\n", text) or any(
        re.search(r"[a-z]-\s*\n", rb.build_variant(pair, "pdf_extraction", 42)["resume"])
        for pair in PAIRS[:10]
    )
    assert re.search(r"\S  +\S|[ ]+\n", text)
    assert rb.transformation_preserves_content(PAIR["resume"], rb.repair_pdf_extraction(text))
    # Losing a line is caught.
    broken = "\n".join(line for line in text.split("\n") if "19%" not in line)
    assert rb.variant_content_problems(PAIR, {**spec, "resume": broken})


def test_web_clutter_wraps_the_jd_without_adding_requirements():
    spec = rb.build_variant(PAIR, "web_clutter", 42)
    assert PAIR["jd"].strip() in spec["jd"]
    assert "cookies" in spec["jd"] and "equal opportunity employer" in spec["jd"]
    assert spec["jd"].index("cookies") < spec["jd"].index(PAIR["jd"].strip())
    assert spec["resume"] == PAIR["resume"]
    assert rb.variant_content_problems(PAIR, {**spec, "jd": spec["jd"][:500]})
    assert rb.variant_content_problems(PAIR, {**spec, "jd": spec["jd"] + "\nReact\n"})


# --- Format variants --------------------------------------------------------------------------


def test_plain_text_and_heading_aliases():
    plain = rb.build_variant(PAIR, "plain_text", 42)
    assert "EXPERIENCE" in plain["resume"] and "## " not in plain["resume"]
    assert "**" not in plain["jd"] and "• Used JavaScript" in plain["resume"]

    aliased = rb.build_variant(PAIR, "heading_aliases", 42)
    headings = re.findall(r"(?m)^## (.+)$", aliased["resume"])
    assert "Experience" not in headings and len(headings) == 5
    assert any(alias in headings for alias in rb._HEADING_ALIASES["experience"])
    assert rb.variant_content_problems(
        PAIR, {**aliased, "resume": aliased["resume"].replace("19%", "29%")}
    )
    assert rb.variant_content_problems(
        PAIR, {**plain, "resume": plain["resume"].replace("Atlas", "Orbit")}
    )


def test_legacy_variants_are_unchanged():
    pair = {
        "resume": "# A\n\n## Summary\n\nText\n\n## Skills\n\nPython\n",
        "jd": "# Role\n\n## Responsibilities\n\n- Build\n\n## Requirements\n\n- Python\n",
    }
    assert rb.perturb_pair(pair, "formatting")[1].count("* Build") == 1
    reordered, _ = rb.perturb_pair(pair, "reordered_sections")
    assert reordered.index("## Skills") < reordered.index("## Summary")


# --- Out-of-scope variants --------------------------------------------------------------------


def test_mismatched_jd_is_cross_category_and_recorded():
    for pair in PAIRS:
        spec = rb.build_variant(pair, "mismatched_jd", 42, JD_POOL)
        other = JD_POOL[spec["metadata"]["mismatched_jd_id"]]
        assert other["category"] != pair["category"]
        assert spec["jd"] == other["jd"] and spec["resume"] == pair["resume"]
        keywords = {item["keyword"] for item in spec["scoring_keywords"]}
        assert keywords >= {item["keyword"] for item in other["keywords"]}
    same = next(jd for jd in JD_POOL.values() if jd["category"] == PAIR["category"])
    spec = rb.build_variant(PAIR, "mismatched_jd", 42, JD_POOL)
    bad = {**spec, "jd": same["jd"], "metadata": {"mismatched_jd_id": same["jd_id"]}}
    assert rb.variant_content_problems(PAIR, bad, JD_POOL)
    with pytest.raises(ValueError):
        rb.build_variant(PAIR, "mismatched_jd", 42)


def test_degenerate_jd_modes():
    modes = Counter()
    for pair in PAIRS:
        spec = rb.build_variant(pair, "degenerate_jd", 42, JD_POOL)
        mode = spec["metadata"]["degenerate_mode"]
        modes[mode] += 1
        assert len(spec["jd"].split()) <= 40
        if mode == "title_only":
            assert spec["jd"].strip() == rb._jd_title(pair["jd"])
        if mode == "non_job_text":
            assert not rb._mentions_keyword(spec["jd"], pair["keywords"])
    assert set(modes) == set(rb.DEGENERATE_MODES)
    spec = rb.build_variant(PAIR, "degenerate_jd", 42)
    assert rb.variant_content_problems(PAIR, {**spec, "jd": PAIR["jd"]})


# --- Sample and identity ----------------------------------------------------------------------


def test_v2_manifest_is_ordered_deterministic_and_versioned():
    manifest = rb.robustness_manifest(PAIRS, "dataset-a", 42)
    assert manifest == rb.robustness_manifest(list(reversed(PAIRS)), "dataset-a", 42)
    assert manifest["version"] == config.ROBUSTNESS_SAMPLE_VERSION == "robustness-sample-v2"
    assert len(manifest["pair_ids"]) == 20
    assert set(manifest["pair_ids"]) == {
        pair["pair_id"] for pair in rb.select_stratified_pairs(PAIRS, 42)
    }
    first_ten = manifest["strata"][:10]
    assert Counter(entry["category"] for entry in first_ten) == {
        category: 2 for category in {pair["category"] for pair in PAIRS}
    }
    assert {entry["fit_level"] for entry in first_ten} == {"strong", "moderate"}
    assert all(entry["mismatched_category"] != entry["category"] for entry in manifest["strata"])
    assert rb.robustness_manifest_v1(PAIRS, "dataset-a", 42)["version"] == "robustness-sample-v1"


def test_identity_moves_with_variant_version(monkeypatch, tmp_path):
    first = evaluate.robustness_root(tmp_path)
    assert first.parent == tmp_path / "robustness"  # v1 files directly under robustness/ remain
    monkeypatch.setattr(config, "ROBUSTNESS_VARIANT_VERSION", "robustness-variants-test")
    assert evaluate.robustness_root(tmp_path) != first


# --- Expected behaviour, deltas and stability -------------------------------------------------


def test_generation_classification():
    resume = PAIR["resume"]
    assert rb.classify_generation("", "TimeoutError: x", True) == "crashed"
    assert rb.classify_generation("", "JD_TOO_SHORT", False) == "handled_error"
    assert rb.classify_generation("", None, False) == "handled_error"
    refusal = "I cannot tailor this resume: the text provided is not a job description."
    assert rb.classify_generation(refusal, None, False) == "refused"
    assert rb.classify_generation(resume, None, False) == "completed"
    # A full resume that mentions a refusal phrase is still a completed output.
    assert rb.classify_generation(refusal + "\n" + resume, None, False) == "completed"


def _oos(outcome, **fields):
    return rb.out_of_scope_expected_behaviour({"generation_outcome": outcome, **fields})


def test_out_of_scope_expected_behaviour():
    clean = {
        "fabrication_bait_violation": False,
        "faithfulness": 0.9,
        "protected_field_violation": False,
    }
    assert _oos("crashed")["expected_behaviour_pass"] is False
    assert _oos("crashed")["oos_crashed"] is True
    for outcome in ("handled_error", "refused"):
        result = _oos(outcome)
        assert result["expected_behaviour_pass"] is True and result["oos_handled"] is True
    assert _oos("completed", **clean)["expected_behaviour_pass"] is True
    assert _oos("completed", **{**clean, "fabrication_bait_violation": True})[
        "expected_behaviour_pass"
    ] is False
    low = _oos("completed", **{**clean, "faithfulness": 0.79})
    assert low["oos_no_fabrication"] is False and low["expected_behaviour_pass"] is False
    assert _oos("completed", **{**clean, "faithfulness": 0.8})["expected_behaviour_pass"] is True
    assert _oos("completed", **{**clean, "protected_field_violation": True})[
        "oos_protected_fields_preserved"
    ] is False
    unscored = _oos("completed", **{**clean, "faithfulness": None})
    assert unscored["expected_behaviour_pass"] is None  # never counted as a zero
    known_failure = _oos(
        "completed", **{**clean, "faithfulness": None, "protected_field_violation": True}
    )
    assert known_failure["expected_behaviour_pass"] is False


def test_paired_deltas_and_stability():
    original = {
        "error": None, "task_success": True, "faithfulness": 0.9, "unsupported_claim_rate": 0.1,
        "protected_field_violation": False, "fabrication_bait_violation": False,
        "requirement_coverage": 1.0, "kw_after": 1.0, "sim_after": 0.8,
    }
    small = {**original, "faithfulness": 0.86, "unsupported_claim_rate": 0.14, "sim_after": 0.82}
    deltas = rb.paired_deltas(original, small)
    assert deltas["faithfulness"] == -0.04 and deltas["judge_overall_mean"] is None
    assert rb.stability_assessment(original, small, 0.05)["stable"] is True
    lost = rb.stability_assessment(original, {**small, "task_success": False}, 0.05)
    assert lost["stable"] is False and lost["violations"] == ["task_success"]
    worse = rb.stability_assessment(original, {**small, "unsupported_claim_rate": 0.2}, 0.05)
    assert worse["violations"] == ["unsupported_claim_rate"]
    failed = rb.stability_assessment(original, {"error": "boom"}, 0.05)
    assert failed == {"comparable": False, "stable": None, "criteria_scored": 0, "violations": []}
    nothing = rb.stability_assessment({"error": None}, {"error": None}, 0.05)
    assert nothing["comparable"] and nothing["stable"] is None


def _row(workflow, pair, variant, **fields):
    return {
        "workflow": workflow, "base_pair_id": pair, "variant": variant,
        "variant_family": rb.variant_family(variant), "variant_scope": rb.variant_scope(variant),
        "error": None, "evaluation_failures": {}, "generation_outcome": "completed", **fields,
    }


def test_robustness_report_denominators(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORT_BOOTSTRAP_SAMPLES", 50)
    good = {"faithfulness": 0.9, "kw_after": 1.0, "task_success": True}
    rows = [
        _row("w1", "p1", "original", **good),
        _row("w1", "p1", "typos", **{**good, "faithfulness": 0.7}),
        _row("w1", "p2", "original", **good),
        _row("w1", "p2", "typos", error="TimeoutError: x", generation_outcome="crashed"),
        _row("w1", "p3", "original", **good),
        _row("w1", "p3", "typos", **{**good, "faithfulness": None},
             evaluation_failures={"faithfulness": "NLI failed"}),
        _row("w1", "p4", "original", error="original generation unavailable",
             generation_outcome="missing_original"),
        _row("w1", "p5", "original", **good),
        _row("w1", "p5", "plain_text", **good),
        _row("w1", "p1", "degenerate_jd", generation_outcome="crashed", error="ValueError: x",
             **_oos("crashed")),
        _row("w1", "p2", "degenerate_jd", generation_outcome="handled_error", error="EMPTY",
             **_oos("handled_error")),
        _row("w1", "p3", "mismatched_jd", **_oos(
            "completed", fabrication_bait_violation=True, faithfulness=0.9,
            protected_field_violation=False,
        ), fabrication_bait_violation=True),
        _row("w1", "p5", "mismatched_jd", **_oos(
            "completed", fabrication_bait_violation=False, faithfulness=None,
            protected_field_violation=False,
        )),
    ]
    reports = reporting.write_reports(tmp_path, [], stage_rows={"robustness": rows})
    for name in ("robustness_deltas", "robustness_variants", "robustness_families",
                 "robustness_out_of_scope", "robustness_out_of_scope_families"):
        assert (tmp_path / f"{name}.csv").exists() and (tmp_path / f"{name}.json").exists()

    faithfulness = next(
        row for row in reports["robustness_deltas"]
        if row["variant"] == "typos" and row["metric"] == "faithfulness"
    )
    assert (faithfulness["eligible"], faithfulness["scored"], faithfulness["unscored"],
            faithfulness["failed"], faithfulness["evaluation_failures"]) == (3, 1, 2, 1, 1)
    assert faithfulness["mean_delta"] == -0.2 and faithfulness["worst_degradation"] == 0.2

    typos = next(row for row in reports["robustness_variants"] if row["variant"] == "typos")
    assert (typos["eligible"], typos["failed"], typos["crashed"]) == (3, 1, 1)
    assert typos["execution_retention"] == round(2 / 3, 6)
    # p1 lost faithfulness beyond the tolerance; p3 is judged on kw/task only (NLI unscored).
    assert (typos["scored"], typos["unscored"], typos["stable"]) == (2, 1, 1)
    assert typos["stability_score"] == 0.5 and typos["instability_reasons"] == "faithfulness x1"

    noisy = next(row for row in reports["robustness_families"] if row["family"] == "noisy")
    fmt = next(row for row in reports["robustness_families"] if row["family"] == "format")
    assert noisy["eligible"] == 3 and fmt["stability_score"] == 1.0

    degenerate = next(
        row for row in reports["robustness_out_of_scope"] if row["variant"] == "degenerate_jd"
    )
    assert (degenerate["eligible"], degenerate["scored"], degenerate["unscored"],
            degenerate["failed"], degenerate["handled_errors"]) == (2, 2, 0, 1, 1)
    assert degenerate["expected_behaviour_pass_rate"] == 0.5
    assert degenerate["graceful_handling_rate"] == 0.5
    mismatched = next(
        row for row in reports["robustness_out_of_scope"] if row["variant"] == "mismatched_jd"
    )
    assert (mismatched["eligible"], mismatched["scored"], mismatched["unscored"],
            mismatched["failed"], mismatched["completed"]) == (2, 1, 1, 0, 2)
    assert mismatched["expected_behaviour_pass_rate"] == 0.0
    assert mismatched["fabrication_failures"] == 1
    family = reports["robustness_out_of_scope_families"][0]
    assert (family["eligible"], family["scored"], family["failed"]) == (4, 3, 1)


# --- Stage wiring (fake workflows, no Ollama) -------------------------------------------------


class EchoWorkflow:
    """Returns the clean source resume; degenerate JDs crash, mismatched JDs are refused."""

    def __init__(self, crash_pair=None):
        self.calls = []
        self.crash_pair = crash_pair
        self.clean = {pair["pair_id"]: pair["resume"] for pair in PAIRS}

    def __call__(self, pair, tracker):
        base, variant = pair["pair_id"].rsplit("__", 1)
        self.calls.append((base, variant, Path(pair["resume_path"]).read_text(encoding="utf-8")))
        assert pair["resume"] == Path(pair["resume_path"]).read_text(encoding="utf-8")
        tracker.log(type="llm", input_tokens=10, output_tokens=5, thinking_chars=0, seconds=0.1)
        if variant == "degenerate_jd" and base == self.crash_pair:
            raise RuntimeError("unexpected input")
        if variant == "mismatched_jd":
            return {"resume_md": "I cannot tailor this resume to an unrelated job description.",
                    "errors": []}
        return {"resume_md": self.clean[base], "errors": []}


@pytest.fixture
def stage(tmp_path, monkeypatch, no_network):
    nli_calls = []

    def fake_faithfulness(original, output, **_kwargs):
        nli_calls.append(original)
        return {"faithfulness": 1.0, "claim_faithfulness": 1.0, "unsupported_claim_rate": 0.0,
                "nli_evaluator_version": config.NLI_EVALUATOR_VERSION}

    monkeypatch.setattr(evaluate, "_quality", lambda pair, text: {"kw_after": 1.0})
    monkeypatch.setattr(evaluate, "_current_generation_hash", lambda workflow: "gen")
    monkeypatch.setattr(evaluate.metrics, "faithfulness", fake_faithfulness)
    manifest = rb.robustness_manifest(PAIRS, "dataset-a", config.ROBUSTNESS_SEED)
    sample = manifest["pair_ids"][:2]
    rows = []
    for pair_id in sample:
        run_dir = tmp_path / "saved" / pair_id
        run_dir.mkdir(parents=True)
        (run_dir / "resume.md").write_text(
            next(p["resume"] for p in PAIRS if p["pair_id"] == pair_id), encoding="utf-8",
        )
        (run_dir / "trace.json").write_text("[]", encoding="utf-8")
        rows.append({
            "workflow": "fake", "pair_id": pair_id, "repeat": 0, "error": None,
            "dataset_hash": "dataset-a", "config_hash": "gen", "llm_calls": 1,
            "resume_path": str(run_dir / "resume.md"), "trace_path": str(run_dir / "trace.json"),
        })
    return {"rows": rows, "sample": sample, "nli_calls": nli_calls, "root": tmp_path / "eval"}


def _run_stage(stage, workflow, **kwargs):
    return evaluate.evaluate_robustness(
        ["fake"], stage["rows"], PAIRS, stage["root"], JudgeConfig(), skip_judge=True,
        limit=2, workflow_registry={"fake": workflow}, dataset_hash_value="dataset-a", **kwargs,
    )


def test_stage_scores_against_clean_resume_and_resumes(stage):
    workflow = EchoWorkflow(crash_pair=stage["sample"][0])
    variants = ["typos", "pdf_extraction", "mismatched_jd", "degenerate_jd"]
    rows = _run_stage(stage, workflow, variants=variants)

    stage_root = evaluate.robustness_root(stage["root"])
    assert (stage_root / "sample_manifest.json").exists()
    assert json.loads((stage_root / "identity.json").read_text())["variant_version"] == (
        config.ROBUSTNESS_VARIANT_VERSION
    )
    assert len(rows) == 2 * 5  # original + four variants per pair
    assert len(workflow.calls) == 2 * 4  # the original reuses the saved generation
    for base, variant, presented in workflow.calls:
        clean = next(p for p in PAIRS if p["pair_id"] == base)
        assert presented == rb.build_variant(clean, variant, 42, JD_POOL)["resume"]
    clean_resumes = {p["resume"] for p in PAIRS if p["pair_id"] in stage["sample"]}
    assert stage["nli_calls"] and set(stage["nli_calls"]) <= clean_resumes  # clean source only
    assert len(evaluate.STAGES["robustness"].load(stage["root"])) == len(rows)  # `report` reload

    by_key = {(row["base_pair_id"], row["variant"]): row for row in rows}
    typo = by_key[(stage["sample"][1], "typos")]
    assert typo["scoring_reference"] == "clean_original_resume"
    assert typo["original_resume"] == next(
        p["resume"] for p in PAIRS if p["pair_id"] == stage["sample"][1]
    )
    assert typo["generation_outcome"] == "completed" and typo["kw_after"] == 1.0
    assert typo["protected_field_violation"] is False and typo["variant_family"] == "noisy"
    assert typo["config_hash"] == "gen" and typo["generation_identity_matches_original"] is True
    assert Path(typo["input_resume_path"]).read_text(encoding="utf-8") != typo["original_resume"]
    crashed = by_key[(stage["sample"][0], "degenerate_jd")]
    assert crashed["generation_outcome"] == "crashed" and crashed["error"]
    assert crashed["expected_behaviour_pass"] is False
    assert by_key[(stage["sample"][1], "degenerate_jd")]["expected_behaviour_pass"] is True
    refused = by_key[(stage["sample"][0], "mismatched_jd")]
    assert refused["generation_outcome"] == "refused" and refused["oos_handled"] is True
    assert refused["scoring_jd"] == "presented_jd"

    again = _run_stage(stage, EchoWorkflow(), variants=variants)
    assert len(again) == len(rows)
    (stage_root / "runs" / "fake" / stage["sample"][0] / "typos" / "evaluation.json").unlink()
    reused = EchoWorkflow()
    _run_stage(stage, reused, variants=variants)
    assert reused.calls == []  # generation.json is reused; only evaluation is recomputed

    more = _run_stage(stage, EchoWorkflow(), variants=["plain_text"])
    assert {row["variant"] for row in more} >= {"plain_text", "typos", "degenerate_jd"}


def test_stage_skips_out_of_scope_quality_components(stage):
    class CountingComponent:
        name = "requirement_coverage"

        def __init__(self):
            self.variants = []

        def identity(self):
            return {"status": "enabled"}

        def is_complete(self, record):
            return "requirement_coverage" in record

        def evaluate(self, record, generation_row, pair, tailored, root, force):
            self.variants.append(pair["jd"] == next(
                p["jd"] for p in PAIRS if p["pair_id"] == generation_row["pair_id"]
            ))
            record["requirement_coverage"] = 1.0

    component = CountingComponent()
    _run_stage(stage, EchoWorkflow(), variants=["web_clutter", "degenerate_jd"],
               components=(component,))
    # original + web_clutter for two pairs, all scored against the clean JD.
    assert component.variants == [True] * 4


def test_stage_records_missing_original_without_generating(stage):
    workflow = EchoWorkflow()
    stage["rows"] = stage["rows"][:1]
    rows = _run_stage(stage, workflow, variants=["typos"])
    missing = [row for row in rows if row["generation_outcome"] == "missing_original"]
    assert len(missing) == 1 and missing[0]["base_pair_id"] == stage["sample"][1]
    assert {call[0] for call in workflow.calls} == {stage["sample"][0]}


def test_cli_variant_options():
    parser = evaluate.build_parser()
    args = parser.parse_args([
        "robustness", "--robustness-families", "noisy",
        "--robustness-variants", "degenerate_jd", "--robustness-limit", "10",
    ])
    assert args.robustness_limit == 10
    assert evaluate.robustness_variant_selection(args) == (
        "original", "typos", "pdf_extraction", "web_clutter", "degenerate_jd",
    )
    assert evaluate.robustness_variant_selection(parser.parse_args(["robustness"])) == rb.VARIANTS
    with pytest.raises(SystemExit):
        parser.parse_args(["robustness", "--robustness-variants", "prompt_injection"])
