"""Cross-workstream integration tests for the evaluator CLI. No network or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import evaluate
from eval import injection, reporting
from eval.judge import JudgeConfig
from eval.requirement_coverage import RequirementCoverageConfig

MODEL_INFO = {"provider": "ollama", "tag": "coverage-test", "digest": "digest-a"}

SOURCE = """# Alex Tan
alex.tan@example.com

## Experience
**Data Analyst, Example Pte Ltd** (Jan 2024 - Dec 2024)
- Built dashboards with PostgreSQL and Python.

## Skills
Python, PostgreSQL
"""


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)))


def _coverage(enabled=True, digest="digest-a", client=None, **config_changes):
    cfg = RequirementCoverageConfig(
        provider="ollama", model="coverage-test", max_retries=0, **config_changes
    )
    return evaluate.RequirementCoverageComponent(
        enabled=enabled,
        coverage_config=cfg,
        model_info={**MODEL_INFO, "digest": digest},
        client=client,
    )


def _pair(pair_id="p1", keywords=None, **extra):
    return {
        "pair_id": pair_id,
        "split": "test",
        "category": "data",
        "fit_level": "strong",
        "resume": SOURCE,
        "jd": "Need SQL and Python.",
        "keywords": keywords if keywords is not None else [
            {"keyword": "SQL", "synonyms": ["PostgreSQL"]},
            {"keyword": "Python", "synonyms": []},
        ],
        **extra,
    }


def _no_ops(monkeypatch):
    monkeypatch.setattr(evaluate, "_quality", lambda *_: {})
    monkeypatch.setattr(evaluate, "evaluate_operational", lambda *_a, **_k: {})


def _evaluate(pair, tailored, tmp_path, components, generation_row=None):
    return evaluate._evaluate_output(
        generation_row or {"workflow": "w1_single_pass", "error": None},
        pair,
        tailored,
        tmp_path,
        JudgeConfig(),
        skip_nli=True,
        skip_judge=True,
        trace=[],
        totals={},
        components=components,
    )


# --- Coverage in saved-output scoring -------------------------------------------------------


def test_lexical_coverage_is_scored_without_a_model(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    component = _coverage()
    record = _evaluate(_pair(), SOURCE + "\nUsed PostgreSQL.", tmp_path, (component,))

    assert record["requirement_coverage"] == 1.0
    assert record["requirement_coverage_status"] == "scored"
    assert record["requirement_coverage_supportable_count"] == 2
    assert record["requirement_coverage_evaluator_version"] == (
        config.REQUIREMENT_COVERAGE_EVALUATOR_VERSION
    )
    assert record["requirement_coverage_evaluator_hash"] == component.identity()["evaluator_hash"]
    assert "requirement_coverage" not in record["evaluation_failures"]
    assert component.is_complete(record)


def test_invalid_semantic_output_is_unscored_not_zero(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    client = FakeClient(["not json"])
    component = _coverage(client=client)
    # Python is supportable lexically but missing from the output, so the coverage stage
    # needs a semantic decision; the fake evaluator returns invalid output.
    record = _evaluate(_pair(), "# Alex Tan\nalex.tan@example.com\nPostgreSQL", tmp_path, (component,))

    assert client.calls == 1
    assert record["requirement_coverage"] is None
    assert record["requirement_coverage_status"] == "unscored"
    assert "requirement_coverage" in record["evaluation_failures"]
    assert component.is_complete(record)  # cached evaluator outcome is final for this identity


def test_component_exception_is_retried_on_resume(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    monkeypatch.setattr(
        evaluate,
        "evaluate_requirement_coverage",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("client unavailable")),
    )
    component = _coverage()
    record = _evaluate(_pair(), SOURCE, tmp_path, (component,))

    assert record["requirement_coverage"] is None
    assert record["requirement_coverage_status"] == "evaluation_error"
    assert record["evaluation_failures"]["requirement_coverage"] == (
        "RuntimeError: client unavailable"
    )
    assert not component.is_complete(record)


def test_generation_failure_and_unsupportable_requirements_are_not_scored(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    component = _coverage()
    failed = _evaluate(
        _pair(), "", tmp_path, (component,),
        generation_row={"workflow": "w1_single_pass", "error": "timeout"},
    )
    assert failed["requirement_coverage_status"] == "generation_failed"
    assert failed["requirement_coverage"] is None

    client = FakeClient([json.dumps({"decisions": [{
        "requirement_id": "req-001", "supported": False, "source_quotes": [],
        "rationale": "No Kubernetes evidence.",
    }]})])
    component = _coverage(client=client)
    unsupported = _evaluate(
        _pair(keywords=[{"keyword": "Kubernetes", "synonyms": []}]),
        SOURCE, tmp_path, (component,),
    )
    assert unsupported["requirement_coverage_status"] == "no_supportable_requirements"
    assert unsupported["requirement_coverage"] is None
    assert unsupported["evaluation_failures"] == {}


def test_coverage_report_has_explicit_denominators(monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 50)
    base = {"dataset_hash": "d", "workflow": "w1", "fit_level": "strong"}
    rows = [
        {**base, "error": None, "requirement_coverage": 0.5,
         "requirement_coverage_status": "scored", "requirement_coverage_supportable_count": 4,
         "requirement_coverage_covered_count": 2, "evaluation_failures": {}},
        {**base, "error": None, "requirement_coverage": 1.0,
         "requirement_coverage_status": "scored", "requirement_coverage_supportable_count": 1,
         "requirement_coverage_covered_count": 1, "evaluation_failures": {}},
        {**base, "error": None, "requirement_coverage": None,
         "requirement_coverage_status": "unscored",
         "evaluation_failures": {"requirement_coverage": "invalid json"}},
        {**base, "error": None, "requirement_coverage": None,
         "requirement_coverage_status": "no_supportable_requirements", "evaluation_failures": {}},
        {**base, "error": "timeout", "requirement_coverage": None,
         "requirement_coverage_status": "generation_failed", "evaluation_failures": {}},
    ]
    [summary] = reporting.requirement_coverage_summary(rows)
    assert summary["outputs"] == 5
    assert summary["generation_failures"] == 1
    assert summary["not_applicable"] == 1
    assert (summary["eligible"], summary["scored"], summary["unscored"], summary["failed"]) == (
        3, 2, 1, 1,
    )
    assert summary["mean_coverage"] == 0.75
    assert summary["micro_coverage"] == 0.6

    generic = next(
        row for row in reporting.summarise(rows, ("workflow",))
        if row["metric"] == "requirement_coverage"
    )
    assert (generic["eligible"], generic["scored"], generic["unscored"]) == (3, 2, 1)
    assert generic["evaluation_failures"] == 1


# --- Evaluator identity and checkpoint completeness -----------------------------------------


def _identity(components):
    return evaluate._evaluation_identity(
        JudgeConfig(), {"digest": "judge", "status": "enabled"}, {"model": "nli"}, components,
    )


def test_evaluator_identity_tracks_coverage_model_prompt_and_status():
    base = _identity((_coverage(),))["evaluator_hash"]

    assert base == _identity((_coverage(),))["evaluator_hash"]
    assert base != _identity((_coverage(digest="digest-b"),))["evaluator_hash"]
    assert base != _identity((_coverage(prompt_version="other-prompt"),))["evaluator_hash"]
    assert base != _identity((_coverage(evaluator_version="other"),))["evaluator_hash"]
    assert base != _identity((_coverage(enabled=False),))["evaluator_hash"]
    identity = _identity((_coverage(),))
    assert identity["schema_version"] == config.EVALUATOR_SCHEMA_VERSION
    coverage = identity["output_components"]["requirement_coverage"]
    assert {"model_digest", "system_prompt_hash", "schema_hash", "prompt_version"} <= coverage.keys()


def _saved_generation(tmp_path, pair_id="p1"):
    run_dir = tmp_path / "runs" / pair_id
    run_dir.mkdir(parents=True)
    (run_dir / "resume.md").write_text(SOURCE + "\nPostgreSQL", encoding="utf-8")
    (run_dir / "trace.json").write_text("[]", encoding="utf-8")
    return {
        "workflow": "w1_single_pass",
        "pair_id": pair_id,
        "repeat": 0,
        "error": None,
        "resume_path": str(run_dir / "resume.md"),
        "trace_path": str(run_dir / "trace.json"),
    }


def test_saved_scoring_resumes_and_recomputes_on_identity_change(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    calls = []
    original = evaluate.evaluate_requirement_coverage

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate, "evaluate_requirement_coverage", counting)
    row = _saved_generation(tmp_path)
    pair_map = {"p1": _pair()}
    root = tmp_path / "eval-root"

    def run(component):
        return evaluate.evaluate_saved(
            [row], pair_map, root, JudgeConfig(), skip_nli=True, skip_judge=True,
            components=(component,),
        )

    first = run(_coverage())
    second = run(_coverage())
    assert len(calls) == 1
    assert first[0]["requirement_coverage"] == second[0]["requirement_coverage"] == 1.0
    run(_coverage(digest="digest-b"))
    assert len(calls) == 2
    assert (root / "evaluations.jsonl").exists()


def test_saved_scoring_does_not_touch_generation_artifacts(monkeypatch, tmp_path):
    _no_ops(monkeypatch)
    row = _saved_generation(tmp_path)
    before = Path(row["resume_path"]).read_bytes(), Path(row["trace_path"]).read_bytes()
    evaluate.evaluate_saved(
        [row], {"p1": _pair()}, tmp_path / "eval-root", JudgeConfig(),
        skip_nli=True, skip_judge=True, components=(_coverage(),),
    )
    assert (Path(row["resume_path"]).read_bytes(), Path(row["trace_path"]).read_bytes()) == before


# --- Injection stage ------------------------------------------------------------------------


@pytest.fixture
def injection_pairs(monkeypatch):
    pairs = [_pair("p1"), _pair("p2"), _pair("p3")]
    monkeypatch.setattr(injection, "select_stratified_pairs", lambda items, seed: items[:2])
    return pairs


class FakeWorkflow:
    """Obeys resume-channel attacks, ignores JD-channel attacks and fails on one pair."""

    def __init__(self, manifest_path=None, fail_pair=None):
        self.calls = []
        self.manifest_path = manifest_path
        self.fail_pair = fail_pair

    def __call__(self, pair, tracker):
        if self.manifest_path is not None:
            assert self.manifest_path.exists(), "manifest must be written before generation"
        self.calls.append((pair["pair_id"], pair["injection_channel"]))
        if pair["pair_id"] == self.fail_pair:
            raise TimeoutError("writer timed out")
        text = SOURCE
        if pair["injection_channel"] == "resume":
            text += f"\n{pair['injection_canary']}\n"
        return {"resume_md": text, "errors": []}


def _run_injection(tmp_path, pairs, workflow, **kwargs):
    return evaluate.evaluate_injection(
        ["fake"], pairs, tmp_path, "dataset-a", skip_nli=True,
        workflow_registry={"fake": workflow}, **kwargs,
    )


def test_injection_writes_manifest_first_and_resumes(tmp_path, injection_pairs):
    stage_root = evaluate.injection_root(tmp_path)
    workflow = FakeWorkflow(manifest_path=stage_root / "sample_manifest.json", fail_pair="p2")
    rows = _run_injection(tmp_path, injection_pairs, workflow)

    assert len(rows) == 4 and len(workflow.calls) == 4
    manifest = json.loads((stage_root / "sample_manifest.json").read_text())
    assert manifest["pair_ids"] == ["p1", "p2"]
    assert manifest["attack_version"] == config.INJECTION_ATTACK_VERSION
    by_key = {(row["pair_id"], row["channel"]): row for row in rows}
    assert by_key[("p1", "resume")]["injection_success"] is True
    assert by_key[("p1", "jd")]["injection_success"] is False
    failed = by_key[("p2", "resume")]
    assert failed["injection_success"] is None
    assert failed["generation_completed"] is False
    assert "TimeoutError" in failed["error"]
    # Diagnostics run against the clean source, so the injected canary is not "protected".
    assert by_key[("p1", "resume")]["protected_field_evaluator_version"] == (
        config.PROTECTED_FIELD_EVALUATOR_VERSION
    )
    run_dir = stage_root / "runs" / "fake" / "p1" / "resume"
    for name in ("resume.md", "trace.json", "generation.json", "evaluation.json"):
        assert (run_dir / name).exists()
    assert (stage_root / "evaluations.jsonl").exists()

    again = FakeWorkflow()
    resumed = _run_injection(tmp_path, injection_pairs, again)
    assert again.calls == []
    assert len(resumed) == 4

    (run_dir / "evaluation.json").unlink()
    rescored = _run_injection(tmp_path, injection_pairs, again)
    assert again.calls == []  # generation.json is reused; only diagnostics are recomputed
    assert next(
        row for row in rescored if (row["pair_id"], row["channel"]) == ("p1", "resume")
    )["injection_success"] is True


def test_injection_limit_and_manifest_drift(tmp_path, injection_pairs, monkeypatch):
    workflow = FakeWorkflow()
    rows = _run_injection(tmp_path, injection_pairs, workflow, limit=1)
    assert {row["pair_id"] for row in rows} == {"p1"}

    monkeypatch.setattr(injection, "select_stratified_pairs", lambda items, seed: items[1:3])
    with pytest.raises(RuntimeError, match="INJECTION_MANIFEST_CHANGED"):
        _run_injection(tmp_path, injection_pairs, workflow)


def test_injection_rejects_generation_identity_mismatch(tmp_path, injection_pairs, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "config_hash", lambda workflow: "current")
    workflow = FakeWorkflow()
    with pytest.raises(RuntimeError, match="INJECTION_GENERATION_IDENTITY_MISMATCH"):
        _run_injection(
            tmp_path, injection_pairs, workflow,
            generation_config_hashes={"fake": {"saved"}},
        )
    assert workflow.calls == []
    rows = _run_injection(
        tmp_path, injection_pairs, workflow, generation_config_hashes={"fake": {"current"}},
    )
    assert {row["config_hash"] for row in rows} == {"current"}


def test_injection_report_has_explicit_denominators(tmp_path, injection_pairs, monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 50)
    rows = _run_injection(tmp_path, injection_pairs, FakeWorkflow(fail_pair="p2"))
    reports = reporting.write_reports(tmp_path / "summaries", [], stage_rows={"injection": rows})

    overall = next(row for row in reports["injection"] if row["scope"] == "overall")
    assert (overall["eligible"], overall["scored"], overall["unscored"], overall["failed"]) == (
        4, 2, 2, 2,
    )
    assert overall["injection_success_rate"] == 0.5
    assert (tmp_path / "summaries" / "injection.csv").exists()
    assert (tmp_path / "summaries" / "injection_diagnostics.json").exists()


def test_write_reports_keeps_legacy_robustness_argument(tmp_path, monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 50)
    rows = [
        {"workflow": "w1", "base_pair_id": "p1", "variant": "original", "faithfulness": 0.9,
         "error": None, "evaluation_failures": {}},
        {"workflow": "w1", "base_pair_id": "p1", "variant": "formatting", "faithfulness": 0.8,
         "error": None, "evaluation_failures": {}},
    ]
    reporting.write_reports(tmp_path, [], robustness_rows=rows)
    assert (tmp_path / "robustness.csv").exists()
    assert (tmp_path / "robustness_status.json").exists()


# --- CLI ------------------------------------------------------------------------------------


def test_cli_exposes_registered_stages():
    parser = evaluate.build_parser()
    for stage in evaluate.STAGES:
        args = parser.parse_args([stage, f"--skip-{stage}", f"--{stage}-limit", "3"])
        assert args.command == stage
        assert getattr(args, f"skip_{stage}") is True
        assert getattr(args, f"{stage}_limit") == 3
        assert set(reporting.STAGE_REPORTERS) >= {stage}
    args = parser.parse_args(["score", "--skip-coverage", "--coverage-model", "m"])
    assert args.skip_coverage and args.coverage_model == "m"


@pytest.fixture
def cli_env(monkeypatch, tmp_path, injection_pairs):
    import run_all

    results_file = tmp_path / "results" / "results.jsonl"
    generation = _saved_generation(tmp_path)
    generation.update({"dataset_hash": "dataset-a", "config_hash": "gen-hash",
                       "metrics_path": generation["resume_path"]})
    resolved = []
    workflow = FakeWorkflow()
    monkeypatch.setattr(config, "RESULTS_FILE", str(results_file))
    monkeypatch.setattr(evaluate, "WORKFLOWS", {"w1_single_pass": workflow})
    monkeypatch.setattr(evaluate, "validate_dataset", lambda: {"dataset_hash": "dataset-a"})
    monkeypatch.setattr(evaluate, "load_pairs", lambda split: injection_pairs)
    monkeypatch.setattr(evaluate, "_generation_rows", lambda workflows: [generation])
    monkeypatch.setattr(evaluate.metrics, "nli_model_identity", lambda: {"model": "nli"})
    monkeypatch.setattr(run_all, "config_hash", lambda workflow: "gen-hash")

    def configured(cfg, client=None):
        resolved.append((cfg.provider, cfg.model))
        return {"provider": cfg.provider, "tag": cfg.model, "digest": "d"}

    monkeypatch.setattr(evaluate, "configured_model", configured)
    _no_ops(monkeypatch)
    return SimpleNamespace(workflow=workflow, resolved=resolved)


def test_cli_all_runs_coverage_then_optional_injection_then_report_reloads(cli_env, capsys):
    flags = ["--skip-faithfulness", "--skip-judge", "--skip-robustness", "--skip-human-review"]
    root = evaluate.main(["all", *flags, "--injection-limit", "1"])

    assert cli_env.resolved == [("ollama", config.JUDGE_MODEL)]  # resolved once per run
    identity = json.loads((root / "evaluator_identity.json").read_text())
    assert identity["output_components"]["requirement_coverage"]["status"] == "enabled"
    record = json.loads((root / "evaluations.jsonl").read_text().splitlines()[0])
    assert record["requirement_coverage"] == 1.0
    assert cli_env.workflow.calls == []  # injection is optional and not part of `all`
    assert evaluate.main(["injection", *flags, "--injection-limit", "1"]) == root
    assert len(cli_env.workflow.calls) == 2
    evaluate.main(["report", *flags])
    assert (root / "summaries" / "requirement_coverage.csv").exists()
    assert (root / "summaries" / "injection.csv").exists()

    (root / "summaries" / "injection.csv").unlink()
    again = evaluate.main(["report", *flags])
    assert again == root
    assert len(cli_env.workflow.calls) == 2  # report never generates
    assert (root / "summaries" / "injection.csv").exists()
    output = capsys.readouterr().out
    assert '"injection_rows": 2' in output

    skipped = evaluate.main(["score", *flags, "--skip-coverage"])
    assert skipped != root


# --- Network guard for the stage tests below ------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def refuse(*_args, **_kwargs):
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# --- Pairwise stage -------------------------------------------------------------------------


def _pairwise_inputs():
    from tests import test_pairwise as tp

    rows = [
        {**tp._row("w2_baseline"), "resume_md": tp.BASELINE},
        {**tp._row("w3_critic"), "resume_md": tp.CHALLENGER},
        {**tp._row("w2_baseline", pair_id="p2"), "resume_md": tp.BASELINE},
    ]
    pair_map = {
        pair_id: {"pair_id": pair_id, "resume": tp.ORIGINAL, "jd": tp.JD, "fit_level": "strong"}
        for pair_id in ("p1", "p2")
    }
    return tp, rows, pair_map


def _run_pairwise_stage(tmp_path, client, rows, pair_map, tp, **kwargs):
    return evaluate.evaluate_pairwise_stage(
        rows, pair_map, tmp_path, tp._config(),
        {"provider": "ollama", "tag": "gemma-test", "digest": "digest-a"},
        challenger_workflows=["w3_critic"], client=client, **kwargs,
    )


def test_pairwise_stage_scores_resumes_and_reports(tmp_path, no_network, monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 50)
    tp, rows, pair_map = _pairwise_inputs()
    client = tp.FakeClient([
        tp._valid_response("candidate_a", tp.CHALLENGER, tp.BASELINE),
        tp._valid_response("candidate_b", tp.BASELINE, tp.CHALLENGER),
    ])
    records = _run_pairwise_stage(tmp_path, client, rows, pair_map, tp)

    by_pair = {record["pair_id"]: record for record in records}
    assert by_pair["p1"]["status"] == "scored"
    assert by_pair["p1"]["consensus_winner"] == "challenger"
    assert by_pair["p2"]["status"] == "generation_unavailable"
    assert by_pair["p2"]["consensus_winner"] is None
    identity = by_pair["p1"]["evaluator"]
    stage_root = evaluate.pairwise_root(tmp_path, identity)
    assert (stage_root / "records" / "w3_critic" / "p1" / "repeat_0.json").exists()
    assert (stage_root / "evaluations.jsonl").exists()
    assert list((tmp_path / "pairwise_cache").glob("*.json"))

    resumed = _run_pairwise_stage(tmp_path, tp.FakeClient([]), rows, pair_map, tp)
    assert {record["pair_id"]: record["status"] for record in resumed} == {
        "p1": "scored", "p2": "generation_unavailable",
    }

    reports = reporting.write_reports(tmp_path / "summaries", [], stage_rows={"pairwise": records})
    [summary] = reports["pairwise"]
    assert (summary["total_pairs"], summary["generation_unavailable"], summary["eligible"],
            summary["scored"], summary["unscored"]) == (2, 1, 1, 1, 0)
    assert (tmp_path / "summaries" / "pairwise.csv").exists()


def test_pairwise_judge_failures_are_retried_on_resume(tmp_path, no_network):
    tp, rows, pair_map = _pairwise_inputs()
    failing = tp.FakeClient([ConnectionError("judge unavailable")] * 4)
    [record] = [
        r for r in _run_pairwise_stage(tmp_path, failing, rows, pair_map, tp)
        if r["pair_id"] == "p1"
    ]
    assert record["status"] == "unscored"
    assert record["failure_kind"] == "judge_failed"
    assert record["consensus_winner"] is None

    healthy = tp.FakeClient([
        tp._valid_response("candidate_a", tp.CHALLENGER, tp.BASELINE),
        tp._valid_response("candidate_b", tp.BASELINE, tp.CHALLENGER),
    ])
    [retried] = [
        r for r in _run_pairwise_stage(tmp_path, healthy, rows, pair_map, tp)
        if r["pair_id"] == "p1"
    ]
    assert retried["status"] == "scored"
    assert len(healthy.calls) == 2


def test_pairwise_identity_change_selects_new_tree(tmp_path, no_network):
    tp, rows, pair_map = _pairwise_inputs()
    model = {"provider": "ollama", "tag": "gemma-test", "digest": "digest-a"}
    first = evaluate.pairwise_root(tmp_path, evaluate.pairwise_identity(tp._config(), model))
    second = evaluate.pairwise_root(
        tmp_path, evaluate.pairwise_identity(tp._config(), {**model, "digest": "digest-b"}),
    )
    assert first != second


def test_pairwise_config_hash_option_parsing():
    assert evaluate._parse_config_hashes(["w2_baseline=abc", "w3_critic=def"]) == {
        "w2_baseline": "abc", "w3_critic": "def",
    }
    with pytest.raises(SystemExit):
        evaluate._parse_config_hashes(["missing-separator"])


# --- CLI paths for the new stages -----------------------------------------------------------


def test_cli_pairwise_fails_clearly_without_baseline(cli_env):
    flags = ["--skip-faithfulness", "--skip-judge"]
    with pytest.raises(SystemExit, match="pairwise needs the baseline"):
        evaluate.main(["pairwise", *flags])


def test_cli_does_not_expose_dropped_stages():
    assert set(evaluate.STAGES) == {"robustness", "injection", "pairwise"}
    assert not evaluate.STAGES["injection"].in_all
    assert "agreement" not in reporting.STAGE_REPORTERS
    assert all(component.name != "ats" for component in evaluate.build_output_components(
        evaluate.build_parser().parse_args(["score", "--skip-coverage"]), JudgeConfig(), {},
    ))


def test_cli_report_does_not_overwrite_started_review_forms(cli_env):
    flags = ["--skip-faithfulness", "--skip-judge", "--skip-robustness", "--skip-injection"]
    root = evaluate.main(["all", *flags])
    rubric = evaluate.review_root(root) / "rubric_ratings.csv"
    assert rubric.exists()
    rubric.write_text(
        rubric.read_text().splitlines()[0] + "\nreview-x,reviewer-1,3,3,3,3,3,\n",
        encoding="utf-8",
    )
    before = rubric.read_text()
    evaluate.main(["report", *flags])
    assert rubric.read_text() == before


def test_coverage_transport_failure_is_not_cached_and_is_retried(monkeypatch, tmp_path):
    _no_ops(monkeypatch)

    class Unreachable:
        calls = 0

        def chat(self, **_kwargs):
            Unreachable.calls += 1
            raise ConnectionError("rate limited")

    component = _coverage(client=Unreachable())
    tailored = "# Alex Tan\nalex.tan@example.com\nPostgreSQL"
    record = _evaluate(_pair(), tailored, tmp_path, (component,))

    assert record["requirement_coverage"] is None
    assert record["requirement_coverage_status"] == "evaluation_error"
    assert record["requirement_coverage_record"]["failure_kind"] == "evaluator_call_failed"
    assert "rate limited" in record["evaluation_failures"]["requirement_coverage"]
    assert not component.is_complete(record)
    assert not list((tmp_path / "cache" / "requirement_coverage").rglob("*.json"))

    healthy = FakeClient([json.dumps({"decisions": [{
        "requirement_id": "req-002", "covered": False, "output_quotes": [],
        "rationale": "Python is not expressed.",
    }]})])
    retried = _evaluate(_pair(), tailored, tmp_path, (_coverage(client=healthy),))
    assert retried["requirement_coverage_status"] == "scored"
    assert retried["requirement_coverage"] == 0.5


# --- Tier 1 end to end: faithfulness, coverage, operational and pairwise vs W2 ---------------


@pytest.fixture
def tier1_env(monkeypatch, tmp_path, no_network):
    import eval.pairwise as pairwise_module
    from tests import test_pairwise as tp

    pairs = [
        {"pair_id": pair_id, "split": "test", "category": "data", "fit_level": "strong",
         "resume": tp.ORIGINAL, "jd": tp.JD,
         "keywords": [{"keyword": "Python", "synonyms": []}]}
        for pair_id in ("p1", "p2", "p3")
    ]
    outputs = {"w2_baseline": tp.BASELINE, "w1_single_pass": tp.CHALLENGER}
    rows = []
    for workflow, text in outputs.items():
        for pair in pairs:
            for config_hash in (f"gen-{workflow}", "stale") if workflow == "w1_single_pass" \
                    and pair["pair_id"] == "p1" else (f"gen-{workflow}",):
                run_dir = tmp_path / "runs" / workflow / config_hash / pair["pair_id"]
                run_dir.mkdir(parents=True)
                (run_dir / "resume.md").write_text(text, encoding="utf-8")
                (run_dir / "trace.json").write_text("[]", encoding="utf-8")
                rows.append({
                    "workflow": workflow, "pair_id": pair["pair_id"], "repeat": 0,
                    "dataset_hash": "dataset-a", "config_hash": config_hash, "error": None,
                    "category": "data", "fit_level": "strong",
                    "llm_calls": 1, "tool_calls": 0, "input_tokens": 100,
                    "output_tokens": 50, "seconds": 2.5,
                    "resume_path": str(run_dir / "resume.md"),
                    "trace_path": str(run_dir / "trace.json"),
                    "metrics_path": str(run_dir / "resume.md"),
                })

    def responses():
        while True:
            yield tp._valid_response("candidate_a", tp.CHALLENGER, tp.BASELINE)
            yield tp._valid_response("candidate_b", tp.BASELINE, tp.CHALLENGER)

    judge = tp.FakeClient([])
    stream = responses()
    judge.responses = type("Endless", (), {"pop": lambda self, _index: next(stream)})()

    monkeypatch.setattr(config, "RESULTS_FILE", str(tmp_path / "results" / "results.jsonl"))
    monkeypatch.setattr(evaluate, "validate_dataset", lambda: {"dataset_hash": "dataset-a"})
    monkeypatch.setattr(evaluate, "load_pairs", lambda split: pairs)
    monkeypatch.setattr(
        evaluate, "_generation_rows",
        lambda workflows: [row for row in rows if row["workflow"] in workflows],
    )
    monkeypatch.setattr(evaluate, "_quality", lambda *_: {})
    monkeypatch.setattr(evaluate.metrics, "nli_model_identity", lambda: {"model": "nli"})
    monkeypatch.setattr(evaluate.metrics, "nli_preflight", lambda: {"ok": True})
    monkeypatch.setattr(evaluate.metrics, "faithfulness", lambda *_a, **_k: {
        "faithfulness": 0.9, "claim_faithfulness": 0.9, "unsupported_claim_rate": 0.1,
        "nli_evaluator_version": config.NLI_EVALUATOR_VERSION,
    })
    monkeypatch.setattr(
        evaluate, "configured_model",
        lambda cfg, client=None: {"provider": cfg.provider, "tag": cfg.model, "digest": "d"},
    )
    monkeypatch.setattr(pairwise_module, "_default_client", lambda _cfg: judge)
    return judge


TIER1 = [
    "all", "--workflows", "w2_baseline", "w1_single_pass",
    "--skip-judge", "--skip-robustness", "--skip-human-review",
    "--judge-model", "gemma-test",
]


def test_tier1_cli_runs_end_to_end_on_a_fixed_pair_subset(tier1_env):
    with pytest.raises(RuntimeError, match="duplicate generation row"):
        evaluate.main(TIER1)  # two saved w1/p1 generations must be disambiguated

    selected = [*TIER1, "--config-hash", "w1_single_pass=gen-w1_single_pass"]
    root = evaluate.main([*selected, "--pair-ids", "p1", "p2"])

    rows = [json.loads(line) for line in (root / "evaluations.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["faithfulness"] == 0.9
        assert row["requirement_coverage"] == 1.0
        assert "task_success" in row and "step_efficiency" in row
    pairwise_rows = [
        json.loads(line)
        for line in evaluate._pairwise_index(root).read_text().splitlines()
    ]
    assert {(row["pair_id"], row["status"], row["consensus_winner"]) for row in pairwise_rows} == {
        ("p1", "scored", "challenger"), ("p2", "scored", "challenger"),
    }
    assert len(tier1_env.calls) == 4  # two orders per comparison
    for name in ("overall.csv", "requirement_coverage.csv", "pairwise.csv"):
        assert (root / "summaries" / name).exists()

    # A second subset adds its rows without re-judging or dropping the first subset.
    assert evaluate.main([*selected, "--pair-ids", "p3"]) == root
    assert len(tier1_env.calls) == 6
    assert len((root / "evaluations.jsonl").read_text().splitlines()) == 6
    assert len(evaluate._pairwise_index(root).read_text().splitlines()) == 3
    summary = json.loads((root / "summaries" / "pairwise.json").read_text())
    assert (summary[0]["eligible"], summary[0]["scored"]) == (3, 3)

    evaluate.main([*selected])  # full rerun is fully cached
    assert len(tier1_env.calls) == 6

    (root / "summaries" / "pairwise.json").unlink()
    evaluate.main(["report", *selected[1:]])  # rebuilt from the compact index alone
    summary = json.loads((root / "summaries" / "pairwise.json").read_text())
    assert summary[0]["scored"] == 3
    assert summary[0]["first_position_selection_rate"] == 0.5
    assert len(tier1_env.calls) == 6


def test_pair_ids_file_and_unknown_ids(tier1_env, tmp_path):
    selected = [*TIER1, "--config-hash", "w1_single_pass=gen-w1_single_pass"]
    ids = tmp_path / "ids.txt"
    ids.write_text("# pilot subset\np2\n", encoding="utf-8")
    root = evaluate.main([*selected, "--pair-ids-file", str(ids)])
    assert {json.loads(line)["pair_id"] for line in (root / "evaluations.jsonl").read_text().splitlines()} == {"p2"}
    with pytest.raises(SystemExit, match="Unknown pair IDs"):
        evaluate.main([*selected, "--pair-ids", "nope"])


def test_transient_failures_make_records_incomplete():
    import evaluate

    base = {
        "nli_evaluator_version": config.NLI_EVALUATOR_VERSION,
        "protected_field_evaluator_version": config.PROTECTED_FIELD_EVALUATOR_VERSION,
        "fabrication_bait_evaluator_version": config.FABRICATION_BAIT_EVALUATOR_VERSION,
        "judge_status": "unscored",
    }
    rate_limited = {**base, "evaluation_failures": {"judge_overall_mean": "RateLimitError: Error code: 429"}}
    offline = {**base, "evaluation_failures": {"embedding_or_keyword": "ConnectionError: Failed to connect to Ollama."}}
    invalid = {**base, "evaluation_failures": {"judge_overall_mean": "ValueError: unverifiable quotation"}}

    assert not evaluate._record_complete(rate_limited)
    assert not evaluate._record_complete(offline)
    assert evaluate._record_complete(invalid)


def test_pairwise_hosted_calls_use_pairwise_completion_budget():
    from eval import pairwise
    from eval.judge import JudgeConfig

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after capturing request")

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    judge_config = JudgeConfig(provider="groq", model="openai/gpt-oss-120b", min_request_interval_seconds=0)
    try:
        pairwise._call_judge(_Client(), judge_config, "prompt")
    except RuntimeError:
        pass
    assert captured["max_completion_tokens"] == config.PAIRWISE_MAX_COMPLETION_TOKENS
    assert pairwise.pairwise_identity(judge_config, {})["pairwise_max_completion_tokens"] == config.PAIRWISE_MAX_COMPLETION_TOKENS
