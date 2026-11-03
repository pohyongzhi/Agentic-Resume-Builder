"""Synthetic evaluation root for Results-tab tests, written with the real evaluation code.

Per-output rows carry the same fields ``evaluate._evaluate_output`` writes (operational metrics
come from ``eval.operational.evaluate_operational``); pairwise records are produced by
``evaluate.evaluate_pairwise_stage`` with a scripted offline judge; robustness rows follow
``evaluate._robustness_record`` and use ``eval.robustness`` for out-of-scope checks; summaries are
written by ``eval.reporting.write_reports``. Nothing touches the network.

    python -m tests.dashboard_fixture /tmp/demo_results   # then ARB_EVALUATIONS_DIR=... python app.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import config
from eval.judge import DIMENSIONS, JudgeConfig
from eval.operational import evaluate_operational
from eval.reporting import write_reports
from eval.robustness import (
    VARIANT_FAMILIES,
    classify_generation,
    out_of_scope_expected_behaviour,
    variant_scope,
)

DATASET_HASH = "f" * 64
GENERATION_SCOPE = "abc123def456"
EVALUATOR_HASH = "0011223344556677"
WORKFLOWS = (
    "w1_single_pass", "w2_baseline", "w3_critic", "w4_evidence",
    "w5_fine_tuned", "w6_best_of_n", "w7_plan_execute",
)
PAIRS = (
    ("bt_resume_001__bt_jd_01", "business_technology", "business_analyst", "strong"),
    ("bt_resume_007__bt_jd_02", "business_technology", "business_analyst", "weak"),
    ("se_resume_002__se_jd_01", "software_engineering", "frontend_engineer", "strong"),
    ("se_resume_010__se_jd_03", "software_engineering", "backend_engineer", "moderate"),
)
# (challenger, pair index) -> scripted pairwise outcome; others default by workflow quality.
PAIRWISE_SCRIPT = {
    ("w3_critic", 1): "inconsistent",
    ("w6_best_of_n", 2): "tie",
    ("w7_plan_execute", 3): "judge_failed",
}
# Mean quality per workflow so the charts have a visible spread.
QUALITY = {
    "w1_single_pass": 0.55, "w2_baseline": 0.80, "w3_critic": 0.84, "w4_evidence": 0.93,
    "w5_fine_tuned": 0.62, "w6_best_of_n": 0.78, "w7_plan_execute": 0.66,
}
COST = {
    "w1_single_pass": (1, 1300, 20), "w2_baseline": (4, 9000, 95), "w3_critic": (9, 26000, 260),
    "w4_evidence": (7, 21000, 210), "w5_fine_tuned": (1, 1250, 18),
    "w6_best_of_n": (5, 15000, 150), "w7_plan_execute": (6, 12000, 120),
}


def original_resume(pair_id):
    name = "Aisha Koh" if pair_id.startswith("bt") else "Ravi Menon"
    return (
        f"# {name}\n{name.split()[0].lower()}@example.test | Singapore\n\n"
        f"## Summary\nAnalyst with SQL and dashboard experience. (source {pair_id})\n\n"
        "## Experience\n**Analyst Intern, Example Labs** (Jan 2024 - Jun 2024)\n"
        "- Built weekly dashboards in Power BI.\n- Cleaned shipment data with SQL.\n\n"
        "## Education\nBSc Information Systems, Example University, 2025\n\n"
        "## Skills\nSQL, Power BI, Excel, Python (basic)\n"
    )


def job_description(pair_id):
    return (
        f"# Junior Analyst ({pair_id})\n\n**Company:** Example Meridian Labs (fictional)\n\n"
        "## Requirements\n- SQL\n- Dashboards (Power BI or Tableau)\n- Stakeholder communication\n"
    )


def tailored_resume(workflow, pair_id):
    """A complete four-section resume; the HTML comment lets the offline judge identify it."""
    words = " ".join(["Delivered clear reporting for regional stakeholders."] * 30)
    return (
        f"<!-- fixture:{workflow}:{pair_id} -->\n# Candidate tailored by {workflow}\n"
        "candidate@example.test | Singapore\n\n## Summary\nAnalyst with SQL and Power BI "
        f"dashboards. {words}\n\n## Experience\n- Built weekly dashboards in Power BI.\n"
        "- Cleaned shipment data with SQL.\n\n## Education\nBSc Information Systems, "
        "Example University, 2025\n\n## Skills\nSQL, Power BI, Excel\n"
    )


def _generation_row(workflow, pair_index, rng, error=None):
    pair_id, category, role, fit = PAIRS[pair_index]
    calls, tokens, seconds = COST[workflow]
    tokens = int(tokens * rng.uniform(0.85, 1.15))
    run = f"results/runs/1.0.0/{GENERATION_SCOPE}/{workflow}/{pair_id}/repeat_0"
    return {
        "llm_calls": calls, "input_tokens": int(tokens * 0.7),
        "output_tokens": tokens - int(tokens * 0.7), "thinking_chars": 0,
        "seconds": round(seconds * rng.uniform(0.8, 1.2), 1),
        "tool_calls": 2 if workflow in {"w2_baseline", "w3_critic", "w4_evidence"} else 0,
        "tool_calls_ok": 2 if workflow in {"w2_baseline", "w3_critic", "w4_evidence"} else 0,
        "tool_calls_repeated": 0, "tool_success": None, "hit_budget": False,
        "finished": error is None, "error": error,
        "workflow": workflow, "pair_id": pair_id, "resume_id": pair_id.split("__")[0],
        "jd_id": pair_id.split("__")[1], "category": category, "role_family": role,
        "seniority": "entry", "fit_level": fit, "split": "test", "repeat": 0,
        "model": (
            config.FINE_TUNED_MODEL if workflow == "w5_fine_tuned" else config.BASE_MODEL
        ),
        "config_hash": f"{workflow[:2]}cfg0000000", "dataset_version": "1.0.0",
        "dataset_hash": DATASET_HASH, "wall_seconds": round(seconds * rng.uniform(0.8, 1.2), 1),
        "resume_path": f"{run}/resume.md", "metrics_path": f"{run}/metrics.json",
        "trace_path": f"{run}/trace.json", "native_output_path": None,
    }


def score_row(workflow, pair_index, rng, *, error=None, coverage_status="scored"):
    """One ``evaluations.jsonl`` row as ``evaluate._evaluate_output`` builds it."""
    pair_id = PAIRS[pair_index][0]
    generation = _generation_row(workflow, pair_index, rng, error)
    text = "" if error else tailored_resume(workflow, pair_id)
    record = {
        **generation,
        "resume_md": text,
        "original_resume": original_resume(pair_id),
        "job_description": job_description(pair_id),
        "evaluation_failures": {},
        "words": len(text.split()), "sim_before": 0.81, "sim_after": None, "sim_gain": None,
        "kw_before": 0.6, "kw_after": None, "kw_missing": [],
    }
    failures = record["evaluation_failures"]
    quality = QUALITY[workflow]
    if error:
        record.update({"protected_field_violation": None, "fabrication_bait_violation": None,
                       "faithfulness": None, "claim_faithfulness": None,
                       "unsupported_claim_rate": None,
                       "requirement_coverage": None,
                       "requirement_coverage_status": "generation_failed"})
    else:
        faith = round(min(1.0, max(0.0, quality + rng.uniform(-0.08, 0.08))), 4)
        violations = 1 if quality < 0.6 and pair_index == 0 else 0
        bait = 1 if workflow == "w7_plan_execute" and pair_index == 1 else 0
        record.update({
            "sim_after": round(0.78 + quality * 0.1, 4),
            "sim_gain": round(quality * 0.1 - 0.03, 4),
            "kw_after": round(min(1.0, 0.5 + quality * 0.5), 4),
            "protected_field_evaluator_version": config.PROTECTED_FIELD_EVALUATOR_VERSION,
            "protected_field_violation": bool(violations),
            "protected_field_violation_count": violations,
            "protected_field_violations": [{"field": "employer"}] * violations,
            "fabrication_bait_evaluator_version": config.FABRICATION_BAIT_EVALUATOR_VERSION,
            "fabrication_bait_violation": bool(bait),
            "fabrication_bait_term_count": 3, "fabrication_bait_match_count": bait,
            "fabrication_bait_terms": ["tableau"], "fabrication_bait_matches": ["tableau"] * bait,
            "faithfulness": faith, "claim_faithfulness": faith,
            "unsupported_claim_rate": round(1 - faith, 4),
            "supported_claims": int(faith * 20), "unsupported": [], "n_claims": 20,
            "claim_scores": [], "nli_evaluator_version": config.NLI_EVALUATOR_VERSION,
            "nli_seconds": 1.2,
        })
        record["requirement_coverage_evaluator_version"] = (
            config.REQUIREMENT_COVERAGE_EVALUATOR_VERSION
        )
        record["requirement_coverage_status"] = coverage_status
        if coverage_status == "scored":
            covered = 2 if quality < 0.7 else 3
            record.update({
                "requirement_coverage": round(covered / 3, 6),
                "requirement_coverage_supportable_count": 3,
                "requirement_coverage_covered_count": covered,
                "requirement_coverage_uncovered": [],
            })
        elif coverage_status == "evaluation_error":
            record["requirement_coverage"] = None
            failures["requirement_coverage"] = "RuntimeError: rate limited"
        else:  # no_supportable_requirements
            record["requirement_coverage"] = None
            record["requirement_coverage_supportable_count"] = 0
            record["requirement_coverage_covered_count"] = 0
        record["requirement_coverage_seconds"] = 0.4
    record.update(evaluate_operational(
        workflow,
        {"split": "test"},
        {"resume_md": text, "finished": not error and bool(text.strip()),
         "hit_budget": False, "errors": [error] if error else []},
        generation, [], scored_row=record, error=error,
    ))
    record["evaluator_hash"] = EVALUATOR_HASH
    return record


def build_score_rows(workflows=WORKFLOWS, seed=7):
    rng = random.Random(seed)
    rows = []
    for workflow in workflows:
        for index in range(len(PAIRS)):
            if workflow == "w3_critic" and index == 3:
                rows.append(score_row(workflow, index, rng, error="W3_FINAL_VALIDATION_FAILED: x"))
            elif workflow == "w6_best_of_n" and index == 2:
                rows.append(score_row(workflow, index, rng, coverage_status="evaluation_error"))
            elif index == 1 and workflow == "w1_single_pass":
                rows.append(score_row(workflow, index, rng,
                                      coverage_status="no_supportable_requirements"))
            else:
                rows.append(score_row(workflow, index, rng))
    return rows


class ScriptedJudge:
    """Offline pairwise judge: reads the fixture markers and answers per ``PAIRWISE_SCRIPT``."""

    def __init__(self):
        self.calls = 0

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model="fixture-judge", digest="d0")])

    def show(self, model):
        return SimpleNamespace(modelinfo={})

    @staticmethod
    def _section(prompt, name):
        return re.search(rf"<{name}>\n(.*?)\n</{name}>", prompt, re.S).group(1)

    def chat(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][-1]["content"]
        a, b = self._section(prompt, "candidate_a"), self._section(prompt, "candidate_b")
        a_workflow, pair_id = re.search(r"fixture:(\w+):(\S+) -->", a).groups()
        b_workflow = re.search(r"fixture:(\w+):", b).group(1)
        challenger = a_workflow if a_workflow != "w2_baseline" else b_workflow
        index = [p[0] for p in PAIRS].index(pair_id)
        outcome = PAIRWISE_SCRIPT.get((challenger, index))
        if outcome == "judge_failed":
            raise RuntimeError("judge unreachable")
        if outcome == "tie":
            winner = "tie"
        elif outcome == "inconsistent":
            winner = "candidate_a"  # position bias: always the first resume
        else:
            better = QUALITY[challenger] >= QUALITY["w2_baseline"]
            winner_workflow = challenger if better else "w2_baseline"
            winner = "candidate_a" if a_workflow == winner_workflow else "candidate_b"
        first = {name: self._section(prompt, name).splitlines()[0]
                 for name in ("original_resume", "job_description")}

        def dimension(extra=None):
            quotes = [{"source": "candidate_a", "quote": a.splitlines()[1]},
                      {"source": "candidate_b", "quote": b.splitlines()[1]}]
            if extra:
                quotes.insert(0, {"source": extra, "quote": first[extra]})
            return {"preferred": winner, "explanation": "Fixture.", "quotations": quotes}

        content = json.dumps({
            "winner": winner, "explanation": "Fixture judgement.",
            "relevance_to_jd": dimension("job_description"),
            "prioritisation_of_supported_experience": dimension("original_resume"),
            **{name: dimension() for name in DIMENSIONS
               if name not in {"relevance_to_jd", "prioritisation_of_supported_experience"}},
        })
        return SimpleNamespace(message=SimpleNamespace(content=content),
                               prompt_eval_count=10, eval_count=20)


def build_pairwise(root, rows):
    import evaluate

    pair_map = {
        pair_id: {"pair_id": pair_id, "resume": original_resume(pair_id),
                  "jd": job_description(pair_id), "category": category, "fit_level": fit}
        for pair_id, category, _, fit in PAIRS
    }
    judge_config = JudgeConfig(model="fixture-judge", max_retries=0,
                               min_request_interval_seconds=0)
    model_info = {"provider": "ollama", "tag": "fixture-judge", "digest": "d0"}
    return evaluate.evaluate_pairwise_stage(
        rows, pair_map, Path(root), judge_config, model_info,
        challenger_workflows=[w for w in {r["workflow"] for r in rows} if w != "w2_baseline"],
        client=ScriptedJudge(),
    )


ROBUSTNESS_IDENTITY = "rb0000000000beef"


def build_robustness_rows(workflows=("w1_single_pass", "w2_baseline", "w4_evidence"), seed=11):
    """Rows shaped like ``evaluate._robustness_record`` output (scored, not generated)."""
    rng = random.Random(seed)
    rows = []
    for workflow in workflows:
        for index in (0, 2):
            pair_id = PAIRS[index][0]
            base = score_row(workflow, index, rng)
            for variant, family in VARIANT_FAMILIES.items():
                scope = variant_scope(variant)
                row = dict(base)
                crashed = workflow == "w1_single_pass" and variant == "degenerate_jd" and index == 2
                failed = workflow == "w2_baseline" and variant == "pdf_extraction" and index == 2
                if variant != "original":
                    drop = rng.uniform(0, 0.12) * (1.5 if workflow == "w1_single_pass" else 0.6)
                    for metric in ("faithfulness", "claim_faithfulness"):
                        row[metric] = round(max(0.0, base[metric] - drop), 4)
                    row["unsupported_claim_rate"] = round(1 - row["faithfulness"], 4)
                    row["kw_after"] = round(max(0.0, base["kw_after"] - drop / 2), 4)
                if failed or crashed:
                    row.update({"error": "TOOL_BUDGET_EXCEEDED: x", "resume_md": "",
                                "faithfulness": None, "claim_faithfulness": None,
                                "unsupported_claim_rate": None, "task_success": None,
                                "requirement_coverage": None})
                if scope == "out_of_scope":
                    row["requirement_coverage"] = None
                    row["requirement_coverage_status"] = None
                refusing = workflow == "w4_evidence" and variant == "mismatched_jd"
                text = (
                    "I cannot tailor this resume to an unrelated job description."
                    if refusing else row["resume_md"]
                )
                row["resume_md"] = text
                row["generation_outcome"] = classify_generation(text, row.get("error"), crashed)
                row.update({
                    "base_pair_id": pair_id, "variant": variant, "variant_family": family,
                    "variant_scope": scope, "variant_metadata": {},
                    "pair_id": f"{pair_id}__{variant}",
                    "variant_version": config.ROBUSTNESS_VARIANT_VERSION,
                    "transformation_version": config.ROBUSTNESS_VARIANT_VERSION,
                    "robustness_identity_hash": ROBUSTNESS_IDENTITY,
                    "scoring_reference": "clean_original_resume",
                    "scoring_jd": "clean_original_jd" if scope == "in_scope" else "presented_jd",
                })
                if scope == "out_of_scope":
                    row.update(out_of_scope_expected_behaviour(row))
                rows.append(row)
    return rows


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_fixture_root(base, *, workflows=WORKFLOWS, pairwise=True, robustness=True,
                       summaries=True, evaluator_hash=EVALUATOR_HASH, bootstrap_samples=None):
    """Write a complete evaluation root under ``base`` and return its path.

    ``bootstrap_samples`` lowers ``config.REPORT_BOOTSTRAP_SAMPLES`` while the summaries are
    written (it changes CI precision only, not any column) to keep tests fast.
    """
    if bootstrap_samples is not None:
        saved = config.REPORT_BOOTSTRAP_SAMPLES
        config.REPORT_BOOTSTRAP_SAMPLES = bootstrap_samples
        try:
            return build_fixture_root(
                base, workflows=workflows, pairwise=pairwise, robustness=robustness,
                summaries=summaries, evaluator_hash=evaluator_hash,
            )
        finally:
            config.REPORT_BOOTSTRAP_SAMPLES = saved
    root = Path(base) / DATASET_HASH / GENERATION_SCOPE / evaluator_hash
    root.mkdir(parents=True, exist_ok=True)
    rows = build_score_rows(workflows)
    _write_jsonl(root / "evaluations.jsonl", rows)
    (root / "evaluator_identity.json").write_text(
        json.dumps({"evaluator_hash": evaluator_hash}) + "\n", encoding="utf-8"
    )
    stage_rows = {}
    if pairwise and "w2_baseline" in workflows and len(workflows) > 1:
        stage_rows["pairwise"] = build_pairwise(root, rows)
    if robustness:
        robust = build_robustness_rows([w for w in ("w1_single_pass", "w2_baseline",
                                                    "w4_evidence") if w in workflows])
        _write_jsonl(root / "robustness" / ROBUSTNESS_IDENTITY / "evaluations.jsonl", robust)
        stage_rows["robustness"] = robust
    if summaries:
        write_reports(root / "summaries", rows, stage_rows=stage_rows)
    return root


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "results_demo/evaluations")
    print(build_fixture_root(target))
