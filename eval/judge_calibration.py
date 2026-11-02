"""Fail-closed calibration for the shared rubric judge."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import config
from eval.judge import JudgeConfig, configured_model, evaluate_judge, judge_identity

CALIBRATION_VERSION = "resume-judge-calibration-v1"

ORIGINAL_RESUME = """## Summary
Operations Coordinator supporting order tracking, supplier follow-up and monthly reporting.

## Experience
### Operations Coordinator | Harbor Lane Foods | Singapore | January 2022 to June 2026
- Maintained an Excel tracker for 35 open orders and followed up delivery dates with suppliers twice weekly.
- Prepared a monthly exception report for the operations manager, who approved schedule changes.
- Reconciled delivery notes against the order register and escalated missing documents.

## Education
Diploma in Business Operations | Bayview Polytechnic | 2021

## Skills
Excel; order tracking; supplier follow-up; delivery-document reconciliation.
"""

JOB_DESCRIPTION = """Operations Coordinator

Coordinate open orders, follow up supplier delivery dates and maintain accurate Excel trackers.
Prepare exception reports and escalate scheduling decisions to the operations manager. Clear written
communication is essential. ERP administration and a PMP qualification are preferred.
"""

STRONG = """## Summary
Operations Coordinator experienced in open-order tracking, supplier follow-up and exception reporting.

## Experience
### Operations Coordinator | Harbor Lane Foods | Singapore | January 2022 to June 2026
- Maintained an Excel tracker for 35 open orders and followed up supplier delivery dates twice weekly.
- Prepared a monthly exception report for the operations manager, who approved schedule changes.
- Reconciled delivery notes against the order register and escalated missing documents.

## Education
Diploma in Business Operations | Bayview Polytechnic | 2021

## Skills
Excel; order tracking; supplier follow-up; delivery-document reconciliation.
"""

GENERIC = """## Summary
Hard-working professional seeking an opportunity to contribute transferable skills and grow with a
dynamic organisation.

## Experience
### Operations Coordinator | Harbor Lane Foods | Singapore | January 2022 to June 2026
- Supported daily operational activities and worked with different stakeholders.
- Helped prepare reports and maintained records.

## Education
Diploma in Business Operations | Bayview Polytechnic | 2021

## Skills
Communication; teamwork; organisation.
"""

UNSUPPORTED = """## Summary
Senior Operations Manager and PMP-certified ERP expert who consistently transforms global supply chains.

## Experience
### Operations Coordinator | Harbor Lane Foods | Singapore | January 2022 to June 2026
- Led a team of 20 and independently approved supplier contracts worth $2 million.
- Implemented an enterprise ERP platform and reduced operating costs by 40%.
- Maintained an Excel tracker for 35 open orders.

## Education
Diploma in Business Operations | Bayview Polytechnic | 2021

## Skills
PMP; enterprise ERP implementation; contract negotiation; global supply-chain leadership.
"""

PADDED = STRONG.replace(
    "## Education",
    """- Maintained an Excel tracker for open orders as part of ongoing operational tracking activities.
- Followed up suppliers about delivery dates as part of supplier follow-up work.
- Prepared reports because reporting was an important part of the role.
- Worked hard, communicated well and always aimed to deliver excellent operational support.

## Education""",
)

CONTROLS = (
    {"id": "strong", "tailored_resume": STRONG},
    {"id": "generic", "tailored_resume": GENERIC},
    {"id": "unsupported", "tailored_resume": UNSUPPORTED},
    {"id": "padded", "tailored_resume": PADDED},
)


def _score(record: dict[str, Any], dimension: str | None = None) -> float | None:
    if record.get("status") != "scored":
        return None
    if dimension is None:
        return record["result"]["overall_mean"]
    return record["result"]["dimensions"][dimension]["score"]


def calibration_checks(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    strong = _score(records["strong"])
    generic = _score(records["generic"])
    unsupported = _score(records["unsupported"])
    padded = _score(records["padded"])
    values = {
        "all_controls_scored": all(value is not None for value in (strong, generic, unsupported, padded)),
        "strong_is_high_quality": strong is not None and strong >= 4.0,
        "generic_is_not_high_quality": generic is not None and generic <= 3.4,
        "unsupported_is_penalised": (
            unsupported is not None
            and unsupported <= 3.0
            and (_score(records["unsupported"], "prioritisation_of_supported_experience") or 5) <= 2
        ),
        "quality_order_resolves": (
            None not in (strong, generic, unsupported) and strong > generic > unsupported
        ),
        "padding_is_not_rewarded": (
            None not in (strong, padded)
            and padded <= strong
            and (_score(records["padded"], "concision") or 5)
            < (_score(records["strong"], "concision") or 0)
        ),
    }
    return [{"name": name, "passed": bool(passed)} for name, passed in values.items()]


def run_calibration(
    output_dir: str | Path,
    judge_config: JudgeConfig,
    client=None,
    judge_fn: Callable[..., dict[str, Any]] = evaluate_judge,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    result_path = output_dir / "calibration-result.json"
    if result_path.exists():
        raise RuntimeError(f"CALIBRATION_OUTPUT_EXISTS: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    for control in CONTROLS:
        records[control["id"]] = judge_fn(
            ORIGINAL_RESUME,
            JOB_DESCRIPTION,
            control["tailored_resume"],
            output_dir / "cache",
            judge_config=judge_config,
            client=client,
        )

    checks = calibration_checks(records)
    model_info = configured_model(judge_config, client)
    attempts = [attempt for record in records.values() for attempt in record.get("attempts", [])]
    result = {
        "calibration_version": CALIBRATION_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "scores": {
            control_id: {
                "status": record["status"],
                "overall_mean": _score(record),
                "dimensions": (
                    {name: value["score"] for name, value in record["result"]["dimensions"].items()}
                    if record.get("result") else None
                ),
                "cache_key": record["cache_key"],
            }
            for control_id, record in records.items()
        },
        "usage": {
            "requests": len(attempts),
            "prompt_tokens": sum(a.get("prompt_tokens") or 0 for a in attempts),
            "output_tokens": sum(a.get("output_tokens") or 0 for a in attempts),
            "reasoning_tokens": sum(a.get("reasoning_tokens") or 0 for a in attempts),
            "total_tokens": sum(a.get("total_tokens") or 0 for a in attempts),
        },
        "judge_config": asdict(judge_config),
        "judge_identity": judge_identity(judge_config, model_info),
        "model_info": model_info,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate the resume rubric judge before bulk evaluation")
    parser.add_argument("--provider", choices=("ollama", "groq"), default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--output-dir")
    parser.add_argument("--execute", action="store_true", help="make the four judge requests")
    args = parser.parse_args(argv)

    config_value = JudgeConfig(
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    dry_run = {
        "execute": args.execute,
        "requests": len(CONTROLS),
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "controls": [control["id"] for control in CONTROLS],
    }
    if not args.execute:
        print(json.dumps(dry_run, indent=2))
        return None
    if not args.output_dir:
        raise SystemExit("--output-dir is required with --execute")
    result = run_calibration(args.output_dir, config_value)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("JUDGE_CALIBRATION_FAILED")
    return result


if __name__ == "__main__":
    main()
