"""Operational evaluator metrics shared by online runs and offline rescoring."""
import re

import config

REQUIRED_SECTIONS = ("summary", "experience", "education", "skills")


def _workflow_spec(workflow):
    """Expected work is workflow-specific; extra intended critic steps are not treated as waste."""
    specs = {
        "w1_single_pass": {
            "tools": set(),
            "sequence": [],
            "min_steps": 1,
            "max_steps": 1,
        },
        "w2_baseline": {
            "tools": {"validate_resume", "render_resume"},
            "sequence": ["validate_resume", "render_resume"],
            "min_steps": 5,  # two extractors, one creator call, validate, render
            "max_steps": 2 + (2 * config.MAX_TOOL_CALLS),
        },
        "w3_critic": {
            "tools": {"validate_resume", "render_resume"},
            "sequence": ["validate_resume", "render_resume"],
            "min_steps": 8,  # extractors, draft, three critics, validate, render
            "max_steps": (
                2 + 1 + (3 * (config.MAX_CRITIC_ROUNDS + 1))
                + config.MAX_CRITIC_ROUNDS + 2
            ),
        },
    }
    return specs.get(workflow, {"tools": None, "sequence": [], "min_steps": 1, "max_steps": None})


def _has_section(markdown, section):
    pattern = rf"^(?:#+\s*)?{re.escape(section)}(?:\s*&\s*certifications)?\s*$"
    return re.search(pattern, markdown or "", flags=re.IGNORECASE | re.MULTILINE) is not None


def _tool_metrics(workflow, trace):
    spec = _workflow_spec(workflow)
    events = [event for event in (trace or []) if event.get("type") == "tool"]
    if spec["tools"] == set():
        return {
            "tool_applicable": False,
            "tool_call_accuracy": None,
            "tool_argument_accuracy": None,
            "tool_sequence_valid": None,
            "tool_recovery_rate": None,
        }
    if spec["tools"] is None:
        return {
            "tool_applicable": bool(events),
            "tool_call_accuracy": None,
            "tool_argument_accuracy": None,
            "tool_sequence_valid": None,
            "tool_recovery_rate": None,
        }

    attempted = len(events)
    allowed_and_ok = sum(
        event.get("name") in spec["tools"] and bool(event.get("ok")) for event in events
    )
    arguments_ok = sum(bool(event.get("ok")) for event in events)
    names = [event.get("name") for event in events]
    cursor = 0
    for required in spec["sequence"]:
        try:
            cursor = names.index(required, cursor) + 1
        except ValueError:
            cursor = -1
            break

    failures = [index for index, event in enumerate(events) if not event.get("ok")]
    recovered = sum(
        any(
            later.get("name") == events[index].get("name") and later.get("ok")
            for later in events[index + 1:]
        )
        for index in failures
    )
    return {
        "tool_applicable": True,
        "tool_call_accuracy": round(allowed_and_ok / attempted, 4) if attempted else 0.0,
        "tool_argument_accuracy": round(arguments_ok / attempted, 4) if attempted else 0.0,
        "tool_sequence_valid": cursor != -1,
        "tool_recovery_rate": round(recovered / len(failures), 4) if failures else None,
    }


def evaluate_operational(workflow, pair, output, totals, trace, scored_row=None, error=None):
    """Return task, tool, step and cost metrics for one workflow run."""
    scored_row = scored_row or {}
    markdown = output.get("resume_md", "") or ""
    words = len(markdown.split())
    section_hits = {section: _has_section(markdown, section) for section in REQUIRED_SECTIONS}
    structure_pass = all(section_hits.values()) and config.TASK_MIN_WORDS <= words <= config.TASK_MAX_WORDS
    faithfulness = scored_row.get("faithfulness")
    protected_field_violation = scored_row.get("protected_field_violation")
    fabrication_bait_violation = scored_row.get("fabrication_bait_violation")
    finished = bool(output.get("finished", markdown.strip())) and not error and not output.get("errors")

    reasons = []
    if not finished:
        reasons.append("workflow_failed")
    if not all(section_hits.values()):
        reasons.append("missing_required_sections")
    if markdown and not (config.TASK_MIN_WORDS <= words <= config.TASK_MAX_WORDS):
        reasons.append("word_count_out_of_range")
    if faithfulness is None:
        reasons.append("faithfulness_not_scored")
    if protected_field_violation is None:
        reasons.append("protected_fields_not_scored")
    if fabrication_bait_violation is None:
        reasons.append("fabrication_bait_not_scored")
    if any(value is None for value in (
        faithfulness, protected_field_violation, fabrication_bait_violation,
    )):
        task_success = None
    else:
        if faithfulness < config.TASK_FAITHFULNESS_THRESHOLD:
            reasons.append("faithfulness_below_threshold")
        if protected_field_violation:
            reasons.append("protected_field_violation")
        if fabrication_bait_violation:
            reasons.append("fabrication_bait_violation")
        task_success = (
            finished
            and structure_pass
            and faithfulness >= config.TASK_FAITHFULNESS_THRESHOLD
            and not protected_field_violation
            and not fabrication_bait_violation
        )

    spec = _workflow_spec(workflow)
    actual_steps = totals.get("llm_calls", 0) + totals.get("tool_calls", 0)
    min_steps = spec["min_steps"]
    max_steps = spec["max_steps"]
    total_tokens = totals.get("input_tokens", 0) + totals.get("output_tokens", 0)
    quality_values = [
        scored_row.get(name)
        for name in ("sim_after", "kw_after", "faithfulness")
        if scored_row.get(name) is not None
    ]
    quality_proxy = sum(quality_values) / len(quality_values) if quality_values else None

    metrics = {
        "held_out": pair.get("split") == "test",
        "required_sections_rate": round(sum(section_hits.values()) / len(REQUIRED_SECTIONS), 4),
        "structure_pass": structure_pass,
        "task_success": task_success,
        "task_failure_reasons": reasons,
        "agent_steps": actual_steps,
        "expected_min_steps": min_steps,
        "step_efficiency": round(min(min_steps / actual_steps, 1.0), 4) if actual_steps else 0.0,
        "budget_utilisation": round(actual_steps / max_steps, 4) if max_steps else None,
        "total_tokens": total_tokens,
        "tokens_per_successful_task": total_tokens if task_success else None,
        "seconds_per_successful_task": totals.get("seconds") if task_success else None,
        "cost_quality_ratio": (
            round(total_tokens / quality_proxy, 2) if quality_proxy and quality_proxy > 0 else None
        ),
    }
    metrics.update(_tool_metrics(workflow, trace))
    return metrics
