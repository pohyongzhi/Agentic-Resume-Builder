"""Stable V5 seven-variant registry and implementation status."""

from dataclasses import dataclass
from typing import Callable, Optional

import config
from workflows import (
    agentic_baseline,
    evidence_grounded_generation,
    fine_tuned_small_generator,
    multi_critic_review,
    multi_draft_selector,
    one_pass,
    planner_section_writers,
)


@dataclass(frozen=True)
class WorkflowSpec:
    number: int
    workflow_id: str
    display_name: str
    status: str
    model: str
    runner: Optional[Callable] = None


WORKFLOW_SPECS = {
    "one_pass": WorkflowSpec(
        1, "one_pass", one_pass.DISPLAY_NAME, one_pass.STATUS,
        config.BASE_MODEL, one_pass.run),
    "agentic_baseline": WorkflowSpec(
        2, "agentic_baseline", agentic_baseline.DISPLAY_NAME,
        agentic_baseline.STATUS, config.BASE_MODEL),
    "multi_critic_review": WorkflowSpec(
        3, "multi_critic_review", multi_critic_review.DISPLAY_NAME,
        multi_critic_review.STATUS, config.BASE_MODEL),
    "evidence_grounded_generation": WorkflowSpec(
        4, "evidence_grounded_generation", evidence_grounded_generation.DISPLAY_NAME,
        evidence_grounded_generation.STATUS, config.BASE_MODEL),
    "fine_tuned_small_generator": WorkflowSpec(
        5, "fine_tuned_small_generator", fine_tuned_small_generator.DISPLAY_NAME,
        fine_tuned_small_generator.STATUS, config.FINE_TUNED_MODEL,
        fine_tuned_small_generator.run),
    "multi_draft_selector": WorkflowSpec(
        6, "multi_draft_selector", multi_draft_selector.DISPLAY_NAME,
        multi_draft_selector.STATUS, config.BASE_MODEL),
    "planner_section_writers": WorkflowSpec(
        7, "planner_section_writers", planner_section_writers.DISPLAY_NAME,
        planner_section_writers.STATUS, config.BASE_MODEL),
}

# Planned workflows are visible in WORKFLOW_SPECS but cannot be executed.
WORKFLOWS = {
    workflow_id: spec.runner
    for workflow_id, spec in WORKFLOW_SPECS.items()
    if spec.status == "implemented" and spec.runner is not None
}
