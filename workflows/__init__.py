"""Stable V1 workflow registry and implementation status."""

from dataclasses import dataclass
from typing import Callable, Optional

import config
from workflows import (
    baseline,
    evidence_grounded_generation,
    fine_tuned_small_generator,
    multi_critic_review,
    multi_draft_selector,
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
    "baseline": WorkflowSpec(
        1, "baseline", baseline.DISPLAY_NAME, baseline.STATUS,
        config.BASE_MODEL, baseline.run),
    "multi_critic_review": WorkflowSpec(
        2, "multi_critic_review", multi_critic_review.DISPLAY_NAME,
        multi_critic_review.STATUS, config.BASE_MODEL),
    "evidence_grounded_generation": WorkflowSpec(
        3, "evidence_grounded_generation", evidence_grounded_generation.DISPLAY_NAME,
        evidence_grounded_generation.STATUS, config.BASE_MODEL),
    "fine_tuned_small_generator": WorkflowSpec(
        4, "fine_tuned_small_generator", fine_tuned_small_generator.DISPLAY_NAME,
        fine_tuned_small_generator.STATUS, config.FINE_TUNED_MODEL,
        fine_tuned_small_generator.run),
    "multi_draft_selector": WorkflowSpec(
        5, "multi_draft_selector", multi_draft_selector.DISPLAY_NAME,
        multi_draft_selector.STATUS, config.BASE_MODEL),
    "planner_section_writers": WorkflowSpec(
        6, "planner_section_writers", planner_section_writers.DISPLAY_NAME,
        planner_section_writers.STATUS, config.BASE_MODEL),
}

# Planned workflows are visible in WORKFLOW_SPECS but cannot be executed.
WORKFLOWS = {
    workflow_id: spec.runner
    for workflow_id, spec in WORKFLOW_SPECS.items()
    if spec.status == "implemented" and spec.runner is not None
}
