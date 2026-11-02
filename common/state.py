"""Shared LangGraph state for the resume pipeline.

Each extraction agent writes only its own field (candidate_data / job_data), so they can run in
parallel. `errors` has a reducer so both parallel nodes can append to it in the same step."""
import operator
from typing import Annotated, TypedDict


class ResumeState(TypedDict, total=False):
    resume_path: str
    job_url: str
    pasted_jd: str
    candidate_data: dict
    job_data: dict
    generated_resume: dict
    resume_md: str
    output_path: str
    hit_budget: bool
    errors: Annotated[list[str], operator.add]
    evidence: list  # W4 evidence report (unused by other workflows)
    w4_stats: dict  # W4 summary numbers (unused by other workflows)
