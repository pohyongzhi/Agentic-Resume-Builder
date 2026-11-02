"""W3 - MULTI-CRITIC REVISION LOOP.

An initial structured draft is reviewed in parallel by a fact-checker, recruiter persona and
formatting reviewer. The creator receives their combined structured feedback and may revise twice.
The latest draft must still pass the shared deterministic harness before it can be rendered.
"""
import json
import time
from typing import TypedDict

from pydantic import ValidationError

from langgraph.graph import END, START, StateGraph

import config
from common.harness import ToolHarness, validate_resume
from common.llm import chat
from common.pipeline import run_pipeline
from common.prompts import STRUCTURED_CREATOR_SYSTEM, creator_prompt
from common.schemas import CriticFeedback, GeneratedResume, llm_schema

NAME = "w3_critic"

FACT_SYSTEM = """You are a resume fact-checker. Compare every claim in the draft with the
candidate data and original resume. Flag unsupported skills, tools, achievements, numbers,
employers, titles, dates, degrees, and strengthened claims. Return only the required structured
review. Approve only when every claim is supported."""

RECRUITER_SYSTEM = """You are a recruiter screening this resume for the supplied job. Review
relevance, evidence, prioritisation, clarity, and whether the strongest supported qualifications
are easy to find. Never recommend inventing a missing qualification. Return only the required
structured review. Approve only when there are no material shortlist weaknesses the candidate's
existing evidence can fix."""

FORMATTING_SYSTEM = """You are a resume formatting and copy reviewer. Check section completeness,
consistent chronology and wording, concise bullets, scanability, duplication, and an approximate
350-500 word target. Review content structure, not visual styling. Return only the required
structured review. Approve only when no material formatting or readability issue remains."""


class CriticState(TypedDict, total=False):
    candidate_data: dict
    job_data: dict
    draft: dict
    revisions: int
    fact_feedback: dict
    recruiter_feedback: dict
    formatting_feedback: dict
    combined_feedback: list[dict]
    all_approved: bool
    resume_md: str
    output_path: str
    generated_resume: dict
    hit_budget: bool
    errors: list[str]


def _structured(system, user, model, tracker, forced_fields=None):
    is_review = model is CriticFeedback
    limit = config.W3_CRITIC_MAX_TOKENS if is_review else config.W3_DRAFT_MAX_TOKENS
    instruction = (
        " Return at most five distinct issues, with short evidence and one concrete recommendation "
        "per issue. Do not repeat issues. Stop after the JSON object."
        if is_review else
        " Return one complete concise resume JSON object. Do not repeat entries or add commentary. "
        "Use [] for missing certifications and other empty lists; never write placeholders such as "
        "None listed or Not applicable as a list item. Retain the candidate education entries."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user + instruction}]
    failures = []
    for attempt in range(config.W3_STRUCTURED_RETRIES + 1):
        msg = chat(messages, tracker, format=llm_schema(model), num_predict=limit)
        raw = msg.content or ""
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("expected one JSON object")
            if forced_fields:
                data.update(forced_fields)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            detail = str(exc)[:1000]
            failures.append(detail)
            tracker.log(type="structured_output", schema=model.__name__,
                        critic=(forced_fields or {}).get("critic"), attempt=attempt + 1,
                        ok=False, error=detail, response_chars=len(raw), raw_response=raw)
            # Start a fresh bounded request rather than feeding a runaway response back in.
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user + instruction +
                         "\n\nYour previous response failed validation: " + detail +
                         "\nReturn a shorter, complete JSON object matching the schema. "
                         "Preserve source facts; do not repair or continue the previous JSON."}]
    raise ValueError(f"{model.__name__} failed after {len(failures)} attempts: {failures[-1]}")


def _prepare_draft(draft, candidate):
    """Remove explicit absence markers and retain source education when a writer omits it.

    This does not remove unknown credentials or alter supplied education facts: the harness
    must still reject those when they are unsupported.
    """
    draft = dict(draft)
    absence_markers = {
        "none", "n/a", "not applicable", "not provided", "none listed",
        "none listed in original resume data", "no certifications listed",
    }
    draft["certifications"] = [
        value for value in draft.get("certifications", [])
        if value.strip().rstrip(".").casefold() not in absence_markers
    ]
    if not draft.get("education") and candidate.get("education"):
        draft["education"] = candidate["education"]
    return draft


def _context(candidate, job, draft):
    return (
        "JOB:\n" + json.dumps(job, indent=2, ensure_ascii=False)
        + "\n\nCANDIDATE AND ORIGINAL RESUME:\n"
        + json.dumps(candidate, indent=2, ensure_ascii=False)
        + "\n\nDRAFT RESUME:\n"
        + json.dumps(draft, indent=2, ensure_ascii=False)
    )


def _tracked_execute(harness, tracker, name, args):
    start = time.time()
    ok, result, stop = harness.execute(name, args)
    tracker.log(type="tool", name=name, args=args, ok=ok, seconds=round(time.time() - start, 3))
    return ok, result, stop


def build_critic_graph(tracker, resume_path=None, run_id=None):
    """Build the W3 creator graph. Exposed so graph behavior can be tested independently."""

    def create_draft(state):
        draft = _structured(
            STRUCTURED_CREATOR_SYSTEM,
            creator_prompt(state["candidate_data"], state["job_data"]),
            GeneratedResume,
            tracker,
        )
        return {"draft": _prepare_draft(draft.model_dump(), state["candidate_data"]), "revisions": 0}

    def critic_node(name, system, output_key):
        def review(state):
            feedback = _structured(
                system,
                _context(state["candidate_data"], state["job_data"], state["draft"]),
                CriticFeedback,
                tracker,
                forced_fields={"critic": name},
            )
            review_data = feedback.model_dump()
            if name == "fact_checker":
                # Feed the same protected-fact checks used by finalisation into the revision loop.
                errors = validate_resume(state["draft"], state["candidate_data"], check_skills=False)
                if errors:
                    review_data["approved"] = False
                    review_data["issues"] = [{
                        "category": "protected_facts", "severity": "critical",
                        "evidence": error,
                        "recommendation": "Restore the source facts or remove unsupported content.",
                    } for error in errors] + review_data["issues"]
            return {output_key: review_data}
        return review

    def aggregate(state):
        reviews = [
            state["fact_feedback"],
            state["recruiter_feedback"],
            state["formatting_feedback"],
        ]
        blocking = any(
            issue["severity"] in ("major", "critical")
            for review in reviews
            for issue in review.get("issues", [])
        )
        approved = all(review.get("approved", False) for review in reviews) and not blocking
        return {"combined_feedback": reviews, "all_approved": approved}

    def route_after_review(state):
        if state["all_approved"] or state["revisions"] >= config.MAX_CRITIC_ROUNDS:
            return "finalise"
        return "revise"

    def revise(state):
        prompt = (
            creator_prompt(state["candidate_data"], state["job_data"])
            + "\n\nCURRENT DRAFT:\n"
            + json.dumps(state["draft"], indent=2, ensure_ascii=False)
            + "\n\nCRITIC FEEDBACK:\n"
            + json.dumps(state["combined_feedback"], indent=2, ensure_ascii=False)
            + "\n\nReturn the complete revised resume JSON. Address every supported recommendation, "
              "but never add facts that are absent from the original resume."
        )
        draft = _structured(
            STRUCTURED_CREATOR_SYSTEM,
            prompt,
            GeneratedResume,
            tracker,
        )
        return {"draft": _prepare_draft(draft.model_dump(), state["candidate_data"]),
                "revisions": state["revisions"] + 1}

    def finalise(state):
        harness = ToolHarness(state["candidate_data"], resume_path=resume_path, run_id=run_id,
                              check_skills=False)
        _, result, _ = _tracked_execute(
            harness, tracker, "validate_resume", {"resume": state["draft"]}
        )
        validation = json.loads(result)
        if not validation.get("valid"):
            return {
                "errors": ["W3_FINAL_VALIDATION_FAILED: " + "; ".join(validation.get("errors", []))],
                "hit_budget": (
                    not state["all_approved"]
                    and state["revisions"] >= config.MAX_CRITIC_ROUNDS
                ),
            }
        _tracked_execute(harness, tracker, "render_resume", {})
        outcome = harness.outcome()
        outcome["hit_budget"] = (
            not state["all_approved"] and state["revisions"] >= config.MAX_CRITIC_ROUNDS
        )
        return outcome

    graph = StateGraph(CriticState)
    graph.add_node("create_draft", create_draft)
    graph.add_node("fact_checker", critic_node("fact_checker", FACT_SYSTEM, "fact_feedback"))
    graph.add_node("recruiter", critic_node("recruiter", RECRUITER_SYSTEM, "recruiter_feedback"))
    graph.add_node(
        "formatting_reviewer",
        critic_node("formatting", FORMATTING_SYSTEM, "formatting_feedback"),
    )
    graph.add_node("aggregate", aggregate)
    graph.add_node("revise", revise)
    graph.add_node("finalise", finalise)

    graph.add_edge(START, "create_draft")
    for node in ("fact_checker", "recruiter", "formatting_reviewer"):
        graph.add_edge("create_draft", node)
        graph.add_edge("revise", node)
    graph.add_edge(["fact_checker", "recruiter", "formatting_reviewer"], "aggregate")
    graph.add_conditional_edges("aggregate", route_after_review, ["revise", "finalise"])
    graph.add_edge("finalise", END)
    return graph.compile()


def make_creator(tracker, resume_path=None, run_id=None):
    def resume_creator(state):
        try:
            graph = build_critic_graph(tracker, resume_path=resume_path, run_id=run_id)
            result = graph.invoke(
                {
                    "candidate_data": state["candidate_data"],
                    "job_data": state["job_data"],
                },
                {"recursion_limit": 50},
            )
            return {
                key: result[key]
                for key in (
                    "resume_md",
                    "output_path",
                    "generated_resume",
                    "hit_budget",
                    "errors",
                )
                if key in result
            }
        except Exception as exc:
            return {"errors": [f"W3_STRUCTURED_OUTPUT_FAILED: {type(exc).__name__}: {exc}"]}
    return resume_creator


def generate(tracker, resume_path, job_url="", pasted_jd="", run_id=None):
    return run_pipeline(
        make_creator,
        tracker,
        resume_path,
        job_url=job_url,
        pasted_jd=pasted_jd,
        run_id=run_id,
    )


def run(pair, tracker):
    state = generate(
        tracker,
        pair["resume_path"],
        job_url=pair.get("job_url", ""),
        pasted_jd=pair["jd"],
        run_id=f"{NAME}__{pair['pair_id']}",
    )
    return {
        "resume_md": state.get("resume_md", ""),
        "hit_budget": bool(state.get("hit_budget")),
        "finished": bool(state.get("resume_md")),
        "errors": state.get("errors", []),
        "output_path": state.get("output_path"),
    }
