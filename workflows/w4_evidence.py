"""W4 - Evidence-grounded generation: every claim must cite and be supported by source spans."""
import json
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from common.grounding import (
    check_citations,
    drop_unsupported,
    filter_skills,
    iter_claims,
    llm_verify,
    rule_check,
    to_generated,
)
from common.harness import ToolHarness, _norm
from common.llm import chat
from common.pipeline import run_pipeline
from common.prompts import STRUCTURED_CREATOR_SYSTEM, creator_prompt
from common.schemas import GroundedResume, llm_schema
from common.spans import citeable, render_spans_for_prompt, split_spans
from common.state import ResumeState

NAME = "w4_evidence"
MAX_GROUNDING_ROUNDS = 2

W4_CITATION_RULES = """
CITATION RULES (mandatory):
- Every summary sentence, experience bullet, education detail and project bullet is an object
  {"text": ..., "sources": [...]}. "sources" lists the IDs (e.g. "S7") of the source spans the
  sentence is based on. Never leave "sources" empty.
- A bullet under a job may only cite spans listed under that same job. Summary sentences may cite
  any span.
- Cite at most 3 spans per sentence.
- The sentence may reword, shorten or combine its cited spans, but must not add anything they don't
  say: no new tools, skills, numbers, scope, seniority ("led", "advanced", "expert") or outcomes.
- If the job asks for something no span supports, leave it out. Do not cite a span that doesn't
  actually say it."""

SYSTEM = STRUCTURED_CREATOR_SYSTEM + "\n\n" + W4_CITATION_RULES


class EvidenceState(TypedDict, total=False):
    candidate_data: dict
    job_data: dict
    spans: list[dict]  # Span.to_dict()
    draft: dict  # GroundedResume.model_dump()
    round: int
    problems: dict  # claim_id -> list[str]
    verdict_cache: dict  # cache_key -> verdict dict
    first_pass: dict  # claim_id -> round in which it first passed
    first_draft_stats: dict
    resume_md: str
    output_path: str
    generated_resume: dict
    evidence: list
    w4_stats: dict
    hit_budget: bool
    errors: list[str]


def draft_prompt(candidate, job, spans) -> str:
    """Build the creator prompt with numbered source spans instead of raw text."""
    # Start with the standard creator prompt structure
    job_fields = {k: v for k, v in job.items() if k != "source_text"}
    data = {k: v for k, v in candidate.items() if k != "raw_text"}

    # Build prompt with source spans
    prompt = f"""JOB (extracted):
{json.dumps(job_fields, indent=2, ensure_ascii=False)}

CANDIDATE DATA (extracted from the resume):
{json.dumps(data, indent=2, ensure_ascii=False)}

{render_spans_for_prompt(spans, candidate)}

Tailor the resume to this job. Cite every claim with its source span IDs."""

    return prompt


def revise_prompt(candidate, job, spans, draft, problems) -> str:
    """Build the revise prompt with the draft and rejected claims."""
    base = draft_prompt(candidate, job, spans)

    # Build the rejected claims list
    rejected_lines = []
    for claim_id, issues in sorted(problems.items()):
        if issues:
            # Get the claim text
            claim_obj = None
            for c in iter_claims(GroundedResume.model_validate(draft)):
                if c["claim_id"] == claim_id:
                    claim_obj = c
                    break
            if claim_obj:
                claim_text = claim_obj["claim"].text
                sources_str = ", ".join(claim_obj["claim"].sources)
                issue_str = "; ".join(issues)
                rejected_lines.append(f'- {claim_id} "{claim_text}" cites [{sources_str}]: {issue_str}')

    rejected_section = "\n".join(rejected_lines)

    return f"""{base}

CURRENT DRAFT (JSON):
{json.dumps(draft, indent=2, ensure_ascii=False)}

REJECTED CLAIMS – fix each one by rewording it so it only says what its cited spans say, citing
the correct spans, or delete it if no span supports it:
{rejected_section}
Keep every claim that is not listed exactly as it is. Return the complete resume JSON."""


def _timed(tracker, node_name):
    """Decorator to time and log a node's execution."""
    def decorator(func):
        def wrapper(state):
            start = time.time()
            result = func(state)
            end = time.time()
            tracker.log(
                type="node",
                workflow=NAME,
                node=node_name,
                start=start,
                end=end,
                seconds=round(end - start, 3),
            )
            return result
        return wrapper
    return decorator


def _tracked_execute(harness, tracker, name, args):
    """Execute a harness operation and log it."""
    start = time.time()
    ok, result, stop = harness.execute(name, args)
    tracker.log(type="tool", name=name, args=args, ok=ok, seconds=round(time.time() - start, 3))
    return ok, result, stop


def build_evidence_graph(tracker, resume_path=None, run_id=None):
    """Build the W4 evidence-grounding LangGraph."""

    @_timed(tracker, "split_spans")
    def split_spans_node(state):
        raw_text = state["candidate_data"]["raw_text"]
        spans = split_spans(raw_text, state["candidate_data"])
        cite_spans = citeable(spans)

        if not cite_spans:
            return {
                "errors": [f"W4_NO_SPANS: no citeable content in resume (only {len(spans)} total spans)"],
            }

        return {
            "spans": [s.to_dict() for s in spans],
            "round": 0,
            "problems": {},
            "verdict_cache": {},
            "first_pass": {},
        }

    @_timed(tracker, "draft")
    def draft_node(state):
        from common.spans import Span

        spans = [Span(**s) for s in state["spans"]]
        prompt = draft_prompt(state["candidate_data"], state["job_data"], spans)

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            msg = chat(messages, tracker, format=llm_schema(GroundedResume))
            draft = GroundedResume.model_validate_json(msg.content or "{}")
            return {"draft": draft.model_dump()}
        except Exception as e:
            # Retry once
            messages.append({"role": "user", "content": f"Validation error: {e}\n\nReturn the complete valid resume JSON."})
            try:
                msg = chat(messages, tracker, format=llm_schema(GroundedResume))
                draft = GroundedResume.model_validate_json(msg.content or "{}")
                return {"draft": draft.model_dump()}
            except Exception as e2:
                return {"errors": [f"W4_DRAFT_FAILED: {type(e2).__name__}: {e2}"]}

    @_timed(tracker, "check")
    def check_node(state):
        from common.spans import Span

        draft = GroundedResume.model_validate(state["draft"])
        spans = [Span(**s) for s in state["spans"]]
        candidate = state["candidate_data"]
        job = state["job_data"]

        # Citation checks
        problems = check_citations(draft, spans, candidate)

        # Rule checks on claims with no citation problems
        job_text = job.get("source_text", "") + " " + " ".join(
            job.get("responsibilities", []) + job.get("required_qualifications", []) +
            job.get("preferred_qualifications", [])
        )

        span_by_id = {s.id: s for s in spans}

        for item in iter_claims(draft):
            claim_id = item["claim_id"]
            claim = item["claim"]

            if problems[claim_id]:
                # Already has citation problems, skip rule check
                continue

            # Build cited_text
            cited_texts = []
            for source_id in claim.sources:
                span = span_by_id.get(source_id)
                if span and span.kind == "content":
                    cited_texts.append(span.text)

            cited_text = " ".join(cited_texts)
            rule_problems = rule_check(claim.text, cited_text, job_text)
            problems[claim_id].extend(rule_problems)

        return {"problems": problems}

    @_timed(tracker, "llm_verify")
    def llm_verify_node(state):
        from common.spans import Span

        draft = GroundedResume.model_validate(state["draft"])
        spans = [Span(**s) for s in state["spans"]]
        problems = state["problems"]
        cache = state["verdict_cache"]
        first_pass = state.get("first_pass", {})
        current_round = state["round"]

        # Build items for LLM verification (claims with no problems so far)
        span_by_id = {s.id: s for s in spans}
        items = []

        for item in iter_claims(draft):
            claim_id = item["claim_id"]
            claim = item["claim"]

            if problems[claim_id]:
                continue  # Already has problems

            # Build cited_text
            cited_texts = []
            for source_id in claim.sources:
                span = span_by_id.get(source_id)
                if span and span.kind == "content":
                    cited_texts.append(span.text)

            cited_text = " ".join(cited_texts)
            items.append({
                "claim_id": claim_id,
                "claim": claim,
                "cited_text": cited_text,
            })

        # Call LLM verifier
        if items:
            verdicts = llm_verify(items, tracker, cache)

            # Add not_entailed for unsupported claims
            for item in items:
                claim_id = item["claim_id"]
                verdict = verdicts.get(claim_id, {"supported": False, "reason": "verifier gave no verdict"})
                if not verdict["supported"]:
                    reason = verdict.get("reason", "not supported")
                    problems[claim_id].append(f"not_entailed: {reason}")

        # Track which claims passed for the first time
        for item in iter_claims(draft):
            claim_id = item["claim_id"]
            if not problems[claim_id] and claim_id not in first_pass:
                first_pass[claim_id] = current_round

        # Calculate stats for this round
        total_claims = len(problems)
        cited_ok = sum(1 for cid, probs in problems.items()
                       if not any(p.startswith(("no_citation", "bad_citation", "too_many_sources", "wrong_entry", "wrong_section")) for p in probs))
        rule_ok = sum(1 for cid, probs in problems.items()
                      if not any(p.startswith(("new_number", "new_skill", "inflated")) for p in probs))
        llm_checked = len(items)
        llm_cached = len([i for i in items if cache_key(i["claim"]) in cache]) if current_round > 0 else 0
        supported = sum(1 for probs in problems.values() if not probs)
        rejected_by_code = {}
        for probs in problems.values():
            for prob in probs:
                code = prob.split(":")[0]
                rejected_by_code[code] = rejected_by_code.get(code, 0) + 1

        tracker.log(
            type="w4_round",
            round=current_round,
            claims=total_claims,
            cited_ok=cited_ok,
            rule_ok=rule_ok,
            llm_checked=llm_checked,
            llm_cached=llm_cached,
            supported=supported,
            rejected_by_code=rejected_by_code,
        )

        # Store first draft stats
        first_draft_stats = state.get("first_draft_stats")
        if current_round == 0:
            first_draft_stats = {
                "citation_coverage": cited_ok / total_claims if total_claims > 0 else 0,
                "supported_rate": supported / total_claims if total_claims > 0 else 0,
            }

        return {
            "problems": problems,
            "verdict_cache": cache,
            "first_pass": first_pass,
            "first_draft_stats": first_draft_stats or state.get("first_draft_stats", {}),
        }

    def decide(state):
        """Route after verification."""
        problems = state["problems"]
        has_problems = any(probs for probs in problems.values())

        if not has_problems:
            return "finalise"
        elif state["round"] < MAX_GROUNDING_ROUNDS:
            return "revise"
        else:
            return "finalise"

    @_timed(tracker, "revise")
    def revise_node(state):
        from common.spans import Span

        draft = GroundedResume.model_validate(state["draft"])
        spans = [Span(**s) for s in state["spans"]]
        problems = state["problems"]
        candidate = state["candidate_data"]
        job = state["job_data"]

        # Build revise prompt
        prompt = revise_prompt(candidate, job, spans, state["draft"], problems)

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            msg = chat(messages, tracker, format=llm_schema(GroundedResume))
            draft = GroundedResume.model_validate_json(msg.content or "{}")
            return {"draft": draft.model_dump(), "round": state["round"] + 1}
        except Exception as e:
            # Retry once
            messages.append({"role": "user", "content": f"Validation error: {e}\n\nReturn the complete valid resume JSON."})
            try:
                msg = chat(messages, tracker, format=llm_schema(GroundedResume))
                draft = GroundedResume.model_validate_json(msg.content or "{}")
                return {"draft": draft.model_dump(), "round": state["round"] + 1}
            except Exception:
                # Keep previous draft and go to finalise
                return {"round": state["round"] + 1}

    @_timed(tracker, "finalise")
    def finalise_node(state):
        draft = GroundedResume.model_validate(state["draft"])
        problems = state["problems"]
        candidate = state["candidate_data"]
        first_pass = state["first_pass"]
        first_draft_stats = state["first_draft_stats"]

        # Build evidence report
        evidence = []
        all_items = iter_claims(draft)

        for item in all_items:
            claim_id = item["claim_id"]
            claim = item["claim"]
            section = item["section"]
            entry_label = item["entry_label"]
            probs = problems.get(claim_id, [])
            status = "dropped" if probs else "kept"
            rounds_to_pass = first_pass.get(claim_id, -1) if not probs else -1

            # Get source span details
            from common.spans import Span
            spans = [Span(**s) for s in state["spans"]]
            span_by_id = {s.id: s for s in spans}
            sources = []
            for source_id in claim.sources:
                span = span_by_id.get(source_id)
                if span:
                    sources.append({"id": span.id, "text": span.text})

            evidence.append({
                "claim_id": claim_id,
                "section": section,
                "entry": entry_label,
                "claim": claim.text,
                "sources": sources,
                "status": status,
                "problems": probs,
                "rounds_to_pass": rounds_to_pass,
            })

        # Drop unsupported claims
        clean_draft = drop_unsupported(draft, problems)

        # Filter skills and certifications
        kept_skills, dropped_skills = filter_skills(
            clean_draft.skills,
            candidate.get("skills", []),
            candidate.get("raw_text", "")
        )
        kept_certs, dropped_certs = filter_skills(
            clean_draft.certifications,
            candidate.get("certifications", []),
            candidate.get("raw_text", "")
        )

        clean_draft = GroundedResume(
            **{**clean_draft.model_dump(), "skills": kept_skills, "certifications": kept_certs}
        )

        # Convert to GeneratedResume
        generated = to_generated(clean_draft)

        # Validate and render
        harness = ToolHarness(candidate, resume_path=resume_path, run_id=run_id)
        _, result, _ = _tracked_execute(harness, tracker, "validate_resume", {"resume": generated})
        validation = json.loads(result)

        if not validation.get("valid"):
            return {
                "errors": ["W4_FINAL_VALIDATION_FAILED: " + "; ".join(validation.get("errors", []))],
                "hit_budget": state["round"] >= MAX_GROUNDING_ROUNDS,
                "evidence": evidence,
            }

        _tracked_execute(harness, tracker, "render_resume", {})
        outcome = harness.outcome()

        # Calculate final stats
        total_first_draft = len(all_items)
        kept = sum(1 for e in evidence if e["status"] == "kept")
        dropped = sum(1 for e in evidence if e["status"] == "dropped")

        w4_stats = {
            "kept": kept,
            "dropped": dropped,
            "first_draft_citation_coverage": first_draft_stats.get("citation_coverage", 0),
            "first_draft_supported_rate": first_draft_stats.get("supported_rate", 0),
            "rounds_used": state["round"],
            "skills_dropped": len(dropped_skills) + len(dropped_certs),
        }

        tracker.log(type="w4_final", **w4_stats)

        return {
            **outcome,
            "evidence": evidence,
            "w4_stats": w4_stats,
            "hit_budget": state["round"] >= MAX_GROUNDING_ROUNDS and dropped > 0,
        }

    def route_after_split(state):
        if state.get("errors"):
            return END
        return "draft"

    def route_after_draft(state):
        if state.get("errors"):
            return END
        return "check"

    # Build the graph
    graph = StateGraph(EvidenceState)
    graph.add_node("split_spans", split_spans_node)
    graph.add_node("draft", draft_node)
    graph.add_node("check", check_node)
    graph.add_node("llm_verify", llm_verify_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalise", finalise_node)

    graph.add_edge(START, "split_spans")
    graph.add_conditional_edges("split_spans", route_after_split, ["draft", END])
    graph.add_conditional_edges("draft", route_after_draft, ["check", END])
    graph.add_edge("check", "llm_verify")
    graph.add_conditional_edges("llm_verify", decide, ["revise", "finalise"])
    graph.add_edge("revise", "check")
    graph.add_edge("finalise", END)

    return graph.compile()


def make_creator(tracker, resume_path=None, run_id=None):
    """Create the W4 creator node for the pipeline."""
    def resume_creator(state):
        try:
            graph = build_evidence_graph(tracker, resume_path=resume_path, run_id=run_id)
            result = graph.invoke(
                {
                    "candidate_data": state["candidate_data"],
                    "job_data": state["job_data"],
                },
                {"recursion_limit": 50},
            )
            return {
                key: result[key]
                for key in ("resume_md", "output_path", "generated_resume", "hit_budget",
                            "errors", "evidence", "w4_stats")
                if key in result
            }
        except Exception as exc:
            return {"errors": [f"W4_FAILED: {type(exc).__name__}: {exc}"]}
    return resume_creator


def generate(tracker, resume_path, job_url="", pasted_jd="", run_id=None):
    """Run the W4 pipeline."""
    return run_pipeline(
        make_creator,
        tracker,
        resume_path,
        job_url=job_url,
        pasted_jd=pasted_jd,
        run_id=run_id,
    )


def run(pair, tracker):
    """Run W4 on a pair (for run_all.py)."""
    state = generate(
        tracker,
        pair["resume_path"],
        job_url=pair.get("job_url") or pair.get("jd_url", ""),
        pasted_jd=pair["jd"],
        run_id=f"{NAME}__{pair['pair_id']}",
    )
    return {
        "resume_md": state.get("resume_md", ""),
        "hit_budget": bool(state.get("hit_budget")),
        "finished": bool(state.get("resume_md")),
        "errors": state.get("errors", []),
        "output_path": state.get("output_path"),
        "evidence": state.get("evidence", []),
        "w4_stats": state.get("w4_stats", {}),
    }


# Import cache_key for the llm_verify_node
from common.grounding import cache_key  # noqa: E402
