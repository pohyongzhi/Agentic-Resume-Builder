r"""W7 - PLAN-AND-EXECUTE  (planner + parallel section writers + assembler)

Technique: a planner first writes a tailoring plan: the resume's sections and their order, which
JD requirement maps to which evidence in the resume, and for each section what to lead with and
what to cut. One writer per planned section then drafts its section, all in parallel, from the
original resume and its own part of the plan. An assembler (one model call) merges the sections in
the planned order and makes them consistent with each other and with the plan.

    START -> plan --Send--> write (one per planned section, in parallel) -> assemble -> END

This follows the W7 description and nothing more, so it can be compared with the other variants.
Every decision is made by the model. The plan reaches the writers exactly as the planner wrote it:
  - no check that its evidence quotes are really in the resume, and no checks of the output
    against the original (that is the evidence-grounded variant's technique; the scorecard's
    faithfulness metric measures it)
  - no critique/revise loop (that is the multi-critic variant's technique)
Code only does plumbing: parse the plan's JSON (one retry), hand each writer its part, tidy
headings, and put the header in front.

The planner chooses the sections freely (no checks on its choice). BASE_RULES still asks for
Summary, Experience, Education and Skills, so a plan that adds e.g. Projects goes beyond the shared
rule; raise this with the group before the full run.

What it costs: 1 plan call + 1 call per planned section + 1 assembler call. Varies per resume.
Course link: Topic 7 §4 plan-and-execute.

Note: all writers share one Ollama server, so "parallel" may just queue. Report wall_seconds
(from run_all.py) as latency; the Tracker's `seconds` adds up call times and overstates overlap.
"""
import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from common.llm import chat
from common.prompts import BASE_RULES

NAME = "w7_plan_execute"

PLANNER = BASE_RULES + """
For this step you do NOT write the resume. You write a tailoring plan as JSON, in exactly this shape:
{
  "header": "<the candidate's name and contact line(s), copied exactly from the top of the resume>",
  "sections": [
    {"name": "<section heading>",
     "lead_with": ["<the most relevant role, project or point from the resume for this section>"],
     "cut": ["<content of this section that is irrelevant to this job and can be dropped>"]}
  ],
  "requirements": [
    {"requirement": "<a requirement from the job description>",
     "evidence": "<a line copied WORD FOR WORD from the original resume that proves it, or null>",
     "section": "<the planned section the evidence belongs in>"}
  ]
}
- "sections" is the tailored resume's sections in order. Decide which sections this candidate and
  this job need.
- List 6-12 requirements. If the resume has no evidence for a requirement, set "evidence" to null.
Never make up evidence. Reply with the JSON only."""

WRITER = BASE_RULES + """
For this step you write ONLY the "{section}" section, not the whole resume. Other writers are
writing the other sections at the same time.
- Start with the heading "## {section}" and write nothing after the section.
- Format: {format}
- Every line must state a fact about the candidate. Never write lines about the job or why something
  is relevant (e.g. "This degree aligns with the role").
- Use only facts from the original resume. Requirements marked UNSUPPORTED must not be claimed.
- Reply with the Markdown section only, no explanations."""

# Each writer works alone, so the format of its section has to be spelled out or sections clash.
FORMATS = {
    "summary": "2-3 sentences in implied first person: no name, no 'I', no 'he', 'she' or 'they'. "
               "Summarise in your own words: who the candidate is and their strongest relevant strengths.",
    "list": "A comma-separated list of short items (1-3 words each), optionally grouped with a bold label "
            "per line. No sentences.",
    "entries": 'Only the entries (roles, projects, degrees, activities) that belong under this heading. For each, '
               'keep its bold title line with the name, organisation and dates exactly as in the original, then '
               'up to 4 bullets starting with "- ", one point per bullet. Entries with nothing to add need no bullets.',
}

ASSEMBLER = BASE_RULES + """
You are the assembler. Different writers drafted the sections below separately, from a shared plan,
without seeing each other's work. Merge them into one resume and make sure they are consistent:
- Keep the sections in the order given, each under its "## " heading.
- Remove any sentence or fact that appears in more than one section; keep it where it fits best.
- The sections must not contradict each other (titles, organisations, dates, what each role or
  project involved). If they do, keep the version in the section that the fact belongs to.
- Follow the plan: leave out the items it says to cut and never claim requirements it marks UNSUPPORTED.
- Do not add new facts. Keep each section's content otherwise as written.
Reply with the merged resume sections in Markdown only, starting with the first "## " heading."""


class PlanState(TypedDict):
    plan: dict
    plan_ok: bool
    sections: Annotated[dict, operator.or_]   # each writer adds {section: markdown}; merged in parallel
    resume_md: str
    assembled_ok: bool


# ---------- helpers (plain code, no model) ----------

def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _same_section(a, b):
    """Planner labels vary ("Experience" vs "Professional Experience"); treat containment as a match."""
    a, b = _norm(a), _norm(b)
    return bool(a and b) and (a == b or a in b or b in a)


def _kind(name):
    """Which writing format a section gets, from the heading the planner chose."""
    n = _norm(name)
    if any(k in n for k in ("summary", "profile", "objective", "about")):
        return "summary"
    if any(k in n for k in ("skill", "tool", "technolog", "language", "competenc", "interest", "hobb")):
        return "list"
    return "entries"


def _parse_json(text):
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        raise ValueError("no JSON object found")
    raw = json.loads(found.group(0))
    if not isinstance(raw, dict):
        raise ValueError("the plan must be a JSON object")
    return raw


def _as_list(v):
    return [str(x) for x in v if isinstance(x, (str, int, float))] if isinstance(v, list) else []


def _clean_plan(raw):
    """Keep the planner's choices exactly; only fix the shape so a malformed plan can't crash the run."""
    listed = lambda key: raw.get(key) if isinstance(raw.get(key), list) else []   # a string would loop over letters
    reqs = []
    for r in listed("requirements"):
        if not isinstance(r, dict) or not r.get("requirement"):
            continue
        ev = r.get("evidence")
        ev = ev.strip() if isinstance(ev, str) and ev.strip() else None
        reqs.append({"requirement": str(r["requirement"]), "evidence": ev, "section": str(r.get("section") or "")})

    sections, seen = [], set()
    for s in listed("sections"):
        name = s.get("name") if isinstance(s, dict) else s
        if not isinstance(name, str) or not _norm(name) or _norm(name) in seen:
            continue
        seen.add(_norm(name))
        s = s if isinstance(s, dict) else {}
        sections.append({"name": name.strip(), "lead_with": _as_list(s.get("lead_with")),
                         "cut": _as_list(s.get("cut"))})
    if not sections:   # no usable sections: fall back to the four BASE_RULES asks for
        sections = [{"name": n, "lead_with": [], "cut": []} for n in ("Summary", "Experience", "Education", "Skills")]

    header = raw.get("header") if isinstance(raw.get("header"), str) else ""
    return {"header": header.strip(), "sections": sections, "requirements": reqs}


def _brief_for(section, plan):
    """What one writer sees of the plan: its requirements (with evidence quotes for entry sections;
    names only for summary/list sections, which otherwise paste the quotes in as sentences),
    every UNSUPPORTED requirement, and its own lead and cut items."""
    reqs = plan["requirements"]
    unsupported = [r["requirement"] for r in reqs if not r["evidence"]]
    if _kind(section["name"]) in ("summary", "list"):
        body = "Requirements the candidate meets: " + (", ".join(r["requirement"] for r in reqs if r["evidence"]) or "-")
    else:
        mine = [r for r in reqs if r["evidence"] and _same_section(r["section"], section["name"])]
        body = "Requirements for this section:\n" + (
            "\n".join(f'- {r["requirement"]}: evidence "{r["evidence"]}"' for r in mine) or "- (none mapped)")
    return (body +
            "\nUNSUPPORTED (do not claim): " + (", ".join(unsupported) or "-") +
            "\nLead with: " + (", ".join(section["lead_with"]) or "-") +
            "\nCut: " + (", ".join(section["cut"]) or "-"))


def _only_section(text, name, all_names):
    """Strip code fences and anything the writer added beyond its own section. Stops only at another
    planned section's heading, so sub-headings inside the section (e.g. '### DBS Bank') are kept."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", (text or "").strip())
    others = {_norm(n) for n in all_names} - {_norm(name)}
    lines = text.splitlines()
    own = [i for i, l in enumerate(lines) if l.startswith("#") and _norm(l.lstrip("#")) == _norm(name)]
    if own:                                   # drop any preamble ("Here is the section:") before its heading
        lines = lines[own[0] + 1:]
    body = []
    for line in lines:
        if line.startswith("#"):
            heading = _norm(line.lstrip("#"))
            if heading == _norm(name) or re.match(r"#\s", line):
                continue                      # its own heading (added back below) or a stray '# Name' title
            if heading in others:
                break                         # the writer started another section
        body.append(line)
    return f"## {name}\n" + _md_bullets("\n".join(body).strip())


def _md_bullets(text):
    """PDF resumes use '•' bullets and writers copy them. Markdown doesn't treat '•' as a list, so the
    page runs the title and its bullets into one paragraph. Turn them into '- ' bullets."""
    return re.sub(r"^(\s*)[•●▪◦‣∙·]\s*", r"\1- ", text, flags=re.M)


def _restore_headings(md, names):
    """The assembler sometimes drops the '## ' (e.g. writes **Summary**). Put the planned headings back."""
    by_norm = {_norm(n): n for n in names}
    out = []
    for line in md.splitlines():
        key = _norm(re.sub(r"^[#*\s]+|[*:\s]+$", "", line.strip()))
        out.append(f"## {by_norm[key]}" if key in by_norm and len(line) < 80 else line)
    return "\n".join(out)


# ---------- the workflow ----------

def run(pair, tracker):
    resume, jd = pair["resume"], pair["jd"]

    def plan_node(state):
        messages = [{"role": "system", "content": PLANNER},
                    {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE'S ORIGINAL RESUME:\n{resume}"}]
        for attempt in range(2):   # one retry if the JSON is broken
            msg = chat(messages, tracker)
            try:
                plan = _clean_plan(_parse_json(msg.content))
                tracker.log(type="w7_plan", ok=True, attempt=attempt + 1, header=plan["header"],
                            sections=plan["sections"], requirements=plan["requirements"],
                            supported=sum(r["evidence"] is not None for r in plan["requirements"]))
                return {"plan": plan, "plan_ok": True}
            except (ValueError, json.JSONDecodeError) as e:
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": f"That was not valid JSON ({e}). Reply with the JSON only."}]
        plan = _clean_plan({})   # writers still run on the four BASE_RULES sections; run not finished cleanly
        tracker.log(type="w7_plan", ok=False, attempt=2, sections=plan["sections"])
        return {"plan": plan, "plan_ok": False}

    def fan_out(state):
        """One writer per planned section, all started in the same step."""
        return [Send("write", {"section": s, "plan": state["plan"]}) for s in state["plan"]["sections"]]

    def write_node(task):
        section, plan = task["section"], task["plan"]
        name = section["name"]
        msg = chat([
            {"role": "system", "content": WRITER.format(section=name, format=FORMATS[_kind(name)])},
            {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE'S ORIGINAL RESUME:\n{resume}\n\n"
                                        f"YOUR PART OF THE PLAN:\n{_brief_for(section, plan)}\n\n"
                                        f"Write the {name} section."},
        ], tracker)
        all_names = [s["name"] for s in plan["sections"]]
        return {"sections": {name: _only_section(msg.content, name, all_names)}}

    def assemble_node(state):
        """One model call: merge the separately written sections and make them consistent."""
        plan = state["plan"]
        names = [s["name"] for s in plan["sections"] if s["name"] in state["sections"]]
        draft = "\n\n".join(state["sections"][n] for n in names)
        plan_text = "\n".join(
            [f'- {r["requirement"]}: ' + (f'evidence "{r["evidence"]}"' if r["evidence"] else "UNSUPPORTED")
             for r in plan["requirements"]] +
            [f'Cut from {s["name"]}: {", ".join(s["cut"])}' for s in plan["sections"] if s["cut"]])
        msg = chat([
            {"role": "system", "content": ASSEMBLER},
            {"role": "user", "content": f"PLAN:\n{plan_text}\n\nSECTIONS (in order):\n\n{draft}"},
        ], tracker)
        merged = _restore_headings(_md_bullets(re.sub(r"^```\w*\s*|\s*```$", "", (msg.content or "").strip())), names)
        # Anything before the first section (a repeated name, "Here is the resume:") is dropped:
        # the header comes from the plan, so keeping it would print the name twice.
        merged = merged[merged.find("## "):] if "## " in merged else merged
        ok = len(merged) >= 200
        if not ok:   # the assembler returned nothing usable: fall back to the sections joined as written
            merged = draft
        header = plan["header"] or "\n".join([l for l in resume.splitlines() if l.strip()][:2])
        return {"resume_md": "\n\n".join(p for p in [header, merged] if p), "assembled_ok": ok}

    g = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    g.add_node("write", write_node)
    g.add_node("assemble", assemble_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["write"])   # fan out: one writer per planned section
    g.add_edge("write", "assemble")                        # fan in: waits for every writer
    g.add_edge("assemble", END)

    out = g.compile().invoke({"plan": {}, "plan_ok": False, "sections": {}, "resume_md": "", "assembled_ok": False})
    return {"resume_md": out["resume_md"], "hit_budget": False, "finished": out["plan_ok"] and out["assembled_ok"]}
