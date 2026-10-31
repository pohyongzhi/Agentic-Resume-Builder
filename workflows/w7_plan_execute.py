r"""W7 - PLAN-AND-EXECUTE  (planner + parallel section writers + assembler)

Technique: a planner first writes a tailoring plan: which sections the resume should have and in
what order, a word budget for each, which JD requirement maps to which evidence in the resume, and
what to cut. One writer per planned section then drafts its section, all in parallel, each seeing
only its part of the plan and its part of the original resume. An assembler (one model call) merges
the sections in the planned order and makes them consistent with each other and with the plan.

    START -> plan --Send--> write (one per planned section, in parallel) -> assemble -> check -> END

This follows the W7 description and nothing more, so it can be compared with the other variants:
  - no critique/revise loop (that is the multi-critic variant's technique)
  - no checking of output claims against the original resume (that is the evidence-grounded
    variant's technique, and the scorecard's faithfulness metric already measures it)
The `check` node is plain code and changes nothing: it logs the inconsistencies it finds in the
draft before the assembler and in the final resume after it, as diagnostics for the report.

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
from eval.metrics import keyword_coverage

NAME = "w7_plan_execute"

TOTAL_WORDS = 450       # BASE_RULES: about 350-500 words. Shared out across however many sections are planned.
MAX_TOTAL_WORDS = 500

PLANNER = BASE_RULES + """
For this step you do NOT write the resume. You write a tailoring plan as JSON, in exactly this shape:
{
  "sections": [
    {"name": "<section heading>", "words": <word budget>, "source": ["<heading(s) in the original resume to draw from>"]}
  ],
  "requirements": [
    {"requirement": "<a requirement from the job description>",
     "evidence": "<a line copied WORD FOR WORD from the original resume that proves it, or null>",
     "section": "<the planned section it belongs in>"}
  ],
  "lead_with": ["<the most relevant role, project or skill from the resume>"],
  "cut": ["<resume content that is irrelevant to this job and can be dropped or shortened>"]
}
- "sections" is the tailored resume's sections in order. Decide which sections this candidate and
  this job need. The word budgets should add up to about """ + str(TOTAL_WORDS) + """.
- List 6-12 requirements. If the resume has no evidence for a requirement, set "evidence" to null.
Never make up evidence. Reply with the JSON only."""

WRITER = BASE_RULES + """
For this step you write ONLY the "{section}" section, not the whole resume.
- Start with the heading "## {section}" and write nothing after the section.
- Use at most {words} words. Shorter is fine; never pad.
- Format: {format}
- Every line must state a fact about the candidate. Never write lines about the job or why something
  is relevant (e.g. "This degree aligns with the role").
- Use only facts from the original resume. Requirements marked UNSUPPORTED must not be claimed.
- Reply with the Markdown section only, no explanations."""

# Each writer works alone, so the format of its section has to be spelled out or sections clash.
FORMATS = {
    "summary": "2-3 sentences in implied first person: no name, no 'I', no 'he', 'she' or 'they'. "
               "Summarise in your own words: who the candidate is and their strongest relevant strengths. "
               "Never reuse a sentence or bullet from the resume; the other sections already contain them.",
    "list": "A comma-separated list of short items (1-3 words each), optionally grouped with a bold label "
            "per line. No sentences.",
    "entries": 'For each entry (role, project, degree, activity) keep its bold title line with the name, '
               'organisation and dates exactly as in the original, then up to 4 bullets starting with "- ", '
               'one point per bullet. Entries with nothing to add (e.g. a degree) need no bullets.',
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


# ---------- reading the original resume (plain text from PDF/DOCX, or Markdown) ----------

def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# A line is a section heading if every word in it is one of these (so "Projects" or "Work Experience"
# is a heading but "Project Manager" is not). Works with or without '#', bold or a trailing colon.
HEADING_WORDS = {
    "summary", "profile", "objective", "about", "me", "professional", "career", "experience", "experiences",
    "work", "employment", "history", "relevant", "projects", "project", "personal", "academic", "education",
    "qualifications", "skills", "skill", "technical", "core", "key", "competencies", "tools", "technologies",
    "leadership", "activities", "extracurricular", "extra", "co", "curricular", "cca", "involvement",
    "community", "volunteer", "volunteering", "awards", "achievements", "honours", "honors", "certifications",
    "certificates", "licenses", "publications", "research", "languages", "interests", "hobbies",
    "additional", "information", "other", "and", "references", "courses", "coursework", "training",
}


def _is_heading(line):
    s = line.strip()
    if not s or s[0] in "-•*" and not s.startswith("**"):
        return False
    words = _norm(s).split()
    return 0 < len(words) <= 5 and all(w in HEADING_WORDS for w in words)


def _parse_resume(resume):
    """Split the original into (header, {heading: text}). The header is the name and contact lines
    before the first heading, copied into the output as-is."""
    header, sections, current = [], {}, None
    for line in resume.splitlines():
        if _is_heading(line):
            current = re.sub(r"^[#*\s]+|[*:\s]+$", "", line.strip())
            sections[current] = ""
        elif current is None:
            header.append(line)
        else:
            sections[current] += line + "\n"
    if not sections:   # no headings found: keep the first two lines as the header, the rest as one block
        lines = [l for l in resume.splitlines() if l.strip()]
        return "\n".join(lines[:2]), {"Resume": "\n".join(lines[2:])}
    return "\n".join(header).strip(), sections


def _kind(name):
    n = _norm(name)
    if any(k in n for k in ("summary", "profile", "objective", "about")):
        return "summary"
    if any(k in n for k in ("skill", "tool", "technolog", "language", "competenc", "interest", "hobb")):
        return "list"
    return "entries"


# ---------- the plan ----------

def _evidence_is_real(evidence, resume):
    """The planner must quote the resume. Accept a near-verbatim quote: at least 80% of its words
    appear in the resume. Anything else is treated as made up and the requirement becomes unsupported."""
    words = _norm(evidence).split()
    if not words:
        return False
    if " ".join(words) in _norm(resume):
        return True
    resume_words = set(_norm(resume).split())
    return sum(w in resume_words for w in words) / len(words) >= 0.8


def _parse_json(text):
    found = re.search(r"\{.*\}", text or "", re.S)
    if not found:
        raise ValueError("no JSON object found")
    return json.loads(found.group(0))


def _word_budgets(sections):
    """Planner budgets where given; the rest of TOTAL_WORDS shared evenly; scaled down if over the max."""
    given = {s["name"]: s["words"] for s in sections if s["words"] > 0}
    missing = [s["name"] for s in sections if s["words"] <= 0]
    if missing:
        share = max(20, (TOTAL_WORDS - sum(given.values())) // len(missing))
        given.update({name: share for name in missing})
    total = sum(given.values())
    scale = min(1.0, MAX_TOTAL_WORDS / total) if total else 1.0
    for s in sections:
        s["words"] = max(15, round(given[s["name"]] * scale))
    return sections


def _clean_plan(raw, resume, original_headings):
    """Keep the planner's choices; only fix the shape, check evidence quotes and share out the words."""
    sections, seen = [], set()
    for s in raw.get("sections") or []:
        name = s.get("name") if isinstance(s, dict) else s
        if not isinstance(name, str) or not name.strip() or _norm(name) in seen:
            continue
        seen.add(_norm(name))
        words = s.get("words") if isinstance(s, dict) else None
        source = s.get("source") if isinstance(s, dict) else None
        sections.append({"name": name.strip(),
                         "words": words if isinstance(words, int) else 0,
                         "source": [str(x) for x in source] if isinstance(source, list) else []})
    if not sections:   # no usable plan: one section per heading in the original
        sections = [{"name": h, "words": 0, "source": [h]} for h in original_headings]
    sections = _word_budgets(sections)

    reqs = []
    for r in raw.get("requirements") or []:
        if not isinstance(r, dict) or not r.get("requirement"):
            continue
        ev = r.get("evidence")
        ev = ev if isinstance(ev, str) and _evidence_is_real(ev, resume) else None
        reqs.append({"requirement": str(r["requirement"]), "evidence": ev, "section": str(r.get("section") or "")})

    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else []
    # A small planner often cuts a line it also uses as evidence. Evidence wins: drop those cuts.
    evidence = [r["evidence"] for r in reqs if r["evidence"]]
    cut = [c for c in as_list(raw.get("cut")) if not any(_overlaps(c, ev) for ev in evidence)]
    return {"sections": sections, "requirements": reqs, "lead_with": as_list(raw.get("lead_with")), "cut": cut}


def _brief_for(section, plan, source):
    """What one writer sees. Entry sections get their mapped requirements with evidence quotes.
    Summary and list sections get requirement names only: given quotes, they paste them in as sentences.
    "Lead with" and "cut" items are only passed to the writer whose source contains them: given a
    project to lead with, the Experience writer files it under a job."""
    reqs = plan["requirements"]
    if _kind(section["name"]) == "summary":
        lead, cut = [], []
    else:
        lead = [x for x in plan["lead_with"] if _evidence_is_real(x, source)]
        cut = [x for x in plan["cut"] if _evidence_is_real(x, source)]
    if _kind(section["name"]) in ("summary", "list"):
        supported = [r["requirement"] for r in reqs if r["evidence"]]
        unsupported = [r["requirement"] for r in reqs if not r["evidence"]]
        body = ("Requirements the candidate meets: " + (", ".join(supported) or "-") +
                "\nUNSUPPORTED (do not claim): " + (", ".join(unsupported) or "-"))
    else:
        mine = [r for r in reqs if _norm(r["section"]) == _norm(section["name"])]
        unsupported = [r["requirement"] for r in reqs if not r["evidence"]]
        body = ("Requirements for this section:\n" + ("\n".join(
            f'- {r["requirement"]}: ' + (f'evidence "{r["evidence"]}"' if r["evidence"] else "UNSUPPORTED")
            for r in mine) or "- (none mapped)") +
            "\nUNSUPPORTED anywhere (do not claim): " + (", ".join(unsupported) or "-"))
    return (body +
            "\nLead with: " + (", ".join(lead) or "-") +
            "\nCut or shorten: " + (", ".join(cut) or "-"))


def _source_text(section, resume, original):
    """The part of the original resume this writer works from. Other sections see the original sections
    the planner named, or everything if none of the names match. Summaries see the original summary
    plus every title line (roles, projects, degrees, skills) but no bullets: shown bullets, a 4B
    model copies them into the summary word for word, whatever the prompt says."""
    if _kind(section["name"]) == "summary":
        if not resume:
            return ""
        parts = []
        for h, t in original.items():
            lines = t.strip().splitlines() if _kind(h) in ("summary", "list") else \
                [l for l in t.splitlines() if l.strip() and not l.strip().startswith(("-", "•", "* "))]
            parts.append(f"{h}\n" + "\n".join(lines))
        return "\n\n".join(parts)
    wanted = {_norm(s) for s in section["source"] + [section["name"]]}
    parts = [f"{h}\n{t.strip()}" for h, t in original.items() if _norm(h) in wanted]
    return "\n\n".join(parts) if parts else resume


def _assign_evidence(plan, resume, original):
    """Where a piece of evidence comes from is a fact, so code decides it, not the planner: each
    requirement goes to the planned entry section whose original text contains its evidence.
    Otherwise the Experience writer gets a project's evidence and files it under a job."""
    entry_sections = [s for s in plan["sections"] if _kind(s["name"]) == "entries"]
    for r in plan["requirements"]:
        if not r["evidence"]:
            continue
        for s in entry_sections:
            if _evidence_is_real(r["evidence"], _source_text(s, "", original)):
                r["section"] = s["name"]
                break
    return plan


# ---------- output clean-up and checks (plain code, no model) ----------

def _only_section(text, name):
    """Strip code fences and anything the writer added beyond its own section."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", (text or "").strip())
    body, started = [], False
    for line in text.splitlines():
        if line.startswith("#"):
            is_mine = _norm(line.lstrip("#")) == _norm(name)
            if started and not is_mine:
                break
            if is_mine:
                started = True
                continue
        body.append(line)
    return f"## {name}\n" + "\n".join(body).strip()


def _restore_headings(md, names):
    """The assembler sometimes drops the '## ' (e.g. writes **Summary**). Put the planned headings back."""
    by_norm = {_norm(n): n for n in names}
    out = []
    for line in md.splitlines():
        key = _norm(re.sub(r"^[#*\s]+|[*:\s]+$", "", line.strip()))
        out.append(f"## {by_norm[key]}" if key in by_norm and len(line) < 80 else line)
    return "\n".join(out)


def _split_sections(md):
    sections, current = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return sections


STOPWORDS = {"and", "the", "for", "with", "from", "into", "using", "use", "used", "strong", "skills",
             "skill", "ability", "experience", "proficient", "proficiency", "familiar", "familiarity",
             "knowledge", "including", "via", "per", "about"}


def _content_words(text):
    return [w for w in _norm(text).split() if len(w) > 2 and w not in STOPWORDS]


def _overlaps(a, b, share=0.8):
    """True if at least `share` of a's content words appear in b, or the other way round."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return False
    return (sum(w in set(wb) for w in wa) / len(wa) >= share or
            sum(w in set(wa) for w in wb) / len(wb) >= share)


def _find_problems(md, plan):
    """Diagnostics only: are the sections consistent with each other and with the plan? Nothing is
    compared with the original resume (that is the evidence-grounded variant's job and the
    faithfulness metric's). Logged before and after the assembler; never changes the output."""
    problems, sections = [], _split_sections(md)
    found = {_norm(h): body for h, body in sections.items()}

    for s in plan["sections"]:
        if len(found.get(_norm(s["name"]), "").split()) < 3:
            problems.append(f'Planned section "{s["name"]}" is missing or empty.')
    words = len(md.split())
    if words > 550:
        problems.append(f"{words} words; BASE_RULES asks for about 350-500.")

    # Writers can't see each other, so the same sentence often lands in several sections.
    # Compare on the first 8 words so near-copies ("...team to track metrics" vs "...team") also count.
    openings = [" ".join(_norm(s).split()[:8]) for s in re.split(r"(?<=[.;,])\s+|\n", md)
                if len(s.split()) >= 8]
    dupes = sorted({o for o in openings if openings.count(o) > 1})
    for d in dupes:
        problems.append(f'Repeated in more than one place: "{d} ..."')

    for r in plan["requirements"]:
        if r["evidence"] is None and keyword_coverage(md, [{"keyword": r["requirement"]}])["hit"]:
            problems.append(f'Plan marks "{r["requirement"]}" UNSUPPORTED, but the resume claims it.')
    for c in plan["cut"]:
        # One direction only: the cut item's words must be in the line, not just the line's words in the item.
        cw = _content_words(c)
        if len(cw) >= 3 and any(sum(w in set(_content_words(line)) for w in cw) / len(cw) >= 0.8
                                for line in md.splitlines()):
            problems.append(f'Plan says to cut "{c}", but it is still there.')
    return problems


# ---------- the workflow ----------

def run(pair, tracker):
    resume, jd = pair["resume"], pair["jd"]
    header, original = _parse_resume(resume)

    def plan_node(state):
        messages = [{"role": "system", "content": PLANNER},
                    {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE'S ORIGINAL RESUME:\n{resume}\n\n"
                                                f"Headings found in the original resume: {', '.join(original)}"}]
        for attempt in range(2):   # one retry if the JSON is broken
            msg = chat(messages, tracker)
            try:
                plan = _assign_evidence(_clean_plan(_parse_json(msg.content), resume, list(original)),
                                        resume, original)
                tracker.log(type="w7_plan", ok=True, attempt=attempt + 1,
                            sections=[f'{s["name"]} ({s["words"]}w)' for s in plan["sections"]],
                            requirements=plan["requirements"],
                            supported=sum(r["evidence"] is not None for r in plan["requirements"]),
                            cut=plan["cut"])
                return {"plan": plan, "plan_ok": True}
            except (ValueError, json.JSONDecodeError) as e:
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": f"That was not valid JSON ({e}). Reply with the JSON only."}]
        # Writers can still work from the original's own sections; the run is marked as not finished cleanly.
        plan = _clean_plan({}, resume, list(original))
        tracker.log(type="w7_plan", ok=False, attempt=2, sections=[s["name"] for s in plan["sections"]])
        return {"plan": plan, "plan_ok": False}

    def fan_out(state):
        """One writer per planned section, all started in the same step."""
        return [Send("write", {"section": s, "plan": state["plan"]}) for s in state["plan"]["sections"]]

    def write_node(task):
        section, plan = task["section"], task["plan"]
        name = section["name"]
        source = _source_text(section, resume, original)
        msg = chat([
            {"role": "system", "content": WRITER.format(section=name, words=section["words"],
                                                         format=FORMATS[_kind(name)])},
            {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\n"
                                        f"CANDIDATE'S ORIGINAL RESUME (the part for this section):\n{source}\n\n"
                                        f"{_brief_for(section, plan, source)}\n\nWrite the {name} section."},
        ], tracker)
        return {"sections": {name: _only_section(msg.content, name)}}

    def assemble_node(state):
        """One model call: merge the separately written sections and make them consistent."""
        names = [s["name"] for s in state["plan"]["sections"] if s["name"] in state["sections"]]
        draft = "\n\n".join(state["sections"][n] for n in names)
        tracker.log(type="w7_check", stage="before_assembler", problems=_find_problems(draft, state["plan"]))
        plan = state["plan"]
        plan_text = "\n".join(
            [f'- {r["requirement"]}: ' + (f'evidence "{r["evidence"]}"' if r["evidence"] else "UNSUPPORTED")
             for r in plan["requirements"]] +
            [f"Cut: {', '.join(plan['cut']) or '-'}"])
        msg = chat([
            {"role": "system", "content": ASSEMBLER},
            {"role": "user", "content": f"PLAN:\n{plan_text}\n\nSECTIONS (in order):\n\n{draft}"},
        ], tracker)
        merged = re.sub(r"^```\w*\s*|\s*```$", "", (msg.content or "").strip())
        ok = len(merged) >= 200
        if not ok:   # the assembler returned nothing usable: fall back to the sections joined as written
            merged = draft
        merged = _restore_headings(merged, names)
        return {"resume_md": "\n\n".join(p for p in [header, merged] if p), "assembled_ok": ok}

    def check_node(state):
        tracker.log(type="w7_check", stage="final", problems=_find_problems(state["resume_md"], state["plan"]))
        return {}

    g = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    g.add_node("write", write_node)
    g.add_node("assemble", assemble_node)
    g.add_node("check", check_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["write"])   # fan out: one writer per planned section
    g.add_edge("write", "assemble")                        # fan in: waits for every writer
    g.add_edge("assemble", "check")
    g.add_edge("check", END)

    out = g.compile().invoke({"plan": {}, "plan_ok": False, "sections": {}, "resume_md": "", "assembled_ok": False})
    return {"resume_md": out["resume_md"], "hit_budget": False, "finished": out["plan_ok"] and out["assembled_ok"]}
