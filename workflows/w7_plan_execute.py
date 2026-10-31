r"""W7 - PLAN-AND-EXECUTE  (planner + parallel section writers + assembler)

Technique: a planner first maps each JD requirement to evidence quoted from the resume (or marks it
unsupported), picks the section order, and decides what to cut. Four writers then draft one section
each, in parallel, seeing only the plan items for their section. An assembler joins the sections
and a checker (plain code) looks for inconsistencies:
  - between sections: repeated sentences, Skills not backed by the original resume
  - with the original: role titles, employers, degrees and numbers must match exactly
  - with the plan: unsupported requirements claimed, items the plan said to cut still present
If it finds any, one fix pass rewrites the whole resume. (The fix pass is an addition to the
W7 description; it costs at most one extra model call.)

    START -> plan -> write_summary    -\
                  -> write_experience -+-> assemble -> check --(ok)-------> END
                  -> write_education  -|                     \--(problems)-> fix -> END
                  -> write_skills     -/

What it should improve: faithfulness (writers only get evidence the code has verified is really
in the resume) and relevance (the plan decides what to lead with and what to cut).
What it costs: ~6-7 model calls instead of 1.
Course link: Topic 7 §4 plan-and-execute.

Note: all four writers share one Ollama server, so "parallel" may just queue. Report wall_seconds
(from run_all.py) as latency; the Tracker's `seconds` adds up call times and overstates overlap.
"""
import json
import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from common.llm import chat
from common.prompts import BASE_RULES
from eval.metrics import keyword_coverage

NAME = "w7_plan_execute"

SECTIONS = ["Summary", "Experience", "Education", "Skills"]
WORD_LIMITS = {"Summary": "40-70", "Experience": "180-280", "Education": "15-50", "Skills": "20-60"}
# Each writer works alone, so the format of its section has to be spelled out or sections clash.
FORMATS = {
    "Summary": "2-3 sentences describing the candidate. Do not copy experience bullets word for word.",
    "Experience": 'For each role keep the bold title line with employer and dates exactly as in the original, '
                  'then 2-4 bullets starting with "- ". One achievement per bullet.',
    "Education": "Copy degree, institution and dates exactly as in the original resume.",
    "Skills": "A comma-separated list of short skill names (1-3 words each), optionally grouped with a bold "
              "label per line. No sentences, no degrees.",
}

PLANNER = BASE_RULES + """
For this step you do NOT write the resume. You write a tailoring plan as JSON, in exactly this shape:
{
  "requirements": [
    {"requirement": "<a requirement from the job description>",
     "evidence": "<a line copied WORD FOR WORD from the original resume that proves it, or null>",
     "section": "<Summary|Experience|Education|Skills>"}
  ],
  "section_order": ["Summary", "Experience", "Education", "Skills"],
  "lead_with": ["<the most relevant role, project or skill from the resume>"],
  "cut": ["<resume content that is irrelevant to this job and can be dropped or shortened>"]
}
List 6-12 requirements. If the resume has no evidence for a requirement, set "evidence" to null.
Never make up evidence. Reply with the JSON only."""

WRITER = BASE_RULES + """
For this step you write ONLY the "{section}" section, not the whole resume.
- Start with the heading "## {section}" and write nothing after the section.
- Use about {words} words.
- Format: {format}
- Use only facts from the original resume. Requirements marked UNSUPPORTED must not be claimed.
- Reply with the Markdown section only, no explanations."""

FIXER = BASE_RULES + """
You are given a tailored resume and a list of problems found in it. Fix every problem and change as
little else as possible. Reply with the complete corrected resume in Markdown only."""


class PlanState(TypedDict):
    plan: dict
    plan_ok: bool
    sections: Annotated[dict, operator.or_]   # each writer adds {section: markdown}; merged in parallel
    resume_md: str
    problems: list


# ---------- helpers (plain code, no model) ----------

def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


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


def _clean_plan(raw, resume):
    """Keep only well-formed requirements, drop fake evidence, and fill in defaults."""
    reqs = []
    for r in raw.get("requirements") or []:
        if not isinstance(r, dict) or not r.get("requirement"):
            continue
        ev = r.get("evidence")
        ev = ev if isinstance(ev, str) and _evidence_is_real(ev, resume) else None
        sec = r.get("section") if r.get("section") in SECTIONS else "Experience"
        reqs.append({"requirement": str(r["requirement"]), "evidence": ev, "section": sec})
    order = [s for s in (raw.get("section_order") or []) if s in SECTIONS]
    order += [s for s in SECTIONS if s not in order]
    as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else []
    # A small planner often cuts a line it also uses as evidence. Evidence wins: drop those cuts.
    evidence = [r["evidence"] for r in reqs if r["evidence"]]
    cut = [c for c in as_list(raw.get("cut")) if not any(_overlaps(c, ev) for ev in evidence)]
    return {"requirements": reqs, "section_order": order,
            "lead_with": as_list(raw.get("lead_with")), "cut": cut}


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


def _backed_by(text, resume):
    """Every content word of `text` appears somewhere in the original resume. Words of 5+ letters
    match on their first 5, a crude stem so "cleaning" matches "cleaned"."""
    words = set(_norm(resume).split())
    stems = {w[:5] for w in words if len(w) >= 5}
    return all(w in words or (len(w) >= 5 and w[:5] in stems) for w in _content_words(text))


def _split_sections(md):
    """{'Summary': body, ...} from a Markdown resume, keyed by '## ' headings."""
    sections, current = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            current = next((s for s in SECTIONS if _norm(line[3:]) == _norm(s)), line[3:].strip())
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return sections


def _find_problems(md, plan, resume):
    """Code-only checks that the assembled sections are consistent with each other, with the plan,
    and with the original resume. Returns a list of problems in words the fix step can act on."""
    problems, sections = [], _split_sections(md)

    # Structure
    for s in SECTIONS:
        if len(sections.get(s, "").split()) < 4:
            problems.append(f"The {s} section is missing or empty; write it from the original resume.")
    if not re.search(r"^\s*[-*] ", sections.get("Experience", ""), re.M):
        problems.append("The Experience section must use bullet points under each role.")
    words = len(md.split())
    if words > 550:
        problems.append(f"The resume is {words} words; cut it to about 350-500.")

    # Writers can't see each other, so the same sentence often lands in several sections.
    # Compare on the first 8 words so near-copies ("...team to track metrics" vs "...team") also count.
    openings = [" ".join(_norm(s).split()[:8]) for s in re.split(r"(?<=[.;,])\s+|\n", md)
                if len(s.split()) >= 8]
    dupes = sorted({o for o in openings if openings.count(o) > 1})
    if dupes:
        problems.append("These sentences appear in more than one place; keep each only once: "
                        + "; ".join(f'"{d} ..."' for d in dupes))

    # Protected facts: role titles, employers, degrees and every number must match the original.
    for line in (sections.get("Experience", "") + sections.get("Education", "")).splitlines():
        if line.strip().startswith(("-", "* ")):
            continue
        for bold in re.findall(r"\*\*(.+?)\*\*", line) or ([line] if line.strip() else []):
            if not _backed_by(bold, resume):
                problems.append(f'"{bold.strip()}" does not match a title, employer or degree in the '
                                "original resume; copy it exactly from the original.")
    for num in sorted(set(re.findall(r"\d+", md)) - set(re.findall(r"\d+", resume))):
        problems.append(f'The number "{num}" is not in the original resume; remove it or use the original figure.')

    # Skills must be backed by the original resume, not just by what another writer wrote.
    for line in sections.get("Skills", "").splitlines():
        line = re.sub(r"\*\*[^*]+\*\*:?", "", line)          # bold group label, e.g. **Programming:**
        line = re.sub(r"^\s*[-*]\s+", "", line)
        for skill in re.split(r"[,;](?![^()]*\))", line):    # don't split inside "SQL (basic, intermediate)"
            if skill.strip() and not _backed_by(skill, resume):
                problems.append(f'Skill "{skill.strip()}" is not in the original resume; remove it.')

    # The plan: unsupported requirements must not be claimed, cut items must be gone.
    for r in plan["requirements"]:
        kw = [{"keyword": r["requirement"]}]
        if r["evidence"] is None and keyword_coverage(md, kw)["hit"] and not keyword_coverage(resume, kw)["hit"]:
            problems.append(f'"{r["requirement"]}" is not in the original resume; remove that claim.')
    for c in plan["cut"]:
        if len(_content_words(c)) >= 3 and any(_overlaps(c, line) for line in md.splitlines()):
            problems.append(f'The plan said to cut "{c}", but it is still in the resume; remove or shorten it.')
    return problems


def _header(resume):
    """Name and contact lines = everything before the first '## ' heading, copied as-is."""
    lines = []
    for line in resume.splitlines():
        if line.startswith("## "):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _only_section(text, section):
    """Strip code fences and anything the writer added beyond its own section."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", (text or "").strip())
    body, started = [], False
    for line in text.splitlines():
        if line.startswith("#"):
            is_mine = _norm(line.lstrip("#")) == _norm(section)
            if started and not is_mine:
                break
            if is_mine:
                started = True
                continue
        body.append(line)
    return f"## {section}\n" + "\n".join(body).strip()


def _brief_for(section, plan):
    """What one writer sees. Experience/Education get their mapped requirements with evidence quotes.
    Summary and Skills get requirement names only: given quotes, they paste them in as sentences."""
    if section in ("Summary", "Skills"):
        reqs = plan["requirements"]
        supported = [r["requirement"] for r in reqs if r["evidence"]]
        unsupported = [r["requirement"] for r in reqs if not r["evidence"]]
        body = ("Requirements the candidate meets: " + (", ".join(supported) or "-") +
                "\nUNSUPPORTED (do not claim): " + (", ".join(unsupported) or "-"))
    else:
        items = [r for r in plan["requirements"] if r["section"] == section]
        body = "Requirements for this section:\n" + ("\n".join(
            f'- {r["requirement"]}: ' + (f'evidence "{r["evidence"]}"' if r["evidence"] else "UNSUPPORTED")
            for r in items) or "- (none mapped)")
    return (body +
            "\nLead with: " + (", ".join(plan["lead_with"]) or "-") +
            "\nCut or shorten: " + (", ".join(plan["cut"]) or "-"))


# ---------- the workflow ----------

def run(pair, tracker):
    resume, jd = pair["resume"], pair["jd"]

    def plan_node(state):
        messages = [{"role": "system", "content": PLANNER},
                    {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE'S ORIGINAL RESUME:\n{resume}"}]
        for attempt in range(2):   # one retry if the JSON is broken
            msg = chat(messages, tracker)
            try:
                plan = _clean_plan(_parse_json(msg.content), resume)
                tracker.log(type="w7_plan", ok=True, attempt=attempt + 1,
                            requirements=len(plan["requirements"]),
                            supported=sum(r["evidence"] is not None for r in plan["requirements"]))
                return {"plan": plan, "plan_ok": True}
            except (ValueError, json.JSONDecodeError) as e:
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": f"That was not valid JSON ({e}). Reply with the JSON only."}]
        tracker.log(type="w7_plan", ok=False, attempt=2)
        # Writers can still work from the resume alone; the run is marked as not finished cleanly.
        return {"plan": _clean_plan({}, resume), "plan_ok": False}

    def make_writer(section):
        def write_node(state):
            msg = chat([
                {"role": "system", "content": WRITER.format(section=section, words=WORD_LIMITS[section],
                                                         format=FORMATS[section])},
                {"role": "user", "content": f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE'S ORIGINAL RESUME:\n{resume}\n\n"
                                            f"{_brief_for(section, state['plan'])}\n\nWrite the {section} section."},
            ], tracker)
            return {"sections": {section: _only_section(msg.content, section)}}
        return write_node

    def assemble_node(state):
        parts = [_header(resume)] + [state["sections"][s] for s in state["plan"]["section_order"]]
        return {"resume_md": "\n\n".join(p for p in parts if p)}

    def check_node(state):
        problems = _find_problems(state["resume_md"], state["plan"], resume)
        tracker.log(type="w7_check", problems=problems)
        return {"problems": problems}

    def fix_node(state):
        """Not in the original W7 description: one repair pass when the check finds problems.
        Costs at most one extra model call; the re-check is only logged, never looped."""
        msg = chat([
            {"role": "system", "content": FIXER},
            {"role": "user", "content": f"CANDIDATE'S ORIGINAL RESUME:\n{resume}\n\nTAILORED RESUME:\n"
                                        f"{state['resume_md']}\n\nPROBLEMS:\n- " + "\n- ".join(state["problems"])},
        ], tracker)
        fixed = re.sub(r"^```\w*\s*|\s*```$", "", (msg.content or "").strip())
        if len(fixed) < 200:   # the fix failed; keep the assembled draft
            return {}
        tracker.log(type="w7_recheck", problems=_find_problems(fixed, state["plan"], resume))
        return {"resume_md": fixed}

    g = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    writers = [f"write_{s.lower()}" for s in SECTIONS]
    for s, node in zip(SECTIONS, writers):
        g.add_node(node, make_writer(s))
        g.add_edge("plan", node)          # fan out: all four writers run in the same step
    g.add_node("assemble", assemble_node)
    g.add_node("check", check_node)
    g.add_node("fix", fix_node)
    g.add_edge(START, "plan")
    g.add_edge(writers, "assemble")       # fan in: waits for all four writers
    g.add_edge("assemble", "check")
    g.add_conditional_edges("check", lambda s: "fix" if s["problems"] else END, ["fix", END])
    g.add_edge("fix", END)

    out = g.compile().invoke({"plan": {}, "plan_ok": False, "sections": {}, "resume_md": "", "problems": []})
    return {"resume_md": out["resume_md"], "hit_budget": False, "finished": out["plan_ok"]}
