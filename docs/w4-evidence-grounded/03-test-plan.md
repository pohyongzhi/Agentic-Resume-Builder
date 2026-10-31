# W4 test plan

All tests run with `pytest -q` and use **fake model replies**. Never call Ollama. Follow the style
of `tests/test_w3_critic.py` (monkeypatch the module's `chat`, return
`SimpleNamespace(content=json.dumps(...))`).

Two `chat` imports need patching in workflow tests: `workflows.w4_evidence.chat` (creator) and
`common.grounding.chat` (verifier). A single fake can serve both by looking at the system prompt:
`messages[0]["content"] == grounding.VERIFIER_SYSTEM` means it's the verifier.

**Don't hard-code span IDs** (`"S7"`) in tests. Look them up by text with a helper:

```python
def sid(spans, needle):
    """ID of the citeable span whose text contains `needle` (exactly one must match)."""
    hits = [s.id for s in spans if s.kind == "content" and needle in s.text]
    assert len(hits) == 1, (needle, hits)
    return hits[0]
```

## Fixtures

Use the existing `candidate` and `good_resume` fixtures from `tests/conftest.py` (Markdown resume
"Alex Tan"). Add these fixtures in `tests/test_w4_spans.py` (or a new `tests/w4_fixtures.py`).

### `PDF_RAW`: text shaped like `pypdf` output from a real two-column resume (fictional person)

Note the trailing spaces, the `•` bullets, the wrapped continuation lines with no marker, the
entry header lines with right-aligned dates, and the `Excel -based` / `self -service` artefacts.
These are all real `pypdf` behaviours.

```python
PDF_RAW = (
    "Jordan Lim \n"
    " \n"
    "Singapore • jordan.lim@example.com • (+65) 9000 0000 \n"
    "About Me\n"
    "I am a final-year Business Analytics student who enjoys turning messy data into clear reports for\n"
    "teams. I like working with people and improving everyday processes. \n"
    "Experience \n"
    "Data Analyst Intern - Example Bank               May 2026 – Present \n"
    "• Built automated reporting workflows in Excel and Python, reducing manual effort for the\n"
    "operations team \n"
    "• Gathered requirements from business users and wrote user stories for dashboard\n"
    "enhancements \n"
    "Operations Assistant (National Service) - Example Agency      Apr 2023 – Aug 2024 \n"
    "• Processed approval requests through an internal SAP-based system.  \n"
    "• Designed an Excel -based tracker to streamline approvals, enabling user self -service \n"
    "Education \n"
    "Example University       Aug 2024 – Dec 2027 \n"
    "Bachelor of Science (Business Analytics), GPA: 3.6/4.0 \n"
    "Skills & Interests \n"
    "Skills: Python, Excel, SQL, SAP, Figma \n"
)

PDF_CANDIDATE = {
    "name": "Jordan Lim",
    "contact": {"email": "jordan.lim@example.com", "phone": "(+65) 9000 0000", "location": "Singapore", "links": []},
    "summary": "",
    "experience": [
        {"employer": "Example Bank", "title": "Data Analyst Intern", "start": "May 2026", "end": "Present", "bullets": []},
        {"employer": "Example Agency", "title": "Operations Assistant (National Service)",
         "start": "Apr 2023", "end": "Aug 2024", "bullets": []},
    ],
    "education": [{"institution": "Example University", "degree": "Bachelor of Science (Business Analytics)",
                   "start": "Aug 2024", "end": "Dec 2027", "details": []}],
    "projects": [], "skills": ["Python", "Excel", "SQL", "SAP", "Figma"],
    "certifications": [], "achievements": [], "raw_text": PDF_RAW,
}
```

---

## §1 `tests/test_w4_schemas.py`

| # | Test | Expect |
|---|---|---|
| 1.1 | `Claim.model_validate({"text": "x", "sources": ["S3", "S4"]})` | sources `["S3", "S4"]` |
| 1.2 | sources `"S3, S4"` | `["S3", "S4"]` |
| 1.3 | sources `["[s3]", "S 4", "S3"]` | `["S3", "S4"]` (normalised, de-duplicated, in order) |
| 1.4 | sources `["foo"]` | `[]` |
| 1.5 | `GroundedExperience` with `bullets=["Did a thing"]` (plain string) | one Claim, text `"Did a thing"`, sources `[]` |
| 1.6 | `GroundedResume` with `summary="One sentence."` (plain string) | one Claim with empty sources |
| 1.7 | `GroundedResume` with `summary=None`, `experience=None` | empty lists, no error (inherits `Schema` null handling) |
| 1.8 | `llm_schema(GroundedResume)` | a dict with `"properties"`, and `json.dumps` works |
| 1.9 | `llm_schema(VerifyBatch)` | has `verdicts` |

---

## §2 `tests/test_w4_spans.py`

### Markdown resume (`candidate` fixture)

| # | Test | Expect |
|---|---|---|
| 2.1 | `split_spans(candidate["raw_text"], candidate)` | IDs are `S1..Sn` in order, no gaps |
| 2.2 | The span containing `"Cleaned and merged shipment data"` | `kind="content"`, `section="experience"`, `entry=0`; text has no leading `-` |
| 2.3 | The line `**Business Analyst Intern, Example Logistics Pte Ltd** (May 2025 - Aug 2025)` | `kind="entry_header"`, not in `citeable(spans)` |
| 2.4 | `## Experience`, `## Skills` | `kind="heading"`, not citeable |
| 2.5 | The skills line | `section="skills"`, citeable |
| 2.6 | Name + email lines at the top | `section="header"`, not citeable |
| 2.7 | Offset invariant, for every span | `_norm(span.text)` is contained in `_norm(" ".join(raw_text[span.start:span.end].split()))` after removing bullet markers (`•-*`) and the `" -"` hyphen artefact |

### PDF-shaped resume (`PDF_RAW`, `PDF_CANDIDATE`)

| # | Test | Expect |
|---|---|---|
| 2.8 | Wrapped bullet | exactly one span whose text is `"Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"` (joined, whitespace collapsed) |
| 2.9 | Second wrapped bullet | `"Gathered requirements from business users and wrote user stories for dashboard enhancements"` |
| 2.10 | Entry assignment | both Example Bank bullets have `entry=0`; the two Example Agency bullets have `entry=1` |
| 2.11 | Entry headers | `"Data Analyst Intern - Example Bank ... May 2026 – Present"` and `"Example University ... Aug 2024 – Dec 2027"` are `entry_header` |
| 2.12 | Hyphen artefacts | a span contains `"Excel-based tracker"` and `"self-service"` |
| 2.13 | About Me paragraph | one citeable span (≤ 300 chars) in `section="summary"`, joining both lines |
| 2.14 | Degree line | `"Bachelor of Science (Business Analytics), GPA: 3.6/4.0"` is a citeable span with `section="education"`, `entry=0` |
| 2.15 | Contact line with `•` separators | in `section="header"`, not treated as a bullet, not citeable |
| 2.16 | Long block split | a paragraph of 3 sentences totalling > 300 chars becomes 3 spans, each with correct offsets (invariant 2.7) |
| 2.17 | Empty text | `split_spans("", candidate)` → `[]` |

### Prompt rendering

| # | Test | Expect |
|---|---|---|
| 2.18 | `render_spans_for_prompt(spans, PDF_CANDIDATE)` | contains `"== EXPERIENCE: Data Analyst Intern, Example Bank (May 2026 - Present) =="`, followed by the two Example Bank spans as `"[S<n>] ..."` lines before the next `==` header |
| 2.19 | Same | contains no `entry_header` or `heading` span text as a `[S..]` line |

---

## §3 `tests/test_w4_grounding.py`

Build `GroundedResume` objects in the tests from `PDF_CANDIDATE` and its spans. Helper:

```python
def resume_with(bullets_bank, bullets_agency=(), summary=()):
    """Each bullet/summary item is (text, [span ids])."""
    return GroundedResume.model_validate({
        "name": "Jordan Lim", "contact": {"email": "jordan.lim@example.com"},
        "summary": [{"text": t, "sources": s} for t, s in summary],
        "experience": [
            {"employer": "Example Bank", "title": "Data Analyst Intern", "start": "May 2026", "end": "Present",
             "bullets": [{"text": t, "sources": s} for t, s in bullets_bank]},
            {"employer": "Example Agency", "title": "Operations Assistant (National Service)",
             "start": "Apr 2023", "end": "Aug 2024",
             "bullets": [{"text": t, "sources": s} for t, s in bullets_agency]},
        ],
        "education": [{"institution": "Example University", "degree": "Bachelor of Science (Business Analytics)",
                       "start": "Aug 2024", "end": "Dec 2027", "details": []}],
        "skills": ["Python", "Excel"],
    })
```

### A. `iter_claims`

| # | Test | Expect |
|---|---|---|
| A1 | 2 summary + 2 Bank + 1 Agency claims | IDs `summary.0, summary.1, exp.0.0, exp.0.1, exp.1.0` in that order; `entry_label` `"Data Analyst Intern, Example Bank"` |

### B. `check_citations`

| # | Claim | Expect problems |
|---|---|---|
| B1 | Bank bullet citing the "Built automated reporting" span | `[]` |
| B2 | Bank bullet with `sources=[]` | `["no_citation"]` |
| B3 | Bank bullet citing `"S999"` | contains `bad_citation` |
| B4 | Bank bullet citing the Bank **entry header** span ID | contains `bad_citation` |
| B5 | Bank bullet citing an **Agency** bullet span | contains `wrong_entry` |
| B6 | Summary sentence citing an Agency span | `[]` (summary is exempt) |
| B7 | Bank bullet citing 4 valid Bank-or-summary spans | contains `too_many_sources` |
| B8 | Education detail citing a Bank span | contains `wrong_section` |
| B9 | Draft entry with an employer not in the candidate (`"Other Corp"`) | no `wrong_entry` (check skipped; the harness rejects it later) |
| B10 | Return value | every claim ID from `iter_claims` is a key |

### C. `rule_check`

`cited` = the cited span text. `job` = a JD string containing
`"Experience with SQL, Tableau and Power BI. Familiarity with A/B testing, KPIs and Jira is a plus."`

| # | Claim | Cited text | Expect |
|---|---|---|---|
| C1 | `"Built automated Excel and Python reporting workflows that cut manual effort for operations"` | Bank bullet 1 | `[]` |
| C2 | `"Cut manual reporting effort by 40% with automated workflows"` | Bank bullet 1 | `["new_number: 40%"]` |
| C3 | `"Built reporting workflows in 2026"` | Bank bullet 1 | `[]` (years ignored) |
| C4 | `"Built automated reporting in Excel, Python and Tableau"` | Bank bullet 1 | `["new_skill: Tableau"]` |
| C5 | `"Gathered requirements and defined KPIs for dashboards"` | Bank bullet 2 | contains `new_skill: KPIs` (or `KPI`) |
| C6 | `"Ran A/B testing on dashboard enhancements"` | Bank bullet 2 | contains `new_skill: A/B testing` (or `A/B`) |
| C7 | `"Led the automation of reporting workflows"` | Bank bullet 1 | `["inflated: led"]` |
| C8 | `"Advanced Excel user who built automated reporting"` | Bank bullet 1 | contains `inflated: advanced` |
| C9 | `"Built reporting workflows; maintained ledger files"` | `"Built reporting workflows and maintained ledger files"` | `[]` (`led` must not match `ledger`) |
| C10 | `"Led weekly stand-ups"` | `"Led weekly stand-ups with the team"` | `[]` (the source says it too) |
| C11 | `"Used SQL to clean data"` | `"Cleaned data using SQL queries"` | `[]` (SQL is in the cited text) |
| C12 | `skill_terms(job)` | | contains `SQL`, `Tableau`, `Power BI`, `Jira`; does **not** contain `Experience`, `Familiarity` (sentence-initial words) |

### D. Finalise helpers

| # | Test | Expect |
|---|---|---|
| D1 | `drop_unsupported(resume, {"exp.0.1": ["inflated: led"], "exp.0.0": []})` | Bank keeps only bullet 0; the entry itself stays |
| D2 | Drop all bullets of an entry | entry kept with `bullets=[]` |
| D3 | `filter_skills(["Python", "Tableau", "excel"], parsed=PDF_CANDIDATE["skills"], raw_text=PDF_RAW)` | kept `["Python", "excel"]`, dropped `["Tableau"]` |
| D4 | `to_generated(resume)` | `GeneratedResume.model_validate(...)` succeeds; summary is a plain string joining the summary claims; bullets are strings; the output contains no `"sources"` key and no `[S` anywhere in `json.dumps` |

### E. `llm_verify` (monkeypatch `common.grounding.chat`)

| # | Test | Expect |
|---|---|---|
| E1 | 3 items, fake returns supported for all | 1 chat call; all `supported=True`; cache now has 3 keys |
| E2 | Same 3 items again with that cache | **0** chat calls; same verdicts |
| E3 | 2 cached + 1 new item | 1 chat call whose user message contains only the new claim's `claim_id` |
| E4 | Fake omits one `claim_id` | that claim → `supported=False`, reason `"verifier gave no verdict"` |
| E5 | Fake returns non-JSON text | all uncached items unsupported with reason `"verifier output invalid"`; no exception |
| E6 | Fake returns `supported=False, reason="adds 'for executives'"` | verdict kept with that reason |
| E7 | Every call | uses `format=llm_schema(VerifyBatch)` (assert on the kwargs the fake received) and the `VERIFIER_SYSTEM` system prompt |

---

## §4 `tests/test_w4_workflow.py`

Set `config.OUTPUT_DIR` to `tmp_path / "out"` with monkeypatch, as `test_w3_critic.py` does.
Use the `candidate` fixture (Markdown, Alex Tan) and `job_data = {"title": "Data Analyst",
"source_text": "Need SQL, Power BI and Tableau. KPIs a plus.", "responsibilities": [],
"required_qualifications": [], "preferred_qualifications": []}`.

Build drafts dynamically from the real spans so the IDs are right:

```python
spans = split_spans(candidate["raw_text"], candidate)
DASH = sid(spans, "Built weekly sales dashboards")
CLEAN = sid(spans, "Cleaned and merged shipment data")
SKILLS = sid(spans, "Excel, Power BI, SQL")

def draft(bank_bullets, summary=(("Information Systems student who builds sales dashboards.", [DASH]),),
          skills=("SQL", "Power BI", "Excel")):
    return {"name": "Alex Tan", "contact": {"email": "alex.tan@example.com"},
            "summary": [{"text": t, "sources": s} for t, s in summary],
            "experience": [{"employer": "Example Logistics Pte Ltd", "title": "Business Analyst Intern",
                            "start": "May 2025", "end": "Aug 2025",
                            "bullets": [{"text": t, "sources": s} for t, s in bank_bullets]}],
            "education": [{"institution": "Example University",
                           "degree": "Bachelor of Science (Information Systems)", "end": "expected 2026"}],
            "skills": list(skills)}

GOOD = [("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
        ("Cleaned and merged shipment data from three spreadsheets with SQL.", [CLEAN])]
BAD = [GOOD[0], ("Led a team that cut data cleaning time by 40%.", [CLEAN])]
```

The fake `chat` keeps a list of calls (system prompt, user prompt, `format`) and returns scripted
creator drafts in order, plus verifier replies (default: all supported).

| # | Scenario | Expect |
|---|---|---|
| W1 | Draft = GOOD, verifier approves | creator called **once**, verifier **once**; `resume_md` starts with `"# Alex Tan"`, contains `"Power BI sales dashboards"`, contains no `"[S"`; every `evidence` row `status="kept"`, `rounds_to_pass=0`; `w4_stats["dropped"] == 0`; tracker has `type="node"` events for `split_spans, draft, check, llm_verify, finalise`, plus one `w4_round` and one `w4_final`; two `tool` events (`validate_resume`, `render_resume`), both `ok=True` |
| W2 | Draft = BAD, then GOOD on revise | creator called twice; the revise user prompt contains `"exp.0.1"`, `"inflated"` and `"new_number"`; final has 2 kept claims; the fixed claim has `rounds_to_pass=1`; the **second verifier call doesn't include the unchanged claim** `exp.0.0` (cache) |
| W3 | Draft = BAD every time | creator called **3** times (draft + 2 revisions); `resume_md` doesn't contain `"40%"` or `"Led a team"`; `w4_stats["dropped"] == 1`; `hit_budget is True`; the evidence row for the dropped claim has `status="dropped"` and problems including `new_number: 40%` |
| W4 | One GOOD bullet with `sources=[]` every time | dropped with `no_citation`; resume still rendered |
| W5 | GOOD draft, but the verifier rejects `summary.0` once with reason `"adds 'regional'"`, then approves | revise prompt contains `"not_entailed"` and the reason; final kept |
| W6 | Creator returns invalid JSON twice | `errors[0]` starts with `"W4_DRAFT_FAILED"`; no `resume_md`; verifier never called |
| W7 | GOOD draft with skills `["SQL", "Tableau"]` | `"Tableau"` not in `resume_md`; `w4_stats["skills_dropped"] == 1`; validation passes |
| W8 | Revise returns invalid JSON twice after a BAD first draft | keeps the BAD draft, goes to finalise, drops the bad claim, renders |
| W9 | Candidate with empty `raw_text` | `errors[0]` starts with `"W4_NO_SPANS"` |
| W10 | `"w4_evidence" in workflows.WORKFLOWS` | True |
| W11 | **End-to-end through `run(pair, tracker)`**: write `candidate["raw_text"]` to `tmp_path/"resume.md"`, monkeypatch `common.extraction._structured` to return the parsed candidate / job models (see `tests/test_pipeline.py` for the pattern), and patch both `chat`s | the returned dict has `resume_md`, `finished=True`, `errors=[]`, a non-empty `evidence` list and `w4_stats`. This proves the extra keys survive the shared pipeline (`ResumeState`) |

---

## §5 Manual checks with the real model (not pytest)

After all tests pass, with Ollama running:

1. `python run_all.py --workflows w4_evidence --limit 1 --repeats 1`
2. Open the last line of `results/results.jsonl` and check:
   - `error` is `null`
   - the trace has `w4_round` events; `first_draft_citation_coverage` ≥ 0.5 (if not, fix the prompt layout first)
   - `resume_md` has no `[S` markers
3. `python app.py`, select W4, upload a real resume and paste a JD. Read the evidence report:
   do the ❌ rows really deserve to be dropped? Write down 3 examples for the presentation.
## Audit regression coverage

Additional regressions cover:

- Project descriptions passing citation, rule and LLM checks, including dropping uncited
  strings, invented metrics and verifier-rejected descriptions before rendering.
- Markdown job headings assigning entry ownership, cross-job citations being rejected, and
  unassigned/other-section spans not supporting a known job's bullets.
- Complete numeric token comparisons (`40%` versus `140%`, currency and decimals).
- Hyphenated inflation terms, explicit inflections, and `led` versus `ledger`.
- Skill extraction tracking sentence position per occurrence and preserving technical punctuation.
- First-pass metrics surviving claim deletion, movement and replacement.
- Null citation lists becoming empty lists without raising an exception.
