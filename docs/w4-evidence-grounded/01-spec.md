# W4 Evidence-grounded generation: specification

## 1. Goal

W4 is variant 4 of 7 in the study. All variants share the same model (`qwen3.5:4b-q4_K_M`), the same
extraction agents, the same harness and the same evaluator; only the **resume creator** differs.

W4's technique: **make the model prove every claim.**

- The original resume text is split into numbered spans (`S1`, `S2`, …).
- The creator must attach span IDs (`sources`) to every bullet and summary sentence it writes.
- Code rejects any claim with no citation or an invalid one.
- A verifier (code rules first, then one LLM check) compares each claim with **only the text of
  the spans it cited**.
- Rejected claims go back to the creator with the reason. After a fixed number of rounds, anything
  still unsupported is **dropped, not shipped**.
- The cleaned resume then goes through the existing `ToolHarness` (`validate_resume` → `render_resume`),
  exactly like W2 and W3.

How W4 differs from W3 (multi-critic): W3 asks a model "is this true?". W4 forces the model to
point at its evidence and checks that pointer in code, so a lenient model can't wave an
unsupported claim through.

## 2. Where W4 sits in the existing repo

Do not change the shared pipeline. W4 supplies a creator node, the same way W2 and W3 do:

```
common/pipeline.py (unchanged)

  START -> resume_parser --\
                            +-> check_extractions --(errors)--> END
  START -> jd_retriever  --/            \--(ok)--> resume_creator -> END
                                                   ^^^^^^^^^^^^^^
                                                   W4's make_creator() returns this node
```

Inside W4's creator node there is a LangGraph subgraph:

```
split_spans ──> draft ──> check_citations ──> rule_check ──> llm_verify ──> decide
   (code)       (LLM)        (code)            (code)         (LLM)          │
                  ▲                                                          │ rejected claims left
                  └──────────────── revise (LLM) <────── and round < MAX ────┤
                                                                             │ all supported,
                                                                             ▼ or out of rounds
                                                                         finalise (code)
                                       drop unsupported → strip citations → ToolHarness validate + render
```

Existing pieces to reuse (read them before implementing):

| Existing | Use it for |
|---|---|
| `common.pipeline.run_pipeline(creator_factory, tracker, resume_path, job_url, pasted_jd, run_id)` | Running parser + JD retriever + W4 creator |
| `common.llm.chat(messages, tracker, format=schema)` | Every model call (tracks tokens and time) |
| `common.schemas.Schema` | Base class that tolerates small-model output quirks (null → default, lone string → list) |
| `common.schemas.llm_schema(model)` | JSON schema for Ollama `format=` |
| `common.schemas.GeneratedResume`, `Contact`, `Education`, `Experience`, `Project` | Final shape handed to the harness |
| `common.harness.ToolHarness` | Final `validate_resume` + `render_resume` (protected facts, safe output path) |
| `common.harness._norm`, `common.harness._in_text` | Text normalisation and whole-word matching (import them; don't copy). Note `_in_text(term, text)` expects `text` already passed through `_norm` |
| `common.prompts.STRUCTURED_CREATOR_SYSTEM` | Shared creator rules. W4 **appends** its citation rules; never edits the shared text |
| `workflows/w3_critic.py` | Pattern for a creator that is its own LangGraph subgraph, plus `_tracked_execute` |

State available to the creator node (from `common/state.py`): `candidate_data` (dict, includes
`raw_text`, the original resume text) and `job_data` (dict, includes `source_text`).

## 3. Data model

### 3.1 Span (code only, never produced by the LLM)

```python
@dataclass(frozen=True)
class Span:
    id: str            # "S1", "S2", ... in document order
    text: str          # cleaned text of the span (bullet marker removed, wrapped lines joined)
    start: int         # char offset in candidate_data["raw_text"] where the span begins
    end: int           # char offset where it ends (raw_text[start:end] covers the original lines)
    section: str       # "header" | "summary" | "experience" | "education" | "projects" | "skills" | "other"
    entry: int | None  # index into candidate_data["experience"] (or ["projects"]/["education"]) this span belongs to
    kind: str          # "heading" (section title), "entry_header" (employer/title/date line) or "content"
```

### 3.2 Pydantic models (add to `common/schemas.py`, below the existing ones)

```python
class Claim(Schema):
    """One sentence/bullet the creator wrote, plus the span IDs it is based on."""
    text: str
    sources: list[str] = Field(default_factory=list)   # e.g. ["S12", "S13"]

class GroundedExperience(Schema):
    employer: str
    title: str
    start: str = ""
    end: str = ""
    bullets: list[Claim] = Field(default_factory=list)

class GroundedEducation(Schema):
    institution: str
    degree: str
    start: str = ""
    end: str = ""
    details: list[Claim] = Field(default_factory=list)

class GroundedProject(Schema):
    name: str
    description: Claim = Field(default_factory=lambda: Claim(text=""))
    bullets: list[Claim] = Field(default_factory=list)

class GroundedResume(Schema):
    name: str
    contact: Contact = Field(default_factory=Contact)
    summary: list[Claim] = Field(default_factory=list)      # summary = list of cited sentences
    experience: list[GroundedExperience] = Field(default_factory=list)
    education: list[GroundedEducation] = Field(default_factory=list)
    projects: list[GroundedProject] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)          # checked by the harness, no citation needed
    certifications: list[str] = Field(default_factory=list)  # same

class ClaimVerdict(Schema):
    claim_id: str
    supported: bool
    reason: str = ""

class VerifyBatch(Schema):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
```

**Small-model tolerance (required).** Give `Claim` a `model_validator(mode="before")` that accepts:

- a bare string instead of a Claim object → `{"text": s, "sources": []}`. This then fails the
  citation check, which is the correct outcome.
- `sources` given as one string like `"S3, S4"` or `"[S3] [S4]"` → `["S3", "S4"]`
- lower-case or bracketed IDs (`"s3"`, `"[S3]"`) → `"S3"`
- `sources: null` → `[]` (the claim then fails citation validation).

Project descriptions are claims too: a nonempty description must pass the same citation,
rule and LLM checks as project bullets. A legacy plain-string description is accepted as an
uncited claim and must be revised or dropped. An empty description is omitted from verification.
The final `GeneratedResume` still contains a plain-string description.

Do this in `Claim` itself so it works wherever a Claim appears (bullets, details, summary).
Note that the existing `Schema` validator wraps a lone string into a list only for `list[str]` fields.
For `list[Claim]` fields, the bare-string-to-Claim conversion must happen in the parent models
(`GroundedExperience.bullets`, `GroundedEducation.details`, `GroundedProject.bullets`,
`GroundedResume.summary`). Do it with a `field_validator(..., mode="before")` on those fields that
maps `str` items to `{"text": item, "sources": []}` and wraps a single string into a one-item list.

### 3.3 Claim IDs

After a draft is parsed, give every claim a stable ID based on its position:

- `summary.0`, `summary.1`, …
- `exp.{i}.{j}`: experience entry i (index in the draft), bullet j
- `edu.{i}.{j}`, `proj.{i}.{j}`
- `proj.{i}.description` for a nonempty project description

Use these IDs in the verifier, the evidence report and the revise prompt.
Positions may change after deletion or reordering. Track first-pass history separately by
section, entry label, claim text and source IDs so revision metrics follow the claim itself.

## 4. Node-by-node behaviour

### 4.1 `split_spans` (code), in `common/spans.py`

`split_spans(raw_text: str, candidate: dict) -> list[Span]`

Real input is messy, so these rules are needed. (Checked against how `pypdf` extracts a real
two-column resume: bullets start with `•`, long bullets wrap onto the next line with no marker,
entry header lines end in a date range, and paragraphs wrap mid-sentence.)

1. Walk `raw_text` line by line, keeping each line's character offsets.
2. Blank / whitespace-only lines end the current block.
3. **Bullet line:** starts with `•`, `-`, `*`, `–`, `▪`, `●` or `\d+[.)]` after optional whitespace.
   It starts a new block. Strip the marker from `text`, but keep the offsets of the full line.
4. **Section heading:** a short line (≤ 4 words, no full stop) whose normalised text matches a known
   heading, or a Markdown heading (`#`, `##`). Known headings, case-insensitive:
   `summary, about me, profile, objective → summary`; `experience, work experience, employment,
   professional experience → experience`; `education → education`; `projects → projects`;
   `skills, skills & interests, technical skills → skills`; `leadership, activities, awards,
   certifications, volunteering → other`. A heading sets the current section and becomes its own
   span with `kind="heading"`.
   A Markdown heading whose text is **not** a known heading (e.g. `# Alex Tan`) does not change the
   section; it's still a `kind="heading"` span (so before the first known heading it's part of the
   header section, rule 9).
   Within experience, education and projects, first test unknown Markdown headings for an
   entry match (rule 5). A matching heading is an `entry_header` and sets its entry. An
   unrecognized Markdown heading clears entry ownership rather than inheriting the prior entry.
5. **Entry header line:** inside experience / education / projects, a line that contains a date
   range (`Jan 2024 – Present`, `May 2025 - Aug 2025`, `2019`, `2018 - 2022`, `expected 2026`),
   **or** contains the normalised employer / institution / project name of a parsed candidate entry.
   It becomes its own span with `kind="entry_header"` and moves the current `entry` to the
   matching candidate entry. Match on employer for experience, institution for education and name
   for projects, using `common.harness._in_text`. If the line has a date range but no name match,
   advance `entry` by one in that section.
6. **Continuation line:** any other non-blank line directly after a bullet or content line in the
   same block is joined onto that block with a single space. This repairs PDF wrapping (e.g.
   `• Drive product enhancements from requirements to` + `stakeholder alignment and delivery`).
   Any other non-blank line that does **not** follow a bullet or content line (e.g. the first line
   after a heading or an entry header) starts a new content block. Joining two different items that
   wrap together under one entry (e.g. a degree line followed by a coursework line) is acceptable,
   because they share the same section and entry.
7. Clean span `text`: collapse whitespace, and fix PDF hyphen artefacts `"Excel -based"` →
   `"Excel-based"` (regex `(\w) -(\w)` → `\1-\2`). Never change offsets.
8. **Long blocks:** if a content block is longer than `SPAN_MAX_CHARS` (300), split it into
   sentences (on `. `, `! `, `? ` followed by a capital letter or `I'`) and make each sentence its
   own span. Offsets must still point into the right part of `raw_text`.
9. Lines before the first heading form the header section (name, contact). Mark them
   `section="header"`, `kind="heading"`. They can never be cited.
10. Assign IDs `S1…Sn` in document order, after all splitting.

Invariant: for every span, `normalise(raw_text[span.start:span.end])` contains
`normalise(span.text)` after removing bullet markers and joining lines. This is a test.

### 4.2 `draft` (LLM)

One `chat(..., format=llm_schema(GroundedResume))` call.

- System: `STRUCTURED_CREATOR_SYSTEM + W4_CITATION_RULES` (W4_CITATION_RULES is defined in the W4 module).
- User: the same job and candidate JSON that `creator_prompt()` builds, but replace its
  "ORIGINAL RESUME TEXT" part with the **numbered source spans, grouped by section and entry**:

```
SOURCE SPANS (cite these by ID):

== EXPERIENCE: Business Analyst Intern, Example Logistics Pte Ltd (May 2025 - Aug 2025) ==
[S6] Built weekly sales dashboards in Excel and Power BI used by the regional sales team.
[S7] Cleaned and merged shipment data from three spreadsheets using SQL queries.

== EXPERIENCE: Student Assistant, University IT Helpdesk (Jan 2024 - Dec 2024) ==
[S9] Answered student queries on email and walk-in, about 30 cases per week.

== SUMMARY ==
[S3] Final-year Information Systems student with internship experience in data reporting.
```

Only list `content` spans (headings and entry headers are not citeable). The JD and candidate JSON
are still included as in `creator_prompt`; build the prompt with a W4 helper rather than editing
`creator_prompt`.

`W4_CITATION_RULES`, appended to the system prompt:

```
CITATION RULES (mandatory):
- Every summary sentence, experience bullet, education detail, project description and project bullet is an object
  {"text": ..., "sources": [...]}. "sources" lists the IDs (e.g. "S7") of the source spans the
  sentence is based on. Never leave "sources" empty.
- A bullet under a job may only cite spans listed under that same job. Summary sentences may cite
  any span.
- Cite at most 3 spans per sentence.
- The sentence may reword, shorten or combine its cited spans, but must not add anything they don't
  say: no new tools, skills, numbers, scope, seniority ("led", "advanced", "expert") or outcomes.
- If the job asks for something no span supports, leave it out. Do not cite a span that doesn't
  actually say it.
```

If the reply fails to parse as `GroundedResume`, retry the call once with the validation error
appended. If it fails again, stop with error `W4_DRAFT_FAILED: ...`.

### 4.3 `check_citations` (code), in `common/grounding.py`

`check_citations(resume: GroundedResume, spans: list[Span], candidate: dict) -> dict[claim_id, list[str]]`
returns the problems per claim (empty list = passed). Problem codes:

| Code | Rule |
|---|---|
| `no_citation` | `sources` is empty |
| `bad_citation` | an ID isn't a known span ID, or points at a `heading` / `entry_header` / `header` span |
| `too_many_sources` | more than `MAX_SOURCES_PER_CLAIM` (3) IDs |
| `wrong_entry` | an experience bullet cites a span whose `entry` is a different job. Match the draft's experience entry to the candidate's by normalised (employer, title), falling back to employer only. If the draft entry can't be matched, skip this check (the harness rejects unknown entries later). Same for education and projects. |
| `wrong_section` | an education detail cites a non-education span, or a project bullet cites a non-project span. Summary claims are exempt. |

Entry citation checks require a resolved source entry for a matched draft entry. Experience
claims must cite experience spans belonging to that same job; unassigned or other-section
spans cannot establish job-specific experience. Education and project claims likewise reject
unassigned spans. Summary claims remain exempt from entry restrictions.

### 4.4 `rule_check` (code), in `common/grounding.py`

`rule_check(claim_text: str, cited_text: str, job_text: str) -> list[str]`, where
`cited_text` is the cited spans joined with `" "`. Only run on claims that passed `check_citations`.

| Code | Rule | Why it's code, not LLM |
|---|---|---|
| `new_number` | Every number in the claim (regex `\d[\d,]*(?:\.\d+)?%?`, also `$`-prefixed) must appear in `cited_text`. Ignore 4-digit years 1950–2035. | Invented metrics are the most common resume fabrication and are trivially checkable |
| `new_skill` | Collect *skill-like terms* from `job_text`: tokens that are ALL-CAPS ≥ 2 letters (SQL, KPI, AWS), contain `/ + # .` inside a word (A/B, C#, Node.js), or are Capitalised words not at the start of a sentence (Tableau, Jira, Figma). Merge consecutive such tokens into one phrase (`Power BI`, `Google Analytics`), and also keep each token on its own. A plural `s` is ignored when matching (`KPIs` ↔ `KPI`). Ignore a stop-list of generic capitalised words (I, We, The, You, Our, Singapore, January–December, Monday–Sunday). Flag any such term that appears in the claim (via `_in_text`) but not in `cited_text`. | This is the "fabrication bait" case: the JD asks for a tool, so the model slips it in |
| `inflated` | Match each `INFLATION_TERMS` word or its explicit inflections in the claim; flag it if none occur in the cited text. Preserve hyphenated terms. `ledger` must not match `led`. See defaults in §7. | Inflation is a vocabulary pattern, so a list is transparent and explainable |

Deterministic matching clarifications:

- Compare complete numeric tokens: `40%` is not supported by `140%`, nor `$40` by `$400`.
- Determine sentence-initial capitalization per occurrence, not by excluding a word everywhere
  because it began an earlier sentence. Join skill phrases across whitespace, not list punctuation.
- Match inflation terms as whole words with explicit inflections (e.g. `lead`, `leading`,
  `leadership`) and preserve hyphenated terms (`world-class`, `cutting-edge`). `ledger` does not
  establish support for `led`.

### 4.5 `llm_verify` (LLM), in `common/grounding.py`

`llm_verify(items: list[dict], tracker, cache: dict) -> dict[claim_id, ClaimVerdict]`

- `items`: claims that passed citation and rule checks, as `{"claim_id", "claim", "cited_text"}`.
- **One batched call per round** with `format=llm_schema(VerifyBatch)`, not one call per claim.
- Skip claims whose `(claim, tuple(sources))` is already in `cache`, and reuse their verdict. Store
  new verdicts in `cache`. Unchanged claims are never re-verified in later rounds.
- A claim missing from the reply, or with a malformed verdict, counts as unsupported with reason
  `"verifier gave no verdict"` (fail safe).

Verifier system prompt (defined in the W4 module):

```
You check resume claims against the source text they cite. For each item, decide if the claim is
fully supported by its cited source text alone.
Supported: the claim says the same thing in different words, shortens it, or combines the cited
sources without adding meaning.
NOT supported: the claim adds any tool, skill, number, scope, seniority, outcome, purpose,
audience or result that the cited text does not state.
Judge only against the cited text, not general knowledge or the job description.
Return one verdict per claim_id with a one-sentence reason when unsupported.
```

### 4.6 `decide` / routing

- Combine the problems: citation problems, then rule problems, then `not_entailed` for LLM-rejected
  claims (with the LLM's reason).
- If no claim has problems → `finalise`.
- Else if `round < MAX_GROUNDING_ROUNDS` (2) → `revise`.
- Else → `finalise` (the remaining rejected claims get dropped there).

### 4.7 `revise` (LLM)

One `chat(..., format=llm_schema(GroundedResume))` call, same system prompt as `draft`. The user
prompt is the draft prompt plus:

```
CURRENT DRAFT (JSON):
{the current GroundedResume}

REJECTED CLAIMS – fix each one by rewording it so it only says what its cited spans say, citing
the correct spans, or delete it if no span supports it:
- exp.0.2 "Cut manual data entry by 40%" cites [S8]: new_number (40% is not in S8)
- summary.1 "Proven leader in data analytics" cites [S3]: inflated (proven, leader)
...
Keep every claim that is not listed exactly as it is. Return the complete resume JSON.
```

Increment `round`. Same parse/retry handling as `draft` (on a second failure, keep the previous
draft and go to `finalise`).

### 4.8 `finalise` (code)

1. Remove every claim that still has problems. Keep entries even if all their bullets were removed;
   the harness requires employer/title/dates, not bullets.
2. Drop `skills` / `certifications` the harness would reject: keep an item only if its normalised
   form is in the candidate's parsed list or `_in_text(item, normalised raw_text)`. This avoids
   failing the whole run over one skill the creator added.
3. Convert to a `GeneratedResume` dict: `summary` = supported summary claim texts joined with
   `" "`; bullets/details and project descriptions = claim texts; everything else copied.
4. `ToolHarness(candidate_data, resume_path=..., run_id=...)`. Run `validate_resume` then
   `render_resume` through a tracked helper (copy `_tracked_execute` from W3) so both tool calls are
   logged.
5. If validation still fails → `errors=["W4_FINAL_VALIDATION_FAILED: " + "; ".join(errors)]`.
6. Return `resume_md`, `output_path`, `generated_resume`, `hit_budget`, plus W4 extras:
   `evidence` (the report, §5) and `w4_stats` (§6).

`hit_budget` = True if the loop stopped because `round == MAX_GROUNDING_ROUNDS` while claims were
still rejected.

The rendered Markdown must contain **no** citation markers (`[S12]`). Citations live only in the
evidence report.

## 5. Evidence report (returned and shown in the app)

A list with one row per claim in the final draft before dropping:

```python
{"claim_id": "exp.0.1", "section": "experience", "entry": "Business Analyst Intern, Example Logistics Pte Ltd",
 "claim": "Built weekly Power BI sales dashboards for the regional sales team.",
 "sources": [{"id": "S6", "text": "Built weekly sales dashboards in Excel and Power BI used by ..."}],
 "status": "kept" | "dropped",
 "problems": [],            # final problem codes, e.g. ["new_skill: Tableau"]
 "rounds_to_pass": 0}       # 0 = passed first time, 1 = fixed in revision 1, ...
```

## 6. What W4 records (workflow metrics from the project brief)

Log through the existing `Tracker` (`tracker.log(type=..., ...)`). `Tracker.totals()` only counts
`llm` and `tool` events, so the new event types don't break any shared metric.

| Event | When | Fields |
|---|---|---|
| `type="node"` | every W4 subgraph node | `workflow="w4_evidence"`, `node`, `start` (epoch s), `end`, `seconds` |
| `type="w4_round"` | after each `decide` | `round`, `claims`, `cited_ok`, `rule_ok`, `llm_checked`, `llm_cached`, `supported`, `rejected_by_code` (dict code → count) |
| `type="w4_final"` | in `finalise` | `kept`, `dropped`, `first_draft_citation_coverage` (share of first-draft claims with valid citations), `first_draft_supported_rate`, `rounds_used`, `skills_dropped` |
| `type="tool"` | validate/render | already done by the tracked helper |

`run()` also returns `w4_stats` (the same numbers as `w4_final`) and `evidence`. `run_all.py`
ignores unknown keys, and the events are saved in the trace automatically.

## 7. Constants (put them in the W4 module, NOT `config.py`)

```python
MAX_GROUNDING_ROUNDS = 2
MAX_SOURCES_PER_CLAIM = 3
SPAN_MAX_CHARS = 300
INFLATION_TERMS = ("led", "lead", "spearhead", "head", "own", "expert", "advanced", "extensive",
                   "proven", "significant", "substantial", "strategic", "senior", "award",
                   "world-class", "cutting-edge", "drastically", "dramatically")
```

Why not `config.py`: `run_all.config_hash()` hashes every upper-case setting in `config.py`, so
adding keys there would invalidate every teammate's cached results and force reruns.
`common/spans.py` and `common/grounding.py` should accept these as parameters with these defaults.

## 8. Errors

| Code | Meaning |
|---|---|
| `W4_DRAFT_FAILED` | Creator output didn't match `GroundedResume` after one retry |
| `W4_FINAL_VALIDATION_FAILED` | The cleaned resume still failed `validate_resume` |
| `W4_NO_SPANS` | `split_spans` returned no citeable spans (e.g. empty text) |

Add these to the `MESSAGES` dict in `app.py` with plain-English explanations.

## 9. Non-goals and fairness rules

- Don't edit shared prompts, `config.py`, the harness, the pipeline, `eval/` or other workflows.
  The only shared edit is adding two optional keys (`evidence`, `w4_stats`) to `ResumeState`, because
  LangGraph drops undeclared keys returned by a node.
- Don't use `eval.metrics.faithfulness` or its NLI model in W4. The evaluator scores every workflow
  with it; using it inside W4 would mean optimising for the test.
- Skills, certifications, employers, titles, dates, degrees and contact details are not cited.
  The shared harness already checks them exactly.
- No retrieval, web search or fine-tuning. W4 tests one technique: forced citation + verification.
