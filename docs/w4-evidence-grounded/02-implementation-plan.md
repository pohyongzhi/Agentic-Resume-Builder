# W4 implementation plan

Work phase by phase. Each phase ends with `pytest -q` passing (old and new tests) before the next
one starts. The spec (`01-spec.md`) is the source of truth for behaviour; this file says **where**
things go and **when** a phase is done.

## Files touched

| File | Action |
|---|---|
| `common/schemas.py` | **add** `Claim`, `GroundedExperience`, `GroundedEducation`, `GroundedProject`, `GroundedResume`, `ClaimVerdict`, `VerifyBatch` (append only, don't edit existing classes) |
| `common/spans.py` | **new** |
| `common/grounding.py` | **new** |
| `workflows/w4_evidence.py` | **new** |
| `workflows/__init__.py` | **add** W4 to `WORKFLOWS` |
| `common/state.py` | **add** two optional keys `evidence`, `w4_stats` to `ResumeState` (nothing else) |
| `app.py` | **add** a W4 dropdown choice, a W4 branch in `_run_selected_workflow`, W4 error messages, evidence report output |
| `HOW_TO_RUN.md` | **add** one example command for W4 |
| `tests/test_w4_schemas.py`, `tests/test_w4_spans.py`, `tests/test_w4_grounding.py`, `tests/test_w4_workflow.py` | **new** |

Do **not** touch (except the two-key addition to `common/state.py` above): `config.py`, `common/prompts.py`, `common/harness.py`, `common/pipeline.py`,
`common/llm.py`, `common/extraction.py`, `eval/`, `run_all.py`, other workflows.

---

## Phase 0: orientation (no code)

1. Run `pytest -q` and note that everything passes before you start.
2. Read the files listed in the README prompt. Confirm these exist with these signatures:
   - `common.llm.chat(messages, tracker, tools=None, think=None, format=None)`
   - `common.schemas.Schema`, `llm_schema(model, exclude=())`, `GeneratedResume`, `Contact`
   - `common.harness.ToolHarness(candidate, resume_path=None, output_dir=None, max_calls=None, run_id=None)`
     with `.execute(name, args) -> (ok, result_json_str, stop)` and `.outcome()`
   - `common.harness._norm(text)`, `common.harness._in_text(term, raw_norm)`
   - `common.pipeline.run_pipeline(creator_factory, tracker, resume_path, job_url="", pasted_jd="", run_id=None)`
   - `common.prompts.STRUCTURED_CREATOR_SYSTEM`, `creator_prompt(candidate, job)`
3. If any differ, stop and report.

---

## Phase 1: schemas

**Add to `common/schemas.py`** the models from spec §3.2, including the tolerance rules:

- `Claim` `model_validator(mode="before")`: if `data` is a `str` → `{"text": data, "sources": []}`.
  Normalise `sources`: if it's a string, extract IDs with regex `[Ss]\s*(\d+)`. If it's a list,
  extract IDs from each item the same way. Output a de-duplicated list `["S3", "S4"]` in order.
  Drop items with no ID.
- A shared helper `_claims_before(v)` used by `field_validator(mode="before")` on
  `GroundedExperience.bullets`, `GroundedEducation.details`, `GroundedProject.bullets`,
  `GroundedResume.summary`: `None` → `[]`, `str` → `[{"text": v, "sources": []}]`, a list →
  map each `str` item to `{"text": item, "sources": []}`.

`llm_schema(GroundedResume)` must still produce a valid JSON schema (it will contain `$defs`;
Ollama `format=` accepts that, as W2 already relies on it for `GeneratedResume`).

**Done when:** `tests/test_w4_schemas.py` (see test plan §1) passes.

---

## Phase 2: spans (`common/spans.py`)

```python
from dataclasses import dataclass, asdict

SPAN_MAX_CHARS = 300

@dataclass(frozen=True)
class Span:
    id: str; text: str; start: int; end: int
    section: str; entry: int | None; kind: str
    def to_dict(self) -> dict: return asdict(self)

def split_spans(raw_text: str, candidate: dict, max_chars: int = SPAN_MAX_CHARS) -> list[Span]: ...
def citeable(spans: list[Span]) -> list[Span]:          # kind == "content" and section != "header"
def spans_by_id(spans: list[Span]) -> dict[str, Span]:
def render_spans_for_prompt(spans: list[Span], candidate: dict) -> str:   # format in spec §4.2
```

Implement the 10 rules in spec §4.1. Suggested internals: a list of `(line_text, start, end)`
tuples from `re.finditer(r"[^\n]*\n?", raw_text)`, and small helpers `_is_bullet(line)`,
`_heading_section(line) -> str | None`, `_has_date_range(line)`,
`_match_entry(line, section, candidate) -> int | None`, `_sentences(text, start) -> list[(text, s, e)]`.

Date-range regex suggestion (case-insensitive):
`(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\b(19|20)\d{2}\b\s*[-–—]\s*((19|20)\d{2}|present|now|current)|expected\s+(19|20)\d{2}`

`render_spans_for_prompt` groups citeable spans under headers. For experience:
`== EXPERIENCE: {title}, {employer} ({start} - {end}) ==` from `candidate["experience"][entry]`.
Similarly for education and projects. Summary spans go under `== SUMMARY ==`, anything else under
`== OTHER ==`. If `entry` is `None` in experience, use `== EXPERIENCE (unassigned) ==`.

**Done when:** `tests/test_w4_spans.py` (test plan §2) passes, including the PDF-wrapping fixture.

---

## Phase 3: deterministic checks (`common/grounding.py`, part 1)

```python
from common.harness import _norm, _in_text
from common.schemas import GroundedResume, Claim

MAX_SOURCES_PER_CLAIM = 3
INFLATION_TERMS = (...)   # spec §7

def iter_claims(resume: GroundedResume) -> list[dict]:
    """[{"claim_id", "section", "entry_index", "entry_label", "claim": Claim}] in document order (spec §3.3)."""

def match_entry(section: str, draft_entry, candidate: dict) -> int | None:
    """Index of the candidate entry this draft entry corresponds to (normalised employer+title,
    then employer only; institution+degree then institution; project name)."""

def check_citations(resume: GroundedResume, spans: list, candidate: dict,
                    max_sources: int = MAX_SOURCES_PER_CLAIM) -> dict[str, list[str]]:
    """claim_id -> problem strings (spec §4.3). Every claim_id from iter_claims is a key."""

def skill_terms(job_text: str) -> set[str]:
    """Skill-like terms from the JD (spec §4.4 new_skill)."""

def rule_check(claim_text: str, cited_text: str, job_text: str,
               inflation_terms=INFLATION_TERMS) -> list[str]:
    """Problem strings like "new_number: 40%", "new_skill: Tableau", "inflated: led"."""

def drop_unsupported(resume: GroundedResume, problems: dict[str, list[str]]) -> GroundedResume:
def filter_skills(items: list[str], parsed: list[str], raw_text: str) -> tuple[list[str], list[str]]:
def to_generated(resume: GroundedResume) -> dict:   # GeneratedResume-shaped dict, citations removed
```

Problem strings are always `"<code>"` or `"<code>: <detail>"`, where code is one of
`no_citation, bad_citation, too_many_sources, wrong_entry, wrong_section, new_number, new_skill,
inflated, not_entailed`.

**Done when:** `tests/test_w4_grounding.py` sections A–D (test plan §3) pass.

---

## Phase 4: LLM verifier (`common/grounding.py`, part 2)

```python
from common.llm import chat
from common.schemas import VerifyBatch, llm_schema

VERIFIER_SYSTEM = """..."""   # spec §4.5, verbatim

def cache_key(claim: Claim) -> str:
    return json.dumps([claim.text.strip(), sorted(claim.sources)])

def llm_verify(items: list[dict], tracker, cache: dict) -> dict[str, dict]:
    """items: [{"claim_id", "claim": Claim, "cited_text"}]. Returns claim_id -> {"supported": bool, "reason": str}.
    One chat() call for all uncached items (format=llm_schema(VerifyBatch)); no call if all are cached.
    Missing or malformed verdicts -> {"supported": False, "reason": "verifier gave no verdict"}.
    If the reply isn't valid JSON or the schema check fails, treat all uncached items as unsupported
    with reason "verifier output invalid" (do not crash)."""
```

The user message lists the items as JSON: `[{"claim_id": ..., "claim": ..., "cited_text": ...}]`.
Tests monkeypatch `common.grounding.chat`.

**Done when:** test plan §3 section E passes.

---

## Phase 5: the workflow (`workflows/w4_evidence.py`)

Mirror the structure of `workflows/w3_critic.py`.

```python
NAME = "w4_evidence"
MAX_GROUNDING_ROUNDS = 2

W4_CITATION_RULES = """..."""         # spec §4.2, verbatim
SYSTEM = STRUCTURED_CREATOR_SYSTEM + "\n\n" + W4_CITATION_RULES

class EvidenceState(TypedDict, total=False):
    candidate_data: dict
    job_data: dict
    spans: list[dict]              # Span.to_dict()
    draft: dict                    # GroundedResume.model_dump()
    round: int
    problems: dict                 # claim_id -> list[str]
    verdict_cache: dict            # cache_key -> verdict dict
    first_pass: dict               # claim_id -> round in which it first passed (for evidence rows)
    first_draft_stats: dict
    resume_md: str
    output_path: str
    generated_resume: dict
    evidence: list
    w4_stats: dict
    hit_budget: bool
    errors: list[str]

def draft_prompt(candidate, job, spans) -> str: ...
def revise_prompt(candidate, job, spans, draft, problems, iter_rows) -> str: ...   # spec §4.7
def build_evidence_graph(tracker, resume_path=None, run_id=None): ...
def make_creator(tracker, resume_path=None, run_id=None): ...       # returns the pipeline node
def generate(tracker, resume_path, job_url="", pasted_jd="", run_id=None): ...
def run(pair, tracker) -> dict: ...
```

Graph nodes, in order (spec §2 diagram):
`split_spans → draft → check (citations + rules) → llm_verify → decide → (revise → check …) | finalise → END`.

It's fine to put citation and rule checks in one `check` node. `decide` is a conditional edge
function, not a node.

Details:

- **Node timing:** wrap every node with a small decorator `_timed(tracker, name)` that logs
  `type="node", workflow=NAME, node=name, start, end, seconds` (spec §6).
- **split_spans:** if no citeable spans, return `errors=["W4_NO_SPANS: ..."]` and route to END.
- **draft / revise:** `chat(messages, tracker, format=llm_schema(GroundedResume))`, parse with
  `GroundedResume.model_validate_json`. On `ValidationError`/`JSONDecodeError`, retry once with the
  error text appended as a user message. Second failure: in draft → `W4_DRAFT_FAILED`, route to END;
  in revise → keep the previous draft and route to `finalise`.
- **check:** `check_citations`, then `rule_check` on claims with no citation problems (`cited_text`
  = cited span texts joined by `" "`, `job_text` =
  `job_data["source_text"]` plus responsibilities and qualifications joined).
- **llm_verify node:** build items for claims with no problems so far and call `llm_verify`. Add
  `"not_entailed: <reason>"` for unsupported ones. Record the first round each claim passed
  everything. Log the `w4_round` event. Store `first_draft_stats` when `round == 0`.
- **decide:** spec §4.6.
- **finalise:** spec §4.8. Use the harness through a `_tracked_execute` helper (copy W3's). Build the
  evidence report (spec §5) and `w4_stats`, and log the `w4_final` event.
- **make_creator** wraps `graph.invoke(...)` in `try/except Exception` and returns
  `{"errors": [f"W4_FAILED: {type(e).__name__}: {e}"]}` on unexpected errors, like W3. It returns only:
  `resume_md, output_path, generated_resume, hit_budget, errors, evidence, w4_stats`.
- **run(pair, tracker)** returns the same keys as W3's `run`, plus `evidence` and `w4_stats`. Use
  `pair.get("job_url") or pair.get("jd_url", "")` (`common/data.py` uses the key `jd_url`).

**Register:** add `w4_evidence` to `workflows/__init__.py` (import + dict entry, keeping the existing order).

**Done when:** `tests/test_w4_workflow.py` (test plan §4) passes and `pytest -q` is fully green.

**Shared state (one small, additive edit):** LangGraph silently drops keys a node returns if they
aren't declared in the graph's state (verified with LangGraph 1.x). Add two optional keys to
`ResumeState` in `common/state.py`:
```python
    evidence: list          # W4 evidence report (unused by other workflows)
    w4_stats: dict          # W4 summary numbers (unused by other workflows)
```
This is the only change allowed in `common/state.py`. `ResumeState` is `total=False`, so other
workflows are unaffected. Write a test proving `run()` returns `evidence` and `w4_stats`.

---

## Phase 6: app integration (`app.py`)

1. Add `("W4 - Evidence-grounded: every bullet cites the original resume", "w4_evidence")` to
   `WORKFLOW_CHOICES`.
2. In `_run_selected_workflow`, add a branch like W3's, calling `w4_evidence.generate(...)` and
   returning its state (including `evidence`).
3. Add `W4_DRAFT_FAILED`, `W4_FINAL_VALIDATION_FAILED`, `W4_NO_SPANS`, `W4_FAILED` to `MESSAGES`.
4. Show the evidence report without restructuring the UI: append a Markdown section to the
   existing stats output:
   ```
   ### Evidence report (W4)
   Kept 14 / 16 claims · 2 dropped · passed first time: 11
   | | Claim | Cited source |
   |---|---|---|
   | ✅ | Built weekly Power BI sales dashboards… | S6: "Built weekly sales dashboards in Excel and Power BI…" |
   | ❌ | Led a team to cut data entry by 40% | S8: "Mapped the order-to-delivery process…" — inflated: led; new_number: 40% |
   ```
   Truncate long texts to about 90 characters. Only show it when the selected workflow is W4.
5. Run `tests/test_app.py`. It must still pass. Add a test only if it's simple to do with the
   existing fakes.

**Done when:** `pytest -q` is green. Then try it manually: `python app.py`, pick W4, upload a resume,
paste a JD.

---

## Phase 7: real-model check (manual, with Ollama running)

```bash
python smoke_test.py
python run_all.py --workflows w4_evidence --limit 1 --repeats 1
python summarise.py
```

Check in `results/results.jsonl` (the last W4 line → `trace`):

- `w4_round` events exist, and `first_draft_citation_coverage` is a number. If it's below about
  0.5, the 4B model isn't citing well: improve the span prompt layout (§4.2) before tuning anything else.
- The rendered `resume_md` has no `[S…]` markers.
- The `w4_final` dropped count looks plausible. Spot-check 3 dropped claims by hand.

Add to `HOW_TO_RUN.md` under "Useful smaller runs":
`python run_all.py --workflows w4_evidence --limit 1 --repeats 1`.
