# W4 Audit Guide

This guide helps auditors verify that the W4 (evidence-grounded generation) implementation matches the specification.

## Quick Start

1. **Read the specification**: Start with `01-spec.md` to understand what W4 should do
2. **Check implementation**: Review `02-implementation-plan.md` for the architecture
3. **Run tests**: All 123 tests should pass (`pytest -q`)
4. **Verify behavior**: Test against `03-test-plan.md` scenarios

## Files to Audit

### Core Implementation

| File | Lines | Purpose | What to Check |
|------|-------|---------|---------------|
| `common/schemas.py` | ~100 | W4 Pydantic models | Claim, GroundedResume, ClaimVerdict match spec §3.2 |
| `common/spans.py` | ~200 | Split resume into citeable chunks | 10 rules from spec §4.1 |
| `common/grounding.py` | ~550 | Citation + verification logic | 5 citation rules + 3 deterministic rules + LLM verifier |
| `workflows/w4_evidence.py` | ~560 | Main W4 LangGraph | Graph: split→draft→check→verify→revise loop→finalise |

### Tests (123 total)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_w4_schemas.py` | 9 | Schema validation, small-model tolerance |
| `tests/test_w4_spans.py` | 19 | Span splitting rules, PDF artifacts |
| `tests/test_w4_grounding.py` | 34 | Citation checks, rule checks, LLM verifier |
| `tests/test_w4_workflow.py` | 11 | End-to-end scenarios from test plan |

### Integration

| File | Changes | Purpose |
|------|---------|---------|
| `common/state.py` | +2 fields | Added `evidence` and `w4_stats` |
| `workflows/__init__.py` | +2 lines | Registered W4 |
| `app.py` | +80 lines | Web UI integration + evidence report |

## Verification Checklist

### 1. Specification Compliance

- [ ] **Model**: Uses `qwen3.5:4b-q4_K_M` (check `config.py` and all `chat()` calls)
- [ ] **Shared pipeline**: W4 uses `common.pipeline.run_pipeline()` like W2/W3
- [ ] **No modification**: Doesn't change shared prompts (only appends W4_CITATION_RULES)
- [ ] **Creator node**: Returns a node compatible with the shared pipeline

### 2. Data Model (spec §3)

```bash
# Check schemas exist and validate
pytest tests/test_w4_schemas.py -v
```

- [ ] `Claim` has `text` and `sources` fields
- [ ] `Claim` accepts bare strings and normalizes sources
- [ ] `GroundedResume` has list[Claim] for summary/bullets/details
- [ ] `ClaimVerdict` and `VerifyBatch` for LLM verification

### 3. Span Splitting (spec §4.1)

```bash
# Check 10 rules are implemented
pytest tests/test_w4_spans.py::test_split_rules -v

# Verify PDF artifact handling
pytest tests/test_w4_spans.py::test_pdf_wrapping -v
```

- [ ] Rule 1-2: Section headings (experience/education/projects/skills)
- [ ] Rule 3: Entry headers (job title/company)
- [ ] Rule 4-6: Bullet detection (-, *, •, wrapped bullets)
- [ ] Rule 7: Hyphen break handling ("Excel -based" → "Excel-based")
- [ ] Rule 8-9: Sentence splitting (>300 chars)
- [ ] Rule 10: Metadata assignment (section, entry, kind)

### 4. Citation Validation (spec §4.3)

```bash
# Check 5 citation rules
pytest tests/test_w4_grounding.py -k citation -v
```

- [ ] `no_citation`: Empty sources rejected
- [ ] `bad_citation`: Invalid span IDs rejected
- [ ] `too_many_sources`: Max 3 sources per claim
- [ ] `wrong_entry`: Can't cite other job's spans
- [ ] `wrong_section`: Education/projects can't cite other sections

### 5. Rule Checks (spec §4.4)

```bash
# Check 3 deterministic rules
pytest tests/test_w4_grounding.py -k rule_check -v
```

- [ ] `new_number`: Numbers in claim must be in cited text
- [ ] `new_skill`: Skills from JD must be in cited text
- [ ] `inflated`: Seniority terms must be in cited text

### 6. LLM Verification (spec §4.5)

```bash
# Check verifier logic
pytest tests/test_w4_grounding.py -k llm_verify -v
```

- [ ] Batches claims for efficiency
- [ ] Caches verdicts by (text, sources)
- [ ] Returns `{claim_id: {supported, reason}}`
- [ ] Uses VERIFIER_SYSTEM prompt from spec

### 7. Workflow Graph (spec §4)

```bash
# Check workflow scenarios
pytest tests/test_w4_workflow.py -v
```

- [ ] W1: Good draft passes first time
- [ ] W2: Bad draft revised to good
- [ ] W3: Bad draft hits MAX_GROUNDING_ROUNDS (2)
- [ ] W4: No citation claims dropped
- [ ] W5: Verifier rejection triggers revision
- [ ] W6-W9: Error handling
- [ ] W10: Registered in WORKFLOWS
- [ ] W11: End-to-end integration

### 8. Evidence Report (spec §5)

```bash
# Run and check report format
python -c "
from workflows.w4_evidence import generate
from common.llm import Tracker
state = generate(Tracker(), 'fixtures/alex-markdown.md', pasted_jd='Data Analyst')
print(len(state['evidence']), 'claims in report')
print(state['w4_stats'])
"
```

- [ ] Each claim has: claim_id, section, entry, claim, sources, status, problems, rounds_to_pass
- [ ] Status is "kept" or "dropped"
- [ ] w4_stats has: kept, dropped, first_draft_citation_coverage, first_draft_supported_rate, rounds_used, skills_dropped (spec §6)

### 9. Integration

```bash
# Check app integration
pytest tests/test_app.py -v
```

- [ ] W4 appears in WORKFLOW_CHOICES
- [ ] Error messages added (W4_DRAFT_FAILED, etc.)
- [ ] Evidence report renders as Markdown table
- [ ] HOW_TO_RUN.md documents W4

## Common Issues to Check

### 1. Citation Rules Enforcement

```python
# Should reject: experience bullet citing education span
draft = {
    "experience": [{"bullets": [{"text": "...", "sources": ["S15"]}]}],
    # If S15 is from education section
}
```

### 2. Small-Model Tolerance

```python
# Should accept all these as Claims:
"Built dashboards"  # bare string
{"text": "Built dashboards"}  # missing sources
{"text": "Built dashboards", "sources": "S3, S4"}  # string sources
{"text": "Built dashboards", "sources": ["[s3]", "S 4"]}  # denormalized
```

### 3. Revision Loop

```python
# Should revise up to MAX_GROUNDING_ROUNDS=2
# Round 0: draft
# Round 1: revise (if problems found)
# Round 2: give up, drop unsupported claims
```

### 4. Evidence Completeness

```python
# Evidence should include ALL claims, even kept ones
assert len(evidence) == total_claims_in_draft
assert all(e['status'] in ('kept', 'dropped') for e in evidence)
```

## Performance Benchmarks

Run with real LLM to check performance:

```bash
# Should complete in reasonable time
time python -c "
from workflows.w4_evidence import generate
from common.llm import Tracker
state = generate(Tracker(), 'fixtures/alex-markdown.md', pasted_jd='Data Analyst')
print(f\"Rounds: {state['w4_stats']['rounds_used']}\")
"
```

Expected:
- **Tokens**: ~3-5k total (varies by resume size)
- **Time**: <30s for small resume
- **Rounds**: Usually 0-1 (occasionally 2 if model struggles)

## Red Flags

Things that indicate incorrect implementation:

❌ **Changes shared prompts**: W4 should only append, never modify `STRUCTURED_CREATOR_SYSTEM`
❌ **Different model**: All workflows use `qwen3.5:4b-q4_K_M`
❌ **Kept unsupported claims**: Claims with problems should be dropped
❌ **Missing citations**: Final resume shouldn't have [S3] markers (stripped by `to_generated()`)
❌ **Infinite loops**: Must respect MAX_GROUNDING_ROUNDS=2
❌ **Tests using real LLM**: All tests should use FakeChat, not call Ollama

## Quick Audit Commands

```bash
# 1. All tests pass
pytest -q  # Should show "123 passed"

# 2. No shared prompt modification
grep -n "STRUCTURED_CREATOR_SYSTEM" common/prompts.py workflows/w4_evidence.py
# Should only append W4_CITATION_RULES, not edit

# 3. Model consistency
grep -r "MODEL\|chat(" . --include="*.py" | grep -v test | grep -v "#"

# 4. Evidence report format
pytest tests/test_w4_workflow.py::test_w1_good_draft_first_time -xvs
# Check evidence has all required fields

# 5. Integration
grep -n "w4_evidence" app.py workflows/__init__.py
# Should be registered and integrated
```

## Documentation Cross-Reference

| Spec Section | Implementation | Tests |
|--------------|----------------|-------|
| §3.1 Span | `common/spans.py:Span` | `test_w4_spans.py` |
| §3.2 Claim | `common/schemas.py:Claim` | `test_w4_schemas.py` |
| §3.3 Claim IDs | `common/grounding.py:iter_claims()` | `test_w4_grounding.py:test_iter_claims()` |
| §4.1 Split | `split_spans()` | `test_split_rules()` |
| §4.2 Draft | `draft_node()` | `test_w1_good_draft_first_time()` |
| §4.3 Citations | `check_citations()` | `test_check_citations_*()` |
| §4.4 Rules | `rule_check()` | `test_rule_check_*()` |
| §4.5 Verify | `llm_verify()` | `test_llm_verify_*()` |
| §4.6 Revise | `revise_node()` | `test_w2_bad_then_good_on_revise()` |
| §4.7 Finalise | `finalise_node()` | All workflow tests |
| §5 Evidence | `finalise_node()` | `test_w1_*` checks evidence |

## Sign-off Checklist

After completing the audit:

- [ ] All 123 tests pass
- [ ] Specification requirements met (§1-§5)
- [ ] No shared code modified
- [ ] Integration works (app.py + workflows registry)
- [ ] Documentation complete (HOW_TO_RUN.md)
- [ ] Performance acceptable (<30s for small resume)
- [ ] Evidence report format correct

## Questions to Ask

1. **Does every claim cite sources?** Check a generated resume
2. **Are citations validated?** Run test_w4_grounding.py
3. **Does revision work?** Run test_w2_bad_then_good_on_revise
4. **Are bad claims dropped?** Check evidence report has "dropped" status
5. **Is caching working?** Check unchanged claims skip LLM in round 2
