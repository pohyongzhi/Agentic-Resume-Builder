# Prompt for External Auditor

Copy and paste this prompt to another LLM to audit the W4 implementation:

---

You are auditing the implementation of **W4 (Evidence-Grounded Generation)** for an agentic resume builder. Your task is to verify that the implementation matches the specification exactly.

## Your Mission

Review the W4 implementation and produce a detailed audit report covering:
1. **Compliance**: Does it match the specification?
2. **Correctness**: Does the code implement the required logic?
3. **Completeness**: Are all components present?
4. **Quality**: Is it well-tested and documented?
5. **Issues**: Any bugs, gaps, or deviations from spec?

## Context

This is a research project implementing 7 resume generation workflows (W1-W7). W4 is variant 4, which uses **evidence-grounded generation**: every claim in the generated resume must cite source spans from the original resume and pass verification.

**Key constraints from spec:**
- Uses `qwen3.5:4b-q4_K_M` model for all LLM calls
- Shares the same pipeline as other workflows (W2, W3)
- Must NOT modify shared prompts (only append W4-specific rules)
- Citation validation happens in code (5 rules), then 3 deterministic rules, then LLM verification
- Maximum 2 revision rounds (MAX_GROUNDING_ROUNDS=2)
- Unsupported claims are dropped, not kept

## Files to Review

### 📋 Start Here
1. **`docs/w4-evidence-grounded/01-spec.md`** - Read this first to understand requirements
2. **`docs/w4-evidence-grounded/AUDIT_GUIDE.md`** - Detailed verification checklist
3. **`docs/w4-evidence-grounded/02-implementation-plan.md`** - Architecture and phases

### 🔍 Core Implementation (verify against spec)
1. **`common/schemas.py`** - Search for "W4" schemas (Claim, GroundedResume, etc.)
   - Check: §3.2 in spec
   - Verify: Claim accepts bare strings, normalizes sources
   
2. **`common/spans.py`** - Span splitting logic
   - Check: §4.1 in spec (10 rules)
   - Verify: PDF artifact handling, metadata assignment
   
3. **`common/grounding.py`** - Citation validation and verification
   - Check: §4.3 (5 citation rules), §4.4 (3 deterministic rules), §4.5 (LLM verifier)
   - Verify: check_citations(), rule_check(), llm_verify()
   
4. **`workflows/w4_evidence.py`** - Main workflow graph
   - Check: §4 (complete workflow)
   - Verify: Graph structure, revision loop, evidence generation

### 🧪 Tests (verify coverage)
1. **`tests/test_w4_schemas.py`** - 9 tests for schemas
2. **`tests/test_w4_spans.py`** - 19 tests for span splitting
3. **`tests/test_w4_grounding.py`** - 34 tests for grounding logic
4. **`tests/test_w4_workflow.py`** - 11 end-to-end workflow tests

### 🔗 Integration
1. **`app.py`** - Web UI integration (search for "w4" or "W4")
2. **`workflows/__init__.py`** - Workflow registration
3. **`common/state.py`** - State additions (evidence, w4_stats fields)
4. **`HOW_TO_RUN.md`** - User documentation (search for "W4")

## Specific Things to Check

### ✅ Critical Requirements

1. **Model Usage**
   ```bash
   # All LLM calls must use common.llm.chat()
   grep -n "from common.llm import chat" workflows/w4_evidence.py common/grounding.py
   
   # Model should be qwen3.5:4b-q4_K_M from config.py
   grep "^MODEL" config.py
   ```

2. **Shared Prompts Not Modified**
   ```bash
   # W4 should ONLY append W4_CITATION_RULES, never edit STRUCTURED_CREATOR_SYSTEM
   grep -A 5 "W4_CITATION_RULES" workflows/w4_evidence.py
   grep -B 2 -A 2 "STRUCTURED_CREATOR_SYSTEM" workflows/w4_evidence.py
   ```

3. **Citation Rules (5 rules from spec §4.3)**
   - [ ] no_citation: sources cannot be empty
   - [ ] bad_citation: span IDs must exist and be citeable
   - [ ] too_many_sources: max 3 sources per claim
   - [ ] wrong_entry: can't cite spans from different job/school
   - [ ] wrong_section: education/projects can't cite from other sections

4. **Deterministic Rules (3 rules from spec §4.4)**
   - [ ] new_number: numbers in claim must appear in cited text
   - [ ] new_skill: skills from JD must appear in cited text
   - [ ] inflated: seniority terms must appear in cited text

5. **Revision Loop**
   - [ ] MAX_GROUNDING_ROUNDS = 2
   - [ ] Round 0: initial draft
   - [ ] Round 1-2: revise with feedback
   - [ ] After round 2: drop unsupported claims, don't keep trying

6. **Evidence Report**
   - [ ] Contains ALL claims (both kept and dropped)
   - [ ] Each claim has: claim_id, section, entry, claim, sources, status, problems, rounds_to_pass
   - [ ] Status is either "kept" or "dropped"
   - [ ] w4_stats includes: kept, dropped, first_draft_citation_coverage, first_draft_supported_rate, rounds_used, skills_dropped (spec §6)

### 🧪 Test Verification

Run these commands (in your mind - analyze the test code):

```python
# All tests should pass
# Check: Do test files have proper assertions?
# Look at: tests/test_w4_workflow.py

# Key scenarios from spec §6 (test plan):
# W1: Good draft, no revisions needed
# W2: Bad draft, revised once, becomes good
# W3: Bad draft, hits budget (2 rounds), drops claims
# W4: Missing citations are dropped
# W5: LLM verifier rejects, then approves after revision
```

### 🔴 Red Flags to Watch For

**Major Issues:**
- ❌ Modifying shared prompts (STRUCTURED_CREATOR_SYSTEM, creator_prompt)
- ❌ Using a different model than qwen3.5:4b-q4_K_M
- ❌ Infinite revision loops (not respecting MAX_GROUNDING_ROUNDS)
- ❌ Keeping unsupported claims instead of dropping them
- ❌ Tests that call real LLM (should use FakeChat)

**Minor Issues:**
- ⚠️ Missing docstrings
- ⚠️ Inconsistent error messages
- ⚠️ Missing test coverage for edge cases
- ⚠️ Performance issues (>1 minute for small resume)

### 📊 Deliverable: Audit Report

Provide a structured report with:

```markdown
# W4 Implementation Audit Report

## Executive Summary
[Pass/Fail with key findings]

## Specification Compliance
- [ ] Data models match spec §3
- [ ] Span splitting implements 10 rules (§4.1)
- [ ] Citation validation implements 5 rules (§4.3)
- [ ] Deterministic checks implement 3 rules (§4.4)
- [ ] LLM verification matches spec (§4.5)
- [ ] Workflow graph correct (§4)
- [ ] Evidence report format correct (§5)

## Code Quality
- Test coverage: [X/123 tests passing]
- Documentation: [Complete/Incomplete]
- Error handling: [Adequate/Inadequate]
- Performance: [Within/Outside expected range]

## Issues Found
### Critical
1. [Issue description]
   - Location: [file:line]
   - Expected: [from spec]
   - Actual: [in code]
   - Fix: [recommendation]

### Minor
[List minor issues]

## Recommendations
[Suggestions for improvement]

## Overall Assessment
[Pass/Conditional Pass/Fail with justification]
```

## Audit Methodology

1. **Read spec first** (`01-spec.md`) - understand what W4 should do
2. **Review implementation** - check each file against spec sections
3. **Analyze tests** - verify they cover all scenarios from `03-test-plan.md`
4. **Check integration** - ensure W4 works with existing system
5. **Look for red flags** - deviations, bugs, missing pieces
6. **Cross-reference** - use the mapping table in `AUDIT_GUIDE.md`

## Key Questions to Answer

1. **Does it work?** Can you trace the flow from resume input to evidence report output?
2. **Is it correct?** Does each function implement its spec requirement exactly?
3. **Is it complete?** Are all 10 span rules, 5 citation rules, 3 deterministic rules present?
4. **Is it tested?** Do tests cover the scenarios in `03-test-plan.md`?
5. **Is it integrated?** Can you select W4 in the web UI and see the evidence report?

## Example: How to Verify One Component

**Verifying Citation Rule "wrong_entry"** (spec §4.3):

1. Read spec: "A bullet under Job A may not cite a span from Job B"
2. Find code: `common/grounding.py` function `check_citations()`
3. Look for logic:
   ```python
   if section == "experience" and entry_index is not None:
       draft_entry = resume.experience[entry_index]
       matched_index = match_entry("experience", draft_entry, candidate)
       if matched_index is not None:
           if span.section == "experience" and span.entry != matched_index:
               issues.append(f"wrong_entry: {source_id}")
   ```
4. Find test: `tests/test_w4_grounding.py::test_check_citations_wrong_entry`
5. Verify test creates scenario where Job A cites Job B's span
6. Check test expects "wrong_entry" in problems list
7. ✅ Confirmed: Rule is implemented and tested correctly

Do this for ALL spec requirements.

## Time Estimate

- Quick review (checklist only): 30 minutes
- Thorough audit (all code): 2-3 hours
- Deep audit (run tests, trace execution): 4-6 hours

## Questions?

If anything is unclear:
1. Check `AUDIT_GUIDE.md` for detailed verification steps
2. Read the relevant spec section in `01-spec.md`
3. Look at the implementation plan in `02-implementation-plan.md`
4. Review test cases in `03-test-plan.md`

---

**Start your audit now. Be thorough, objective, and detail-oriented. Good luck! 🔍**
