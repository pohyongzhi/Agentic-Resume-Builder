"""Historical design notes retained for provenance.

The stable V1 planned-workflow modules now live in their own files. This note is
not registered and the legacy W4 ATS example is not one of the six workflows.

Starting points for the other four owners originally copied the W4 pattern:
    1. Write a SYSTEM prompt = BASE_RULES + how your technique works.
    2. Define your tools as plain Python functions and wrap them with Tool(...).
    3. Return run_agent(...) results in the same dict shape.
Move your workflow into its own file (e.g. workflows/w1_research.py) and register it in workflows/__init__.py.
"""

# W1 RESEARCH  (Topic 7 tools + Topic 8 RAG)
#   Tools: web_search(query) -> top results about the company/role (e.g. `pip install ddgs`)
#          search_guide(query) -> top-3 chunks from the Harvard resume guide, embedded with common.llm.embed
#   Watch: web pages can contain junk or hidden instructions; only use them for context, never as candidate facts.

# W2 RECRUITER ROLE-PLAY  (Topic 7 multi-agent)
#   Two prompts: WRITER (BASE_RULES) and RECRUITER ("You are the hiring manager for this JD. Would you shortlist
#   this resume? List the 3 biggest gaps or weaknesses.").
#   Graph: write -> recruiter_review -> rewrite -> recruiter_review -> rewrite -> END  (max 2 rounds)
#   The recruiter can only comment; it never edits the resume itself.

# W3 CRITIC INSIDE  (Topic 7 reflection, Topic 9 hallucination)
#   Tools: critique(resume_md) -> scores + fixes for clarity/impact/relevance
#          verify_claims(resume_md) -> lines not supported by the original resume (can reuse eval.metrics.faithfulness)
#   Stop after 2 rewrites or when the critique has no major issues.

# W5 PLAN FIRST  (Topic 7 §4 plan-and-execute)
#   Node 1 'plan': model writes JSON {must_haves:[], evidence:{requirement: resume_line}, lead_with:[], drop:[]}
#   Node 2 'write': writes the resume following the plan (plan is in the prompt).
#   Optional node 'replan' if the draft ignores part of the plan.
