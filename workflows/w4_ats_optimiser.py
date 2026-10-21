"""W4 - ATS OPTIMISER LOOP  (worked example: copy this pattern for your own workflow)

Technique: the agent extracts the JD's keywords itself, checks how many the resume covers,
rewrites to cover the missing ones, and repeats until it is happy or runs out of steps.

What it should improve: keyword coverage.
What it might hurt: faithfulness (it may add skills the candidate doesn't have to hit the score).
Course link: Topic 7 tools + agent loop + step budget.

Note: the agent uses ITS OWN keyword list (like a real ATS optimiser would).
The evaluator uses our HAND-LABELLED list, so W4 can't simply copy the answer key.
"""
import json
import re

from common.agent import Tool, run_agent
from common.llm import chat
from common.prompts import BASE_RULES, task_prompt
from eval.metrics import keyword_coverage

NAME = "w4_ats_optimiser"

SYSTEM = BASE_RULES + """
You work like an ATS (applicant tracking system) optimiser:
1. Call extract_keywords on the job description.
2. Write a draft and call check_keywords with it.
3. If important keywords are missing AND the original resume supports them, revise and check again.
4. When coverage is good, or the remaining keywords are not supported by the original resume, call submit_resume."""


def run(pair, tracker):
    memo = {}

    def extract_keywords(jd_text: str) -> str:
        """Ask the model for the JD's key skills (cached so repeated calls are cheap)."""
        if "kw" not in memo:
            msg = chat([{"role": "user", "content":
                         "List the 10-15 most important skills/keywords in this job description. "
                         "Reply with a JSON array of short strings only.\n\n" + jd_text}], tracker)
            found = re.search(r"\[.*\]", msg.content or "", re.S)
            memo["kw"] = json.loads(found.group(0)) if found else []
        return json.dumps(memo["kw"])

    def check_keywords(resume_md: str) -> str:
        if "kw" not in memo:
            return "Error: call extract_keywords first."
        kw = [{"keyword": k, "synonyms": []} for k in memo["kw"]]
        cov = keyword_coverage(resume_md, kw)
        return json.dumps({"coverage": round(cov["coverage"], 2), "missing": cov["missing"]})

    tools = [
        Tool(extract_keywords, "Extract the key skills from the job description.",
             {"jd_text": {"type": "string", "description": "The full job description"}}),
        Tool(check_keywords, "Check which job keywords a draft resume covers and which are missing.",
             {"resume_md": {"type": "string", "description": "The draft resume in Markdown"}}),
    ]
    resume, hit_budget, finished = run_agent(SYSTEM, task_prompt(pair), tools, tracker)
    return {"resume_md": resume, "hit_budget": hit_budget, "finished": finished}
