"""Run this FIRST (day 1-2). Answers: can our 4B model make valid tool calls at all?

    python smoke_test.py

If valid tool calls < ~80%, fix prompts/tools or try a bigger model before anyone builds a workflow.
"""
import json

from common import llm

import config
from common.agent import Tool, run_agent
from common.llm import Tracker


def add(a: int, b: int) -> int:
    return a + b


def word_count(text: str) -> int:
    return len(text.split())


TASKS = [
    "What is 17 + 25? Use the add tool, then submit your answer with submit_resume.",
    "How many words are in 'Led a team of five analysts'? Use word_count, then submit with submit_resume.",
    "Add 3 and 4, then add 10 to the result. Use the add tool, then submit with submit_resume.",
    "Count the words in 'Built SQL dashboards for sales' using word_count, then submit with submit_resume.",
    "What is 100 + 1? Use add, then submit with submit_resume.",
]

JSON_TASKS = [
    "Save this profile with save_profile, then submit 'done' with submit_resume: Alex Tan, Business Analyst Intern "
    "at Example Logistics from May 2025 to Aug 2025, and Student Assistant at University IT Helpdesk from Jan 2024 to Dec 2024.",
    "Use save_profile to save: Priya Nair, Data Analyst at Northwind (2022-01 to 2024-06). Then submit 'done' with submit_resume.",
    "Save with save_profile: Wei Ling, Software Engineer at Brightline from 2021 to 2023, Intern at Acme in 2020. "
    "Then call submit_resume with 'done'.",
]

if __name__ == "__main__":
    try:
        llm._client.show(config.MODEL)
    except Exception as e:
        raise SystemExit(f"Can't reach model {config.MODEL}. Is Ollama running and the model pulled? ({e})")

    tools = [Tool(add, "Add two integers.", {"a": {"type": "integer"}, "b": {"type": "integer"}}),
             Tool(word_count, "Count words in a text.", {"text": {"type": "string"}})]
    ok = total = finished = 0
    for t in TASKS:
        tr = Tracker()
        _, _, fin = run_agent("You are a careful assistant. Use the tools provided.", t, tools, tr, max_steps=5)
        tc = tr.totals()
        ok, total, finished = ok + tc["tool_calls_ok"], total + tc["tool_calls"], finished + fin
        print(f"- {t[:45]}... tool calls {tc['tool_calls_ok']}/{tc['tool_calls']}, finished={fin}")
    print(f"\nValid tool calls: {ok}/{total} ({100 * ok / max(total, 1):.0f}%), tasks finished: {finished}/{len(TASKS)}")

    # The baseline creator passes a whole resume as a JSON object argument to validate_resume.
    # Check the model can do that before relying on it.
    print("\nJSON-object arguments (needed by the baseline's validate_resume):")
    json_ok = 0
    for i, task in enumerate(JSON_TASKS):
        tr = Tracker()
        got = {}

        def save_profile(profile):
            if isinstance(profile, str):
                profile = json.loads(profile)
            if not (isinstance(profile, dict) and profile.get("name") and isinstance(profile.get("jobs"), list)):
                raise ValueError("profile must be an object with 'name' (string) and 'jobs' (a list of "
                                 "{employer, title, start, end}). Call save_profile again with that shape.")
            got["profile"] = profile
            return "Saved."

        job = {"type": "object", "properties": {k: {"type": "string"} for k in ("employer", "title", "start", "end")},
               "required": ["employer", "title"]}
        tool = Tool(save_profile, "Save a candidate profile.",
                    {"profile": {"type": "object", "description": "The candidate profile",
                                 "properties": {"name": {"type": "string"}, "jobs": {"type": "array", "items": job}},
                                 "required": ["name", "jobs"]}})
        run_agent("You are a careful assistant. Use the tools provided.", task, [tool], tr, max_steps=4)
        passed = bool(got.get("profile"))
        json_ok += passed
        print(f"- task {i + 1}: {'ok' if passed else 'FAILED'}")
    print(f"Saved a correctly shaped JSON object (retries allowed): {json_ok}/{len(JSON_TASKS)}. If this is low, "
          f"use the fallback: the creator drafts with format=GeneratedResume and the harness calls validate/render itself.")
