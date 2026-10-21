"""Run this FIRST (day 1-2). Answers: can our 4B model make valid tool calls at all?

    python smoke_test.py

If valid tool calls < ~80%, fix prompts/tools or try a bigger model before anyone builds a workflow.
"""
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
