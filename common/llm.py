"""Thin wrapper around Ollama that logs every call (tokens, time, tool calls) into a Tracker.

Every workflow MUST call the model through `chat()` so cost is measured the same way for everyone.
"""
import time
import ollama

import config
from common.prompts import clean_model_output

_client = ollama.Client()  # talks to the local Ollama server (default http://localhost:11434)


class Tracker:
    """Collects one row per model call and per tool call. Becomes the 'trace' in the results."""

    def __init__(self):
        self.events = []

    def log(self, **event):
        event["step"] = len(self.events) + 1
        self.events.append(event)

    def totals(self):
        llm = [e for e in self.events if e["type"] == "llm"]
        tools = [e for e in self.events if e["type"] == "tool"]
        return {
            "llm_calls": len(llm),
            "input_tokens": sum(e["input_tokens"] for e in llm),
            "output_tokens": sum(e["output_tokens"] for e in llm),   # includes thinking tokens if THINK=True
            "thinking_chars": sum(e["thinking_chars"] for e in llm),
            "seconds": round(sum(e["seconds"] for e in llm) + sum(e.get("seconds", 0) for e in tools), 2),
            "tool_calls": len(tools),
            "tool_calls_ok": sum(1 for e in tools if e["ok"]),
            "tool_calls_repeated": _count_repeats(tools),
        }


def _count_repeats(tool_events):
    seen, repeats = set(), 0
    for e in tool_events:
        key = (e["name"], str(e.get("args")))
        repeats += key in seen
        seen.add(key)
    return repeats


def chat(messages, tracker, tools=None, think=None, model=None):
    """Call the writer model once. Returns the ollama Message (has .content, .tool_calls, .thinking)."""
    selected_model = model or config.BASE_MODEL
    start = time.time()
    resp = _client.chat(
        model=selected_model,
        messages=messages,
        tools=tools,
        think=config.THINK if think is None else think,
        options={
            "temperature": config.TEMPERATURE,
            "num_ctx": config.NUM_CTX,
            "num_predict": config.NUM_PREDICT,
        },
    )
    msg = resp.message
    cleaned = clean_model_output(msg.content)
    if cleaned != msg.content:
        try:
            msg.content = cleaned
        except (AttributeError, TypeError):
            msg = msg.model_copy(update={"content": cleaned})
    tracker.log(
        type="llm",
        model=selected_model,
        input_tokens=resp.prompt_eval_count or 0,
        output_tokens=resp.eval_count or 0,
        thinking_chars=len(msg.thinking or ""),
        seconds=round(time.time() - start, 2),
        asked_tools=[tc.function.name for tc in (msg.tool_calls or [])],
    )
    return msg


def embed(texts):
    """Embeddings for cosine similarity. Uses a separate small embedding model."""
    return _client.embed(model=config.EMBED_MODEL, input=texts).embeddings
