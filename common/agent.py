r"""Shared ReAct agent built with LangGraph. Used by every tool-using workflow.

The graph is the loop from Topic 7 §2, drawn as nodes:

    START -> agent --(asked for a tool?)--> tools -> agent -> ...
                 \--(called submit_resume, or out of steps)--> END

Each workflow only supplies: a system prompt + a dict of tools. Everything else
(step budget, validation, logging, how the final resume is collected) is identical.
"""
import json
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

import config
from common.llm import chat


class Tool:
    """A tool = a Python function + the JSON schema the model reads to decide when/how to call it."""

    def __init__(self, fn, description, params, required=None):
        self.fn = fn
        self.name = fn.__name__
        self.schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {"type": "object", "properties": params,
                               "required": required or list(params)},
            },
        }


def submit_resume(resume_md: str) -> str:
    """Every agent finishes by calling this with the final tailored resume."""
    return "Submitted."


SUBMIT = Tool(submit_resume,
              "Call this ONCE when you are finished, with the complete final tailored resume in Markdown.",
              {"resume_md": {"type": "string", "description": "The full final resume in Markdown"}})


class AgentState(TypedDict):
    messages: list
    steps: int
    final_resume: str
    last_draft: str
    stopped: bool


def build_agent(system_prompt, tools, tracker, max_steps=None, executor=None, include_submit=True,
                finish_tools=("submit_resume",), nudge=None):
    """Returns a compiled LangGraph app. Call app.invoke({...}) via run_agent().

    executor: optional fn(name, args) -> (ok, result_text, stop). When given, EVERY tool call
    (including unknown names) goes through it instead of the tool functions, so a harness can
    enforce its own rules. finish_tools: tool names that end the loop when they succeed.
    nudge: optional message sent back when the model replies without a tool call, instead of
    ending the loop (still limited by max_steps)."""
    max_steps = max_steps or config.MAX_STEPS
    tools = list(tools) + ([SUBMIT] if include_submit else [])
    by_name = {t.name: t for t in tools}
    schemas = [t.schema for t in tools]

    def agent_node(state):
        msg = chat(state["messages"], tracker, tools=schemas)
        entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            entry["tool_calls"] = [{"function": {"name": tc.function.name,
                                                 "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls]
        draft = state["last_draft"]
        if not msg.tool_calls and msg.content and len(msg.content) > 200:
            draft = msg.content  # model wrote a resume in plain text instead of submitting
        new = [entry]
        if nudge and not msg.tool_calls:
            new.append({"role": "user", "content": nudge})
        return {"messages": state["messages"] + new, "steps": state["steps"] + 1, "last_draft": draft}

    def tools_node(state):
        last = state["messages"][-1]
        new_msgs, final, draft, stopped = [], state["final_resume"], state["last_draft"], state["stopped"]
        for call in last.get("tool_calls", []):
            if stopped:
                break
            name, args = call["function"]["name"], call["function"]["arguments"]
            start = time.time()
            if isinstance(args, str):           # some models send arguments as a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            tool = by_name.get(name)
            if executor is not None:
                ok, result, stop = executor(name, args if isinstance(args, dict) else {})
                stopped = stopped or stop
            elif tool is None:
                ok, result = False, f"Error: no tool called '{name}'. Available: {list(by_name)}"
            elif not isinstance(args, dict):
                ok, result = False, "Error: arguments must be a JSON object."
            else:
                missing = [r for r in tool.schema["function"]["parameters"]["required"] if r not in args]
                if missing:
                    ok, result = False, f"Error: missing arguments {missing}."
                else:
                    try:
                        result, ok = tool.fn(**args), True
                    except Exception as e:   # return errors the model can recover from (Topic 7 §1)
                        ok, result = False, f"Error running {name}: {e}"
            tracker.log(type="tool", name=name, args=args, ok=ok, seconds=round(time.time() - start, 3))
            if ok and name == "submit_resume":
                final = args["resume_md"]
            if ok and name in finish_tools:
                stopped = True
            if ok and isinstance(args, dict) and "resume_md" in args:
                draft = args["resume_md"]
            new_msgs.append({"role": "tool", "tool_name": name, "content": str(result)})
        return {"messages": state["messages"] + new_msgs, "final_resume": final, "last_draft": draft,
                "stopped": stopped}

    def route_after_agent(state):
        last = state["messages"][-1]
        if state["steps"] >= max_steps:
            return END
        if last["role"] == "user":    # nudged after a reply without a tool call
            return "agent"
        return "tools" if last.get("tool_calls") else END

    def route_after_tools(state):
        if state["stopped"] or state["final_resume"] or state["steps"] >= max_steps:
            return END
        return "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "agent", END])
    g.add_conditional_edges("tools", route_after_tools, ["agent", END])
    return g.compile()


def run_agent(system_prompt, user_prompt, tools, tracker, max_steps=None, **agent_kwargs):
    """Runs the agent and returns (final_resume, hit_budget, finished_cleanly).
    agent_kwargs (executor, include_submit, finish_tools) are passed to build_agent."""
    app = build_agent(system_prompt, tools, tracker, max_steps, **agent_kwargs)
    out = app.invoke({
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "steps": 0, "final_resume": "", "last_draft": "", "stopped": False,
    }, {"recursion_limit": 100})
    finished = bool(out["final_resume"])
    resume = out["final_resume"] or out["last_draft"]
    return resume, out["steps"] >= (max_steps or config.MAX_STEPS), finished
