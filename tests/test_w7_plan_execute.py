"""Checks that W7 does what its description and the README require. Runs without Ollama:
the Ollama client is replaced by a fake, so the real common.llm.chat() (and its Tracker logging)
still runs for every call.

    python -m unittest tests.test_w7_plan_execute -v

W7 description: "A planner first produces a tailoring plan: which JD requirement maps to which
evidence in the resume, the section order, and what to cut. Separate writer agents then draft each
section in parallel, and an assembler merges the sections and checks they are consistent."
"""
import json
import re
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import config
import common.llm
from common.llm import Tracker
from common.prompts import BASE_RULES
from workflows import WORKFLOWS, w7_plan_execute as w7

RESUME = "Alex Tan\nalex@example.com\n\nExperience\nAnalyst Intern, Acme (2025)\n- Built SQL dashboards\n"
JD = "Data Analyst. Requirements: SQL, dashboards, Tableau."

WRITER_DELAY = 0.3   # seconds each fake writer call takes, to see whether writers overlap in time


def make_plan(sections=("Summary", "Experience", "Education", "Skills")):
    return {
        "header": "Alex Tan\nalex@example.com",
        "sections": [{"name": s, "lead_with": [f"lead-{s}"], "cut": [f"cut-{s}"]} for s in sections],
        "requirements": [
            {"requirement": "SQL", "evidence": "Built SQL dashboards", "section": "Experience"},
            # Not in the resume: W7 must pass the planner's mapping on unchanged (no evidence check).
            {"requirement": "Leadership", "evidence": "Led a team of 50 engineers", "section": "Experience"},
            {"requirement": "Tableau", "evidence": None, "section": "Skills"},
        ],
    }


class FakeOllama:
    """Stands in for common.llm._client. Records every call and answers by role."""

    def __init__(self, plan_replies):
        self.plan_replies = list(plan_replies)   # one reply per planner call
        self.calls = []
        self.lock = threading.Lock()

    def chat(self, model, messages, tools=None, think=None, options=None):
        start = time.perf_counter()
        system, user = messages[0]["content"], messages[-1]["content"]
        if "tailoring plan" in system:
            role, content = "plan", self.plan_replies.pop(0)
        elif "write ONLY the" in system:
            name = re.search(r'write ONLY the "(.+?)" section', system).group(1)
            time.sleep(WRITER_DELAY)
            # Realistic length, and '•' bullets the way writers copy them from PDF resumes.
            role, content = f"write:{name}", (f"## {name}\n**Entry for {name}** | 2025\n"
                                              f"• first fact for {name}, written out in a full sentence\n"
                                              f"• second fact for {name}, also a full sentence")
        elif "You are the assembler" in system:
            role, content = "assemble", user.split("SECTIONS (in order):", 1)[1].strip()
        else:
            raise AssertionError("unexpected prompt: " + system[:80])
        with self.lock:
            self.calls.append({"role": role, "model": model, "messages": messages, "tools": tools,
                               "think": think, "options": options, "start": start, "end": time.perf_counter()})
        return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None, thinking=None),
                               prompt_eval_count=10, eval_count=5)


def run_w7(plan_replies):
    fake = FakeOllama(plan_replies)
    tracker = Tracker()
    with mock.patch.object(common.llm, "_client", fake):
        out = w7.run({"resume": RESUME, "jd": JD}, tracker)
    return out, fake, tracker


class W7MatchesDescription(unittest.TestCase):

    def setUp(self):
        self.plan = make_plan()
        self.out, self.fake, self.tracker = run_w7([json.dumps(self.plan)])
        self.writers = [c for c in self.fake.calls if c["role"].startswith("write:")]

    # --- "A planner first produces a tailoring plan" ---
    def test_planner_runs_first_and_once(self):
        self.assertEqual(self.fake.calls[0]["role"], "plan")
        self.assertEqual(sum(c["role"] == "plan" for c in self.fake.calls), 1)

    def test_plan_has_mapping_order_and_cut(self):
        logged = next(e for e in self.tracker.events if e["type"] == "w7_plan")
        self.assertTrue(logged["requirements"], "requirement -> evidence mapping")
        self.assertEqual([s["name"] for s in logged["sections"]], [s["name"] for s in self.plan["sections"]])
        self.assertTrue(all(s["cut"] for s in logged["sections"]), "what to cut")

    def test_plan_reaches_writers_unchanged(self):
        """No code checks: even evidence that isn't in the resume is passed on as the planner wrote it."""
        experience = next(c for c in self.writers if c["role"] == "write:Experience")
        brief = experience["messages"][-1]["content"]
        self.assertIn("Led a team of 50 engineers", brief)
        self.assertIn("lead-Experience", brief)
        self.assertIn("cut-Experience", brief)

    # --- "Separate writer agents then draft each section in parallel" ---
    def test_one_writer_per_planned_section(self):
        self.assertEqual(sorted(c["role"] for c in self.writers),
                         sorted(f"write:{s['name']}" for s in self.plan["sections"]))

    def test_each_writer_gets_only_its_own_lead_and_cut(self):
        for c in self.writers:
            name, brief = c["role"][6:], c["messages"][-1]["content"]
            for other in self.plan["sections"]:
                if other["name"] != name:
                    self.assertNotIn(f"cut-{other['name']}", brief)
                    self.assertNotIn(f"lead-{other['name']}", brief)

    def test_writers_run_in_parallel(self):
        latest_start = max(c["start"] for c in self.writers)
        earliest_end = min(c["end"] for c in self.writers)
        self.assertLess(latest_start, earliest_end, "all writers were running at the same moment")
        span = max(c["end"] for c in self.writers) - min(c["start"] for c in self.writers)
        self.assertLess(span, WRITER_DELAY * len(self.writers) * 0.6, "faster than running them one by one")

    # --- "an assembler merges the sections and checks they are consistent" ---
    def test_assembler_runs_last_once_with_plan_and_all_sections(self):
        self.assertEqual(self.fake.calls[-1]["role"], "assemble")
        self.assertEqual(sum(c["role"] == "assemble" for c in self.fake.calls), 1)
        prompt = self.fake.calls[-1]["messages"][-1]["content"]
        self.assertIn("PLAN:", prompt)
        for s in self.plan["sections"]:
            self.assertIn(f"## {s['name']}", prompt)
        system = self.fake.calls[-1]["messages"][0]["content"]
        self.assertIn("consistent", system)

    def test_output_follows_planned_order_with_header(self):
        md = self.out["resume_md"]
        self.assertTrue(md.startswith("Alex Tan\nalex@example.com"))
        headings = re.findall(r"^## (.+)$", md, re.M)
        self.assertEqual(headings, [s["name"] for s in self.plan["sections"]])

    def test_pdf_bullets_become_markdown_bullets(self):
        md = self.out["resume_md"]
        self.assertNotIn("•", md)
        self.assertIn("- first fact for Experience", md)

    def test_call_count_is_1_plus_N_plus_1(self):
        self.assertEqual(len(self.fake.calls), 1 + len(self.plan["sections"]) + 1)
        _, fake, _ = run_w7([json.dumps(make_plan(("Summary", "Projects", "Skills")))])
        self.assertEqual(len(fake.calls), 1 + 3 + 1, "N follows the planner's choice of sections")


class W7OutputCleanup(unittest.TestCase):
    """Formatting plumbing: none of these change what the model wrote, only how it is laid out."""

    def test_reworded_own_heading_is_not_duplicated(self):
        out = w7._only_section("## Professional Experience\n- did X", "Experience", ["Summary", "Experience"])
        self.assertEqual(out, "## Experience\n- did X")

    def test_writer_stops_at_another_planned_section(self):
        out = w7._only_section("## Experience\n- did X\n## Skills\nPython", "Experience", ["Experience", "Skills"])
        self.assertEqual(out, "## Experience\n- did X")

    def test_subheadings_inside_a_section_are_kept(self):
        out = w7._only_section("## Experience\n### DBS Bank\n- did X", "Experience", ["Experience", "Skills"])
        self.assertIn("### DBS Bank", out)

    def test_assembler_name_line_is_dropped_not_turned_into_a_heading(self):
        plan = make_plan(("Summary", "Experience"))
        name_first = "### Alex Tan\n" + "\n\n".join(
            f"## {s}\n- a full sentence about {s} that makes the section long enough" * 3 for s in ("Summary", "Experience"))

        class Assembler(FakeOllama):
            def chat(self, model, messages, **kw):
                reply = super().chat(model, messages, **kw)
                if "You are the assembler" in messages[0]["content"]:
                    reply.message.content = name_first
                return reply

        fake, tracker = Assembler([json.dumps(plan)]), Tracker()
        with mock.patch.object(common.llm, "_client", fake):
            md = w7.run({"resume": RESUME, "jd": JD}, tracker)["resume_md"]
        self.assertEqual(md.count("Alex Tan"), 1)
        self.assertEqual(re.findall(r"^## (.+)$", md, re.M), ["Summary", "Experience"])


class W7FollowsReadmeRules(unittest.TestCase):

    def setUp(self):
        self.out, self.fake, self.tracker = run_w7([json.dumps(make_plan())])

    def test_registered_in_workflows(self):
        self.assertIs(WORKFLOWS["w7_plan_execute"], w7.run)

    def test_returns_the_required_dict(self):
        self.assertEqual(set(self.out), {"resume_md", "hit_budget", "finished"})
        self.assertIsInstance(self.out["resume_md"], str)
        self.assertIs(self.out["hit_budget"], False)
        self.assertIs(self.out["finished"], True)

    def test_every_prompt_starts_with_base_rules(self):
        for c in self.fake.calls:
            self.assertTrue(c["messages"][0]["content"].startswith(BASE_RULES), c["role"])

    def test_every_call_goes_through_chat_with_shared_settings(self):
        llm_events = [e for e in self.tracker.events if e["type"] == "llm"]
        self.assertEqual(len(llm_events), len(self.fake.calls), "every call is logged, so cost is counted")
        for c in self.fake.calls:
            self.assertEqual(c["model"], config.MODEL)
            self.assertEqual(c["options"], {"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX})
            self.assertEqual(c["think"], config.THINK)
            self.assertIsNone(c["tools"], "fixed pipeline, no tool loop")

    def test_bad_plan_json_retries_once_then_finishes_with_default_sections(self):
        out, fake, _ = run_w7(["not json", "still not json"])
        self.assertEqual(sum(c["role"] == "plan" for c in fake.calls), 2)
        self.assertEqual(sorted(c["role"] for c in fake.calls if c["role"].startswith("write:")),
                         sorted(f"write:{n}" for n in ("Summary", "Experience", "Education", "Skills")))
        self.assertIs(out["finished"], False, "a run without a real plan is not counted as finished")
        self.assertTrue(out["resume_md"].startswith("Alex Tan"), "header falls back to the resume's first lines")


if __name__ == "__main__":
    unittest.main()
