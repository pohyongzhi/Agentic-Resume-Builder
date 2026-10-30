import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config
from common import llm
from common.llm import Tracker
from workflows import WORKFLOWS, WORKFLOW_SPECS


def response(content="## Summary\nSupported resume"):
    message = SimpleNamespace(content=content, tool_calls=[], thinking="")
    return SimpleNamespace(message=message, prompt_eval_count=10, eval_count=5)


class ModelRoutingTests(unittest.TestCase):
    def test_seven_stable_v5_variant_ids_and_statuses(self):
        self.assertEqual(list(WORKFLOW_SPECS), [
            "one_pass",
            "agentic_baseline",
            "multi_critic_review",
            "evidence_grounded_generation",
            "fine_tuned_small_generator",
            "multi_draft_selector",
            "planner_section_writers",
        ])
        self.assertEqual(set(WORKFLOWS), {"one_pass", "fine_tuned_small_generator"})
        self.assertEqual(WORKFLOW_SPECS["one_pass"].number, 1)
        self.assertEqual(WORKFLOW_SPECS["one_pass"].status, "implemented")
        self.assertEqual(WORKFLOW_SPECS["fine_tuned_small_generator"].number, 5)
        self.assertEqual(WORKFLOW_SPECS["fine_tuned_small_generator"].status, "implemented")
        for variant_id in (
            "agentic_baseline",
            "multi_critic_review",
            "evidence_grounded_generation",
            "multi_draft_selector",
            "planner_section_writers",
        ):
            self.assertEqual(WORKFLOW_SPECS[variant_id].status, "planned")

    def test_chat_defaults_to_base_model_and_logs_it(self):
        tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response()) as call:
            llm.chat([{"role": "user", "content": "test"}], tracker)
        self.assertEqual(call.call_args.kwargs["model"], config.BASE_MODEL)
        self.assertEqual(call.call_args.kwargs["options"]["num_predict"], config.NUM_PREDICT)
        self.assertEqual(tracker.events[0]["model"], config.BASE_MODEL)

    def test_chat_explicit_override_uses_and_logs_fine_tuned_model(self):
        tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response()) as call:
            llm.chat(
                [{"role": "user", "content": "test"}], tracker,
                model=config.FINE_TUNED_MODEL,
            )
        self.assertEqual(call.call_args.kwargs["model"], config.FINE_TUNED_MODEL)
        self.assertEqual(tracker.events[0]["model"], config.FINE_TUNED_MODEL)

    def test_variants_one_and_five_route_identical_inputs_to_distinct_models(self):
        pair = {"resume": "Supported resume", "jd": "Supported job"}

        one_pass_tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response("one-pass output")):
            one_pass_output = WORKFLOWS["one_pass"](pair, one_pass_tracker)

        fine_tuned_tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response("fine-tuned output")):
            fine_tuned_output = WORKFLOWS["fine_tuned_small_generator"](
                pair, fine_tuned_tracker)

        self.assertEqual(one_pass_tracker.events[0]["model"], config.BASE_MODEL)
        self.assertEqual(fine_tuned_tracker.events[0]["model"], config.FINE_TUNED_MODEL)
        self.assertEqual(one_pass_output["resume_md"], "one-pass output")
        self.assertEqual(fine_tuned_output["resume_md"], "fine-tuned output")

    def test_application_defaults_to_one_pass_selector(self):
        import app

        self.assertEqual(inspect.signature(app.run).parameters["workflow_id"].default, "one_pass")
        self.assertEqual(app.IMPLEMENTED_CHOICES[0][1], "one_pass")


if __name__ == "__main__":
    unittest.main()
