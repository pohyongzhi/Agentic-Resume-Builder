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
    def test_six_stable_workflow_ids_and_statuses(self):
        self.assertEqual(list(WORKFLOW_SPECS), [
            "baseline",
            "multi_critic_review",
            "evidence_grounded_generation",
            "fine_tuned_small_generator",
            "multi_draft_selector",
            "planner_section_writers",
        ])
        self.assertEqual(set(WORKFLOWS), {"baseline", "fine_tuned_small_generator"})
        self.assertEqual(WORKFLOW_SPECS["baseline"].status, "implemented")
        self.assertEqual(WORKFLOW_SPECS["fine_tuned_small_generator"].status, "implemented")
        for workflow_id in (
            "multi_critic_review", "evidence_grounded_generation",
            "multi_draft_selector", "planner_section_writers",
        ):
            self.assertEqual(WORKFLOW_SPECS[workflow_id].status, "planned")

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

    def test_workflow_one_and_four_route_identical_inputs_to_distinct_models(self):
        pair = {"resume": "Supported resume", "jd": "Supported job"}

        baseline_tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response("baseline output")):
            baseline_output = WORKFLOWS["baseline"](pair, baseline_tracker)

        fine_tuned_tracker = Tracker()
        with patch.object(llm._client, "chat", return_value=response("fine-tuned output")):
            fine_tuned_output = WORKFLOWS["fine_tuned_small_generator"](
                pair, fine_tuned_tracker)

        self.assertEqual(baseline_tracker.events[0]["model"], config.BASE_MODEL)
        self.assertEqual(fine_tuned_tracker.events[0]["model"], config.FINE_TUNED_MODEL)
        self.assertEqual(baseline_output["resume_md"], "baseline output")
        self.assertEqual(fine_tuned_output["resume_md"], "fine-tuned output")

    def test_application_defaults_to_baseline_selector(self):
        import app

        self.assertEqual(inspect.signature(app.run).parameters["workflow_id"].default, "baseline")
        self.assertEqual(app.IMPLEMENTED_CHOICES[0][1], "baseline")


if __name__ == "__main__":
    unittest.main()
