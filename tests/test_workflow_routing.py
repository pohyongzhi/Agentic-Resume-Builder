from types import SimpleNamespace
from unittest.mock import patch

import config
from common import llm
from common.llm import Tracker
from workflows import WORKFLOWS, WORKFLOW_MODELS


def response(content="## Summary\nSupported resume"):
    message = SimpleNamespace(content=content, tool_calls=[], thinking="")
    return SimpleNamespace(
        message=message,
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )


def test_current_workflows_are_registered_in_order():
    assert list(WORKFLOWS) == [
        "w1_single_pass",
        "w2_baseline",
        "w3_critic",
        "w4_evidence",
        "w5_fine_tuned",
        "w6_best_of_n",
        "w7_plan_execute",
    ]


def test_only_w5_routes_to_the_fine_tuned_model():
    assert WORKFLOW_MODELS["w5_fine_tuned"] == config.FINE_TUNED_MODEL
    assert all(
        model == config.BASE_MODEL
        for workflow, model in WORKFLOW_MODELS.items()
        if workflow != "w5_fine_tuned"
    )


def test_chat_defaults_to_base_model_and_logs_it():
    tracker = Tracker()
    with patch.object(llm._client, "chat", return_value=response()) as call:
        llm.chat([{"role": "user", "content": "test"}], tracker)
    assert call.call_args.kwargs["model"] == config.BASE_MODEL
    assert tracker.events[0]["model"] == config.BASE_MODEL


def test_chat_preserves_options_with_explicit_model_override():
    tracker = Tracker()
    schema = {"type": "object"}
    with patch.object(llm._client, "chat", return_value=response()) as call:
        llm.chat(
            [{"role": "user", "content": "test"}],
            tracker,
            model=config.FINE_TUNED_MODEL,
            format=schema,
            num_predict=321,
        )
    assert call.call_args.kwargs["model"] == config.FINE_TUNED_MODEL
    assert call.call_args.kwargs["format"] == schema
    assert call.call_args.kwargs["options"]["num_predict"] == 321
    assert tracker.events[0]["model"] == config.FINE_TUNED_MODEL


def test_w1_and_w5_send_identical_inputs_to_distinct_models():
    pair = {"resume": "Supported resume", "jd": "Supported job"}

    w1_tracker = Tracker()
    with patch.object(llm._client, "chat", return_value=response("w1 output")):
        w1_output = WORKFLOWS["w1_single_pass"](pair, w1_tracker)

    w5_tracker = Tracker()
    with patch.object(llm._client, "chat", return_value=response("w5 output<|im_end|>")):
        w5_output = WORKFLOWS["w5_fine_tuned"](pair, w5_tracker)

    assert w1_tracker.events[0]["model"] == config.BASE_MODEL
    assert w5_tracker.events[0]["model"] == config.FINE_TUNED_MODEL
    assert w1_output["resume_md"] == "w1 output"
    assert w5_output["resume_md"] == "w5 output"


def test_application_defaults_to_w1_and_displays_w5():
    import app

    assert app.WORKFLOW_CHOICES[0][1] == "w1_single_pass"
    assert ("W5 - Fine-tuned: selected V4.1 small generator", "w5_fine_tuned") in app.WORKFLOW_CHOICES
