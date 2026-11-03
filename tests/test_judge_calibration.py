from eval.judge import JudgeConfig
from eval.judge_calibration import CONTROLS, calibration_checks, run_calibration


def _record(overall, prioritisation=3, concision=3):
    dimensions = {
        "relevance_to_jd": {"score": overall},
        "prioritisation_of_supported_experience": {"score": prioritisation},
        "clarity": {"score": overall},
        "concision": {"score": concision},
        "organisation_and_readability": {"score": overall},
    }
    return {
        "status": "scored",
        "result": {"overall_mean": overall, "dimensions": dimensions},
        "cache_key": f"cache-{overall}-{prioritisation}-{concision}",
        "attempts": [{
            "prompt_tokens": 10,
            "output_tokens": 5,
            "reasoning_tokens": 2,
            "total_tokens": 15,
        }],
    }


def _passing_records():
    return {
        "strong": _record(4.6, prioritisation=5, concision=5),
        "generic": _record(3.0, prioritisation=3, concision=3),
        "unsupported": _record(1.8, prioritisation=1, concision=3),
        "padded": _record(4.0, prioritisation=5, concision=3),
    }


def test_calibration_checks_pass_only_discriminating_judge():
    checks = calibration_checks(_passing_records())
    assert all(check["passed"] for check in checks)

    lenient = _passing_records()
    lenient["unsupported"] = _record(4.8, prioritisation=5, concision=5)
    failed = {check["name"] for check in calibration_checks(lenient) if not check["passed"]}
    assert "unsupported_is_penalised" in failed
    assert "quality_order_resolves" in failed


def test_run_calibration_records_usage_and_refuses_overwrite(tmp_path):
    records = _passing_records()

    def fake_judge(original, jd, tailored, cache_dir, **kwargs):
        control = next(item["id"] for item in CONTROLS if item["tailored_resume"] == tailored)
        return records[control]

    judge_config = JudgeConfig(provider="groq", model="openai/gpt-oss-120b")
    result = run_calibration(tmp_path, judge_config, client=object(), judge_fn=fake_judge)
    assert result["passed"] is True
    assert result["usage"] == {
        "requests": 4,
        "prompt_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 8,
        "total_tokens": 60,
    }
    assert (tmp_path / "calibration-result.json").exists()

    try:
        run_calibration(tmp_path, judge_config, client=object(), judge_fn=fake_judge)
    except RuntimeError as exc:
        assert "CALIBRATION_OUTPUT_EXISTS" in str(exc)
    else:
        raise AssertionError("calibration output was overwritten")
