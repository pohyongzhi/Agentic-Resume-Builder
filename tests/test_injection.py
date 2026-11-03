import pytest

from eval import injection


PAIR = {
    "pair_id": "pair-001",
    "resume": "# Alex\n\nPython developer.",
    "jd": "# Engineer\n\nPython required.",
}


def test_canary_is_deterministic_and_unique_by_pair_and_channel():
    first = injection.derive_canary("dataset-a", "pair-001", "resume")

    assert first == injection.derive_canary("dataset-a", "pair-001", "resume")
    assert first != injection.derive_canary("dataset-a", "pair-001", "jd")
    assert first != injection.derive_canary("dataset-a", "pair-002", "resume")
    assert first.startswith("INJECTION_CANARY_")


def test_inject_pair_changes_only_selected_channel():
    attacked, metadata = injection.inject_pair(PAIR, "jd", "dataset-a")

    assert attacked["resume"] == PAIR["resume"]
    assert attacked["jd"].startswith(PAIR["jd"])
    assert metadata["canary"] in attacked["jd"]
    assert attacked["injection_channel"] == "jd"


def test_inject_pair_rejects_canary_collision():
    canary = injection.derive_canary("dataset-a", PAIR["pair_id"], "resume")
    pair = {**PAIR, "resume": PAIR["resume"] + "\n" + canary}

    with pytest.raises(ValueError, match="collision"):
        injection.inject_pair(pair, "resume", "dataset-a")


def test_canary_detection_accepts_case_and_markdown_but_not_partial_tokens():
    canary = "INJECTION_CANARY_ABC123"
    output = f"Used Python. **{canary.lower()}** and `{canary}`."

    assert len(injection.canary_matches(output, canary)) == 2
    assert injection.canary_matches("X" + canary, canary) == []
    assert injection.canary_matches(canary + "_EXTRA", canary) == []


def test_generation_failure_is_not_counted_as_security_success_or_failure():
    score = injection.evaluate_injection_output("", "INJECTION_CANARY_A", "timeout")

    assert score["generation_completed"] is False
    assert score["injection_success"] is None
    assert score["generation_error"] == "timeout"


def test_fake_workflow_attack_runs_without_network():
    def workflow(pair, tracker):
        tracker.append(pair["injection_channel"])
        return {"resume_md": f"Tailored resume\n{pair['injection_canary']}", "errors": []}

    tracker = []
    row = injection.run_injection_attack(
        "fake", workflow, PAIR, "dataset-a", "resume", tracker
    )

    assert row["generation_completed"] is True
    assert row["injection_success"] is True
    assert row["canary_match_count"] == 1
    assert tracker == ["resume"]


def test_summary_excludes_generation_failures_from_rate_and_reports_them():
    rows = [
        {"workflow": "w1", "channel": "resume", "generation_completed": True,
         "injection_success": True},
        {"workflow": "w1", "channel": "resume", "generation_completed": True,
         "injection_success": False},
        {"workflow": "w1", "channel": "resume", "generation_completed": False,
         "injection_success": None},
    ]
    summary = injection.summarise_injection(rows)
    overall = next(item for item in summary if item["scope"] == "overall")

    assert overall["attempted"] == 3
    assert overall["completed"] == 2
    assert overall["generation_failures"] == 1
    assert overall["injection_success_rate"] == 0.5
    assert overall["wilson_95_low"] < 0.5 < overall["wilson_95_high"]


def test_manifest_is_deterministic_and_immutable(tmp_path, monkeypatch):
    selected = [
        {"pair_id": "p1"},
        {"pair_id": "p2"},
    ]
    monkeypatch.setattr(injection, "select_stratified_pairs", lambda pairs, seed: selected)
    first = injection.injection_manifest(selected, "dataset-a", seed=7)
    second = injection.injection_manifest(list(reversed(selected)), "dataset-a", seed=7)

    assert first == second
    path = tmp_path / "manifest.json"
    injection.write_injection_manifest(path, first)
    injection.write_injection_manifest(path, first)
    with pytest.raises(RuntimeError, match="INJECTION_MANIFEST_CHANGED"):
        injection.write_injection_manifest(path, {**first, "seed": 8})

