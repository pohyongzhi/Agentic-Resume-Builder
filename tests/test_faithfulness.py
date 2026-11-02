import json
import numpy as np
import pytest

from eval import metrics


def test_nli_probabilities_accepts_probabilities_and_stably_softmaxes_logits():
    probabilities = np.array([[0.1, 0.8, 0.1]])
    assert np.allclose(metrics._probabilities(probabilities), probabilities)
    output = metrics._probabilities(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(output).all()
    assert output.argmax(axis=1).tolist() == [1]
    assert np.allclose(output.sum(axis=1), 1)


def test_nli_probabilities_rejects_bad_shapes_and_nonfinite_values():
    with pytest.raises(RuntimeError, match="SHAPE"):
        metrics._probabilities([[1, 2]])
    with pytest.raises(RuntimeError, match="NONFINITE"):
        metrics._probabilities([[1, np.nan, 2]])


class FakeNLI:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return np.array([[0.0, 5.0, 0.0] for _ in pairs])


def test_faithfulness_uses_semantic_label_mapping_and_keeps_claim_details(monkeypatch):
    metadata = {
        "model": "fake",
        "model_revision": "r1",
        "id2label": {"0": "neutral", "1": "contradiction", "2": "entailment"},
        "label2id": {"neutral": 0, "contradiction": 1, "entailment": 2},
    }

    class PermutedNLI:
        def predict(self, pairs):
            return np.array([[0.0, 0.0, 5.0] for _ in pairs])

    monkeypatch.setattr(metrics, "_load_nli", lambda: (PermutedNLI(), metadata))
    result = metrics.faithfulness(
        "Alex built Python APIs for customers.",
        "- Alex built Python APIs for customers and documented them.",
    )
    assert result["faithfulness"] == 1
    assert result["claim_scores"][0]["predicted_relation"] == "entailment"
    assert result["claim_scores"][0]["best_source_chunk"]
    assert result["claim_scores"][0]["best_evidence_id"]
    assert result["claim_scores"][0]["support_method"] == "nli"
    assert len(result["claim_scores"][0]["evidence_candidates"]) <= 3
    assert result["claim_scores"][0]["evidence_candidates"][0]["probabilities"]


def test_faithfulness_skips_nli_for_normalized_exact_support(monkeypatch):
    model = FakeNLI()
    metadata = {
        "model": "fake",
        "model_revision": "r1",
        "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
    }
    monkeypatch.setattr(metrics, "_load_nli", lambda: (model, metadata))
    result = metrics.faithfulness(
        "## Experience\n- Built reliable Python APIs.\n",
        "## Experience\n- Built reliable Python APIs.\n",
    )
    assert result["faithfulness"] == 1
    assert result["unsupported_claim_rate"] == 0
    assert result["claim_scores"][0]["support_method"] == "normalized_exact"
    assert model.pairs == []


def test_faithfulness_only_sends_top_evidence_candidates_to_nli(monkeypatch):
    model = FakeNLI()
    metadata = {
        "model": "fake",
        "model_revision": "r1",
        "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
    }
    monkeypatch.setattr(metrics, "_load_nli", lambda: (model, metadata))
    source = "## Experience\n" + "\n".join(
        f"- Delivered source evidence item {index}." for index in range(8)
    )
    result = metrics.faithfulness(source, "## Summary\nDelivered a useful result.")
    assert len(model.pairs) == metrics.config.NLI_EVIDENCE_LIMIT
    assert result["claim_scores"][0]["best_evidence_unit"] in {
        pair[0] for pair in model.pairs
    }


def test_deterministic_support_does_not_use_partial_word_matches(monkeypatch):
    model = FakeNLI()
    metadata = {
        "model": "fake",
        "model_revision": "r1",
        "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
    }
    monkeypatch.setattr(metrics, "_load_nli", lambda: (model, metadata))
    result = metrics.faithfulness("## Skills\nPostgreSQL\n", "## Skills\nSQL\n")
    assert result["claim_scores"][0]["support_method"] == "nli"
    assert model.pairs


def test_extract_claims_includes_individual_skill_items():
    assert metrics.extract_claims("## Skills\nPython, SQL; AWS\n") == [
        "Python", "SQL", "AWS",
    ]


def test_sentence_retrieval_does_not_rank_one_word_skill_above_matching_evidence():
    units = metrics.extract_evidence_units(
        "## Experience\n"
        "- Used JavaScript and TypeScript to build accessible user interfaces, improving "
        "team turnaround by 19%.\n"
        "## Skills\nJavaScript, TypeScript\n"
    )
    retrieved = metrics.retrieve_evidence_units(
        "Built accessible user interfaces using JavaScript and TypeScript, improving team "
        "turnaround by 19%.",
        units,
    )
    assert retrieved[0]["kind"] == "claim"
    assert "19%" in retrieved[0]["text"]


def test_faithfulness_cache_records_evidence_identity_and_reuses_result(
    monkeypatch, tmp_path,
):
    model = FakeNLI()
    metadata = {
        "model": "fake",
        "model_revision": "r1",
        "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"},
        "label2id": {"entailment": 0, "neutral": 1, "contradiction": 2},
    }
    monkeypatch.setattr(metrics, "_load_nli", lambda: (model, metadata))
    arguments = (
        "## Experience\n- Built Python APIs.\n",
        "## Experience\n- Developed Python services.\n",
    )
    first = metrics.faithfulness(*arguments, cache_dir=tmp_path)
    prediction_count = len(model.pairs)
    second = metrics.faithfulness(*arguments, cache_dir=tmp_path)
    cache_record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert second == first
    assert len(model.pairs) == prediction_count
    assert cache_record["identity"]["evidence_limit"] == metrics.config.NLI_EVIDENCE_LIMIT
    assert cache_record["identity"]["evaluator_version"] == "claim-evidence-nli-v3"


def test_claim_extraction_ignores_contact_and_section_labels():
    claims = metrics.extract_claims(
        "## Summary\nalex@example.test | +65 6000 0000 | Singapore\n"
        "- Built reliable Python APIs for enterprise customers.\n"
    )
    assert claims == ["Built reliable Python APIs for enterprise customers."]
