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
    def predict(self, pairs):
        # Entail every claim against the first source chunk.
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


def test_claim_extraction_ignores_contact_and_section_labels():
    claims = metrics.extract_claims(
        "## Summary\nalex@example.test | +65 6000 0000 | Singapore\n"
        "- Built reliable Python APIs for enterprise customers.\n"
    )
    assert claims == ["Built reliable Python APIs for enterprise customers."]
