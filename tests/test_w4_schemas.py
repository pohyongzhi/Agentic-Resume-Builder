"""Tests for W4 schemas (test plan §1)."""
import json

from common.schemas import Claim, GroundedExperience, GroundedResume, VerifyBatch, llm_schema


def test_claim_with_list_sources():
    """1.1: Claim with list of sources."""
    c = Claim.model_validate({"text": "x", "sources": ["S3", "S4"]})
    assert c.sources == ["S3", "S4"]


def test_claim_with_string_sources():
    """1.2: sources given as a string."""
    c = Claim.model_validate({"text": "x", "sources": "S3, S4"})
    assert c.sources == ["S3", "S4"]


def test_claim_sources_normalized():
    """1.3: Normalize and deduplicate sources."""
    c = Claim.model_validate({"text": "x", "sources": ["[s3]", "S 4", "S3"]})
    assert c.sources == ["S3", "S4"]


def test_claim_sources_invalid():
    """1.4: Invalid source strings are dropped."""
    c = Claim.model_validate({"text": "x", "sources": ["foo"]})
    assert c.sources == []


def test_grounded_experience_plain_string_bullet():
    """1.5: Plain string bullet converted to Claim."""
    exp = GroundedExperience.model_validate({
        "employer": "X", "title": "Y", "bullets": ["Did a thing"]
    })
    assert len(exp.bullets) == 1
    assert exp.bullets[0].text == "Did a thing"
    assert exp.bullets[0].sources == []


def test_grounded_resume_plain_string_summary():
    """1.6: Plain string summary converted to Claim."""
    resume = GroundedResume.model_validate({
        "name": "A", "summary": "One sentence."
    })
    assert len(resume.summary) == 1
    assert resume.summary[0].text == "One sentence."
    assert resume.summary[0].sources == []


def test_grounded_resume_null_fields():
    """1.7: Null fields become empty lists (inherits Schema null handling)."""
    resume = GroundedResume.model_validate({
        "name": "A", "summary": None, "experience": None
    })
    assert resume.summary == []
    assert resume.experience == []


def test_llm_schema_grounded_resume():
    """1.8: llm_schema produces valid JSON schema for GroundedResume."""
    schema = llm_schema(GroundedResume)
    assert "properties" in schema
    assert "name" in schema["properties"]
    # Should be serializable
    json.dumps(schema)


def test_llm_schema_verify_batch():
    """1.9: llm_schema has verdicts."""
    schema = llm_schema(VerifyBatch)
    assert "verdicts" in schema["properties"]
def test_null_claim_sources_become_empty_citations():
    from common.schemas import Claim, GroundedProject

    assert Claim.model_validate({"text": "Built reports", "sources": None}).sources == []
    project = GroundedProject.model_validate({"name": "Reporting Tool", "description": "Built reports"})
    assert project.description.text == "Built reports"
    assert project.description.sources == []
