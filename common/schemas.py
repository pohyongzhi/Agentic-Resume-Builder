"""Fixed schemas passed between agents. The JSON schema of each model is also sent to Ollama
(`format=`), so the model's reply is constrained to this shape."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Schema(BaseModel):
    """Small models often send null for empty fields, or one string where a list is expected.
    Treat null as 'use the default' and wrap a lone string into a one-item list."""

    @model_validator(mode="before")
    @classmethod
    def _tolerate_small_model_output(cls, data):
        if not isinstance(data, dict):
            return data
        out = {}
        for k, v in data.items():
            if v is None:
                continue
            field = cls.model_fields.get(k)
            if field is not None and field.annotation == list[str] and isinstance(v, str):
                v = [v] if v.strip() else []
            out[k] = v
        return out


class Contact(Schema):
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)


class Experience(Schema):
    employer: str
    title: str
    start: str = ""
    end: str = ""
    bullets: list[str] = Field(default_factory=list)


class Education(Schema):
    institution: str
    degree: str
    start: str = ""
    end: str = ""
    details: list[str] = Field(default_factory=list)


class Project(Schema):
    name: str
    description: str = ""
    bullets: list[str] = Field(default_factory=list)


class CandidateData(Schema):
    """What the resume parser extracts. raw_text is the original text, kept for checking."""
    name: str
    contact: Contact = Field(default_factory=Contact)
    summary: str = ""
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    raw_text: str = ""


class JobData(Schema):
    """What the JD retriever extracts. source_text is the page or pasted text it came from."""
    title: str
    company: str = ""
    responsibilities: list[str] = Field(default_factory=list)
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    source_text: str = ""


class GeneratedResume(Schema):
    """What the resume creator produces. The harness validates it, then renders it to Markdown."""
    name: str
    contact: Contact = Field(default_factory=Contact)
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class CriticIssue(Schema):
    """One actionable problem found by a W3 critic."""
    category: str
    severity: Literal["minor", "major", "critical"]
    evidence: str
    recommendation: str


class CriticFeedback(Schema):
    """Common structured response returned by every specialist critic."""
    critic: Literal["fact_checker", "recruiter", "formatting"]
    approved: bool
    score: int = Field(ge=1, le=5)
    summary: str
    issues: list[CriticIssue] = Field(default_factory=list)


def llm_schema(model: type[BaseModel], exclude: tuple[str, ...] = ()) -> dict:
    """JSON schema for Ollama's `format=`, minus fields the model should not fill (e.g. raw_text)."""
    schema = model.model_json_schema()
    for field in exclude:
        schema.get("properties", {}).pop(field, None)
        if field in schema.get("required", []):
            schema["required"].remove(field)
    return schema


# ========== W4 Evidence-grounded schemas ==========

import re


def _claims_before(v):
    """Shared field_validator for bullets/details/summary that may be strings or missing."""
    if v is None:
        return []
    if isinstance(v, str):
        return [{"text": v, "sources": []}]
    if isinstance(v, list):
        return [{"text": item, "sources": []} if isinstance(item, str) else item for item in v]
    return v


class Claim(Schema):
    """One sentence/bullet the creator wrote, plus the span IDs it is based on."""
    text: str
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_claim(cls, data):
        # Accept a bare string as a Claim
        if isinstance(data, str):
            return {"text": data, "sources": []}
        if not isinstance(data, dict):
            return data

        # Normalize sources field
        sources = data.get("sources") or []
        if isinstance(sources, str):
            # Extract IDs from string like "S3, S4" or "[S3] [S4]"
            sources = re.findall(r"[Ss]\s*(\d+)", sources)
        elif isinstance(sources, list):
            # Extract IDs from each item in the list
            extracted = []
            for item in sources:
                if isinstance(item, str):
                    ids = re.findall(r"[Ss]\s*(\d+)", item)
                    extracted.extend(ids)
            sources = extracted

        # Normalize to ["S3", "S4"] format, deduplicate while preserving order
        seen = set()
        normalized = []
        for s_id in sources:
            normalized_id = f"S{s_id}"
            if normalized_id not in seen:
                seen.add(normalized_id)
                normalized.append(normalized_id)

        data["sources"] = normalized
        return data


class GroundedExperience(Schema):
    employer: str
    title: str
    start: str = ""
    end: str = ""
    bullets: list[Claim] = Field(default_factory=list)

    @classmethod
    def _validate_bullets(cls, v):
        return _claims_before(v)

    # Register the field validator
    _bullets_validator = model_validator(mode="before")(
        lambda cls, data: {**data, "bullets": cls._validate_bullets(data.get("bullets"))}
        if isinstance(data, dict) else data
    )


class GroundedEducation(Schema):
    institution: str
    degree: str
    start: str = ""
    end: str = ""
    details: list[Claim] = Field(default_factory=list)

    @classmethod
    def _validate_details(cls, v):
        return _claims_before(v)

    _details_validator = model_validator(mode="before")(
        lambda cls, data: {**data, "details": cls._validate_details(data.get("details"))}
        if isinstance(data, dict) else data
    )


class GroundedProject(Schema):
    name: str
    description: Claim = Field(default_factory=lambda: Claim(text=""))
    bullets: list[Claim] = Field(default_factory=list)

    @classmethod
    def _validate_bullets(cls, v):
        return _claims_before(v)

    _bullets_validator = model_validator(mode="before")(
        lambda cls, data: {**data, "bullets": cls._validate_bullets(data.get("bullets"))}
        if isinstance(data, dict) else data
    )


class GroundedResume(Schema):
    name: str
    contact: Contact = Field(default_factory=Contact)
    summary: list[Claim] = Field(default_factory=list)
    experience: list[GroundedExperience] = Field(default_factory=list)
    education: list[GroundedEducation] = Field(default_factory=list)
    projects: list[GroundedProject] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    @classmethod
    def _validate_summary(cls, v):
        return _claims_before(v)

    _summary_validator = model_validator(mode="before")(
        lambda cls, data: {**data, "summary": cls._validate_summary(data.get("summary"))}
        if isinstance(data, dict) else data
    )


class ClaimVerdict(Schema):
    claim_id: str
    supported: bool
    reason: str = ""


class VerifyBatch(Schema):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
