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
