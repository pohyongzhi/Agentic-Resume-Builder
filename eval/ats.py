"""Versioned, reproducible ATS proxy and provider boundary.

The offline score is a transparent compatibility proxy, not a commercial ATS score.
Provider adapters may be supplied by callers, but this module performs no network calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable


ATS_SCHEMA_VERSION = "1.0"
OFFLINE_PROXY_VERSION = "offline-ats-proxy-v1"
_REQUIRED_SECTIONS = ("experience", "education", "skills")
_SECTION_ALIASES = {
    "experience": ("experience", "work experience", "professional experience"),
    "education": ("education", "academic background"),
    "skills": ("skills", "technical skills", "core competencies"),
}


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"(?i)bearer\s+[a-z0-9._~+/=-]+", "Bearer [redacted]", message)
    message = re.sub(
        r"(?i)(api[-_ ]?key|token|secret)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2[redacted]",
        message,
    )
    return f"{type(exc).__name__}: {message}"[:1000]


def _contains_term(text: str, term: str) -> bool:
    return re.search(
        r"(?<![\w])" + re.escape(term.lower()) + r"(?![\w])",
        (text or "").lower(),
    ) is not None


def _normalise_keywords(keywords: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalised = []
    for item in keywords or ():
        keyword = str(item.get("keyword", "")).strip()
        if not keyword:
            continue
        synonyms = sorted({
            str(value).strip() for value in item.get("synonyms", []) if str(value).strip()
        }, key=str.lower)
        normalised.append({"keyword": keyword, "synonyms": synonyms})
    return sorted(normalised, key=lambda item: item["keyword"].lower())


def _term_coverage(text: str, keywords: Sequence[dict[str, Any]]) -> float | None:
    if not keywords:
        return None
    hits = 0
    for item in keywords:
        terms = [item["keyword"], *item["synonyms"]]
        hits += any(_contains_term(text, term) for term in terms)
    return hits / len(keywords)


def _section_score(text: str) -> tuple[float, list[str]]:
    headings = set()
    for line in text.splitlines():
        candidate = re.sub(r"^#{1,6}\s*", "", line.strip()).strip(" :").lower()
        if candidate:
            headings.add(candidate)
    found = []
    for section in _REQUIRED_SECTIONS:
        if any(alias in headings for alias in _SECTION_ALIASES[section]):
            found.append(section)
    return len(found) / len(_REQUIRED_SECTIONS), found


def _format_score(text: str) -> tuple[float, list[str]]:
    hazards = []
    if re.search(r"^\s*<table\b", text, re.MULTILINE | re.IGNORECASE):
        hazards.append("html_table")
    if re.search(r"^\s*\|.*\|\s*$", text, re.MULTILINE):
        hazards.append("markdown_table")
    if re.search(r"[\uE000-\uF8FF]", text):
        hazards.append("private_use_characters")
    return max(0.0, 1.0 - 0.34 * len(hazards)), hazards


@runtime_checkable
class ATSProvider(Protocol):
    """Stable boundary for offline or explicitly configured external scorers."""

    def identity(self) -> dict[str, Any]: ...

    def score(
        self,
        original_resume: str,
        job_description: str,
        tailored_resume: str,
        keywords: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OfflineATSProvider:
    """Transparent local proxy with no model, service, or mutable dependency."""

    version: str = OFFLINE_PROXY_VERSION

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "offline_reference",
            "provider_version": self.version,
            "schema_version": ATS_SCHEMA_VERSION,
            "score_kind": "ats_proxy_score",
        }

    def score(
        self,
        original_resume: str,
        job_description: str,
        tailored_resume: str,
        keywords: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del job_description
        if not tailored_resume or not tailored_resume.strip():
            raise ValueError("tailored resume is empty")
        normalised_keywords = _normalise_keywords(keywords)
        supported_keywords = [
            item for item in normalised_keywords
            if any(
                _contains_term(original_resume, term)
                for term in [item["keyword"], *item["synonyms"]]
            )
        ]
        section_completeness, sections_found = _section_score(tailored_resume)
        formatting_compatibility, formatting_hazards = _format_score(tailored_resume)
        printable = sum(character.isprintable() or character in "\n\t" for character in tailored_resume)
        parseability = printable / len(tailored_resume)
        components = {
            "parseability": parseability,
            "section_completeness": section_completeness,
            "formatting_compatibility": formatting_compatibility,
            "keyword_coverage": _term_coverage(tailored_resume, normalised_keywords),
            "supported_requirement_coverage": _term_coverage(
                tailored_resume, supported_keywords
            ),
        }
        weights = {
            "parseability": 0.20,
            "section_completeness": 0.25,
            "formatting_compatibility": 0.15,
            "supported_requirement_coverage": 0.40,
        }
        available_weight = sum(
            weight for name, weight in weights.items() if components[name] is not None
        )
        proxy_score = sum(
            components[name] * weight
            for name, weight in weights.items() if components[name] is not None
        ) / available_weight
        return {
            "ats_proxy_score": round(proxy_score, 6),
            "ats_components": {
                name: round(value, 6) if value is not None else None
                for name, value in components.items()
            },
            "details": {
                "sections_found": sections_found,
                "formatting_hazards": formatting_hazards,
                "keyword_count": len(normalised_keywords),
                "source_supported_keyword_count": len(supported_keywords),
                "composite_weights": weights,
                "visual_layout_assessed": False,
            },
        }


@dataclass(frozen=True)
class CallableATSProvider:
    """Injection adapter for tests or an application-owned external integration."""

    provider: str
    provider_version: str
    scorer: Callable[..., dict[str, Any]]
    score_kind: str = "ats_vendor_score"

    def identity(self) -> dict[str, Any]:
        if not self.provider or not self.provider_version:
            raise ValueError("provider and provider_version are required")
        return {
            "provider": self.provider,
            "provider_version": self.provider_version,
            "schema_version": ATS_SCHEMA_VERSION,
            "score_kind": self.score_kind,
        }

    def score(
        self,
        original_resume: str,
        job_description: str,
        tailored_resume: str,
        keywords: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.scorer(
            original_resume=original_resume,
            job_description=job_description,
            tailored_resume=tailored_resume,
            keywords=keywords,
        )


def _validate_payload(identity: dict[str, Any], payload: dict[str, Any]) -> None:
    score_kind = identity["score_kind"]
    score = payload.get(score_kind)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"provider response lacks numeric {score_kind}")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{score_kind} must be between 0 and 1")
    components = payload.get("ats_components", {})
    if not isinstance(components, dict):
        raise ValueError("ats_components must be an object")
    for name, value in components.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError(f"ATS component {name!r} must be null or between 0 and 1")


def evaluate_ats(
    provider: ATSProvider,
    original_resume: str,
    job_description: str,
    tailored_resume: str,
    keywords: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate once; failures remain explicitly unscored and never become zero."""
    identity = provider.identity()
    required_identity = {"provider", "provider_version", "schema_version", "score_kind"}
    if required_identity - identity.keys():
        raise ValueError("ATS provider identity is incomplete")
    request = {
        "input_hashes": {
            "original_resume": _text_hash(original_resume),
            "job_description": _text_hash(job_description),
            "tailored_resume": _text_hash(tailored_resume),
        },
        "keywords": _normalise_keywords(keywords),
        "provider": identity,
    }
    request_hash = _canonical_hash(request)
    started = time.perf_counter()
    try:
        payload = provider.score(
            original_resume,
            job_description,
            tailored_resume,
            keywords=request["keywords"],
        )
        if not isinstance(payload, dict):
            raise ValueError("provider response must be an object")
        _validate_payload(identity, payload)
        response_hash = _canonical_hash(payload)
        return {
            **payload,
            "ats_status": "scored",
            **identity,
            "ats_cache_key": request_hash,
            "ats_request_hash": request_hash,
            "ats_response_hash": response_hash,
            "ats_failure_reason": None,
            "ats_seconds": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        return {
            "ats_status": "unscored",
            **identity,
            "ats_cache_key": request_hash,
            "ats_request_hash": request_hash,
            "ats_response_hash": None,
            identity["score_kind"]: None,
            "ats_components": {},
            "ats_failure_reason": _safe_error(exc),
            "ats_seconds": round(time.perf_counter() - started, 6),
        }


def comparable_ats_records(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only records from the same versioned scoring contract are comparable."""
    fields = ("provider", "provider_version", "schema_version", "score_kind")
    return all(left.get(field) == right.get(field) for field in fields)
