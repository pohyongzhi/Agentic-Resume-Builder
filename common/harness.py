"""The harness: code that decides which tools the resume creator may run and enforces their rules.

Rules enforced here (not in the prompt):
  - only validate_resume and render_resume can run
  - render_resume only renders the latest draft, and only if that exact draft passed validation
  - files are saved only inside OUTPUT_DIR, never over the original resume or an existing file
  - at most MAX_TOOL_CALLS tool calls, then the run stops with TOOL_BUDGET_EXCEEDED

validate_resume catches structural errors and changes to protected facts (employers, titles, dates,
degrees, institutions, skills). It does NOT prove every rewritten bullet is accurate.
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

import config
from common.agent import Tool
from common.schemas import CandidateData, GeneratedResume


def _norm(text):
    text = (text or "").lower().replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[^\w\s+#.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _in_text(term, raw_norm):
    term = _norm(term)
    return bool(term) and re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", raw_norm) is not None


# ---------- validate_resume ----------

def validate_resume(data, candidate):
    """Returns a list of error strings (empty = valid). `data` is a dict or JSON string,
    `candidate` is the parsed CandidateData (dict or model)."""
    if isinstance(candidate, dict):
        candidate = CandidateData.model_validate(candidate)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return [f"resume is not valid JSON: {e}"]
    try:
        resume = GeneratedResume.model_validate(data)
    except ValidationError as e:
        return [f"schema: {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]

    errors = []
    raw = _norm(candidate.raw_text)

    if not resume.name.strip():
        errors.append("name is required")
    elif candidate.name and _norm(resume.name) != _norm(candidate.name):
        errors.append(f"name '{resume.name}' does not match the original resume ('{candidate.name}')")
    if not (resume.contact.email or resume.contact.phone):
        errors.append("contact needs an email or phone number")
    for field in ("email", "phone"):
        given, original = getattr(resume.contact, field), getattr(candidate.contact, field)
        if given and _norm(given) != _norm(original):
            errors.append(f"contact {field} '{given}' does not match the original resume")
    if not (resume.experience or resume.education):
        errors.append("at least one experience or education entry is required")

    def all_in_raw(*terms):
        return bool(raw) and all(_in_text(t, raw) for t in terms if t.strip())

    by_role = {(_norm(x.employer), _norm(x.title)): x for x in candidate.experience}
    for x in resume.experience:
        label = f"experience '{x.title} at {x.employer}'"
        match = by_role.get((_norm(x.employer), _norm(x.title)))
        if match is None:
            # the parser may have missed the entry: accept it if every protected detail is in the original text
            if not all_in_raw(x.employer, x.title, x.start, x.end):
                errors.append(f"{label} is not in the original resume; copy employer, title and dates exactly")
            continue
        for field in ("start", "end"):
            if _norm(getattr(x, field)) != _norm(getattr(match, field)):
                errors.append(f"{label}: {field} date '{getattr(x, field)}' should be '{getattr(match, field)}'")
        if raw and not _in_text(x.employer, raw):
            errors.append(f"{label}: employer '{x.employer}' does not appear in the original resume text")

    by_school = {(_norm(e.institution), _norm(e.degree)): e for e in candidate.education}
    for e in resume.education:
        label = f"education '{e.degree}, {e.institution}'"
        match = by_school.get((_norm(e.institution), _norm(e.degree)))
        if match is None:
            if not all_in_raw(e.institution, e.degree, e.start, e.end):
                errors.append(f"{label} is not in the original resume; copy institution, degree and dates exactly")
            continue
        for field in ("start", "end"):
            if _norm(getattr(e, field)) != _norm(getattr(match, field)):
                errors.append(f"{label}: {field} date '{getattr(e, field)}' should be '{getattr(match, field)}'")
        if raw and not _in_text(e.institution, raw):
            errors.append(f"{label}: institution does not appear in the original resume text")

    known_skills = {_norm(s) for s in candidate.skills}
    for s in resume.skills:
        if _norm(s) not in known_skills and not _in_text(s, raw):
            errors.append(f"skill '{s}' is not in the original resume")
    known_certs = {_norm(c) for c in candidate.certifications}
    for c in resume.certifications:
        if _norm(c) not in known_certs and not _in_text(c, raw):
            errors.append(f"certification '{c}' is not in the original resume")
    known_projects = {_norm(p.name) for p in candidate.projects}
    for p in resume.projects:
        if _norm(p.name) not in known_projects and not _in_text(p.name, raw):
            errors.append(f"project '{p.name}' is not in the original resume")
    return errors


# ---------- render_resume ----------

def _dates(start, end):
    if start and end:
        return f" ({start} - {end})"
    return f" ({start or end})" if start or end else ""


def to_markdown(resume):
    """The one fixed template every resume is rendered with."""
    r = resume if isinstance(resume, GeneratedResume) else GeneratedResume.model_validate(resume)
    contact = " | ".join(v for v in [r.contact.email, r.contact.phone, r.contact.location, *r.contact.links] if v)
    lines = [f"# {r.name}"] + ([contact] if contact else [])
    if r.summary:
        lines += ["", "## Summary", r.summary]
    if r.experience:
        lines += ["", "## Experience"]
        for x in r.experience:
            lines += ["", f"### {x.title}, {x.employer}{_dates(x.start, x.end)}"] + [f"- {b}" for b in x.bullets]
    if r.education:
        lines += ["", "## Education"]
        for e in r.education:
            lines += ["", f"### {e.degree}, {e.institution}{_dates(e.start, e.end)}"] + [f"- {d}" for d in e.details]
    if r.projects:
        lines += ["", "## Projects"]
        for p in r.projects:
            lines += ["", f"### {p.name}"] + ([p.description] if p.description else []) + [f"- {b}" for b in p.bullets]
    if r.skills:
        lines += ["", "## Skills", ", ".join(r.skills)]
    if r.certifications:
        lines += ["", "## Certifications"] + [f"- {c}" for c in r.certifications]
    return "\n".join(lines) + "\n"


class PathNotAllowed(Exception):
    pass


def safe_output_path(output_dir, filename, resume_path=None):
    """Resolves output_dir/filename and refuses anything outside output_dir, the original resume,
    or an existing file."""
    out_dir = Path(output_dir).resolve()
    target = (out_dir / filename).resolve()
    if not target.is_relative_to(out_dir) or target == out_dir:
        raise PathNotAllowed(f"'{filename}' is outside the output folder")
    if resume_path and target == Path(resume_path).resolve():
        raise PathNotAllowed("refusing to overwrite the original resume")
    if target.exists():
        raise PathNotAllowed(f"refusing to overwrite existing file '{target.name}'")
    return target


def render_resume(resume, output_dir, filename, resume_path=None):
    """Renders a validated resume with the fixed template and saves it. Returns (markdown, path)."""
    md = to_markdown(resume)
    target = safe_output_path(output_dir, filename, resume_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as f:   # "x" fails if the file appeared in the meantime
        f.write(md)
    return md, str(target)


# ---------- the harness ----------

def inline_schema(model):
    """JSON schema with $refs expanded and titles dropped; tool parameters can't use $defs."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def expand(node):
        if isinstance(node, list):
            return [expand(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return expand(defs[node["$ref"].split("/")[-1]])
        out = {k: expand(v) for k, v in node.items() if k not in ("title", "default", "properties")}
        if "properties" in node:   # field names (which may be called "title") are kept
            out["properties"] = {name: expand(sub) for name, sub in node["properties"].items()}
        return out
    return expand(schema)


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class ToolHarness:
    ALLOWED = ("validate_resume", "render_resume")

    def __init__(self, candidate, resume_path=None, output_dir=None, max_calls=None, run_id=None):
        self.candidate = CandidateData.model_validate(candidate) if isinstance(candidate, dict) else candidate
        self.resume_path = resume_path
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.max_calls = max_calls or config.MAX_TOOL_CALLS
        self.run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id or "resume")
        self.calls = 0
        self.budget_exceeded = False
        self.draft = None            # latest draft the model submitted for validation
        self.validated_hash = None   # hash of the latest draft that passed validation
        self.last_errors = []
        self.rendered = None         # {"resume_md", "output_path", "generated_resume"}

    def execute(self, name, args):
        """Runs one tool call under the harness rules. Returns (ok, result_text, stop)."""
        if self.calls >= self.max_calls:
            self.budget_exceeded = True
            return False, f"TOOL_BUDGET_EXCEEDED: limit of {self.max_calls} tool calls reached.", True
        self.calls += 1
        if name not in self.ALLOWED:
            return False, f"Error: tool '{name}' is not allowed. Allowed tools: {list(self.ALLOWED)}.", False
        if name == "validate_resume":
            return self._validate(args)
        return self._render()

    def _validate(self, args):
        data = (args or {}).get("resume")
        if data is None:
            return False, "Error: missing argument 'resume'.", False
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                self.draft, self.last_errors = None, [f"resume is not valid JSON: {e}"]
                return False, json.dumps({"valid": False, "errors": self.last_errors}), False
        if isinstance(data, dict):
            # name and contact must equal the original anyway; fill them if the model left them out
            data = dict(data)
            if not data.get("name"):
                data["name"] = self.candidate.name
            if not data.get("contact"):
                data["contact"] = self.candidate.contact.model_dump()
        self.draft = data
        self.last_errors = validate_resume(data, self.candidate)
        if self.last_errors:
            self.validated_hash = None
            return True, json.dumps({"valid": False, "errors": self.last_errors}), False
        self.validated_hash = _hash(data)
        return True, json.dumps({"valid": True, "next": "call render_resume"}), False

    def _render(self):
        if self.draft is None or self.validated_hash is None or _hash(self.draft) != self.validated_hash:
            return False, "Error: render_resume requires the latest draft to pass validate_resume first.", False
        filename = f"{self.run_id}_{datetime.now():%Y%m%d-%H%M%S-%f}.md"
        try:
            md, path = render_resume(self.draft, self.output_dir, filename, self.resume_path)
        except PathNotAllowed as e:
            return False, f"Error: {e}", False
        self.rendered = {"resume_md": md, "output_path": path,
                         "generated_resume": GeneratedResume.model_validate(self.draft).model_dump()}
        return True, f"Rendered and saved to {path}.", True

    def outcome(self):
        """State update for the pipeline once the creator stops."""
        if self.rendered:
            return dict(self.rendered)
        last = f" Last validation errors: {self.last_errors}" if self.last_errors else ""
        # Reaching the limit without rendering is exhaustion even if the agent did not attempt
        # one extra call (which is where execute() normally flips budget_exceeded).
        if self.budget_exceeded or self.calls >= self.max_calls:
            self.budget_exceeded = True
            return {"errors": [f"TOOL_BUDGET_EXCEEDED: no valid resume after {self.max_calls} tool calls.{last}"],
                    "hit_budget": True}
        return {"errors": [f"CREATOR_DID_NOT_FINISH: the creator stopped without rendering a validated resume.{last}"]}

    def tools(self):
        """Schemas the model sees. Execution goes through execute(), not these functions."""
        def validate_resume(resume):  # noqa: ARG001 - signature only, run via execute()
            pass

        def render_resume():
            pass

        return [
            Tool(validate_resume,
                 "Check a draft resume against the candidate's original resume. Returns {valid, errors}. "
                 "Fix every error and validate again before rendering.",
                 {"resume": {**inline_schema(GeneratedResume),
                             "description": "The full draft resume as a JSON object"}}),
            Tool(render_resume,
                 "Render and save the most recently validated draft. Only works after validate_resume "
                 "returned valid=true for that exact draft. Call this once to finish.", {}),
        ]
