"""The two extraction agents. They run in parallel in the pipeline graph (common/pipeline.py).

Each agent calls its tool in code (small local models are unreliable at deciding when to call a
tool), then asks the model to turn the text into a fixed schema using Ollama's `format=`.
Each node writes only its own state field, plus `errors` on failure."""
import re

from pydantic import ValidationError

from common.llm import chat
from common.schemas import CandidateData, JobData, llm_schema
from common.tools import ToolError, extract_resume_text, fetch_job_page

PARSER_PROMPT = """Extract the candidate's information from this resume into the JSON schema.
Capture every section: summary, every job, every education entry (degree and institution, even if
they are on one line), projects, skills, certifications and achievements.
Copy names, employers, job titles, dates, degrees, institutions and skills exactly as written.
Do not infer, summarise away, or add anything that is not in the text. Leave a field empty if it is missing."""

JD_PROMPT = """Extract this job posting into the JSON schema: job title, company, responsibilities,
required qualifications and preferred ("nice to have", "a plus") qualifications.
Use only what the posting says. Leave a field empty if it is missing."""


def _structured(prompt, text, schema_model, tracker, exclude=()):
    msg = chat([{"role": "system", "content": prompt}, {"role": "user", "content": text}],
               tracker, format=llm_schema(schema_model, exclude))
    return schema_model.model_validate_json(msg.content or "")


def _recover_contact(candidate, raw_text):
    """Recover obvious protected contact fields that a small parser model omitted."""
    if not candidate.contact.email:
        email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", raw_text)
        if email:
            candidate.contact.email = email.group(0)
    if not candidate.contact.phone:
        for match in re.finditer(r"(?<!\w)\+?[\d() .-]{7,}\d(?!\w)", raw_text):
            value = match.group(0).strip()
            digits = re.sub(r"\D", "", value)
            if 7 <= len(digits) <= 15:
                candidate.contact.phone = value
                break


def make_resume_parser(tracker):
    def resume_parser(state):
        try:
            raw_text = extract_resume_text(state.get("resume_path", ""))
        except ToolError as e:
            return {"errors": [str(e)]}
        try:
            candidate = _structured(PARSER_PROMPT, raw_text, CandidateData, tracker, exclude=("raw_text",))
        except ValidationError as e:
            return {"errors": [f"RESUME_PARSE_FAILED: the model's output did not match the schema ({e.error_count()} errors)"]}
        _recover_contact(candidate, raw_text)
        candidate.raw_text = raw_text
        return {"candidate_data": candidate.model_dump()}
    return resume_parser


def make_jd_retriever(tracker):
    def jd_retriever(state):
        text = (state.get("pasted_jd") or "").strip()
        if not text:
            url = (state.get("job_url") or "").strip()
            if not url:
                return {"errors": ["JD_MISSING: provide a job link or paste the job description"]}
            try:
                text = fetch_job_page(url)
            except ToolError as e:
                return {"errors": [f"{e}. Please paste the job description instead."]}
        try:
            job = _structured(JD_PROMPT, text, JobData, tracker, exclude=("source_text",))
        except ValidationError as e:
            return {"errors": [f"JD_PARSE_FAILED: the model's output did not match the schema ({e.error_count()} errors)"]}
        job.source_text = text
        return {"job_data": job.model_dump()}
    return jd_retriever
