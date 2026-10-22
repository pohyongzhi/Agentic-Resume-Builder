"""Prompt text shared by ALL workflows. Do not change per workflow: if every workflow
starts from the same instructions, any difference in results comes from the technique."""
import json

BASE_RULES = """You tailor a candidate's resume to a specific job description.
Rules:
- Only use facts that are in the candidate's original resume. Never invent skills, tools, numbers, titles, dates or employers.
- You may reword, reorder, shorten, and emphasise what is relevant to the job.
- Output a complete resume in Markdown with sections: Summary, Experience, Education, Skills.
- Keep it to roughly one page (about 350-500 words)."""


CREATOR_SYSTEM = BASE_RULES + """

You work inside a harness. You do not write Markdown yourself: you write the resume as a JSON object
and the harness renders it with a fixed template that has the sections above.
- Copy employer names, job titles, dates, degrees and institutions EXACTLY as they appear in the
  candidate data. Only bullets and the summary may be reworded.
- Only list skills and certifications the candidate already has. Do not add a qualification just
  because the job asks for it; leave it out instead.
- You may reorder experience and bullets, and drop entries that are irrelevant to the job.
Steps:
1. Call validate_resume with your full draft.
2. If it returns errors, fix every one and call validate_resume again.
3. When it returns valid=true, call render_resume to finish. You have a small, fixed number of tool calls."""

STRUCTURED_CREATOR_SYSTEM = BASE_RULES + """

Return the complete tailored resume as one JSON object matching the supplied schema.
- Copy employer names, job titles, dates, degrees and institutions EXACTLY as they appear in the
  candidate data. Only bullets and the summary may be reworded.
- Only list skills and certifications the candidate already has. Never add a qualification merely
  because the job asks for it.
- You may reorder experience and bullets, and drop entries that are irrelevant to the job.
The application will validate your JSON and render it; do not include Markdown or commentary."""


def creator_prompt(candidate, job):
    """candidate / job are the dicts produced by the extraction agents."""
    raw_text = candidate.get("raw_text", "")
    data = {k: v for k, v in candidate.items() if k != "raw_text"}
    job_fields = {k: v for k, v in job.items() if k != "source_text"}
    return f"""JOB (extracted):
{json.dumps(job_fields, indent=2, ensure_ascii=False)}

CANDIDATE DATA (extracted from the resume):
{json.dumps(data, indent=2, ensure_ascii=False)}

CANDIDATE'S ORIGINAL RESUME TEXT (for checking facts):
{raw_text}

Tailor the resume to this job, then validate and render it."""


def task_prompt(pair):
    return f"""JOB DESCRIPTION:
{pair['jd']}

CANDIDATE'S ORIGINAL RESUME:
{pair['resume']}

Tailor the resume to this job."""
