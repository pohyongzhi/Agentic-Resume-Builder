"""Prompt text shared by ALL workflows. Do not change per workflow: if every workflow
starts from the same instructions, any difference in results comes from the technique."""

BASE_RULES = """You tailor a candidate's resume to a specific job description.
Rules:
- Only use facts that are in the candidate's original resume. Never invent skills, tools, numbers, titles, dates or employers.
- You may reword, reorder, shorten, and emphasise what is relevant to the job.
- Output a complete resume in Markdown with sections: Summary, Experience, Education, Skills.
- Keep it to roughly one page (about 350-500 words)."""


def task_prompt(pair):
    return f"""JOB DESCRIPTION:
{pair['jd']}

CANDIDATE'S ORIGINAL RESUME:
{pair['resume']}

Tailor the resume to this job."""
