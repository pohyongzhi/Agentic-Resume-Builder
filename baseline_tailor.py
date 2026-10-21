"""SINGLE PASS (V0): the simplest possible resume tailor. One prompt, one model call, no tools, no LangGraph.
Used by workflows/single_pass.py as the lower bound. The agentic baseline is workflows/baseline.py.

Use it from the terminal:
    python baseline_tailor.py --resume my_resume.pdf --jd job.txt
    python baseline_tailor.py --resume my_resume.docx --jd job.txt --out tailored.md
"""
import argparse

from common.llm import Tracker, chat
from common.prompts import BASE_RULES
from common.tools import ToolError, extract_resume_text


def read_file(path):
    """Turns a resume or job file into plain text (.pdf, .docx, .txt, .md). Returns "" if unreadable."""
    try:
        return extract_resume_text(path)
    except ToolError:
        return ""


def tailor(resume_text, jd_text, tracker=None):
    """One model call: original resume + job description -> tailored resume (Markdown).
    Returns (tailored_resume, stats)."""
    tracker = tracker or Tracker()
    msg = chat([
        {"role": "system", "content": BASE_RULES},
        {"role": "user", "content": f"JOB DESCRIPTION:\n{jd_text}\n\n"
                                    f"CANDIDATE'S ORIGINAL RESUME:\n{resume_text}\n\n"
                                    f"Tailor the resume to this job."},
    ], tracker)
    return (msg.content or "").strip(), tracker.totals()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tailor a resume to a job description (baseline).")
    ap.add_argument("--resume", required=True, help="path to your resume (.pdf, .docx, .txt, .md)")
    ap.add_argument("--jd", required=True, help="path to the job description (.txt, .md, .pdf, .docx)")
    ap.add_argument("--out", default="tailored_resume.md", help="where to save the result")
    args = ap.parse_args()

    resume, jd = read_file(args.resume), read_file(args.jd)
    if not resume:
        raise SystemExit("Couldn't read any text from the resume. If it's a scanned PDF, save it as .docx or .txt.")
    print("Tailoring... (first run can take a minute while the model loads)")
    result, stats = tailor(resume, jd)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(result)
    print(result)
    print(f"\nSaved to {args.out} | tokens in {stats['input_tokens']}, out {stats['output_tokens']}, "
          f"{stats['seconds']}s")
