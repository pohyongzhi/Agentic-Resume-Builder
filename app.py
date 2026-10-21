"""Simple web page for the agentic baseline: upload a resume, give a job link (or paste the job ad),
get a tailored resume.

    python app.py      -> then open http://127.0.0.1:7860 in your browser

Later, add a dropdown here to pick W1-W5 so you can demo every workflow side by side.
"""
import gradio as gr

from common.llm import Tracker
from workflows.baseline import generate

MESSAGES = {
    "JD_FETCH_FAILED": "Couldn't read that job page. Please paste the job description in the box below.",
    "JD_MISSING": "Please give a job link or paste the job description.",
    "SCANNED_PDF": "This looks like a scanned PDF with no text. Please upload a text-based PDF, .docx or .txt.",
    "EMPTY_RESUME": "Couldn't read any text from that resume.",
    "UNSUPPORTED_FILE": "Please upload a .pdf, .docx, .txt or .md resume.",
    "TOOL_BUDGET_EXCEEDED": "The agent couldn't produce a resume that passed validation within its tool-call limit.",
}


def _explain(errors):
    lines = []
    for err in errors:
        code = err.split(":", 1)[0]
        lines.append(f"- **{MESSAGES.get(code, code)}**  \n  `{err}`")
    return "\n".join(lines)


def run(resume_file, job_url, pasted_jd):
    if resume_file is None:
        return "Please upload your resume.", "", None, gr.update()
    if not (job_url or "").strip() and not (pasted_jd or "").strip():
        return "Please give a job link or paste the job description.", "", None, gr.update()

    tracker = Tracker()
    path = resume_file if isinstance(resume_file, str) else resume_file.name
    state = generate(tracker, path, job_url=job_url, pasted_jd=pasted_jd)
    t = tracker.totals()
    stats = (f"**Tokens:** {t['input_tokens']} in, {t['output_tokens']} out  |  **Time:** {t['seconds']}s  |  "
             f"**Tool calls:** {t['tool_calls_ok']}/{t['tool_calls']} valid")

    errors = state.get("errors") or []
    if errors:
        needs_paste = any(e.startswith(("JD_FETCH_FAILED", "JD_MISSING")) for e in errors)
        highlight = gr.update(label="Paste the job description here (the link didn't work)") if needs_paste else gr.update()
        return "### Couldn't generate a resume\n" + _explain(errors), stats, None, highlight
    return state["resume_md"], stats + f"  \nSaved to `{state['output_path']}`", state["output_path"], gr.update()


with gr.Blocks(title="Resume Tailor - Agentic baseline") as demo:
    gr.Markdown("# Resume Tailor (agentic baseline)\n"
                "Resume parser + job description agents run in parallel, then a creator agent writes the resume "
                "through a harness that validates facts before saving.")
    with gr.Row():
        with gr.Column():
            resume_in = gr.File(label="Your resume (.pdf, .docx, .txt, .md)",
                                file_types=[".pdf", ".docx", ".txt", ".md"])
            url_in = gr.Textbox(label="Job posting link")
            jd_in = gr.Textbox(label="Or paste the job description (used instead of the link if filled)", lines=12)
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            out_md = gr.Markdown(label="Tailored resume")
            stats = gr.Markdown()
            download = gr.File(label="Download (.md)")
    go.click(run, [resume_in, url_in, jd_in], [out_md, stats, download, jd_in])

if __name__ == "__main__":
    demo.launch()
