"""Simple web page for the agentic baseline: upload a resume, give a job link (or paste the job ad),
get a tailored resume.

    python app.py      -> then open http://127.0.0.1:7860 in your browser

Later, add a dropdown here to pick W1-W5 so you can demo every workflow side by side.
"""
from pathlib import Path
from uuid import uuid4

import gradio as gr

import config
from common.llm import Tracker
from common.tools import ToolError, extract_resume_text, fetch_job_page
from workflows import WORKFLOWS
from workflows.w2_baseline import NAME as BASELINE_NAME, generate
from workflows.w3_critic import NAME as CRITIC_NAME, generate as generate_w3

WORKFLOW_CHOICES = [
    ("W1 - Single pass: one prompt, no guardrails", "w1_single_pass"),
    ("W2 - Agentic baseline: extract, validate and render", "w2_baseline"),
    ("W3 - Multi-critic: fact, recruiter and formatting review", "w3_critic"),
]

APP_CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 10% 0%, rgba(99, 102, 241, .10), transparent 30rem),
        radial-gradient(circle at 95% 10%, rgba(14, 165, 233, .08), transparent 26rem);
}
.hero {
    padding: 2.2rem 2rem;
    margin-bottom: 1.25rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(79, 70, 229, .13), rgba(14, 165, 233, .07));
}
.eyebrow {
    display: inline-block;
    padding: .35rem .7rem;
    margin-bottom: .75rem;
    border: 1px solid rgba(99, 102, 241, .28);
    border-radius: 999px;
    color: var(--body-text-color);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.hero h1 {
    margin: 0 0 .55rem;
    font-size: clamp(2rem, 5vw, 3.3rem);
    line-height: 1.02;
    letter-spacing: -.04em;
}
.hero p {
    max-width: 720px;
    margin: 0;
    color: var(--body-text-color-subdued);
    font-size: 1.02rem;
    line-height: 1.65;
}
.privacy-note {
    display: flex;
    gap: .45rem;
    align-items: center;
    margin-top: 1rem;
    color: var(--body-text-color-subdued);
    font-size: .86rem;
}
.panel {
    padding: 1.15rem !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 18px !important;
    background: var(--block-background-fill) !important;
    box-shadow: 0 12px 35px rgba(15, 23, 42, .06);
}
.panel-heading h2 {
    margin: 0 0 .15rem;
    font-size: 1.2rem;
}
.panel-heading p {
    margin: 0 0 .8rem;
    color: var(--body-text-color-subdued);
    font-size: .9rem;
}
.generate-btn {
    min-height: 48px !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
.empty-state {
    min-height: 455px;
    display: grid;
    place-items: center;
    padding: 2rem;
    border: 1px dashed var(--border-color-primary);
    border-radius: 14px;
    color: var(--body-text-color-subdued);
    text-align: center;
}
.resume-preview {
    min-height: 455px;
    padding: 1.5rem !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    background: var(--background-fill-primary);
}
.resume-preview h1 {
    padding-bottom: .6rem;
    border-bottom: 2px solid var(--border-color-primary);
}
.status-card {
    margin-top: .8rem;
    padding: .75rem 1rem;
    border-radius: 12px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued);
    font-size: .9rem;
}
.how-it-works {
    margin-top: 1.15rem;
}
.steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
}
.step {
    padding: .9rem;
    border: 1px solid var(--border-color-primary);
    border-radius: 13px;
    background: var(--block-background-fill);
}
.step b {
    display: block;
    margin-bottom: .25rem;
}
.step span {
    color: var(--body-text-color-subdued);
    font-size: .84rem;
}
@media (max-width: 760px) {
    .hero { padding: 1.5rem 1.2rem; }
    .steps { grid-template-columns: 1fr 1fr; }
}
"""

MESSAGES = {
    "JD_FETCH_FAILED": "Couldn't read that job page. Please paste the job description in the box below.",
    "JD_MISSING": "Please give a job link or paste the job description.",
    "SCANNED_PDF": "This looks like a scanned PDF with no text. Please upload a text-based PDF, .docx or .txt.",
    "EMPTY_RESUME": "Couldn't read any text from that resume.",
    "UNSUPPORTED_FILE": "Please upload a .pdf, .docx, .txt or .md resume.",
    "TOOL_BUDGET_EXCEEDED": "The agent couldn't produce a resume that passed validation within its tool-call limit.",
    "W3_FINAL_VALIDATION_FAILED": "The critics finished, but the final draft failed the factual validation harness.",
    "W3_STRUCTURED_OUTPUT_FAILED": "A critic or creator returned an invalid structured response.",
}


def _explain(errors):
    lines = []
    for err in errors:
        code = err.split(":", 1)[0]
        lines.append(f"- **{MESSAGES.get(code, code)}**  \n  `{err}`")
    return "\n".join(lines)


def _save_markdown(markdown, workflow):
    """Give non-baseline workflows a safe download file; the baseline saves through its harness."""
    output_dir = Path(config.OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ui_{workflow}_{uuid4().hex}.md"
    with open(path, "x", encoding="utf-8") as f:
        f.write(markdown)
    return str(path)


def _run_selected_workflow(workflow, path, job_url, pasted_jd, tracker):
    if workflow == BASELINE_NAME:
        return generate(tracker, path, job_url=job_url, pasted_jd=pasted_jd)
    if workflow == CRITIC_NAME:
        return generate_w3(tracker, path, job_url=job_url, pasted_jd=pasted_jd)

    resume_text = extract_resume_text(path)
    jd_text = (pasted_jd or "").strip()
    if not jd_text:
        if not (job_url or "").strip():
            raise ToolError("JD_MISSING", "provide a job link or paste the job description")
        try:
            jd_text = fetch_job_page(job_url.strip())
        except ToolError as e:
            raise ToolError(e.code, f"{e}. Please paste the job description instead.") from e

    pair = {
        "pair_id": f"ui_{uuid4().hex[:8]}",
        "resume_path": path,
        "resume": resume_text,
        "jd": jd_text,
        "job_url": job_url or "",
        "keywords": [],
    }
    result = WORKFLOWS[workflow](pair, tracker)
    if result.get("errors") or not result.get("resume_md", "").strip():
        return {"errors": result.get("errors") or ["EMPTY_OUTPUT: the workflow produced no resume"],
                "hit_budget": result.get("hit_budget", False)}
    return {
        "resume_md": result["resume_md"],
        "output_path": _save_markdown(result["resume_md"], workflow),
        "errors": [],
    }


def run(workflow, resume_file, job_url, pasted_jd):
    if resume_file is None:
        return (
            "### Upload a resume to get started\nYour tailored resume will appear here.",
            "Please upload your resume.",
            gr.update(value=None, visible=False),
            gr.update(),
        )
    if not (job_url or "").strip() and not (pasted_jd or "").strip():
        return (
            "### Add a job description\nEnter a link or paste the full posting on the left.",
            "Please give a job link or paste the job description.",
            gr.update(value=None, visible=False),
            gr.update(),
        )

    tracker = Tracker()
    path = resume_file if isinstance(resume_file, str) else resume_file.name
    try:
        state = _run_selected_workflow(workflow, path, job_url, pasted_jd, tracker)
    except ToolError as e:
        state = {"errors": [str(e)]}
    t = tracker.totals()
    stats = (
        f"**{workflow.replace('_', ' ').title()}** &nbsp;·&nbsp; "
        f"completed in {t['seconds']}s &nbsp;·&nbsp; "
        f"{t['input_tokens'] + t['output_tokens']:,} tokens &nbsp;·&nbsp; "
        f"{t['tool_calls_ok']}/{t['tool_calls']} valid tool calls"
    )

    errors = state.get("errors") or []
    if errors:
        needs_paste = any(e.startswith(("JD_FETCH_FAILED", "JD_MISSING")) for e in errors)
        highlight = gr.update(label="Paste the job description here (the link didn't work)") if needs_paste else gr.update()
        return (
            "### We couldn't generate this resume\n" + _explain(errors),
            stats,
            gr.update(value=None, visible=False),
            highlight,
        )
    return (
        state["resume_md"],
        stats + (" &nbsp;·&nbsp; **Validation passed**" if workflow in (BASELINE_NAME, CRITIC_NAME)
                 else " &nbsp;·&nbsp; **Experimental workflow**"),
        gr.update(value=state["output_path"], visible=True),
        gr.update(),
    )


theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
)

with gr.Blocks(title="Resume Tailor · Agentic baseline") as demo:
    gr.HTML(
        """
        <section class="hero">
          <span class="eyebrow">Agentic resume builder</span>
          <h1>Tailor your resume.<br>Keep every claim true.</h1>
          <p>Match your experience to a role with parallel resume and job-analysis agents.
             A validation harness checks protected facts before your resume can be saved.</p>
          <div class="privacy-note">◉ Runs locally with Ollama &nbsp;·&nbsp; Your resume is not sent to a hosted API</div>
        </section>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["panel"]):
            gr.HTML(
                """<div class="panel-heading">
                     <h2>1. Add your information</h2>
                     <p>Choose a workflow, upload a resume, then add the target role.</p>
                   </div>"""
            )
            workflow_in = gr.Dropdown(
                choices=WORKFLOW_CHOICES,
                value=BASELINE_NAME,
                label="Workflow",
                info="Use the agentic baseline for validated output; the others are comparison workflows.",
            )
            resume_in = gr.File(
                label="Resume",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                file_count="single",
            )
            with gr.Tabs():
                with gr.Tab("Paste job description"):
                    jd_in = gr.Textbox(
                        label="Job description",
                        placeholder="Paste the full job title, responsibilities and qualifications…",
                        lines=11,
                    )
                with gr.Tab("Use a public link"):
                    url_in = gr.Textbox(
                        label="Job posting URL",
                        placeholder="https://company.com/jobs/role",
                        info="Some sites block automated retrieval. Paste the JD if the link cannot be read.",
                    )
            go = gr.Button("Generate tailored resume", variant="primary", elem_classes=["generate-btn"])
            gr.Markdown(
                "Text-based PDFs only. Scanned documents need OCR before upload.",
                elem_classes=["privacy-note"],
            )

        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.HTML(
                """<div class="panel-heading">
                     <h2>2. Review your tailored resume</h2>
                     <p>The download appears after validation and rendering succeed.</p>
                   </div>"""
            )
            out_md = gr.Markdown(
                "### Your tailored resume will appear here\n"
                "Upload a resume and add a job description to begin.",
                elem_classes=["resume-preview", "empty-state"],
            )
            stats = gr.Markdown(
                "Waiting for input",
                elem_classes=["status-card"],
            )
            download = gr.File(label="Download validated Markdown", visible=False)

    with gr.Accordion("How the agentic workflow works", open=False, elem_classes=["how-it-works"]):
        gr.HTML(
            """
            <div class="steps">
              <div class="step"><b>01 · Parse</b><span>Extract structured, checkable candidate facts.</span></div>
              <div class="step"><b>02 · Analyse</b><span>Read the JD and identify its requirements.</span></div>
              <div class="step"><b>03 · Tailor</b><span>Reorder and rewrite only supported experience.</span></div>
              <div class="step"><b>04 · Validate</b><span>Check protected details, then render and save.</span></div>
            </div>
            """
        )

    go.click(
        run,
        [workflow_in, resume_in, url_in, jd_in],
        [out_md, stats, download, jd_in],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.launch(theme=theme, css=APP_CSS)
