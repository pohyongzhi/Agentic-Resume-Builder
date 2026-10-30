"""Local V1 web app for implemented resume-tailoring workflows.

    python app.py      -> then open http://127.0.0.1:7860 in your browser

"""
import tempfile

import gradio as gr

import config
from baseline_tailor import read_file
from common.llm import Tracker
from eval.metrics import keyword_coverage
from workflows import WORKFLOWS, WORKFLOW_SPECS


def run(resume_file, jd_text, keywords_text, workflow_id="one_pass"):
    if resume_file is None:
        return "Please upload your resume.", "", None
    if not (jd_text or "").strip():
        return "Please paste the job description.", "", None

    resume = read_file(resume_file if isinstance(resume_file, str) else resume_file.name)
    if not resume:
        return "Couldn't read text from that file. If it's a scanned PDF, upload .docx or .txt instead.", "", None

    if workflow_id not in WORKFLOWS:
        return "That workflow is not implemented yet.", "", None
    tracker = Tracker()
    output = WORKFLOWS[workflow_id]({"resume": resume, "jd": jd_text}, tracker)
    result, stats = output["resume_md"], tracker.totals()
    spec = WORKFLOW_SPECS[workflow_id]

    info = (f"**Workflow:** `{workflow_id}`  |  **Model:** `{spec.model}`  \n"
            f"**Tokens:** {stats['input_tokens']} in, {stats['output_tokens']} out  |  "
            f"**Time:** {stats['seconds']}s")
    kws = [k.strip() for k in (keywords_text or "").split(",") if k.strip()]
    if kws:  # optional: quick keyword check, e.g. "SQL, Power BI, Python"
        kw = [{"keyword": k, "synonyms": []} for k in kws]
        before, after = keyword_coverage(resume, kw), keyword_coverage(result, kw)
        info += (f"  \n**Keywords covered:** {len(before['hit'])}/{len(kws)} before → "
                 f"{len(after['hit'])}/{len(kws)} after"
                 + (f"  \nStill missing: {', '.join(after['missing'])}" if after["missing"] else ""))

    out = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    out.write(result)
    out.close()
    return result, info, out.name


IMPLEMENTED_CHOICES = [
    (spec.display_name, workflow_id)
    for workflow_id, spec in WORKFLOW_SPECS.items()
    if spec.status == "implemented"
]


with gr.Blocks(title=f"{config.PRODUCT_NAME} {config.PRODUCT_VERSION}") as demo:
    gr.Markdown(
        f"# {config.PRODUCT_NAME} v{config.PRODUCT_VERSION}\n"
        f"Experiment {config.EXPERIMENT_VERSION}: compare implemented local variants. "
        "Variant 1 One-pass is selected by default."
    )
    with gr.Row():
        with gr.Column():
            resume_in = gr.File(label="Your resume (.pdf, .docx, .txt, .md)",
                                file_types=[".pdf", ".docx", ".txt", ".md"])
            workflow_in = gr.Dropdown(
                label="Variant", choices=IMPLEMENTED_CHOICES, value="one_pass")
            jd_in = gr.Textbox(label="Job description (paste the full ad)", lines=14)
            kw_in = gr.Textbox(label="Optional: keywords to check, comma-separated (e.g. SQL, Power BI, Python)")
            go = gr.Button("Tailor my resume", variant="primary")
        with gr.Column():
            out_md = gr.Markdown(label="Tailored resume")
            stats = gr.Markdown()
            download = gr.File(label="Download (.md)")
    go.click(run, [resume_in, jd_in, kw_in, workflow_in], [out_md, stats, download])

if __name__ == "__main__":
    demo.launch()
