"""Simple web page for the baseline: upload a resume, paste a job ad, get a tailored resume.

    python app.py      -> then open http://127.0.0.1:7860 in your browser

Later, add a dropdown here to pick W1-W5 so you can demo every workflow side by side.
"""
import tempfile

import gradio as gr

import config
from baseline_tailor import read_file, tailor
from eval.metrics import keyword_coverage


def run(resume_file, jd_text, keywords_text):
    if resume_file is None:
        return "Please upload your resume.", "", None
    if not (jd_text or "").strip():
        return "Please paste the job description.", "", None

    resume = read_file(resume_file if isinstance(resume_file, str) else resume_file.name)
    if not resume:
        return "Couldn't read text from that file. If it's a scanned PDF, upload .docx or .txt instead.", "", None

    result, stats = tailor(resume, jd_text)

    info = (f"**Tokens:** {stats['input_tokens']} in, {stats['output_tokens']} out  |  "
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


with gr.Blocks(title="Resume Tailor - V4.1") as demo:
    gr.Markdown(
        f"# Resume Tailor (V4.1)\n"
        f"Local model: `{config.MODEL}`. One prompt, no tools, no external API."
    )
    with gr.Row():
        with gr.Column():
            resume_in = gr.File(label="Your resume (.pdf, .docx, .txt, .md)",
                                file_types=[".pdf", ".docx", ".txt", ".md"])
            jd_in = gr.Textbox(label="Job description (paste the full ad)", lines=14)
            kw_in = gr.Textbox(label="Optional: keywords to check, comma-separated (e.g. SQL, Power BI, Python)")
            go = gr.Button("Tailor my resume", variant="primary")
        with gr.Column():
            out_md = gr.Markdown(label="Tailored resume")
            stats = gr.Markdown()
            download = gr.File(label="Download (.md)")
    go.click(run, [resume_in, jd_in, kw_in], [out_md, stats, download])

if __name__ == "__main__":
    demo.launch()
