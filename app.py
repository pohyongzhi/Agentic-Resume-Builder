"""Simple web page: upload a resume, paste a job ad, pick a workflow, get a tailored resume.

    python app.py      -> then open http://127.0.0.1:7860 in your browser

Every workflow registered in workflows/__init__.py shows up in the dropdown.
"""
import tempfile

import gradio as gr

from baseline_tailor import read_file
from common.llm import Tracker
from eval.metrics import keyword_coverage
from workflows import WORKFLOWS


def _w7_details(events):
    """W7 logs its plan and consistency checks into the Tracker; show them to explain its output."""
    lines = []
    for e in events:
        if e["type"] == "w7_plan":
            sections = e.get("sections", [])
            if not e["ok"]:
                lines.append("**Plan:** JSON failed twice; writers used the default sections: " +
                             ", ".join(s["name"] for s in sections))
                continue
            reqs = e.get("requirements", [])
            lines.append("**Plan sections:** " + " → ".join(s["name"] for s in sections) + "\n" +
                         "\n".join(f"- **{s['name']}**: lead with " + ("; ".join(s["lead_with"]) or "-") +
                                   " | cut " + ("; ".join(s["cut"]) or "-") for s in sections))
            lines.append(f"**Plan mapping:** {e.get('supported', 0)}/{len(reqs)} requirements backed by resume evidence\n" +
                         "\n".join(f"- {r['requirement']} → " +
                                   (f"*\"{r['evidence']}\"* ({r['section']})" if r["evidence"] else "**unsupported**")
                                   for r in reqs))
        elif e["type"] == "w7_check":
            label = ("Consistency before assembler" if e.get("stage") == "before_assembler"
                     else "Consistency after assembler")
            lines.append(f"**{label}:** " + (f"{len(e['problems'])} issue(s)\n" + "\n".join(f"- {p}" for p in e["problems"])
                                             if e["problems"] else "no issues"))
    return "\n\n".join(lines)


def run(resume_file, jd_text, keywords_text, workflow):
    if resume_file is None:
        return "Please upload your resume.", "", None
    if not (jd_text or "").strip():
        return "Please paste the job description.", "", None

    resume = read_file(resume_file if isinstance(resume_file, str) else resume_file.name)
    if not resume:
        return "Couldn't read text from that file. If it's a scanned PDF, upload .docx or .txt instead.", "", None

    tracker = Tracker()
    out = WORKFLOWS[workflow]({"resume": resume, "jd": jd_text}, tracker)
    result, stats = out["resume_md"].strip(), tracker.totals()

    info = (f"**Workflow:** {workflow}  |  **Model calls:** {stats['llm_calls']}  |  "
            f"**Tokens:** {stats['input_tokens']} in, {stats['output_tokens']} out  |  "
            f"**Time:** {stats['seconds']}s")
    if not out["finished"]:
        info += "  \n**Note:** the workflow did not finish cleanly; this is its last draft."
    kws = [k.strip() for k in (keywords_text or "").split(",") if k.strip()]
    if kws:  # optional: quick keyword check, e.g. "SQL, Power BI, Python"
        kw = [{"keyword": k, "synonyms": []} for k in kws]
        before, after = keyword_coverage(resume, kw), keyword_coverage(result, kw)
        info += (f"  \n**Keywords covered:** {len(before['hit'])}/{len(kws)} before → "
                 f"{len(after['hit'])}/{len(kws)} after"
                 + (f"  \nStill missing: {', '.join(after['missing'])}" if after["missing"] else ""))
    details = _w7_details(tracker.events)
    if details:
        info += "\n\n" + details

    file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
    file.write(result)
    file.close()
    return result, info, file.name


with gr.Blocks(title="Resume Tailor") as demo:
    gr.Markdown("# Resume Tailor\nPick a workflow, upload a resume and paste a job ad. Runs on the local model.")
    with gr.Row():
        with gr.Column():
            workflow_in = gr.Dropdown(label="Workflow", choices=list(WORKFLOWS), value="w7_plan_execute"
                                      if "w7_plan_execute" in WORKFLOWS else list(WORKFLOWS)[0])
            resume_in = gr.File(label="Your resume (.pdf, .docx, .txt, .md)",
                                file_types=[".pdf", ".docx", ".txt", ".md"])
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
