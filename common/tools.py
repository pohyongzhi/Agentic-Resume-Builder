"""Tools used by the extraction agents. Each one fails with a ToolError carrying a short code
(SCANNED_PDF, JD_FETCH_FAILED, ...) so the pipeline can stop with a clear message."""
import os

import config

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class ToolError(Exception):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def extract_resume_text(path):
    """Turns a resume file into plain text. Supports .pdf, .docx, .txt, .md"""
    if not path or not os.path.isfile(path):
        raise ToolError("RESUME_NOT_FOUND", f"no resume file at {path!r}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        pages = PdfReader(path).pages
        text = "\n".join(page.extract_text() or "" for page in pages).strip()
        if len(text) < config.PDF_MIN_CHARS_PER_PAGE * max(len(pages), 1):
            raise ToolError("SCANNED_PDF", "this looks like a scanned PDF with no text layer; "
                                           "please upload a text-based PDF, .docx or .txt")
        return text
    if ext == ".docx":
        from docx import Document
        text = "\n".join(p.text for p in Document(path).paragraphs).strip()
    elif ext in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    else:
        raise ToolError("UNSUPPORTED_FILE", f"{ext or 'this file type'} is not supported; use .pdf, .docx, .txt or .md")
    if not text:
        raise ToolError("EMPTY_RESUME", "no text could be read from the resume")
    return text


def fetch_job_page(url):
    """Fetches a job posting and returns its readable text (navigation, scripts and ads stripped)."""
    import httpx
    import trafilatura

    if not url.startswith(("http://", "https://")):
        raise ToolError("JD_FETCH_FAILED", "the job link must start with http:// or https://")
    try:
        resp = httpx.get(url, headers={"User-Agent": BROWSER_UA}, timeout=config.FETCH_TIMEOUT,
                         follow_redirects=True)
    except httpx.HTTPError as e:
        raise ToolError("JD_FETCH_FAILED", f"could not reach the page ({type(e).__name__})") from e
    if resp.status_code != 200:
        raise ToolError("JD_FETCH_FAILED", f"the page returned HTTP {resp.status_code}")
    text = (trafilatura.extract(resp.text) or "").strip()
    if len(text) < config.JD_MIN_CHARS:
        raise ToolError("JD_FETCH_FAILED", "the page had too little readable text "
                                           "(it may need a login or JavaScript)")
    return text
