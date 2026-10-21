from types import SimpleNamespace

import pytest

from common.tools import ToolError, extract_resume_text, fetch_job_page


def test_scanned_pdf_detected(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    path = tmp_path / "scan.pdf"
    with open(path, "wb") as f:
        w.write(f)
    with pytest.raises(ToolError) as e:
        extract_resume_text(str(path))
    assert e.value.code == "SCANNED_PDF"


def test_markdown_resume_read(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("# Alex Tan\nalex@example.com")
    assert extract_resume_text(str(p)).startswith("# Alex Tan")


@pytest.mark.parametrize("name,code", [("r.png", "UNSUPPORTED_FILE"), ("missing.md", "RESUME_NOT_FOUND")])
def test_bad_resume_files(tmp_path, name, code):
    if name.endswith(".png"):
        (tmp_path / name).write_bytes(b"\x89PNG")
    with pytest.raises(ToolError) as e:
        extract_resume_text(str(tmp_path / name))
    assert e.value.code == code


def _fake_get(status, html):
    return lambda *a, **k: SimpleNamespace(status_code=status, text=html)


def test_fetch_non_200(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", _fake_get(403, ""))
    with pytest.raises(ToolError) as e:
        fetch_job_page("https://example.com/job")
    assert e.value.code == "JD_FETCH_FAILED" and "403" in str(e.value)


def test_fetch_too_little_text(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", _fake_get(200, "<html><body><p>Loading...</p></body></html>"))
    with pytest.raises(ToolError) as e:
        fetch_job_page("https://example.com/job")
    assert e.value.code == "JD_FETCH_FAILED"


def test_fetch_ok(monkeypatch):
    import httpx
    body = " ".join(["We are hiring a Junior Data Analyst to build dashboards and write SQL."] * 8)
    html = f"<html><body><article><h1>Junior Data Analyst</h1><p>{body}</p></article></body></html>"
    monkeypatch.setattr(httpx, "get", _fake_get(200, html))
    assert "Junior Data Analyst" in fetch_job_page("https://example.com/job")


def test_fetch_rejects_non_http():
    with pytest.raises(ToolError):
        fetch_job_page("file:///etc/passwd")
