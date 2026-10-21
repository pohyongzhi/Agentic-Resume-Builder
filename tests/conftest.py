import copy

import pytest

RAW_TEXT = """# Alex Tan
alex.tan@example.com

## Experience
**Business Analyst Intern, Example Logistics Pte Ltd** (May 2025 - Aug 2025)
- Built weekly sales dashboards in Excel and Power BI used by the regional sales team.
- Cleaned and merged shipment data from three spreadsheets using SQL queries.

## Education
Bachelor of Science (Information Systems), Example University, expected 2026

## Skills
Excel, Power BI, SQL (basic), Python (basic), process mapping, stakeholder communication
"""

CANDIDATE = {
    "name": "Alex Tan",
    "contact": {"email": "alex.tan@example.com", "phone": "", "location": "", "links": []},
    "summary": "Final-year Information Systems student.",
    "education": [{"institution": "Example University", "degree": "Bachelor of Science (Information Systems)",
                   "start": "", "end": "expected 2026", "details": []}],
    "experience": [{"employer": "Example Logistics Pte Ltd", "title": "Business Analyst Intern",
                    "start": "May 2025", "end": "Aug 2025",
                    "bullets": ["Built weekly sales dashboards in Excel and Power BI."]}],
    "projects": [],
    "skills": ["Excel", "Power BI", "SQL (basic)", "Python (basic)", "process mapping",
               "stakeholder communication"],
    "certifications": [],
    "achievements": [],
    "raw_text": RAW_TEXT,
}

GOOD_RESUME = {
    "name": "Alex Tan",
    "contact": {"email": "alex.tan@example.com"},
    "summary": "Information Systems student who builds sales dashboards and cleans data with SQL.",
    "experience": [{"employer": "Example Logistics Pte Ltd", "title": "Business Analyst Intern",
                    "start": "May 2025", "end": "Aug 2025",
                    "bullets": ["Built weekly Power BI sales dashboards for the regional sales team."]}],
    "education": [{"institution": "Example University", "degree": "Bachelor of Science (Information Systems)",
                   "end": "expected 2026"}],
    "skills": ["SQL", "Power BI", "Excel", "Python"],
}


@pytest.fixture
def candidate():
    return copy.deepcopy(CANDIDATE)


@pytest.fixture
def good_resume():
    return copy.deepcopy(GOOD_RESUME)
