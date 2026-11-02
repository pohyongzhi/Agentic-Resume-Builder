"""Deterministically materialise the 100-pair synthetic evaluation dataset.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --force
"""
import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPEC_PATH = DATA / "generation" / "spec.json"

PREFIXES = {
    "software_engineering": "se",
    "data_ai": "da",
    "cybersecurity": "cy",
    "cloud_devops": "cd",
    "business_technology": "bt",
}

COMPANIES = [
    "Example Meridian Labs",
    "Example Northstar Systems",
    "Example Harbour Digital",
    "Example Orchard Technologies",
]
EMPLOYERS = [
    "Example Atlas Solutions",
    "Example Beacon Services",
    "Example Cedar Innovations",
    "Example Delta Operations",
    "Example Ember Technologies",
]
SCHOOLS = [
    "Example Institute of Technology",
    "Example Metropolitan University",
    "Example Polytechnic",
    "Example School of Computing",
]
FIRST_NAMES = [
    "Aisha", "Benjamin", "Chloe", "Daniel", "Elena", "Farhan", "Grace", "Hiro",
    "Inez", "Jonas", "Kavya", "Leon", "Mei", "Noah", "Olivia", "Priya",
    "Quentin", "Rina", "Samuel", "Talia",
]
LAST_NAMES = ["Tan", "Lim", "Ng", "Lee", "Koh"]

SYNONYMS = {
    "CI/CD": ["continuous integration", "continuous delivery"],
    "REST APIs": ["RESTful APIs", "API development"],
    "large language models": ["LLMs", "language models"],
    "RAG": ["retrieval-augmented generation", "retrieval augmented generation"],
    "Power BI": ["business intelligence dashboards"],
    "data cleaning": ["data cleansing", "cleaned data"],
    "dashboarding": ["dashboards", "data visualisation"],
    "incident response": ["incident handling"],
    "stakeholder communication": ["stakeholder management"],
    "unit testing": ["automated tests"],
    "requirements gathering": ["requirements analysis"],
    "process mapping": ["business process modelling"],
    "machine learning": ["ML"],
    "service level objectives": ["SLOs"],
    "A/B testing": ["experimentation"],
}


def _write(path, text, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; use --force to regenerate")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _jd_markdown(role, company):
    responsibilities = "\n".join(f"- {x.capitalize()}." for x in role["responsibilities"])
    required = "\n".join(f"- {x}" for x in role["skills"])
    preferred = "\n".join(f"- {x}" for x in role["preferred"])
    return f"""# {role['title']}

**Company:** {company} (fictional)  
**Location:** Singapore  
**Employment type:** Full-time  
**Seniority:** {role['seniority'].title()}

## About the role

{company} is seeking a {role['title']} to join a multidisciplinary technology team. The successful
candidate will turn operational and customer needs into reliable, maintainable solutions. This role
works closely with engineering, product and business stakeholders, values clear documentation and
expects evidence-based decisions. The position offers ownership of meaningful work while maintaining
appropriate review, security and quality controls.

## Responsibilities

{responsibilities}
- Communicate progress, risks and technical trade-offs to relevant stakeholders.
- Maintain clear documentation and contribute to team planning and reviews.

## Required qualifications

{required}
- Clear written and verbal communication
- Ability to work collaboratively and solve ambiguous problems

## Preferred qualifications

{preferred}

## What success looks like

In the first six months, the successful candidate will understand the existing environment, deliver
well-tested improvements and build trusted working relationships. Success is measured through
quality, reliability, responsible delivery and useful outcomes rather than activity alone.
"""


def _candidate_name(index):
    first = FIRST_NAMES[(index - 1) % len(FIRST_NAMES)]
    last = LAST_NAMES[(index - 1) // len(FIRST_NAMES)]
    return f"{first} {last}"


def _fit_skills(role, fit):
    if fit == "strong":
        return role["skills"] + role["preferred"][:2]
    if fit == "moderate":
        return role["skills"][:4] + role["preferred"][:1] + ["Excel", "documentation", "Git"]
    return role["skills"][:2] + ["Excel", "documentation", "stakeholder communication"]


def _dates(seniority, variant):
    if seniority == "senior":
        return ("Jan 2021", "Present", "Jun 2017", "Dec 2020")
    if seniority == "mid":
        return ("Jul 2023", "Present", "Jan 2021", "Jun 2023")
    return ("Jan 2025", "Present", "May 2024", "Dec 2024")


def _resume_markdown(role, fit, candidate_index, variant):
    name = _candidate_name(candidate_index)
    skills = _fit_skills(role, fit)
    current_start, current_end, prior_start, prior_end = _dates(role["seniority"], variant)
    employer = EMPLOYERS[(candidate_index - 1) % len(EMPLOYERS)]
    prior_employer = EMPLOYERS[candidate_index % len(EMPLOYERS)]
    school = SCHOOLS[(candidate_index - 1) % len(SCHOOLS)]
    metric = 12 + (candidate_index * 7) % 31
    volume = 20 + (candidate_index * 13) % 90
    skill_a, skill_b = skills[0], skills[1]
    skill_c = skills[2] if len(skills) > 2 else "documentation"
    skill_d = skills[3] if len(skills) > 3 else "Excel"

    if fit == "strong":
        current_title = role["title"]
        summary = (
            f"{role['title']} with hands-on experience in {skill_a}, {skill_b} and {skill_c}. "
            "Delivers documented, measurable improvements with cross-functional teams."
        )
    elif fit == "moderate":
        current_title = "Technology Associate"
        summary = (
            f"Technology associate with transferable experience in {skill_a}, {skill_b} and "
            f"{skill_c}, seeking to deepen capability as a {role['title']}."
        )
    else:
        current_title = "Operations Support Coordinator"
        summary = (
            f"Operations support professional experienced in {skill_a}, {skill_b}, documentation "
            f"and stakeholder coordination, transitioning toward {role['title']} work."
        )

    cert = ""
    if "AWS" in skills:
        cert = "\n## Certifications\n\n- AWS Certified Cloud Practitioner\n"
    elif "Security+" in skills:
        cert = "\n## Certifications\n\n- CompTIA Security+\n"

    return f"""# {name}

{name.lower().replace(' ', '.')}@example.test | +65 6000 {candidate_index:04d} | Singapore

## Summary

{summary}

## Experience

### {current_title}, {employer} (fictional) ({current_start} - {current_end})

- Used {skill_a} and {skill_b} to {role['responsibilities'][0]}, improving team turnaround by {metric}%.
- Applied {skill_c} to {role['responsibilities'][1]} and documented decisions for technical and business reviewers.
- Managed approximately {volume} work items per month while tracking quality checks and follow-up actions.
- Collaborated with four colleagues to {role['responsibilities'][2]} and presented progress during fortnightly reviews.

### Technical Assistant, {prior_employer} (fictional) ({prior_start} - {prior_end})

- Supported recurring operational work using {skill_d} and documented step-by-step procedures.
- Investigated service requests, recorded findings and escalated issues with clear supporting evidence.
- Created a simple reporting template that reduced manual preparation time by {max(metric - 5, 5)}%.

## Projects

### {role['title']} Practice Project

- Built a fictional demonstration project using {skill_a}, {skill_b} and {skill_c}.
- Tested the project with representative scenarios and recorded limitations and next steps.
- Presented the design and outcomes to a five-person peer review group.

## Education

### Bachelor of Science in Information Systems, {school} (fictional) (Aug 2020 - May 2024)

- Coursework included software engineering, databases, cybersecurity and applied analytics.

## Skills

{', '.join(dict.fromkeys(skills))}
{cert}"""


def _dataset_hash(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(DATA))):
        digest.update(str(path.relative_to(DATA)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generate(force=False):
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    rows, hash_paths = [], []
    candidate_index = 0
    category_counts = {}

    for category in spec["categories"]:
        prefix = PREFIXES[category["id"]]
        category_counts[category["id"]] = 0
        for role_number, role in enumerate(category["roles"], 1):
            jd_id = f"{prefix}_jd_{role_number:02d}"
            company = COMPANIES[(role_number - 1) % len(COMPANIES)]
            jd_path = DATA / "jds" / f"{jd_id}.md"
            keywords_path = DATA / "jds" / f"{jd_id}.keywords.json"
            jd_md = _jd_markdown(role, company)
            keywords = [
                {"keyword": skill, "synonyms": SYNONYMS.get(skill, [])}
                for skill in role["skills"]
            ]
            _write(jd_path, jd_md, force)
            _write(keywords_path, json.dumps(keywords, indent=2) + "\n", force)
            hash_paths += [jd_path, keywords_path]

            raw_resumes = []
            for variant, fit in enumerate(spec["fit_pattern"], 1):
                candidate_index += 1
                resume_id = f"{prefix}_resume_{(role_number - 1) * 5 + variant:03d}"
                pair_id = f"{resume_id}__{jd_id}"
                resume_md = _resume_markdown(role, fit, candidate_index, variant)
                resume_path = DATA / "resumes" / f"{resume_id}.md"
                _write(resume_path, resume_md, force)
                hash_paths.append(resume_path)
                rows.append({
                    "pair_id": pair_id,
                    "resume_id": resume_id,
                    "jd_id": jd_id,
                    "category": category["id"],
                    "role_family": role["id"],
                    "seniority": role["seniority"],
                    "fit_level": fit,
                    "split": "test",
                    "source": "synthetic_template_v1",
                    "jd_url": "",
                })
                category_counts[category["id"]] += 1
                raw_resumes.append({
                    "resume_id": resume_id,
                    "fit_level": fit,
                    "markdown": resume_md,
                })

            raw = {
                "category": category["id"],
                "jd_id": jd_id,
                "role": role,
                "jd_markdown": jd_md,
                "keywords": keywords,
                "resumes": raw_resumes,
            }
            raw_path = DATA / "generation" / "raw" / f"{jd_id}.json"
            _write(raw_path, json.dumps(raw, indent=2, ensure_ascii=False) + "\n", force)

    pairs_path = DATA / "pairs.csv"
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    if pairs_path.exists() and not force:
        raise FileExistsError(f"{pairs_path} already exists; use --force to replace it")
    fieldnames = list(rows[0])
    tmp = pairs_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(pairs_path)
    hash_paths.append(pairs_path)

    manifest = {
        "name": spec["dataset_name"],
        "version": spec["version"],
        "created": date.today().isoformat(),
        "generator": "scripts/generate_dataset.py",
        "generation_spec": str(SPEC_PATH.relative_to(ROOT)),
        "seed": spec["seed"],
        "frozen": True,
        "counts": {
            "categories": len(spec["categories"]),
            "job_descriptions": 20,
            "resumes": 100,
            "pairs": len(rows),
        },
        "category_pairs": category_counts,
        "dataset_hash": _dataset_hash(hash_paths),
    }
    _write(DATA / "dataset.json", json.dumps(manifest, indent=2) + "\n", force)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace existing materialised files")
    args = parser.parse_args()
    result = generate(force=args.force)
    print(json.dumps(result, indent=2))
