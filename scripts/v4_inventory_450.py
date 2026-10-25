#!/usr/bin/env python3
"""Build or validate the ignored 450-pair V4 source inventory.

This command performs deterministic source cleanup and pairing only. It never
calls a model and never generates a tailored resume.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v4_data.contracts import canonical, digest, read_jsonl, validate
from v4_data.holdout import check_holdout
from v4_data.pipeline import build_request, check_ledger, privacy_findings
from v4_data.splitting import admit_sources, assign_splits, normalize, pair_sources, similar

DATASET_CANDIDATES = "lang-uk/recruitment-dataset-candidate-profiles-english"
DATASET_JDS = "lang-uk/recruitment-dataset-job-descriptions-english"
CARD_CANDIDATES = "https://huggingface.co/datasets/" + DATASET_CANDIDATES
CARD_JDS = "https://huggingface.co/datasets/" + DATASET_JDS

SKILLS = {
    "active mq": ("active mq", "activemq"), "adobe illustrator": ("illustrator",),
    "agile": ("agile",), "android": ("android",), "angular": ("angular",),
    "api testing": ("api testing", "api test"), "aws": ("aws", "amazon web services"),
    "azure": ("azure",), "bootstrap": ("bootstrap",), "c#": ("c#", ".net"),
    "c++": ("c++",), "ci/cd": ("ci/cd", "continuous integration"),
    "confluence": ("confluence",), "css": ("css", "css3"), "cypress": ("cypress",),
    "docker": ("docker",), "figma": ("figma",), "firebase": ("firebase",),
    "git": ("git", "github", "gitlab"), "gcp": ("gcp", "google cloud"),
    "gradle": ("gradle",), "graphql": ("graphql",), "hibernate": ("hibernate",),
    "html": ("html", "html5"), "java": ("java",), "javascript": ("javascript", " js "),
    "jira": ("jira",), "jmeter": ("jmeter",), "jpa": ("jpa",), "junit": ("junit",),
    "kanban": ("kanban",), "kafka": ("kafka",), "kotlin": ("kotlin",),
    "kubernetes": ("kubernetes",), "mariadb": ("mariadb",), "maven": ("maven",),
    "microservices": ("microservice",), "mobile testing": ("mobile testing", "mobile test"),
    "mongodb": ("mongodb",), "mysql": ("mysql",), "ngrx": ("ngrx",),
    "node.js": ("node.js", "nodejs"), "openapi": ("openapi", "swagger"),
    "oracle": ("oracle",), "playwright": ("playwright",), "postgresql": ("postgres",),
    "postman": ("postman",), "python": ("python",), "rabbitmq": ("rabbitmq", "rabbit mq"),
    "react": ("react.js", "reactjs", "react "), "redis": ("redis",),
    "redux": ("redux",), "regression testing": ("regression testing", "regression test"),
    "rest": ("rest api", "restful", " rest "), "rxjs": ("rxjs",),
    "sass": ("sass", "scss"), "scrum": ("scrum",), "selenium": ("selenium",),
    "smoke testing": ("smoke testing", "smoke test"), "soap": ("soap",),
    "spring": ("spring framework", "spring boot", " spring "),
    "spring boot": ("spring boot",), "sql": (" sql ", "sql server"),
    "test automation": ("automation testing", "automated testing", "test automation"),
    "test cases": ("test case",), "testng": ("testng",), "typescript": ("typescript",),
    "ui testing": ("ui testing", "ui test"), "vue": ("vue.js", "vuejs", " vue "),
    "web testing": ("web testing", "web test"), "webpack": ("webpack",),
}

ROLE_PATTERNS = {
    "frontend": ("front-end", "frontend", "front end"),
    "java": ("java",),
    "qa": ("quality assurance", " qa ", "qa ", "tester", "testing engineer"),
}

CONTACT = re.compile(
    r"(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|https?://|www\.|linkedin|t\.me/|telegram|skype|"
    r"github\.com/|behance|dribbble|(?:\+?\d[\d ()-]{7,}\d))"
)
PERSONAL = re.compile(
    r"(?i)\b(date of birth|born|marital status|married|single|citizenship|nationality|"
    r"home address|postal address|full address|gender|sex:|military status|passport)\b"
)
UNIQUE_RISK = re.compile(
    r"(?i)\b(certificate (?:no|number)|patent|publication|award|olympiad|world record|"
    r"conference speaker|youtube channel|instagram|facebook|audience of|followers|"
    r"project name|coursera for business|far cry|heineken|wl pay|aimienpay)\b"
)
COURSE_MARKETING = re.compile(
    r"(?i)\b(?:udemy course|by the end of this course|students? (?:will )?(?:learn|build)|"
    r"course curriculum|course developer|project/course)\b|(?i:^how to\b)"
)
ASPIRATION = re.compile(
    r"(?i)\b(i am looking for|i'm looking for|looking for|want to|"
    r"would like to|my goal is|my objective is|areas? of interest|prefer to|dream of|seeking (?:a|an)|"
    r"i aspire|i am searching|i'm willing|i am planning to|interested in|searching for a team|"
    r"currently seeking|seeking new opportunities|motivated to|eager to|studying to|"
    r"strong desire to|advancing (?:my )?expertise|enhancing (?:my )?skills|growing (?:my )?understanding|"
    r"obtain a [^.!\n]{0,40} position|have an opportunity to develop|interesting project where|"
    r"positions? below|where i can (?:apply|improve|develop)|"
    r"would be nice to|enrich my knowledge|enjoy studying|improve my [^.!\n]{0,30} skills?|"
    r"not (?:interested|interesed|intreseted)|(?:interesed|intreseted) in|"
    r"interviews? as a max|jobs? with a requirement|grow in automation|"
    r"expectations? for my next career move|looking forward to join|open to (?:learn|job|opportunit)|"
    r"confident that i can|for me good work|where i (?:can|will|would)|i['’]d like to work|"
    r"continuously expand my knowledge|stay up-to-date|passionate about creating|"
    r"self-development direction|opportunit(?:y|ies) to (?:learn|collaborate|acquire|grow)|"
    r"goals? for improvement|looking forward to|continuous learning|continually improve|"
    r"attaining the position|opportunit(?:y|ies) (?:that offer|to grow|for further development)|"
    r"possibility of development|expect opportunit(?:y|ies)|wish i could learn|"
    r"would be happy to consider|valuable asset to any|motivated for professional growth|"
    r"wouldn['’]t like to|i['’]d like to (?:dive|find)|"
    r"allowed me to learn and grow|opportunit(?:y|ies) to develop and improve|"
    r"always open to new knowledge|excited to join|suitable candidate|"
    r"moving forward|in the (?:ongoing )?future|possibilities to grow|development opportunities)\b|"
    r"^professional growth\b"
)
SELF_IDENTIFIER = re.compile(r"(?i)\b(?:my name is|i am named|this is)\s+[A-Z][a-z]{1,30}\b")
JD_DROP = re.compile(
    r"(?i)\b(apply now|send (?:us |your )?(?:cv|resume)|recruiter|contact us|salary|"
    r"compensation|benefits|we offer|(?:is|are) looking for|operates (?:community|a |the )|"
    r"about (?:us|the company)|equal opportunity|"
    r"our (?:client|company|team)|the company|headquartered|work schedule|days off|"
    r"reimbursement|relocation|visa support|training program|career advancement|career development|"
    r"professional growth|family atmosphere|"
    r"award-winning developer|with offices in|global audience|phase of growth|unique opportunity|"
    r"their claim|they develop and market|they build an|there is a small team|is looking to hire|"
    r"lifelong love of learning|technical test|built to change the nature|"
    r"look no further|tech startup seeking|is seeking a skilled|always looking for talented|"
    r"unique situation|chance to learn|"
    r"company parties|personal development|culture and people|centrally located offices?|"
    r"working conditions|"
    r"team buildings?|travel to|multinational corporation|drivers of change|join (?:the|our) team|"
    r"patent|innovation award|leadership award|leading manufacturers|solutions provide|"
    r"purposefully designed for|a few details:|team of \d|used by \d|they deliver|if you are|"
    r"leaders in next generation|innovators in|best-of-breed|success built|team specialises|"
    r"team has developed|free of charge|minimum bureaucracy|business-oriented|every day deployments|"
    r"apply for this vacancy|join us|"
    r"client(?:'s)? product|product team consists|project is|"
    r"we (?:are|build|create|develop|have|pride|reach|provide|work)|"
    r"age|gender|nationality|race|religion|family status)\b"
)
INJECTION = re.compile(r"(?i)\b(ignore (?:all |the )?(?:previous|above)|assistant:|system prompt|do not follow)\b")
JD_MARKETING = re.compile(
    r"(?i)(?:^(?:we|our|the client|client)\b|^at\s+[A-Z]|^as a part of \[employer withheld\]|"
    r"^\[employer withheld\]|\bour mission\b|"
    r"\b(?:leading(?:\s+\w+){0,3}\s+company|one of the leaders|world[’']s leading|software studio from|"
    r"venture-backed company|subscriber locations|corporate parties|market leader|pioneer)\b)"
)
ORG_SUFFIX = re.compile(
    r"\b[A-Z][\w&.-]+(?:\s+[A-Z][\w&.-]+){0,3}\s+"
    r"(?:Inc|Ltd|LLC|GmbH|Corp|Corporation|Systems|Bank|University|Academy|Institute|Center|Group)\b"
)
RESPONSIBILITY = re.compile(
    r"(?i)\b(develop|design|implement|test|analy[sz]e|maintain|create|build|support|"
    r"manage|lead|review|document|write|configure|deploy|monitor|coordinate|troubleshoot)\w*\b"
)
DATE_RANGE = re.compile(
    r"(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ./-]*)?"
    r"(20\d{2})\s*(?:-|–|—|to)\s*(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ./-]*)?"
    r"(20\d{2}|present|current|now)"
)


class InventoryError(ValueError):
    pass


def words(text):
    return re.findall(r"\b[\w+#./-]+\b", text)


def clean_space(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def skill_set(text):
    haystack = " " + clean_space(text).casefold() + " "
    def present(variant):
        if variant != variant.strip():
            return variant in haystack
        if re.fullmatch(r"[a-z0-9+#.\-/]+", variant):
            return bool(re.search(r"(?<![\w+#])" + re.escape(variant) + r"(?![\w+#])", haystack))
        return variant in haystack
    return {name for name, variants in SKILLS.items() if any(present(variant) for variant in variants)}


def role_family(title):
    value = " " + clean_space(title).casefold() + " "
    if any(pattern in value for pattern in ROLE_PATTERNS["qa"]):
        return "qa"
    if (any(pattern in value for pattern in ROLE_PATTERNS["frontend"])
            or re.search(r"\bjavascript\b", value)):
        return "frontend"
    if re.search(r"\bjava\b", value):
        return "java"
    return None


def seniority(title, experience=None):
    value = clean_space(title).casefold()
    if (any(x in value for x in ("head", "director", "principal", "architect"))
            or re.search(r"\blead\b", value)):
        return 4
    if "senior" in value or re.search(r"\bsr\.?\b", value):
        return 3
    if any(x in value for x in ("junior", "intern", "trainee", "graduate")):
        return 1
    if re.search(r"\b(?:middle|mid-level|midlevel|mid)\b", value):
        return 2
    years = 0.0
    try:
        years = float(experience or 0)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(experience or ""))
        years = float(match.group()) if match else 0.0
    return 3 if years >= 6 else 2 if years >= 2 else 1


def candidate_seniority(title, experience):
    try:
        years = float(experience)
    except (TypeError, ValueError):
        return seniority(title)
    return 3 if years >= 6 else 2 if years >= 2 else 1


def jd_seniority(title, experience, text):
    rank = seniority(title, experience)
    scan = re.sub(r"[*_`#]+", " ", clean_space(text))
    if re.search(r"(?i)\b(?:principal|lead)\s+(?:(?:software|java|qa|front\w*)\s+)?(?:engineer|developer)\b", scan):
        rank = 4
    if re.search(r"(?i)\bQA\s+Lead\b|\bleading a (?:group|team) of QAs?\b", scan):
        rank = 4
    requirements = [int(value) for value in re.findall(
        r"(?i)\b(\d{1,2})\+?\s*(?:years?|yrs?)\b", scan)]
    if requirements:
        years = max(requirements)
        rank = max(rank, 3 if years >= 6 else 2 if years >= 2 else 1)
    return rank


def redact_candidate_unit(unit):
    patterns = (
        r"\b((?i:contractor for|worked at|worked for|working at|working for|role at|"
        r"(?:engineer|developer|architect|tester) at))\s+"
        r"(?!\[(?:Organization|Employer) withheld\])"
        r"[A-Za-z0-9&.-]+(?:\s+[A-Z][\w&.-]*){0,3}",
        r"\b((?i:client|company|employer))\s*:\s*[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}",
    )
    value = unit.replace("**", "")
    if re.search(r"(?i)\bprojects?\b", value):
        value = re.sub(r"[\"“][A-Z][^\"”\n]{1,40}[\"”]", "[Product withheld]", value)
    value = re.sub(
        r"^[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3}"
        r"(?=\s+\[[^\]]{2,80}\]\s*[-–—])",
        "[Product withheld]", value)
    for pattern in patterns:
        value = re.sub(pattern, lambda match: match.group(1) + " [Organization withheld]", value)
    value = re.sub(
        r"^[^,:]{2,60},\s*((?:(?:senior|junior|lead|middle|mid-level)\s+)?"
        r"(?:software|java|qa|front[- ]?end|quality assurance)\s+[^:]{0,35}:)",
        r"[Organization withheld], \1", value, flags=re.IGNORECASE)
    value = re.sub(
        r"(\b(?i:(?:(?:senior|junior|lead|middle|mid-level)\s+)?"
        r"(?:java|software|front[- ]?end|full[- ]?stack|qa)\s+"
        r"(?:developer|engineer|architect)(?:\s*/\s*(?:full[- ]?stack\s+)?lead)?)),\s*"
        r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}(?=\s*(?:\(|,|-|$))",
        r"\1, [Organization withheld]", value)
    value = re.sub(r"\b((?i:company)),\s*[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}",
                   r"\1, [Organization withheld]", value)
    value = re.sub(
        r"^[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]+){0,3}\s*"
        r"(\((?i:Provider|Software company|IT company)\b)",
        r"[Organization withheld] \1", value)
    value = re.sub(
        r"^([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]+){0,4})\s*[–—-]\s*"
        r"((?i:(?:(?:senior|junior|middle|lead)\s+)?(?:java|software|front[- ]?end|full[- ]?stack|qa)?\s*"
        r"(?:developer|engineer|tester|manager|architect)\b))",
        r"[Organization withheld] – \2", value)
    value = re.sub(
        r"^[A-Z][A-Z0-9&.-]{2,}(?:,\s*[A-Z][A-Za-z -]+)?\s*[–—-]\s*"
        r"((?i:(?:front[- ]?end|full[- ]?stack|java|software|qa)?\s*"
        r"(?:developer|engineer|tester|manager|architect|lead)\b))",
        r"[Organization withheld] — \1", value)
    value = re.sub(
        r"(\b(?i:developer|engineer|tester|manager|architect|lead)\b)\s*-\s*"
        r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}(?=\s*\()",
        r"\1 - [Organization withheld]", value)
    value = re.sub(
        r"^([A-Z][^:]{1,70}\b(?:Engineering|Agency|Technologies|Solutions|Systems|Group))\s*:",
        "[Organization withheld]:", value)
    if DATE_RANGE.search(value) and re.search(
            r"(?i)\b(?:developer|engineer|tester|manager|architect|lead)\b", value):
        value = re.sub(r"\((?![^)]*\b(?:years?|yrs?|months?)\b)[^)]{2,60}\)\s*$",
                       "([Organization withheld])", value)
    value = re.sub(r"\[Organization withheld\]\s*\([A-Z][\w&.-]{1,30}\)",
                   "[Organization withheld]", value)
    technology_domains = {"asp.net", "cypress.io", "next.js", "node.js", "qase.io",
                          "react.js", "socket.io", "vue.js"}
    value = re.sub(
        r"(?i)\b[a-z0-9][a-z0-9.-]+\.(?:agency|ai|co|com|dev|io|net|org|tech|ua|uk)\b",
        lambda match: match.group(0) if match.group(0).casefold() in technology_domains
        else "[Organization withheld]", value)
    value = re.sub(
        r"(?i)\b(?:company|IT company)\s+(?:which is )?named\s+"
        r"[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}",
        "company [Organization withheld]", value)
    value = re.sub(r"\b(?i:at)[ \t]+[“\"][^”\"]{2,60}[”\"]",
                   "at [Organization withheld]", value)
    value = re.sub(
        r"\b(?i:for)[ \t]+[A-Z][\w&.-]*"
        r"(?:[ \t]+(?:and|&|[A-Z][\w&.-]*)){0,4}[ \t]+Engineering\b",
        "for [Organization withheld]", value)
    def redact_candidate_product(match):
        return match.group(0) if skill_set(match.group(2)) else (
            match.group(1) + " [Product withheld]")
    value = re.sub(
        r"\b((?i:interface of|frontend of|technical document for))[ \t]+"
        r"([A-Z][A-Za-z0-9.-]+)\b", redact_candidate_product, value)
    value = re.sub(r"(?i)\bfor the [A-Z][\w&.-]+ customers\b", "for customers", value)
    value = ORG_SUFFIX.sub("[Organization withheld]", value)
    value = re.sub(
        r"\[Organization withheld\]\s*:\s*[A-Za-z][\w.-]+",
        "[Organization withheld]", value)
    return value


def candidate_organization_names(text):
    names = set()
    patterns = (
        r"(?m)^([A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]+){0,3})[ \t]*"
        r"\((?i:Provider|Software company|IT company)\b",
        r"\b(?i:worked|working|role|internship)[ \t]+(?i:at|for)[ \t]+"
        r"([A-Za-z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})",
        r"\b(?i:engineer|developer|architect|tester)[ \t]+(?i:at)[ \t]+"
        r"([A-Za-z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})",
        r"\b(?i:course|training)[ \t]+(?i:at|of|from)[ \t]+"
        r"([A-Za-z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})",
        r"\b(?i:(?:the[ \t]+)?company)[ \t]+"
        r"([A-Z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})",
        r"\b(?i:worked)[ \t]+(?i:in)[ \t]+(?i:(?:the[ \t]+)?company)[ \t]+"
        r"([A-Z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})",
        r"\b(?i:for)[ \t]+([A-Z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})[ \t]+"
        r"(?i:company|online marketplace)\b",
        r"\b(?i:in)[ \t]+([A-Z][\w&.-]*(?:[ \t]+[A-Z][\w&.-]*){0,3})[ \t]+"
        r"(?i:social network(?: product)?)\b",
    )
    generic = {"a", "an", "client", "company", "employer", "organization", "the"}
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip(" ,.;:-")
            if name.casefold() not in generic and len(name) >= 3:
                names.add(name)
    for match in re.finditer(
            r"\b(?i:(?:the[ \t]+)?company)[ \t]+[A-Z][\w&.-]*,[ \t]*"
            r"([A-Z][\w&.-]+)", text):
        names.add(match.group(1).strip(" ,.;:-"))
    return names


def candidate_product_names(text):
    names = set()
    patterns = (
        r"(?m)^([A-Z][A-Za-z0-9.-]+)[ \t]+is[ \t]+an?[ \t]+[^.\n]{0,80}"
        r"\b(?:platform|application)\b",
        r"\b(?i:interface|frontend|technical document)[ \t]+(?i:of|for)[ \t]+"
        r"([A-Z][A-Za-z0-9.-]+)\b",
        r"\b(?i:on)[ \t]+(?i:the)[ \t]+([A-Z][A-Za-z0-9.-]+)[ \t]+(?i:project)\b",
        r"\b(?i:application),[ \t]+(?i:the)[ \t]+"
        r"([A-Z][A-Za-z0-9.-]+(?:[ \t]+[A-Z][A-Za-z0-9.-]+){1,4})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1)
            if not skill_set(name):
                names.add(name)
    for line in text.splitlines():
        if re.search(r"(?i)\bprojects?\b", line):
            for match in re.finditer(r"[\"“]([A-Z][^\"”\n]{1,40})[\"”]", line):
                names.add(match.group(1))
        match = re.match(
            r"([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3})"
            r"(?=\s+\[[^\]]{2,80}\](?:\s*[-–—])?)", line)
        if match and not skill_set(match.group(1)):
            names.add(match.group(1))
        match = re.search(
            r"\b(?i:developed|built|created)\s+(?:an?\s+)?(?:the\s+)?"
            r"([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3})\s+"
            r"(?i:website|application|platform)\b", line)
        if match and not skill_set(match.group(1)):
            names.add(match.group(1))
        match = re.match(
            r"([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*){0,3})\s*[-–—]\s*"
            r"\((?=[^)]*\b(?:HTML|CSS|SASS|JavaScript|JS|React|Redux|REST|NPM)\b)", line)
        if match and not skill_set(match.group(1)):
            names.add(match.group(1))
    return names


def sanitize_jd_title(title, company):
    value = clean_space(title).replace("**", "")
    value = re.sub(r"[\u0400-\u04ff]+", "", value)
    value = re.sub(r"^\d{2,}/\d{2,}\s+", "", value)
    if company:
        value = re.sub(re.escape(clean_space(company)), "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?i)\s*\((?:sign-on bonus|for product company[^)]*|irc\d+)\)?", "", value)
    value = re.sub(r"(?i)\s+in a product company\s*$", "", value)
    value = re.sub(r"(?i),\s*(?:sign-on\s+)?bonus\b.*$", "", value)
    trailing = re.search(r",\s*([A-Z][A-Za-z0-9&.-]{2,})\s*$", value)
    locations = {"poland", "portugal", "ukraine", "germany", "uk", "usa", "canada"}
    if trailing and trailing.group(1).casefold() not in locations:
        value = value[:trailing.start()]
    value = re.sub(r"(?i)\s+for\s+[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4}(?:\s+(?:Inc|Ltd|LLC|GmbH))?$", "", value)
    value = ORG_SUFFIX.sub("", value)
    value = re.sub(r"\s+\(\s*\)", "", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" -(),")
    if value.count("(") != value.count(")"):
        value = value.replace("(", "").replace(")", "")
    return value


def raw_rows(paths, required_field):
    seen = {}
    for directory in paths:
        for path in sorted(Path(directory).glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise InventoryError(f"Invalid raw JSON: {path}") from exc
            for wrapper in payload.get("rows", []):
                row = wrapper.get("row", {})
                if not row.get(required_field):
                    continue
                key = str(row.get("id", ""))
                if key:
                    seen.setdefault(key, (row, path))
    return list(seen.values())


def split_units(text):
    value = clean_space(text)
    pieces = re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z•-])", value)
    result = []
    for piece in pieces:
        piece = re.sub(r"^\s*(?:(?:[•*\-–—]+)|(?:\d+[.)]))\s*", "", piece).strip(" ;")
        if len(words(piece)) >= 4 and piece.casefold() not in {x.casefold() for x in result}:
            result.append(piece)
    return result


def chronology_findings(text):
    inverted, future = [], []
    for start, end in DATE_RANGE.findall(text):
        a = int(start)
        b = 2023 if end.casefold() in {"present", "current", "now"} else int(end)
        if b < a:
            inverted.append(f"{start}-{end}")
        if a > 2023 or b > 2023:
            future.append(f"{start}-{end}")
    return inverted, future


def sanitize_candidate(row):
    raw = clean_space(row.get("CV"))
    reasons, actions, kept = [], [], []
    if role_family(row.get("Position")) is None:
        reasons.append("unsupported_occupation")
    if CONTACT.search(raw):
        actions.append("removed_contact_or_profile_link_lines")
    if PERSONAL.search(raw):
        actions.append("removed_sensitive_personal_lines")
    if UNIQUE_RISK.search(raw):
        reasons.append("distinctive_personal_detail")
    if re.search(r"[\u0400-\u04ff]", raw):
        actions.append("removed_non_english_lines")
    organization_names = candidate_organization_names(raw)
    for name in sorted(organization_names, key=len, reverse=True):
        raw, count = re.subn(r"(?<![\w])" + re.escape(name) + r"(?![\w])",
                             "[Organization withheld]", raw, flags=re.IGNORECASE)
        if count:
            actions.append("redacted_organization_names")
    for name in sorted(candidate_product_names(raw), key=len, reverse=True):
        raw, count = re.subn(r"(?<![\w])" + re.escape(name) + r"(?![\w])",
                             "[Product withheld]", raw)
        if count:
            actions.append("redacted_product_names")
    raw, slash_count = re.subn(
        r"\[Organization withheld\](?:/[A-Za-z][\w.-]*)+",
        "[Organization withheld]", raw)
    if slash_count:
        actions.append("redacted_organization_names")
    raw, suffix_count = re.subn(
        r"\[Organization withheld\]\s*:\s*[A-Za-z][\w.-]+",
        "[Organization withheld]", raw)
    if suffix_count:
        actions.append("redacted_organization_names")
    inverted, future = chronology_findings(raw)
    if inverted:
        reasons.append("impossible_timeline")
    if future:
        reasons.append("post_publication_timeline")
    try:
        reported_years = float(row.get("Experience Years"))
    except (TypeError, ValueError):
        reported_years = None
    stated_years = [(float(value), bool(plus)) for value, plus in re.findall(
        r"(?i)\b(?:over\s+|more than\s+)?(\d{1,2}(?:\.\d+)?)\s*(\+?)\s+years?\s+(?:of\s+)?"
        r"(?:experience|[\w -]{0,35}(?:testing|development))\b", raw)]
    if reported_years is not None and any(
            (reported_years < value if plus else abs(value - reported_years) > 0.6)
            for value, plus in stated_years):
        reasons.append("contradictory_experience")
    for unit in split_units(raw):
        if (CONTACT.search(unit) or PERSONAL.search(unit) or SELF_IDENTIFIER.search(unit)
                or ASPIRATION.search(unit)
                or COURSE_MARKETING.search(unit)
                or re.search(r"[\u0400-\u04ff]", unit)):
            continue
        if re.search(r"(?i)\b(name|phone|email|address)\s*:", unit):
            continue
        named_heading = re.match(r"^([A-Z][\w&'-]+(?:\s+[A-Z][\w&'-]+){0,5}):", unit)
        generic_headings = {"position", "responsibilities", "technologies", "technology", "stack",
                            "team", "tools", "description", "frameworks", "role", "contribution"}
        if named_heading and named_heading.group(1).casefold() not in generic_headings:
            continue
        redacted = redact_candidate_unit(unit)
        if redacted != unit:
            actions.append("redacted_organization_names")
        unit = redacted
        if (unit.endswith(":") or re.search(
                r"\[Organization withheld\].*\b(?:helps|provides|offers|is an?)\b",
                unit, flags=re.IGNORECASE)):
            continue
        # Retain explicit work evidence and dense skill statements. Identity-only
        # employers, clients, products, courses and project descriptions are removed.
        if not RESPONSIBILITY.search(unit) and len(skill_set(unit)) < 2:
            continue
        kept.append(unit)
    deduplicated = {}
    for unit in kept:
        deduplicated.setdefault(unit.casefold(), unit)
    kept = list(deduplicated.values())
    # Bound unique narrative combinations while retaining enough evidence.
    kept = kept[:24]
    evidence_words = sum(len(words(x)) for x in kept)
    skills = sorted(skill_set(" ".join(kept)))
    responsibility_units = sum(bool(RESPONSIBILITY.search(x)) for x in kept)
    if evidence_words < 120:
        reasons.append("thin_candidate")
    if len(kept) < 6:
        reasons.append("insufficient_fact_units")
    if len(skills) < 4:
        reasons.append("insufficient_explicit_skills")
    if responsibility_units < 3:
        reasons.append("insufficient_responsibilities")
    if not kept:
        return None, sorted(set(reasons or ["empty_after_sanitization"])), actions
    years = row.get("Experience Years")
    try:
        years_text = f"{float(years):g}"
    except (TypeError, ValueError):
        years_text = "not stated"
    text = "## Profile evidence\nReported professional experience: " + years_text + " years.\n\n"
    text += "## Experience and responsibilities\n" + "\n".join("- " + x for x in kept) + "\n\n"
    text += "## Explicitly mentioned skills\n" + "; ".join(skills) + ".\n"
    if privacy_findings(text) or CONTACT.search(text) or PERSONAL.search(text):
        reasons.append("pii_after_sanitization")
    return text, sorted(set(reasons)), sorted(set(actions))


def sanitize_jd(row):
    company = clean_space(row.get("Company Name"))
    title = sanitize_jd_title(row.get("Position"), company)
    raw = clean_space(row.get("Long Description"))
    reasons, actions, kept = [], [], []
    family = role_family(title)
    if family is None:
        reasons.append("unsupported_occupation")
    if (len(words(title)) > 14
            or re.search(r"(?i)\b(?:good knowledge|basic knowledge|experience in).*(?:will be a plus|knowledge of)\b", title)):
        reasons.append("invalid_jd_title")
    if re.search(r"(?i)\bwith\s+[A-Z][A-Za-z-]+\s+language\b", title):
        reasons.append("unsupported_language_requirement")
    if jd_seniority(title, row.get("Exp Years"), raw) == 4:
        reasons.append("leadership_role")
    if company:
        raw = re.sub(r"(?<!\w)" + re.escape(company) + r"(?!\w)",
                     "[Employer withheld]", raw, flags=re.IGNORECASE)
        actions.append("redacted_company_name")
    for unit in split_units(raw):
        if (CONTACT.search(unit) or JD_DROP.search(unit) or JD_MARKETING.search(unit)
                or INJECTION.search(unit) or re.search(r"[\u0400-\u04ff]", unit)
                or re.search(r"(?i)\b(?:NYSE|NASDAQ|LSE)\s*:", unit)):
            continue
        if re.search(r"(?i)\b(?:mr|ms|mrs|dr)\.?\s+[A-Z][a-z]+", unit):
            continue
        redacted = redact_candidate_unit(unit)
        if redacted != unit:
            actions.append("redacted_organization_names")
        unit = redacted
        def redact_product(match):
            return match.group(0) if skill_set(match.group(1)) else (
                "[Product withheld] " + match.group(2) + " " + match.group(3))
        unit, product_count = re.subn(
            r"\b([A-Z][A-Za-z0-9.-]+)\s+"
            r"(backend|frontend|web|mobile)\s+(applications?|platforms?|services?)\b",
            redact_product, unit)
        if product_count and "[Product withheld]" in unit:
            actions.append("redacted_product_names")
        if not RESPONSIBILITY.search(unit) and len(skill_set(unit)) < 2:
            continue
        kept.append(unit)
    kept = kept[:32]
    body = "\n".join("- " + x for x in kept)
    text = "# " + title + "\n\n## Responsibilities and requirements\n" + body + "\n"
    skills = sorted(skill_set(title + " " + body))
    if len(words(body)) < 80:
        reasons.append("thin_jd")
    if len(kept) < 6 or sum(bool(RESPONSIBILITY.search(x)) for x in kept) < 3:
        reasons.append("insufficient_responsibilities")
    if family == "qa" and sum(bool(re.search(
            r"(?i)\b(test|testing|quality|bug|defect|selenium|cypress|playwright|junit|testng)\w*\b", unit))
            for unit in kept) < 3:
        reasons.append("insufficient_occupation_evidence")
    if len(skills) < 4:
        reasons.append("insufficient_explicit_skills")
    if privacy_findings(text) or CONTACT.search(text) or PERSONAL.search(text):
        reasons.append("pii_after_sanitization")
    return text, sorted(set(reasons)), sorted(set(actions)), skills


def source_record(kind, row, text, revision, raw_path, created_at):
    raw_hash = digest(row)
    short = digest([kind, row["id"]])[:20]
    prefix = "clean450-c" if kind == "candidate" else "clean450-j"
    source_id = f"{prefix}-{short}-v1"
    dataset = DATASET_CANDIDATES if kind == "candidate" else DATASET_JDS
    uri = (CARD_CANDIDATES if kind == "candidate" else CARD_JDS) + f"?row={row['id']}"
    return validate({
        "schema_version": "1.0", "id": source_id, "kind": kind,
        "family_id": source_id.removesuffix("-v1") + "-family",
        "template_family_id": source_id.removesuffix("-v1") + "-source-template",
        "parent_ids": [], "related_family_ids": [], "created_at": created_at,
        "source": {"dataset": dataset, "uri": uri, "revision": revision,
            "source_id": str(row["id"]), "retrieved_at": created_at, "license": "MIT",
            "restrictions": ["Local development only; derived release requires renewed privacy and license review."],
            "attribution": "Djinni Recruitment Dataset (English)", "language": "en", "fictional": False,
            "raw_sha256": raw_hash, "use_decision": "local-approved", "privacy_review": "passed",
            "decision_note": "Deterministically sanitized and admitted by inventory-450-v1; raw cache remains ignored."},
        "text": text, "text_sha256": digest(text), "normalized_sha256": digest(normalize(text)),
        "normalization_version": "nfkc-words-v1",
    }, "source")


def ledger_for(source, row, created_at):
    facts, offset, number = [], 0, 0
    section = None
    for line in source["text"].splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped[3:]
        elif stripped:
            claim = stripped[2:] if stripped.startswith("- ") else stripped
            start = offset + line.index(claim)
            category = "summary" if section == "Profile evidence" else "skill" if section == "Explicitly mentioned skills" else "experience"
            number += 1
            facts.append({"id": f"{source['id']}-f{number:02d}", "category": category,
                "claim": claim, "evidence": {"start": start, "end": start + len(claim), "text": claim},
                "confidence": 1})
        offset += len(line)
    ledger = {"schema_version": "1.0", "id": "ledger-" + digest(source["id"])[:24],
        "candidate_id": source["id"], "candidate_sha256": source["text_sha256"],
        "version": "extractive-v1", "created_at": created_at, "facts": facts}
    return validate(ledger, "ledger")


def deduplicate(items, threshold):
    accepted, rejected = [], []
    hashes = set()
    prepared = []
    for item in sorted(items, key=lambda x: x["source"]["source_id"]):
        norm_hash = item["normalized_sha256"]
        if norm_hash in hashes:
            rejected.append((item, "exact_duplicate"))
            continue
        norm = normalize(item["text"])
        tokens = norm.split()
        shingles = {tuple(tokens[i:i + 3]) for i in range(max(0, len(tokens) - 2))}
        possible = []
        for index, (other_norm, other_shingles) in enumerate(prepared):
            if min(len(norm), len(other_norm)) >= 30 and (norm in other_norm or other_norm in norm):
                possible.append(index)
                continue
            if not shingles or not other_shingles:
                continue
            length_ratio = min(len(norm), len(other_norm)) / max(len(norm), len(other_norm))
            if length_ratio < 0.65:
                continue
            overlap = len(shingles & other_shingles) / len(shingles | other_shingles)
            if overlap >= 0.3:
                possible.append(index)
        if any(similar(item["text"], accepted[index]["text"], threshold) for index in possible):
            rejected.append((item, "near_duplicate"))
            continue
        accepted.append(item)
        prepared.append((norm, shingles))
        hashes.add(norm_hash)
    return accepted, rejected


def pair_score(candidate, jd, cmeta, jmeta):
    if cmeta["role_family"] != jmeta["role_family"]:
        return None
    if jmeta["seniority"] > cmeta["seniority"] or cmeta["seniority"] - jmeta["seniority"] > 1:
        return None
    if (cmeta.get("experience_years") is not None and jmeta.get("required_years") is not None
            and jmeta["required_years"] > cmeta["experience_years"]):
        return None
    shared = sorted(cmeta["skills"] & jmeta["skills"])
    coverage = len(shared) / max(1, len(jmeta["skills"]))
    if len(shared) < 3 or coverage < 0.3:
        return None
    if "automation" in jmeta.get("raw_position", "").casefold() and not (
            cmeta["skills"] & {"test automation", "selenium", "cypress", "playwright", "testng"}):
        return None
    if (cmeta["role_family"] == "qa"
            and re.search(r"\bjava\b", jmeta.get("raw_position", "").casefold())
            and "java" not in cmeta["skills"]):
        return None
    specializations = {
        "machine learning": ("machine learning", " ml "),
        "blockchain": ("blockchain", "web3"),
        "digital identity": ("identity", "security"),
        "etl": (" etl ", "data pipeline"),
        "three.js": ("three.js", "threejs"),
        "kotlin": ("kotlin",),
        "react": ("react",),
        "angular": ("angular",),
        "vue": (" vue ", "vue.js", "vuejs"),
    }
    candidate_text = " " + cmeta.get("text", "").casefold() + " "
    title = jmeta.get("raw_position", "").casefold()
    for label, evidence in specializations.items():
        if label in title and not any(term in candidate_text for term in evidence):
            return None
    if ("full stack" in title or "fullstack" in title) and not (
            cmeta["skills"] & {"javascript", "react", "angular", "vue", "node.js"}):
        return None
    score = round(0.5 + 0.3 * coverage + 0.1 * min(len(shared), 6) / 6 +
                  0.1 * (cmeta["seniority"] == jmeta["seniority"]), 6)
    return score, coverage, shared


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate_pairs(candidates, options, blocked_jds=(), jd_capacity=2):
    """Find two distinct JDs per candidate with a capacity network."""
    blocked = set(blocked_jds)
    option_map = {candidate_id: {jd_id: (score, detail) for score, jd_id, detail in ranked}
                  for candidate_id, ranked in options.items()}
    graph = defaultdict(list)

    def add_edge(left, right, capacity):
        forward = {"to": right, "reverse": len(graph[right]), "capacity": capacity,
                   "initial": capacity}
        reverse = {"to": left, "reverse": len(graph[left]), "capacity": 0, "initial": 0}
        graph[left].append(forward)
        graph[right].append(reverse)

    source_node, sink_node = ("source",), ("sink",)
    usable_jds = set()
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        candidate_id = candidate["id"]
        candidate_node = ("candidate", candidate_id)
        add_edge(source_node, candidate_node, 2)
        for _, jd_id, _ in options[candidate_id]:
            if jd_id not in blocked:
                add_edge(candidate_node, ("jd", jd_id), 1)
                usable_jds.add(jd_id)
    for jd_id in sorted(usable_jds):
        add_edge(("jd", jd_id), sink_node, jd_capacity)

    total, target = 0, 2 * len(candidates)
    while total < target:
        level = {source_node: 0}
        queue = deque([source_node])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge["capacity"] and edge["to"] not in level:
                    level[edge["to"]] = level[node] + 1
                    queue.append(edge["to"])
        if sink_node not in level:
            break
        cursor = defaultdict(int)

        def send(node, amount):
            if node == sink_node:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                if edge["capacity"] and level.get(edge["to"]) == level[node] + 1:
                    sent = send(edge["to"], min(amount, edge["capacity"]))
                    if sent:
                        edge["capacity"] -= sent
                        graph[edge["to"]][edge["reverse"]]["capacity"] += sent
                        return sent
                cursor[node] += 1
            return 0

        while total < target:
            sent = send(source_node, target - total)
            if not sent:
                break
            total += sent
    if total != target:
        raise InventoryError(f"Insufficient compatible JD capacity: allocated {total}/{target}")

    result = []
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        candidate_id = candidate["id"]
        for edge in graph[("candidate", candidate_id)]:
            if edge["to"][0] == "jd" and edge["initial"] == 1 and edge["capacity"] == 0:
                jd_id = edge["to"][1]
                score, detail = option_map[candidate_id][jd_id]
                result.append((candidate_id, jd_id, score, detail))
    if any(count != 2 for count in Counter(candidate_id for candidate_id, _, _, _ in result).values()):
        raise InventoryError("Candidate allocation cardinality drift")
    if len({(candidate_id, jd_id) for candidate_id, jd_id, _, _ in result}) != len(result):
        raise InventoryError("Candidate allocated the same JD twice")
    return result


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_holdout(path):
    rows = read_jsonl(path, "holdout")
    if len(rows) != 1:
        raise InventoryError("Expected one frozen holdout snapshot")
    return rows[0]


def build(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise InventoryError("Output directory already exists")
    if not out.is_relative_to((ROOT / "runs").resolve()):
        raise InventoryError("Output must be a fresh directory under runs/")
    cfg = load_config(args.config)
    created_at = args.at
    datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    holdout = load_holdout(args.holdout)
    candidate_rows = raw_rows(args.candidate_raw, "CV")
    jd_rows = raw_rows(args.jd_raw, "Long Description")

    candidate_items, jd_items = [], []
    candidate_rejects, jd_rejects = [], []
    stats = Counter()
    for row, path in candidate_rows:
        text, reasons, actions = sanitize_candidate(row)
        stats["candidate_screened"] += 1
        for action in set(actions):
            stats["candidate_action:" + action] += 1
        if reasons:
            candidate_rejects.append({"source_id_sha256": digest(str(row["id"])), "reasons": reasons,
                                      "actions": actions, "raw_file": path.name})
            continue
        source = source_record("candidate", row, text, cfg["candidate_revision"], path, created_at)
        try:
            check_holdout([source], holdout, load_config(args.policy))
        except Exception:
            candidate_rejects.append({"source_id_sha256": digest(str(row["id"])),
                                      "reasons": ["frozen_holdout_overlap"], "actions": actions,
                                      "raw_file": path.name})
            continue
        candidate_items.append((source, row, actions))

    for row, path in jd_rows:
        text, reasons, actions, skills = sanitize_jd(row)
        stats["jd_screened"] += 1
        for action in set(actions):
            stats["jd_action:" + action] += 1
        if reasons:
            jd_rejects.append({"source_id_sha256": digest(str(row["id"])), "reasons": reasons,
                               "actions": actions, "raw_file": path.name})
            continue
        source = source_record("jd", row, text, cfg["jd_revision"], path, created_at)
        try:
            check_holdout([source], holdout, load_config(args.policy))
        except Exception:
            jd_rejects.append({"source_id_sha256": digest(str(row["id"])),
                               "reasons": ["frozen_holdout_overlap"], "actions": actions,
                               "raw_file": path.name})
            continue
        jd_items.append((source, row, actions, skills))

    candidate_unique, candidate_dupes = deduplicate([x[0] for x in candidate_items], cfg["near_duplicate_threshold"])
    jd_unique, jd_dupes = deduplicate([x[0] for x in jd_items], cfg["near_duplicate_threshold"])
    c_lookup = {x[0]["id"]: x for x in candidate_items}
    j_lookup = {x[0]["id"]: x for x in jd_items}
    for item, reason in candidate_dupes:
        candidate_rejects.append({"source_id_sha256": digest(item["source"]["source_id"]), "reasons": [reason], "actions": []})
    for item, reason in jd_dupes:
        jd_rejects.append({"source_id_sha256": digest(item["source"]["source_id"]), "reasons": [reason], "actions": []})

    cmeta, jmeta = {}, {}
    for source in candidate_unique:
        row = c_lookup[source["id"]][1]
        try:
            experience_years = float(row.get("Experience Years"))
        except (TypeError, ValueError):
            experience_years = None
        cmeta[source["id"]] = {"role_family": role_family(row.get("Position")),
            "seniority": candidate_seniority(row.get("Position"), row.get("Experience Years")),
            "skills": skill_set(source["text"]), "raw_position": clean_space(row.get("Position")),
            "text": source["text"], "experience_years": experience_years}
    for source in jd_unique:
        row = j_lookup[source["id"]][1]
        jmeta[source["id"]] = {"role_family": role_family(row.get("Position")),
            "seniority": jd_seniority(row.get("Position"), row.get("Exp Years"), row.get("Long Description", "")),
            "skills": skill_set(source["text"]),
            "raw_position": source["text"].splitlines()[0].removeprefix("# "),
            "required_years": max([int(value) for value in re.findall(
                r"(?i)\b(\d{1,2})\+?\s*(?:years?|yrs?)\b", row.get("Long Description", ""))] or [0])}

    # Prefer candidates with more viable choices and stronger explicit evidence.
    options = {}
    for candidate in candidate_unique:
        ranked = []
        for jd in jd_unique:
            scored = pair_score(candidate, jd, cmeta[candidate["id"]], jmeta[jd["id"]])
            if scored:
                ranked.append((scored[0], jd["id"], scored))
        options[candidate["id"]] = sorted(ranked, key=lambda x: (-x[0], x[1]))
    candidates = [c for c in candidate_unique if len(options[c["id"]]) >= 2]
    jd_validation_ids = {jd["id"] for jd in jd_unique
                         if int(digest(jd["id"])[:8], 16) % 5 == 0}

    # Reserve both sides before pairing. A source family's stable hash decides
    # its eligible split; candidate selection then uses only JDs from that split.
    selected, val_ids = [], set()
    target_by_family = {"frontend": 75, "java": 75, "qa": 75}
    validation_by_family = {"frontend": 7, "java": 7, "qa": 6}
    for family, count in target_by_family.items():
        pool = [c for c in candidates if cmeta[c["id"]]["role_family"] == family]
        validation_pool = [c for c in pool if sum(jd_id in jd_validation_ids
                           for _, jd_id, _ in options[c["id"]]) >= 2]
        validation_pool.sort(key=lambda c: (-sum(jd_id in jd_validation_ids
                             for _, jd_id, _ in options[c["id"]]), -len(words(c["text"])), c["id"]))
        chosen_validation = validation_pool[:validation_by_family[family]]
        val_ids.update(c["id"] for c in chosen_validation)
        train_pool = [c for c in pool if c["id"] not in val_ids and
                      sum(jd_id not in jd_validation_ids for _, jd_id, _ in options[c["id"]]) >= 2]
        train_pool.sort(key=lambda c: (-sum(jd_id not in jd_validation_ids
                        for _, jd_id, _ in options[c["id"]]), -len(words(c["text"])), c["id"]))
        selected.extend(chosen_validation)
        selected.extend(train_pool[:count - validation_by_family[family]])
    if len(selected) != cfg["candidate_count"]:
        counts = Counter(cmeta[c["id"]]["role_family"] for c in candidates)
        raise InventoryError(f"Insufficient compatible candidate families: {counts}")
    if len(val_ids) != cfg["validation_candidate_families"]:
        raise InventoryError("Validation family count drift")

    # Reuse is capped and always stays within one split. This supports candidate
    # comparisons without forcing 450 near-interchangeable postings.
    validation_candidates = [c for c in selected if c["id"] in val_ids]
    train_candidates = [c for c in selected if c["id"] not in val_ids]
    validation_plans = allocate_pairs(
        validation_candidates, options,
        {jd["id"] for jd in jd_unique if jd["id"] not in jd_validation_ids},
        cfg["max_pairs_per_jd"])
    train_plans = allocate_pairs(train_candidates, options, jd_validation_ids,
                                 cfg["max_pairs_per_jd"])
    plans = validation_plans + train_plans
    jd_split = {jd_id: "validation" for _, jd_id, _, _ in validation_plans}
    jd_split.update({jd_id: "train" for _, jd_id, _, _ in train_plans})
    used_jds = set(jd_split)
    selected_jds = [j for j in jd_unique if j["id"] in used_jds]
    selected_candidate_ids = {source["id"] for source in selected}
    selected_jd_ids = {source["id"] for source in selected_jds}
    for source in candidate_unique:
        if source["id"] not in selected_candidate_ids:
            reason = "insufficient_compatible_jds" if len(options[source["id"]]) < 2 else "eligible_not_selected"
            candidate_rejects.append({"source_id_sha256": digest(source["source"]["source_id"]),
                                      "reasons": [reason], "actions": []})
    for source in jd_unique:
        if source["id"] not in selected_jd_ids:
            jd_rejects.append({"source_id_sha256": digest(source["source"]["source_id"]),
                               "reasons": ["eligible_not_selected"], "actions": []})

    all_sources = sorted(selected + selected_jds, key=lambda x: x["id"])
    reservations = {}
    for source in all_sources:
        if source["kind"] == "candidate":
            split = "validation" if source["id"] in val_ids else "train"
        else:
            split = jd_split[source["id"]]
        reservations[source["id"]] = split
    planned_pairs = [{"schema_version": "1.0", "id": "pair-" + digest([candidate_id, jd_id])[:24],
                      "candidate_id": candidate_id, "jd_id": jd_id}
                     for candidate_id, jd_id, _, _ in plans]
    assignments = assign_splits(all_sources, load_config(args.policy), created_at,
                                existing_pairs=planned_pairs, reservations=reservations)
    by_assignment = {x["source_id"]: x for x in assignments}
    pairs, quality = [], []
    for candidate_id, jd_id, score, detail in sorted(plans):
        pair = pair_sources(candidate_id, jd_id, all_sources, assignments)
        pairs.append(pair)
        quality.append({"schema_version": "1.0", "pair_id": pair["id"],
            "candidate_id": candidate_id, "jd_id": jd_id,
            "split": by_assignment[candidate_id]["split"], "occupation_family": cmeta[candidate_id]["role_family"],
            "candidate_seniority": cmeta[candidate_id]["seniority"], "jd_seniority": jmeta[jd_id]["seniority"],
            "candidate_experience_years": cmeta[candidate_id]["experience_years"],
            "jd_required_years": jmeta[jd_id]["required_years"],
            "candidate_position": cmeta[candidate_id]["raw_position"], "jd_position": jmeta[jd_id]["raw_position"],
            "shared_skills": detail[2], "skill_coverage": round(detail[1], 6), "score": score,
            "reason": "Same occupation family, supported experience range, non-increasing seniority, and explicit shared skills.",
            "accepted": True})
    if Counter(q["split"] for q in quality) != {"train": 410, "validation": 40}:
        raise InventoryError("Pair split count drift")

    ledgers = [ledger_for(c, c_lookup[c["id"]][1], created_at) for c in sorted(selected, key=lambda x: x["id"])]
    by_ledger = {x["candidate_id"]: x for x in ledgers}
    requests = [build_request(pair, all_sources, by_ledger[pair["candidate_id"]], assignments) for pair in pairs]
    check_holdout(all_sources, holdout, load_config(args.policy))
    admit_sources(all_sources)
    for ledger in ledgers:
        check_ledger(ledger, next(x for x in all_sources if x["id"] == ledger["candidate_id"]))

    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "sources.jsonl", all_sources)
    write_jsonl(out / "ledgers.jsonl", ledgers)
    write_jsonl(out / "assignments.jsonl", assignments)
    write_jsonl(out / "pairs.jsonl", pairs)
    write_jsonl(out / "requests.jsonl", requests)
    write_jsonl(out / "pair-quality.jsonl", quality)
    write_jsonl(out / "rejected-candidates.jsonl", candidate_rejects)
    write_jsonl(out / "rejected-jds.jsonl", jd_rejects)

    sample_ids = set()
    for family in ("frontend", "java", "qa"):
        rows = [q for q in quality if q["occupation_family"] == family]
        rows.sort(key=lambda q: (q["score"], q["pair_id"]))
        sample_ids.update(q["pair_id"] for q in rows[:4] + rows[len(rows)//2:len(rows)//2+3] + rows[-3:])
    review = [{**q, "candidate_excerpt": next(s["text"] for s in all_sources if s["id"] == q["candidate_id"])[:600],
               "jd_excerpt": next(s["text"] for s in all_sources if s["id"] == q["jd_id"])[:600],
               "review_status": "selected_for_manual_review"} for q in quality if q["pair_id"] in sample_ids]
    write_jsonl(out / "manual-review-sample.jsonl", review)

    source_counts = Counter(s["kind"] for s in all_sources)
    summary = {"schema_version": "1.0", "created_at": created_at,
        "candidate_rows_screened": stats["candidate_screened"], "jd_rows_screened": stats["jd_screened"],
        "candidate_rows_rejected": len(candidate_rejects), "jd_rows_rejected": len(jd_rejects),
        "candidate_sanitization_actions": {key.removeprefix("candidate_action:"): value for key, value in sorted(stats.items()) if key.startswith("candidate_action:")},
        "jd_sanitization_actions": {key.removeprefix("jd_action:"): value for key, value in sorted(stats.items()) if key.startswith("jd_action:")},
        "candidate_rejection_reasons": dict(Counter(reason for row in candidate_rejects for reason in row["reasons"])),
        "jd_rejection_reasons": dict(Counter(reason for row in jd_rejects for reason in row["reasons"])),
        "candidates_retained": source_counts["candidate"], "jds_retained": source_counts["jd"],
        "pairs": len(pairs), "train_pairs": 410, "validation_pairs": 40,
        "role_pairs": dict(Counter(q["occupation_family"] for q in quality)),
        "minimum_pair_score": min(q["score"] for q in quality),
        "median_pair_score": sorted(q["score"] for q in quality)[len(quality)//2],
        "minimum_skill_coverage": min(q["skill_coverage"] for q in quality),
        "exact_duplicates_selected": 0, "near_duplicates_selected": 0,
        "cross_split_family_leakage": 0, "holdout_overlaps": 0,
        "targets_generated": 0, "external_model_calls": 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    provenance = {"schema_version": "1.0", "retrieved_at": created_at, "sources": [
        {"name": "Djinni candidate profiles (English)", "url": CARD_CANDIDATES,
         "revision": cfg["candidate_revision"], "license": "MIT (dataset card declaration)",
         "restrictions": "Local development; release requires renewed source, privacy, and attribution review.",
         "provenance": "Anonymous profiles supplied by Djinni; posts date from 2020-2023.",
         "original_size": 210250, "schema": ["Position", "Moreinfo", "Looking For", "Highlights", "Primary Keyword", "English Level", "Experience Years", "CV", "CV_lang", "id"],
         "fields_used": ["Position", "Experience Years", "CV", "id"],
         "privacy_concerns": "Residual organizations, dates, links, contacts, certificate numbers, and distinctive achievements can remain in nominally anonymous free text.",
         "filtering": "Occupation restriction, contact/sensitive/aspiration line removal, distinctive-detail and timeline rejection, evidence/skill minimums, exact and near deduplication, frozen-holdout exclusion.",
         "screened": stats["candidate_screened"], "retained": source_counts["candidate"], "decision": "included"},
        {"name": "Djinni job descriptions (English)", "url": CARD_JDS,
         "revision": cfg["jd_revision"], "license": "MIT (dataset card declaration)",
         "restrictions": "Local development; release requires renewed source, privacy, and attribution review.",
         "provenance": "Job postings supplied by Djinni; posts date from 2020-2023.",
         "original_size": 141897, "schema": ["Position", "Long Description", "Company Name", "Exp Years", "Primary Keyword", "English Level", "Published", "Long Description_lang", "id"],
         "fields_used": ["Position", "Long Description", "Company Name", "Exp Years", "id"],
         "privacy_concerns": "Recruiter contacts and tracking links may occur in free text.",
         "filtering": "Company/contact/application/benefit/discriminatory boilerplate removal, role/seniority/responsibility/skill minimums, exact and near deduplication, frozen-holdout exclusion.",
         "screened": stats["jd_screened"], "retained": source_counts["jd"], "decision": "included"},
        {"name": "datasetmaster/resumes", "url": "https://huggingface.co/datasets/datasetmaster/resumes",
         "license": "MIT card declaration; upstream submissions not separately documented", "decision": "rejected",
         "reason": "Mixed real/synthetic records cannot be reliably separated and upstream consent/provenance is insufficient for this inventory."},
        {"name": "candidate-matching-synthetic", "url": "https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic",
         "license": "MIT card declaration", "decision": "rejected",
         "reason": "Generic generated responsibility bullets and narrow templates do not meet the V2 evidence-quality reference."},
        {"name": "RJDB", "url": "https://github.com/megagonlabs/rjdb", "license": "CC BY-SA 4.0 repository declaration",
         "decision": "rejected", "reason": "Synthetic matching task and share-alike/upstream constraints are unnecessary for this local inventory."},
        {"name": "USAJOBS", "url": "https://developer.usajobs.gov", "license": "Open public-data terms with separately restricted-item exceptions",
         "decision": "rejected", "reason": "No credentials were used; federal-domain prose would add skew and was unnecessary after Djinni screening."}
    ]}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": "1.0", "created_at": created_at,
        "config_sha256": file_hash(args.config), "holdout_sha256": file_hash(args.holdout),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(canonical(summary))


def validate_inventory(args):
    path = Path(args.inventory)
    sources = read_jsonl(path / "sources.jsonl", "source")
    ledgers = read_jsonl(path / "ledgers.jsonl", "ledger")
    assignments = read_jsonl(path / "assignments.jsonl", "assignment")
    pairs = read_jsonl(path / "pairs.jsonl", "pair")
    requests = read_jsonl(path / "requests.jsonl", "request")
    with (path / "pair-quality.jsonl").open(encoding="utf-8") as handle:
        quality = [json.loads(x) for x in handle]
    policy = load_config(args.policy)
    cfg = load_config(args.config)
    holdout = load_holdout(args.holdout)
    jd_count = sum(source["kind"] == "jd" for source in sources)
    if (sum(source["kind"] == "candidate" for source in sources) != 225
            or not 225 <= jd_count <= 450 or len(ledgers) != 225
            or len(pairs) != 450 or len(requests) != 450):
        raise InventoryError("Inventory count mismatch")
    if Counter(q["split"] for q in quality) != {"train": 410, "validation": 40}:
        raise InventoryError("Quality split mismatch")
    if any(not q["accepted"] or q["score"] < 0.5 or len(q["shared_skills"]) < 3
           or q["skill_coverage"] < cfg["minimum_skill_coverage"]
           or q["candidate_seniority"] < q["jd_seniority"]
           or (q["candidate_experience_years"] is not None
               and q["jd_required_years"] > q["candidate_experience_years"])
           for q in quality):
        raise InventoryError("Pair quality gate failed")
    if any(privacy_findings(source["text"]) or CONTACT.search(source["text"])
           or PERSONAL.search(source["text"]) for source in sources):
        raise InventoryError("PII gate failed")
    for source in sources:
        if source["kind"] == "candidate":
            if len(words(source["text"])) < cfg["minimum_candidate_words"]:
                raise InventoryError("Candidate evidence gate failed")
            inverted, future = chronology_findings(source["text"])
            if inverted or future:
                raise InventoryError("Candidate chronology gate failed")
        elif len(words(source["text"])) < cfg["minimum_jd_words"]:
            raise InventoryError("JD evidence gate failed")
    if any(count != cfg["pairs_per_candidate"] for count in
           Counter(pair["candidate_id"] for pair in pairs).values()):
        raise InventoryError("Candidate pair cardinality failed")
    if max(Counter(pair["jd_id"] for pair in pairs).values()) > cfg["max_pairs_per_jd"]:
        raise InventoryError("JD reuse cap failed")
    check_holdout(sources, holdout, policy)
    rebuilt = assign_splits(sources, policy, "2000-01-01T00:00:00Z",
                            existing_pairs=pairs, previous=assignments)
    if sorted(rebuilt, key=lambda x: x["source_id"]) != sorted(assignments, key=lambda x: x["source_id"]):
        raise InventoryError("Assignment drift")
    by_ledger = {x["candidate_id"]: x for x in ledgers}
    expected = [build_request(pair, sources, by_ledger[pair["candidate_id"]], assignments) for pair in pairs]
    if expected != requests:
        raise InventoryError("Request reconstruction mismatch")
    families = defaultdict(set)
    for assignment in assignments:
        families[assignment["family_id"]].add(assignment["split"])
    if any(len(x) != 1 for x in families.values()):
        raise InventoryError("Family leakage")
    manifest = json.loads((path / "manifest.json").read_text())
    for name, meta in manifest["files"].items():
        if file_hash(path / name) != meta["sha256"]:
            raise InventoryError(f"Manifest mismatch: {name}")
    print(canonical({"status": "ready", "sources": len(sources), "candidates": 225, "jds": jd_count,
        "pairs": 450, "train": 410, "validation": 40, "targets": 0, "holdout_overlaps": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--candidate-raw", action="append", required=True)
    build_p.add_argument("--jd-raw", action="append", required=True)
    build_p.add_argument("--holdout", required=True)
    build_p.add_argument("--out", required=True)
    build_p.add_argument("--at", required=True)
    build_p.add_argument("--config", default=str(ROOT / "configs/v4/inventory-450.json"))
    build_p.add_argument("--policy", default=str(ROOT / "configs/v4/split-policy.json"))
    val_p = sub.add_parser("validate")
    val_p.add_argument("--inventory", required=True)
    val_p.add_argument("--holdout", required=True)
    val_p.add_argument("--config", default=str(ROOT / "configs/v4/inventory-450.json"))
    val_p.add_argument("--policy", default=str(ROOT / "configs/v4/split-policy.json"))
    args = parser.parse_args()
    try:
        build(args) if args.command == "build" else validate_inventory(args)
    except (InventoryError, ValueError, OSError) as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
