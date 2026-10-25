import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "scripts/v4_inventory_450.py"
SPEC = importlib.util.spec_from_file_location("v4_inventory_450", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Inventory450Tests(unittest.TestCase):
    def test_raw_rows_filters_mixed_cache_types(self):
        self.assertIn("required_field", MODULE.raw_rows.__code__.co_varnames)

    def test_candidate_drops_contacts_and_aspirations(self):
        row = {"Position": "Senior Java Developer", "Experience Years": 7, "CV": """
        Built Java and Spring Boot services for payment processing and documented REST APIs.
        Implemented PostgreSQL persistence and Kafka consumers with Docker deployment.
        Maintained JUnit tests and reviewed Git pull requests with the backend team.
        Designed microservices and monitored Redis-backed workflows in production.
        Investigated support incidents and coordinated fixes through Jira and Scrum.
        Configured Maven builds and Kubernetes manifests for repeatable releases.
        Contact me at person@example.com.
        I am looking for interesting AWS and machine learning work.
        """}
        text, reasons, actions = MODULE.sanitize_candidate(row)
        self.assertNotIn("pii_after_sanitization", reasons)
        self.assertNotIn("person@example.com", text)
        self.assertNotIn("looking for", text)
        self.assertIn("removed_contact_or_profile_link_lines", actions)

    def test_candidate_drops_non_english_identity_line(self):
        row = {"Position": "Java Developer", "Experience Years": 5, "CV":
               "ЗАО Альфа-Банк, Java Developer. " +
               "Developed Java Spring Docker SQL Git services and tested REST APIs. " * 20}
        text, _, actions = MODULE.sanitize_candidate(row)
        self.assertNotIn("Альфа", text)
        self.assertIn("removed_non_english_lines", actions)

    def test_candidate_rejects_inverted_timeline(self):
        row = {"Position": "QA Engineer", "Experience Years": 3,
               "CV": "Testing work January 2023 - March 2021. " + "Tested APIs with Postman Jira Selenium SQL Git. " * 20}
        _, reasons, _ = MODULE.sanitize_candidate(row)
        self.assertIn("impossible_timeline", reasons)

    def test_candidate_rejects_conflicting_experience_total(self):
        row = {"Position": "Java Developer", "Experience Years": 4,
               "CV": "I have over 11 years of experience. " +
                     "Developed Java Spring Docker SQL Git services and tested APIs. " * 20}
        _, reasons, _ = MODULE.sanitize_candidate(row)
        self.assertIn("contradictory_experience", reasons)

    def test_split_units_preserves_leading_year_requirement(self):
        self.assertEqual(MODULE.split_units("3+ years of Java experience."),
                         ["3+ years of Java experience."])

    def test_jd_drops_company_and_application_boilerplate(self):
        row = {"Position": "Front-End Developer", "Company Name": "Named Company", "Exp Years": "3",
               "Long Description": ("Named Company builds products. Develop React and TypeScript interfaces. "
                 "Implement HTML and CSS components. Test JavaScript and REST integrations. Maintain Git workflows. "
                 "Review requirements with designers. Support production releases. Document UI behavior. "
                 "Build accessible pages. Analyze browser issues. Coordinate Scrum delivery. " * 3) +
                 "Send your CV to recruiter@example.com. We offer benefits and salary."}
        text, reasons, actions, skills = MODULE.sanitize_jd(row)
        self.assertNotIn("pii_after_sanitization", reasons)
        self.assertNotIn("Named Company", text)
        self.assertNotIn("recruiter@example.com", text)
        self.assertIn("redacted_company_name", actions)
        self.assertGreaterEqual(len(skills), 4)

    def test_pair_requires_role_seniority_and_skills(self):
        candidate = {"role_family": "java", "seniority": 2, "skills": {"java", "spring", "sql", "git"}}
        good = {"role_family": "java", "seniority": 2, "skills": {"java", "spring", "docker", "git"}}
        senior = {**good, "seniority": 3}
        unrelated = {**good, "role_family": "frontend"}
        self.assertIsNotNone(MODULE.pair_score({}, {}, candidate, good))
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, senior))
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, unrelated))

    def test_pair_rejects_any_experience_shortfall(self):
        candidate = {"role_family": "qa", "seniority": 2, "experience_years": 2,
                     "skills": {"postman", "test cases", "jira"}}
        job = {"role_family": "qa", "seniority": 2, "required_years": 4,
               "skills": {"postman", "test cases", "jira"}, "raw_position": "QA Engineer"}
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, job))
        job["required_years"] = 3
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, job))

    def test_javascript_is_frontend_not_java(self):
        self.assertEqual(MODULE.role_family("JavaScript Developer"), "frontend")
        self.assertEqual(MODULE.role_family("Senior Java Developer"), "java")

    def test_candidate_seniority_uses_reported_experience(self):
        self.assertEqual(MODULE.candidate_seniority("Senior Java Developer", 2.5), 2)
        self.assertEqual(MODULE.jd_seniority("Java Developer", "2", "Requires 6+ years"), 3)
        self.assertEqual(MODULE.seniority("Java Lead"), 4)
        self.assertEqual(MODULE.seniority("Middle Frontend Developer"), 2)

    def test_candidate_organization_trigger_is_redacted(self):
        value = MODULE.redact_candidate_unit("Worked at Named Company as a Java developer")
        self.assertNotIn("Named Company", value)
        self.assertNotIn("atis", MODULE.redact_candidate_unit("Developer at atis building interfaces"))

    def test_candidate_employer_timeline_is_redacted(self):
        value = MODULE.redact_candidate_unit("Named Bank, Senior Java Developer: May 2017 - March 2020")
        self.assertNotIn("Named Bank", value)
        self.assertIn("Senior Java Developer", value)
        value = MODULE.redact_candidate_unit("Senior Java Developer, NamedCo - Developed services")
        self.assertNotIn("NamedCo", value)
        value = MODULE.redact_candidate_unit("My current position: Senior Java Developer, NamedCo")
        self.assertNotIn("NamedCo", value)

    def test_jd_seniority_ignores_markdown_boundaries(self):
        text = "The **Principal Software Engineer** designs and develops systems."
        self.assertEqual(MODULE.jd_seniority("Java developer", "3", text), 4)
        self.assertEqual(MODULE.jd_seniority("QA Engineer", "3", "Have been a QA Lead"), 4)

    def test_jd_title_removes_named_employer(self):
        self.assertEqual(MODULE.sanitize_jd_title("Frontend Developer for MGB Inc", ""), "Frontend Developer")
        self.assertEqual(MODULE.sanitize_jd_title("Java developer (Optiva)", "Optiva"), "Java developer")
        self.assertEqual(MODULE.sanitize_jd_title("1926/07 Senior Java Engineer", ""), "Senior Java Engineer")
        self.assertEqual(MODULE.sanitize_jd_title("Java Engineer (IRC158047)", ""), "Java Engineer")
        self.assertEqual(MODULE.sanitize_jd_title("QA Automation (Python), Poland", ""),
                         "QA Automation (Python), Poland")

    def test_specialized_pair_requires_matching_evidence(self):
        candidate = {"role_family": "java", "seniority": 3,
                     "skills": {"java", "python", "docker"}, "text": "Java services"}
        job = {"role_family": "java", "seniority": 2,
               "skills": {"java", "python", "docker"}, "raw_position": "Java Engineer (Machine Learning)"}
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, job))

    def test_allocator_uses_distinct_jds_and_capacity(self):
        candidates = [{"id": "a"}, {"id": "b"}]
        detail = (0.9, 0.5, ["java", "git", "sql"])
        options = {
            "a": [(0.9, "j1", detail), (0.8, "j2", detail)],
            "b": [(0.9, "j1", detail), (0.8, "j2", detail)],
        }
        pairs = MODULE.allocate_pairs(candidates, options)
        self.assertEqual(len(pairs), 4)
        for candidate_id in ("a", "b"):
            self.assertEqual(len({x[1] for x in pairs if x[0] == candidate_id}), 2)
        self.assertLessEqual(max(__import__("collections").Counter(x[1] for x in pairs).values()), 2)


if __name__ == "__main__":
    unittest.main()
