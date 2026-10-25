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
        for phrase in ("I want to learn AWS.", "I am searching for a friendly team.",
                       "Professional growth and development opportunities.",
                       "Currently seeking new opportunities as a frontend developer.",
                       "I am seeking an engaging company in frontend development.",
                       "I am highly motivated to achieve new heights in my career.",
                       "I am always eager to develop my skills and abilities.",
                       "Interested in learning automation testing.",
                       "Looking for a full time remote position.",
                       "Now I'm studying to automate testing on Java.",
                       "Obtain a QA Engineer position in a product company.",
                       "Not interested in legacy projects.",
                       "Would be nice to grow into Automation Quality Assurance.",
                       "I enjoy studying and developing myself.",
                       "I am interesed in team leading.",
                       "My expectations for my next career move include challenging projects.",
                       "Looking forward to join a corporation where I can create value.",
                       "I am confident that I can make a significant contribution.",
                       "For me good work is where I can improve my skills.",
                       "I continuously expand my knowledge and stay up-to-date.",
                       "I am eager to participate in innovative web development.",
                       "I have a strong desire to continuously learn.",
                       "Advancing expertise in frontend frameworks like React.",
                       "Self-development direction: Principal Software Engineer.",
                       "An opportunity to learn and collaborate with experienced colleagues is valuable.",
                       "My goal for improvement is to stay current with industry trends.",
                       "I am looking forward to helping you build modern interfaces.",
                       "Attaining the position of Technical Leader would be desirable.",
                       "I am open to opportunities that offer professional growth.",
                       "I wish I could learn more about automation testing.",
                       "I would be happy to consider opportunities for further development.",
                       "I am looking to continually improve my testing skills.",
                       "This challenge allowed me to learn and grow as a developer.",
                       "Testing gave me opportunities to develop and improve my skills.",
                       "I am flexible and always open to new knowledge.",
                       "I am excited to join a collaborative team.",
                       "I am excited to combine my skills in web development.",
                       "My experience makes me a suitable candidate for the position.",
                       "I have leadership qualities and strive for high quality.",
                       "Let's discuss the details; write me to schedule a convenient time.",
                       "Can work on a new project or support an existing one—no preferences.",
                       "I like to use WebFlux, REST, SQL, and Spring.",
                       "I value a company culture that encourages professional development."):
            changed = {**row, "CV": row["CV"] + "\n" + phrase}
            changed_text, _, _ = MODULE.sanitize_candidate(changed)
            self.assertNotIn(phrase, changed_text)

    def test_candidate_drops_self_identifier(self):
        row = {"Position": "Frontend Developer", "Experience Years": 2, "CV": """
        My name is Yaroslav, and I am a frontend developer.
        Developed React and JavaScript interfaces with responsive HTML and CSS components.
        Implemented Redux state management and tested REST API integrations.
        Maintained Git workflows and reviewed pull requests with the development team.
        Built accessible TypeScript pages and analyzed browser compatibility defects.
        Documented UI behavior and coordinated Scrum delivery through Jira.
        Supported production releases and investigated frontend incidents.
        """}
        text, _, _ = MODULE.sanitize_candidate(row)
        self.assertNotIn("Yaroslav", text)

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
        row["Experience Years"] = 5
        row["CV"] = row["CV"].replace("over 11 years", "6 years")
        _, reasons, _ = MODULE.sanitize_candidate(row)
        self.assertIn("contradictory_experience", reasons)
        row["CV"] = row["CV"].replace("6 years", "4+ years")
        _, reasons, _ = MODULE.sanitize_candidate(row)
        self.assertNotIn("contradictory_experience", reasons)

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

    def test_jd_company_redaction_does_not_corrupt_larger_words(self):
        row = {"Position": "Java Developer", "Company Name": "Block", "Exp Years": "3",
               "Long Description": ("Support developer teams for Block and blockchain releases. "
                 "Our mission is to help every developer grow. Develop Java and Spring services. "
                 "Implement SQL persistence and test REST APIs. Maintain Git and Docker workflows. "
                 "Review code and document service behavior. Support production releases. " * 3)}
        text, _, _, _ = MODULE.sanitize_jd(row)
        self.assertIn("blockchain", text)
        self.assertNotIn("[Employer withheld]chain", text)
        self.assertNotIn("Our mission", text)

    def test_jd_drops_company_marketing_without_structured_name(self):
        row = {"Position": "Front-End Developer", "Company Name": "", "Exp Years": "3",
               "Long Description": ("They develop and market a leading documentation platform. "
                 "Their claim is to provide the best solution worldwide. "
                 "An award-winning developer with offices in several cities serves a global audience. "
                 "From company parties to personal development, our culture and people drive success. "
                 "Cool, centrally located offices create good working conditions. "
                 "Develop React and TypeScript interfaces. Implement HTML and CSS components. "
                 "Test JavaScript and REST integrations. Maintain Git workflows. Review requirements. "
                 "Support production releases. Document UI behavior. Build accessible pages. " * 3)}
        text, _, _, _ = MODULE.sanitize_jd(row)
        self.assertNotIn("They develop and market", text)
        self.assertNotIn("Their claim", text)
        self.assertNotIn("award-winning", text)
        self.assertNotIn("company parties", text)
        self.assertNotIn("working conditions", text)

    def test_jd_redacts_branded_product_but_preserves_technology(self):
        row = {"Position": "Java Engineer", "Company Name": "Different Employer", "Exp Years": "3",
               "Long Description": ("Design and implement Varjo backend applications. "
                 "Develop Java backend services with Spring Boot. Implement SQL persistence. "
                 "Test REST APIs and maintain Git and Docker workflows. Review code. "
                 "Support production releases and document service behavior. " * 3)}
        text, _, actions, _ = MODULE.sanitize_jd(row)
        self.assertNotIn("Varjo", text)
        self.assertIn("Java backend services", text)
        self.assertIn("redacted_product_names", actions)

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

    def test_qa_java_automation_requires_java_evidence(self):
        candidate = {"role_family": "qa", "seniority": 2, "experience_years": 4,
                     "skills": {"test automation", "selenium", "jmeter"}, "text": "Selenium JMeter"}
        job = {"role_family": "qa", "seniority": 2, "required_years": 3,
               "skills": {"test automation", "selenium", "jmeter"},
               "raw_position": "QA Automation Engineer Java"}
        self.assertIsNone(MODULE.pair_score({}, {}, candidate, job))
        candidate["skills"].add("java")
        self.assertIsNotNone(MODULE.pair_score({}, {}, candidate, job))

    def test_javascript_is_frontend_not_java(self):
        self.assertEqual(MODULE.role_family("JavaScript Developer"), "frontend")
        self.assertEqual(MODULE.role_family("Senior Java Developer"), "java")
        self.assertEqual(MODULE.skill_set("JavaScript, React and CSS"), {"javascript", "react", "css"})
        self.assertIn("java", MODULE.skill_set("Java, Spring Boot and Docker"))

    def test_candidate_seniority_uses_reported_experience(self):
        self.assertEqual(MODULE.candidate_seniority("Senior Java Developer", 2.5), 2)
        self.assertEqual(MODULE.jd_seniority("Java Developer", "2", "Requires 6+ years"), 3)
        self.assertEqual(MODULE.seniority("Java Lead"), 4)
        self.assertEqual(MODULE.seniority("Middle Frontend Developer"), 2)

    def test_candidate_organization_trigger_is_redacted(self):
        value = MODULE.redact_candidate_unit("Worked at Named Company as a Java developer")
        self.assertNotIn("Named Company", value)
        self.assertNotIn("atis", MODULE.redact_candidate_unit("Developer at atis building interfaces"))
        value = MODULE.redact_candidate_unit("Netcracker (Provider BSS and OSS): Java developer")
        self.assertNotIn("Netcracker", value)
        value = MODULE.redact_candidate_unit("Front-End Developer at Upwork (Outsourcing industry)")
        self.assertNotIn("Upwork", value)
        self.assertIn("(Outsourcing industry)", value)
        value = MODULE.redact_candidate_unit(
            'Led two projects "IceCream" and "Filmoteka" using React and Redux')
        self.assertNotIn("IceCream", value)
        self.assertNotIn("Filmoteka", value)
        value = MODULE.redact_candidate_unit(
            "So Yummy [JS, React, Redux] - a digital cookbook with authorization")
        self.assertNotIn("So Yummy", value)

    def test_candidate_organization_names_are_reused_for_redaction(self):
        raw = "Netcracker (Provider BSS and OSS): Java developer\nTechnologies: Netcracker ORM"
        self.assertEqual(MODULE.candidate_organization_names(raw), {"Netcracker"})
        row = {"Position": "Java Developer", "Experience Years": 5,
               "CV": raw + "\n" +
                     "Developed Java Spring Docker SQL Git REST services and tested APIs.\n" * 15}
        text, _, actions = MODULE.sanitize_candidate(row)
        self.assertNotIn("Netcracker", text)
        self.assertIn("redacted_organization_names", actions)
        names = MODULE.candidate_organization_names(
            "Frontend Developer Internship at codeVillage.\nCompleted a six-month internship.")
        self.assertIn("codeVillage", names)
        self.assertNotIn("codeVillage.\nCompleted", names)
        names = MODULE.candidate_organization_names(
            "Worked in the company Vorodis as a developer.\n"
            "Developed a store for the company PowerLife, It-Group.")
        self.assertEqual(names, {"Vorodis", "PowerLife", "It-Group"})
        names = MODULE.candidate_organization_names(
            "Built a toolbox for Kaospilot company.\n"
            "Senior Java Developer in AskFm social network product.\n"
            "Developed features for Jumia online marketplace.\n"
            "Worked at an international IT company 'Amdaris'.")
        self.assertEqual(names, {"Kaospilot", "AskFm", "Jumia", "Amdaris"})

    def test_candidate_product_names_are_reused_for_redaction(self):
        raw = ("TechBoss is an e-commerce platform for merchants.\n"
               "Developed the TechBoss mobile apps.\n"
               "Developed the interface of Adfury and wrote its technical documentation.\n"
               "Worked on the Filmoteka project.\n"
               "Developed an application, the Visa Exception System, for travel workflows.\n"
               'Led projects "IceCream" and "Filmoteka Plus" using React.\n'
               "So Yummy [JS, React, Redux] - a digital cookbook.\n"
               "Filmoteka - (HTML, CSS, SASS, JS, REST API) responsive website.\n"
               "MIMIHO [HTML, CSS, JS, Vite]\n"
               "Developed the Mimino website with an adaptive layout.\n"
               "1 Sonalysis (in progress) Sonalysis is an analytics application.\n"
               "2 Resume builder: A resume builder for document creation.\n"
               "5 The Bulb Africa website : a company website.\n"
               "Created the Unified Information Portal of Roshydromet.\n")
        self.assertEqual(MODULE.candidate_product_names(raw),
                         {"TechBoss", "Adfury", "Filmoteka", "Visa Exception System",
                          "IceCream", "Filmoteka Plus", "So Yummy", "MIMIHO", "Mimino",
                          "Sonalysis", "Resume builder", "The Bulb Africa", "Roshydromet"})
        row = {"Position": "Frontend Developer", "Experience Years": 4,
               "CV": raw + "Developed React JavaScript TypeScript HTML CSS Redux interfaces.\n" * 15}
        text, _, actions = MODULE.sanitize_candidate(row)
        self.assertNotIn("TechBoss", text)
        self.assertNotIn("Adfury", text)
        self.assertNotIn("Filmoteka", text)
        self.assertNotIn("Visa Exception System", text)
        self.assertIn("redacted_product_names", actions)

    def test_candidate_redacts_course_provider_and_slash_separated_platforms(self):
        raw = ("Completed a FrontEnd Development course at DAN.IT EDUCATION with high grades.\n"
               "Attended an online course of QA Testlab.\n"
               "CMS Developer at Q-team/Freelancehunt/Upwork.\n")
        row = {"Position": "Frontend Developer", "Experience Years": 2,
               "CV": raw +
                     "Developed React JavaScript TypeScript HTML CSS interfaces and tested REST APIs.\n" * 15}
        text, _, actions = MODULE.sanitize_candidate(row)
        self.assertNotIn("DAN.IT EDUCATION", text)
        self.assertNotIn("QA Testlab", text)
        self.assertNotIn("Freelancehunt", text)
        self.assertNotIn("Upwork", text)
        self.assertIn("redacted_organization_names", actions)

    def test_candidate_drops_course_title_and_redacts_organization_suffix(self):
        raw = ("How to Build a Startup for Remote Drone Operations.\n"
               "In this Project/Course we are building a Cloud Application.\n"
               "Acme: Acme helps founders build sustainable businesses.\n"
               "Developed REST services for accounts of [Organization withheld]: ticos-systems.\n"
               "CMS Developer at Q-team/Freelancehunt/Upwork.\n"
               "Developed React JavaScript TypeScript HTML CSS interfaces and tested REST APIs.\n"
               "developed react javascript typescript html css interfaces and tested rest apis.\n")
        row = {"Position": "Frontend Developer", "Experience Years": 2,
               "CV": raw +
                     "Developed React JavaScript TypeScript HTML CSS interfaces and tested REST APIs.\n" * 15}
        text, _, _ = MODULE.sanitize_candidate(row)
        self.assertNotIn("How to Build", text)
        self.assertNotIn("Project/Course", text)
        self.assertNotIn("helps founders", text)
        self.assertNotIn("ticos-systems", text)
        self.assertEqual(text.casefold().count("developed react javascript typescript html css"), 1)

    def test_candidate_employer_timeline_is_redacted(self):
        value = MODULE.redact_candidate_unit("Named Bank, Senior Java Developer: May 2017 - March 2020")
        self.assertNotIn("Named Bank", value)
        self.assertIn("Senior Java Developer", value)
        value = MODULE.redact_candidate_unit("Senior Java Developer, NamedCo - Developed services")
        self.assertNotIn("NamedCo", value)
        value = MODULE.redact_candidate_unit("My current position: Senior Java Developer, NamedCo")
        self.assertNotIn("NamedCo", value)
        value = MODULE.redact_candidate_unit(
            "Jul 2018 - Jan 2020, Junior Front End developer (webisland.agency)")
        self.assertNotIn("webisland.agency", value)
        value = MODULE.redact_candidate_unit(
            "Luxoft – Senior Java Developer (Krakow, Poland February 2019 – till now)")
        self.assertNotIn("Luxoft", value)
        value = MODULE.redact_candidate_unit("Content manager - ViDi AutoCity (Automotive business)")
        self.assertNotIn("ViDi AutoCity", value)
        value = MODULE.redact_candidate_unit(
            "Siduna & Ndou Engineering: Designed and developed the site")
        self.assertNotIn("Siduna & Ndou Engineering", value)
        value = MODULE.redact_candidate_unit(
            "Developed a site for a real estate company, improving user experience")
        self.assertIn("improving user experience", value)
        self.assertNotIn("[Organization withheld] user experience", value)

    def test_candidate_redacts_non_technology_domains(self):
        value = MODULE.redact_candidate_unit(
            "Built webisland.agency using React.js, ASP.NET, Socket.io, Qase.io and Cypress.io")
        self.assertNotIn("webisland.agency", value)
        for technology in ("React.js", "ASP.NET", "Socket.io", "Qase.io", "Cypress.io"):
            self.assertIn(technology, value)
        value = MODULE.redact_candidate_unit(
            'Completed test automation training at “ACADEMIA TESTARII” with a diploma')
        self.assertNotIn("ACADEMIA TESTARII", value)
        value = MODULE.redact_candidate_unit("EVDEV, Ukraine — Front-end Developer")
        self.assertNotIn("EVDEV", value)
        value = MODULE.redact_candidate_unit(
            "Developed the site for Siduna and Ndou Engineering")
        self.assertNotIn("Siduna", value)
        value = MODULE.redact_candidate_unit(
            "Developed the interface of TechBoss and wrote technical documentation")
        self.assertNotIn("TechBoss", value)

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

    def test_jd_rejects_malformed_and_language_restricted_titles(self):
        body = ("Develop and test Java Spring SQL Git REST services. "
                "Implement APIs and review code with the team. " * 10)
        malformed = {"Position": ("QA automatization Good knowledge SQL database Experience in testing projects "
                                  "based on microservices Basic knowledge JAVA Knowledge of Jira will be a plus"),
                     "Company Name": "", "Exp Years": "3", "Long Description": body}
        _, reasons, _, _ = MODULE.sanitize_jd(malformed)
        self.assertIn("invalid_jd_title", reasons)
        language = {**malformed, "Position": "Java developer with Lithuanian language"}
        _, reasons, _, _ = MODULE.sanitize_jd(language)
        self.assertIn("unsupported_language_requirement", reasons)

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
