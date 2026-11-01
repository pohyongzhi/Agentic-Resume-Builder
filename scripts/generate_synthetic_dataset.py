"""Generate the project's fictional 100-resume / 25-JD evaluation dataset.

Occupation duties and skill families are paraphrased from O*NET OnLine occupation
profiles. All people, employers, universities, projects, and performance figures in
the generated files are fictional. Run from the repository root:

    python scripts/generate_synthetic_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = ROOT / "data" / "resumes"
JD_DIR = ROOT / "data" / "jds"


def occupation(title, code, degree, company, focus, skills, duties, evidence, keywords):
    return {
        "title": title,
        "code": code,
        "source": f"https://www.onetonline.org/link/summary/{code}",
        "degree": degree,
        "company": company,
        "focus": focus,
        "skills": skills,
        "duties": duties,
        "evidence": evidence,
        "keywords": keywords,
    }


ROLES = [
    occupation(
        "Junior Data Analyst", "15-2051.01", "Information Systems", "Example Retail Group",
        "commercial reporting and sales analysis",
        ["Excel", "Power BI", "SQL", "Python", "data cleaning", "dashboard development", "sales analysis", "stakeholder communication"],
        ["Build and maintain commercial dashboards", "Write SQL queries to extract and clean data", "Analyse sales performance and explain trends", "Automate recurring reports with Python", "Document data definitions and reporting logic"],
        ["Built weekly sales dashboards in Excel and Power BI for a regional team.", "Cleaned and merged shipment records from three spreadsheets using SQL queries.", "Compared monthly sales performance by product category and prepared a short briefing.", "Used Python to combine recurring CSV extracts for a student analytics project.", "Documented field definitions for a shared reporting workbook.", "Checked dashboard totals against source files and corrected mapping errors.", "Mapped an order-to-delivery process and suggested two data-entry improvements.", "Answered reporting questions and wrote concise user guidance."],
        [("SQL", []), ("Power BI", ["Tableau"]), ("dashboard", ["dashboards"]), ("Excel", []), ("Python", []), ("data cleaning", ["cleaned", "clean data"]), ("sales analysis", ["sales performance"]), ("stakeholder communication", ["stakeholder", "stakeholders"]), ("automation", ["automate", "automated"]), ("data documentation", ["data definitions", "reporting logic"])],
    ),
    occupation(
        "Junior Data Scientist", "15-2051.00", "Data Science", "Northstar Mobility Labs",
        "predictive modelling and experiment analysis",
        ["Python", "pandas", "scikit-learn", "SQL", "statistics", "data visualisation", "Git", "model evaluation"],
        ["Prepare structured datasets for analysis", "Build and evaluate predictive models", "Explain model results to non-technical partners", "Write reproducible Python notebooks", "Monitor data and model quality"],
        ["Prepared trip records with pandas and resolved missing values before modelling.", "Compared regression and tree-based models using cross-validation.", "Created Python charts explaining demand patterns to a project supervisor.", "Queried a PostgreSQL database to assemble training features.", "Tracked notebook and code changes with Git branches.", "Calculated precision, recall, and confusion matrices for a classification assignment.", "Tested feature distributions for drift between two monthly datasets.", "Wrote a model card describing assumptions, limitations, and evaluation results."],
        [("Python", []), ("pandas", []), ("scikit-learn", ["sklearn"]), ("SQL", ["PostgreSQL"]), ("statistics", ["statistical analysis"]), ("data visualisation", ["visualization"]), ("Git", []), ("predictive modelling", ["machine learning", "predictive model"]), ("model evaluation", ["cross-validation"]), ("data quality", ["missing values", "data validation"])],
    ),
    occupation(
        "Business Analyst", "13-1111.00", "Business Analytics", "CivicBridge Services",
        "requirements analysis and process improvement",
        ["requirements gathering", "process mapping", "Excel", "SQL", "Power BI", "user stories", "stakeholder interviews", "UAT"],
        ["Gather and document business requirements", "Map current and future processes", "Translate needs into user stories", "Support user acceptance testing", "Prepare decision-ready analysis"],
        ["Interviewed five users and consolidated their reporting requirements.", "Mapped an approval workflow in Visio and identified duplicate review steps.", "Converted workshop notes into user stories with acceptance criteria.", "Prepared UAT scenarios and recorded outcomes in a defect log.", "Analysed service volumes in Excel and highlighted peak periods.", "Queried a training database with SQL to verify business rules.", "Built a Power BI page summarising turnaround times.", "Maintained a requirements traceability table for a semester project."],
        [("requirements gathering", ["business requirements"]), ("process mapping", ["workflow mapping"]), ("user stories", ["acceptance criteria"]), ("UAT", ["user acceptance testing"]), ("stakeholder interviews", ["workshops"]), ("Excel", []), ("SQL", []), ("Power BI", []), ("business rules", []), ("requirements traceability", ["traceability matrix"])],
    ),
    occupation(
        "Junior Software Developer", "15-1252.00", "Computer Science", "BlueFern Software",
        "web application development and maintenance",
        ["Python", "JavaScript", "React", "REST APIs", "Git", "unit testing", "SQL", "Agile"],
        ["Implement maintainable application features", "Integrate front ends with REST APIs", "Write automated unit tests", "Review code and resolve defects", "Collaborate through an Agile delivery process"],
        ["Implemented a Python API endpoint for a campus booking application.", "Built reusable React components for search and filtering.", "Connected a JavaScript interface to REST API responses.", "Wrote unit tests for validation and date-handling functions.", "Used Git pull requests to review code with three teammates.", "Designed SQL tables for users, reservations, and audit events.", "Investigated application logs and fixed a duplicate-submission defect.", "Participated in weekly sprint planning and retrospectives."],
        [("Python", []), ("JavaScript", []), ("React", []), ("REST API", ["REST APIs"]), ("Git", ["GitHub"]), ("unit testing", ["automated tests"]), ("SQL", []), ("Agile", ["Scrum", "sprint"]), ("code review", ["pull requests"]), ("debugging", ["defect resolution"])],
    ),
    occupation(
        "Junior QA Analyst", "15-1253.00", "Software Engineering", "QualityHarbour Technologies",
        "manual and automated software testing",
        ["test cases", "Jira", "Selenium", "Postman", "SQL", "regression testing", "defect reporting", "Git"],
        ["Design and execute functional test cases", "Document defects with reproducible evidence", "Run regression tests before releases", "Test APIs and data outcomes", "Maintain basic automated test scripts"],
        ["Designed functional test cases from acceptance criteria for a booking portal.", "Logged defects in Jira with steps, screenshots, and expected results.", "Ran regression checks across Chrome and Edge before a class release.", "Used Postman to verify API status codes and response fields.", "Queried SQL tables to confirm transaction records after tests.", "Updated Selenium scripts for login and form-validation scenarios.", "Retested resolved defects and maintained a release checklist.", "Summarised test coverage and open risks for the development team."],
        [("test cases", ["test scenarios"]), ("Jira", ["bug tracking"]), ("Selenium", ["test automation"]), ("Postman", ["API testing"]), ("SQL", []), ("regression testing", ["regression tests"]), ("defect reporting", ["defect log", "bug report"]), ("Git", []), ("cross-browser testing", ["browser compatibility"]), ("release readiness", ["release checklist"])],
    ),
    occupation(
        "Junior Cybersecurity Analyst", "15-1212.00", "Cybersecurity", "RedMaple Security",
        "security monitoring and incident response",
        ["SIEM", "incident response", "Wireshark", "Linux", "vulnerability assessment", "network security", "Python", "security documentation"],
        ["Monitor alerts and investigate suspicious activity", "Support incident triage and documentation", "Review vulnerabilities and remediation status", "Analyse network traffic and system logs", "Maintain security procedures and evidence"],
        ["Reviewed simulated SIEM alerts and documented triage decisions.", "Analysed packet captures in Wireshark to identify unusual DNS traffic.", "Used Linux commands to inspect authentication and web-server logs.", "Scanned a lab environment and prioritised vulnerabilities by severity.", "Wrote a Python script that parsed failed-login events from a text file.", "Created an incident timeline from endpoint and firewall evidence.", "Updated a security checklist for account provisioning and removal.", "Presented remediation findings from a controlled penetration-testing lab."],
        [("SIEM", ["security monitoring"]), ("incident response", ["incident triage"]), ("Wireshark", ["packet analysis"]), ("Linux", []), ("vulnerability assessment", ["vulnerability scanning"]), ("network security", ["firewall"]), ("Python", []), ("log analysis", ["system logs"]), ("security documentation", ["incident documentation"]), ("remediation", ["risk remediation"])],
    ),
    occupation(
        "Cloud Support Associate", "15-1232.00", "Information Technology", "NimbusCove Cloud Services",
        "cloud operations and technical support",
        ["AWS", "Linux", "TCP/IP", "ticketing", "Python", "cloud monitoring", "IAM", "troubleshooting"],
        ["Resolve cloud service tickets", "Monitor platform health and escalate incidents", "Troubleshoot Linux and network issues", "Support access and identity requests", "Document repeatable support procedures"],
        ["Resolved simulated support tickets involving storage permissions and service availability.", "Configured AWS EC2 and S3 resources for a cloud computing assignment.", "Used Linux tools to diagnose disk, process, and connectivity issues.", "Checked TCP/IP settings and DNS resolution during lab incidents.", "Created CloudWatch alarms for CPU and storage thresholds.", "Applied IAM roles following least-privilege requirements.", "Wrote a Python health-check script for three test endpoints.", "Documented troubleshooting steps in a shared knowledge base."],
        [("AWS", ["Amazon Web Services"]), ("Linux", []), ("TCP/IP", ["networking"]), ("ticketing", ["support tickets"]), ("Python", []), ("cloud monitoring", ["CloudWatch"]), ("IAM", ["identity and access management"]), ("troubleshooting", ["technical support"]), ("DNS", []), ("knowledge base", ["support documentation"])],
    ),
    occupation(
        "Junior Network Administrator", "15-1244.00", "Network Engineering", "HarbourLink Networks",
        "network operations and user connectivity",
        ["Cisco IOS", "TCP/IP", "DNS", "DHCP", "Wireshark", "VPN", "network monitoring", "technical support"],
        ["Configure and support network devices", "Monitor availability and investigate outages", "Maintain addressing and name services", "Support secure remote connectivity", "Document network changes and diagrams"],
        ["Configured VLANs and access ports on Cisco lab switches.", "Assigned IPv4 subnets and verified routes between three networks.", "Troubleshot DNS and DHCP issues in a virtual lab.", "Captured traffic with Wireshark to diagnose retransmissions.", "Configured a test VPN and validated remote access.", "Monitored device availability with an open-source monitoring dashboard.", "Updated a network diagram and device inventory after changes.", "Resolved classroom connectivity tickets and recorded root causes."],
        [("Cisco IOS", ["Cisco"]), ("TCP/IP", ["IPv4"]), ("DNS", []), ("DHCP", []), ("Wireshark", ["packet capture"]), ("VPN", ["remote access"]), ("network monitoring", ["availability monitoring"]), ("VLAN", ["VLANs"]), ("technical support", ["connectivity tickets"]), ("network documentation", ["network diagram"])],
    ),
    occupation(
        "Junior Database Administrator", "15-1242.00", "Information Systems", "DataGrove Platforms",
        "database reliability and access management",
        ["PostgreSQL", "SQL", "database backup", "access control", "query optimisation", "data integrity", "Linux", "monitoring"],
        ["Maintain relational database environments", "Manage user access and permissions", "Monitor queries and capacity", "Support backup and recovery tests", "Investigate data integrity issues"],
        ["Designed PostgreSQL tables with primary and foreign key constraints.", "Created SQL roles and tested read-only permissions in a lab database.", "Executed backup and restore procedures for a sample application.", "Reviewed slow queries with execution plans and added an index.", "Checked duplicate and orphaned records with data-quality queries.", "Monitored storage use and connection counts in a test environment.", "Used Linux shell commands to manage database log files.", "Documented recovery steps and verified restored row counts."],
        [("PostgreSQL", []), ("SQL", []), ("database backup", ["backup and recovery"]), ("access control", ["permissions", "SQL roles"]), ("query optimisation", ["query optimization", "execution plans"]), ("data integrity", ["data quality"]), ("Linux", []), ("monitoring", ["database monitoring"]), ("indexing", ["index"]), ("relational database", ["database design"])],
    ),
    occupation(
        "Junior Web Developer", "15-1254.00", "Web Development", "PixelQuay Studio",
        "accessible responsive web development",
        ["HTML", "CSS", "JavaScript", "responsive design", "accessibility", "Git", "REST APIs", "web performance"],
        ["Develop responsive web pages", "Implement accessible interface behaviour", "Integrate pages with APIs", "Test compatibility and performance", "Maintain code through version control"],
        ["Built responsive pages with semantic HTML, CSS Grid, and media queries.", "Added keyboard navigation and accessible labels to an event website.", "Implemented JavaScript form validation with clear error messages.", "Fetched and displayed REST API data in a course project.", "Used Git branches and pull requests with a four-person team.", "Tested layouts across mobile and desktop browser sizes.", "Reduced image file sizes and measured page loading improvements.", "Documented reusable components and site setup instructions."],
        [("HTML", ["semantic HTML"]), ("CSS", ["CSS Grid"]), ("JavaScript", []), ("responsive design", ["mobile responsive"]), ("accessibility", ["keyboard navigation", "accessible labels"]), ("Git", []), ("REST API", ["REST APIs"]), ("cross-browser testing", ["browser compatibility"]), ("web performance", ["page loading"]), ("documentation", ["setup instructions"])],
    ),
    occupation(
        "Junior UX Designer", "15-1255.00", "Interaction Design", "HumanKind Digital",
        "user research and interface prototyping",
        ["Figma", "wireframing", "prototyping", "user interviews", "usability testing", "design systems", "accessibility", "journey mapping"],
        ["Translate user needs into interface designs", "Create wireframes and interactive prototypes", "Plan and conduct usability tests", "Apply accessible design practices", "Maintain reusable design components"],
        ["Conducted five user interviews for a campus navigation project.", "Mapped the current journey and identified three recurring pain points.", "Created low-fidelity wireframes and an interactive Figma prototype.", "Planned usability tasks and recorded observations from six sessions.", "Revised navigation labels based on test findings.", "Checked colour contrast and keyboard focus requirements.", "Built reusable buttons and form fields in a small design system.", "Presented design decisions with links to user evidence."],
        [("Figma", []), ("wireframing", ["wireframes"]), ("prototyping", ["interactive prototype"]), ("user interviews", ["user research"]), ("usability testing", ["usability tests"]), ("design systems", ["reusable components"]), ("accessibility", ["colour contrast", "keyboard focus"]), ("journey mapping", ["user journey"]), ("information architecture", ["navigation"]), ("research synthesis", ["user evidence"])],
    ),
    occupation(
        "Digital Marketing Analyst", "13-1161.01", "Digital Marketing", "BrightPath Commerce",
        "search and campaign performance analysis",
        ["Google Analytics", "SEO", "Google Ads", "Excel", "A/B testing", "campaign reporting", "keyword research", "data visualisation"],
        ["Analyse digital campaign performance", "Conduct keyword and competitor research", "Maintain search and paid-media reports", "Support landing-page experiments", "Explain channel results and recommendations"],
        ["Built a Google Analytics report for traffic, engagement, and conversion events.", "Researched search terms and grouped keywords by user intent.", "Compared Google Ads campaign results in an Excel pivot table.", "Drafted SEO titles and descriptions for eight product pages.", "Designed an A/B test plan for two landing-page headlines.", "Created charts showing weekly channel performance.", "Reviewed competitor content and summarised topic gaps.", "Presented campaign observations and next-step recommendations."],
        [("Google Analytics", ["web analytics"]), ("SEO", ["search engine optimisation", "search engine optimization"]), ("Google Ads", ["paid search"]), ("Excel", ["pivot table"]), ("A/B testing", ["experiment"]), ("campaign reporting", ["campaign performance"]), ("keyword research", ["search terms"]), ("data visualisation", ["charts"]), ("conversion", ["conversion events"]), ("competitor analysis", ["competitor research"])],
    ),
    occupation(
        "Market Research Analyst", "13-1161.00", "Marketing Analytics", "Meridian Consumer Insights",
        "consumer research and market sizing",
        ["survey design", "Excel", "SPSS", "data analysis", "market research", "presentation", "segmentation", "competitive analysis"],
        ["Design and administer research instruments", "Analyse customer and market data", "Segment respondents and compare behaviour", "Research competitors and category trends", "Present findings with clear limitations"],
        ["Drafted a survey and piloted it with twelve participants.", "Cleaned questionnaire responses and coded open-text categories.", "Analysed cross-tabulations in SPSS for a consumer study.", "Created respondent segments using purchase and preference variables.", "Estimated a classroom market size from public statistics.", "Compared competitor pricing and product features in Excel.", "Built presentation charts with notes on sample limitations.", "Summarised interview themes for a product concept review."],
        [("survey design", ["questionnaire"]), ("Excel", []), ("SPSS", []), ("data analysis", ["cross-tabulation"]), ("market research", ["consumer research"]), ("segmentation", ["respondent segments"]), ("competitive analysis", ["competitor pricing"]), ("market sizing", ["market size"]), ("presentation", ["presentation charts"]), ("research limitations", ["sample limitations"])],
    ),
    occupation(
        "Junior Financial Analyst", "13-2051.00", "Finance", "Oakline Advisory",
        "financial planning and performance reporting",
        ["financial modelling", "Excel", "variance analysis", "forecasting", "Power BI", "financial statements", "scenario analysis", "presentation"],
        ["Maintain financial models and forecasts", "Analyse actual results against plans", "Prepare management reporting", "Review financial statements and operating drivers", "Support scenario and investment analysis"],
        ["Built a three-statement financial model for a listed-company case study.", "Compared monthly actuals with budget and explained major variances.", "Prepared an Excel forecast using volume and price assumptions.", "Created a Power BI page for revenue and expense trends.", "Calculated profitability and liquidity ratios from annual reports.", "Tested upside and downside scenarios for a capital project.", "Checked model formulas and reconciled totals to source statements.", "Presented financial findings and assumptions to a seminar group."],
        [("financial modelling", ["financial model"]), ("Excel", []), ("variance analysis", ["budget variance"]), ("forecasting", ["forecast"]), ("Power BI", []), ("financial statements", ["three-statement"]), ("scenario analysis", ["sensitivity analysis"]), ("management reporting", ["financial reporting"]), ("financial ratios", ["profitability", "liquidity"]), ("reconciliation", ["reconciled"] )],
    ),
    occupation(
        "Junior Accountant", "13-2011.00", "Accountancy", "LedgerStone Services",
        "financial records and month-end support",
        ["general ledger", "Excel", "reconciliation", "accounts payable", "journal entries", "financial reporting", "audit support", "ERP"],
        ["Prepare reconciliations and journal entries", "Support accounts payable and receivable", "Maintain accurate ledger records", "Assist month-end reporting", "Organise schedules and evidence for audit"],
        ["Reconciled a training bank account to the general ledger.", "Prepared sample journal entries with supporting calculations.", "Checked supplier invoices against purchase orders and receipts.", "Maintained an accounts payable ageing schedule in Excel.", "Assisted with a mock month-end close checklist.", "Exported transaction records from a teaching ERP system.", "Prepared a fixed-asset roll-forward for an accounting assignment.", "Organised audit evidence and cross-referenced working papers."],
        [("general ledger", ["ledger"]), ("Excel", []), ("reconciliation", ["reconciled"]), ("accounts payable", ["supplier invoices"]), ("journal entries", ["journals"]), ("financial reporting", ["month-end reporting"]), ("audit support", ["audit evidence"]), ("ERP", ["enterprise resource planning"]), ("month-end close", ["close checklist"]), ("fixed assets", ["fixed-asset"] )],
    ),
    occupation(
        "Human Resources Coordinator", "13-1071.00", "Human Resource Management", "PeopleSpring Group",
        "employee records and recruitment coordination",
        ["HRIS", "recruitment", "onboarding", "Excel", "employee records", "interview scheduling", "HR reporting", "confidentiality"],
        ["Maintain accurate employee records", "Coordinate interviews and candidate communication", "Support onboarding activities", "Prepare routine HR reports", "Handle employee information confidentially"],
        ["Updated fictional employee records in a teaching HRIS environment.", "Coordinated interview times for candidates and three panel members.", "Prepared onboarding checklists and first-day information packs.", "Tracked recruitment stages in an Excel workbook.", "Checked personnel files for missing acknowledgement forms.", "Created a monthly headcount and leave summary.", "Drafted standard candidate emails for review by a supervisor.", "Applied access controls when organising confidential HR documents."],
        [("HRIS", ["human resources information system"]), ("recruitment", ["hiring"]), ("onboarding", ["onboarding checklist"]), ("Excel", []), ("employee records", ["personnel files"]), ("interview scheduling", ["interview coordination"]), ("HR reporting", ["headcount"]), ("confidentiality", ["confidential"]), ("candidate communication", ["candidate emails"]), ("document control", ["access controls"] )],
    ),
    occupation(
        "Learning and Development Coordinator", "13-1151.00", "Organisational Psychology", "GrowWell Academy",
        "training coordination and learning evaluation",
        ["training coordination", "LMS", "facilitation", "needs analysis", "Excel", "learning evaluation", "content development", "stakeholder communication"],
        ["Coordinate training schedules and enrolment", "Maintain learning records in an LMS", "Support needs analysis and course design", "Prepare learning materials", "Evaluate attendance, feedback, and outcomes"],
        ["Scheduled four workshops and maintained participant lists.", "Uploaded course materials and completion records to a sample LMS.", "Summarised staff learning needs from survey responses.", "Created facilitator slides and participant worksheets.", "Supported a peer-learning session and managed questions.", "Analysed attendance and feedback scores in Excel.", "Drafted a post-course evaluation with five rating items.", "Shared training updates with participants and programme sponsors."],
        [("training coordination", ["workshop scheduling"]), ("LMS", ["learning management system"]), ("facilitation", ["facilitator"]), ("needs analysis", ["learning needs"]), ("Excel", []), ("learning evaluation", ["course evaluation"]), ("content development", ["learning materials"]), ("stakeholder communication", ["training updates"]), ("attendance tracking", ["participant lists"]), ("survey", ["feedback scores"] )],
    ),
    occupation(
        "Supply Chain Analyst", "13-1081.00", "Supply Chain Management", "RouteLeaf Distribution",
        "inventory and supply chain performance analysis",
        ["Excel", "SQL", "Power BI", "inventory analysis", "demand planning", "supplier performance", "logistics", "ERP"],
        ["Analyse inventory and fulfilment performance", "Support demand and replenishment planning", "Maintain supplier scorecards", "Build supply chain reports", "Investigate delays and stock discrepancies"],
        ["Analysed inventory turns and stockout events in Excel.", "Queried order records with SQL to compare fulfilment lead times.", "Built a Power BI dashboard for inbound and outbound volumes.", "Prepared a simple demand forecast from twelve months of history.", "Compared supplier delivery performance against agreed dates.", "Mapped a warehouse receiving process and noted two bottlenecks.", "Reviewed ERP extracts for duplicate purchase order lines.", "Summarised late-delivery reasons for a weekly logistics review."],
        [("Excel", []), ("SQL", []), ("Power BI", []), ("inventory analysis", ["inventory turns", "stockout"]), ("demand planning", ["demand forecast"]), ("supplier performance", ["supplier scorecard"]), ("logistics", ["fulfilment"]), ("ERP", []), ("lead time", ["delivery time"]), ("process mapping", ["warehouse process"] )],
    ),
    occupation(
        "Logistics Coordinator", "43-5061.00", "Logistics Management", "TransitField Logistics",
        "shipment coordination and delivery tracking",
        ["shipment tracking", "Excel", "inventory records", "carrier coordination", "delivery scheduling", "ERP", "documentation", "customer communication"],
        ["Coordinate shipment and delivery schedules", "Update transport and inventory records", "Communicate exceptions to carriers and customers", "Prepare shipping documentation", "Track service performance and recurring delays"],
        ["Updated a shipment tracker with pickup and delivery milestones.", "Scheduled delivery windows for a simulated distribution exercise.", "Reconciled carton counts against warehouse dispatch records.", "Prepared sample packing lists and delivery documents.", "Contacted project teammates acting as carriers to resolve timing conflicts.", "Recorded delivery exceptions and customer updates in a case log.", "Exported order status information from a teaching ERP system.", "Summarised on-time delivery performance in Excel."],
        [("shipment tracking", ["delivery tracking"]), ("Excel", []), ("inventory records", ["warehouse records"]), ("carrier coordination", ["carriers"]), ("delivery scheduling", ["delivery windows"]), ("ERP", []), ("shipping documentation", ["packing lists"]), ("customer communication", ["customer updates"]), ("delivery exceptions", ["exceptions"]), ("on-time delivery", ["service performance"] )],
    ),
    occupation(
        "Project Coordinator", "13-1082.00", "Project Management", "Stonebridge Projects",
        "project schedules, risks, and delivery coordination",
        ["project planning", "Microsoft Project", "Excel", "risk register", "meeting minutes", "stakeholder communication", "budget tracking", "Agile"],
        ["Maintain project plans and milestones", "Coordinate meetings and action items", "Track risks, issues, and decisions", "Support budget and resource reporting", "Communicate progress to project stakeholders"],
        ["Created a project schedule with milestones and dependencies.", "Recorded meeting minutes and followed up outstanding action items.", "Maintained a risk register with owners and response actions.", "Tracked planned and actual project expenses in Excel.", "Prepared a weekly status summary for a student project sponsor.", "Updated a responsibility matrix for a five-person delivery team.", "Facilitated a sprint review and documented feedback.", "Organised project files using consistent naming and version control."],
        [("project planning", ["project schedule"]), ("Microsoft Project", ["project management software"]), ("Excel", []), ("risk register", ["project risks"]), ("meeting minutes", ["action items"]), ("stakeholder communication", ["status summary"]), ("budget tracking", ["project expenses"]), ("Agile", ["sprint review"]), ("milestones", ["dependencies"]), ("document control", ["version control"] )],
    ),
    occupation(
        "Operations Research Analyst", "15-2031.00", "Operations Research", "OptimaWorks Consulting",
        "quantitative decision modelling",
        ["Python", "linear programming", "statistics", "Excel Solver", "SQL", "simulation", "data visualisation", "optimisation"],
        ["Structure operational problems as quantitative models", "Analyse constraints and scenarios", "Build optimisation or simulation models", "Validate results and assumptions", "Explain recommendations to decision makers"],
        ["Formulated a linear programming model for a production-mix case.", "Used Excel Solver to compare capacity allocation scenarios.", "Built a Python simulation of queue waiting times.", "Queried operational data with SQL for model inputs.", "Calculated confidence intervals for simulated outcomes.", "Tested model sensitivity to demand and cost assumptions.", "Created charts comparing baseline and optimised scenarios.", "Documented constraints, assumptions, and recommendation limits."],
        [("Python", []), ("linear programming", ["optimisation model"]), ("statistics", ["confidence intervals"]), ("Excel Solver", ["Solver"]), ("SQL", []), ("simulation", ["queue simulation"]), ("data visualisation", ["charts"]), ("optimisation", ["optimization"]), ("sensitivity analysis", ["scenario analysis"]), ("model assumptions", ["constraints"] )],
    ),
    occupation(
        "Customer Success Associate", "43-4051.00", "Business Management", "ClientNest Software",
        "customer onboarding and product adoption",
        ["customer onboarding", "CRM", "ticketing", "product training", "customer communication", "usage reporting", "issue escalation", "knowledge base"],
        ["Guide new customers through onboarding", "Answer product questions and coordinate resolution", "Monitor usage and follow up adoption risks", "Maintain accurate CRM notes", "Create reusable customer guidance"],
        ["Prepared an onboarding checklist for a simulated software customer.", "Answered product questions through a classroom support queue.", "Recorded contact notes and next steps in a sample CRM.", "Created a usage report identifying inactive training accounts.", "Escalated technical issues with clear reproduction steps.", "Delivered a short product walkthrough to six classmates.", "Wrote two knowledge-base articles for common setup questions.", "Summarised customer feedback themes for a product exercise."],
        [("customer onboarding", ["onboarding checklist"]), ("CRM", ["customer relationship management"]), ("ticketing", ["support queue"]), ("product training", ["product walkthrough"]), ("customer communication", ["customer feedback"]), ("usage reporting", ["usage report"]), ("issue escalation", ["escalated"]), ("knowledge base", ["knowledge-base articles"]), ("product adoption", ["inactive accounts"]), ("case notes", ["contact notes"] )],
    ),
    occupation(
        "Sales Operations Analyst", "41-4012.00", "Business Analytics", "RevenueTrail Technologies",
        "sales pipeline and performance reporting",
        ["Salesforce", "Excel", "Power BI", "SQL", "pipeline analysis", "forecasting", "data quality", "sales reporting"],
        ["Maintain sales pipeline reporting", "Analyse conversion and forecast trends", "Improve CRM data quality", "Prepare territory and account analysis", "Support recurring sales reviews"],
        ["Built a Salesforce training report for opportunity stages and values.", "Analysed pipeline conversion rates in Excel.", "Created a Power BI view of bookings by segment.", "Queried sample account data with SQL to check duplicates.", "Prepared a simple sales forecast from weighted opportunities.", "Reviewed missing CRM fields and proposed validation rules.", "Compared territory performance against quarterly targets.", "Produced a weekly sales review pack with documented definitions."],
        [("Salesforce", ["CRM"]), ("Excel", []), ("Power BI", []), ("SQL", []), ("pipeline analysis", ["sales pipeline"]), ("forecasting", ["sales forecast"]), ("data quality", ["validation rules"]), ("sales reporting", ["sales review"]), ("conversion rate", ["conversion rates"]), ("territory analysis", ["territory performance"] )],
    ),
    occupation(
        "Sustainability Coordinator", "13-1199.05", "Environmental Management", "GreenArc Facilities",
        "environmental data and sustainability programmes",
        ["sustainability reporting", "Excel", "carbon accounting", "waste analysis", "stakeholder engagement", "data collection", "green procurement", "project coordination"],
        ["Collect and validate sustainability data", "Support emissions and waste reporting", "Coordinate environmental initiatives", "Research sustainable procurement options", "Communicate progress to internal stakeholders"],
        ["Collected monthly electricity and water data from sample facility records.", "Calculated basic emissions estimates using provided conversion factors.", "Analysed waste categories and identified the largest material stream.", "Built an Excel tracker for sustainability actions and owners.", "Researched environmental criteria for a fictional supplier review.", "Coordinated a campus reuse campaign with four volunteers.", "Prepared charts for a short sustainability progress update.", "Documented data sources and assumptions for a reporting exercise."],
        [("sustainability reporting", ["sustainability progress"]), ("Excel", []), ("carbon accounting", ["emissions estimates"]), ("waste analysis", ["waste categories"]), ("stakeholder engagement", ["internal stakeholders"]), ("data collection", ["facility data"]), ("green procurement", ["supplier criteria"]), ("project coordination", ["action tracker"]), ("energy data", ["electricity"]), ("documentation", ["data sources", "assumptions"] )],
    ),
    occupation(
        "Junior Industrial Engineer", "17-2112.00", "Industrial Engineering", "FlowForge Manufacturing",
        "process improvement and production analysis",
        ["process mapping", "time study", "Excel", "Lean", "quality analysis", "capacity planning", "AutoCAD", "data analysis"],
        ["Analyse production processes and capacity", "Conduct time and workflow studies", "Support quality and waste reduction", "Prepare layouts and standard work", "Track improvement results"],
        ["Mapped a simulated assembly process and identified waiting steps.", "Conducted a time study across twelve production cycles.", "Calculated workstation utilisation and capacity in Excel.", "Applied a basic value-stream map to a coursework case.", "Created a simple facility layout in AutoCAD.", "Analysed defect categories with a Pareto chart.", "Drafted standard-work instructions for a lab assembly task.", "Compared cycle times before and after a layout change."],
        [("process mapping", ["workflow study"]), ("time study", ["cycle time"]), ("Excel", []), ("Lean", ["value-stream map"]), ("quality analysis", ["Pareto chart"]), ("capacity planning", ["utilisation"]), ("AutoCAD", ["facility layout"]), ("data analysis", []), ("standard work", ["work instructions"]), ("process improvement", ["waste reduction"] )],
    ),
]


FIRST_NAMES = [
    "Alex", "Aisha", "Benjamin", "Chloe", "Daniel", "Eunice", "Farhan", "Grace", "Harish", "Isabelle",
    "Javier", "Kai", "Leila", "Marcus", "Nadia", "Owen", "Priya", "Qistina", "Ryan", "Sofia",
    "Tariq", "Uma", "Victor", "Wei Lin", "Yasmin",
]
LAST_NAMES = ["Tan", "Lim", "Ng", "Rahman"]
UNIVERSITIES = [
    "Northbridge University", "Harbour City University", "Meridian Institute of Technology",
    "Eastpoint University", "Cedar Valley College", "Riverside Institute",
    "Westlake University", "Summit Metropolitan University",
]
EMPLOYER_PREFIXES = ["Aurora", "Cedar", "Harbour", "Keystone", "Meridian", "Northstar", "Oak", "River", "Summit", "Willow"]
EMPLOYER_NOUNS = ["Bridge", "Cove", "Field", "Forge", "Grove", "Lane", "Peak", "Quay", "Stone", "Trail"]
def candidate_name(index):
    return f"{FIRST_NAMES[index % 25]} {LAST_NAMES[index // 25]}"


def employer_name(index):
    return f"{EMPLOYER_PREFIXES[index % 10]}{EMPLOYER_NOUNS[(index // 10) % 10]} Solutions Pte Ltd"


def clean_title(title):
    return title.removeprefix("Junior ")


def render_resume(candidate_number, role_index, variant):
    if candidate_number == 1:
        return """# Alex Tan (fictional sample candidate)
alex.tan@example.com

## Summary
Final-year Information Systems student with internship experience in data reporting and process improvement.

## Experience
**Business Analyst Intern, Example Logistics Pte Ltd** (May 2025 - Aug 2025)
- Built weekly sales dashboards in Excel and Power BI used by the regional sales team.
- Cleaned and merged shipment data from three spreadsheets using SQL queries.
- Mapped the order-to-delivery process and suggested two changes that cut manual data entry.

**Student Assistant, University IT Helpdesk** (Jan 2024 - Dec 2024)
- Answered student queries on email and walk-in, about 30 cases per week.
- Wrote short how-to guides for common login and Wi-Fi problems.

## Education
Bachelor of Science (Information Systems), Example University, expected 2026

## Skills
Excel, Power BI, SQL (basic), Python (basic), process mapping, stakeholder communication
"""

    role = ROLES[role_index]
    name = candidate_name(candidate_number - 1)
    base_title = clean_title(role["title"])
    first_titles = [
        f"{base_title} Intern",
        f"Junior {base_title}",
        f"{base_title} Project Assistant",
        f"{base_title} Trainee",
    ]
    first_title = first_titles[variant]
    second_titles = ["Data Support Assistant", "Operations Project Assistant", "Research Assistant", "Student Technology Assistant"]
    start_year = 2023 + ((candidate_number + variant) % 2)
    end_year = start_year + 1
    evidence = role["evidence"]
    first_bullets = [evidence[(variant * 2 + offset) % len(evidence)] for offset in range(3)]
    second_bullets = [evidence[(variant * 2 + 4 + offset) % len(evidence)] for offset in range(2)]
    skill_rotation = role["skills"][variant:] + role["skills"][:variant]
    skill_count = 8 - variant
    skills = skill_rotation[:skill_count] + (["Microsoft Office", "team collaboration"] if variant >= 2 else [])
    summary_styles = [
        f"Early-career {base_title.lower()} with project and internship evidence in {role['focus']}, {role['skills'][0]}, and {role['skills'][1]}.",
        f"Final-year {role['degree']} student with applied experience in {role['focus']} through structured team projects.",
        f"Entry-level analyst bringing transferable experience in {role['skills'][0]}, {role['skills'][2]}, and written documentation.",
        f"Final-year student seeking an entry-level {base_title.lower()} role after completing coursework and projects related to {role['focus']}.",
    ]
    university = UNIVERSITIES[(candidate_number + role_index) % len(UNIVERSITIES)]
    graduation = 2026 + (candidate_number % 2)
    return f"""# {name} (fictional sample candidate)
cand{candidate_number:03d}@example.com

## Summary
{summary_styles[variant]}

## Experience
**{first_title}, {employer_name(candidate_number - 1)}** (May {start_year} - Aug {start_year})
{chr(10).join('- ' + item for item in first_bullets)}

**{second_titles[variant]}, {university}** (Jan {end_year} - Dec {end_year})
{chr(10).join('- ' + item for item in second_bullets)}

## Education
Bachelor of Science ({role['degree']}), {university}, expected {graduation}

## Skills
{', '.join(skills)}
"""


def render_jd(jd_number, role):
    requirements = [
        f"Degree or diploma in {role['degree']} or a related field",
        f"Practical exposure to {role['skills'][0]}, {role['skills'][1]}, and {role['skills'][2]}",
        f"Ability to work with {role['skills'][3]} and produce clear documentation",
        "Careful analytical thinking and professional communication",
        "Internship, coursework, or project evidence is welcome for this entry-level role",
    ]
    return f"""# {role['title']} (synthetic evaluation job description)
Company: {role['company']} (fictional)
Occupation reference: {role['source']}

We are looking for a {role['title']} to support {role['focus']}.

Responsibilities:
{chr(10).join('- ' + item for item in role['duties'])}

Requirements:
{chr(10).join('- ' + item for item in requirements)}

This posting is synthetic and was written for evaluation only. It paraphrases the linked O*NET occupation profile; it is not a real vacancy.
"""


def render_sources():
    rows = "\n".join(
        f"| `jd_{index:03d}` | {role['title']} | [{role['code']}]({role['source']}) |"
        for index, role in enumerate(ROLES, 1)
    )
    return f"""# Synthetic dataset provenance

This dataset contains fictional candidates and synthetic job descriptions. No resume belongs to a real person, and no job description is a live vacancy.

The job descriptions paraphrase task and skill families from O*NET OnLine, which is sponsored by the U.S. Department of Labor. O*NET content is available under the [CC BY 4.0 licence](https://www.onetcenter.org/license.html).

| JD | Synthetic role | Occupation reference |
|---|---|---|
{rows}

The `*.keywords.json` files are automatically generated draft labels. Before the final experiment, two group members should independently review each file and reconcile disagreements, as required by the project README.
"""


def validate_dataset(pair_rows):
    resume_files = sorted(RESUME_DIR.glob("cand_[0-9][0-9][0-9].md"))
    jd_files = sorted(JD_DIR.glob("jd_[0-9][0-9][0-9].md"))
    keyword_files = sorted(JD_DIR.glob("jd_[0-9][0-9][0-9].keywords.json"))
    assert len(resume_files) == 100, len(resume_files)
    assert len(jd_files) == 25, len(jd_files)
    assert len(keyword_files) == 25, len(keyword_files)
    assert len(pair_rows) == 100, len(pair_rows)
    assert len({(row["resume_id"], row["jd_id"]) for row in pair_rows}) == 100
    assert set(Counter(row["jd_id"] for row in pair_rows).values()) == {4}
    assert set(Counter(row["resume_id"] for row in pair_rows).values()) == {1}
    assert {row["resume_id"] for row in pair_rows} == {f"cand_{i:03d}" for i in range(1, 101)}
    assert {row["jd_id"] for row in pair_rows} == {f"jd_{i:03d}" for i in range(1, 26)}

    resume_hashes = set()
    names = set()
    emails = set()
    for path in resume_files:
        text = path.read_text(encoding="utf-8")
        for heading in ("## Summary", "## Experience", "## Education", "## Skills"):
            assert heading in text, (path, heading)
        lines = text.splitlines()
        names.add(lines[0])
        emails.add(lines[1])
        assert 100 <= len(text.split()) <= 300, (path, len(text.split()))
        resume_hashes.add(hashlib.sha256(text.encode()).hexdigest())
    assert len(resume_hashes) == 100, "Resume contents are not unique"
    assert len(names) == 100, "Candidate names are not unique"
    assert len(emails) == 100, "Candidate emails are not unique"

    for jd_path, path in zip(jd_files, keyword_files):
        jd_text = jd_path.read_text(encoding="utf-8")
        assert "synthetic evaluation job description" in jd_text
        assert "Occupation reference: https://www.onetonline.org/" in jd_text
        labels = json.loads(path.read_text(encoding="utf-8"))
        assert 10 <= len(labels) <= 15, (path, len(labels))
        assert all(set(item) == {"keyword", "synonyms"} for item in labels)
        assert len({item["keyword"].lower() for item in labels}) == len(labels)

    for row in pair_rows:
        assert (RESUME_DIR / f"{row['resume_id']}.md").exists()
        assert (JD_DIR / f"{row['jd_id']}.md").exists()
        assert (JD_DIR / f"{row['jd_id']}.keywords.json").exists()


def main():
    if len(ROLES) != 25:
        raise RuntimeError(f"Expected 25 roles, found {len(ROLES)}")
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    candidate_number = 1
    for jd_number, role in enumerate(ROLES, 1):
        jd_id = f"jd_{jd_number:03d}"
        (JD_DIR / f"{jd_id}.md").write_text(render_jd(jd_number, role), encoding="utf-8")
        labels = [{"keyword": keyword, "synonyms": synonyms} for keyword, synonyms in role["keywords"]]
        (JD_DIR / f"{jd_id}.keywords.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for variant in range(4):
            resume_id = f"cand_{candidate_number:03d}"
            (RESUME_DIR / f"{resume_id}.md").write_text(
                render_resume(candidate_number, jd_number - 1, variant), encoding="utf-8"
            )
            pair_rows.append({"resume_id": resume_id, "jd_id": jd_id, "jd_url": role["source"]})
            candidate_number += 1

    with (ROOT / "data" / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["resume_id", "jd_id", "jd_url"])
        writer.writeheader()
        writer.writerows(pair_rows)

    (ROOT / "data" / "SOURCES.md").write_text(render_sources(), encoding="utf-8")
    validate_dataset(pair_rows)
    print("Generated and validated 100 unique resumes, 25 JDs, 25 keyword files, and 100 pairs.")


if __name__ == "__main__":
    main()
