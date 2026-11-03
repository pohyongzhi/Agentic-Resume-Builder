# Synthetic evaluation-data specification

Generate one fictional job description and five fictional candidate resumes from one role record in
`spec.json`. The five candidates must be two strong fits, two moderate fits and one weak fit.

Rules:

- Use only fictional people, employers, schools, email addresses and telephone numbers.
- Never claim that a candidate was interviewed, selected, hired or rejected.
- A strong candidate should support most required skills, a moderate candidate roughly half, and a
  weak candidate only a small transferable subset.
- Missing qualifications must be omitted, not contradicted or hidden.
- Every metric, skill and achievement later used in tailoring must be stated in the original resume.
- Job descriptions must separate required and preferred qualifications.
- Resumes must contain contact information, summary, experience, education, projects, skills and
  certifications where applicable.
- Return structured JSON. Markdown files are produced only by the materialisation script.

The checked-in dataset is generated deterministically by `scripts/generate_dataset.py`. This prompt
is retained as the versioned contract if a stronger teacher model is used to regenerate a future
dataset version.
