# W7 – Plan-and-execute (owner: Jaxsen)

> A planner first produces a tailoring plan: which JD requirement maps to which evidence in the
> resume, the section order, and what to cut. Separate writer agents then draft each section in
> parallel, and an assembler merges the sections and checks they are consistent.

| File | What it is |
|---|---|
| `workflow.png` | Diagram of the graph and where each model call happens |
| `../../workflows/w7_plan_execute.py` | The workflow |
| `../../tests/test_w7_plan_execute.py` | Tests with a fake Ollama client (no model needed) |

![W7 plan-and-execute](workflow.png)

## The graph

```text
START -> plan --Send--> write (one per planned section, in parallel) -> assemble -> END
```

| Node | Model calls | What it does |
|---|---|---|
| `plan` | 1 (2 if the JSON is broken) | Returns the header, the sections in order with `lead_with` and `cut`, and 6-12 JD requirements, each mapped to a line quoted from the resume or `null` |
| `write` | 1 per section | Writes only its own section from the original resume and its part of the plan |
| `assemble` | 1 | Merges the sections in order, removes repeats and contradictions, follows the plan's cuts and UNSUPPORTED requirements |

Model calls per resume = 1 + N + 1, where N is the number of planned sections.

## Design choices

- **Follows the variant description only.** Every decision is the model's. There is no code check
  that evidence quotes really appear in the resume, and no critique/revise loop: those are W4's and
  W3's techniques, and the shared faithfulness metric measures the result.
- **No layout rules.** Like the baseline, only `BASE_RULES` shapes the output.
- **Code is plumbing only:** parse the plan JSON (one retry), give each writer its part, cut each
  writer's reply down to its own section, and turn PDF `•` bullets into Markdown bullets.
- **Fallbacks.** If the plan JSON fails twice, writers use the four `BASE_RULES` sections and the
  run is marked not finished. If the assembler returns nothing usable, the header and sections are
  joined as written.

## Run it

```bash
python run_all.py --workflows w7_plan_execute --split test --limit 1 --repeats 1
python app.py      # pick "W7 - Plan-and-execute" in the dropdown; the plan is shown under the stats
pytest -q tests/test_w7_plan_execute.py
```

## Things to know when comparing results

- **Latency:** all writers share one Ollama server, so "parallel" may just queue. Report
  `wall_seconds` from `run_all.py`; the Tracker's `seconds` adds up call times.
- **Diagnostics:** the plan is logged in the trace as a `w7_plan` event (sections, requirements,
  how many are backed by evidence, which attempt succeeded).
- **Not on the shared pipeline:** like W1 and W6, W7 reads the raw resume and JD and skips the
  shared extraction and validate/render harness. Compare it with that in mind.
- **Sections:** the planner chooses the sections freely, so a plan can add e.g. Projects beyond the
  four `BASE_RULES` asks for.
