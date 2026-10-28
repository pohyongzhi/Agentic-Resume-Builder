# Workflow contract

All workflows accept the same resume and job-description pair and return a Markdown resume. The shared model wrapper records the actual selected model for every call.

| No. | Stable ID | Model route | Status | V1 behavior |
| ---: | --- | --- | --- | --- |
| 1 | `baseline` | `qwen3.5:4b-q4_K_M` | Implemented | One deterministic shared-prompt call; application default. |
| 2 | `multi_critic_review` | Base | Planned | Reserved for a multi-critic review loop; unavailable. |
| 3 | `evidence_grounded_generation` | Base | Planned | Reserved for explicit evidence-grounded generation; unavailable. |
| 4 | `fine_tuned_small_generator` | `agentic-resume-v4-1:q4_K_M` | Implemented | Same input/output contract and prompt path as the baseline, with an explicit model override. |
| 5 | `multi_draft_selector` | Base | Planned | Reserved for several drafts plus a selector; unavailable. |
| 6 | `planner_section_writers` | Base | Planned | Reserved for a planner and section writers; unavailable. |

The historical `w4_ats_optimiser` code is not registered as one of these six workflows and must not be interpreted as the fine-tuned Workflow 4.
