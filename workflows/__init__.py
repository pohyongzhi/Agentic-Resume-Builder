"""Registry of workflows. Add yours here once it runs."""
from workflows import baseline, single_pass, w4_ats_optimiser

WORKFLOWS = {
    single_pass.NAME: single_pass.run,
    baseline.NAME: baseline.run,
    w4_ats_optimiser.NAME: w4_ats_optimiser.run,
    # "w1_research": w1_research.run,
    # "w2_recruiter": w2_recruiter.run,
    # "w3_critic": w3_critic.run,
    # "w5_plan_first": w5_plan_first.run,
}
