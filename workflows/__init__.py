"""Registry of workflows. Add yours here once it runs."""
from workflows import w1_single_pass, w2_baseline, w3_critic, w4_evidence

WORKFLOWS = {
    w1_single_pass.NAME: w1_single_pass.run,
    w2_baseline.NAME: w2_baseline.run,
    w3_critic.NAME: w3_critic.run,
    w4_evidence.NAME: w4_evidence.run,
}
