"""Registry of executable workflow variants and their writer-model routes."""
import config
from workflows import (
    w1_single_pass,
    w2_baseline,
    w3_critic,
    w4_evidence,
    w5_fine_tuned,
    w6_best_of_n,
)

WORKFLOWS = {
    w1_single_pass.NAME: w1_single_pass.run,
    w2_baseline.NAME: w2_baseline.run,
    w3_critic.NAME: w3_critic.run,
    w4_evidence.NAME: w4_evidence.run,
    w5_fine_tuned.NAME: w5_fine_tuned.run,
    w6_best_of_n.NAME: w6_best_of_n.run,
}

WORKFLOW_MODELS = {
    name: config.FINE_TUNED_MODEL if name == w5_fine_tuned.NAME else config.BASE_MODEL
    for name in WORKFLOWS
}
