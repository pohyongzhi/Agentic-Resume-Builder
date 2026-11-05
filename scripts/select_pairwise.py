"""Pick the workflows that go on to pairwise after the day-1 screening.

    python scripts/select_pairwise.py results/evaluations [--top 3] [--min-faithfulness 0.9]

Rule, fixed before any results were seen: among challengers (every workflow
except w2_baseline) whose mean claim faithfulness on all their resumes is at
least --min-faithfulness, take the --top highest mean rubric scores on the
rubric pairs; ties go to the more faithful workflow. Writes
eval_plan/pairwise_selected.txt with the scores used, so the choice is auditable.
"""
import argparse
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = "w2_baseline"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evaluations", type=pathlib.Path, help="folder holding everyone's merged results/evaluations")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--min-faithfulness", type=float, default=0.9)
    a = ap.parse_args()

    faith, rubric = {}, {}
    for f in a.evaluations.glob("*/*/*/records/*/*/repeat_0.json"):
        r = json.loads(f.read_text())
        wf = r.get("workflow")
        if r.get("judge_status") == "scored" and r.get("judge_overall_mean") is not None:
            rubric.setdefault(wf, {})[r["pair_id"]] = r["judge_overall_mean"]
        if r.get("claim_faithfulness") is not None and r.get("requirement_coverage_status"):
            faith.setdefault(wf, {})[r["pair_id"]] = r["claim_faithfulness"]

    rows = []
    for wf in sorted(set(faith) | set(rubric)):
        if wf == BASELINE:
            continue
        f = statistics.fmean(faith[wf].values()) if faith.get(wf) else None
        s = statistics.fmean(rubric[wf].values()) if rubric.get(wf) else None
        rows.append((wf, f, s, len(faith.get(wf, {})), len(rubric.get(wf, {}))))

    eligible = [r for r in rows if r[1] is not None and r[2] is not None and r[1] >= a.min_faithfulness]
    chosen = sorted(eligible, key=lambda r: (r[2], r[1]), reverse=True)[: a.top]

    lines = [f"# Selected for pairwise: top {a.top} challengers by mean rubric score among those with",
             f"# mean claim faithfulness >= {a.min_faithfulness}. Scores used (faithfulness / rubric, n):"]
    for wf, f, s, nf, ns in rows:
        mark = "selected" if any(c[0] == wf for c in chosen) else "not selected"
        fs = "–" if f is None else f"{f:.3f}"
        ss = "–" if s is None else f"{s:.2f}"
        lines.append(f"#   {wf}: {fs} (n={nf}) / {ss} (n={ns}) -> {mark}")
    lines += [c[0] for c in chosen]
    out = ROOT / "eval_plan" / "pairwise_selected.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out.relative_to(ROOT)}. Commit and push it so the selected owners can run pairwise.")


if __name__ == "__main__":
    main()
