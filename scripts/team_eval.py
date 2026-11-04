"""Run one workflow's share of the team evaluation with your own Groq key.

    python scripts/team_eval.py <workflow>

That one command does everything that is due for your workflow, then packs the
results into results/handback_<workflow>.zip for you to send back:

  day 1   local metrics on all 100 resumes, requirement coverage (gpt-oss-20b),
          rubric judge on the 30 pairs in eval_plan/rubric30.txt (gpt-oss-120b)
  day 2   pairwise against w2_baseline on eval_plan/pairwise20.txt, only if your
          workflow is listed in eval_plan/pairwise_selected.txt (added after the
          day-1 screening; pull, then run the same command again)

Every step is resumable. If your daily Groq limit runs out, run the same command
again the next day and only the missing work is redone.

Options: --status (progress only), --test-pairs FILE (dry run on a few pairs).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ["w1_single_pass", "w2_baseline", "w3_critic", "w4_evidence",
             "w5_fine_tuned", "w6_best_of_n", "w7_plan_execute"]
BASELINE = "w2_baseline"
PLAN = ROOT / "eval_plan"
JUDGE = ["--judge-provider", "groq", "--judge-model", "openai/gpt-oss-120b"]
COVERAGE = ["--coverage-provider", "groq", "--coverage-model", "openai/gpt-oss-20b"]
QUIET = ["--skip-robustness", "--skip-injection", "--skip-human-review", "--skip-reports"]
TRANSIENT = ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
             "ConnectionError", "TimeoutError")


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def newest_hashes():
    hashes = {}
    results = ROOT / "results" / "results.jsonl"
    if results.exists():
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if row.get("workflow") in WORKFLOWS and row.get("config_hash"):
                hashes[row["workflow"]] = row["config_hash"]
    return hashes


def ollama_up():
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not url.startswith("http"):
        url = "http://" + url
    try:
        urllib.request.urlopen(url + "/api/version", timeout=3)
        return True
    except OSError:
        return False


def run(step, args):
    """Run evaluate.py, show its output, return the evaluation folder it wrote."""
    print(f"\n=== {step} ===", flush=True)
    proc = subprocess.Popen([sys.executable, "evaluate.py", *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not any(noise in line for noise in ("it/s]", "Batches", "Warning")):
            print(line, end="", flush=True)
    if proc.wait() != 0:
        sys.exit(f"\n{step} stopped with an error (shown above). Fix it, then run the same command again.")
    text = "".join(lines)
    start = text.rfind("{\n")
    try:
        return ROOT / json.loads(text[start:])["output_root"]
    except (ValueError, KeyError):
        return None


def state_file(wf):
    return PLAN / f".roots_{wf}.json"


def remember(wf, roots):
    known = set(json.loads(state_file(wf).read_text())) if state_file(wf).exists() else set()
    known |= {str(r.relative_to(ROOT)) for r in roots if r}
    state_file(wf).write_text(json.dumps(sorted(known)))
    return [ROOT / r for r in sorted(known)]


def status(wf, roots):
    done = {"coverage": [0, 0], "rubric": [0, 0], "pairwise": [0, 0]}
    waiting = 0
    for root in roots:
        for f in (root / "records" / wf).glob("*/repeat_0.json"):
            r = json.loads(f.read_text())
            fails = r.get("evaluation_failures") or {}
            waiting += any(isinstance(v, str) and v.startswith(TRANSIENT) for v in fails.values())
            if r.get("requirement_coverage_status"):
                done["coverage"][1] += 1
                done["coverage"][0] += r["requirement_coverage_status"] == "scored"
            if r.get("judge_status") in ("scored", "unscored"):
                done["rubric"][1] += 1
                done["rubric"][0] += r["judge_status"] == "scored"
        for f in root.glob(f"pairwise/*/records/{wf}/*/repeat_0.json"):
            r = json.loads(f.read_text())
            done["pairwise"][1] += 1
            done["pairwise"][0] += r.get("status") == "scored"
            waiting += r.get("failure_kind") == "judge_failed"
    print(f"\n=== progress for {wf} ===")
    for step, (ok, total) in done.items():
        print(f"  {step:9} {ok} scored of {total} attempted")
    if waiting:
        print(f"  {waiting} item(s) hit a rate limit or connection error: run the same command again later (tomorrow if the daily limit is used up).")
    return waiting


def pack(wf, roots):
    out = ROOT / "results" / f"handback_{wf}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root in roots:
            for f in root.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(ROOT))
    print(f"\nResults packed into {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB). Send this file back.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workflow", choices=WORKFLOWS)
    ap.add_argument("--status", action="store_true", help="show progress and stop")
    ap.add_argument("--test-pairs", help="limit every step to the pairs in this file (dry run)")
    a = ap.parse_args()
    wf = a.workflow
    os.chdir(ROOT)
    load_env()
    PLAN.mkdir(exist_ok=True)

    if a.status:
        roots = remember(wf, [])
        status(wf, roots)
        return

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Put GROQ_API_KEY=<your key> in a .env file in the repo folder.")
    hashes = newest_hashes()
    if wf not in hashes or BASELINE not in hashes:
        sys.exit("The shared resumes are missing from results/results.jsonl. Run `git pull` first.")
    if not ollama_up():
        print("Note: Ollama is not running, so the similarity metric will be retried on the next run. "
              "Start it with `ollama serve` (and `ollama pull nomic-embed-text` once) for a complete first run.")
    os.environ.setdefault("HOSTED_MIN_INTERVAL_SECONDS", "55")  # free tier: 8k tokens a minute
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    own = ["--config-hash", f"{wf}={hashes[wf]}"]
    rubric_pairs = a.test_pairs or str(PLAN / "rubric30.txt")
    pairwise_pairs = a.test_pairs or str(PLAN / "pairwise20.txt")
    limit_all = ["--pair-ids-file", a.test_pairs] if a.test_pairs else []
    roots = []

    roots.append(run("1/3 local metrics + requirement coverage (all resumes)",
                     ["score", "--workflows", wf, *own, "--skip-judge", "--skip-pairwise",
                      *limit_all, *JUDGE, *COVERAGE, *QUIET]))
    roots.append(run("2/3 rubric judge (30 pairs)",
                     ["score", "--workflows", wf, *own, "--skip-coverage", "--skip-pairwise",
                      "--pair-ids-file", rubric_pairs, *JUDGE, *COVERAGE, *QUIET]))

    selected_file = PLAN / "pairwise_selected.txt"
    selected = []
    if selected_file.exists():
        selected = [l.strip() for l in selected_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    if wf == BASELINE:
        print("\n=== 3/3 pairwise: not needed, w2_baseline is the baseline ===")
    elif not selected_file.exists():
        print("\n=== 3/3 pairwise: waiting for the day-1 screening. When eval_plan/pairwise_selected.txt "
              "appears after `git pull`, run this command again. ===")
    elif wf not in selected:
        print(f"\n=== 3/3 pairwise: {wf} was not selected for pairwise; nothing more to run ===")
    else:
        roots.append(run("3/3 pairwise vs w2_baseline (20 pairs)",
                         ["pairwise", "--workflows", BASELINE, wf,
                          "--config-hash", f"{BASELINE}={hashes[BASELINE]}", *own,
                          "--pairwise-challengers", wf, "--skip-judge", "--skip-coverage",
                          "--pair-ids-file", pairwise_pairs, *JUDGE, *COVERAGE, *QUIET]))

    roots = remember(wf, roots)
    status(wf, roots)
    pack(wf, roots)


if __name__ == "__main__":
    main()
