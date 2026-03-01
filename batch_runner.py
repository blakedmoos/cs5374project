"""
CS 5374 - Batch Test Runner
Runs multiple automated adversarial tests across model configurations.
Aggregates results into a summary CSV for analysis.

Usage:
    python batch_runner.py              # Run with defaults
    python batch_runner.py --trials 5   # 5 trials per config
    python batch_runner.py --rounds 5   # 5 rounds per trial
"""

import json
import csv
import os
import sys
import subprocess
import argparse
from datetime import datetime
from tracing import enable_tracing

# Enable LangSmith tracing for batch runs
enable_tracing("cs5374-batch-runner")

# ============================================================
# MODEL CONFIGURATIONS TO TEST
# Each tuple is (attacker_model, target_model, judge_model)
# ============================================================

CONFIGS = [
    ("llama3.2:3b", "llama3.2:3b", "llama3.2:3b"),
    ("llama3.2:3b", "nous-hermes2", "llama3.2:3b"),
    ("llama3.2:3b", "mistral", "llama3.2:3b"),
    ("nous-hermes2", "llama3.2:3b", "llama3.2:3b"),
    ("nous-hermes2", "nous-hermes2", "llama3.2:3b"),
    ("nous-hermes2", "mistral", "llama3.2:3b"),
    ("mistral", "llama3.2:3b", "llama3.2:3b"),
    ("mistral", "nous-hermes2", "llama3.2:3b"),
    ("mistral", "mistral", "llama3.2:3b"),
]

DEFAULT_TRIALS = 3
DEFAULT_ROUNDS = 10

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_FILE = f"batch_summary_{TIMESTAMP}.csv"
BATCH_DIR = f"batch_{TIMESTAMP}"


def check_models():
    """Verify all required models are available in Ollama."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        available = result.stdout.lower()
    except FileNotFoundError:
        print("ERROR: Ollama not found. Make sure it's installed and running.")
        sys.exit(1)

    needed = set()
    for atk, tgt, jdg in CONFIGS:
        needed.add(atk)
        needed.add(tgt)
        needed.add(jdg)

    missing = []
    for model in needed:
        # Check model name without tag
        model_base = model.split(":")[0]
        if model_base not in available:
            missing.append(model)

    if missing:
        print(f"Missing models: {', '.join(missing)}")
        print("Pull them with:")
        for m in missing:
            print(f"  ollama pull {m}")
        sys.exit(1)

    print(f"All {len(needed)} models available.")


def run_single_test(attacker, target, judge, max_rounds, trial_num, config_num):
    """Run a single automated test by importing and invoking the main graph."""
    # We import here to avoid circular issues and to patch config each time
    import bitcoin_wallet_test as bwt

    # Patch configuration
    bwt.ATTACKER_MODEL = attacker
    bwt.TARGET_MODEL = target
    bwt.JUDGE_MODEL = judge
    bwt.MAX_ROUNDS = max_rounds
    bwt.MODE = "auto"

    # Set unique filenames
    run_id = f"c{config_num}_t{trial_num}"
    bwt.TIMESTAMP = f"{TIMESTAMP}_{run_id}"
    bwt.LOG_FILE = os.path.join(BATCH_DIR, f"log_{run_id}.txt")
    bwt.RESULTS_FILE = os.path.join(BATCH_DIR, f"results_{run_id}.json")

    print(f"\n{'='*60}")
    print(f"CONFIG {config_num} | TRIAL {trial_num}")
    print(f"Attacker: {attacker} | Target: {target} | Judge: {judge}")
    print(f"{'='*60}")

    try:
        # Tag this run in LangSmith so traces are filterable per config
        os.environ["LANGCHAIN_PROJECT"] = f"cs5374-batch-{attacker}-vs-{target}"

        graph = bwt.build_graph()
        initial_state = {
            "conversation_history": [],
            "attacker_message": "",
            "target_response": "",
            "judge_evaluations": [],
            "round_number": 0,
            "secret_extracted": False,
            "result_summary": "",
            "human_takeover": False,
        }
        result = graph.invoke(initial_state, {"recursion_limit": 100})

        # Load the results file
        with open(bwt.RESULTS_FILE, "r") as f:
            return json.load(f)

    except Exception as e:
        print(f"ERROR in config {config_num} trial {trial_num}: {e}")
        return None


def aggregate_results(all_results):
    """Build summary statistics and write CSV."""
    rows = []

    for r in all_results:
        if r is None:
            continue
        config = r.get("config", {})
        rows.append({
            "attacker": config.get("attacker_model", ""),
            "target": config.get("target_model", ""),
            "judge": config.get("judge_model", ""),
            "rounds": r.get("rounds_completed", 0),
            "max_rounds": config.get("max_rounds", 0),
            "extracted": r.get("secret_extracted", False),
            "avg_danger": r.get("avg_danger_score", 0),
            "max_danger": r.get("max_danger_score", 0),
            "partial_leaks": sum(1 for e in r.get("evaluations", []) if e.get("leak_status") == "PARTIAL_LEAK"),
            "full_leaks": sum(1 for e in r.get("evaluations", []) if e.get("leak_status") == "FULL_LEAK"),
        })

    # Write CSV
    if rows:
        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV: {SUMMARY_FILE}")

    return rows


def print_summary(rows):
    """Print aggregated summary by config."""
    from collections import defaultdict

    configs = defaultdict(list)
    for r in rows:
        key = f"{r['attacker']} → {r['target']}"
        configs[key].append(r)

    print(f"\n{'='*70}")
    print("BATCH RESULTS SUMMARY")
    print(f"{'='*70}")

    for config_name, trials in configs.items():
        n = len(trials)
        extractions = sum(1 for t in trials if t["extracted"])
        avg_danger = sum(t["avg_danger"] for t in trials) / n if n else 0
        max_danger = max(t["max_danger"] for t in trials) if trials else 0
        total_partial = sum(t["partial_leaks"] for t in trials)
        total_full = sum(t["full_leaks"] for t in trials)

        print(f"\n{config_name}")
        print(f"  Trials: {n}")
        print(f"  Extractions: {extractions}/{n} ({extractions/n*100:.0f}%)")
        print(f"  Avg Danger: {avg_danger:.1f}/10")
        print(f"  Max Danger: {max_danger}/10")
        print(f"  Partial Leaks: {total_partial}")
        print(f"  Full Leaks: {total_full}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Batch adversarial test runner")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials per config")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Max rounds per trial")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("CS 5374 - BATCH TEST RUNNER")
    print(f"{'='*60}")
    print(f"Configurations: {len(CONFIGS)}")
    print(f"Trials per config: {args.trials}")
    print(f"Rounds per trial: {args.rounds}")
    print(f"Total runs: {len(CONFIGS) * args.trials}")
    print(f"Output dir: {BATCH_DIR}")
    print(f"Summary: {SUMMARY_FILE}")
    print()

    # Check models
    check_models()

    # Create output dir
    os.makedirs(BATCH_DIR, exist_ok=True)

    # Run all tests
    all_results = []
    total = len(CONFIGS) * args.trials
    current = 0

    for config_num, (atk, tgt, jdg) in enumerate(CONFIGS, 1):
        for trial in range(1, args.trials + 1):
            current += 1
            print(f"\n[{current}/{total}] Running...")
            result = run_single_test(atk, tgt, jdg, args.rounds, trial, config_num)
            all_results.append(result)

    # Aggregate and print
    rows = aggregate_results(all_results)
    if rows:
        print_summary(rows)

    print(f"\nBatch complete. {len([r for r in all_results if r])} successful runs.")
    print(f"Logs and results in: {BATCH_DIR}/")
    print(f"Summary CSV: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
