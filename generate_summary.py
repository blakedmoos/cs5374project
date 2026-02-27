"""Generate batch summary CSV from existing result JSON files."""
import json
import csv
import os
from collections import defaultdict

BATCH_DIR = "batch_20260225_180032"
SUMMARY_FILE = "batch_summary_20260225_180032.csv"

rows = []
for filename in sorted(os.listdir(BATCH_DIR)):
    if not filename.startswith("results_") or not filename.endswith(".json"):
        continue
    filepath = os.path.join(BATCH_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            r = json.load(f)
        config = r.get("config", {})
        rows.append({
            "run_id": filename.replace("results_", "").replace(".json", ""),
            "attacker": config.get("attacker_model", ""),
            "target": config.get("target_model", ""),
            "judge": config.get("judge_model", ""),
            "rounds": r.get("rounds_completed", 0),
            "max_rounds": config.get("max_rounds", 0),
            "extracted": r.get("secret_extracted", False),
            "avg_danger": round(r.get("avg_danger_score", 0), 2),
            "max_danger": r.get("max_danger_score", 0),
            "partial_leaks": sum(1 for e in r.get("evaluations", []) if e.get("leak_status") == "PARTIAL_LEAK"),
            "full_leaks": sum(1 for e in r.get("evaluations", []) if e.get("leak_status") == "FULL_LEAK"),
        })
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if rows:
    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {SUMMARY_FILE}")

configs = defaultdict(list)
for r in rows:
    key = f"{r['attacker']} -> {r['target']}"
    configs[key].append(r)

print(f"\n{'='*70}")
print("BATCH RESULTS SUMMARY")
print(f"{'='*70}")

for config_name, trials in sorted(configs.items()):
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
