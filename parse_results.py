import json, glob, os, csv

results = []
for f in sorted(glob.glob('blind_results_*.json')):
    with open(f) as fh:
        d = json.load(fh)
    m = d.get('metadata', {})
    s = d.get('summary', {})
    results.append({
        'timestamp': m.get('timestamp', '?'),
        'target': m.get('target_model', '?').replace('google_genai/', '').replace('anthropic/', ''),
        'attacker': m.get('attacker_model', '?').replace('anthropic/', '').replace('claude-', ''),
        'prompt': m.get('prompt_level', '?'),
        'resumed': 'yes' if m.get('resumed_from') else 'no',
        'sessions': s.get('sessions_used', '?'),
        'rounds': s.get('total_rounds', '?'),
        'leaked': s.get('leak_count', len(s.get('unique_words_leaked', []))),
        'full_leak': 'YES' if s.get('full_leak_achieved', False) else 'no',
        'best_danger': s.get('best_danger_score', '?'),
        'avg_danger': round(s.get('avg_danger_score', 0), 1),
    })

# Print table
header = f"{'Target':<42} {'Attacker':<22} {'R':<3} {'Ses':<4} {'Rnd':<4} {'Lk':<4} {'Full':<5} {'Dgr':<4}"
print(header)
print('-' * len(header))
for r in results:
    print(f"{r['target']:<42} {r['attacker']:<22} {r['resumed']:<3} {str(r['sessions']):<4} {str(r['rounds']):<4} {str(r['leaked']):<4} {r['full_leak']:<5} {str(r['best_danger']):<4}")

# Write CSV
with open('all_results_tracker.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)

print(f"\nSaved {len(results)} results to all_results_tracker.csv")

# Summary by target
print("\n\n=== SUMMARY BY TARGET ===")
from collections import defaultdict
by_target = defaultdict(list)
for r in results:
    by_target[r['target']].append(r)

for target in sorted(by_target.keys()):
    runs = by_target[target]
    total = len(runs)
    breached = sum(1 for r in runs if r['full_leak'] == 'YES')
    fresh = [r for r in runs if r['resumed'] == 'no']
    fresh_breached = sum(1 for r in fresh if r['full_leak'] == 'YES')
    rounds_to_breach = [r['rounds'] for r in runs if r['full_leak'] == 'YES']
    print(f"\n{target}")
    print(f"  Total runs: {total} ({len(fresh)} fresh, {total - len(fresh)} resumed)")
    print(f"  Breached: {breached}/{total} ({breached/total*100:.0f}%)")
    if fresh:
        print(f"  Fresh breached: {fresh_breached}/{len(fresh)} ({fresh_breached/len(fresh)*100:.0f}%)")
    if rounds_to_breach:
        print(f"  Rounds to breach: {rounds_to_breach} (min={min(rounds_to_breach)}, max={max(rounds_to_breach)})")
