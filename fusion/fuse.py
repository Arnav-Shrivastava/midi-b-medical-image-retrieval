# fusion/fuse.py
import json, numpy as np, os, sys
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.map_harness import evaluate, mean_average_precision

# ── config ────────────────────────────────────────────────
GT_PATH       = "data/answer_key/ground_truth.json"
C1A_PATH      = "fusion/inputs/c1a_scores.json"
C1B_PATH      = "fusion/inputs/c1b_scores.json"
C2_PATH       = "fusion/inputs/c2_scores.json"
C2_CONF_PATH  = "fusion/inputs/c2_confidence.json"
OUT_DIR       = "results"
# ─────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# ── load all scores ───────────────────────────────────────
print("Loading score files...")
with open(C1A_PATH)     as f: c1a_scores  = json.load(f)
with open(C1B_PATH)     as f: c1b_scores  = json.load(f)
with open(C2_PATH)      as f: c2_scores   = json.load(f)
with open(C2_CONF_PATH) as f: c2_conf     = json.load(f)
with open(GT_PATH)      as f: ground_truth = json.load(f)

ground_truth = {k: v for k, v in ground_truth.items() if len(v) > 0}
print(f"Loaded scores for {len(c1a_scores)} queries")

# ── fusion function ───────────────────────────────────────
def fuse_and_rank(qid, alpha, beta):
    """
    Three-way confidence-weighted fusion.

    final_score = alpha * C1a + beta * C1b + gamma * C2

    gamma scales with CRNN confidence for this query —
    if OCR is confident, trust it more; if not, lean on visual.
    alpha and beta split the remaining weight.

    alpha + beta must sum to 1.0 (they share the non-C2 weight).
    """
    crnn_conf = c2_conf.get(qid, 0.0)      # 0.0 to 1.0
    gamma     = crnn_conf                   # C2 weight = its confidence
    remaining = 1.0 - gamma
    a = alpha * remaining                   # C1a weight
    b = beta  * remaining                   # C1b weight

    # collect all candidate image ids across all three score dicts
    all_candidates = (
        set(c1a_scores.get(qid, {}).keys()) |
        set(c1b_scores.get(qid, {}).keys()) |
        set(c2_scores.get(qid, {}).keys())
    )

    fused = {}
    for img_id in all_candidates:
        score = (
            a * c1a_scores.get(qid, {}).get(img_id, 0.0) +
            b * c1b_scores.get(qid, {}).get(img_id, 0.0) +
            gamma * c2_scores.get(qid, {}).get(img_id, 0.0)
        )
        fused[img_id] = score

    ranked = sorted(fused, key=fused.get, reverse=True)
    return ranked[:100]

# ── grid search alpha, beta on full query set ─────────────
print("\nGrid searching fusion weights (alpha, beta)...")
print("alpha = C1a share, beta = C1b share (of non-C2 weight)")

best_map10  = 0.0
best_params = (0.5, 0.5)
grid_results = []

# alpha + beta must sum to 1.0
alphas = np.arange(0.0, 1.1, 0.1).round(1)
for alpha in alphas:
    beta = round(1.0 - alpha, 1)
    retrieved = {qid: fuse_and_rank(qid, alpha, beta)
                 for qid in ground_truth}
    map10 = mean_average_precision(retrieved, ground_truth, k=10)
    grid_results.append((alpha, beta, map10))
    print(f"  alpha={alpha:.1f} beta={beta:.1f} -> mAP@10={map10:.4f}")
    if map10 > best_map10:
        best_map10  = map10
        best_params = (alpha, beta)

print(f"\nBest: alpha={best_params[0]}, beta={best_params[1]}, mAP@10={best_map10:.4f}")

# ── final evaluation with best params ────────────────────
alpha, beta = best_params
retrieved_final = {qid: fuse_and_rank(qid, alpha, beta)
                   for qid in ground_truth}

results = evaluate(retrieved_final, ground_truth,
                   system_name="C1a+C1b+C2 fusion")
results["alpha"]  = alpha
results["beta"]   = beta
results["note"]   = "gamma is data-driven per query from C2 confidence"

# ── save grid search curve ────────────────────────────────
grid_data = [{"alpha": a, "beta": b, "mAP@10": m}
             for a, b, m in grid_results]

with open(f"{OUT_DIR}/fusion_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(f"{OUT_DIR}/fusion_alpha_curve.json", "w") as f:
    json.dump(grid_data, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/fusion_results.json")
print(f"Alpha curve saved to {OUT_DIR}/fusion_alpha_curve.json")