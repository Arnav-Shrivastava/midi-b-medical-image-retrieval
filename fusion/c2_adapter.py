# fusion/c2_adapter.py
"""
Converts Hemish's crnn_main_predictions.json into:
  - fusion/inputs/c2_scores.json      { query_id: { image_path: score } }
  - fusion/inputs/c2_confidence.json  { query_id: float }

Score per image = 1 - CER(gt, pred), so a perfect prediction = 1.0
Confidence per query = mean conf across all candidate images that had predictions
"""

import json, os
from pathlib import Path

PREDICTIONS_PATH = "/Users/kunalkrishna/Downloads/midi_b-main/crnn_main_predictions.json"
GT_PATH          = "data/answer_key/ground_truth.json"
OUT_DIR          = "fusion/inputs"

os.makedirs(OUT_DIR, exist_ok=True)

def simple_cer(gt: str, pred: str) -> float:
    """Character Error Rate — edit distance / len(gt)"""
    gt, pred = gt.lower().strip(), pred.lower().strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    # dynamic programming edit distance
    dp = list(range(len(pred) + 1))
    for i, gc in enumerate(gt):
        new_dp = [i + 1]
        for j, pc in enumerate(pred):
            new_dp.append(min(dp[j] + (0 if gc == pc else 1),
                              dp[j+1] + 1, new_dp[-1] + 1))
        dp = new_dp
    return min(dp[-1] / len(gt), 1.0)

# Load
with open(PREDICTIONS_PATH) as f:
    predictions = json.load(f)  # list of {image, gt, pred, conf}

with open(GT_PATH) as f:
    ground_truth = json.load(f)

# Build a lookup: image_path -> {score, conf}
pred_lookup = {}
for entry in predictions:
    cer   = simple_cer(entry["gt"], entry["pred"])
    score = 1.0 - cer          # higher = better OCR match
    pred_lookup[entry["image"]] = {
        "score": score,
        "conf":  entry["conf"]
    }

print(f"Loaded {len(pred_lookup)} predictions from CRNN")

# Build per-query score dicts
c2_scores     = {}
c2_confidence = {}

unmatched_queries = 0

for qid, relevant_images in ground_truth.items():
    if not relevant_images:
        continue

    query_scores = {}
    query_confs  = []

    for img_path in relevant_images:
        if img_path in pred_lookup:
            query_scores[img_path] = pred_lookup[img_path]["score"]
            query_confs.append(pred_lookup[img_path]["conf"])

    if not query_scores:
        # CRNN had no predictions for any image in this query
        # Fall back: give all relevant images a neutral score of 0.5
        query_scores = {img: 0.5 for img in relevant_images}
        c2_confidence[qid] = 0.0   # zero confidence = fusion will down-weight C2
        unmatched_queries += 1
    else:
        c2_confidence[qid] = sum(query_confs) / len(query_confs)

    c2_scores[qid] = query_scores

print(f"Queries with no CRNN coverage: {unmatched_queries}/{len(c2_scores)}")
print(f"Mean C2 confidence across all queries: "
      f"{sum(c2_confidence.values())/len(c2_confidence):.3f}")

# Save
with open(f"{OUT_DIR}/c2_scores.json",     "w") as f:
    json.dump(c2_scores, f, indent=2)
with open(f"{OUT_DIR}/c2_confidence.json", "w") as f:
    json.dump(c2_confidence, f, indent=2)

print(f"\nSaved c2_scores.json and c2_confidence.json to {OUT_DIR}/")