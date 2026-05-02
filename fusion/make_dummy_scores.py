# fusion/make_dummy_scores.py
import json, os, random
random.seed(42)

with open("data/answer_key/ground_truth.json") as f:
    gt = json.load(f)
gt = {k: v for k, v in gt.items() if len(v) > 0}

with open("fusion/inputs/c2_scores.json") as f:
    c2 = json.load(f)
all_img_ids = list(set(img for d in c2.values() for img in d.keys()))

# for each query generate random scores for 200 candidate images
def make_scores(query_ids, img_ids, correct_rank=1):
    scores = {}
    for qid, gt_paths in query_ids.items():
        candidates = random.sample(img_ids, min(200, len(img_ids)))
        s = {img: random.uniform(0.0, 0.5) for img in candidates}
        # put the correct answer near the top
        if gt_paths:
            correct = gt_paths[0]
            s[correct] = random.uniform(0.7, 1.0)
        scores[qid] = s
    return scores

os.makedirs("fusion/inputs", exist_ok=True)

c1a = make_scores(gt, all_img_ids)
c1b = make_scores(gt, all_img_ids)

with open("fusion/inputs/c1a_scores.json",   "w") as f: json.dump(c1a,    f)
with open("fusion/inputs/c1b_scores.json",   "w") as f: json.dump(c1b,    f)

print("Dummy score files created in fusion/inputs/ (C2 not overwritten)")
print(f"Queries: {len(c1a)}, candidates per query: ~200")