import json
import os
from map_harness import evaluate

# load gt
GT_PATH = "../data/answer_key/ground_truth.json"
with open(GT_PATH) as f:
    ground_truth = json.load(f)

# evaluate standalone
def eval_standalone(path, name):
    print(f"Loading {name}...")
    with open(path) as f:
        scores = json.load(f)
    
    # rank them
    retrieved = {}
    for qid, img_scores in scores.items():
        ranked = sorted(img_scores, key=img_scores.get, reverse=True)
        retrieved[qid] = ranked[:100]
    
    evaluate(retrieved, ground_truth, system_name=name)

if __name__ == "__main__":
    eval_standalone("../fusion/inputs/c1a_scores.json", "C1a")
    eval_standalone("../fusion/inputs/c1b_scores.json", "C1b")
    eval_standalone("../fusion/inputs/c2_scores.json", "C2")
