# eval/map_harness.py
import os

def normalize(path):
    """Extract a unique relative path to prevent filename collisions."""
    path = path.replace("\\", "/")
    if "data/processed/jpg/" in path:
        return path.split("data/processed/jpg/")[-1]
    return path

def average_precision_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    hits, sum_p = 0, 0.0
    for rank, img_id in enumerate(retrieved_ids[:k], start=1):
        if normalize(img_id) in relevant_ids:
            hits += 1
            sum_p += hits / rank
    return sum_p / min(len(relevant_ids), k)

def mean_average_precision(all_retrieved, ground_truth, k):
    scores = []
    for qid, retrieved in all_retrieved.items():
        relevant = set(normalize(p) for p in ground_truth.get(qid, []))
        scores.append(average_precision_at_k(retrieved, relevant, k))
    return sum(scores) / len(scores) if scores else 0.0

def evaluate(all_retrieved, ground_truth, system_name="system"):
    map10  = mean_average_precision(all_retrieved, ground_truth, k=10)
    map100 = mean_average_precision(all_retrieved, ground_truth, k=100)
    print(f"\n{'='*40}")
    print(f"System  : {system_name}")
    print(f"mAP@10  : {map10:.4f}")
    print(f"mAP@100 : {map100:.4f}")
    print(f"Queries : {len(all_retrieved)}")
    print(f"{'='*40}\n")
    return {"system": system_name, "mAP@10": map10, "mAP@100": map100}