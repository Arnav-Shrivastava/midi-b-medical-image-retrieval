# eval/test_harness.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.map_harness import average_precision_at_k, mean_average_precision, normalize

def test_perfect():
    ap = average_precision_at_k(["1-1.jpg", "1-2.jpg"], {"1-1.jpg"}, k=10)
    assert abs(ap - 1.0) < 1e-6
    print("PASS: perfect retrieval")

def test_rank2_hit():
    ap = average_precision_at_k(["bad.jpg", "1-1.jpg"], {"1-1.jpg"}, k=10)
    assert abs(ap - 0.5) < 1e-6
    print("PASS: rank-2 hit -> AP=0.5")

def test_two_relevant():
    ap = average_precision_at_k(["1-1.jpg", "bad.jpg", "1-2.jpg"], {"1-1.jpg", "1-2.jpg"}, k=10)
    assert abs(ap - (1.0 + 2/3) / 2) < 1e-6
    print("PASS: two relevant items")

def test_no_relevant():
    ap = average_precision_at_k(["1-1.jpg"], set(), k=10)
    assert ap == 0.0
    print("PASS: no relevant items -> AP=0.0")

def test_cutoff():
    retrieved = [f"{i}-1.jpg" for i in range(20)]
    ap = average_precision_at_k(retrieved, {"14-1.jpg"}, k=10)
    assert ap == 0.0
    print("PASS: item beyond K cutoff ignored")

def test_multi_query():
    r  = {"q1": ["1-1.jpg", "bad.jpg", "1-2.jpg"], "q2": ["bad.jpg", "1-3.jpg"]}
    gt = {"q1": ["1-1.jpg", "1-2.jpg"], "q2": ["1-3.jpg"]}
    expected = ((1.0 + 2/3) / 2 + 0.5) / 2
    result = mean_average_precision(r, gt, k=10)
    assert abs(result - expected) < 1e-6
    print("PASS: multi-query mAP")

def test_normalize():
    full = r"C:\Users\arnav\Desktop\medical image retrieval system\data\processed\jpg\1011144680\abc\1-1.jpg"
    assert normalize(full) == "1-1.jpg"
    print("PASS: path normalization works")

def test_real_gt_format():
    """Test with actual ground_truth.json structure."""
    import json
    with open("data/answer_key/ground_truth.json") as f:
        gt = json.load(f)

    qid   = list(gt.keys())[0]
    paths = gt[qid]
    relevant = set(normalize(p) for p in paths)

    # Simulate a retrieval that returns the correct image at rank 1
    correct_fname = normalize(paths[0])
    fake_retrieved = {qid: [correct_fname, "wrong1.jpg", "wrong2.jpg"]}

    from eval.map_harness import mean_average_precision
    score = mean_average_precision(fake_retrieved, gt, k=10)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print(f"PASS: real GT format works (query={qid}, relevant={relevant})")

if __name__ == "__main__":
    print("Running harness validation...\n")
    test_perfect()
    test_rank2_hit()
    test_two_relevant()
    test_no_relevant()
    test_cutoff()
    test_multi_query()
    test_normalize()
    test_real_gt_format()
    print("\nAll 8 tests passed. Harness validated for CP1.")