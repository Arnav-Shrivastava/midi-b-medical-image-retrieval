# results/ablation_table.py
# Generates the full §5 ablation table from all result JSON files.
# Run anytime — shows — for results not yet available.
# When all scores are in, run once and the paper table is ready.

import json, os

RESULTS_DIR  = "baselines/results"
FUSION_DIR   = "results"

# ── define all rows in display order ─────────────────────
ROWS = [
    # SOTA baselines
    {
        "label":    "CLIP ViT-B/32",
        "tag":      "SOTA",
        "file":     f"{RESULTS_DIR}/clip_results.json",
        "hardware": "GPU only",
    },
    {
        "label":    "TrOCR + BM25",
        "tag":      "SOTA",
        "file":     f"{RESULTS_DIR}/trocr_bm25_results.json",
        "hardware": "GPU only",
    },
    {
        "label":    "CRAFT + TrOCR + BM25",
        "tag":      "SOTA",
        "file":     f"{RESULTS_DIR}/craft_trocr_results.json",
        "hardware": "CPU",
    },
    # Our contributions
    {
        "label":    "C1a: HOG+LBP pyramid",
        "tag":      "ours",
        "file":     f"{RESULTS_DIR}/c1a_results.json",
        "hardware": "CPU only",
    },
    {
        "label":    "C1b: MobileNetV3",
        "tag":      "ours",
        "file":     f"{RESULTS_DIR}/c1b_results.json",
        "hardware": "GPU train / CPU infer",
    },
    {
        "label":    "C1a + C1b",
        "tag":      "ours",
        "file":     f"{RESULTS_DIR}/c1a_c1b_results.json",
        "hardware": "CPU infer",
    },
    {
        "label":    "C2: Domain CRNN",
        "tag":      "ours",
        "file":     f"{RESULTS_DIR}/c2_results.json",
        "hardware": "GPU train / CPU infer",
    },
    {
        "label":    "C1a + C1b + C2 (full fusion)",
        "tag":      "ours",
        "file":     f"{FUSION_DIR}/fusion_results.json",
        "hardware": "CPU infer",
    },
]

# ── load a result file safely ─────────────────────────────
def load(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except:
        return None

# ── print the table ───────────────────────────────────────
def fmt(val, decimals=4):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

def print_table():
    col1 = 30   # method name width
    col2 = 10   # mAP@10
    col3 = 10   # mAP@100
    col4 = 12   # latency
    col5 = 22   # hardware

    header = (f"{'Method':<{col1}} {'mAP@10':>{col2}} {'mAP@100':>{col3}} "
              f"{'Latency':>{col4}}  {'Hardware':<{col5}}")
    divider = "─" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    sota_done = False
    for row in ROWS:
        # print separator between SOTA and ours
        if row["tag"] == "ours" and not sota_done:
            print(divider)
            sota_done = True

        data = load(row["file"])

        map10   = data.get("mAP@10")  if data else None
        map100  = data.get("mAP@100") if data else None
        latency = data.get("latency_ms_per_query") if data else None

        lat_str = f"{latency:.2f} ms" if latency else "—"
        label   = row["label"]
        if row["tag"] == "ours":
            label = f"{label} *"

        print(f"{label:<{col1}} {fmt(map10):>{col2}} {fmt(map100):>{col3}} "
              f"{lat_str:>{col4}}  {row['hardware']:<{col5}}")

    print(divider)
    print("* our contributions\n")

# ── show completion status ────────────────────────────────
def print_status():
    print("Result file status:")
    all_done = True
    for row in ROWS:
        exists = os.path.exists(row["file"])
        status = "✓" if exists else "⏳ missing"
        if not exists:
            all_done = False
        print(f"  {status}  {row['file']}")

    print()
    if all_done:
        print("All results available. Table is complete for paper submission.")
    else:
        print("Some results still pending. Re-run this script when they arrive.")

# ── delta vs best SOTA ────────────────────────────────────
def print_delta():
    # find best SOTA mAP@10
    sota_rows   = [r for r in ROWS if r["tag"] == "SOTA"]
    sota_scores = []
    for r in sota_rows:
        d = load(r["file"])
        if d and d.get("mAP@10"):
            sota_scores.append(d["mAP@10"])

    if not sota_scores:
        return

    best_sota = max(sota_scores)
    fusion    = load(f"{FUSION_DIR}/fusion_results.json")

    if fusion and fusion.get("mAP@10"):
        delta = fusion["mAP@10"] - best_sota
        print(f"Delta mAP@10 (fusion vs best SOTA): {delta:+.4f}")
        if delta > 0:
            print("=> Full fusion BEATS best SOTA baseline")
        else:
            print("=> Full fusion does not beat best SOTA — check fusion weights")
        print()

# ── main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print_status()
    print_table()
    print_delta()