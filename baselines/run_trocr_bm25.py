# baselines/run_trocr_bm25.py
import torch, json, time, os, sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.map_harness import evaluate

# ── config ────────────────────────────────────────────────
IMAGE_DIR  = "data/processed/jpg"
GT_PATH    = "data/answer_key/ground_truth.json"
OUT_DIR    = "baselines/results"
BATCH_SIZE = 16        # TrOCR is heavy, smaller batch
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Step 1: load model ────────────────────────────────────
print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model     = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed").to(DEVICE)
model.eval()
print("Model loaded.")

# ── Step 2: collect all image paths ──────────────────────
print("Scanning images...")
image_paths = sorted(Path(IMAGE_DIR).rglob("*.jpg"))
img_ids     = [str(p) for p in image_paths]
print(f"Found {len(img_ids)} images")

# ── Step 3: OCR every image ───────────────────────────────
ocr_cache_path = f"{OUT_DIR}/trocr_texts.json"

if os.path.exists(ocr_cache_path):
    print(f"Loading cached OCR results from {ocr_cache_path}")
    with open(ocr_cache_path) as f:
        corpus_texts = json.load(f)
else:
    print(f"\nRunning TrOCR on {len(img_ids)} images (batch={BATCH_SIZE})...")
    corpus_texts = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(img_ids), BATCH_SIZE)):
            batch_paths = img_ids[i : i + BATCH_SIZE]
            images, valid_ids = [], []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    valid_ids.append(p)
                except Exception:
                    pass

            if not images:
                continue

            pixel_values = processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(DEVICE)

            generated_ids = model.generate(pixel_values)
            texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for path, text in zip(valid_ids, texts):
                corpus_texts[path] = text.strip()

    # save so we never rerun OCR
    with open(ocr_cache_path, "w") as f:
        json.dump(corpus_texts, f)
    print(f"OCR done. Saved to {ocr_cache_path}")

print(f"OCR results: {len(corpus_texts)} images")

# sample a few to see what TrOCR is extracting
sample_ids = list(corpus_texts.keys())[:5]
print("\nSample OCR outputs:")
for sid in sample_ids:
    print(f"  {os.path.basename(sid)}: '{corpus_texts[sid]}'")

# ── Step 4: build BM25 index ──────────────────────────────
print("\nBuilding BM25 index...")
indexed_ids    = list(corpus_texts.keys())
tokenized      = [corpus_texts[i].lower().split() for i in indexed_ids]

# handle empty texts — BM25 needs at least something
tokenized = [t if t else ["__empty__"] for t in tokenized]
bm25 = BM25Okapi(tokenized)
print(f"BM25 index built over {len(indexed_ids)} documents")

# ── Step 5: load ground truth ─────────────────────────────
with open(GT_PATH) as f:
    ground_truth = json.load(f)
ground_truth = {k: v for k, v in ground_truth.items() if len(v) > 0}

fname_to_fullpath = {os.path.basename(p): p for p in indexed_ids}

# ── Step 6: retrieve for each query ──────────────────────
print(f"\nRunning BM25 retrieval for {len(ground_truth)} queries...")
all_retrieved = {}
latencies     = []
skipped       = 0

for qid, gt_paths in ground_truth.items():
    gt_fnames = [os.path.basename(p) for p in gt_paths]

    query_fullpath = None
    for fname in gt_fnames:
        if fname in fname_to_fullpath:
            query_fullpath = fname_to_fullpath[fname]
            break

    if query_fullpath is None:
        skipped += 1
        continue

    # use the OCR text of the query image as the BM25 query
    query_text = corpus_texts.get(query_fullpath, "").lower().split()
    if not query_text:
        query_text = ["__empty__"]

    t0     = time.time()
    scores = bm25.get_scores(query_text)

    # rank by score, exclude the query image itself
    query_idx      = indexed_ids.index(query_fullpath)
    ranked_indices = sorted(
        range(len(indexed_ids)),
        key=lambda i: scores[i],
        reverse=True
    )
    ranked_indices = [r for r in ranked_indices if r != query_idx]
    retrieved      = [indexed_ids[r] for r in ranked_indices[:100]]
    latencies.append((time.time() - t0) * 1000)

    all_retrieved[qid] = retrieved

print(f"Skipped {skipped} queries")
avg_latency = round(sum(latencies) / len(latencies), 2)
print(f"Avg retrieval latency: {avg_latency} ms/query")

# ── Step 7: evaluate ─────────────────────────────────────
results = evaluate(all_retrieved, ground_truth, system_name="TrOCR+BM25")
results["latency_ms_per_query"] = avg_latency
results["n_images"]             = len(img_ids)
results["n_queries"]            = len(all_retrieved)

with open(f"{OUT_DIR}/trocr_bm25_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUT_DIR}/trocr_bm25_results.json")