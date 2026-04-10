# baselines/run_clip.py
import torch, numpy as np, json, time, os, sys
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.map_harness import evaluate

# ── config ────────────────────────────────────────────────
IMAGE_DIR   = "data/processed/jpg"
GT_PATH     = "data/answer_key/ground_truth.json"
OUT_DIR     = "baselines/results"
BATCH_SIZE  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("Loading CLIP model...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# ── Step 1: collect all image paths ──────────────────────
print("Scanning images...")
image_paths = sorted(Path(IMAGE_DIR).rglob("*.jpg"))
img_ids     = [str(p) for p in image_paths]   # full path as ID
print(f"Found {len(img_ids)} images")

# ── Step 2: embed all images in batches ──────────────────
print(f"\nEmbedding {len(img_ids)} images (batch={BATCH_SIZE})...")
all_embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(img_ids), BATCH_SIZE)):
        batch_paths = img_ids[i : i + BATCH_SIZE]
        images = []
        valid_indices = []
        for j, p in enumerate(batch_paths):
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_indices.append(j)
            except Exception as e:
                print(f"  Skipping {p}: {e}")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        feats  = model.get_image_features(**inputs)
        feats  = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalize
        all_embeddings.append(feats.cpu().numpy())

E = np.vstack(all_embeddings).astype("float32")   # shape: [21770, 512]
print(f"\nEmbedding matrix: {E.shape}")

# save embeddings so you never have to recompute
np.save(f"{OUT_DIR}/clip_embeddings.npy", E)
with open(f"{OUT_DIR}/clip_img_ids.json", "w") as f:
    json.dump(img_ids, f)
print("Embeddings saved.")

# ── Step 3: load ground truth ─────────────────────────────
with open(GT_PATH) as f:
    ground_truth = json.load(f)

# skip queries with empty GT
ground_truth = {k: v for k, v in ground_truth.items() if len(v) > 0}
print(f"Queries after removing empty GT: {len(ground_truth)}")

# build a lookup: filename -> full path
fname_to_fullpath = {}
for p in img_ids:
    fname = os.path.basename(p)
    fname_to_fullpath[fname] = p

# ── Step 4: retrieve for each query ──────────────────────
print(f"\nRunning retrieval for {len(ground_truth)} queries...")
all_retrieved = {}
latencies     = []
skipped       = 0

for qid, gt_paths in ground_truth.items():
    gt_fnames = [os.path.basename(p) for p in gt_paths]

    # find first GT image that exists in our index
    query_fullpath = None
    for fname in gt_fnames:
        if fname in fname_to_fullpath:
            query_fullpath = fname_to_fullpath[fname]
            break

    if query_fullpath is None:
        print(f"  WARNING: no image found for query {qid}, skipping")
        skipped += 1
        continue

    query_idx = img_ids.index(query_fullpath)

    t0   = time.time()
    sims = E @ E[query_idx]
    ranked_indices = np.argsort(-sims).tolist()
    ranked_indices = [r for r in ranked_indices if r != query_idx]
    retrieved = [img_ids[r] for r in ranked_indices[:100]]
    latencies.append((time.time() - t0) * 1000)

    all_retrieved[qid] = retrieved

print(f"Skipped {skipped} queries (image not found in index)")
avg_latency = round(sum(latencies) / len(latencies), 2)
print(f"Avg retrieval latency: {avg_latency} ms/query")

# ── Step 5: evaluate ─────────────────────────────────────
results = evaluate(all_retrieved, ground_truth, system_name="CLIP ViT-B/32")
results["latency_ms_per_query"] = avg_latency
results["n_images"]             = len(img_ids)
results["n_queries"]            = len(all_retrieved)

with open(f"{OUT_DIR}/clip_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {OUT_DIR}/clip_results.json")