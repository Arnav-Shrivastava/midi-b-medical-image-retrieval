# corpus/generate_stub.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, json, os, random

# ── config ────────────────────────────────────────────────
OUTPUT_DIR   = "corpus/stub_images"
PAIRS_FILE   = "corpus/stub_pairs.json"
NUM_PAIRS    = 5000
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── medical text templates ────────────────────────────────
TEMPLATES = [
    "PA view {side} lung",
    "T2W axial slice {n}",
    "FLAIR sequence brain",
    "AP chest {n}mm",
    "DWI b{b} trace",
    "Coronal T1W {side}",
    "Sagittal MRI L{n}",
    "Ultrasound {organ}",
    "CT HU {n}",
    "LAT view thorax",
    "MG {side} CC view",
    "US {organ} longitudinal",
    "T1W post contrast {n}",
    "Axial FLAIR brain {n}mm",
    "Right {organ} US",
]

FILLS = {
    "{side}":  ["left", "right", "bilateral"],
    "{n}":     [str(i) for i in range(1, 20)],
    "{b}":     ["0", "500", "1000", "1500"],
    "{organ}": ["liver", "kidney", "thyroid", "spleen", "gallbladder"],
}

def fill_template(tmpl):
    text = tmpl
    for key, options in FILLS.items():
        text = text.replace(key, random.choice(options))
    return text

# ── rendering + degradation ───────────────────────────────
def render_patch(text):
    dpi    = random.randint(50, 100)
    width  = max(80, int(dpi * 3.5))
    height = max(20, int(dpi * 0.7))

    # background: light gray like a medical scan background
    bg_color = random.randint(190, 245)
    img  = Image.new("L", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # text color: dark
    text_color = random.randint(10, 80)
    x = random.randint(2, 8)
    y = random.randint(2, 6)
    draw.text((x, y), text, fill=text_color)

    # degradation 1: gaussian blur
    blur_radius = random.uniform(0.3, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # degradation 2: gaussian noise
    arr   = np.array(img, dtype=np.float32)
    sigma = random.uniform(5, 25)
    arr  += np.random.normal(0, sigma, arr.shape)
    arr   = np.clip(arr, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr)

    # degradation 3: random rotation (±3 degrees)
    angle = random.uniform(-3, 3)
    img   = img.rotate(angle, fillcolor=int(bg_color))

    return img

# ── generate pairs ────────────────────────────────────────
print(f"Generating {NUM_PAIRS} synthetic medical text patches...")
pairs = []

for i in range(NUM_PAIRS):
    tmpl = random.choice(TEMPLATES)
    text = fill_template(tmpl)

    img      = render_patch(text)
    filename = f"{i:05d}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # save with JPEG degradation (quality 20-50 like real MIDI-B)
    quality = random.randint(20, 50)
    img.save(filepath, "JPEG", quality=quality)

    pairs.append({"image": filepath, "text": text})

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{NUM_PAIRS} patches generated")

# ── save pairs manifest ───────────────────────────────────
with open(PAIRS_FILE, "w") as f:
    json.dump(pairs, f, indent=2)

print(f"\nDone. {len(pairs)} pairs saved to {PAIRS_FILE}")
print(f"Images saved to {OUTPUT_DIR}/")
print(f"\nSample texts generated:")
for p in random.sample(pairs, 5):
    print(f"  {p['text']}")