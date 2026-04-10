"""
MIDI-B · DICOM to JPG Batch Converter
======================================
Converts the MIDI-B-Synthetic-Validation dataset to JPG images,
stripping metadata to simulate the metadata-stripped real-world scenario.

Expected project layout:
    your-project/
    ├── data/
    │   ├── raw/
    │   │   └── MIDI-B-Synthetic-Validation/
    │   │       ├── 31780971/          ← numbered study folders
    │   │       ├── 66359456/
    │   │       └── ...
    │   └── processed/
    │       └── jpg/                   ← converted images written here
    ├── eval/
    └── dicom_to_jpg.py                ← run from project root

Usage (run from your project root):
    # Default — reads data/raw/MIDI-B-Synthetic-Validation, writes data/processed/jpg
    python dicom_to_jpg.py

    # Also generate a blank ground_truth.json stub
    python dicom_to_jpg.py --gt-stub

    # Override paths if needed
    python dicom_to_jpg.py --input data/raw/MIDI-B-Synthetic-Validation --output data/processed/jpg

Requirements:
    pip install pydicom Pillow numpy tqdm
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project paths  (relative to wherever you run this script from)
# ---------------------------------------------------------------------------
# Resolves to the directory containing this file, so paths work regardless
# of your shell's current working directory.
_HERE = Path(__file__).resolve().parent

# Where your raw MIDI-B DICOM files live
DEFAULT_INPUT  = _HERE / "data" / "raw" / "MIDI-B-Synthetic-Validation"

# Where converted JPGs will be written (created automatically if missing)
DEFAULT_OUTPUT = _HERE / "data" / "processed" / "jpg"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windowing helpers
# ---------------------------------------------------------------------------

def apply_window(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Apply DICOM window center / window width for proper contrast.
    Falls back to simple min-max normalisation if tags are missing.

    Window center and width are radiologist-chosen values that map a
    specific range of Hounsfield Units to the visible grey-scale range.
    Without this, many CT images look washed out or pitch-black.
    """
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is None or ww is None:
        # No windowing info — plain min-max stretch
        lo, hi = float(pixel_array.min()), float(pixel_array.max())
        if hi == lo:
            return np.zeros_like(pixel_array, dtype=np.uint8)
        normalised = (pixel_array.astype(np.float64) - lo) / (hi - lo) * 255.0
        return normalised.clip(0, 255).astype(np.uint8)

    # DICOM tags can be stored as MultiValue sequences — take the first element
    if hasattr(wc, "__iter__"):
        wc = float(wc[0])
        ww = float(ww[0])
    else:
        wc, ww = float(wc), float(ww)

    lo = wc - ww / 2.0
    hi = wc + ww / 2.0

    arr = pixel_array.astype(np.float64)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def apply_rescale(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """
    Apply RescaleSlope / RescaleIntercept (common in CT).
    Converts stored pixel values → Hounsfield Units (or equivalent).
    """
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return pixel_array.astype(np.float64) * slope + intercept


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------

def convert_dicom(dcm_path: Path, out_path: Path, fmt: str = "JPEG") -> dict:
    """
    Convert a single DICOM file to JPG (or PNG).

    Returns a metadata dict that gets written to the companion JSON log —
    useful later when you need to match images back to their modality,
    body part, or study ID without the full DICOM header.

    Parameters
    ----------
    dcm_path : Path   Path to the .dcm file.
    out_path : Path   Destination image path (with .jpg or .png extension).
    fmt      : str    Pillow format string — "JPEG" or "PNG".

    Returns
    -------
    dict with status, original filename, saved path, and key DICOM tags.
    """
    ds = pydicom.dcmread(str(dcm_path))

    # --- Pull pixel data ---
    pixel_array = ds.pixel_array  # shape: (H, W) or (frames, H, W) for multi-frame

    # Multi-frame DICOM (e.g. ultrasound cine loops) — take the middle frame
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]
    elif pixel_array.ndim == 3 and pixel_array.shape[2] in (3, 4):
        # Already RGB/RGBA — just normalise
        pixel_array = pixel_array[:, :, :3]
        img = Image.fromarray(pixel_array.astype(np.uint8), mode="RGB")
        img.save(str(out_path), format=fmt, quality=95)
        return _meta(ds, dcm_path, out_path, "ok", note="RGB passthrough")

    # --- Apply Rescale (CT Hounsfield conversion) ---
    pixel_array = apply_rescale(pixel_array, ds)

    # --- Apply photometric inversion if needed ---
    # MONOCHROME1 means 0 = white (inverted from normal); MONOCHROME2 is normal
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").strip()

    # --- Apply windowing → 0–255 uint8 ---
    uint8_array = apply_window(pixel_array, ds)

    if photometric == "MONOCHROME1":
        uint8_array = 255 - uint8_array

    img = Image.fromarray(uint8_array, mode="L")  # L = 8-bit greyscale

    # Upscale tiny images (some DICOM thumbnails are very small)
    if min(img.size) < 64:
        img = img.resize((max(img.width, 64), max(img.height, 64)), Image.LANCZOS)

    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95

    img.save(str(out_path), **save_kwargs)

    return _meta(ds, dcm_path, out_path, "ok")


def _meta(ds, dcm_path, out_path, status, note=""):
    """Extract key DICOM tags for the companion JSON log."""

    def safe(tag, default="unknown"):
        val = getattr(ds, tag, default)
        return str(val) if val is not None else default

    return {
        "status": status,
        "source": str(dcm_path),
        "output": str(out_path),
        "note": note,
        # Tags useful for grouping/analysis later
        "Modality": safe("Modality"),              # CT, MR, CR, US, XR ...
        "BodyPartExamined": safe("BodyPartExamined"),
        "StudyInstanceUID": safe("StudyInstanceUID"),
        "SeriesInstanceUID": safe("SeriesInstanceUID"),
        "SOPInstanceUID": safe("SOPInstanceUID"),  # unique image ID
        "PatientID": safe("PatientID"),            # anonymised in real datasets
        "Rows": safe("Rows"),
        "Columns": safe("Columns"),
        "BitsStored": safe("BitsStored"),
        "PhotometricInterpretation": safe("PhotometricInterpretation"),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def batch_convert(
    input_dir: Path,
    output_dir: Path,
    fmt: str = "JPEG",
    recursive: bool = True,
) -> dict:
    """
    Convert every DICOM file in input_dir to JPG/PNG in output_dir,
    preserving the folder structure.

    Returns a summary dict with counts and paths to log files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all DICOM files
    glob = "**/*.dcm" if recursive else "*.dcm"
    dcm_files = sorted(input_dir.glob(glob))

    if not dcm_files:
        # Some datasets use no extension for DICOM files
        log.warning(
            "No .dcm files found. Trying files with no extension..."
        )
        glob_no_ext = "**/*" if recursive else "*"
        dcm_files = [
            p for p in input_dir.glob(glob_no_ext)
            if p.is_file() and p.suffix == ""
        ]

    if not dcm_files:
        log.error("No DICOM files found in %s", input_dir)
        return {"converted": 0, "failed": 0, "skipped": 0}

    log.info("Found %d DICOM files. Starting conversion...", len(dcm_files))

    ext = ".jpg" if fmt == "JPEG" else ".png"
    records = []
    counts = {"converted": 0, "failed": 0, "skipped": 0}

    for dcm_path in tqdm(dcm_files, desc="Converting", unit="file", ncols=80):
        # Mirror directory structure in output
        try:
            rel = dcm_path.relative_to(input_dir)
        except ValueError:
            rel = Path(dcm_path.name)

        out_path = output_dir / rel.with_suffix(ext)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already converted (re-run friendly)
        if out_path.exists():
            counts["skipped"] += 1
            records.append({
                "status": "skipped",
                "source": str(dcm_path),
                "output": str(out_path),
            })
            continue

        try:
            record = convert_dicom(dcm_path, out_path, fmt=fmt)
            counts["converted"] += 1
        except Exception as exc:
            log.debug("Failed %s: %s", dcm_path.name, exc)
            log.debug(traceback.format_exc())
            record = {
                "status": "failed",
                "source": str(dcm_path),
                "output": str(out_path),
                "error": str(exc),
            }
            counts["failed"] += 1

        records.append(record)

    # --- Write companion JSON log ---
    log_path = output_dir / "conversion_log.json"
    with open(log_path, "w") as f:
        json.dump({"summary": counts, "records": records}, f, indent=2)

    # --- Write a simple image-id list (useful for mAP harness) ---
    id_list_path = output_dir / "image_ids.txt"
    converted_paths = [
        r["output"] for r in records if r["status"] == "ok"
    ]
    with open(id_list_path, "w") as f:
        f.write("\n".join(converted_paths))

    log.info(
        "Done. Converted: %d  |  Failed: %d  |  Skipped: %d",
        counts["converted"], counts["failed"], counts["skipped"],
    )
    log.info("Log saved to: %s", log_path)
    log.info("Image ID list saved to: %s", id_list_path)

    return {**counts, "log": str(log_path), "id_list": str(id_list_path)}


# ---------------------------------------------------------------------------
# Ground-truth stub helper
# ---------------------------------------------------------------------------

def build_gt_stub(id_list_path: Path, out_path: Path) -> None:
    """
    Reads the image_ids.txt file and writes a blank ground_truth.json
    with empty relevance lists. Fill this in from the MIDI-B answer keys.

    Format expected by the mAP harness:
        { "query_001": ["img_a.jpg", "img_b.jpg", ...], ... }
    """
    with open(id_list_path) as f:
        image_ids = [line.strip() for line in f if line.strip()]

    gt = {f"query_{i:04d}": [] for i in range(len(image_ids))}

    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2)

    log.info(
        "Blank ground_truth.json written to %s — fill from MIDI-B answer keys.",
        out_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert MIDI-B DICOM files to JPG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder containing DICOM files. Default: data/raw/MIDI-B-Synthetic-Validation",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for converted images. Default: data/processed/jpg",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["JPEG", "PNG"],
        default="JPEG",
        help="Output image format. JPEG is smaller; PNG is lossless.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only convert files directly in --input, not subfolders.",
    )
    parser.add_argument(
        "--gt-stub",
        action="store_true",
        help="After conversion, write a blank ground_truth.json stub.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("Input  : %s", args.input)
    log.info("Output : %s", args.output)

    if not args.input.exists():
        log.error(
            "Input folder not found: %s\n"
            "Make sure your DICOM data is at data/raw/MIDI-B-Synthetic-Validation/ "
            "relative to this script, or pass --input <your path>.",
            args.input,
        )
        raise SystemExit(1)

    result = batch_convert(
        input_dir=args.input,
        output_dir=args.output,
        fmt=args.format,
        recursive=not args.no_recursive,
    )

    if args.gt_stub and result.get("id_list"):
        build_gt_stub(
            id_list_path=Path(result["id_list"]),
            out_path=args.output / "ground_truth.json",
        )

    # Exit with error code if any conversions failed (useful in CI/scripts)
    if result.get("failed", 0) > 0:
        log.warning(
            "%d files failed to convert. Check conversion_log.json for details.",
            result["failed"],
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()