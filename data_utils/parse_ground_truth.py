"""
MIDI-B · Ground Truth Parser
=============================
Builds ground_truth.json from the MIDI-B answer key database and
the two CSV mapping files, then cross-references with your converted
JPG files to produce a retrieval-ready ground truth.

Output files (written to data/answer_key/):
    ground_truth.json       — {query_id: [jpg_path, ...]} for mAP harness
    query_index.json        — {query_id: patient_folder} for reference
    parser_report.json      — stats and any warnings

Usage:
    python parse_ground_truth.py

Requirements:
    pip install pandas tqdm
"""

import json
import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

DB_PATH              = _HERE / "data" / "answer_key" / "MIDI-B-Answer-Key-Validation.db"
PATIENT_MAP_PATH     = _HERE / "data" / "answer_key" / "MIDI-B-Patient-Mapping-Validation.csv"
UID_MAP_PATH         = _HERE / "data" / "answer_key" / "MIDI-B-UID-Mapping-Validation.csv"
JPG_DIR              = _HERE / "data" / "processed" / "jpg"
OUT_DIR              = _HERE / "data" / "answer_key"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Load mapping CSVs
# ---------------------------------------------------------------------------

def load_patient_map(path: Path) -> dict:
    """
    validation_patient_mapping.csv
    id_old = folder name in MIDI-B-Synthetic-Validation  (e.g. 31780971)
    id_new = query ID used in the answer key             (e.g. MIDI_1_1_001)

    Returns: {folder_name_str: query_id}
    """
    df = pd.read_csv(path, dtype=str)
    mapping = dict(zip(df["id_old"].str.strip(), df["id_new"].str.strip()))
    log.info("Patient map loaded: %d entries", len(mapping))
    return mapping


def load_uid_map(path: Path) -> dict:
    """
    validation_uid_mapping.csv
    id_old = original SOP Instance UID (stored in the .db col[7])
    id_new = standardized UID used externally

    Returns: {old_uid: new_uid}
    """
    df = pd.read_csv(path, dtype=str)
    mapping = dict(zip(df["id_old"].str.strip(), df["id_new"].str.strip()))
    log.info("UID map loaded: %d entries", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Step 2 — Load answer key DB
# ---------------------------------------------------------------------------

def load_answer_db(db_path: Path) -> pd.DataFrame:
    """
    Reads answer_data table and assigns meaningful column names.

    Schema (inferred from data inspection):
        col[0]  row_index
        col[1]  group_id
        col[2]  modality        (PT, CT, MR, ...)
        col[3]  sop_class_uid
        col[4]  patient_id      (matches folder name)
        col[5]  study_uid
        col[6]  series_uid
        col[7]  sop_instance_uid  (unique per image, matches uid_map id_old)
        col[8]  pixel_hash      (MD5 of pixel data)
        col[9]  audit_json      (de-identification log — ignore)
    """
    conn = sqlite3.connect(str(db_path))
    df = conn.execute("SELECT * FROM answer_data").fetchall()
    conn.close()

    columns = [
        "row_index", "group_id", "modality", "sop_class_uid",
        "patient_id", "study_uid", "series_uid", "sop_instance_uid",
        "pixel_hash", "audit_json"
    ]
    df = pd.DataFrame(df, columns=columns)
    df = df.drop(columns=["audit_json"])  # large JSON, not needed
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["sop_instance_uid"] = df["sop_instance_uid"].astype(str).str.strip()

    log.info("Answer DB loaded: %d rows, %d unique patients",
             len(df), df["patient_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Step 3 — Build JPG lookup table
# ---------------------------------------------------------------------------

def build_jpg_index(jpg_dir: Path) -> dict:
    """
    Scans data/processed/jpg/ and builds two lookup dicts:

    by_patient  : {patient_folder_name: [jpg_path, ...]}
    by_stem     : {filename_without_extension: jpg_path}

    The DICOM conversion script preserves the original filename stem
    (which is the SOP Instance UID or a hash), so we can match on stem.
    """
    log.info("Scanning JPG directory: %s", jpg_dir)
    jpg_files = list(jpg_dir.rglob("*.jpg"))
    log.info("Found %d JPG files", len(jpg_files))

    by_patient = defaultdict(list)
    by_stem    = {}

    for p in jpg_files:
        # Patient folder is the top-level subfolder under jpg/
        # e.g. data/processed/jpg/31780971/series/.../image.jpg
        parts = p.relative_to(jpg_dir).parts
        patient_folder = parts[0] if parts else "unknown"
        by_patient[patient_folder].append(str(p))
        by_stem[p.stem] = str(p)

    log.info("JPG index: %d patient folders", len(by_patient))
    return by_patient, by_stem


# ---------------------------------------------------------------------------
# Step 4 — Build ground truth
# ---------------------------------------------------------------------------

def build_ground_truth(
    df: pd.DataFrame,
    patient_map: dict,
    uid_map: dict,
    by_patient: dict,
    by_stem: dict,
) -> tuple[dict, dict]:
    """
    Constructs:
        ground_truth  : {query_id: [jpg_path, ...]}
        query_index   : {query_id: {patient_id, modalities, image_count}}

    Strategy:
    1. Group DB rows by patient_id
    2. Map patient_id → query_id via patient_map
    3. For each image row, find its JPG via:
       a. sop_instance_uid stem match (most reliable)
       b. pixel_hash match (fallback)
       c. patient folder match (last resort — includes ALL images for patient)
    """
    ground_truth = defaultdict(list)
    query_index  = {}
    warnings     = []

    # Build pixel_hash → jpg_path lookup from stems
    # (some converters name files by hash)
    by_hash = {}
    for stem, path in by_stem.items():
        by_hash[stem] = path

    # Group by patient
    grouped = df.groupby("patient_id")

    unmatched_patients = []
    total_matched_images = 0

    for patient_id, group in tqdm(grouped, desc="Building ground truth"):
        # Map patient → query ID
        query_id = patient_map.get(patient_id)
        if query_id is None:
            warnings.append(f"No query ID for patient {patient_id}")
            unmatched_patients.append(patient_id)
            continue

        matched_jpgs = []
        modalities = set(group["modality"].tolist())

        for _, row in group.iterrows():
            sop_uid = row["sop_instance_uid"]
            pixel_hash = row["pixel_hash"]
            jpg_path = None

            # Strategy A: match by SOP Instance UID (exact stem)
            if sop_uid in by_stem:
                jpg_path = by_stem[sop_uid]

            # Strategy B: match by pixel hash (MD5 used as filename)
            if jpg_path is None and pixel_hash in by_stem:
                jpg_path = by_stem[pixel_hash]

            # Strategy C: match by pixel hash in hash index
            if jpg_path is None and pixel_hash in by_hash:
                jpg_path = by_hash[pixel_hash]

            if jpg_path is not None:
                matched_jpgs.append(jpg_path)

        # Strategy D (fallback): if no UID-level matches, use all JPGs
        # in the patient folder. This is less precise but still valid
        # since each patient folder = one study = one query.
        if not matched_jpgs:
            patient_folder_jpgs = by_patient.get(patient_id, [])
            if patient_folder_jpgs:
                matched_jpgs = patient_folder_jpgs
                warnings.append(
                    f"Patient {patient_id} ({query_id}): used folder fallback "
                    f"({len(patient_folder_jpgs)} images)"
                )
            else:
                warnings.append(
                    f"Patient {patient_id} ({query_id}): NO images found at all"
                )

        ground_truth[query_id] = matched_jpgs
        total_matched_images += len(matched_jpgs)

        query_index[query_id] = {
            "patient_id": patient_id,
            "modalities": sorted(modalities),
            "db_image_count": len(group),
            "matched_jpg_count": len(matched_jpgs),
        }

    log.info(
        "Ground truth built: %d queries, %d total images matched, "
        "%d unmatched patients",
        len(ground_truth), total_matched_images, len(unmatched_patients)
    )

    return dict(ground_truth), query_index, warnings


# ---------------------------------------------------------------------------
# Step 5 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(ground_truth, query_index, warnings, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ground_truth.json — used directly by mAP harness
    gt_path = out_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    log.info("Saved: %s", gt_path)

    # query_index.json — useful for debugging and analysis
    qi_path = out_dir / "query_index.json"
    with open(qi_path, "w") as f:
        json.dump(query_index, f, indent=2)
    log.info("Saved: %s", qi_path)

    # parser_report.json — stats and warnings
    report = {
        "total_queries": len(ground_truth),
        "total_images": sum(len(v) for v in ground_truth.values()),
        "queries_with_no_images": sum(1 for v in ground_truth.values() if not v),
        "avg_images_per_query": round(
            sum(len(v) for v in ground_truth.values()) / max(len(ground_truth), 1), 1
        ),
        "warnings": warnings,
    }
    rp_path = out_dir / "parser_report.json"
    with open(rp_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved: %s", rp_path)

    # Print summary
    print("\n" + "="*55)
    print("GROUND TRUTH PARSER — SUMMARY")
    print("="*55)
    print(f"  Total queries        : {report['total_queries']}")
    print(f"  Total images matched : {report['total_images']}")
    print(f"  Avg images/query     : {report['avg_images_per_query']}")
    print(f"  Queries with 0 images: {report['queries_with_no_images']}")
    print(f"  Warnings             : {len(warnings)}")
    print("="*55)

    if warnings:
        print("\nFirst 10 warnings:")
        for w in warnings[:10]:
            print(f"  ! {w}")

    print(f"\nOutputs written to: {out_dir}")
    print("  ground_truth.json  — feed this to your mAP harness")
    print("  query_index.json   — query metadata for reference")
    print("  parser_report.json — full stats and warnings")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Validate inputs exist
    for path, label in [
        (DB_PATH,          "Answer key DB"),
        (PATIENT_MAP_PATH, "Patient mapping CSV"),
        (UID_MAP_PATH,     "UID mapping CSV"),
        (JPG_DIR,          "JPG directory"),
    ]:
        if not path.exists():
            log.error("%s not found: %s", label, path)
            raise SystemExit(1)

    log.info("Loading mapping files...")
    patient_map = load_patient_map(PATIENT_MAP_PATH)
    uid_map     = load_uid_map(UID_MAP_PATH)

    log.info("Loading answer key database...")
    df = load_answer_db(DB_PATH)

    log.info("Building JPG index...")
    by_patient, by_stem = build_jpg_index(JPG_DIR)

    log.info("Building ground truth...")
    ground_truth, query_index, warnings = build_ground_truth(
        df, patient_map, uid_map, by_patient, by_stem
    )

    save_outputs(ground_truth, query_index, warnings, OUT_DIR)


if __name__ == "__main__":
    main()