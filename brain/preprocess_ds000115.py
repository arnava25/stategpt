"""
brain/preprocess_ds000115.py

Extracts regional fMRI timeseries from ds000115 (working memory in
healthy and schizophrenic individuals) for use in the unified-mind pipeline.

For each subject, extracts timeseries for each task condition:
  - letter0backtask  (easiest, baseline cognitive load)
  - letter1backtask  (medium load)
  - letter2backtask  (hardest, highest load)

Saves per-subject per-condition .npy arrays ready for run.py tokenize.

Usage:
    # Test on one subject first
    python brain/preprocess_ds000115.py --data ~/Desktop/ds000115 --test

    # Full dataset
    python brain/preprocess_ds000115.py --data ~/Desktop/ds000115

Requirements:
    pip install nilearn nibabel pandas
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = [
    "letter0backtask",
    "letter1backtask",
    "letter2backtask",
]

TR = 2.0  # repetition time in seconds (from dataset description)


def get_atlas():
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    print(f"  Atlas: Schaefer 2018, 100 ROIs")
    return atlas.maps, [str(l) for l in atlas.labels]


def extract_timeseries(fmri_path, atlas_img, t_r=2.0):
    """Extract mean regional timeseries from a 4D fMRI file."""
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img  = atlas_img,
        standardize = "zscore_sample",
        detrend     = True,
        low_pass    = 0.1,
        high_pass   = 0.01,
        t_r         = t_r,
        memory_level= 1,
        verbose     = 0,
    )
    ts = masker.fit_transform(str(fmri_path))
    return ts.astype(np.float32)


def find_fmri_files(data_dir, subject_id):
    """
    Find all task fMRI files for a subject.
    BIDS layout: sub-XX/func/sub-XX_task-{task}_bold.nii.gz
    """
    sub_dir  = data_dir / subject_id / "func"
    found    = {}

    if not sub_dir.exists():
        return found

    for task in TASKS:
        # Try both compressed and uncompressed
        for ext in ["_bold.nii.gz", "_bold.nii"]:
            candidate = sub_dir / f"{subject_id}_task-{task}{ext}"
            if candidate.exists():
                found[task] = candidate
                break

    return found


def load_participants(data_dir):
    tsv_path = data_dir / "participants.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"participants.tsv not found in {data_dir}")
    return pd.read_csv(tsv_path, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True,
                        help="Path to ds000115 directory")
    parser.add_argument("--outdir", default="brain/data/subjects",
                        help="Output directory for .npy files")
    parser.add_argument("--test",   action="store_true",
                        help="Process first subject only")
    parser.add_argument("--tr",     type=float, default=2.0)
    args = parser.parse_args()

    data_dir = Path(args.data)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading participants...")
    participants = load_participants(data_dir)
    print(f"  {len(participants)} subjects")
    print(f"  Groups: {participants['condit'].value_counts().to_dict()}")

    print("\nLoading atlas...")
    atlas_img, atlas_labels = get_atlas()

    if args.test:
        participants = participants.iloc[:1]
        print(f"\nTest mode: processing {participants.iloc[0]['participant_id']} only")

    results  = []
    failed   = []
    metadata = []

    for _, row in participants.iterrows():
        sub_id = row["participant_id"]
        condit = row["condit"]
        print(f"\n{sub_id} ({condit})")

        fmri_files = find_fmri_files(data_dir, sub_id)

        if not fmri_files:
            print(f"  No fMRI files found — skipping")
            failed.append(sub_id)
            continue

        sub_results = {"subject_id": sub_id, "condit": condit,
                       "age": row.get("age", None),
                       "gender": row.get("gender", None)}

        # Add symptom scores for SCZ subjects
        for col in ["saps7", "saps20", "saps25", "saps34",
                    "sans8", "sans13", "sans17", "sans22", "sans25"]:
            sub_results[col] = row.get(col, None)

        # Add nback behavioral scores
        for col in ["nback0_targ", "nback1_targ", "nback2_targ",
                    "nback0_targ_medrt", "nback1_targ_medrt", "nback2_targ_medrt"]:
            sub_results[col] = row.get(col, None)

        for task, fmri_path in fmri_files.items():
            out_path = outdir / f"{sub_id}_task-{task}_timeseries.npy"

            if out_path.exists():
                existing = np.load(out_path)
                print(f"  {task}: already exists {existing.shape} — skipping")
                sub_results[f"{task}_path"]    = str(out_path)
                sub_results[f"{task}_volumes"] = existing.shape[0]
                sub_results[f"{task}_regions"] = existing.shape[1]
                continue

            print(f"  {task}: extracting...", end=" ", flush=True)
            try:
                ts = extract_timeseries(fmri_path, atlas_img, t_r=args.tr)
                np.save(out_path, ts)
                print(f"shape={ts.shape} ✓")
                sub_results[f"{task}_path"]    = str(out_path)
                sub_results[f"{task}_volumes"] = ts.shape[0]
                sub_results[f"{task}_regions"] = ts.shape[1]
                results.append(sub_id)
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append(f"{sub_id}_{task}")

        metadata.append(sub_results)

    # Save metadata
    meta_df   = pd.DataFrame(metadata)
    meta_path = outdir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved metadata: {meta_path}")

    print(f"\nProcessed: {len(results)} subjects")
    if failed:
        print(f"Failed: {failed}")

    # Print summary of what's available
    print("\nAvailable timeseries per task:")
    for task in TASKS:
        col = f"{task}_path"
        if col in meta_df.columns:
            n = meta_df[col].notna().sum()
            print(f"  {task}: {n}/{len(meta_df)} subjects")

    print(f"\nNext: python run.py group-analysis --subjects {outdir}")


if __name__ == "__main__":
    main()
