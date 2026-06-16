"""
brain/preprocess_fmri.py

Converts raw resting-state fMRI (NIfTI) to regional timeseries arrays
ready for the unified-mind pipeline.

Workflow:
  1. Load each subject's fMRI .nii.gz file
  2. Extract regional mean timeseries using Schaefer 100-region atlas
  3. Apply basic signal cleaning (detrend, bandpass, confound regression)
  4. Save as (T, N) float32 .npy array per subject
  5. Save group metadata CSV (subject_id, group, filepath)

Usage:
  # Single subject (test)
  python brain/preprocess_fmri.py --data path/to/cobre/ --test

  # Full dataset
  python brain/preprocess_fmri.py --data path/to/cobre/

  # Then run the pipeline per subject:
  python run.py tokenize --timeseries brain/data/subjects/sub-001_timeseries.npy

Requirements:
  pip install nilearn nibabel pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def get_atlas():
    """Load Schaefer 100-region parcellation via nilearn."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    print(f"Atlas: Schaefer 2018, 100 ROIs")
    return atlas.maps, atlas.labels


def extract_timeseries(fmri_path: str, atlas_img,
                       confounds_path: str = None,
                       t_r: float = 2.0) -> np.ndarray:
    """
    Extract mean regional timeseries from a 4D fMRI NIfTI file.

    Args:
        fmri_path:      path to 4D .nii.gz file
        atlas_img:      parcellation atlas image
        confounds_path: optional path to confounds .tsv (motion params etc)
        t_r:            repetition time in seconds (COBRE default: 2.0s)

    Returns:
        timeseries: (T, N) float32 array, T=volumes, N=regions
    """
    from nilearn.input_data import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img      = atlas_img,
        standardize     = True,       # z-score each region
        detrend         = True,       # remove linear trend
        low_pass        = 0.1,        # bandpass filter (Hz)
        high_pass       = 0.01,
        t_r             = t_r,
        memory_level    = 1,
        verbose         = 0,
    )

    confounds = None
    if confounds_path and Path(confounds_path).exists():
        conf_df   = pd.read_csv(confounds_path, sep="\t")
        # Use standard motion parameters if available
        motion_cols = [c for c in conf_df.columns
                       if any(m in c for m in
                              ["trans_x","trans_y","trans_z",
                               "rot_x","rot_y","rot_z"])]
        if motion_cols:
            confounds = conf_df[motion_cols].fillna(0).values

    ts = masker.fit_transform(fmri_path, confounds=confounds)
    return ts.astype(np.float32)


def find_subjects_cobre(data_dir: Path) -> pd.DataFrame:
    """
    Find all fMRI files in COBRE directory structure.
    COBRE layout: data_dir/sub-*/func/sub-*_task-rest_bold.nii.gz
                  or: data_dir/sub-*/func/sub-*_bold.nii.gz
    Also looks for a participants.tsv with group labels.
    """
    rows = []

    # Try to find participants.tsv for group labels
    group_map = {}
    for tsv in data_dir.rglob("participants.tsv"):
        df = pd.read_csv(tsv, sep="\t")
        if "participant_id" in df.columns and "diagnosis" in df.columns:
            for _, row in df.iterrows():
                pid = str(row["participant_id"]).replace("sub-", "")
                group_map[pid] = str(row["diagnosis"])
        elif "participant_id" in df.columns and "group" in df.columns:
            for _, row in df.iterrows():
                pid = str(row["participant_id"]).replace("sub-", "")
                group_map[pid] = str(row["group"])
        break

    # Find all fMRI files
    patterns = [
        "**/*task-rest*bold.nii.gz",
        "**/*_bold.nii.gz",
        "**/*rest*.nii.gz",
        "**/*.nii.gz",
    ]

    found = set()
    for pattern in patterns:
        for f in data_dir.glob(pattern):
            if f not in found:
                found.add(f)
                # Extract subject ID from path
                parts = f.parts
                sub_id = None
                for part in parts:
                    if part.startswith("sub-") or part.startswith("0") or part.isdigit():
                        sub_id = part.replace("sub-", "")
                        break
                if sub_id is None:
                    sub_id = f.stem.split("_")[0].replace("sub-", "")

                group = group_map.get(sub_id, "unknown")

                # Look for confounds file nearby
                confounds = None
                for cf in f.parent.glob("*confounds*.tsv"):
                    confounds = str(cf)
                    break

                rows.append({
                    "subject_id": sub_id,
                    "group":      group,
                    "fmri_path":  str(f),
                    "confounds":  confounds,
                })

    if not rows:
        raise FileNotFoundError(
            f"No NIfTI files found in {data_dir}. "
            "Check that the COBRE data downloaded correctly."
        )

    df = pd.DataFrame(rows).drop_duplicates("subject_id")
    print(f"Found {len(df)} subjects")
    if group_map:
        print(f"  Groups: {df['group'].value_counts().to_dict()}")
    else:
        print("  No participants.tsv found — group labels will be 'unknown'")
        print("  You can add them manually to brain/data/subjects/metadata.csv")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True,
                        help="Path to COBRE data directory")
    parser.add_argument("--outdir", default="brain/data/subjects",
                        help="Where to save per-subject .npy files")
    parser.add_argument("--tr",     type=float, default=2.0,
                        help="fMRI repetition time in seconds (COBRE=2.0)")
    parser.add_argument("--test",   action="store_true",
                        help="Process only first subject (sanity check)")
    parser.add_argument("--n-rois", type=int, default=100,
                        help="Number of atlas regions (100 or 200)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading atlas...")
    atlas_img, atlas_labels = get_atlas()

    print(f"\nScanning {data_dir} for subjects...")
    subjects = find_subjects_cobre(data_dir)

    if args.test:
        subjects = subjects.iloc[:1]
        print("Test mode: processing 1 subject only")

    results = []
    failed  = []

    for _, row in subjects.iterrows():
        sub_id   = row["subject_id"]
        out_path = outdir / f"sub-{sub_id}_timeseries.npy"

        if out_path.exists():
            print(f"  sub-{sub_id}: already exists, skipping")
            results.append({**row.to_dict(), "timeseries_path": str(out_path),
                            "n_volumes": "cached", "n_regions": "cached"})
            continue

        print(f"  sub-{sub_id} ({row['group']})...", end=" ", flush=True)
        try:
            ts = extract_timeseries(
                fmri_path      = row["fmri_path"],
                atlas_img      = atlas_img,
                confounds_path = row["confounds"],
                t_r            = args.tr,
            )
            np.save(out_path, ts)
            print(f"shape={ts.shape}  ✓")
            results.append({**row.to_dict(), "timeseries_path": str(out_path),
                            "n_volumes": ts.shape[0], "n_regions": ts.shape[1]})
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(sub_id)

    # Save metadata
    meta_path = outdir / "metadata.csv"
    pd.DataFrame(results).to_csv(meta_path, index=False)
    print(f"\nSaved metadata: {meta_path}")

    if failed:
        print(f"Failed subjects ({len(failed)}): {failed}")

    print(f"\nSuccessfully processed: {len(results) - len(failed)}/{len(subjects)}")
    print(f"\nNext: python run.py group-analysis")


if __name__ == "__main__":
    main()
