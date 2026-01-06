#!/usr/bin/env python3
# (cat /home/jd/t/git/WsiCtl/build/t0/all_csv/train/all_samples.csv && cat /home/jd/t/git/WsiCtl/build/t0/all_csv/val/all_samples.csv|grep -v dnaIndex) > di_all.csv

"""
Shuffle and split di_all.csv into train/val sets with different dnaIndex filters.

Usage:
    python shuffle_split_di_all_20_18.py --csv $ROOT/di_all.csv

Outputs:
    - $ROOT/diall/train/all_samples.csv, $ROOT/diall/val/all_samples.csv (no filter)
    - $ROOT/di20/train/all_samples.csv, $ROOT/di20/val/all_samples.csv (dnaIndex >= 2.0)
    - $ROOT/di18/train/all_samples.csv, $ROOT/di18/val/all_samples.csv (dnaIndex >= 1.8)
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shuffle and split CSV into train/val with dnaIndex filters"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV file (di_all.csv)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42)"
    )
    return parser.parse_args()


def ensure_dirs(root_dir, subdirs):
    """Create subdirectories if they don't exist."""
    for subdir in subdirs:
        train_dir = root_dir / subdir / "train"
        val_dir = root_dir / subdir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured directories: {train_dir}, {val_dir}")


def split_and_save(df, output_dir, train_ratio, name):
    """Split dataframe and save to train/val directories."""
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:]
    
    train_path = output_dir / "train" / "all_samples.csv"
    val_path = output_dir / "val" / "all_samples.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"[INFO] {name}: total={n_total}, train={len(train_df)}, val={len(val_df)}")
    print(f"       -> {train_path}")
    print(f"       -> {val_path}")


def main():
    args = parse_args()
    
    csv_path = Path(args.csv).resolve()
    root_dir = csv_path.parent
    
    print(f"[INFO] Input CSV: {csv_path}")
    print(f"[INFO] Root directory: {root_dir}")
    print(f"[INFO] Train ratio: {args.train_ratio}, Val ratio: {1 - args.train_ratio}")
    print(f"[INFO] Random seed: {args.seed}")
    print()
    
    # Step 1: Read CSV with header
    print("[STEP 1] Reading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns")
    print()
    
    # Step 2: Shuffle rows
    print("[STEP 2] Shuffling rows...")
    df_shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"[INFO] Shuffled {len(df_shuffled)} rows")
    print()
    
    # Ensure output directories exist
    subdirs = ["diall", "di20", "di18"]
    print("[INFO] Creating output directories...")
    ensure_dirs(root_dir, subdirs)
    print()
    
    # Step 3-1: No filter, split all data
    print("[STEP 3-1] Processing diall (no filter)...")
    split_and_save(df_shuffled, root_dir / "diall", args.train_ratio, "diall")
    print()
    
    # Step 3-2: Filter dnaIndex >= 2.0
    print("[STEP 3-2] Processing di20 (dnaIndex >= 2.0)...")
    df_di20 = df_shuffled[df_shuffled["dnaIndex"] >= 2.0].reset_index(drop=True)
    print(f"[INFO] Filtered: {len(df_di20)} rows (dnaIndex >= 2.0)")
    split_and_save(df_di20, root_dir / "di20", args.train_ratio, "di20")
    print()
    
    # Step 3-3: Filter dnaIndex >= 1.8
    print("[STEP 3-3] Processing di18 (dnaIndex >= 1.8)...")
    df_di18 = df_shuffled[df_shuffled["dnaIndex"] >= 1.8].reset_index(drop=True)
    print(f"[INFO] Filtered: {len(df_di18)} rows (dnaIndex >= 1.8)")
    split_and_save(df_di18, root_dir / "di18", args.train_ratio, "di18")
    print()
    
    print("[DONE] All splits completed successfully!")


if __name__ == "__main__":
    main()
