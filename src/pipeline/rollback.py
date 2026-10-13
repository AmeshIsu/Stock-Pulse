"""Stock Pulse Rollback CLI.

Usage: python -m src.pipeline.rollback --to VERSION
"""
import argparse
import json
import os
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
VERSIONS_DIR = BASE_DIR / "data/versions"
SERVING_DIR = BASE_DIR / "data/serving"
PROCESSED_DIR = BASE_DIR / "data/processed"


def atomic_write_text(file_path: Path, content: str):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, file_path)


def list_versions():
    if not VERSIONS_DIR.exists():
        print("No versions found (data/versions directory does not exist).")
        return

    cur_file = VERSIONS_DIR / "CURRENT"
    current_ver = cur_file.read_text(encoding="utf-8").strip() if cur_file.exists() else "None"
    
    versions = [d.name for d in VERSIONS_DIR.iterdir() if d.is_dir()]
    print("\nAvailable Data Versions:")
    print(f"Current Active Version: {current_ver}\n")
    for v in sorted(versions):
        log_file = VERSIONS_DIR / v / "refresh_log.json"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            as_of = log_data.get("as_of_date", "Unknown")
            mode = log_data.get("mode", "Unknown")
            ts = log_data.get("timestamp", "Unknown")
            mark = " (ACTIVE)" if v == current_ver else ""
            print(f"- {v}: as-of {as_of}, mode={mode}, created={ts}{mark}")
        else:
            print(f"- {v}")
    print()


def rollback_to(version_id: str):
    target_dir = VERSIONS_DIR / version_id
    if not target_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {target_dir}")

    log_file = target_dir / "refresh_log.json"
    if not log_file.exists():
        raise FileNotFoundError(f"refresh_log.json missing in version directory: {target_dir}")

    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)
        
    as_of_date = log_data.get("as_of_date")
    if not as_of_date:
        raise ValueError("Could not determine as_of_date from refresh_log.json")

    print(f"Rolling back to version {version_id} (As-of date: {as_of_date})...")

    # 1. Restore processed files
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["prices_clean.parquet", "features_targets.parquet", "latest_features.parquet", "rankable_mask.parquet"]:
        v_file = target_dir / fname
        if v_file.exists():
            shutil.copy2(v_file, PROCESSED_DIR / fname)
            print(f"Restored data/processed/{fname}")

    # 2. Restore serving directory
    v_serving = target_dir / "serving"
    if v_serving.exists():
        active_serving_target = SERVING_DIR / as_of_date
        if active_serving_target.exists():
            shutil.rmtree(active_serving_target)
        shutil.copytree(v_serving, active_serving_target)
        print(f"Restored data/serving/{as_of_date}/")

    # 3. Atomically update pointers
    atomic_write_text(SERVING_DIR / "CURRENT", as_of_date)
    atomic_write_text(VERSIONS_DIR / "CURRENT", version_id)
    print(f"Updated pointers: SERVING CURRENT -> {as_of_date}, VERSIONS CURRENT -> {version_id}")
    print(f"Rollback to {version_id} completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Stock Pulse Rollback CLI")
    parser.add_argument("--to", help="Version ID (e.g. timestamp folder) to roll back to")
    parser.add_argument("--list", action="store_true", help="List all available versions")

    args = parser.parse_args()
    if args.list or not args.to:
        list_versions()
        if not args.to:
            print("To rollback to a specific version, run: python -m src.pipeline.rollback --to <VERSION_ID>")
    else:
        rollback_to(args.to)


if __name__ == "__main__":
    main()
