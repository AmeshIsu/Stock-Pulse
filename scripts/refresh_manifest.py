"""Refresh reports/results_manifest.json with all tracked prediction files and configs."""
import glob
import hashlib
import json
import datetime
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]

files_to_hash = [
    "reports/results_long.csv",
    "config.yaml",
    "reports/08_provenance.csv"
]

# Add all 75 oos prediction parquets
oos_files = sorted(glob.glob(str(base_dir / "reports" / "oos_predictions" / "*.parquet")))
for f in oos_files:
    rel_path = Path(f).relative_to(base_dir).as_posix()
    files_to_hash.append(rel_path)

file_hashes = {}
for rel_path in sorted(files_to_hash):
    p = base_dir / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Missing file to track: {rel_path}")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    file_hashes[rel_path] = h

manifest = {
    "created_at": datetime.datetime.now().isoformat(),
    "description": "Cryptographic SHA-256 results freeze manifest for Stock Pulse",
    "files": file_hashes
}

manifest_path = base_dir / "reports" / "results_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Refreshed {manifest_path} with {len(file_hashes)} tracked files.")
