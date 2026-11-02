"""Extract the bundled dataset archive into ./data."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data_bundle" / "dataset_bundle.zip"
DEST = ROOT / "data"

def main():
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE}")
    required = [DEST / "Hindi" / "train.json", DEST / "Telugu" / "train.json", DEST / "English" / "train-00000-of-00001 (1).json"]
    if all(p.exists() for p in required):
        print(f"Dataset already extracted at {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(ROOT)
    if not all(p.exists() for p in required):
        raise RuntimeError("Extraction completed but expected dataset files were not found.")
    print(f"Dataset extracted to {DEST}")

if __name__ == "__main__":
    main()
