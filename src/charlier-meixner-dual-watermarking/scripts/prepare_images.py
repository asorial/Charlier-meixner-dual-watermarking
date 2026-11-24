"""
Copy the test images used in the paper from a local copy of the RGB-BMP Steganalysis
Dataset (Al-Jarrah, 2018, Mendeley Data, doi:10.17632/sp4g8h7v8k.1) and check them
against the SHA-256 checksums in data/images_manifest.csv.

The images are not redistributed in this repository. Download the dataset from
https://doi.org/10.17632/sp4g8h7v8k.1, extract it anywhere, and run:

    python scripts/prepare_images.py --source /path/to/extracted/dataset

Destinations:
    src/static/cover_images/   24 test images   (Secs. 3.1-3.4, 3.7-3.10; E1, E2, E4-E6, E8)
    data/covers100/            100 images       (Secs. 3.5-3.6; E3, E9)

The source folder is searched recursively, so the internal layout of the archive
does not matter. Files whose checksum differs from the manifest are reported and
not copied.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MANIFEST = os.path.join(ROOT, "data", "images_manifest.csv")
DEST = {
    "test24": os.path.join(ROOT, "src", "static", "cover_images"),
    "ablation100": os.path.join(ROOT, "data", "covers100"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder with the extracted dataset")
    ap.add_argument("--sets", nargs="+", default=list(DEST), choices=list(DEST),
                    help="image sets to prepare (default: all)")
    a = ap.parse_args()

    with open(MANIFEST, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["set"] in a.sets]
    wanted = {r["file"].lower() for r in rows}

    found = {}
    for dirpath, _, files in os.walk(a.source):
        for name in files:
            if name.lower() in wanted:
                found.setdefault(name.lower(), []).append(os.path.join(dirpath, name))

    copied, missing, mismatched = 0, [], []
    for r in rows:
        candidates = found.get(r["file"].lower(), [])
        match = next((p for p in candidates if sha256(p) == r["sha256"]), None)
        if match is None:
            (mismatched if candidates else missing).append(r["file"])
            continue
        os.makedirs(DEST[r["set"]], exist_ok=True)
        shutil.copy2(match, os.path.join(DEST[r["set"]], r["file"]))
        copied += 1

    print(f"Copied {copied} of {len(rows)} images.")
    if missing:
        print(f"Not found ({len(missing)}): {', '.join(sorted(missing))}")
    if mismatched:
        print(f"Checksum mismatch ({len(mismatched)}): {', '.join(sorted(mismatched))}")
    sys.exit(0 if copied == len(rows) else 1)


if __name__ == "__main__":
    main()
