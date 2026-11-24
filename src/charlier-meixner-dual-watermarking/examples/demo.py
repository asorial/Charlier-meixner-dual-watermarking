"""
Minimal example of the dual watermarking scheme: embed, extract, verify, and localize tampering.

    python examples/demo.py [--cover IMAGE] [--config "C I"] [--key SECRET]

Without --cover, the first image of src/static/cover_images is used; if that folder is empty,
a synthetic 512x512 RGB image is generated so the example runs without the dataset.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import imageio.v3 as iio

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme  # noqa: E402
from helpers.orthonornal_matrix import orthmatrix  # noqa: E402
from helpers.utils import arnold_transform, binarize_watermark, unbinarize_watermark  # noqa: E402

CONFIGS = {"C I": 0, "C II": 1, "M I": 2, "M II": 3}
DELTA, COEFFICIENT, ARNOLD_ITERS = 150, 62, 27   # Table 1 of the paper; inverse Arnold = 64 - 27


def quiet(f, *args, **kwargs):
    """Run f without its progress bar."""
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def load_cover(path):
    if path:
        return iio.imread(path)[:, :, :3]
    folder = os.path.join(ROOT, "src", "static", "cover_images")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".bmp", ".png"))) if os.path.isdir(folder) else []
    if names:
        print(f"Cover: {names[0]}")
        return iio.imread(os.path.join(folder, names[0]))[:, :, :3]
    print("Cover: synthetic image (run scripts/prepare_images.py to use the test images)")
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:512, 0:512]
    base = np.stack([120 + 60 * np.sin(x / 40), 110 + 50 * np.cos(y / 55), 100 + 40 * np.sin((x + y) / 70)], -1)
    return np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", help="512x512 RGB cover image")
    ap.add_argument("--config", default="C I", choices=list(CONFIGS))
    ap.add_argument("--key", default="example-key-change-me",
                    help="secret key K of the fragile layer (use a random key of at least 128 bits)")
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "output"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cover = load_cover(a.cover)
    logo = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))  # 64x64 binary
    bits = binarize_watermark(arnold_transform(logo, ARNOLD_ITERS).reshape(-1))

    A = orthmatrix(CONFIGS[a.config])
    scheme = DualWatermarkingScheme(a.key, DELTA, COEFFICIENT, 0.58, 0.24, A, A)

    # 1. Embedding (robust layer, then fragile tags)
    marked = quiet(scheme.insert, cover, bits)
    iio.imwrite(os.path.join(a.out, "watermarked.png"), marked)
    marked = iio.imread(os.path.join(a.out, "watermarked.png"))
    print(f"[{a.config}] PSNR = {psnr(cover, marked):.2f} dB")

    # 2. Extraction of the robust watermark and verification of the fragile layer
    raw = quiet(scheme.extraction, marked.copy(), "demo.png", save_tampered=False)
    recovered = arnold_transform(np.array(unbinarize_watermark([int(b) for b in raw])).reshape(64, 64),
                                 64 - ARNOLD_ITERS)
    print(f"Robust watermark: BER = {np.mean(recovered != logo):.4f}")
    print(f"Fragile layer, untouched image: {int(scheme.last_tamper_map.sum())} of 4096 blocks flagged")

    # 3. Tampering: paste a 64x64 patch from another area and localize it
    y0, x0, size = 200, 300, 64
    tampered = marked.copy()
    tampered[y0:y0 + size, x0:x0 + size] = marked[20:20 + size, 20:20 + size]
    tamper_map = scheme.authenticate(tampered)
    rows, cols = np.nonzero(tamper_map)
    print(f"Fragile layer, tampered image: {tamper_map.sum()} blocks flagged, "
          f"block rows {rows.min()}-{rows.max()}, columns {cols.min()}-{cols.max()} "
          f"(region covers rows {y0 // 8}-{(y0 + size - 1) // 8}, columns {x0 // 8}-{(x0 + size - 1) // 8})")
    iio.imwrite(os.path.join(a.out, "recovered_watermark.png"), (recovered * 255).astype(np.uint8)
                if recovered.max() <= 1 else recovered.astype(np.uint8))
    iio.imwrite(os.path.join(a.out, "tamper_map.png"), (tamper_map * 255).astype(np.uint8))
    print(f"Outputs written to {a.out}")


if __name__ == "__main__":
    main()
