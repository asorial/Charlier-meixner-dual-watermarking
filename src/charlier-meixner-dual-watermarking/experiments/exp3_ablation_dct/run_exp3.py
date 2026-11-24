"""
Experimento 3 - Ablación controlada: momentos de Charlier/Meixner frente a la DCT.

Única diferencia entre variantes: la matriz de transformación A del bloque 8x8.
Comunes a todas: canal azul, coeficiente 62 (zigzag), Dither Modulation con Delta = 150,
capa frágil HMAC idéntica, misma marca (QR 64x64, Arnold 27), mismas imágenes, mismos ataques
con la misma semilla por imagen y nivel (comparación pareada).

Variantes: C I, C II, M I, M II (matrices del artículo), y DCT (ortonormal exacta)
(referencia: modificación directa del píxel (7,6)).
Inserción: "original" (la del artículo) y "safe" (evita que el recorte a [0,255] invierta el bit;
véase el Experimento 2).

Ataques (niveles):
  gaussian     sigma = p*std(I), p = 0.02, 0.05, 0.10, 0.15, 0.20   (función del repositorio)
  salt_pepper  densidad 0.01, 0.02, 0.05, 0.10, 0.20                  (función del repositorio)
  median       mediana 2D por canal, ventana 3, 5, 7
  mean         media 2D por canal, ventana 3, 5, 7
  jpeg         calidad 100, 95, 90, 80, 70, 50 (Pillow)
  rotation     0.5, 1, 2, 5, 10, 45 grados, centro de la imagen, bilineal, fondo negro, sin
               resincronización (como rotation_noise del repositorio)

Uso (desde la raíz del repositorio):
    python experiments/exp3_ablation_dct/run_exp3.py [--covers DIR] [--workers 2]
"""
import argparse, csv, os, sys, time, platform
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp2_parameter_sensitivity"))
import fastwm as F
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import binarize_watermark, arnold_transform
from scipy import ndimage
import imageio.v3 as iio
import cv2


def dct_matrix(N=8):
    x, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.cos(np.pi * (2 * x + 1) * n / (2 * N)) * np.sqrt(2 / N)
    A[:, 0] /= np.sqrt(2)
    return A


VARIANTS = {"C I": orthmatrix(0), "C II": orthmatrix(1), "M I": orthmatrix(2), "M II": orthmatrix(3),
            "DCT": dct_matrix()}
DELTA, COEF = 150, 62
ATTACKS = [("none", 0)] + \
    [("gaussian", p) for p in [0.02, 0.05, 0.10, 0.15, 0.20]] + \
    [("salt_pepper", d) for d in [0.01, 0.02, 0.05, 0.10, 0.20]] + \
    [("median", k) for k in [3, 5, 7]] + \
    [("mean", k) for k in [3, 5, 7]] + \
    [("jpeg", q) for q in [100, 95, 90, 80, 70, 50]] + \
    [("rotation", a) for a in [0.5, 1, 2, 5, 10, 45]]

_G = {}


def attack(img, kind, lev, seed):
    if kind == "median":
        return ndimage.median_filter(img, size=(int(lev), int(lev), 1))
    if kind == "mean":
        return np.rint(ndimage.uniform_filter(img.astype(float), size=(int(lev), int(lev), 1))).astype(np.uint8)
    if kind == "rotation":
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), float(lev), 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return F.attack(img, kind, lev, seed)


def _init(covers_dir, wm_path):
    names = sorted(f for f in os.listdir(covers_dir) if f.lower().endswith((".bmp", ".png", ".tif", ".tiff")))
    _G["names"], _G["dir"] = names, covers_dir
    wm = iio.imread(wm_path)
    _G["wm"] = wm
    _G["bits"] = np.array([int(b) for b in binarize_watermark(arnold_transform(wm, 27).reshape(-1))])


def evaluate(ii):
    name = _G["names"][ii]
    cover = iio.imread(os.path.join(_G["dir"], name))[..., :3]
    rows = []
    for emb in ["original", "safe"]:
        for var, A in VARIANTS.items():
            fs = F.FastScheme(A, DELTA, COEF, overflow_safe=(emb == "safe"))
            wm_img = fs.insert(cover, _G["bits"])
            base = dict(image=name, dataset="Caltech" if name.upper().startswith("C") else "NRC",
                        variant=var, embedding=emb, psnr=F.psnr(cover, wm_img),
                        ssim=F.ssim_standard(cover, wm_img), ssim_paper=F.ssim_paper(cover, wm_img))
            for ai, (kind, lev) in enumerate(ATTACKS):
                att = attack(wm_img, kind, lev, seed=10_000 * ii + ai)
                ber, ncc = F.ber_ncc(_G["wm"], F.recover_watermark(fs.extract_bits(att)))
                rows.append(dict(base, attack=kind, level=lev, ber=ber, ncc=ncc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", default=os.path.join(F.ROOT, "data", "covers100"))
    ap.add_argument("--watermark", default=os.path.join(F.ROOT, "src", "static", "watermark", "qr_asl.png"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="exp3_trials.csv")
    a = ap.parse_args()
    res = os.path.join(HERE, "results"); os.makedirs(res, exist_ok=True)
    _init(a.covers, a.watermark)
    n = len(_G["names"]); t0 = time.time(); rows = []
    with Pool(a.workers, initializer=_init, initargs=(a.covers, a.watermark)) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate, range(n)), 1):
            rows += r
            print(f"[{time.time() - t0:7.1f}s] {i}/{n} imágenes", flush=True)
    with open(os.path.join(res, a.out), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(res, "exp3_environment.txt"), "w") as f:
        f.write(f"images={n} ({a.covers}) | variants={list(VARIANTS)} | Delta={DELTA} coef={COEF}\n"
                f"python {platform.python_version()} | numpy {np.__version__} | opencv {cv2.__version__}"
                f" | {platform.platform()} | {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
