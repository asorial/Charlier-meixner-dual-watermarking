"""
Experimento 9 - (A) Ablación de la transformada y (B) interacción entre las dos capas del esquema dual.

(A) Seis variantes del mismo algoritmo, que solo difieren en la matriz A del bloque 8x8:
    C I, C II, M I, M II (matrices del artículo), DCT ortonormal e identidad (inserción directa
    en el píxel (7,6) del bloque). Las seis matrices son ortonormales (los momentos, salvo el
    defecto de truncamiento <= 5.6e-3), así que un cambio Delta en el coeficiente produce un cambio
    de energía Delta^2 en el bloque en todas ellas: la normalización es comparable.
    Comunes: imágenes, marca (QR 64x64, Arnold 27), canal azul, bloques 8x8, coeficiente 62,
    Delta = 150, capa frágil HMAC y ataques, con la misma realización aleatoria por imagen y ataque.
(B) Para cada configuración de momentos: solo capa robusta, solo capa frágil, esquema dual
    completo y, como control, orden invertido (frágil y después robusta).
    Robustez: BER/NCC tras los mismos ataques (solo robusta frente a dual).
    Autenticación: imágenes intactas y manipulaciones con posición y tamaño conocidos
    (copy-move, splicing, eliminación; 1, 5, 10, 20 % del área; 2 posiciones por imagen y tamaño,
    las mismas para todas las variantes).
Tiempos (aparte, secuencial, sin procesos en paralelo): inserción y extracción de cada variante.

Uso (desde la raíz del repositorio):
    python experiments/exp9_ablation_layers/run_exp9.py transforms --covers DIR
    python experiments/exp9_ablation_layers/run_exp9.py layers --covers DIR
    python experiments/exp9_ablation_layers/run_exp9.py timing --covers DIR
"""
import argparse, csv, json, os, sys, time, platform, contextlib, io
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in [os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity"),
          os.path.join(ROOT, "experiments", "exp3_ablation_dct"), os.path.join(ROOT, "experiments", "exp4_complexity"),
          os.path.join(ROOT, "experiments", "exp1_tamper_localization")]:
    sys.path.insert(0, p)
import fastwm as F
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import binarize_watermark, arnold_transform
import imageio.v3 as iio
import run_exp3 as E3            # ataques (mismas definiciones y semillas que el Experimento 3)
import run_exp1 as E1            # manipulaciones (copy-move, splicing, eliminación)
import run_exp4 as E4            # verificación vectorizada e implementación de referencia

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
DELTA, COEF = 150, 62
TRANSFORMS = {"C I": np.array(orthmatrix(0), float), "C II": np.array(orthmatrix(1), float),
              "M I": np.array(orthmatrix(2), float), "M II": np.array(orthmatrix(3), float),
              "DCT": E3.dct_matrix(), "Identity": np.eye(8)}
MOMENTS = ["C I", "C II", "M I", "M II"]
ATTACKS = E3.ATTACKS
TAMPERS = ["copy-move", "splicing", "removal"]
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
REPS = 2
SEED = 20261124
_G = {}


class Scheme(F.FastScheme):
    """FastScheme con las dos capas separables."""

    def robust_only(self, cover, bits):
        self._write_tags = lambda out: None                      # se omite la capa frágil
        try:
            return F.FastScheme.insert(self, cover, bits)
        finally:
            del self._write_tags

    def fragile_only(self, img):
        out = img.copy(); self._write_tags(out); return out

    def dual(self, cover, bits):
        return F.FastScheme.insert(self, cover, bits)

    def reversed_order(self, cover, bits):
        return self.robust_only(self.fragile_only(cover), bits)

    def verify(self, img):
        return E4.vec_verify(self, img)


def _init(covers_dir, wm_path):
    names = sorted(f for f in os.listdir(covers_dir) if f.lower().endswith((".bmp", ".png", ".tif", ".tiff")))
    _G.update(names=names, dir=covers_dir, wm=iio.imread(wm_path))
    _G["bits"] = np.array([int(b) for b in binarize_watermark(arnold_transform(_G["wm"], 27).reshape(-1))])


def load(ii):
    return iio.imread(os.path.join(_G["dir"], _G["names"][ii]))[..., :3]


def robust_rows(base, fs, img, ii):
    rows = []
    for ai, (kind, lev) in enumerate(ATTACKS):
        att = E3.attack(img, kind, lev, seed=10_000 * ii + ai)          # misma realización para todas
        bits = fs.extract_bits(att)
        ber, ncc = F.ber_ncc(_G["wm"], F.recover_watermark(bits))
        rows.append(dict(base, attack=kind, level=lev, ber=ber, ncc=ncc,
                         raw_bit_errors=int((bits != _G["bits"]).sum())))
    return rows


# ------------------------------------------------------------------ (A) transformadas
def eval_transforms(ii):
    name = _G["names"][ii]; cover = load(ii); rows = []
    for var, A in TRANSFORMS.items():
        fs = Scheme(A, DELTA, COEF)
        w = fs.dual(cover, _G["bits"])
        base = dict(image=name, variant=var, psnr=F.psnr(cover, w), ssim=F.ssim_standard(cover, w),
                    flagged_intact=int(fs.verify(w).sum()))
        rows += robust_rows(base, fs, w, ii)
    return rows


# ------------------------------------------------------------------ (B) capas
def eval_layers(ii):
    name = _G["names"][ii]; cover = load(ii); donor = load((ii + 1) % len(_G["names"]))
    rob, auth, qual = [], [], []
    fs0 = Scheme(TRANSFORMS["C I"], DELTA, COEF)
    frag = fs0.fragile_only(cover)                       # no depende de la matriz
    images = {("all", "fragile only"): (frag, fs0)}
    for cfg in MOMENTS:
        fs = Scheme(TRANSFORMS[cfg], DELTA, COEF)
        images[(cfg, "robust only")] = (fs.robust_only(cover, _G["bits"]), fs)
        images[(cfg, "dual")] = (fs.dual(cover, _G["bits"]), fs)
        images[(cfg, "reversed order")] = (fs.reversed_order(cover, _G["bits"]), fs)
    for (cfg, var), (img, fs) in images.items():
        q = dict(image=name, config=cfg, variant=var, psnr=F.psnr(cover, img), ssim=F.ssim_standard(cover, img),
                 changed_pixels=float((img != cover).any(axis=2).mean()))
        if var != "robust only":
            q["flagged_intact"] = int(fs.verify(img).sum())
        if var != "fragile only":
            q["bit_errors_no_attack"] = int((fs.extract_bits(img) != _G["bits"]).sum())
        qual.append(q)
        if var in ("robust only", "dual"):
            rob += robust_rows(dict(image=name, config=cfg, variant=var), fs, img, ii)
        if var in ("fragile only", "dual"):
            for ti, t in enumerate(TAMPERS):
                for fi, fr in enumerate(FRACTIONS):
                    for r in range(REPS):
                        rng = np.random.default_rng([SEED, ii, ti, fi, r])   # misma región en todas las variantes
                        tam, region = E1.tamper(img, t, fr, rng, donor)
                        gt = (tam != img).any(axis=2).reshape(64, 8, 64, 8).any(axis=(1, 3))
                        det = fs.verify(tam)
                        m = E1.metrics(det, gt, region)
                        auth.append(dict(image=name, config=cfg, variant=var, tamper=t, fraction=fr, rep=r,
                                         gt_blocks=int(gt.sum()), detected_image=bool(det.any()),
                                         **{k: m[k] for k in ["TP", "FP", "FN", "TN", "precision", "recall", "F1"]}))
    return rob, auth, qual


def write(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))      # unión de columnas (vacío = no aplicable)
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)


def run_pool(fn, covers, wm, workers):
    n = len(_G["names"]); out = []; t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(covers, wm)) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, range(n)), 1):
            out.append(r)
            if i % 5 == 0 or i == n:
                print(f"[{time.time() - t0:7.1f}s] {i}/{n}", flush=True)
    return out, time.time() - t0


# ------------------------------------------------------------------ tiempos
def timing(reps, n_ref):
    import gc
    from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
    rows = []

    def tm(f, *a):
        gc.disable(); t = time.perf_counter_ns(); r = f(*a); dt = time.perf_counter_ns() - t; gc.enable()
        return dt * 1e-6, r
    names = _G["names"]
    for ii, name in enumerate(names):
        cover = load(ii)
        for var, A in TRANSFORMS.items():
            fs = Scheme(A, DELTA, COEF)
            ops = [("vectorized", "insert dual", fs.dual, (cover, _G["bits"])),
                   ("vectorized", "insert robust only", fs.robust_only, (cover, _G["bits"])),
                   ("vectorized", "insert fragile only", fs.fragile_only, (cover,))]
            w = fs.dual(cover, _G["bits"])
            ops += [("vectorized", "extract robust", fs.extract_bits, (w,)),
                    ("vectorized", "verify fragile", fs.verify, (w,))]
            if ii < n_ref:
                s = DualWatermarkingScheme(E4.KEY, DELTA, COEF, E4.P, E4.X0, A, A)
                bits_str = "".join(map(str, _G["bits"]))
                ops += [("reference", "insert dual", E4.ref_insert, (s, cover, bits_str)),
                        ("reference", "extract robust", E4.ref_extract, (s, w)),
                        ("reference", "verify fragile", E4.ref_verify, (s, w))]
            for impl, op, f, a in ops:
                f(*a)                                                   # calentamiento
                for r in range(reps):
                    dt, _ = tm(f, *a)
                    rows.append(dict(image=name, variant=var, impl=impl, op=op, rep=r, ms=dt))
        print(f"  timing {ii + 1}/{len(names)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["transforms", "layers", "timing"])
    ap.add_argument("--covers", default=os.path.join(ROOT, "data", "covers100"))
    ap.add_argument("--watermark", default=os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--images", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ref-images", type=int, default=24)
    a = ap.parse_args()
    _init(a.covers, a.watermark)
    if a.images:
        _G["names"] = _G["names"][:a.images]
    info = dict(part=a.part, images=len(_G["names"]), covers=a.covers, delta=DELTA, coef=COEF,
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                singular_values={k: [float(v.min()), float(v.max())] for k, v in
                                 ((k, np.linalg.svd(A, compute_uv=False)) for k, A in TRANSFORMS.items())})
    if a.part == "transforms":
        out, dt = run_pool(eval_transforms, a.covers, a.watermark, a.workers)
        write("exp9_transforms.csv", [r for rows in out for r in rows])
    elif a.part == "layers":
        out, dt = run_pool(eval_layers, a.covers, a.watermark, a.workers)
        write("exp9_layers_robust.csv", [r for o in out for r in o[0]])
        write("exp9_layers_auth.csv", [r for o in out for r in o[1]])
        write("exp9_layers_quality.csv", [r for o in out for r in o[2]])
    else:
        t0 = time.time(); rows = timing(a.reps, a.ref_images); dt = time.time() - t0
        info.update(E4.system_info())
        write("exp9_timing.csv", rows)
    info["seconds"] = round(dt, 1)
    json.dump(info, open(os.path.join(RES, f"exp9_{a.part}_info.json"), "w"), indent=2)
    print(f"done {a.part} in {dt:.0f} s")


if __name__ == "__main__":
    main()
