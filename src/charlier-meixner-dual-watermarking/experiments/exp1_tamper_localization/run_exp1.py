"""
Experimento 1 - Detección y localización de manipulaciones (capa frágil HMAC).

Diseño
  * 24 imágenes de src/static/cover_images (512x512 RGB), configuraciones C I, C II, M I, M II.
  * Manipulaciones: copy-move (misma imagen), splicing (región de otra portadora sin marca)
    y eliminación de objeto (inpainting Telea de OpenCV).
  * Área manipulada: 1, 5, 10 y 20 % de la imagen (cuadrado de lado round(512*sqrt(f))),
    posición aleatoria (no alineada con la rejilla 8x8), 5 repeticiones.
  * Semillas: rng([SEED, imagen, ataque, tamaño, repetición]) -> la misma región para las
    cuatro configuraciones (comparación pareada).
  * Verdad de referencia por bloque: bloque 8x8 con al menos un píxel distinto (cualquier canal)
    respecto de la imagen marcada. A nivel de píxel: la región nominal manipulada.
  * Métricas por bloque: TP, FP, FN, TN, precisión, recall, F1, FPR, FNR, error de
    accuracy = (TP+TN)/N_bloques, error de localización = (FP+FN)/N_bloques.
    Por píxel (región nominal): precisión, recall, F1, IoU, accuracy.

Uso (desde la raíz del repositorio):
    python experiments/exp1_tamper_localization/run_exp1.py [--reps 5]
Salida: experiments/exp1_tamper_localization/results/exp1_trials.csv (+ ejemplos .npz)
"""
import argparse, csv, io, os, sys, time, contextlib, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dual_scheme.dual_watermarking_scheme import DualWatermarkingScheme
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import arnold_transform, binarize_watermark
import imageio.v3 as iio
import cv2

SEED = 20261101
KEY, DELTA, COEF, P, X0 = "N;fpr-y7", 150, 62, 0.58, 0.24
CONFIGS = {"C I": 0, "C II": 1, "M I": 2, "M II": 3}
ATTACKS = ["copy-move", "splicing", "removal"]
FRACTIONS = [0.01, 0.05, 0.10, 0.20]
EXAMPLE = {"image": "C0309.bmp", "config": "C I", "rep": 0}   # se guardan todos los ataques y áreas


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def square(rng, side, H, W, avoid=None):
    for _ in range(1000):
        y, x = int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))
        if avoid is None:
            return y, x
        ay, ax = avoid
        if abs(y - ay) >= side or abs(x - ax) >= side:      # sin solapamiento
            return y, x
    raise RuntimeError("no non-overlapping position")


def tamper(wm, attack, frac, rng, donor):
    H, W = wm.shape[:2]
    side = int(round(H * np.sqrt(frac)))
    if attack == "copy-move":                      # destino y origen sin solapamiento
        n = H if rng.random() < 0.5 else W        # eje de separación (H == W aquí)
        p1 = int(rng.integers(0, n - 2 * side + 1))
        p2 = int(rng.integers(p1 + side, n - side + 1))
        if rng.random() < 0.5:
            p1, p2 = p2, p1                        # orden destino/origen al azar
        q1, q2 = int(rng.integers(0, n - side + 1)), int(rng.integers(0, n - side + 1))
        if rng.random() < 0.5:                     # separación vertical
            (y, x), (sy, sx) = (p1, q1), (p2, q2)
        else:                                      # separación horizontal
            (y, x), (sy, sx) = (q1, p1), (q2, p2)
    else:
        y, x = square(rng, side, H, W)
    t = wm.copy()
    if attack == "copy-move":
        t[y:y + side, x:x + side] = wm[sy:sy + side, sx:sx + side]
    elif attack == "splicing":
        sy, sx = square(rng, side, H, W)
        t[y:y + side, x:x + side] = donor[sy:sy + side, sx:sx + side]
    elif attack == "removal":
        m = np.zeros((H, W), np.uint8); m[y:y + side, x:x + side] = 255
        bgr = cv2.cvtColor(wm, cv2.COLOR_RGB2BGR)
        t = cv2.cvtColor(cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    region = np.zeros((H, W), bool); region[y:y + side, x:x + side] = True
    return t, region


def metrics(det_blocks, gt_blocks, region_px):
    TP = int(np.sum(det_blocks & gt_blocks)); FP = int(np.sum(det_blocks & ~gt_blocks))
    FN = int(np.sum(~det_blocks & gt_blocks)); TN = int(np.sum(~det_blocks & ~gt_blocks))
    sd = lambda a, b: a / b if b else float("nan")
    prec, rec = sd(TP, TP + FP), sd(TP, TP + FN)
    f1 = sd(2 * prec * rec, prec + rec) if TP else 0.0
    det_px = np.kron(det_blocks, np.ones((8, 8), bool))
    tp = int(np.sum(det_px & region_px)); fp = int(np.sum(det_px & ~region_px))
    fn = int(np.sum(~det_px & region_px))
    pp, pr = sd(tp, tp + fp), sd(tp, tp + fn)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec, F1=f1,
                FPR=sd(FP, FP + TN), FNR=sd(FN, FN + TP),
                accuracy=(TP + TN) / det_blocks.size,
                loc_error=(FP + FN) / det_blocks.size,
                px_precision=pp, px_recall=pr,
                px_F1=sd(2 * pp * pr, pp + pr) if tp else 0.0,
                px_IoU=sd(tp, tp + fp + fn),
                px_accuracy=1 - (fp + fn) / region_px.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--images", type=int, default=0, help="0 = todas")
    args = ap.parse_args()

    covers_dir = os.path.join(ROOT, "src", "static", "cover_images")
    names = sorted(f for f in os.listdir(covers_dir) if f.lower().endswith(".bmp"))
    if args.images:
        names = names[:args.images]
    covers = {n: iio.imread(os.path.join(covers_dir, n)) for n in names}
    wmk = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
    bits = binarize_watermark(arnold_transform(wmk, 27).reshape(-1))

    out_dir = os.path.join(HERE, "results"); os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    timing = {"insert": [], "authenticate": []}
    for ci, (cname, mid) in enumerate(CONFIGS.items()):
        sch = DualWatermarkingScheme(KEY, DELTA, COEF, P, X0, orthmatrix(mid), orthmatrix(mid))
        for ii, name in enumerate(names):
            ts = time.perf_counter(); wm = quiet(sch.insert, covers[name], bits)
            timing["insert"].append(time.perf_counter() - ts)
            assert not sch.authenticate(wm).any(), f"falso positivo sin manipular: {name} {cname}"
            donor = covers[names[(ii + 1) % len(names)]]
            for ai, att in enumerate(ATTACKS):
                for fi, frac in enumerate(FRACTIONS):
                    for rep in range(args.reps):
                        rng = np.random.default_rng([SEED, ii, ai, fi, rep])
                        t, region = tamper(wm, att, frac, rng, donor)
                        changed = np.any(t != wm, axis=2)
                        gt = changed.reshape(64, 8, 64, 8).any(axis=(1, 3))
                        ts = time.perf_counter(); det = sch.authenticate(t)
                        timing["authenticate"].append(time.perf_counter() - ts)
                        m = metrics(det, gt, region)
                        rows.append(dict(image=name, config=cname, attack=att,
                                         fraction=frac, rep=rep, gt_blocks=int(gt.sum()), **m))
                        if (name == EXAMPLE["image"] and cname == EXAMPLE["config"]
                                and rep == EXAMPLE["rep"]):
                            ex_dir = os.path.join(out_dir, "examples"); os.makedirs(ex_dir, exist_ok=True)
                            if ai == 0 and fi == 0:
                                np.savez_compressed(os.path.join(ex_dir, "example_base.npz"),
                                                    original=covers[name], watermarked=wm)
                            np.savez_compressed(
                                os.path.join(ex_dir, f"example_{att}_{int(round(frac * 100))}.npz"),
                                tampered=t, region=region, changed=changed, gt_blocks=gt, detected=det)
            print(f"[{time.time() - t0:7.1f}s] {cname} {name} hecho", flush=True)

    fields = list(rows[0].keys())
    with open(os.path.join(out_dir, "exp1_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "exp1_environment.txt"), "w") as f:
        f.write(f"python {platform.python_version()} | numpy {np.__version__} | "
                f"opencv {cv2.__version__} | {platform.platform()} | {platform.processor()}\n")
        f.write(f"seed={SEED} reps={args.reps} images={len(names)} trials={len(rows)}\n")
        for k, v in timing.items():
            f.write(f"{k}: median {np.median(v):.4f} s, IQR "
                    f"{np.percentile(v, 25):.4f}-{np.percentile(v, 75):.4f} s (n={len(v)})\n")
    print("trials:", len(rows), "->", out_dir)


if __name__ == "__main__":
    main()
