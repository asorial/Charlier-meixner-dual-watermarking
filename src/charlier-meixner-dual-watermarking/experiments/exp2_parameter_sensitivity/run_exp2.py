"""
Experimento 2 - Análisis de sensibilidad de parámetros (respuesta a R2.4, R3.3, R1.E6).

  E2.1  Paso de cuantización Delta en {75,100,125,150,175,200}; C I, C II, M I, M II (matrices del
        artículo); coeficiente 62. Métricas: PSNR, SSIM, BER, NCC.
  E2.1safe  Igual que E2.1 con la variante de inserción que evita el desbordamiento de [0,255]
        (FastScheme(overflow_safe=True)); sirve para separar el efecto de Delta del recorte.
  gate  Coef. 62, Delta en {75,...,200}: C I, C II, M I, M II y DCT, con y sin la variante anterior
        (decisión sobre la novedad: ¿qué aportan los momentos frente a la DCT?).
  E2.2  Parámetros de los polinomios (Delta=150, coeficiente 62), matrices recalculadas con mpmath:
        a) variaciones pequeñas x{0.5,0.75,0.9,1,1.1,1.25,1.5,2} de a (Charlier) y de mu, gamma
           (Meixner), un parámetro cada vez;
        b) barrido amplio de a (Charlier) y de mu, gamma (Meixner), con la DCT como referencia.
        Métricas: PSNR, SSIM, BER, NCC y error de ortogonalidad ||A^T A - I||_F.
  E2.3  Coeficiente de inserción k = 0..63 (orden zigzag), Delta=150; C I, C II, M I, M II y, como
        referencia y DCT. Métricas: PSNR, BER, NCC.

Escenarios de ataque (funciones del repositorio, semilla fija por imagen y escenario):
  sin ataque; gaussiano (sigma = p * std(imagen)), p = 0.05, 0.10, 0.20; sal y pimienta,
  densidad 0.01, 0.05, 0.10; JPEG calidad 90, 70, 50 (E2.3: sin ataque, gaussiano 0.10,
  sal y pimienta 0.05, JPEG 90 y 70).

Uso (desde la raíz del repositorio):
    python experiments/exp2_parameter_sensitivity/run_exp2.py [e21|e21safe|gate|e22|e22wide|e23|all] [--workers 2]
"""
import argparse, csv, os, sys, time, platform
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastwm as F
from matrices import charlier_matrix, meixner_matrix
from helpers.orthonornal_matrix import orthmatrix
from helpers.utils import binarize_watermark, arnold_transform
import imageio.v3 as iio

RES = os.path.join(HERE, "results")
COVERS_DIR = os.path.join(F.ROOT, "src", "static", "cover_images")
WM_PATH = os.path.join(F.ROOT, "src", "static", "watermark", "qr_asl.png")

PAPER = {"C I": orthmatrix(0), "C II": orthmatrix(1), "M I": orthmatrix(2), "M II": orthmatrix(3)}
# parámetros del artículo (Meixner: mu -> g, gamma -> m en la parametrización del código)
PARAMS = {"C I": {"a": 7e-4}, "C II": {"a": 5e-4},
          "M I": {"mu": 8e-4, "gamma": 4.1e-5}, "M II": {"mu": 1e-4, "gamma": 7.5e-5}}


def dct_matrix(N=8):
    x, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.cos(np.pi * (2 * x + 1) * n / (2 * N)) * np.sqrt(2 / N)
    A[:, 0] /= np.sqrt(2)
    return A


REFS = {"DCT": dct_matrix()}

COND_FULL = [("none", 0), ("gaussian", 0.05), ("gaussian", 0.10), ("gaussian", 0.20),
             ("salt_pepper", 0.01), ("salt_pepper", 0.05), ("salt_pepper", 0.10),
             ("jpeg", 90), ("jpeg", 70), ("jpeg", 50)]
COND_E23 = [("none", 0), ("gaussian", 0.10), ("salt_pepper", 0.05), ("jpeg", 90), ("jpeg", 70)]
MULT = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]

_G = {}


def _init():
    names = sorted(f for f in os.listdir(COVERS_DIR) if f.lower().endswith(".bmp"))
    _G["names"] = names
    _G["covers"] = [iio.imread(os.path.join(COVERS_DIR, n)) for n in names]
    wm = iio.imread(WM_PATH)
    _G["wm"] = wm
    _G["bits"] = np.array([int(b) for b in binarize_watermark(arnold_transform(wm, 27).reshape(-1))])


def evaluate(task):
    meta, A, delta, coef, conds, do_ssim = task
    rows = []
    fs = F.FastScheme(A, delta, coef, overflow_safe=meta.get("overflow_safe", False))
    for ii, cover in enumerate(_G["covers"]):
        wm_img = fs.insert(cover, _G["bits"])
        base = dict(meta, image=_G["names"][ii], delta=delta, coef=coef,
                    psnr=F.psnr(cover, wm_img), orth_err=float(np.linalg.norm(A.T @ A - np.eye(8))))
        if do_ssim:
            base["ssim"] = F.ssim_standard(cover, wm_img)
            base["ssim_paper"] = F.ssim_paper(cover, wm_img)
        for ci, (kind, lev) in enumerate(conds):
            att = F.attack(wm_img, kind, lev, seed=1000 * ii + ci)
            ber, ncc = F.ber_ncc(_G["wm"], F.recover_watermark(fs.extract_bits(att)))
            rows.append(dict(base, attack=kind, level=lev, ber=ber, ncc=ncc))
    return rows


def tasks_e21():
    return [(dict(exp="E2.1", config=c), A, d, 62, COND_FULL, True)
            for c, A in PAPER.items() for d in [75, 100, 125, 150, 175, 200]]


def tasks_e21safe():
    """E2.1 con inserción a prueba de desbordamiento (variante propuesta, no es el método del artículo)."""
    return [(dict(exp="E2.1safe", config=c, overflow_safe=True), A, d, 62, COND_FULL, True)
            for c, A in PAPER.items() for d in [75, 100, 125, 150, 175, 200]]


def tasks_gate():
    """Comparación directa (coef. 62): matrices del artículo y DCT, para todo el barrido de
    Delta, con la inserción original y con la variante que evita el desbordamiento. Permite comparar
    a igual PSNR (curvas BER-PSNR)."""
    mats = dict(PAPER, **REFS)
    return [(dict(exp="gate", config=c, overflow_safe=safe), A, d, 62, COND_FULL, True)
            for c, A in mats.items() for safe in (False, True) for d in [75, 100, 125, 150, 175, 200]]


def tasks_e22():
    T = []
    for c in ["C I", "C II"]:
        for m in MULT:
            a = PARAMS[c]["a"] * m
            T.append((dict(exp="E2.2", config=c, param="a", mult=m, value=a),
                      charlier_matrix(a), 150, 62, COND_FULL, True))
    for c in ["M I", "M II"]:
        p = PARAMS[c]
        for name in ["mu", "gamma"]:
            for m in MULT:
                mu = p["mu"] * (m if name == "mu" else 1)
                ga = p["gamma"] * (m if name == "gamma" else 1)
                T.append((dict(exp="E2.2", config=c, param=name, mult=m,
                               value=mu if name == "mu" else ga),
                          meixner_matrix(ga, mu), 150, 62, COND_FULL, True))
    return T


def tasks_e22wide():
    T = []
    for a in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]:
        T.append((dict(exp="E2.2wide", config="Charlier", param="a", value=a),
                  charlier_matrix(a), 150, 62, COND_FULL, True))
    p = PARAMS["M I"]
    for ga in [1e-5, 4.1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.9]:
        T.append((dict(exp="E2.2wide", config="Meixner", param="gamma", value=ga),
                  meixner_matrix(ga, p["mu"]), 150, 62, COND_FULL, True))
    for mu in [1e-4, 8e-4, 1e-2, 0.1, 1.0, 10.0]:
        T.append((dict(exp="E2.2wide", config="Meixner", param="mu", value=mu),
                  meixner_matrix(p["gamma"], mu), 150, 62, COND_FULL, True))
    for r, A in REFS.items():
        T.append((dict(exp="E2.2wide", config=r, param="ref", value=np.nan), A, 150, 62,
                  COND_FULL, True))
    return T


def tasks_e23():
    mats = dict(PAPER, **REFS)
    return [(dict(exp="E2.3", config=c), A, 150, k, COND_E23, False)
            for c, A in mats.items() for k in range(64)]


def run(name, tasks, workers):
    os.makedirs(RES, exist_ok=True)
    t0, rows = time.time(), []
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(evaluate, tasks), 1):
            rows += r
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                print(f"[{name}] {i}/{len(tasks)}  {time.time() - t0:7.1f}s", flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(RES, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RES, "exp2_environment.txt"), "a") as f:
        f.write(f"{name}: {len(rows)} rows, {time.time() - t0:.0f} s | python "
                f"{platform.python_version()} | numpy {np.__version__} | {platform.platform()}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    todo = {"e21": tasks_e21, "e21safe": tasks_e21safe, "gate": tasks_gate, "e22": tasks_e22,
            "e22wide": tasks_e22wide, "e23": tasks_e23}
    for k in (todo if a.which == "all" else [a.which]):
        run(k, todo[k](), a.workers)
