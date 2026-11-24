"""
Experimento 8 - Propiedades de las matrices de momentos usadas en el esquema
(estabilidad numérica, ortogonalidad, cercanía a la identidad, localización y precisión de reconstrucción).

Uso (desde la raíz del repositorio):
    python experiments/exp8_moment_properties/run_exp8.py
Salida: results/exp8_matrix_properties.csv, results/exp8_reconstruction.csv, results/exp8_basis.npz,
        results/exp8_coefficients.csv, results/exp8_numbers.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import imageio.v3 as iio
from mpmath import mp, mpf, exp, sqrt, rf, factorial

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity")]
from helpers.orthonornal_matrix import orthmatrix
from matrices import charlier_matrix, meixner_matrix
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
N = 8


def dct_matrix(N=8):
    x, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    A = np.cos(np.pi * (2 * x + 1) * n / (2 * N)) * np.sqrt(2 / N)
    A[:, 0] /= np.sqrt(2)
    return A


# ------------------------------------------------ float64 evaluation by the three-term recurrence
def charlier_f64(a, N=8):
    A = np.zeros((N, N))
    for x in range(N):
        P = [1.0, x - a]
        for n in range(1, N - 1):
            P.append((x - (n + a)) * P[n] - n * a * P[n - 1])
        w = np.exp(-a) * a ** x / np.math.factorial(x) if hasattr(np, "math") else None
        from math import factorial as f
        w = np.exp(-a) * a ** x / f(x)
        for n in range(N):
            A[x, n] = P[n] * np.sqrt(w / (f(n) * a ** n))
    return A


def meixner_f64(m, g, N=8):
    from math import factorial as f, gamma as G
    poch = lambda q, k: np.prod([q + i for i in range(k)]) if k > 0 else 1.0
    A = np.zeros((N, N))
    b = lambda n: (n * (1 + m) + m * g) / (1 - m)
    c = lambda n: n * m * (n - 1 + g) / (1 - m) ** 2
    for x in range(N):
        P = [1.0, x - b(0)]
        for n in range(1, N - 1):
            P.append((x - b(n)) * P[n] - c(n) * P[n - 1])
        w = m ** x * poch(g, x) / f(x)
        for n in range(N):
            d = f(n) * poch(g, n) * m ** n / (1 - m) ** (g + 2 * n)
            A[x, n] = P[n] * np.sqrt(w / d)
    return A


# parameters as used in the code (orthmatrix 0-3): Charlier a; Meixner (m, g) with m*g = mu*gamma of the paper
CFG = {"C I": ("charlier", (7e-4,)), "C II": ("charlier", (5e-4,)),
       "M I": ("meixner", (4.1e-5, 8e-4)), "M II": ("meixner", (7.5e-5, 1e-4))}
MAT = {"C I": np.array(orthmatrix(0), float), "C II": np.array(orthmatrix(1), float),
       "M I": np.array(orthmatrix(2), float), "M II": np.array(orthmatrix(3), float), "DCT": dct_matrix()}


def basis(A, u=7, v=6):
    """Pixel weights of coefficient (u, v) of M = A^T C A: psi[x, y] = A[x, u] A[y, v]."""
    return np.outer(A[:, u], A[:, v])


rows, bas = [], {}
for name, A in MAT.items():
    r = dict(config=name)
    r["orth_err_max"] = float(np.abs(A.T @ A - np.eye(N)).max())
    r["orth_err_fro"] = float(np.linalg.norm(A.T @ A - np.eye(N)))
    r["cond"] = float(np.linalg.cond(A))
    r["dist_identity_max"] = float(np.abs(A - np.eye(N)).max()) if name != "DCT" else np.nan
    r["dist_identity_fro"] = float(np.linalg.norm(A - np.eye(N))) if name != "DCT" else np.nan
    if name in CFG:
        fam, p = CFG[name]
        Amp = charlier_matrix(*p) if fam == "charlier" else meixner_matrix(*p)
        A64 = charlier_f64(*p) if fam == "charlier" else meixner_f64(*p)
        r["sqrt_param"] = float(np.sqrt(p[0] if fam == "charlier" else p[0] * p[1]))
        r["f64_vs_mp_max"] = float(np.abs(A64 - Amp).max())
        r["stored_vs_mp_max"] = float(np.abs(A - Amp).max())
        r["orth_err_mp"] = float(np.abs(Amp.T @ Amp - np.eye(N)).max())
    psi = basis(A)
    r["psi_peak_weight"] = float(psi[7, 6])
    r["psi_energy_at_76"] = float(psi[7, 6] ** 2 / (psi ** 2).sum())
    r["psi_participation"] = float((psi ** 2).sum() ** 2 / (psi ** 4).sum())   # effective number of pixels
    r["psi_l1_outside"] = float(np.abs(psi).sum() - abs(psi[7, 6]))
    bas[name] = psi
    rows.append(r)
props = pd.DataFrame(rows); props.to_csv(os.path.join(RES, "exp8_matrix_properties.csv"), index=False)
np.savez(os.path.join(RES, "exp8_basis.npz"), **{k.replace(" ", "_"): v for k, v in bas.items()})

# ------------------------------------------------ reconstruction accuracy on the 24 test images (blue channel)
cdir = os.path.join(ROOT, "src", "static", "cover_images")
names = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".bmp"))
rec = []
for name in names:
    ch = iio.imread(os.path.join(cdir, name))[..., 2].astype(float)
    H, W = ch.shape
    B = ch.reshape(H // N, N, W // N, N).transpose(0, 2, 1, 3)
    for cfg, A in MAT.items():
        R = A @ (A.T @ B @ A) @ A.T
        Rq = np.clip(np.rint(R), 0, 255)
        err = R - B
        mse_q = np.mean((Rq - B) ** 2)
        rec.append(dict(image=name, config=cfg, max_abs_err=float(np.abs(err).max()),
                        rmse=float(np.sqrt(np.mean(err ** 2))), changed_after_rounding=float((Rq != B).mean()),
                        psnr_rounded=float(99 if mse_q == 0 else 10 * np.log10(255 ** 2 / mse_q))))
rec = pd.DataFrame(rec); rec.to_csv(os.path.join(RES, "exp8_reconstruction.csv"), index=False)

# ------------------------------------------------ coefficient position (from Experiment 2.3)
e23 = pd.read_csv(os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity", "results", "e23_summary.csv"))
co = []
for cfg in ["C I", "C II", "M I", "M II", "DCT"]:
    d = e23[e23.config == cfg].set_index("coef")
    sc = (d["ber_gaussian_0.1"] + d["ber_salt_pepper_0.05"]) / 2
    ok = d.psnr_none_0 >= 39
    co.append(dict(config=cfg, n_admissible=int(ok.sum()), rank_62=int(sc[ok].rank(method="min")[62]),
                   score_62=float(sc[62]), score_best=float(sc[ok].min()), best_coef=int(sc[ok].idxmin()),
                   score_median=float(sc[ok].median()), score_worst=float(sc[ok].max()),
                   psnr_62=float(d.psnr_none_0[62]), ber0_62=float(d.ber_none_0[62])))
co = pd.DataFrame(co); co.to_csv(os.path.join(RES, "exp8_coefficients.csv"), index=False)

summ = rec.groupby("config")[["max_abs_err", "rmse", "changed_after_rounding", "psnr_rounded"]].agg(["mean", "max", "min"])
print(props.round(6).to_string())
print(summ.round(5).to_string())
print(co.round(4).to_string())

# ------------------------------------------------ read-back accuracy of the marked coefficient (no attack)
import fastwm as F
wm = iio.imread(os.path.join(ROOT, "src", "static", "watermark", "qr_asl.png"))
from helpers.utils import binarize_watermark, arnold_transform
bits = np.array([int(b) for b in binarize_watermark(arnold_transform(wm, 27).reshape(-1))])
rb = []
for name in names:
    cover = iio.imread(os.path.join(cdir, name))[..., :3]
    ch = cover[..., 2].astype(float)
    B = ch.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    for cfg in ["C I", "C II", "M I", "M II", "DCT"]:
        A = MAT[cfg]
        M = A.T @ B @ A
        nu = M[:, 7, 6]
        nub = F.dm_embed(nu, bits, 150.0)                      # quantized coefficient
        Mb = M.copy(); Mb[:, 7, 6] = nub
        R = A @ Mb @ A.T                                         # inverse, before rounding
        nu_f = (A.T @ R @ A)[:, 7, 6]                            # read back, exact arithmetic
        Rq = np.clip(np.rint(R), 0, 255)
        nu_q = (A.T @ Rq @ A)[:, 7, 6]                           # read back after rounding and clipping
        clipped = ((R < -0.5) | (R > 255.5)).reshape(len(R), -1).any(1)
        rb.append(dict(image=name, config=cfg,
                       dev_transform_max=float(np.abs(nu_f - nub).max()),
                       dev_transform_p99=float(np.percentile(np.abs(nu_f - nub), 99)),
                       dev_rounded_p99=float(np.percentile(np.abs(nu_q - nub)[~clipped], 99)) if (~clipped).any() else np.nan,
                       dev_rounded_max_unclipped=float(np.abs(nu_q - nub)[~clipped].max()),
                       frac_blocks_clipped=float(clipped.mean()),
                       ber_no_attack=float((F.dm_extract(nu_q, 150.0) != bits).mean()),
                       ber_unclipped_blocks=float((F.dm_extract(nu_q, 150.0)[~clipped] != bits[~clipped]).mean())))
rb = pd.DataFrame(rb); rb.to_csv(os.path.join(RES, "exp8_readback.csv"), index=False)
print(rb.drop(columns="image").groupby("config").agg(["median", "max"]).round(4).to_string())
