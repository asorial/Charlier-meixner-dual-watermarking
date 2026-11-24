"""
Tabla LaTeX, figura PDF y cifras del Experimento 8 (propiedades de las matrices de momentos).

Uso (desde la raíz del repositorio, después de run_exp8.py):
    python experiments/exp8_moment_properties/plot_exp8.py
Salida:
    results/exp8_properties_table.tex
    figures/exp8_localization.pdf
    results/exp8_numbers.json
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6})
MUTED = "#52514e"
ORDER = ["C I", "C II", "M I", "M II", "DCT"]
TEX = {"C I": r"C\,I", "C II": r"C\,II", "M I": r"M\,I", "M II": r"M\,II", "DCT": "DCT"}

p = pd.read_csv(os.path.join(RES, "exp8_matrix_properties.csv")).set_index("config")
rec = pd.read_csv(os.path.join(RES, "exp8_reconstruction.csv"))
rb = pd.read_csv(os.path.join(RES, "exp8_readback.csv"))
co = pd.read_csv(os.path.join(RES, "exp8_coefficients.csv")).set_index("config")
bas = np.load(os.path.join(RES, "exp8_basis.npz"))
e23 = pd.read_csv(os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity", "results", "e23_summary.csv"))
wide = pd.read_csv(os.path.join(ROOT, "experiments", "exp2_parameter_sensitivity", "results", "e22wide_summary.csv"))


def sci(v, nd=1):
    if v == 0 or not np.isfinite(v):
        return "0" if v == 0 else "--"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    if -2 <= e <= 1:
        return f"{v:.{max(0, nd + 1 - e) if e < 0 else nd}f}"
    return f"${m:.{nd}f}\\times10^{{{e}}}$"


recm = rec.groupby("config").agg(max_err=("max_abs_err", "max"), psnr_med=("psnr_rounded", "median"),
                                 changed_med=("changed_after_rounding", "median"))
rbm = rb.groupby("config").agg(dev_t=("dev_transform_max", "max"), dev_r=("dev_rounded_max_unclipped", "max"),
                               ber_unclip=("ber_unclipped_blocks", "max"), ber0=("ber_no_attack", "median"))

# ------------------------------------------------------------------ table
L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Properties of the $8\times8$ moment matrices used in the scheme and of the orthonormal "
     r"DCT. $\varepsilon_{\perp}=\max|\mathcal{A}^{T}\mathcal{A}-\mathcal{I}|$: orthogonality defect "
     r"caused by truncating the polynomials to $x=0,\dots,7$; $\kappa(\mathcal{A})$: condition number; "
     r"$\delta_{\mathcal{I}}=\max|\mathcal{A}-\mathcal{I}|$; float64: largest difference between the matrix "
     r"evaluated with the three-term recurrence in double precision and with 60 significant digits. "
     r"Localization of coefficient $(7,6)$: weight of pixel $(7,6)$, fraction of the energy of the "
     r"basis function at that pixel and effective number of pixels $(\sum\psi^2)^2/\sum\psi^4$. "
     r"Round trip $\mathcal{A}(\mathcal{A}^{T}\mathcal{C}\mathcal{A})\mathcal{A}^{T}$ without embedding, blue "
     r"channel of the 24 test images: largest error before rounding and median PSNR (dB) after rounding. "
     r"Read-back: largest deviation of $\nu^{k}_{62}$ from its quantized value after the inverse and "
     r"forward transforms, before rounding (the decision margin is $\Delta/4=37.5$). Rank: position of "
     r"coefficient 62 among the 64 coefficients by mean BER under Gaussian noise (0.10) and "
     r"salt-and-pepper noise (0.05).}",
     r"\label{tab:moment_properties}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{lcccccccccc}", r"\toprule",
     r" & \multicolumn{4}{c}{Numerical properties} & \multicolumn{3}{c}{Localization of $(7,6)$} & "
     r"\multicolumn{2}{c}{Round trip} & Coefficient \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-11}",
     r" & $\varepsilon_{\perp}$ & $\kappa(\mathcal{A})$ & $\delta_{\mathcal{I}}$ & float64 & Weight & Energy & "
     r"Support & Max.\ err. & PSNR & Rank \\", r"\midrule"]
for c in ORDER:
    r = p.loc[c]
    f64 = "--" if c == "DCT" else sci(r.f64_vs_mp_max)
    dI = "--" if c == "DCT" else f"{r.dist_identity_max:.3f}"
    psnr = r"$\infty$" if recm.loc[c, "max_err"] < 1e-9 else f"{recm.loc[c, 'psnr_med']:.1f}"
    rank = f"{int(co.loc[c, 'rank_62'])}/{int(co.loc[c, 'n_admissible'])}"
    L.append(" & ".join([TEX[c], sci(r.orth_err_max), f"{r.cond:.4f}", dI, f64, f"{r.psi_peak_weight:.3f}",
                         f"{100 * r.psi_energy_at_76:.1f}\\%", f"{r.psi_participation:.2f}",
                         f"{recm.loc[c, 'max_err']:.2f}" if c != "DCT" else sci(recm.loc[c, 'max_err']),
                         psnr, rank]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(RES, "exp8_properties_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.95), gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
for ax, (key, title) in zip(axes[:3], [("C_I", r"(a) C$\,$I, $\psi_{7,6}$"), ("M_I", r"(b) M$\,$I, $\psi_{7,6}$"),
                                       ("DCT", r"(c) DCT, $\psi_{7,6}$")]):
    psi = bas[key]; vmax = np.abs(psi).max()
    im = ax.imshow(psi, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="none")
    ax.add_patch(Rectangle((6 - .5, 7 - .5), 1, 1, fill=False, ec="#2b2a27", lw=0.9))
    ax.set_title(title, loc="left"); ax.set_xticks([0, 7]); ax.set_yticks([0, 7])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.4); cb.locator = matplotlib.ticker.MaxNLocator(3); cb.update_ticks()
ax = axes[3]
d = e23[e23.config == "C I"]
M = np.full((8, 8), np.nan)
for _, r in d.iterrows():
    M[int(r.row), int(r.col)] = (r["ber_gaussian_0.1"] + r["ber_salt_pepper_0.05"]) / 2
im = ax.imshow(M, cmap="Blues", interpolation="none", vmin=0)
ax.add_patch(Rectangle((6 - .5, 7 - .5), 1, 1, fill=False, ec="#e34948", lw=1.1))
ax.set_title("(d) C$\\,$I, BER by position", loc="left"); ax.set_xticks([0, 7]); ax.set_yticks([0, 7])
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_linewidth(0.4); cb.locator = matplotlib.ticker.MaxNLocator(3); cb.update_ticks()
for a in axes:
    a.set_xlabel("$y$", fontsize=8, labelpad=1)
axes[0].set_ylabel("$x$", fontsize=8, labelpad=1)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(FIG, "exp8_localization.pdf"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ numbers for the text
w = wide.set_index(["config", "param", "value"])
num = dict(
    orth={c: float(p.loc[c, "orth_err_max"]) for c in ORDER},
    cond={c: float(p.loc[c, "cond"]) for c in ORDER},
    dist_identity={c: float(p.loc[c, "dist_identity_max"]) for c in ORDER[:4]},
    f64_err={c: float(p.loc[c, "f64_vs_mp_max"]) for c in ORDER[:4]},
    stored_vs_exact={c: float(p.loc[c, "stored_vs_mp_max"]) for c in ORDER[:4]},
    weight76={c: float(p.loc[c, "psi_peak_weight"]) for c in ORDER},
    energy76={c: float(p.loc[c, "psi_energy_at_76"]) for c in ORDER},
    support76={c: float(p.loc[c, "psi_participation"]) for c in ORDER},
    roundtrip_max_err={c: float(recm.loc[c, "max_err"]) for c in ORDER},
    roundtrip_psnr_median={c: float(recm.loc[c, "psnr_med"]) for c in ORDER},
    roundtrip_changed_median={c: float(recm.loc[c, "changed_med"]) for c in ORDER},
    readback_dev_transform={c: float(rbm.loc[c, "dev_t"]) for c in ORDER},
    readback_dev_rounded={c: float(rbm.loc[c, "dev_r"]) for c in ORDER},
    ber_unclipped_blocks={c: float(rbm.loc[c, "ber_unclip"]) for c in ORDER},
    coef_rank=co["rank_62"].to_dict(), coef_score=co[["score_62", "score_best", "score_median", "score_worst"]].to_dict(),
    wide_charlier={str(v): dict(psnr=float(r.PSNR), ber_sp=float(r["BER SP"]), orth=float(r.orth_err))
                   for (cfg, par, v), r in w.iterrows() if cfg == "Charlier"},
    ber0_by_position_CI={f"({int(r.row)},{int(r.col)})": float(r.ber_none_0) for _, r in d.iterrows()
                         if (int(r.row), int(r.col)) in [(7, 6), (0, 6), (3, 3), (7, 0), (6, 7)]},
)
json.dump(num, open(os.path.join(RES, "exp8_numbers.json"), "w"), indent=1)
print(open(os.path.join(RES, "exp8_properties_table.tex")).read())
print(json.dumps({k: num[k] for k in ["coef_rank", "ber0_by_position_CI", "wide_charlier"]}, indent=1))
