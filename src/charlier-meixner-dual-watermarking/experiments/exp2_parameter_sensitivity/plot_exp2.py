"""
Tablas (LaTeX/CSV) y figuras (PDF) del Experimento 2.

Uso (desde la raíz del repositorio):
    python experiments/exp2_parameter_sensitivity/plot_exp2.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
})
INK, MUTED = "#0b0b0b", "#52514e"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # paleta validada
MARK = ["o", "s", "^", "D", "v", "P"]
SEQ = {1: "#9ec5f4", 2: "#4a90e2", 3: "#1c4f9c"}                               # rampa azul: intensidad
NONE_C = "#8a8984"
ATT_NAME = {"gaussian": "Gaussian noise", "salt_pepper": "Salt & pepper", "jpeg": "JPEG"}
LEVELS = {"gaussian": [0.05, 0.10, 0.20], "salt_pepper": [0.01, 0.05, 0.10], "jpeg": [90, 70, 50]}
LEVEL_LABEL = {"gaussian": lambda v: f"$p$ = {v:.2f}", "salt_pepper": lambda v: f"$d$ = {v:.2f}",
               "jpeg": lambda v: f"$Q$ = {int(v)}"}


def load(name):
    p = os.path.join(RES, f"{name}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d.config != "A = I"].copy() if "config" in d else d      # se excluye A = I


def cond(d, kind, lev=0):
    return d[(d.attack == kind) & np.isclose(d.level, lev)]


def bold_best(df, cols, lower):
    out = df.copy().astype(object)
    for c in cols:
        v = df[c].astype(float)
        best = v.min() if lower[c] else v.max()
        out[c] = [f"\\textbf{{{x:{FMT[c]}}}}" if np.isclose(x, best) else f"{x:{FMT[c]}}" for x in v]
    return out


FMT = {"PSNR": ".2f", "SSIM": ".4f", "BER none": ".4f", "BER G": ".4f", "BER SP": ".4f",
       "BER J90": ".4f", "NCC G": ".4f", "NCC SP": ".4f"}
LOWER = {"PSNR": False, "SSIM": False, "BER none": True, "BER G": True, "BER SP": True,
         "BER J90": True, "NCC G": False, "NCC SP": False}


def summary(d, keys):
    g = d.groupby(keys)
    s = pd.DataFrame({
        "PSNR": cond(d, "none").groupby(keys).psnr.mean(),
        "SSIM": cond(d, "none").groupby(keys).ssim.mean() if "ssim" in d else np.nan,
        "BER none": cond(d, "none").groupby(keys).ber.mean(),
        "BER G": cond(d, "gaussian", 0.10).groupby(keys).ber.mean(),
        "BER SP": cond(d, "salt_pepper", 0.05).groupby(keys).ber.mean(),
        "BER J90": cond(d, "jpeg", 90).groupby(keys).ber.mean(),
        "NCC G": cond(d, "gaussian", 0.10).groupby(keys).ncc.mean(),
        "NCC SP": cond(d, "salt_pepper", 0.05).groupby(keys).ncc.mean(),
    })
    return s


def latex_table(df, caption, label, first_cols):
    cols = list(df.columns)
    head = " & ".join(first_cols + [c.replace("BER G", "BER$_{G,0.10}$").replace("BER SP", "BER$_{SP,0.05}$")
                                      .replace("BER J90", "BER$_{Q90}$").replace("BER none", "BER$_{0}$")
                                      .replace("NCC G", "NCC$_{G,0.10}$").replace("NCC SP", "NCC$_{SP,0.05}$")
                                      .replace("PSNR", "PSNR (dB)")
                                      for c in cols]) + r" \\"
    L = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\begin{tabular}{" + "l" * len(first_cols) + "c" * len(cols) + "}", r"\toprule", head,
         r"\midrule"]
    return L


CAPTION_COMMON = (r"Means over 24 images; for $\Delta$ also over C\,I, C\,II, M\,I and M\,II. "
                  r"BER$_0$: no attack; G: Gaussian noise with $\sigma = 0.10\,\mathrm{std}(I)$; "
                  r"SP: salt-and-pepper density 0.05; Q90: JPEG quality 90. "
                  r"SSIM is the standard local SSIM (7$\times$7 windows). Best value per column in bold.")

# =============================================================================== E2.1 (Delta)
e21, e21s, gate = load("e21"), load("e21safe"), load("gate")
if e21s is None and gate is not None:
    e21s = gate[gate.overflow_safe & gate.config.isin(["C I", "C II", "M I", "M II"])].copy()
if e21 is not None:
    blocks = [("Original embedding", e21)] + ([("Overflow-safe embedding", e21s)] if e21s is not None else [])
    L = latex_table(None if False else summary(e21, ["delta"]),
                    r"Sensitivity to the quantization step $\Delta$ (coefficient 62). " + CAPTION_COMMON +
                    r" Overflow-safe: when rounding and clipping to $[0,255]$ would flip the embedded bit "
                    r"or leave it closer than $\Delta/8$ to the decision boundary, the nearest admissible "
                    r"point of the same dither sub-lattice ($\pm\Delta$) is used.",
                    "tab:delta_sensitivity", [r"$\Delta$"])
    allrows = []
    for name, d in blocks:
        s = summary(d, ["delta"]); s.to_csv(os.path.join(RES, f"e21_summary_{'safe' if 'safe' in name else 'orig'}.csv"))
        summary(d, ["config", "delta"]).to_csv(
            os.path.join(RES, f"e21_summary_by_config_{'safe' if 'safe' in name else 'orig'}.csv"))
        b = bold_best(s, s.columns, LOWER)
        L.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(s.columns) + 1, name))
        for dlt, r in b.iterrows():
            L.append(f"{int(dlt)} & " + " & ".join(r.values) + r" \\")
        L.append(r"\midrule" if name != blocks[-1][0] else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(RES, "exp2_delta_table.tex"), "w").write("\n".join(L) + "\n")

    # ---- figura
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    ax = axes[0, 0]
    for name, d, ls in [("Original", e21, "-")] + ([("Overflow-safe", e21s, "--")] if e21s is not None else []):
        m = cond(d, "none").groupby("delta").psnr.mean()
        ax.plot(m.index, m.values, ls, color=CAT[0], marker="o", ms=4, lw=1.6, label=f"{name} embedding")
    ax.axhline(40, color=MUTED, lw=0.6, ls=":")
    ax.text(201, 40.15, "40 dB", color=MUTED, fontsize=7, ha="right", va="bottom")
    ax.set_title("(a) Imperceptibility", loc="left"); ax.set_ylabel("PSNR (dB)")
    ax.legend(frameon=False, loc="upper right")
    for ax, kind, tag in [(axes[0, 1], "gaussian", "(b)"), (axes[1, 0], "salt_pepper", "(c)"),
                          (axes[1, 1], "jpeg", "(d)")]:
        for name, d, ls in [("orig", e21, "-")] + ([("safe", e21s, "--")] if e21s is not None else []):
            if kind != "jpeg":
                m = cond(d, "none").groupby("delta").ber.mean()
                ax.plot(m.index, m.values, ls, color=NONE_C, marker="o", ms=3, lw=1.2)
            for i, lev in enumerate(LEVELS[kind], 1):
                m = cond(d, kind, lev).groupby("delta").ber.mean()
                ax.plot(m.index, m.values, ls, color=SEQ[i], marker="o", ms=3.5, lw=1.5)
        handles = ([Line2D([], [], color=NONE_C, marker="o", ms=3, lw=1.2, label="No attack")]
                   if kind != "jpeg" else [])
        handles += [Line2D([], [], color=SEQ[i], marker="o", ms=3.5, lw=1.5, label=LEVEL_LABEL[kind](v))
                    for i, v in enumerate(LEVELS[kind], 1)]
        ax.legend(handles=handles, frameon=False, fontsize=7,
                  loc="lower right" if kind == "jpeg" else "upper right", ncol=1)
        ax.set_title(f"{tag} {ATT_NAME[kind]}", loc="left"); ax.set_ylabel("BER")
        ax.set_ylim(0, 0.5 if kind == "jpeg" else None)
    for ax in axes.flat:
        ax.set_xticks([75, 100, 125, 150, 175, 200]); ax.grid(axis="y")
    for ax in axes[1]:
        ax.set_xlabel(r"Quantization step $\Delta$")
    if e21s is not None:
        fig.text(0.5, -0.01, "Solid: original embedding.  Dashed: overflow-safe embedding.",
                 ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp2_delta.pdf"), bbox_inches="tight")
    plt.close(fig)

# =============================================================================== gate
if gate is not None:
    g = summary(gate, ["config", "overflow_safe", "delta"])
    g.to_csv(os.path.join(RES, "gate_summary.csv"))
    order = ["C I", "C II", "M I", "M II", "DCT"]
    L = latex_table(g, r"Moment transforms versus the orthonormal DCT, all with $\Delta=150$ and coefficient 62. "
                       + CAPTION_COMMON, "tab:moments_vs_baselines", ["Transform"])
    for safe, name in [(False, "Original embedding"), (True, "Overflow-safe embedding")]:
        s_ = g.xs((safe, 150), level=["overflow_safe", "delta"]).reindex(order)
        b_ = bold_best(s_, s_.columns, LOWER)
        L.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(s_.columns) + 1, name))
        for cfg, r in b_.iterrows():
            L.append(cfg.replace("C I", r"C\,I").replace(
                "C II", r"C\,II").replace("M I", r"M\,I").replace("M II", r"M\,II") + " & " +
                " & ".join(r.values) + r" \\")
        L.append(r"\midrule" if not safe else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(RES, "exp2_baselines_table.tex"), "w").write("\n".join(L) + "\n")

    # ---- curvas BER-PSNR (comparación a igual imperceptibilidad), inserción sin desbordamiento
    show = [("C I", CAT[0], "o"), ("M I", CAT[1], "s"), ("DCT", CAT[2], "D")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharex=True)
    for ax, (col, ttl) in zip(axes, [("BER G", "(a) Gaussian noise, $p$ = 0.10"),
                                     ("BER SP", "(b) Salt & pepper, $d$ = 0.05"),
                                     ("BER J90", "(c) JPEG, $Q$ = 90")]):
        for cfg, c, mk in show:
            sub = g.xs((cfg, True), level=["config", "overflow_safe"]).sort_values("PSNR")
            ax.plot(sub["PSNR"], sub[col], color=c, marker=mk, ms=3.5, lw=1.4,
                    label=cfg)
        ax.set_title(ttl, loc="left"); ax.grid(axis="y"); ax.set_ylim(bottom=0)
        ax.set_xlabel("PSNR (dB)")
    axes[0].set_ylabel("BER")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp2_ber_psnr.pdf"), bbox_inches="tight")
    plt.close(fig)

# =============================================================================== E2.2 local
e22 = load("e22")
if e22 is not None:
    e22["series"] = e22.config + " – " + e22.param.map({"a": "a", "mu": "μ", "gamma": "γ"})
    s = summary(e22, ["config", "param", "mult"])
    s.to_csv(os.path.join(RES, "e22_summary.csv"))
    rng = s.groupby(level=["config", "param"]).agg(lambda v: v.max() - v.min())
    rng.to_csv(os.path.join(RES, "e22_ranges.csv"))
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Stability under local changes of the polynomial parameters "
         r"(each parameter multiplied by 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5 and 2, one at a time; "
         r"$\Delta=150$, coefficient 62). Each cell gives the minimum--maximum of the mean over the 24 "
         r"images. In the Meixner configurations $\mu$ enters the basis multiplied by $\gamma$ "
         r"($\gamma\ll 1$), so its effect is negligible.}", r"\label{tab:poly_param_sensitivity}",
         r"\setlength{\tabcolsep}{3pt}", r"\small", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Config. & Param. & PSNR (dB) & SSIM & BER$_0$ & BER$_{G,0.10}$ & BER$_{SP,0.05}$ \\",
         r"\midrule"]
    for (cfg, par), _ in rng.iterrows():
        sub = s.loc[(cfg, par)]
        mm = lambda c, f: f"{sub[c].min():{f}}--{sub[c].max():{f}}"
        L.append(f"{cfg.replace(' ', chr(92) + ',')} & ${dict(a='a', mu=r'\mu', gamma=r'\gamma')[par]}$ & "
                 f"{mm('PSNR', '.2f')} & {mm('SSIM', '.4f')} & {mm('BER none', '.4f')} & "
                 f"{mm('BER G', '.4f')} & {mm('BER SP', '.4f')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(RES, "exp2_params_table.tex"), "w").write("\n".join(L) + "\n")

    series = [("C I", "a"), ("C II", "a"), ("M I", "gamma"), ("M II", "gamma")]
    lab = {("C I", "a"): r"C I ($a$)", ("C II", "a"): r"C II ($a$)",
           ("M I", "gamma"): r"M I ($\gamma$)", ("M II", "gamma"): r"M II ($\gamma$)"}
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1), sharex=True)
    for ax, (col, ttl) in zip(axes, [("PSNR", "(a) PSNR (dB)"), ("BER none", "(b) BER, no attack"),
                                     ("BER G", "(c) BER, Gaussian 0.10"),
                                     ("BER SP", "(d) BER, S&P 0.05")]):
        for i, key in enumerate(series):
            v = s.loc[key][col]
            ax.plot(v.index, v.values, color=CAT[i], marker=MARK[i], ms=3.5, lw=1.4, label=lab[key])
        ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(["×0.5", "×1", "×2"])
        ax.minorticks_off(); ax.set_title(ttl, loc="left"); ax.grid(axis="y")
        if col == "PSNR":
            ax.set_ylim(39, 41)
        else:
            ax.set_ylim(0, 0.1)
    axes[0].set_ylabel("Mean over 24 images")
    fig.supxlabel("Parameter multiplier (log scale)", fontsize=9, y=0.02)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp2_params_local.pdf"), bbox_inches="tight")
    plt.close(fig)

# =============================================================================== E2.2 wide
w = load("e22wide")
if w is not None:
    s = summary(w.fillna({"value": -1}), ["config", "param", "value"])
    s["orth_err"] = w.fillna({"value": -1}).groupby(["config", "param", "value"]).orth_err.mean()
    s.to_csv(os.path.join(RES, "e22wide_summary.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.25))
    curves = [("Charlier", "a", r"Charlier, $a$", CAT[0], "o"), ("Meixner", "gamma", r"Meixner, $\gamma$", CAT[1], "s")]
    paper_vals = {"a": [7e-4, 5e-4], "gamma": [4.1e-5, 7.5e-5]}
    for ax, (col, ttl) in zip(axes, [("PSNR", "(a) PSNR (dB)"), ("BER none", "(b) BER, no attack"),
                                     ("BER G", "(c) BER, Gaussian 0.10"),
                                     ("BER SP", "(d) BER, S&P 0.05")]):
        for cfg, par, label, c, mk in curves:
            v = s.loc[(cfg, par)][col]
            ax.plot(v.index, v.values, color=c, marker=mk, ms=3.2, lw=1.4, label=label)
        for ref, ls in [("DCT", ":")]:
            yv = s.loc[(ref, "ref")][col].iloc[0]
            ax.axhline(yv, color=MUTED, lw=0.9, ls=ls)
        ax.axvspan(3e-5, 1e-3, color="#e4e3df", alpha=0.6, lw=0)
        ax.set_xscale("log"); ax.set_xlim(7e-6, 15); ax.set_xticks([1e-5, 1e-3, 1e-1, 10])
        ax.minorticks_off(); ax.set_title(ttl, loc="left"); ax.grid(axis="y")
    axes[0].set_ylabel("Mean over 24 images")
    hs = [Line2D([], [], color=c, marker=mk, ms=3.2, lw=1.4, label=lb) for _, _, lb, c, mk in curves]
    hs += [Line2D([], [], color=MUTED, ls=":", lw=0.9, label="DCT"),
           matplotlib.patches.Patch(color="#e4e3df", label="Values used in the paper")]
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.5, 0.075, "Polynomial parameter (log scale)", ha="center", fontsize=9)
    fig.legend(handles=hs, frameon=False, fontsize=7.5, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(FIG, "exp2_params_wide.pdf"), bbox_inches="tight")
    plt.close(fig)

# =============================================================================== E2.3 coefficient
e23 = load("e23")
if e23 is not None:
    import sys; sys.path.insert(0, HERE); from fastwm import ZZ
    order = ["C I", "C II", "M I", "M II", "DCT"]
    cols = [("psnr", "none", 0, "PSNR (dB)"), ("ber", "none", 0, "BER, no attack"),
            ("ber", "gaussian", 0.10, "BER, Gaussian 0.10"), ("ber", "salt_pepper", 0.05, "BER, S&P 0.05"),
            ("ber", "jpeg", 90, "BER, JPEG Q=90")]
    tab = {}
    for metric, kind, lev, _ in cols:
        tab[(metric, kind, lev)] = cond(e23, kind, lev).groupby(["config", "coef"])[metric].mean()
    out = pd.DataFrame({f"{m}_{k}_{l}": v for (m, k, l), v in tab.items()})
    out["ncc_gaussian_0.1"] = cond(e23, "gaussian", 0.10).groupby(["config", "coef"]).ncc.mean()
    out["ncc_salt_pepper_0.05"] = cond(e23, "salt_pepper", 0.05).groupby(["config", "coef"]).ncc.mean()
    out["ber_jpeg_70"] = cond(e23, "jpeg", 70).groupby(["config", "coef"]).ber.mean()
    pos = {k: divmod(int(ZZ[k]), 8) for k in range(64)}
    out["row"] = [pos[k][0] for _, k in out.index]; out["col"] = [pos[k][1] for _, k in out.index]
    out.to_csv(os.path.join(RES, "e23_summary.csv"))

    fig, axes = plt.subplots(len(order), len(cols), figsize=(6.6, 6.8))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.955, bottom=0.085, wspace=0.08, hspace=0.12)
    for j, (metric, kind, lev, ttl) in enumerate(cols):
        v = tab[(metric, kind, lev)]
        vmin, vmax = (v.min(), v.max()) if metric == "psnr" else (0, v.max())
        cmap = "Blues_r" if metric == "psnr" else "Blues"            # oscuro = peor en todas las columnas
        for i, cfg in enumerate(order):
            M = np.full((8, 8), np.nan)
            for k in range(64):
                r, c = pos[k]; M[r, c] = v.loc[(cfg, k)]
            ax = axes[i, j]
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
            r62, c62 = pos[62]
            ax.add_patch(Rectangle((c62 - .5, r62 - .5), 1, 1, fill=False, ec="#e34948", lw=1.0))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(ttl, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(cfg, fontsize=8.5)
        pb = axes[-1, j].get_position()
        cax = fig.add_axes([pb.x0 + 0.01, pb.y0 - 0.032, pb.width - 0.02, 0.011])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
        cb.locator = matplotlib.ticker.MaxNLocator(3); cb.update_ticks()
    fig.text(0.5, 0.012, "Each map: 8×8 moment positions (u, v); red box: coefficient 62 used in the paper. "
             "Darker = worse.", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(FIG, "exp2_coefficient_maps.pdf"), bbox_inches="tight")
    plt.close(fig)

    # tabla: coeficiente 62 frente a los mejores por configuración (criterio: BER media G + SP, PSNR >= 39 dB)
    rows = []
    for cfg in order:
        sub = out.loc[cfg]
        score = (sub["ber_gaussian_0.1"] + sub["ber_salt_pepper_0.05"]) / 2
        ok = sub["psnr_none_0"] >= 39.0
        best = score[ok].nsmallest(3).index.tolist()
        for k in [62] + [b for b in best if b != 62]:
            r = sub.loc[k]
            rows.append(dict(config=cfg, coef=k, pos=f"({int(r.row)},{int(r.col)})", psnr=r["psnr_none_0"],
                             ber0=r["ber_none_0"], berG=r["ber_gaussian_0.1"], berSP=r["ber_salt_pepper_0.05"],
                             berJ90=r["ber_jpeg_90"], nccG=r["ncc_gaussian_0.1"], used=k == 62))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(RES, "e23_best_coefficients.csv"), index=False)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Coefficient used for the robust watermark (zigzag index $k$, position $(u,v)$ in the "
         r"8$\times$8 moment matrix; $\Delta=150$). For each transform the coefficient of the paper "
         r"($k=62$) is compared with the three best coefficients by mean BER under Gaussian noise (0.10) "
         r"and salt-and-pepper noise (0.05) among those with PSNR $\geq$ 39 dB. Means over 24 images.}",
         r"\label{tab:coefficient_sensitivity}", r"\setlength{\tabcolsep}{3pt}", r"\small",
         r"\begin{tabular}{lrcccccccc}", r"\toprule",
         r"Transform & $k$ & $(u,v)$ & PSNR (dB) & BER$_0$ & BER$_{G,0.10}$ & BER$_{SP,0.05}$ & BER$_{Q90}$ & NCC$_{G,0.10}$ \\",
         r"\midrule"]
    for cfg in order:
        sub = t[t.config == cfg]
        for i, r in enumerate(sub.itertuples()):
            name = cfg.replace(" ", r"\,") if i == 0 else ""
            k = f"\\textbf{{{r.coef}}}" if r.used else f"{r.coef}"
            L.append(f"{name} & {k} & {r.pos} & {r.psnr:.2f} & {r.ber0:.4f} & {r.berG:.4f} & "
                     f"{r.berSP:.4f} & {r.berJ90:.4f} & {r.nccG:.4f} \\\\")
        L.append(r"\midrule" if cfg != order[-1] else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(RES, "exp2_coefficient_table.tex"), "w").write("\n".join(L) + "\n")

print("ok")
