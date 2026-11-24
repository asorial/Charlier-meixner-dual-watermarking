"""
Tablas y figuras (PDF) del Experimento 1 a partir de results/exp1_trials.csv.

Uso (desde la raíz del repositorio):
    python experiments/exp1_tamper_localization/plot_exp1.py
Salida:
    results/exp1_summary_by_config.csv   media y d.t. por configuración, ataque y área
    results/exp1_summary.csv             agregado sobre configuraciones
    results/exp1_table.tex               tabla LaTeX (booktabs): Precision, Recall, F1, FPR, FNR, Accuracy
    figures/exp1_examples.pdf            3 ataques (10 %): original, manipulada, región real, detectada
    figures/exp1_examples_areas.pdf      copy-move a 1, 5, 10 y 20 %: mismas cuatro columnas
    figures/exp1_pixel_iou.pdf           IoU por píxel frente al área (granularidad de bloque)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
EXD = os.path.join(RES, "examples")
os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,            # fuentes TrueType incrustadas
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "grid.color": "#e4e3df", "grid.linewidth": 0.5,
})
ATTACKS = ["copy-move", "splicing", "removal"]
LABEL = {"copy-move": "Copy-move", "splicing": "Splicing", "removal": "Object removal"}
STYLE = {"copy-move": ("#2a78d6", "o", "-"),          # paleta categórica validada, slots 1-3
         "splicing": ("#eb6834", "s", "--"),
         "removal": ("#1baf7a", "^", ":")}
OUTLINE = "#e34948"
METRICS = ["precision", "recall", "F1", "FPR", "FNR", "accuracy", "loc_error",
           "px_precision", "px_recall", "px_F1", "px_IoU", "px_accuracy"]

d = pd.read_csv(os.path.join(RES, "exp1_trials.csv"))
d["area"] = (d["fraction"] * 100).round().astype(int)
AREAS = sorted(d["area"].unique())

# ---------------------------------------------------------------- tablas
d.groupby(["config", "attack", "area"])[["gt_blocks"] + METRICS].agg(["mean", "std"]).to_csv(
    os.path.join(RES, "exp1_summary_by_config.csv"))
agg = d.groupby(["attack", "area"])[["gt_blocks"] + METRICS].agg(["mean", "std"])
agg.to_csv(os.path.join(RES, "exp1_summary.csv"))
spread = d.groupby(["attack", "area", "config"])[METRICS].mean().groupby(["attack", "area"]).agg(
    lambda s: s.max() - s.min()).max().max()


def pm(m, s, nd=4):
    return f"${m:.{nd}f} \\pm {s:.{nd}f}$" if s >= 0.5 * 10 ** -nd else f"${m:.{nd}f}$"


def sci(v):
    e = int(np.floor(np.log10(v)))
    return f"${v / 10 ** e:.1f}\\times 10^{{{e}}}$"


n_row = len(d) // (len(ATTACKS) * len(AREAS))
max_sd = float(agg.xs("std", axis=1, level=1)[["precision", "recall", "F1", "FPR", "FNR", "accuracy"]].max().max())
miss = f"{int(d.FN.sum())} of {int(d.gt_blocks.sum()):,}".replace(",", "\\,") + " tampered blocks were missed " \
       f"({sci(d.FN.sum() / d.gt_blocks.sum())})" if d.FN.sum() else "no tampered block was missed"
lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Block-level tamper detection and localization of the fragile layer "
         f"(24 images, 4 configurations, 5 random positions; mean over {n_row} trials per row; "
         f"largest standard deviation {max_sd:.4f}). Ground truth: 8$\\times$8 blocks with at least one modified pixel. "
         f"No false positives were observed and {miss}, consistent with the $2^{{-16}}$ collision "
         r"probability of the 16-bit tag. Accuracy is dominated by the authentic blocks "
         r"(95--99\% of each image) and is reported for completeness. Results are identical for "
         r"C\,I, C\,II, M\,I and M\,II (maximum difference between configurations "
         f"{sci(spread) if spread > 0 else '0'}).}}", r"\label{tab:tamper_localization}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrcccccc}", r"\toprule",
         r"Attack & Area & GT blocks & Precision & Recall & F1 & FPR & FNR & Accuracy \\",
         r"\midrule"]
for att in ATTACKS:
    for i, area in enumerate(AREAS):
        r = agg.loc[(att, area)]
        cell = lambda k: f"{r[(k, 'mean')]:.5f}"
        lines.append(" & ".join([LABEL[att] if i == 0 else "", f"{area}\\%",
                                 f"{r[('gt_blocks', 'mean')]:.0f}", cell("precision"),
                                 cell("recall"), cell("F1"), cell("FPR"), cell("FNR"),
                                 cell("accuracy")]) + r" \\")
    lines.append(r"\midrule" if att != ATTACKS[-1] else r"\bottomrule")
lines += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(RES, "exp1_table.tex"), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- figuras de ejemplos
base = np.load(os.path.join(EXD, "example_base.npz"))
HEADS = ["Original", "Tampered", "Tampered region\n(ground truth)", "Detected region\n(8$\\times$8 blocks)"]


def example_figure(cases, row_labels, fname):
    fig, axes = plt.subplots(len(cases), 4, figsize=(6.6, 1.72 * len(cases) + 0.35))
    for r, (att, area) in enumerate(cases):
        z = np.load(os.path.join(EXD, f"example_{att}_{area}.npz"))
        ys, xs = np.where(z["region"]); y0, x0 = ys.min(), xs.min(); side = ys.max() - y0 + 1
        panels = [base["original"], z["tampered"], z["changed"].astype(float),
                  np.kron(z["detected"], np.ones((8, 8)))]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            if c < 2:
                ax.imshow(img, interpolation="none")
            else:
                ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="none")
            if c >= 1:
                ax.add_patch(Rectangle((x0 - .5, y0 - .5), side, side, fill=False,
                                       ec=OUTLINE, lw=0.8, ls="--"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("#c3c2b7"); s.set_linewidth(0.5)
            if r == 0:
                ax.set_title(HEADS[c])
        tp = int((z["detected"] & z["gt_blocks"]).sum()); n = int(z["gt_blocks"].sum())
        fp = int((z["detected"] & ~z["gt_blocks"]).sum())
        axes[r, 0].set_ylabel(row_labels[r])
        axes[r, 3].set_xlabel(f"TP {tp}/{n} blocks, FP {fp}", fontsize=7.5, labelpad=2)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.5)
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight", dpi=300)
    plt.close(fig)


example_figure([(a, 10) for a in ATTACKS], [f"{LABEL[a]} (10%)" for a in ATTACKS],
               "exp1_examples.pdf")
example_figure([("copy-move", a) for a in AREAS], [f"Copy-move {a}%" for a in AREAS],
               "exp1_examples_areas.pdf")

# ---------------------------------------------------------------- IoU por píxel
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for k, att in enumerate(ATTACKS):
    col, mk, ls = STYLE[att]
    g = d[d.attack == att].groupby("area")["px_IoU"]
    m, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
    x = np.array(AREAS) * (1 + 0.04 * (k - 1))
    ax.errorbar(x, m.values, yerr=[m.values - lo.values, hi.values - m.values],
                color=col, marker=mk, ms=4.5, lw=1.4, ls=ls, capsize=2, elinewidth=0.8,
                label=LABEL[att])
ax.set_xscale("log"); ax.set_xticks(AREAS); ax.set_xticklabels([f"{a}%" for a in AREAS])
ax.minorticks_off(); ax.set_xlabel("Tampered area (% of image)")
ax.set_ylabel("Pixel-level IoU")
ax.grid(axis="y"); ax.set_ylim(0.6, 1.005)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "exp1_pixel_iou.pdf"), bbox_inches="tight")
plt.close(fig)

print(f"max diferencia entre configuraciones (cualquier métrica): {spread:.2e}")
print(f"TP={d.TP.sum()} FP={d.FP.sum()} FN={d.FN.sum()} GT={d.gt_blocks.sum()}")
print(agg[[(k, "mean") for k in ["precision", "recall", "F1", "FPR", "FNR", "accuracy",
                                  "px_IoU"]]].round(5).to_string())
