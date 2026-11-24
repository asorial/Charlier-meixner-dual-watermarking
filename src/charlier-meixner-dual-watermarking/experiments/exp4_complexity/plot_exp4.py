"""
Tablas (LaTeX/CSV), figuras (PDF) y texto de complejidad del Experimento 4.

Uso (desde la raíz del repositorio):
    python experiments/exp4_complexity/plot_exp4.py [--label NOMBRE]
(--label debe coincidir con el usado en run_exp4.py; sin etiqueta lee results/)
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--label", default=""); a = ap.parse_args()
RES = os.path.join(HERE, "results" + (f"_{a.label}" if a.label else ""))
FIG = os.path.join(HERE, "figures" + (f"_{a.label}" if a.label else "")); os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e4e3df", "grid.linewidth": 0.5})
MUTED = "#52514e"
C_REF, C_VEC, C_LOOP = "#2a78d6", "#eb6834", "#1baf7a"
CONFIGS = ["C I", "C II", "M I", "M II", "DCT"]
TEX = {"C I": r"C\,I", "C II": r"C\,II", "M I": r"M\,I", "M II": r"M\,II", "DCT": "DCT"}
OPS = [("insert", "Embedding"), ("extract", "Robust extraction"), ("verify", "Authentication")]

info = json.load(open(os.path.join(RES, "system_info.json")))
ops = pd.read_csv(os.path.join(RES, "exp4_operations.csv"))
mem = pd.read_csv(os.path.join(RES, "exp4_memory.csv"))
ops["ms"] = ops.seconds * 1e3


def q(x, p):
    return np.percentile(x, p)


summ = ops.groupby(["impl", "op", "config"]).ms.agg(median="median", q1=lambda x: q(x, 25),
                                                     q3=lambda x: q(x, 75), mean="mean", n="size")
summ["iqr"] = summ.q3 - summ.q1
summ.to_csv(os.path.join(RES, "exp4_times_summary.csv"))
msum = mem.groupby(["impl", "op", "config"]).peak_mb.median()
msum.to_csv(os.path.join(RES, "exp4_memory_summary.csv"))

# diferencia relativa máxima entre configuraciones (por implementación y operación)
spread = summ["median"].groupby(level=["impl", "op"]).agg(lambda v: (v.max() - v.min()) / v.min())

# ------------------------------------------------------------------ texto del sistema
pk = info.get("packages", {})
sysline = (f"{info.get('cpu_model', info.get('processor', 'CPU'))}, {info.get('logical_cpus')} logical cores, "
           f"{info.get('ram_gb', '?')} GB RAM, {info.get('os')}, Python {info.get('python')}, "
           f"NumPy {pk.get('numpy', '?')} ({info.get('blas', '?')}), OpenCV {pk.get('cv2', '?')}")
sysline_tex = sysline.replace("_", r"\_").replace("%", r"\%")
n_img, reps = len(info.get("images", [])), info.get("reps")

# ------------------------------------------------------------------ tabla de tiempos
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Execution time per $512\times512$ RGB image (median [interquartile range] over "
     f"{n_img} images $\\times$ {reps} runs after one warm-up run) and peak memory of each operation "
     r"(tracemalloc). Embedding includes the robust and the fragile layers; robust extraction "
     r"includes the inverse Arnold transform of the $64\times64$ watermark (37 iterations in the reference "
     r"implementation, a precomputed permutation in the vectorized one); authentication returns "
     r"the map of tampered blocks. Reference: block-by-block implementation; vectorized: batched "
     r"NumPy implementation producing bit-identical outputs. Hardware: " + sysline_tex + ".}",
     r"\label{tab:execution_time}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{llcccc}", r"\toprule",
     r"Implem. & Transf. & Embedding (ms) & Extraction (ms) & Authentication (ms) & Memory (MB) \\",
     r"\midrule"]
for impl, lab in [("reference", "Ref."), ("vectorized", "Vect.")]:
    for i, cfg in enumerate(CONFIGS):
        cells = []
        for op, _ in OPS:
            r = summ.loc[(impl, op, cfg)]
            cells.append(f"{r['median']:.1f} [{r.q1:.1f}, {r.q3:.1f}]")
        pm = max(msum.loc[(impl, op, cfg)] for op, _ in OPS)
        L.append(f"{lab if i == 0 else ''} & {TEX[cfg]} & " + " & ".join(cells) + f" & {pm:.1f} \\\\")
    L.append(r"\midrule" if impl == "reference" else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(RES, "exp4_times_table.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ figura 1: tiempos por operación
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True, sharex=True)
y = np.arange(len(CONFIGS))[::-1]
for ax, (op, ttl) in zip(axes, OPS):
    for impl, c, mk, dy in [("reference", C_REF, "o", 0.12), ("vectorized", C_VEC, "s", -0.12)]:
        s = summ.loc[(impl, op)].reindex(CONFIGS)
        ax.errorbar(s["median"], y + dy, xerr=[s["median"] - s.q1, s.q3 - s["median"]], fmt=mk, color=c,
                    ms=4, capsize=2, elinewidth=0.9, lw=0)
    ax.set_xscale("log"); ax.set_title(ttl, loc="left"); ax.grid(axis="x")
    ax.set_xlim(0.4, 600); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"]); ax.minorticks_off()
    ax.set_xlabel("Time per image (ms, log scale)")
axes[0].set_yticks(y); axes[0].set_yticklabels(CONFIGS)
hs = [Line2D([], [], color=C_REF, marker="o", lw=0, ms=4, label="Reference (block loop)"),
      Line2D([], [], color=C_VEC, marker="s", lw=0, ms=4, label="Vectorized (NumPy)")]
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.savefig(os.path.join(FIG, "exp4_times.pdf"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ figura 2: escalado
slopes = {}
bs_path, is_path = os.path.join(RES, "exp4_block_size.csv"), os.path.join(RES, "exp4_image_size.csv")
if os.path.exists(bs_path) and os.path.exists(is_path):
    bs = pd.read_csv(bs_path); isz = pd.read_csv(is_path)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    for (impl, fam), g in bs.groupby(["impl", "family"]):
        m = g.groupby("N").seconds.median() * 1e3
        c = C_VEC if impl == "vectorized" else C_LOOP
        ls = {"Charlier": "-", "Meixner": "--", "DCT": ":"}[fam]
        ax.plot(m.index, m.values, color=c, ls=ls, marker="o", ms=3, lw=1.3)
        if fam == "Charlier":
            slopes[f"blocksize_{impl}"] = float(np.polyfit(np.log(m.index), np.log(m.values), 1)[0])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"])
    ax.minorticks_off(); ax.set_xlabel(r"Block size $N$"); ax.set_ylabel("Time (ms, log scale)")
    ax.set_title(r"(a) Forward + inverse transform, $512\times512$", loc="left"); ax.grid(axis="y")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    hs = [Line2D([], [], color=C_VEC, lw=1.3, label="Batched (BLAS)"), Line2D([], [], color=C_LOOP, lw=1.3, label="Block loop"),
          Line2D([], [], color=MUTED, ls="-", lw=1, label="Charlier"), Line2D([], [], color=MUTED, ls="--", lw=1, label="Meixner"),
          Line2D([], [], color=MUTED, ls=":", lw=1, label="DCT")]
    ax.legend(handles=hs, frameon=False, fontsize=6.8, loc="upper center", ncol=2)
    ax = axes[1]
    for (op, ttl), c, mk in zip(OPS, [C_REF, C_VEC, C_LOOP], ["o", "s", "^"]):
        m = isz[isz.op == op].groupby("pixels").seconds.median()
        ax.plot(m.index, m.values, color=c, marker=mk, ms=3.5, lw=1.4,
                label=f"{ttl} (slope {np.polyfit(np.log(m.index), np.log(m.values), 1)[0]:.2f})")
        slopes[f"imagesize_{op}"] = float(np.polyfit(np.log(m.index), np.log(m.values), 1)[0])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    sizes = sorted(isz["size"].unique()); ax.set_xticks([s * s for s in sizes])
    ax.set_xticklabels([f"{s}²" for s in sizes]); ax.set_xlabel("Image size (pixels, log scale)")
    ax.set_ylabel("Time (s, log scale)"); ax.set_title(r"(b) Reference implementation, $N=8$", loc="left")
    ax.grid(axis="y"); ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp4_scaling.pdf"), bbox_inches="tight"); plt.close(fig)
    json.dump(slopes, open(os.path.join(RES, "exp4_slopes.json"), "w"), indent=2)

# ------------------------------------------------------------------ texto de complejidad (LaTeX)
med = lambda impl, op: summ.loc[(impl, op)]["median"].median()
txt = rf"""% Generado por plot_exp4.py a partir de {os.path.basename(RES)}/ ({sysline})
\subsection{{Computational complexity and execution time}}\label{{sec:complexity}}

Let the host image have $H\times W$ pixels, divided into $B=HW/N^2$ non-overlapping blocks of
size $N\times N$ ($N=8$). The moment matrix of a block, $\mathcal{{M}}=\mathcal{{A}}^{{T}}\mathcal{{C}}\mathcal{{A}}$,
requires two dense $N\times N$ products, i.e.\ $2N^3$ multiply--accumulate operations, and the
reconstruction $\mathcal{{A}}\mathcal{{M}}\mathcal{{A}}^{{T}}$ the same. Embedding therefore costs
$4N^3$ operations per block and $O(BN^3)=O(HWN)$ per channel, i.e.\ $4N=32$ operations per pixel for
$N=8$. Extraction needs a single coefficient, $\nu_{{uv}}=\mathbf a_u^{{T}}\mathcal{{C}}\,\mathbf a_v$,
which costs $O(N^2)$ per block and $O(HW)$ in total. The matrices $\mathcal{{A}}$ depend only on
$N$ and the polynomial parameters and are computed once, off-line. The fragile layer computes one
HMAC-SHA256 over the $3N^2$ bytes of each block, $O(HW)$, and the Arnold scrambling of the
$64\times64$ watermark costs $O(\eta\,n_w^2)$ with $n_w=64$, independent of the host size. Memory
is $O(HW)$. The cost is identical for the Charlier, Meixner and DCT bases, since all are dense
$N\times N$ matrices applied in the same way.

Table~\ref{{tab:execution_time}} reports the measured times on {sysline_tex}. With the reference
block-by-block implementation a $512\times512$ image is watermarked in about
{med('reference', 'insert'):.0f}\,ms, its robust watermark is extracted in {med('reference', 'extract'):.0f}\,ms
and its integrity is verified in {med('reference', 'verify'):.0f}\,ms; a batched NumPy implementation
that produces bit-identical outputs reduces these times to {med('vectorized', 'insert'):.0f},
{med('vectorized', 'extract'):.1f} and {med('vectorized', 'verify'):.0f}\,ms. The five transforms differ by
less than {100 * spread.max():.0f}\,\% in every operation."""
if slopes:
    bsv = (bs[(bs.impl == "vectorized") & (bs.family == "Charlier")].groupby("N").seconds.median() * 1e3).to_dict()
    txt += rf""" The empirical scaling agrees with the analysis (Fig.~\ref{{fig:scaling}}): with $N=8$, the
log--log slope of time versus number of pixels is {slopes['imagesize_insert']:.2f} for embedding,
{slopes['imagesize_extract']:.2f} for extraction and {slopes['imagesize_verify']:.2f} for authentication
(1 for linear cost). For a fixed $512\times512$ image the number of arithmetic operations grows
linearly with $N$, but at these sizes the batched (BLAS) transform is memory-bound: forward and
inverse transforms of all blocks take {bsv[8]:.1f}\,ms for $N=8$ and between {min(bsv.values()):.1f} and
{max(bsv.values()):.1f}\,ms for $N=4$--32, whereas a block-by-block loop is dominated by the per-block
interpreter overhead, proportional to the number of blocks $HW/N^2$ (log--log slope
{slopes['blocksize_loop']:.2f} in $N$; $-2$ in the overhead-dominated limit)."""
txt += "\n"
open(os.path.join(RES, "exp4_complexity.tex"), "w").write(txt)

print(sysline)
print(summ["median"].unstack("config").round(2).to_string())
print("max relative spread between transforms:", spread.round(3).to_dict())
print("slopes:", {k: round(v, 2) for k, v in slopes.items()})
