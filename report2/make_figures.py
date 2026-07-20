"""Generate column-sized vector figures for the Phase 2 report.

Reads the real experiment artefacts in trained_models/ rather than hardcoding,
so the figures stay consistent with the JSON results.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TM = "../trained_models"
res = json.load(open(f"{TM}/model_c_cmaes_danube_tune/results.json"))
hpo = json.load(open(f"{TM}/hpo_comparison_summary.json"))

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "figure.dpi": 300,
})

C = {"fused": "#1f4e79", "temporal": "#c0392b", "spatial": "#27865f"}

# --- Fig 1: per-stream R2 across targets -----------------------------------
targets = ["dissolved_oxygen", "total_phosphorus", "nitrate_n",
           "electrical_conductance", "chlorophyll_a"]
short = ["DO", "TP", "NO$_3$-N", "EC", "Chl-a"]
m = res["metrics"]
streams = ["fused", "temporal_stream", "spatial_stream"]
labels = ["Fused", "Temporal", "Spatial"]
cols = [C["fused"], C["temporal"], C["spatial"]]

fig, ax = plt.subplots(figsize=(3.4, 2.0))
x = np.arange(len(targets)); w = 0.26
for i, (s, lab, c) in enumerate(zip(streams, labels, cols)):
    vals = [m[s][t]["R2"] for t in targets]
    ax.bar(x + (i - 1) * w, vals, w, label=lab, color=c, edgecolor="none")
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylabel("$R^2$")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
          frameon=False, columnspacing=1.2, handlelength=1.2)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_streams.pdf", bbox_inches="tight")
plt.close(fig)

# --- Fig 2: HPO strategy accuracy vs cost ----------------------------------
rows = hpo["summary_table"]["rows"]
names = {"random_search_ga": "Random+GA", "optuna_tpe_ga": "TPE+GA",
         "bohb_ga": "BOHB+GA", "random_search_cmaes": "Random+CMA-ES"}
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for r in rows:
    key, wall, mean_r2 = r[0], r[4], r[10]
    hl = key == "random_search_cmaes"
    ax.scatter(wall, mean_r2, s=46 if hl else 30,
               color=C["fused"] if hl else "#8899aa",
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.annotate(names[key], (wall, mean_r2), textcoords="offset points",
                xytext=(0, 7 if not hl else -12), ha="center", fontsize=6.3)
ax.set_xlabel("Tuning wall time (min)")
ax.set_ylabel("Mean fused $R^2$")
ax.set_xlim(150, 355); ax.set_ylim(0.30, 0.47)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_hpo.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote figures/fig_streams.pdf and figures/fig_hpo.pdf")
