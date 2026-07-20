"""Regenerate report figures from saved test-set predictions.

The plots in old-model/plots/ are 15x6 inches, which become illegible when
scaled into an IEEE column. These are sized for the column width instead.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = "../old-model"
yp = np.load(f"{OLD}/model_assets/y_pred_actual.npy")
yt = np.load(f"{OLD}/model_assets/y_test_actual.npy")
feats = list(pd.read_csv(f"{OLD}/waterdata.csv").columns.drop("Date"))

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "figure.dpi": 300,
})

LABEL = {
    "Temperature_C": "Temperature (°C)",
    "pH": "pH",
    "Conductivity_uS_cm": "Conductivity (µS/cm)",
    "Ammonia_mgL": "Ammonia (mg/L)",
    "Phosphates_mgL": "Phosphates (mg/L)",
    "Turbidity_NTU": "Turbidity (NTU)",
    "Nitrates_mgL": "Nitrates (mg/L)",
    "Dissolved_Oxygen_mgL": "Dissolved oxygen (mg/L)",
    "TDS_mgL": "TDS (mg/L)",
}


def panel(ax, name, legend=False):
    i = feats.index(name)
    ax.plot(yt[:, i], color="#1f4e79", lw=0.9, label="Actual")
    ax.plot(yp[:, i], color="#c0392b", lw=0.9, ls="--", label="Predicted")
    ax.set_title(LABEL[name], pad=3)
    ax.set_xlabel("Test time step", labelpad=2)
    ax.margins(x=0.01)
    if legend:
        ax.legend(loc="upper right", framealpha=0.9, borderpad=0.3)


# Fig 1: temperature, full column width
fig, ax = plt.subplots(figsize=(3.4, 1.9))
panel(ax, "Temperature_C", legend=True)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_temperature.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 2: 2x2 grid, full column width
fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.8))
for ax, name in zip(axes.ravel(), ["Turbidity_NTU", "Nitrates_mgL",
                                   "Dissolved_Oxygen_mgL", "TDS_mgL"]):
    panel(ax, name, legend=(name == "Turbidity_NTU"))
fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.7)
fig.savefig("figures/fig_grid.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote figures/fig_temperature.pdf and figures/fig_grid.pdf")


# Fig 3: remaining four parameters, 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.8))
for ax, name in zip(axes.ravel(), ["pH", "Conductivity_uS_cm",
                                   "Ammonia_mgL", "Phosphates_mgL"]):
    panel(ax, name, legend=(name == "pH"))
fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.7)
fig.savefig("figures/fig_grid2.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 4: residual histogram for the six pollution-indicative parameters
POLL = ["Conductivity_uS_cm", "Turbidity_NTU", "TDS_mgL",
        "Ammonia_mgL", "Nitrates_mgL", "Phosphates_mgL"]
fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.1))
for ax, name in zip(axes.ravel(), POLL):
    i = feats.index(name)
    r = yt[:, i] - yp[:, i]
    ax.hist(r, bins=22, color="#1f4e79", edgecolor="white", linewidth=0.3)
    ax.axvline(3 * r.std(), color="#c0392b", ls="--", lw=0.9)
    ax.set_title(LABEL.get(name, name).split(" (")[0], pad=2, fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.set_yticks([])
fig.suptitle("Residual distributions with $3\\sigma$ alert threshold (dashed)",
             fontsize=7, y=1.04)
fig.tight_layout(pad=0.25, w_pad=0.4, h_pad=0.6)
fig.savefig("figures/fig_residuals.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig_grid2.pdf and fig_residuals.pdf")


# Fig 5: lag-1 autocorrelation, explains which parameters persistence can win
raw = pd.read_csv(f"{OLD}/waterdata.csv", parse_dates=["Date"])
acf = {f: np.corrcoef(raw[f].values[:-1], raw[f].values[1:])[0, 1] for f in feats}
order = sorted(feats, key=lambda f: -acf[f])
fig, ax = plt.subplots(figsize=(3.4, 1.75))
cols = ["#c0392b" if acf[f] > 0.5 else "#1f4e79" for f in order]
ax.bar(range(len(order)), [acf[f] for f in order], color=cols, edgecolor="none")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABEL.get(f, f).split(" (")[0] for f in order],
                   rotation=38, ha="right", fontsize=5.6)
ax.set_ylabel("Lag-1 autocorrelation")
ax.axhline(0, color="black", lw=0.6)
fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_acf.pdf", bbox_inches="tight")
plt.close(fig)

# Fig 6: predicted-vs-actual scatter showing amplitude damping
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75))
for ax, name in zip(axes, ["Temperature_C", "Dissolved_Oxygen_mgL"]):
    i = feats.index(name)
    a, p = yt[:, i], yp[:, i]
    ax.scatter(a, p, s=3, color="#1f4e79", alpha=0.5, edgecolor="none")
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", label="1:1")
    b, c = np.polyfit(a, p, 1)
    ax.plot([lo, hi], [b * lo + c, b * hi + c], color="#c0392b", lw=1.0,
            label=f"slope {b:.2f}")
    ax.set_title(LABEL[name].split(" (")[0], pad=3)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.1)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig("figures/fig_scatter.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote fig_acf.pdf and fig_scatter.pdf")
