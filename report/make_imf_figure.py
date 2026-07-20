"""Decompose one parameter with CEEMDAN and plot the resulting IMFs."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyEMD.CEEMDAN import CEEMDAN

df = pd.read_csv("../old-model/waterdata.csv", parse_dates=["Date"])
sig = df["Temperature_C"].values
np.random.seed(42)
imfs = CEEMDAN()(sig)          # package defaults, as used in training
print("IMFs extracted:", imfs.shape[0])

plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5,
                     "xtick.labelsize": 5, "ytick.labelsize": 5,
                     "figure.dpi": 300})
show = min(6, imfs.shape[0])
fig, axes = plt.subplots(show + 1, 1, figsize=(3.4, 4.0), sharex=True)
axes[0].plot(sig, lw=0.5, color="#1f4e79")
axes[0].set_ylabel("Signal", fontsize=5.5, rotation=0, ha="right", va="center")
for k in range(show):
    lab = f"IMF {k+1}" if k < show - 1 else "Residual"
    series = imfs[k] if k < show - 1 else imfs[show - 1:].sum(axis=0)
    axes[k + 1].plot(series, lw=0.4, color="#c0392b")
    axes[k + 1].set_ylabel(lab, fontsize=5.5, rotation=0, ha="right", va="center")
for ax in axes:
    ax.tick_params(labelsize=4.5); ax.margins(x=0.01)
axes[-1].set_xlabel("Observation index", fontsize=6)
fig.tight_layout(pad=0.25, h_pad=0.15)
fig.savefig("figures/fig_imf.pdf", bbox_inches="tight")
print("wrote figures/fig_imf.pdf")
