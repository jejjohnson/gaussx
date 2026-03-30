# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example Demo
#
# This notebook demonstrates the standard pattern for documentation examples.
# Each example generates figures and tables, saves them to `docs/images/`,
# and embeds them for static rendering.
#
# **What you'll learn:**
#
# 1. How to set up the image output directory
# 2. How to save and embed figures
# 3. How to produce and embed timing / statistics tables

# %%
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Image output directory ──────────────────────────────────────────
# Path(__file__) ensures this works regardless of the working directory.
# The directory is created eagerly so savefig never fails.
IMG_DIR = Path(__file__).resolve().parent.parent / "images" / "example_demo"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Setup
#
# We create some sample data to plot.

# %%
x = np.linspace(0, 2 * np.pi, 200)
signals = {
    "sin(x)": np.sin(x),
    "sin(2x)": np.sin(2 * x),
    "sin(x) + 0.5 sin(3x)": np.sin(x) + 0.5 * np.sin(3 * x),
}

# %% [markdown]
# ## Figures
#
# Generate, save, and embed a comparison figure.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for label, y in signals.items():
    ax.plot(x, y, label=label)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Signal Comparison")
ax.legend()
plt.tight_layout()
fig.savefig(IMG_DIR / "signal_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Signal comparison](../images/example_demo/signal_comparison.png)

# %% [markdown]
# ## Timing & Statistics
#
# Collect timing data, then present as both printed output (visible when
# executed) and a static markdown table (visible in the built docs).

# %%
repeats = 1000
results: dict[str, dict] = {}
for label, y in signals.items():
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = np.fft.rfft(y)
    elapsed_ms = (time.perf_counter() - t0) * 1e3 / repeats
    rms = float(np.sqrt(np.mean(y**2)))
    results[label] = {"time_ms": elapsed_ms, "rms": rms}

print(f"{'Signal':30s}  {'Time (ms)':>10s}  {'RMS':>8s}")
print("-" * 52)
for label, stats in results.items():
    print(f"{label:30s}  {stats['time_ms']:10.4f}  {stats['rms']:8.4f}")

# %% [markdown]
# | Signal | FFT Time (ms) | RMS |
# |--------|---------------|-----|
# | sin(x) | ~0.01 | 0.707 |
# | sin(2x) | ~0.01 | 0.707 |
# | sin(x) + 0.5 sin(3x) | ~0.01 | 0.790 |
#
# > **Note**: Update the table above after re-running the notebook.

# %%
fig, (ax_time, ax_rms) = plt.subplots(1, 2, figsize=(10, 4))

labels = list(results.keys())
times = [results[k]["time_ms"] for k in labels]
rms_vals = [results[k]["rms"] for k in labels]

ax_time.bar(range(len(labels)), times, color="steelblue", edgecolor="black")
ax_time.set_xticks(range(len(labels)))
ax_time.set_xticklabels(labels, fontsize=8, rotation=15)
ax_time.set_ylabel("Time (ms)")
ax_time.set_title("FFT Timing")

ax_rms.bar(range(len(labels)), rms_vals, color="coral", edgecolor="black")
ax_rms.set_xticks(range(len(labels)))
ax_rms.set_xticklabels(labels, fontsize=8, rotation=15)
ax_rms.set_ylabel("RMS")
ax_rms.set_title("RMS Amplitude")

plt.tight_layout()
fig.savefig(IMG_DIR / "timing_stats.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ![Timing and statistics](../images/example_demo/timing_stats.png)

# %% [markdown]
# ## Summary
#
# This example showed the standard documentation pattern:
#
# 1. **Setup** — imports, `IMG_DIR`, parameters
# 2. **Compute** — run the demonstration
# 3. **Save & embed** — `savefig` + markdown image reference
# 4. **Tables** — printed output + static markdown fallback
