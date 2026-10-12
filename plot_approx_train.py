"""Val loss of the approximate-attention LMs (train_lm.py, SEQ=1024 BATCH=4 STEPS=3000) vs softmax.
    python plot_approx_train.py -> runs/train/approx_train.png"""
import csv
import os

import matplotlib.pyplot as plt

NAMES = {"softmax": "softmax (exact)", "sparse": "softmax on window 32 + 16 random", "ppr": "PPR a=0.7 on the same set",
         "bitonic": "soft bitonic top-16 of last 64", "bandit": "bandit elimination z=2"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for name, label in NAMES.items():
    path = f"runs/train/{name}_seq1024.csv"
    if not os.path.exists(path): continue
    rows = [r for r in csv.DictReader(open(path)) if r.get("val_loss")]
    if not rows: continue
    tok = [int(r["tokens"]) / 1e6 for r in rows]; vl = [float(r["val_loss"]) for r in rows]; sec = [float(r["seconds"]) / 60 for r in rows]
    ax[0].plot(tok, vl, "o-", ms=3, label=f"{label} ({vl[-1]:.2f})"); ax[1].plot(sec, vl, "o-", ms=3, label=label)
ax[0].set(xlabel="tokens (M)", ylabel="val loss (nats / BPE token)", title="validation loss, 3000 steps x 4096 tokens"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="minutes", ylabel="val loss", title="validation loss vs wall clock")
fig.tight_layout(); fig.savefig("runs/train/approx_train.png", dpi=120)
