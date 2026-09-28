# Real-data illustration for the rotation two-sample test.
# Uses the paper's own tested procedures (alignment_study.py) on a real,
# non-Gaussian bivariate dataset (sklearn breast-cancer). No network needed.
import os, numpy as np
os.environ.setdefault("OUT", "/tmp/rd_out"); os.makedirs("/tmp/rd_out", exist_ok=True)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import alignment_study as A
from sklearn.datasets import load_breast_cancer
from scipy.stats import skew, kurtosis

SEED = 20250510; B = 199; alpha = 0.05; xi = A.CONTRASTS["u4"]
def rot(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])
tests = {
    "phi-PP (ours)":     lambda X, Y, r: A.phipp_p(X, Y, B, r, xi),
    "energy+ICA":        lambda X, Y, r: A.energy_ica_p(X, Y, B, r),
    "energy rot-align":  lambda X, Y, r: A.energy_rotalign_p(X, Y, B, r),
    "energy naive":      lambda X, Y, r: A.energy_naive_p(X, Y, B, r),
}
d = load_breast_cancer(); nm = d.feature_names
pairs = {"area / concavity": [3, 6], "radius / worst-concave-pts": [0, 27]}

# ---- Part 1: calibration on real marginals (rotation null true by construction)
reps = 100
p1 = {}
for label, idx in pairs.items():
    Z = d.data[:, idx].astype(float); m = 150
    ss = np.random.SeedSequence(SEED + hash(label) % 1000).spawn(reps)
    rej = {k: 0 for k in tests}
    for i in range(reps):
        rng = np.random.default_rng(ss[i]); perm = rng.permutation(len(Z))
        Xa = Z[perm[:m]] @ rot(rng.uniform(0, 2*np.pi)).T
        Xb = Z[perm[m:2*m]] @ rot(rng.uniform(0, 2*np.pi)).T
        for k, fn in tests.items(): rej[k] += fn(Xa, Xb, rng) <= alpha
    p1[label] = {k: rej[k]/reps for k in tests}
    print("Part1", label, {k: round(v,3) for k,v in p1[label].items()})

# ---- Part 2: real two-sample, benign vs malignant (area/concavity)
Z = d.data[:, [3, 6]].astype(float); y = d.target
Zb, Zm = Z[y == 1], Z[y == 0]; rng = np.random.default_rng(SEED + 1)
p2 = {k: fn(Zb, Zm, rng) for k, fn in tests.items()}
print("Part2 p-values", {k: round(v,3) for k,v in p2.items()})

# ---- figure: Part-1 rejection rates, both feature pairs
fig, ax = plt.subplots(figsize=(7, 4.4))
meth = list(tests); x = np.arange(len(meth)); w = 0.38
hatches = ["///", "..."]
for j, (label, r) in enumerate(p1.items()):
    ax.bar(x + (j-0.5)*w, [r[m] for m in meth], w, label=label,
           hatch=hatches[j], edgecolor="black", linewidth=.5)
ax.axhline(alpha, ls=":", c="grey"); ax.text(len(meth)-1, alpha+.02, "nominal 0.05", fontsize=8, ha="right", color="grey")
ax.set_xticks(x); ax.set_xticklabels(meth, fontsize=9)
ax.set_ylabel("rejection rate under the rotation null"); ax.set_ylim(0, 1)
ax.set_title("Calibration on real marginals (breast-cancer): naive energy and\nenergy+ICA over-reject; the projected test controls the level")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig("/tmp/rd_out/fig_realdata.png", dpi=140); plt.close(fig)
print("figure saved")

# ---- CSV
import csv
with open("/tmp/rd_out/table_realdata.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["part","feature_pair/method"]+meth if False else ["setting"]+meth)
    for label in p1: w.writerow(["null-calib: "+label]+[round(p1[label][m],3) for m in meth])
    w.writerow(["two-sample p (benign vs malignant)"]+[round(p2[m],3) for m in meth])
print("csv saved")
