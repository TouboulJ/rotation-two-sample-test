import os, csv, time
import numpy as np
import hashlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20250510
def _dh(x): return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 1000
QUICK = os.environ.get("Q", "0") == "1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_results")
os.makedirs(OUT, exist_ok=True)

def sc(full, quick): return quick if QUICK else full
def savetable(p, rows):
    with open(p, "w", newline="") as f: csv.writer(f).writerows(rows)

G = 96
GRID = np.linspace(0, 2 * np.pi, G, endpoint=False)
AX = np.c_[np.cos(GRID), np.sin(GRID)]
CONTRASTS = {"u4": lambda u: u ** 4, "u6": lambda u: u ** 6,
             "logcosh": lambda u: np.log(np.cosh(u)), "absu": lambda u: np.abs(u)}

def standardize(Z):
    Z = Z - Z.mean(0, keepdims=True); s = Z.std(0, keepdims=True); s[s < 1e-9] = 1.0
    return Z / s

def rot(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])

def profile(Zw, xi):
    M = xi(Zw @ AX.T); return M.mean(0), M - M.mean(0, keepdims=True)

def min_shift(kY, kX):
    def dist(a, b):
        cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), len(a))
        return (a @ a + b @ b - 2 * cc.max()) / len(a)
    return min(dist(kY, kX), dist(kY, kX[::-1].copy()))

def src_light(n, rng):
    return np.c_[rng.uniform(-np.sqrt(3), np.sqrt(3), n), rng.choice([-1.0, 1.0], n)]

def comp(kind, n, rng):
    phi = rng.uniform(0, 2 * np.pi); X = src_light(n, rng)
    if kind == "H0": Y = src_light(n, rng) @ rot(phi).T
    elif kind == "gauss": Y = rng.standard_normal((n, 2)) @ rot(phi).T
    else: Y = np.c_[rng.uniform(-np.sqrt(3), np.sqrt(3), n), rng.uniform(-np.sqrt(3), np.sqrt(3), n)] @ rot(phi).T
    return X, Y

def phipp_p(X, Y, B, rng, xi):
    kX, MX = profile(standardize(X), xi); kY, MY = profile(standardize(Y), xi)
    T = min_shift(kY, kX); n = len(MX); ge = 1
    for _ in range(B):
        gX = (MX.T @ rng.standard_normal(n)) / n; gY = (MY.T @ rng.standard_normal(len(MY))) / len(MY)
        ge += min_shift(kX + gY, kX + gX) >= T
    return ge / (B + 1)


def pdist(A, Bm):
    s = lambda M: (M ** 2).sum(1); D2 = s(A)[:, None] + s(Bm)[None, :] - 2 * A @ Bm.T
    return np.sqrt(np.maximum(D2, 0))
def energy(U, V):
    n, m = len(U), len(V); return 2 * pdist(U, V).mean() - pdist(U, U).sum() / (n * n) - pdist(V, V).sum() / (m * m)
def mmd(U, V, g):
    kf = lambda A, Bm: np.exp(-g * pdist(A, Bm) ** 2)
    return kf(U, U).mean() + kf(V, V).mean() - 2 * kf(U, V).mean()
def energy_naive_p(X, Y, B, rng):
    U, V = standardize(X), standardize(Y); Z = np.vstack([U, V]); n = len(U); obs = energy(U, V); ge = 1
    for _ in range(B):
        p = rng.permutation(len(Z)); ge += energy(Z[p][:n], Z[p][n:]) >= obs
    return ge / (B + 1)
def mmd_naive_p(X, Y, B, rng):
    U, V = standardize(X), standardize(Y); Z = np.vstack([U, V]); n = len(U)
    g = 1.0 / (2 * np.median(pdist(Z, Z)[np.triu_indices(len(Z), 1)]) ** 2 + 1e-9); obs = mmd(U, V, g); ge = 1
    for _ in range(B):
        p = rng.permutation(len(Z)); ge += mmd(Z[p][:n], Z[p][n:], g) >= obs
    return ge / (B + 1)
def canon_ica(Zw, rng, it=120):
    W = rng.standard_normal((2, 2)); U, _, Vt = np.linalg.svd(W); W = U @ Vt; n = len(Zw)
    for _ in range(it):
        WX = Zw @ W.T; g = np.tanh(WX); gp = 1 - g ** 2
        Wn = (g.T @ Zw) / n - (gp.mean(0)[:, None] * W); U, _, Vt = np.linalg.svd(Wn); W = U @ Vt
    S = Zw @ W.T; S = S[:, np.argsort(-np.abs((S ** 4).mean(0) - 3))]
    for j in range(2):
        if np.mean(S[:, j] ** 3) < 0: S[:, j] = -S[:, j]
    return S
def energy_ica_p(X, Y, B, rng):
    U = canon_ica(standardize(X), rng); V = canon_ica(standardize(Y), rng); Z = np.vstack([U, V]); n = len(U)
    obs = energy(U, V); ge = 1
    for _ in range(B):
        p = rng.permutation(len(Z)); ge += energy(Z[p][:n], Z[p][n:]) >= obs
    return ge / (B + 1)

ANG = np.linspace(0, 2 * np.pi, 12, endpoint=False)
def energy_rotalign_p(X, Y, B, rng):
    U, V = standardize(X), standardize(Y)
    def stat(U, V):
        Un2 = (U ** 2).sum(1); Vn2 = (V ** 2).sum(1)
        within = pdist(U, U).sum() / (len(U) ** 2) + pdist(V, V).sum() / (len(V) ** 2)
        best = np.inf
        for a in ANG:
            RV = V @ rot(a).T
            cross = np.sqrt(np.maximum(Un2[:, None] + Vn2[None, :] - 2 * U @ RV.T, 0))
            e = 2 * cross.mean() - within
            if e < best: best = e
        return best
    obs = stat(U, V); Z = np.vstack([U, V]); n = len(U); ge = 1
    for _ in range(B):
        p = rng.permutation(len(Z)); ge += stat(Z[p][:n], Z[p][n:]) >= obs
    return ge / (B + 1)


def study_calibration():
    B = sc(199, 79); alpha = 0.05; reps = sc(60, 15); NS = sc([200, 500, 1000], [150, 300])
    rows = [["contrast", "n", "H0_level", "gauss_power", "unif_power"]]; curve = {}
    for cname in ["u4", "u6"]:
        xi = CONTRASTS[cname]
        for n in NS:
            r = {}
            for kind in ["H0", "gauss", "unif"]:
                ss = np.random.SeedSequence(SEED + _dh(kind + cname) + n).spawn(reps); k = 0
                for i in range(reps):
                    rng = np.random.default_rng(ss[i]); X, Y = comp(kind, n, rng); k += phipp_p(X, Y, B, rng, xi) <= alpha
                r[kind] = k / reps
            rows.append([cname, n, round(r["H0"], 3), round(r["gauss"], 3), round(r["unif"], 3)]); curve[(cname, n)] = r
    savetable(os.path.join(OUT, "table_calibration.csv"), rows)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    style = {"u4": ("o", "#1f77b4"), "u6": ("^", "#d62728")}
    for cname in ["u4", "u6"]:
        mk, cl = style[cname]
        ax.plot(NS, [curve[(cname, n)]["gauss"] for n in NS], marker=mk, ls="-", color=cl, label="%s gauss-power" % cname)
        ax.plot(NS, [curve[(cname, n)]["H0"] for n in NS], marker=mk, ls="--", color=cl, label="%s H0-level" % cname)
    ax.axhline(alpha, ls=":", c="grey"); ax.set_xlabel("n"); ax.set_ylabel("rate"); ax.set_ylim(-.03, 1.03)
    ax.set_title("calibration, isotropic light-tailed regime"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_calibration.png"), dpi=130); plt.close(fig)


def study_hermite():
    from numpy.polynomial.hermite_e import hermegauss
    nodes, wts = hermegauss(80); w = wts / np.sqrt(2 * np.pi)
    def He(j, x):
        H = [np.ones_like(x), x]
        for k in range(1, j): H.append(x * H[-1] - k * H[-2])
        return H[j]
    def fact(j):
        r = 1
        for k in range(2, j + 1): r *= k
        return r
    rows = [["contrast", "c0", "c2", "c4", "c6", "supra_quadratic_variance", "data_factor_vs_u4"]]
    supra = {}
    for name, xi in CONTRASTS.items():
        fx = xi(nodes); c = {j: np.sum(w * fx * He(j, nodes)) / fact(j) for j in [0, 2, 4, 6]}
        var = np.sum(w * fx * fx) - c[0] ** 2
        supra[name] = var - 2 * c[2] ** 2
        rows.append([name, round(c[0], 3), round(c[2], 3), round(c[4], 4), round(c[6], 5), round(supra[name], 4), None])
    for r in rows[1:]: r[-1] = round(supra["u4"] / supra[r[0]], 1)
    savetable(os.path.join(OUT, "table_hermite_coefficients.csv"), rows)


def src_skew(n, rng): return rng.exponential(1, (n, 2)) - 1.0

def comp_diss(alt, n, rng):
    phi = rng.uniform(0, 2 * np.pi); X = src_light(n, rng)
    if alt == "H0": Y = src_light(n, rng) @ rot(phi).T
    elif alt == "kurt": Y = rng.standard_normal((n, 2)) @ rot(phi).T
    else: Y = src_skew(n, rng) @ rot(phi).T
    return X, Y

def study_dissociation():
    B, alpha, reps, n = 99, 0.05, sc(60, 20), sc(400, 250)
    u3 = lambda u: u ** 3; u4 = lambda u: u ** 4
    def rule(name, X, Y, rng):
        if name == "u3": return phipp_p(X, Y, B, rng, u3) <= alpha
        if name == "u4": return phipp_p(X, Y, B, rng, u4) <= alpha
        return min(phipp_p(X, Y, B, rng, u3), phipp_p(X, Y, B, rng, u4)) <= alpha / 2
    methods = ["u3", "u4", "bank"]; res = {m: {} for m in methods}
    for m in methods:
        for alt in ["H0", "kurt", "skew"]:
            ss = np.random.SeedSequence(SEED + _dh(m + alt)).spawn(reps); k = 0
            for r in range(reps):
                rng = np.random.default_rng(ss[r]); X, Y = comp_diss(alt, n, rng); k += rule(m, X, Y, rng)
            res[m][alt] = k / reps
    rows = [["method", "H0", "kurt_alt", "skew_alt"]]
    for m in methods: rows.append([m, round(res[m]["H0"], 3), round(res[m]["kurt"], 3), round(res[m]["skew"], 3)])
    savetable(os.path.join(OUT, "table_dissociation.csv"), rows)
    lab = {"u3": "u^3 (skewness)", "u4": "u^4 (kurtosis)", "bank": "bank {u3,u4} Bonf."}
    x = np.arange(len(methods)); wdt = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - wdt, [res[m]["H0"] for m in methods], wdt, label="H0 (should be <=.05)", color="#c44", hatch="///", edgecolor="black", linewidth=.5)
    ax.bar(x, [res[m]["kurt"] for m in methods], wdt, label="power vs kurtosis alt.", color="#4488cc", hatch="...", edgecolor="black", linewidth=.5)
    ax.bar(x + wdt, [res[m]["skew"] for m in methods], wdt, label="power vs skewness alt.", color="#55aa66", hatch="xxx", edgecolor="black", linewidth=.5)
    ax.axhline(alpha, ls="--", c="grey"); ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in methods], fontsize=9)
    ax.set_ylabel("rejection rate"); ax.set_ylim(0, 1.05)
    ax.set_title("no single contrast dominates; a corrected bank handles both (n=%d)" % n)
    ax.legend(fontsize=8, loc="upper center"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dissociation.png"), dpi=130); plt.close(fig)


def study_power():
    n, reps, B, alpha = sc(200, 150), sc(80, 25), 99, 0.05
    METHODS = {"phiPP": lambda X, Y, B, rng: phipp_p(X, Y, B, rng, CONTRASTS["u4"]),
               "energy_ICA": energy_ica_p, "energy_rotalign": energy_rotalign_p,
               "energy_naive": energy_naive_p, "mmd_naive": mmd_naive_p}
    order = ["phiPP", "energy_ICA", "energy_rotalign", "energy_naive", "mmd_naive"]
    res = {m: {} for m in METHODS}
    for m in order:
        for kind in ["H0", "gauss", "unif"]:
            ss = np.random.SeedSequence(SEED + _dh(kind + m)).spawn(reps); k = 0
            for r in range(reps):
                rng = np.random.default_rng(ss[r]); X, Y = comp(kind, n, rng); k += METHODS[m](X, Y, B, rng) <= alpha
            p = k / reps; res[m][kind] = (p, (p * (1 - p) / reps) ** 0.5)
    rows = [["method", "H0_level", "H0_se", "power_gauss", "gauss_se", "power_unif", "unif_se"]]
    for m in order:
        rows.append([m] + [round(v, 3) for kind in ["H0", "gauss", "unif"] for v in res[m][kind]])
    savetable(os.path.join(OUT, "table_power_comparison.csv"), rows)
    lab = {"phiPP": "phi-PP\n(valid)", "energy_ICA": "energy+ICA\n(valid)",
           "energy_rotalign": "energy+rot.\n(INVALID)", "energy_naive": "energy naive\n(INVALID)", "mmd_naive": "MMD naive"}
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for off, kind, col, hat, labk in [(-w, "H0", "#c44", "///", "H0 rotation null (valid <=.05)"),
                                      (0, "gauss", "#4488cc", "...", "power vs Gaussian"),
                                      (w, "unif", "#55aa66", "xxx", "power vs uniform")]:
        ax.bar(x + off, [res[m][kind][0] for m in order], w, yerr=[res[m][kind][1] for m in order],
               capsize=2, label=labk, color=col, hatch=hat, edgecolor="black", linewidth=.5)
    ax.axhline(alpha, ls="--", c="grey", lw=1); ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in order], fontsize=8)
    ax.set_ylabel("rejection rate"); ax.set_ylim(0, 1.08)
    ax.set_title("validity and power (n=%d, %d reps, +/-1 s.e.): naive aligned tests over-reject" % (n, reps))
    ax.legend(fontsize=8, loc="upper center", ncol=3); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_power_comparison.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    GENERATES = {
        "calibration":      ["fig_calibration.png", "table_calibration.csv"],
        "power_comparison": ["fig_power_comparison.png", "table_power_comparison.csv"],
        "hermite":          ["table_hermite_coefficients.csv"],
        "dissociation":     ["fig_dissociation.png", "table_dissociation.csv"],
    }
    for name, fn in [("calibration", study_calibration), ("power_comparison", study_power),
                     ("hermite", study_hermite), ("dissociation", study_dissociation)]:
        t0 = time.time()
        try:
            fn(); print("[ok]   %-18s %5.0fs   ->  %s" % (name, time.time() - t0, ", ".join(GENERATES[name])))
        except Exception as e:
            print("[FAIL] %-18s %s" % (name, e))
    print("results in", OUT)
