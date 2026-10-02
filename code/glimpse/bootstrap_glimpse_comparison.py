from project_paths import PATHS
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# INPUT
# ============================================================
B = Path(str(PATHS.work_root / 'test'))

OURS_FILE = B / "square_ours_metrics.csv"
GLIMPSE_FILE = B / "glimpse_test_metrics.csv"

OUT_SUMMARY = B / "square_method_summary.csv"
OUT_PAIRWISE = B / "square_pairwise_bootstrap.csv"
OUT_BYCOSMO = B / "square_method_by_cosmology.csv"

NBOOT = 50000
SEED = 12345

# ============================================================
# LOAD
# ============================================================
ours = pd.read_csv(OURS_FILE)
gl = pd.read_csv(GLIMPSE_FILE)

print("OURS:")
print(ours.groupby("method").size())
print()
print("GLIMPSE rows:", len(gl))

# GLIMPSE file already contains lambda=3 frozen.
gl = gl.copy()
gl["method"] = "GLIMPSE"

# Keep common columns.
cols = [
    "method",
    "c", "r", "p",
    "f_obs", "n_obs",
    "nmse", "pcc", "std_ratio",
]

ours = ours[cols].copy()
gl = gl[cols].copy()

df = pd.concat(
    [ours, gl],
    ignore_index=True
)

# ============================================================
# SANITY CHECKS
# ============================================================
methods = [
    "KS",
    "GLIMPSE",
    "STARLET_NOMASK",
    "STARLET_MASK",
]

expected_r = set(range(10, 30))
expected_c = set(range(3))
expected_p = set(range(8))

for method in methods:

    d = df[df["method"] == method]

    assert len(d) == 480, (
        method, len(d)
    )

    assert set(d["r"]) == expected_r
    assert set(d["c"]) == expected_c
    assert set(d["p"]) == expected_p

    # exactly one value per C/R/P
    assert not d.duplicated(
        ["c", "r", "p"]
    ).any()

print("\nSanity checks: OK")

# ============================================================
# GLOBAL SUMMARY
# ============================================================
summary = (
    df.groupby("method")
    .agg(
        n=("nmse", "size"),
        nmse_mean=("nmse", "mean"),
        nmse_std=("nmse", "std"),
        nmse_median=("nmse", "median"),
        pcc_mean=("pcc", "mean"),
        pcc_std=("pcc", "std"),
        std_ratio_mean=("std_ratio", "mean"),
    )
    .reindex(methods)
    .reset_index()
)

summary.to_csv(
    OUT_SUMMARY,
    index=False
)

print("\n" + "="*92)
print("GLOBAL TEST SUMMARY")
print("="*92)
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# BY COSMOLOGY
# ============================================================
bycosmo = (
    df.groupby(["method", "c"])
    .agg(
        n=("nmse", "size"),
        nmse_mean=("nmse", "mean"),
        pcc_mean=("pcc", "mean"),
        std_ratio_mean=("std_ratio", "mean"),
    )
    .reset_index()
)

bycosmo.to_csv(
    OUT_BYCOSMO,
    index=False
)

print("\n" + "="*92)
print("BY COSMOLOGY")
print("="*92)
print(
    bycosmo.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# WIDE TABLE:
# one row per C/R/P, one column per method
# ============================================================
wide_nmse = df.pivot(
    index=["c", "r", "p"],
    columns="method",
    values="nmse"
).reset_index()

wide_pcc = df.pivot(
    index=["c", "r", "p"],
    columns="method",
    values="pcc"
).reset_index()

for method in methods:
    assert wide_nmse[method].notna().all()
    assert wide_pcc[method].notna().all()

# ============================================================
# PAIR DEFINITIONS
#
# delta_abs = mean(NMSE_A - NMSE_B)
#
# positive -> A has larger error, therefore B is better.
#
# relative improvement of B over A:
#
# 100 * (mean_A - mean_B) / mean_A
#
# positive -> B improves on A.
# ============================================================
pairs = [
    ("KS", "STARLET_NOMASK"),
    ("KS", "STARLET_MASK"),
    ("STARLET_NOMASK", "STARLET_MASK"),
    ("KS", "GLIMPSE"),
    ("GLIMPSE", "STARLET_NOMASK"),
    ("GLIMPSE", "STARLET_MASK"),
]

# ============================================================
# BLOCK BOOTSTRAP
#
# Resampling unit = realization R.
#
# Each sampled R carries:
#   3 cosmologies x 8 patches = 24 paired cases.
#
# 20 realization blocks total.
# ============================================================
rng = np.random.default_rng(SEED)

realizations = np.array(
    sorted(wide_nmse["r"].unique())
)

assert len(realizations) == 20

# Precompute one block mean per realization and method.
block_nmse = {}

block_pcc = {}

for method in methods:

    block_nmse[method] = np.array([
        wide_nmse.loc[
            wide_nmse["r"] == r,
            method
        ].mean()
        for r in realizations
    ])

    block_pcc[method] = np.array([
        wide_pcc.loc[
            wide_pcc["r"] == r,
            method
        ].mean()
        for r in realizations
    ])

# Bootstrap indices:
# shape = [NBOOT, 20]
boot_idx = rng.integers(
    0,
    len(realizations),
    size=(NBOOT, len(realizations))
)

results = []

print("\n" + "="*110)
print("PAIRED REALIZATION-BLOCK BOOTSTRAP")
print("="*110)

for A, Bm in pairs:

    a = block_nmse[A]
    b = block_nmse[Bm]

    # Observed global means.
    mean_a = a.mean()
    mean_b = b.mean()

    # A - B:
    # positive means B has smaller NMSE.
    delta = mean_a - mean_b

    relative = (
        100.0 * delta / mean_a
    )

    # Bootstrap sample means.
    a_boot = a[boot_idx].mean(axis=1)
    b_boot = b[boot_idx].mean(axis=1)

    delta_boot = (
        a_boot - b_boot
    )

    rel_boot = (
        100.0
        * delta_boot
        / a_boot
    )

    delta_lo, delta_hi = np.percentile(
        delta_boot,
        [2.5, 97.5]
    )

    rel_lo, rel_hi = np.percentile(
        rel_boot,
        [2.5, 97.5]
    )

    # PCC difference B - A.
    pa = block_pcc[A]
    pb = block_pcc[Bm]

    pcc_delta = (
        pb.mean() - pa.mean()
    )

    pcc_boot = (
        pb[boot_idx].mean(axis=1)
        -
        pa[boot_idx].mean(axis=1)
    )

    pcc_lo, pcc_hi = np.percentile(
        pcc_boot,
        [2.5, 97.5]
    )

    # Case-level win counts are descriptive only.
    # They are NOT used for uncertainty.
    va = wide_nmse[A].to_numpy()
    vb = wide_nmse[Bm].to_numpy()

    wins_B = int(
        np.sum(vb < va)
    )

    ties = int(
        np.sum(
            np.isclose(
                vb, va,
                rtol=0.0,
                atol=1e-12
            )
        )
    )

    result = {
        "reference_A": A,
        "comparison_B": Bm,
        "nmse_A": mean_a,
        "nmse_B": mean_b,
        "abs_improvement_A_minus_B": delta,
        "abs_ci95_low": delta_lo,
        "abs_ci95_high": delta_hi,
        "relative_improvement_pct": relative,
        "relative_ci95_low_pct": rel_lo,
        "relative_ci95_high_pct": rel_hi,
        "pcc_B_minus_A": pcc_delta,
        "pcc_ci95_low": pcc_lo,
        "pcc_ci95_high": pcc_hi,
        "B_wins_cases": wins_B,
        "ties_cases": ties,
        "n_cases": len(va),
        "n_realization_blocks": len(realizations),
    }

    results.append(result)

    print()
    print(f"{Bm}  vs  {A}")
    print("-"*70)

    print(
        f"NMSE {A:15s} = "
        f"{mean_a:.9f}"
    )

    print(
        f"NMSE {Bm:15s} = "
        f"{mean_b:.9f}"
    )

    print(
        f"absolute improvement = "
        f"{delta:+.9f} "
        f"[{delta_lo:+.9f}, "
        f"{delta_hi:+.9f}]"
    )

    print(
        f"relative improvement = "
        f"{relative:+.3f}% "
        f"[{rel_lo:+.3f}%, "
        f"{rel_hi:+.3f}%]"
    )

    print(
        f"PCC change B-A       = "
        f"{pcc_delta:+.6f} "
        f"[{pcc_lo:+.6f}, "
        f"{pcc_hi:+.6f}]"
    )

    print(
        f"B lower NMSE in      = "
        f"{wins_B}/{len(va)} cases"
    )

# ============================================================
# SAVE PAIRWISE
# ============================================================
pairwise = pd.DataFrame(results)

pairwise.to_csv(
    OUT_PAIRWISE,
    index=False
)

# ============================================================
# EXTRA: ranking consistency per realization block
# ============================================================
print("\n" + "="*110)
print("REALIZATION-BLOCK CONSISTENCY")
print("="*110)

for A, Bm in pairs:

    a = block_nmse[A]
    b = block_nmse[Bm]

    wins = np.sum(
        b < a
    )

    print(
        f"{Bm:16s} vs {A:16s}: "
        f"{wins:2d}/{len(realizations)} "
        f"realization blocks with lower mean NMSE"
    )

# ============================================================
# FINAL TABLE
# ============================================================
print("\n" + "="*110)
print("FINAL COMPACT TABLE")
print("="*110)

ks_nmse = float(
    summary.loc[
        summary.method == "KS",
        "nmse_mean"
    ].iloc[0]
)

compact = summary[
    [
        "method",
        "nmse_mean",
        "pcc_mean"
    ]
].copy()

compact["delta_vs_KS_pct"] = (
    100.0
    * (
        ks_nmse
        - compact["nmse_mean"]
    )
    / ks_nmse
)

print(
    compact.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print()
print("Saved:")
print(" ", OUT_SUMMARY)
print(" ", OUT_PAIRWISE)
print(" ", OUT_BYCOSMO)
