from project_paths import PATHS
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================
COSMOSTAT_ROOT = PATHS.mcalens_root

if str(COSMOSTAT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(COSMOSTAT_ROOT),
    )

from pycs.astro.wl.mass_mapping import (
    massmap2d,
    shear_data,
)


# =============================================================================
# FIXED SOURCE VERSION
# =============================================================================
EXPECTED_COMMIT = (
    "7ae457db1697ac966fc6e739c00b91fa6a47ec2c"
)

try:
    CURRENT_COMMIT = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(COSMOSTAT_ROOT),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )
except Exception:
    CURRENT_COMMIT = "UNKNOWN"

print()
print("=" * 110)
print("MCALENS VALIDATION MAP GENERATION")
print("=" * 110)

print(
    "CosmoStat source :",
    COSMOSTAT_ROOT,
)

print(
    "commit           :",
    CURRENT_COMMIT,
)

if (
    CURRENT_COMMIT != "UNKNOWN"
    and CURRENT_COMMIT != EXPECTED_COMMIT
):
    raise RuntimeError(
        "\nWrong CosmoStat commit.\n"
        f"Current : {CURRENT_COMMIT}\n"
        f"Expected: {EXPECTED_COMMIT}\n"
    )


# =============================================================================
# DATA
# =============================================================================
BASE = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

OUT = Path(
    str(PATHS.work_root / 'validation/mcalens')
)

RECON_DIR = (
    OUT / "recon"
)

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_FILE = (
    OUT
    / "mcalens_validation_metrics.csv"
)

SUMMARY_FILE = (
    OUT
    / "mcalens_validation_summary.csv"
)

BYCOSMO_FILE = (
    OUT
    / "mcalens_validation_by_cosmology.csv"
)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================
N = 176
NSCALES = 4

NITER = 100
NSIGMA = 5.0

INPAINT = False
INP_NITER = 20

ONLY_POS = True
FIRST_DETECT_SCALE = 1

BMODE = True

PROPAGATE_NOISE = False
NREA = None

PATCH_ROWS = [
    (0, 176),
    (176, 352),
    (352, 528),
    (528, 704),
    (704, 880),
    (880, 1056),
    (1056, 1232),
    (1232, 1408),
]


# =============================================================================
# REFERENCE VALUES FROM FROZEN VALIDATION
# =============================================================================
EXPECTED_GLOBAL_NMSE = 0.658007
EXPECTED_GLOBAL_PCC = 0.591189
EXPECTED_GLOBAL_STD = 0.691173

# Previous C0/R0/P0 MCALens pilot with exactly this setup.
EXPECTED_PILOT_NMSE = 0.670929329
EXPECTED_PILOT_PCC = 0.585494

# Loose enough for insignificant numerical/platform differences,
# tight enough to catch a convention error.
PILOT_TOL = 3.0e-3
GLOBAL_TOL = 3.0e-3


# =============================================================================
# LOAD FIXED MASK AND P_KAPPA
# =============================================================================
mask_full = np.load(
    MASK_FILE
).astype(bool)

if mask_full.shape != (
    1424,
    176,
):
    raise RuntimeError(
        f"Unexpected mask shape: {mask_full.shape}"
    )

pk = np.load(
    PK_FILE
).astype(
    np.float64
)

print(
    "P_kappa shape   :",
    pk.shape,
)

print(
    "P_kappa min/max :",
    float(np.min(pk)),
    float(np.max(pk)),
)

print(
    "niter           :",
    NITER,
)

print(
    "Nsigma          :",
    NSIGMA,
)

print(
    "nscales         :",
    NSCALES,
)

print(
    "Inpaint         :",
    INPAINT,
)

print(
    "OnlyPos         :",
    ONLY_POS,
)

print(
    "FirstDetectScale:",
    FIRST_DETECT_SCALE,
)

print(
    "Bmode           :",
    BMODE,
)

print()


# =============================================================================
# MASS-SHEET-INVARIANT METRICS
# =============================================================================
def metrics(
    rec,
    truth,
    mask,
):
    m = mask.astype(bool)

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[m]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[m]

    r -= np.mean(r)
    t -= np.mean(t)

    err = r - t

    mse = float(
        np.mean(
            err**2
        )
    )

    nmse = float(
        mse
        / np.mean(
            t**2
        )
    )

    pcc = float(
        np.corrcoef(
            r,
            t,
        )[0, 1]
    )

    std_ratio = float(
        np.std(r)
        / np.std(t)
    )

    return (
        mse,
        nmse,
        pcc,
        std_ratio,
    )


# =============================================================================
# LOAD ONE FAIR CASE
# =============================================================================
def load_case(
    c,
    r,
    p,
):
    fn = (
        BASE
        / f"C{c}_R{r}.npz"
    )

    if not fn.exists():
        raise FileNotFoundError(
            fn
        )

    with np.load(fn) as z:

        truth_full = np.asarray(
            z["truth"],
            dtype=np.float64,
        )

        gamma_full = np.asarray(
            z["gamma"],
            dtype=np.complex128,
        )

        if "sigma_n" in z:
            sigma_n = float(
                np.asarray(
                    z["sigma_n"]
                ).reshape(-1)[0]
            )
        else:
            # FAIR physical shape-noise benchmark:
            # sigma_e=0.4, ng=30 arcmin^-2,
            # 2 arcmin pixels.
            sigma_n = (
                0.025819888974716113
            )

    r0, r1 = (
        PATCH_ROWS[p]
    )

    truth = truth_full[
        r0:r1,
        :
    ].copy()

    mask = mask_full[
        r0:r1,
        :
    ].copy()

    # Zero-filled observed shear.
    gamma = (
        gamma_full[
            r0:r1,
            :
        ].copy()
        * mask
    )

    if truth.shape != (
        N,
        N,
    ):
        raise RuntimeError(
            truth.shape
        )

    if gamma.shape != (
        N,
        N,
    ):
        raise RuntimeError(
            gamma.shape
        )

    return (
        truth,
        gamma,
        mask,
        sigma_n,
    )


# =============================================================================
# BUILD MCALENS INPUT
# =============================================================================
def make_shear_data(
    gamma,
    mask,
    sigma_n,
):
    d = shear_data()

    # Same shear convention as the FAIR forward operator:
    #
    #     gamma = gamma1 + i gamma2
    #
    d.g1 = np.asarray(
        gamma.real,
        dtype=np.float64,
    )

    d.g2 = np.asarray(
        gamma.imag,
        dtype=np.float64,
    )

    d.mask = np.asarray(
        mask,
        dtype=np.int32,
    )

    # MCALens defines:
    #
    #   Ncov / 2 = variance of either g1 or g2
    #
    # therefore observed pixels need
    #
    #   Ncov = 2 sigma_n^2.
    #
    # Missing pixels are initially zero.
    # _prepare_data() converts Ncov==0 to 1e9.
    Ncov = np.zeros(
        (
            N,
            N,
        ),
        dtype=np.float64,
    )

    Ncov[
        mask
    ] = (
        2.0
        * sigma_n**2
    )

    d.Ncov = Ncov

    return d


# =============================================================================
# RUN ONE MCALENS CASE
# =============================================================================
def run_mcalens(
    gamma,
    mask,
    sigma_n,
):
    d = make_shear_data(
        gamma,
        mask,
        sigma_n,
    )

    M = massmap2d(
        name="mass"
    )

    M.init_massmap(
        N,
        N,
        ns=NSCALES,
    )

    M.DEF_niter = NITER
    M.Verbose = False

    t0 = (
        time.perf_counter()
    )

    (
        e_rec,
        b_rec,
        e_sparse,
        b_sparse,
    ) = M.sparse_wiener_filtering(
        d,
        pk.copy(),
        niter=NITER,
        Nsigma=NSIGMA,
        Inpaint=INPAINT,
        InpNiter=INP_NITER,
        OnlyPos=ONLY_POS,
        FirstDetectScale=FIRST_DETECT_SCALE,
        Bmode=BMODE,
        ktr=None,
        PropagateNoise=PROPAGATE_NOISE,
        Nrea=NREA,
    )

    runtime = (
        time.perf_counter()
        - t0
    )

    e_rec = np.asarray(
        e_rec,
        dtype=np.float64,
    )

    b_rec = np.asarray(
        b_rec,
        dtype=np.float64,
    )

    e_sparse = np.asarray(
        e_sparse,
        dtype=np.float64,
    )

    b_sparse = np.asarray(
        b_sparse,
        dtype=np.float64,
    )

    if e_rec.shape != (
        N,
        N,
    ):
        raise RuntimeError(
            f"Unexpected MCALens output "
            f"shape: {e_rec.shape}"
        )

    # Gaussian component follows directly from
    # total = Gaussian + sparse.
    e_gaussian = (
        e_rec
        - e_sparse
    )

    b_gaussian = (
        b_rec
        - b_sparse
    )

    active = np.asarray(
        M.WT_ActiveCoef
    )

    active_count = float(
        np.sum(
            active
        )
    )

    return {
        "mcalens": e_rec,

        "bmode": b_rec,

        "sparse_e": e_sparse,

        "sparse_b": b_sparse,

        "gaussian_e": e_gaussian,

        "gaussian_b": b_gaussian,

        "active_count": (
            active_count
        ),

        "runtime": (
            runtime
        ),
    }


# =============================================================================
# SAVE / LOAD ONE RECONSTRUCTION
# =============================================================================
def recon_filename(
    c,
    r,
    p,
):
    return (
        RECON_DIR
        / f"C{c}_R{r}_P{p}.npz"
    )


def save_reconstruction(
    fn,
    result,
    c,
    r,
    p,
    sigma_n,
):
    np.savez_compressed(
        fn,

        mcalens=np.asarray(
            result["mcalens"],
            dtype=np.float32,
        ),

        bmode=np.asarray(
            result["bmode"],
            dtype=np.float32,
        ),

        sparse_e=np.asarray(
            result["sparse_e"],
            dtype=np.float32,
        ),

        sparse_b=np.asarray(
            result["sparse_b"],
            dtype=np.float32,
        ),

        gaussian_e=np.asarray(
            result["gaussian_e"],
            dtype=np.float32,
        ),

        gaussian_b=np.asarray(
            result["gaussian_b"],
            dtype=np.float32,
        ),

        active_count=np.float64(
            result[
                "active_count"
            ]
        ),

        runtime=np.float64(
            result[
                "runtime"
            ]
        ),

        sigma_n=np.float64(
            sigma_n
        ),

        cosmology=np.int32(
            c
        ),

        realization=np.int32(
            r
        ),

        patch=np.int32(
            p
        ),

        nsigma=np.float64(
            NSIGMA
        ),

        niter=np.int32(
            NITER
        ),
    )


def load_reconstruction(
    fn,
):
    with np.load(fn) as z:

        if "mcalens" not in z:
            raise RuntimeError(
                f"{fn} has no "
                f"'mcalens' key."
            )

        result = {
            "mcalens": np.asarray(
                z["mcalens"],
                dtype=np.float64,
            ),

            "active_count": float(
                np.asarray(
                    z[
                        "active_count"
                    ]
                )
            )
            if "active_count" in z
            else np.nan,

            "runtime": float(
                np.asarray(
                    z["runtime"]
                )
            )
            if "runtime" in z
            else np.nan,
        }

    return result


# =============================================================================
# RUN / RESUME ALL 240 VALIDATION MAPS
# =============================================================================
rows = []

case_index = 0

for c in range(3):

    for r in range(10):

        for p in range(8):

            case_index += 1

            (
                truth,
                gamma,
                mask,
                sigma_n,
            ) = load_case(
                c,
                r,
                p,
            )

            fn = recon_filename(
                c,
                r,
                p,
            )

            if fn.exists():

                result = (
                    load_reconstruction(
                        fn
                    )
                )

                source = "cached"

            else:

                result = (
                    run_mcalens(
                        gamma,
                        mask,
                        sigma_n,
                    )
                )

                save_reconstruction(
                    fn,
                    result,
                    c,
                    r,
                    p,
                    sigma_n,
                )

                source = "new"

            (
                mse,
                nmse,
                pcc,
                std_ratio,
            ) = metrics(
                result[
                    "mcalens"
                ],
                truth,
                mask,
            )

            row = {
                "method": "MCALENS",

                "c": c,
                "r": r,
                "p": p,

                "nsigma": NSIGMA,

                "mse": mse,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": (
                    std_ratio
                ),

                "active_count": (
                    result[
                        "active_count"
                    ]
                ),

                "runtime": (
                    result[
                        "runtime"
                    ]
                ),

                "f_obs": float(
                    np.mean(
                        mask
                    )
                ),

                "sigma_n": (
                    sigma_n
                ),

                "source": source,
            }

            rows.append(
                row
            )

            # -------------------------------------------------
            # CRITICAL EARLY PILOT CHECK
            # -------------------------------------------------
            if (
                c == 0
                and r == 0
                and p == 0
            ):

                print()
                print("=" * 110)
                print(
                    "C0 / R0 / P0 EARLY SANITY CHECK"
                )
                print("=" * 110)

                print(
                    f"NMSE      : "
                    f"{nmse:.9f}"
                )

                print(
                    f"expected  : "
                    f"{EXPECTED_PILOT_NMSE:.9f}"
                )

                print(
                    f"PCC       : "
                    f"{pcc:.9f}"
                )

                print(
                    f"expected  : "
                    f"{EXPECTED_PILOT_PCC:.9f}"
                )

                print(
                    f"std ratio : "
                    f"{std_ratio:.9f}"
                )

                print(
                    f"active    : "
                    f"{result['active_count']:.1f}"
                )

                print(
                    f"source    : "
                    f"{source}"
                )

                if (
                    abs(
                        nmse
                        - EXPECTED_PILOT_NMSE
                    )
                    > PILOT_TOL
                    or
                    abs(
                        pcc
                        - EXPECTED_PILOT_PCC
                    )
                    > PILOT_TOL
                ):
                    raise RuntimeError(
                        "\nMCALens convention mismatch "
                        "on C0/R0/P0.\n"
                        "Stopping before generating "
                        "the full validation set."
                    )

                print()
                print(
                    "PILOT CHECK PASSED"
                )
                print()

            print(
                f"{case_index:3d}/240  "
                f"C{c} R{r:02d} P{p}  "
                f"NMSE={nmse:.6f}  "
                f"PCC={pcc:.6f}  "
                f"std={std_ratio:.6f}  "
                f"[{source}]",
                flush=True,
            )

            # Resume-safe metrics.
            pd.DataFrame(
                rows
            ).to_csv(
                METRICS_FILE,
                index=False,
            )


# =============================================================================
# FINAL TABLE
# =============================================================================
tab = pd.DataFrame(
    rows
)

tab = (
    tab
    .sort_values(
        [
            "c",
            "r",
            "p",
        ]
    )
    .reset_index(
        drop=True
    )
)

if len(tab) != 240:
    raise RuntimeError(
        f"Expected 240 cases, "
        f"got {len(tab)}."
    )

tab.to_csv(
    METRICS_FILE,
    index=False,
)


# =============================================================================
# GLOBAL SUMMARY
# =============================================================================
summary = pd.DataFrame(
    [
        {
            "method": (
                "MCALENS"
            ),

            "nsigma": (
                NSIGMA
            ),

            "n": len(tab),

            "nmse_mean": float(
                tab.nmse.mean()
            ),

            "nmse_std": float(
                tab.nmse.std()
            ),

            "nmse_median": float(
                tab.nmse.median()
            ),

            "mse_mean": float(
                tab.mse.mean()
            ),

            "pcc_mean": float(
                tab.pcc.mean()
            ),

            "std_ratio_mean": float(
                tab.std_ratio.mean()
            ),

            "active_mean": float(
                tab.active_count.mean()
            ),

            "active_median": float(
                tab.active_count.median()
            ),

            "runtime_mean": float(
                tab.runtime.mean()
            ),
        }
    ]
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)

print()
print("=" * 125)
print("GLOBAL VALIDATION SUMMARY")
print("=" * 125)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =============================================================================
# BY COSMOLOGY
# =============================================================================
byc = (
    tab
    .groupby(
        "c"
    )
    .agg(
        n=(
            "nmse",
            "size",
        ),

        nmse_mean=(
            "nmse",
            "mean",
        ),

        nmse_std=(
            "nmse",
            "std",
        ),

        nmse_median=(
            "nmse",
            "median",
        ),

        pcc_mean=(
            "pcc",
            "mean",
        ),

        std_ratio_mean=(
            "std_ratio",
            "mean",
        ),

        active_mean=(
            "active_count",
            "mean",
        ),
    )
    .reset_index()
)

byc.to_csv(
    BYCOSMO_FILE,
    index=False,
)

print()
print("=" * 110)
print("BY COSMOLOGY")
print("=" * 110)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =============================================================================
# FINAL FROZEN BENCHMARK CHECK
# =============================================================================
got_nmse = float(
    summary.iloc[0][
        "nmse_mean"
    ]
)

got_pcc = float(
    summary.iloc[0][
        "pcc_mean"
    ]
)

got_std = float(
    summary.iloc[0][
        "std_ratio_mean"
    ]
)

print()
print("=" * 110)
print("FROZEN BENCHMARK CHECK")
print("=" * 110)

print(
    f"NMSE      : "
    f"{got_nmse:.9f}   "
    f"expected ~{EXPECTED_GLOBAL_NMSE:.9f}"
)

print(
    f"PCC       : "
    f"{got_pcc:.9f}   "
    f"expected ~{EXPECTED_GLOBAL_PCC:.9f}"
)

print(
    f"std ratio : "
    f"{got_std:.9f}   "
    f"expected ~{EXPECTED_GLOBAL_STD:.9f}"
)

if (
    abs(
        got_nmse
        - EXPECTED_GLOBAL_NMSE
    )
    > GLOBAL_TOL
    or
    abs(
        got_pcc
        - EXPECTED_GLOBAL_PCC
    )
    > GLOBAL_TOL
):

    raise RuntimeError(
        "\nFinal MCALens validation metrics "
        "do not reproduce the frozen benchmark."
    )

print()
print(
    "FROZEN MCALENS VALIDATION CHECK PASSED"
)


# =============================================================================
# VERIFY FUSION SCRIPT EXPECTED FILES
# =============================================================================
missing = []

for c in range(3):
    for r in range(10):
        for p in range(8):

            fn = recon_filename(
                c,
                r,
                p,
            )

            if not fn.exists():
                missing.append(
                    fn
                )

if missing:
    raise RuntimeError(
        f"{len(missing)} reconstruction "
        "files are missing."
    )

print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)

print(
    "Reconstructions:"
)

print(
    RECON_DIR
)

print()
print(
    "Example:"
)

print(
    RECON_DIR
    / "C0_R0_P0.npz"
)

print()
print(
    "Metrics:"
)

print(
    METRICS_FILE
)

print(
    SUMMARY_FILE
)

print(
    BYCOSMO_FILE
)

print()
print(
    "240/240 VALIDATION MAPS READY "
    "FOR MULTISCALE FUSION"
)
