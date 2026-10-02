from project_paths import PATHS
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import numpy as np
import pandas as pd

# ============================================================
# FROZEN TEST CONFIGURATION
# ============================================================
BASE = Path(str(PATHS.work_root / 'test/fair_npz'))

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

# IMPORTANT:
# This spectrum was built exclusively from validation:
# C0-C2, R0-R9, P0-P7.
PK_FILE = Path(
    str(PATHS.prior_file)
)

OUT = Path(
    str(PATHS.work_root / 'test/mcalens')
)
OUT.mkdir(parents=True, exist_ok=True)

RECON_DIR = OUT / "recon"
RECON_DIR.mkdir(parents=True, exist_ok=True)

CSV = OUT / "mcalens_test_metrics.csv"

COSMOS = range(3)
REALS = range(10, 30)       # TEST ONLY
PATCHES = range(8)

NSIGMA = 5                  # FROZEN
NITER = 100
NSCALES = 4

INPAINT = False
ONLY_POS = True
FIRST_DETECT_SCALE = 1
BMODE = True

NWORKERS = int(
    os.environ.get(
        "MCALENS_WORKERS",
        "8",
    )
)

NBOOT = 30000
BOOT_SEED = 12345

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


# ============================================================
# METRICS
# ============================================================
def ms_metrics(rec, truth, mask):
    """
    Mass-sheet-invariant metrics on observed pixels.
    """
    mask = np.asarray(mask, dtype=bool)

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[mask]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[mask]

    r -= r.mean()
    t -= t.mean()

    err = r - t

    mse = np.mean(err**2)
    nmse = mse / np.mean(t**2)

    pcc = np.corrcoef(
        r,
        t,
    )[0, 1]

    std_ratio = (
        np.std(r)
        / np.std(t)
    )

    return {
        "mse": float(mse),
        "nmse": float(nmse),
        "pcc": float(pcc),
        "std_ratio": float(std_ratio),
    }


# ============================================================
# ONE PATCH
# ============================================================
def run_one(job):
    c, r, p = job

    # Import in worker.
    from pycs.astro.wl.mass_mapping import (
        massmap2d,
        shear_data,
    )

    cache = (
        BASE
        / f"C{c}_R{r}.npz"
    )

    if not cache.exists():
        raise FileNotFoundError(cache)

    with np.load(cache) as z:

        truth_full = np.asarray(
            z["truth"],
            dtype=np.float64,
        )

        gamma_full = np.asarray(
            z["gamma"],
            dtype=np.complex128,
        )

        sigma_n = float(
            z["sigma_n"]
        )

        omega_m = float(
            z["omega_m"]
        )

    mask_full = np.load(
        MASK_FILE
    ).astype(bool)

    pk = np.load(
        PK_FILE
    ).astype(np.float64)

    r0, r1 = PATCH_ROWS[p]

    truth = truth_full[
        r0:r1,
        :
    ].copy()

    gamma = gamma_full[
        r0:r1,
        :
    ].copy()

    mask = mask_full[
        r0:r1,
        :
    ].copy()

    assert truth.shape == (176, 176)
    assert gamma.shape == (176, 176)
    assert mask.shape == (176, 176)

    # Same zero-filled observed shear used by all
    # square-field comparators.
    gamma_obs = (
        gamma
        * mask
    )

    g1 = gamma_obs.real.copy()
    g2 = gamma_obs.imag.copy()

    # --------------------------------------------------------
    # MCALens data object
    # --------------------------------------------------------
    d = shear_data()

    d.g1 = g1
    d.g2 = g2

    Ncov = np.zeros(
        (176, 176),
        dtype=np.float64,
    )

    # MCALens:
    # variance per shear component = Ncov / 2
    Ncov[mask] = (
        2.0
        * sigma_n**2
    )

    d.Ncov = Ncov

    d.mask = mask.astype(
        np.float64
    )

    # --------------------------------------------------------
    # Frozen MCALens
    # --------------------------------------------------------
    M = massmap2d(
        name="mass"
    )

    M.init_massmap(
        176,
        176,
        ns=NSCALES,
    )

    M.DEF_niter = NITER
    M.Verbose = False

    t0 = time.perf_counter()

    (
        kappa,
        kb,
        ksparse,
        ksparse_b,
    ) = M.sparse_wiener_filtering(
        d,
        pk.copy(),
        niter=NITER,
        Nsigma=NSIGMA,
        Inpaint=INPAINT,
        OnlyPos=ONLY_POS,
        FirstDetectScale=FIRST_DETECT_SCALE,
        Bmode=BMODE,
        ktr=None,
        PropagateNoise=False,
    )

    runtime = (
        time.perf_counter()
        - t0
    )

    kgaussian = (
        kappa
        - ksparse
    )

    met = ms_metrics(
        kappa,
        truth,
        mask,
    )

    # --------------------------------------------------------
    # Save reconstruction
    # --------------------------------------------------------
    recon_file = (
        RECON_DIR
        / f"C{c}_R{r}_P{p}.npz"
    )

    np.savez_compressed(
        recon_file,

        truth=truth.astype(
            np.float32
        ),

        mask=mask,

        gamma=gamma_obs.astype(
            np.complex64
        ),

        mcalens=kappa.astype(
            np.float32
        ),

        bmode=kb.astype(
            np.float32
        ),

        sparse=ksparse.astype(
            np.float32
        ),

        sparse_b=ksparse_b.astype(
            np.float32
        ),

        gaussian=kgaussian.astype(
            np.float32
        ),

        sigma_n=sigma_n,

        omega_m=omega_m,

        Nsigma=NSIGMA,
    )

    return {
        "method": "MCALENS",
        "c": c,
        "r": r,
        "p": p,

        "omega_m": omega_m,
        "f_obs": float(mask.mean()),
        "n_obs": int(mask.sum()),

        "Nsigma": NSIGMA,
        "niter": NITER,

        "mse": met["mse"],
        "nmse": met["nmse"],
        "pcc": met["pcc"],
        "std_ratio": met["std_ratio"],

        "runtime": float(runtime),

        "active_coeffs": int(
            np.sum(
                M.WT_ActiveCoef
            )
        ),

        "bmode_std": float(
            np.std(kb)
        ),

        "sparse_std": float(
            np.std(ksparse)
        ),

        "gaussian_std": float(
            np.std(kgaussian)
        ),

        "finite": bool(
            np.isfinite(kappa).all()
        ),

        "recon_file": str(
            recon_file
        ),
    }


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================
def ci(x):
    return (
        float(
            np.percentile(x, 2.5)
        ),
        float(
            np.percentile(x, 97.5)
        ),
    )


def paired_block_bootstrap(
    base,
    mc,
    method_name,
):
    """
    Resample the 20 held-out realization indices R10-R29.

    Each realization block contains:
        3 cosmologies x 8 patches = 24 cases.
    """
    left = base[
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    left = left.rename(
        columns={
            "nmse": "nmse_base",
            "pcc": "pcc_base",
        }
    )

    right = mc[
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    right = right.rename(
        columns={
            "nmse": "nmse_mc",
            "pcc": "pcc_mc",
        }
    )

    q = left.merge(
        right,
        on=["c", "r", "p"],
        how="inner",
        validate="one_to_one",
    )

    if len(q) != 480:
        raise RuntimeError(
            f"{method_name}: expected 480 matched "
            f"cases, got {len(q)}"
        )

    # Mean within each realization block.
    block = (
        q.groupby("r")
        .agg(
            nmse_base=("nmse_base", "mean"),
            nmse_mc=("nmse_mc", "mean"),
            pcc_base=("pcc_base", "mean"),
            pcc_mc=("pcc_mc", "mean"),
        )
        .sort_index()
    )

    if len(block) != 20:
        raise RuntimeError(
            f"{method_name}: expected 20 blocks, "
            f"got {len(block)}"
        )

    nb = len(block)

    b_nmse = block["nmse_base"].to_numpy()
    m_nmse = block["nmse_mc"].to_numpy()

    b_pcc = block["pcc_base"].to_numpy()
    m_pcc = block["pcc_mc"].to_numpy()

    # Observed.
    obs_base = q["nmse_base"].mean()
    obs_mc = q["nmse_mc"].mean()

    obs_abs = (
        obs_base
        - obs_mc
    )

    obs_rel = (
        100.0
        * obs_abs
        / obs_base
    )

    obs_pcc = (
        q["pcc_mc"].mean()
        - q["pcc_base"].mean()
    )

    # Bootstrap by realization.
    rng = np.random.default_rng(
        BOOT_SEED
    )

    ind = rng.integers(
        0,
        nb,
        size=(NBOOT, nb),
    )

    bs_base_nmse = (
        b_nmse[ind]
        .mean(axis=1)
    )

    bs_mc_nmse = (
        m_nmse[ind]
        .mean(axis=1)
    )

    bs_abs = (
        bs_base_nmse
        - bs_mc_nmse
    )

    bs_rel = (
        100.0
        * bs_abs
        / bs_base_nmse
    )

    bs_pcc = (
        m_pcc[ind].mean(axis=1)
        - b_pcc[ind].mean(axis=1)
    )

    abs_lo, abs_hi = ci(bs_abs)
    rel_lo, rel_hi = ci(bs_rel)
    pcc_lo, pcc_hi = ci(bs_pcc)

    patch_wins = int(
        np.sum(
            q["nmse_mc"]
            < q["nmse_base"]
        )
    )

    block_wins = int(
        np.sum(
            m_nmse
            < b_nmse
        )
    )

    return {
        "comparison": (
            f"MCALENS vs {method_name}"
        ),

        "n_cases": len(q),
        "n_blocks": nb,

        "base_nmse": float(obs_base),
        "mcalens_nmse": float(obs_mc),

        "nmse_gain_abs": float(obs_abs),
        "nmse_gain_abs_ci_lo": abs_lo,
        "nmse_gain_abs_ci_hi": abs_hi,

        "nmse_gain_pct": float(obs_rel),
        "nmse_gain_pct_ci_lo": rel_lo,
        "nmse_gain_pct_ci_hi": rel_hi,

        "pcc_gain": float(obs_pcc),
        "pcc_gain_ci_lo": pcc_lo,
        "pcc_gain_ci_hi": pcc_hi,

        "patch_wins": patch_wins,
        "patch_total": len(q),

        "block_wins": block_wins,
        "block_total": nb,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 92)
    print("MCALENS — FROZEN HELD-OUT TEST")
    print("=" * 92)

    print("C             : 0,1,2")
    print("R             : 10...29")
    print("patches       : 0...7")
    print("total cases   : 480")

    print()
    print("FROZEN PARAMETERS")
    print("-----------------")
    print("Nsigma        :", NSIGMA)
    print("niter         :", NITER)
    print("nscales       :", NSCALES)
    print("Inpaint       :", INPAINT)
    print("OnlyPos       :", ONLY_POS)
    print(
        "FirstScale    :",
        FIRST_DETECT_SCALE,
    )
    print("Bmode         :", BMODE)
    print("Pk            :", PK_FILE)
    print("workers       :", NWORKERS)

    print()

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------
    if CSV.exists():

        old = pd.read_csv(
            CSV
        )

        done = set(
            zip(
                old["c"].astype(int),
                old["r"].astype(int),
                old["p"].astype(int),
            )
        )

        rows = old.to_dict(
            "records"
        )

        print(
            "Existing cases:",
            len(rows),
        )

    else:

        done = set()
        rows = []

    jobs = [
        (c, r, p)
        for c in COSMOS
        for r in REALS
        for p in PATCHES
        if (
            c,
            r,
            p,
        ) not in done
    ]

    print(
        "Remaining cases:",
        len(jobs),
    )
    print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    with ProcessPoolExecutor(
        max_workers=NWORKERS
    ) as ex:

        futures = {
            ex.submit(
                run_one,
                job,
            ): job
            for job in jobs
        }

        completed = 0

        for fut in as_completed(
            futures
        ):

            job = futures[fut]

            try:
                row = fut.result()

            except Exception as exc:

                print(
                    "FAILED:",
                    job,
                    repr(exc),
                    flush=True,
                )

                raise

            rows.append(row)
            completed += 1

            if (
                completed % 8 == 0
                or completed == len(jobs)
            ):

                tab = pd.DataFrame(
                    rows
                )

                tab = tab.sort_values(
                    ["c", "r", "p"]
                )

                tab.to_csv(
                    CSV,
                    index=False,
                )

                print(
                    f"{completed:4d}/"
                    f"{len(jobs):4d} "
                    f"new cases saved",
                    flush=True,
                )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------
    tab = pd.DataFrame(
        rows
    )

    tab = (
        tab.drop_duplicates(
            subset=["c", "r", "p"],
            keep="last",
        )
        .sort_values(
            ["c", "r", "p"]
        )
        .reset_index(drop=True)
    )

    if len(tab) != 480:
        raise RuntimeError(
            f"Expected 480 test cases; got {len(tab)}"
        )

    tab.to_csv(
        CSV,
        index=False,
    )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================
    summary = pd.DataFrame(
        [
            {
                "method": "MCALENS",
                "n": len(tab),

                "nmse_mean": (
                    tab["nmse"].mean()
                ),

                "nmse_std": (
                    tab["nmse"].std()
                ),

                "nmse_median": (
                    tab["nmse"].median()
                ),

                "mse_mean": (
                    tab["mse"].mean()
                ),

                "pcc_mean": (
                    tab["pcc"].mean()
                ),

                "std_ratio_mean": (
                    tab["std_ratio"].mean()
                ),

                "runtime_mean": (
                    tab["runtime"].mean()
                ),

                "active_mean": (
                    tab["active_coeffs"].mean()
                ),

                "finite_rate": (
                    tab["finite"].mean()
                ),
            }
        ]
    )

    summary_file = (
        OUT
        / "mcalens_test_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print("=" * 92)
    print("GLOBAL TEST SUMMARY")
    print("=" * 92)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # ========================================================
    # BY COSMOLOGY
    # ========================================================
    byc = (
        tab.groupby("c")
        .agg(
            n=("nmse", "size"),

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
                "active_coeffs",
                "mean",
            ),
        )
        .reset_index()
    )

    byc_file = (
        OUT
        / "mcalens_test_by_cosmology.csv"
    )

    byc.to_csv(
        byc_file,
        index=False,
    )

    print()
    print("=" * 92)
    print("BY COSMOLOGY")
    print("=" * 92)

    print(
        byc.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # ========================================================
    # FIND OUR EXISTING MATCHED SQUARE METRICS
    # ========================================================
    candidates = [
        Path(
            str(PATHS.work_root / 'test/square_ours_metrics.csv')
        ),

        Path(
            str(PATHS.work_root / 'test/square_method_metrics.csv')
        ),

        Path(
            str(PATHS.work_root / 'test/our_square_metrics.csv')
        ),
    ]

    ours_file = None

    for fn in candidates:
        if fn.exists():
            ours_file = fn
            break

    if ours_file is None:

        print()
        print(
            "No per-case square comparator CSV found."
        )

        print(
            "MCALens results are complete; "
            "pairwise bootstrap skipped."
        )

    else:

        print()
        print(
            "Comparator metrics:",
            ours_file,
        )

        ours = pd.read_csv(
            ours_file
        )

        required = {
            "method",
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        }

        missing = (
            required
            - set(ours.columns)
        )

        if missing:
            raise RuntimeError(
                f"{ours_file} missing columns: "
                f"{sorted(missing)}"
            )

        pair_rows = []

        for method in [
            "KS",
            "STARLET_NOMASK",
            "STARLET_MASK",
        ]:

            base = ours[
                ours["method"] == method
            ].copy()

            if len(base) == 0:
                print(
                    "Method absent:",
                    method,
                )
                continue

            res = paired_block_bootstrap(
                base,
                tab,
                method,
            )

            pair_rows.append(res)

        if pair_rows:

            pair = pd.DataFrame(
                pair_rows
            )

            pair_file = (
                OUT
                / "mcalens_test_pairwise_bootstrap.csv"
            )

            pair.to_csv(
                pair_file,
                index=False,
            )

            print()
            print("=" * 92)
            print("PAIRED REALIZATION-BLOCK BOOTSTRAP")
            print("=" * 92)

            print(
                pair.to_string(
                    index=False,
                    float_format=lambda x: (
                        f"{x:.6f}"
                    ),
                )
            )

            print()
            print(
                "Saved:",
                pair_file,
            )

    print()
    print("=" * 92)
    print("FILES")
    print("=" * 92)

    print(CSV)
    print(summary_file)
    print(byc_file)
    print(RECON_DIR)

    print()
    print("MCALENS TEST COMPLETE")
