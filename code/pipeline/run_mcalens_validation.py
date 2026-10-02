from project_paths import PATHS
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE = Path(str(PATHS.work_root / 'validation/fair_npz'))
MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

OUT = Path(
    str(PATHS.work_root / 'validation/mcalens')
)
OUT.mkdir(parents=True, exist_ok=True)

CSV = OUT / "mcalens_validation_metrics.csv"

COSMOS = range(3)
REALS = range(10)
PATCHES = range(8)

NSIGMAS = [6]

NITER = 100
NSCALES = 4

# Change from shell if desired:
# export MCALENS_WORKERS=8
NWORKERS = int(
    os.environ.get(
        "MCALENS_WORKERS",
        "8",
    )
)

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
    mask = mask.astype(bool)

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[mask]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[mask]

    # mass-sheet invariant
    r -= r.mean()
    t -= t.mean()

    mse = np.mean(
        (r - t) ** 2
    )

    den = np.mean(
        t ** 2
    )

    nmse = mse / den

    pcc = np.corrcoef(
        r,
        t,
    )[0, 1]

    std_ratio = (
        np.std(r)
        / np.std(t)
    )

    return (
        float(mse),
        float(nmse),
        float(pcc),
        float(std_ratio),
    )


# ============================================================
# ONE JOB
# ============================================================
def run_one(job):
    c, r, p, nsigma = job

    # Import inside worker.
    from pycs.astro.wl.mass_mapping import (
        massmap2d,
        shear_data,
    )

    cache = (
        BASE
        / f"C{c}_R{r}.npz"
    )

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

    gamma_obs = (
        gamma
        * mask
    )

    g1 = gamma_obs.real.copy()
    g2 = gamma_obs.imag.copy()

    # --------------------------------------------------------
    # MCALens data
    # --------------------------------------------------------
    d = shear_data()

    d.g1 = g1
    d.g2 = g2

    Ncov = np.zeros(
        truth.shape,
        dtype=np.float64,
    )

    # Complex-shear covariance convention:
    # var(component) = Ncov / 2
    Ncov[mask] = (
        2.0
        * sigma_n**2
    )

    d.Ncov = Ncov
    d.mask = mask.astype(
        np.float64
    )

    # --------------------------------------------------------
    # MCALens
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
        Nsigma=nsigma,
        Inpaint=False,
        OnlyPos=True,
        FirstDetectScale=1,
        Bmode=True,
        ktr=None,
        PropagateNoise=False,
    )

    runtime = (
        time.perf_counter()
        - t0
    )

    mse, nmse, pcc, std_ratio = (
        ms_metrics(
            kappa,
            truth,
            mask,
        )
    )

    return {
        "c": c,
        "r": r,
        "p": p,
        "Nsigma": nsigma,

        "f_obs": float(
            mask.mean()
        ),

        "n_obs": int(
            mask.sum()
        ),

        "mse": mse,
        "nmse": nmse,
        "pcc": pcc,
        "std_ratio": std_ratio,

        "runtime": runtime,

        "bmode_std": float(
            np.std(kb)
        ),

        "sparse_std": float(
            np.std(ksparse)
        ),

        "active_coeffs": int(
            np.sum(
                M.WT_ActiveCoef
            )
        ),

        "finite": bool(
            np.isfinite(kappa).all()
        ),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("=" * 90)
    print("MCALENS VALIDATION")
    print("=" * 90)

    print(
        "Nsigma     :",
        NSIGMAS,
    )

    print(
        "iterations :",
        NITER,
    )

    print(
        "workers    :",
        NWORKERS,
    )

    print(
        "Pk         :",
        PK_FILE,
    )

    print()

    # --------------------------------------------------------
    # Resume support
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
                old["Nsigma"].astype(int),
            )
        )

        rows = old.to_dict(
            "records"
        )

        print(
            "Existing rows:",
            len(rows),
        )

    else:
        done = set()
        rows = []

    jobs = [
        (c, r, p, nsigma)
        for nsigma in NSIGMAS
        for c in COSMOS
        for r in REALS
        for p in PATCHES
        if (
            c,
            r,
            p,
            nsigma,
        ) not in done
    ]

    print(
        "Remaining jobs:",
        len(jobs),
    )

    print()

    # --------------------------------------------------------
    # Parallel
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
                    [
                        "Nsigma",
                        "c",
                        "r",
                        "p",
                    ]
                )

                tab.to_csv(
                    CSV,
                    index=False,
                )

                print(
                    f"{completed:4d}/"
                    f"{len(jobs):4d} new jobs "
                    f"saved",
                    flush=True,
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    tab = pd.DataFrame(
        rows
    )

    tab = tab.sort_values(
        [
            "Nsigma",
            "c",
            "r",
            "p",
        ]
    )

    tab.to_csv(
        CSV,
        index=False,
    )

    print()
    print("=" * 90)
    print("GLOBAL VALIDATION SUMMARY")
    print("=" * 90)

    summary = (
        tab.groupby(
            "Nsigma"
        )
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

            runtime_mean=(
                "runtime",
                "mean",
            ),

            finite_rate=(
                "finite",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("=" * 90)
    print("BY COSMOLOGY")
    print("=" * 90)

    byc = (
        tab.groupby(
            [
                "Nsigma",
                "c",
            ]
        )
        .agg(
            n=("nmse", "size"),

            nmse_mean=(
                "nmse",
                "mean",
            ),

            pcc_mean=(
                "pcc",
                "mean",
            ),

            std_ratio_mean=(
                "std_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        byc.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        "Saved:",
        CSV,
    )
