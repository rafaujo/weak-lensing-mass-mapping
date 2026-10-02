from project_paths import PATHS
from pathlib import Path
import csv
import os
import subprocess
import time

import numpy as np
import pandas as pd
from astropy.io import fits

# ============================================================
# Frozen validation configuration
# ============================================================
B = Path(str(PATHS.work_root))

VAL = B / "validation"
NPZ_DIR = VAL / "fair_npz"
CAT_DIR = VAL / "catalogs"
OUT_DIR = VAL / "recon"
CFG_DIR = VAL / "configs"
LOG_DIR = VAL / "logs"

for d in [
    CAT_DIR,
    OUT_DIR,
    CFG_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

GLIMPSE = PATHS.glimpse_root / 'build' / 'glimpse'

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

mask_full = np.load(MASK_FILE).astype(bool)

PATCH = 176

PATCHES = [
    (p, p * PATCH, (p + 1) * PATCH)
    for p in range(8)
]

# Bracket fixed from C0/R0/P0 pilot.
LAMBDAS = [2.5, 3.0, 3.5]

NITER = 500
NRANDOM = 1000
NREWEIGHTS = 5
NSCALES = 4

PIXEL_SIZE = 2.0
SIZE = 352.0

RESULTS_CSV = VAL / "glimpse_validation_metrics.csv"

# ============================================================
# Helpers
# ============================================================
def tag(lam):
    return str(lam).replace(".", "p")


def metrics_ms(rec, truth, m):

    r = np.asarray(rec[m], dtype=np.float64).copy()
    t = np.asarray(truth[m], dtype=np.float64).copy()

    r -= r.mean()
    t -= t.mean()

    nmse = np.dot(r-t, r-t) / np.dot(t, t)

    rr = np.dot(r, r)
    tt = np.dot(t, t)

    pcc = np.dot(r, t) / np.sqrt(rr * tt)

    sr = r.std() / t.std()

    return float(nmse), float(pcc), float(sr)


def write_catalog(path, gamma, mask):

    if path.exists():
        return

    N = 176

    coord = (
        np.arange(N) - (N - 1) / 2.0
    ) * PIXEL_SIZE

    RA, DEC = np.meshgrid(
        coord,
        coord
    )

    sel = mask

    ra = RA[sel].astype(np.float64)
    dec = DEC[sel].astype(np.float64)

    e1 = gamma.real[sel].astype(np.float64)
    e2 = gamma.imag[sel].astype(np.float64)

    # Required by survey::load().
    # Ignored physically because zlens=-1.
    z = np.ones(
        sel.sum(),
        dtype=np.float64
    )

    cols = [
        fits.Column(
            name="ra",
            format="D",
            array=ra
        ),
        fits.Column(
            name="dec",
            format="D",
            array=dec
        ),
        fits.Column(
            name="e1",
            format="D",
            array=e1
        ),
        fits.Column(
            name="e2",
            format="D",
            array=e2
        ),
        fits.Column(
            name="z",
            format="D",
            array=z
        ),
    ]

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(cols)
    ]).writeto(
        path,
        overwrite=True
    )


def write_config(path, omega_m, lam):

    txt = f"""[survey]
center_ra=0.0
center_dec=0.0
size={SIZE}
units=arcmin
flip_e2=true
hdu=1
ra=ra
dec=dec
e1=e1
e2=e2
z=z

[cosmology]
Omega_m={omega_m:.10f}
h=0.70

[field]
units=arcmin
pixel_size={PIXEL_SIZE}
padding=0
include_flexion=false
zlens=-1

[parameters]
niter={NITER}
nrandom={NRANDOM}
nreweights={NREWEIGHTS}
nscales={NSCALES}
lambda={lam}
positivity=false
battle_lemarie_reg=0.1
last_scale_reg=0.1
"""

    path.write_text(txt)


# ============================================================
# Resume database
# ============================================================
fields = [
    "method",
    "c", "r", "p",
    "row0", "row1",
    "omega_m",
    "f_obs", "n_obs",
    "lambda",
    "lambda_tag",
    "nmse",
    "pcc",
    "std_ratio",
    "runtime_s",
    "gsl_seed",
    "output_file",
]

done = set()

if RESULTS_CSV.exists():

    old = pd.read_csv(RESULTS_CSV)

    for _, x in old.iterrows():
        done.add((
            int(x["c"]),
            int(x["r"]),
            int(x["p"]),
            str(x["lambda_tag"])
        ))

    append_header = False

else:
    append_header = True


fout = open(
    RESULTS_CSV,
    "a",
    newline=""
)

writer = csv.DictWriter(
    fout,
    fieldnames=fields
)

if append_header:
    writer.writeheader()
    fout.flush()


# ============================================================
# Main
# ============================================================
TOTAL = 3 * 10 * 8 * len(LAMBDAS)
counter = 0

try:

    for c in range(3):

        for r in range(10):

            npz_file = NPZ_DIR / f"C{c}_R{r}.npz"

            if not npz_file.exists():
                raise FileNotFoundError(npz_file)

            d = np.load(npz_file)

            truth_full = d["truth"]
            gamma_full = d["gamma"]

            omega_m = float(d["omega_m"])

            for p, r0, r1 in PATCHES:

                truth_p = truth_full[r0:r1, :]
                gamma_p = gamma_full[r0:r1, :]
                mask_p = mask_full[r0:r1, :]

                if mask_p.sum() == 0:
                    print(
                        f"SKIP C{c} R{r} P{p}: empty mask"
                    )
                    continue

                cat_file = (
                    CAT_DIR /
                    f"catalog_C{c}_R{r}_P{p}.fits"
                )

                write_catalog(
                    cat_file,
                    gamma_p,
                    mask_p
                )

                # Different random threshold realization per
                # map/patch, but IDENTICAL across lambda values.
                gsl_seed = (
                    c * 10000 +
                    r * 100 +
                    p
                )

                for lam in LAMBDAS:

                    counter += 1

                    ltag = tag(lam)

                    key = (
                        c, r, p, ltag
                    )

                    if key in done:

                        print(
                            f"[{counter:03d}/{TOTAL}] "
                            f"C{c} R{r} P{p} L={lam}: cached"
                        )

                        continue

                    cfg = (
                        CFG_DIR /
                        f"config_C{c}_L{ltag}.ini"
                    )

                    write_config(
                        cfg,
                        omega_m,
                        lam
                    )

                    rec_file = (
                        OUT_DIR /
                        f"glimpse_C{c}_R{r}_P{p}_L{ltag}.fits"
                    )

                    log_file = (
                        LOG_DIR /
                        f"C{c}_R{r}_P{p}_L{ltag}.log"
                    )

                    print(
                        f"[{counter:03d}/{TOTAL}] "
                        f"C{c} R{r} P{p} "
                        f"lambda={lam}"
                    )

                    need_run = True

                    # If reconstruction already exists from an
                    # interrupted run, validate it before rerunning.
                    if rec_file.exists():

                        try:
                            x = fits.getdata(rec_file)

                            if (
                                x.shape == (176, 176)
                                and np.isfinite(x).all()
                            ):
                                need_run = False

                        except Exception:
                            need_run = True

                    runtime = np.nan

                    if need_run:

                        env = os.environ.copy()

                        env["GSL_RNG_SEED"] = str(
                            gsl_seed
                        )

                        cmd = [
                            str(GLIMPSE),
                            "--config", str(cfg),
                            "--data", str(cat_file),
                            "--output", str(rec_file),
                        ]

                        t0 = time.perf_counter()

                        with open(
                            log_file,
                            "w"
                        ) as log:

                            proc = subprocess.run(
                                cmd,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                env=env
                            )

                        runtime = (
                            time.perf_counter() - t0
                        )

                        if proc.returncode != 0:

                            print(
                                "\nGLIMPSE FAILED:"
                            )
                            print(
                                " ", log_file
                            )

                            raise RuntimeError(
                                f"GLIMPSE return code "
                                f"{proc.returncode}"
                            )

                    rec = fits.getdata(
                        rec_file
                    ).astype(np.float64)

                    assert rec.shape == (176, 176)

                    nmse, pcc, sr = metrics_ms(
                        rec,
                        truth_p,
                        mask_p
                    )

                    row = {
                        "method": "GLIMPSE",
                        "c": c,
                        "r": r,
                        "p": p,
                        "row0": r0,
                        "row1": r1,
                        "omega_m": omega_m,
                        "f_obs": float(mask_p.mean()),
                        "n_obs": int(mask_p.sum()),
                        "lambda": lam,
                        "lambda_tag": ltag,
                        "nmse": nmse,
                        "pcc": pcc,
                        "std_ratio": sr,
                        "runtime_s": runtime,
                        "gsl_seed": gsl_seed,
                        "output_file": str(rec_file),
                    }

                    writer.writerow(row)
                    fout.flush()

                    done.add(key)

                    print(
                        f"      NMSE={nmse:.6f} "
                        f"PCC={pcc:.6f} "
                        f"std={sr:.4f}"
                    )

finally:

    fout.close()


# ============================================================
# Summary
# ============================================================
df = pd.read_csv(RESULTS_CSV)

print()
print("=" * 78)
print("GLIMPSE VALIDATION SUMMARY")
print("=" * 78)

summary = (
    df.groupby("lambda")
    .agg(
        n=("nmse", "size"),
        nmse_mean=("nmse", "mean"),
        nmse_std=("nmse", "std"),
        nmse_median=("nmse", "median"),
        pcc_mean=("pcc", "mean"),
        std_ratio_mean=("std_ratio", "mean"),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

best = summary.loc[
    summary["nmse_mean"].idxmin()
]

print()
print("BEST GLIMPSE BY MEAN VALIDATION MS-NMSE")
print("lambda =", best["lambda"])
print("NMSE   =", best["nmse_mean"])
print("PCC    =", best["pcc_mean"])

# ============================================================
# Per-cosmology summary
# ============================================================
print()
print("BY COSMOLOGY")

byc = (
    df.groupby(["lambda", "c"])
    .agg(
        n=("nmse", "size"),
        nmse=("nmse", "mean"),
        pcc=("pcc", "mean")
    )
    .reset_index()
)

print(
    byc.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# KS comparison
# ============================================================
ks_file = VAL / "ks_validation_metrics.csv"

if ks_file.exists():

    ks = pd.read_csv(ks_file)

    print()
    print("=" * 78)
    print("KS sigma=1.5 ON SAME SQUARE PATCHES")
    print("=" * 78)

    print(
        "N          =", len(ks)
    )
    print(
        "NMSE mean  =", ks["nmse"].mean()
    )
    print(
        "NMSE median=", ks["nmse"].median()
    )
    print(
        "PCC mean   =", ks["pcc"].mean()
    )

    print()
    print("GLIMPSE BEST - KS")

    print(
        "Delta NMSE =",
        best["nmse_mean"] -
        ks["nmse"].mean()
    )
