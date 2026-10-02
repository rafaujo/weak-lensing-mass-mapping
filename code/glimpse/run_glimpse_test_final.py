from project_paths import PATHS
from pathlib import Path
import csv
import os
import subprocess
import time

import numpy as np
from astropy.io import fits

B = Path(str(PATHS.work_root))
TEST = B / "test"

NPZ_DIR = TEST / "fair_npz"
CAT_DIR = TEST / "catalogs"
REC_DIR = TEST / "glimpse_recon"
CFG_DIR = TEST / "configs"
LOG_DIR = TEST / "logs"

for d in [
    CAT_DIR,
    REC_DIR,
    CFG_DIR,
    LOG_DIR,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )

GLIMPSE = PATHS.glimpse_root / 'build' / 'glimpse'

MASK = np.load(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
).astype(bool)

LAMBDA = 3.0

N = 176
PIX = 2.0

RESULTS = (
    TEST /
    "glimpse_test_metrics.csv"
)

fields = [
    "method",
    "c", "r", "p",
    "omega_m",
    "f_obs", "n_obs",
    "lambda",
    "nmse",
    "pcc",
    "std_ratio",
    "runtime_s",
]


def metrics(rec, truth, mask):

    r = rec[mask].astype(
        np.float64
    ).copy()

    t = truth[mask].astype(
        np.float64
    ).copy()

    r -= r.mean()
    t -= t.mean()

    nmse = (
        np.dot(r-t, r-t)
        / np.dot(t, t)
    )

    pcc = (
        np.dot(r, t)
        / np.sqrt(
            np.dot(r, r)
            * np.dot(t, t)
        )
    )

    return (
        float(nmse),
        float(pcc),
        float(
            r.std()/t.std()
        )
    )


def write_catalog(fn, gamma, mask):

    if fn.exists():
        return

    coord = (
        np.arange(N)
        - (N-1)/2
    ) * PIX

    RA, DEC = np.meshgrid(
        coord,
        coord
    )

    sel = mask

    cols = [
        fits.Column(
            name="ra",
            format="D",
            array=RA[sel]
        ),
        fits.Column(
            name="dec",
            format="D",
            array=DEC[sel]
        ),
        fits.Column(
            name="e1",
            format="D",
            array=gamma.real[sel]
        ),
        fits.Column(
            name="e2",
            format="D",
            array=gamma.imag[sel]
        ),
        fits.Column(
            name="z",
            format="D",
            array=np.ones(
                sel.sum()
            )
        ),
    ]

    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            cols
        )
    ]).writeto(
        fn,
        overwrite=True
    )


def write_config(fn, omega_m):

    fn.write_text(
f"""[survey]
center_ra=0.0
center_dec=0.0
size=352.0
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
pixel_size=2.0
padding=0
include_flexion=false
zlens=-1

[parameters]
niter=500
nrandom=1000
nreweights=5
nscales=4
lambda=3.0
positivity=false
battle_lemarie_reg=0.1
last_scale_reg=0.1
"""
    )


done = set()

if RESULTS.exists():

    with open(RESULTS) as f:

        for x in csv.DictReader(f):

            done.add((
                int(x["c"]),
                int(x["r"]),
                int(x["p"]),
            ))

    header = False

else:
    header = True


fout = open(
    RESULTS,
    "a",
    newline=""
)

writer = csv.DictWriter(
    fout,
    fieldnames=fields
)

if header:
    writer.writeheader()


try:

    counter = 0
    total = 3*20*8

    for c in range(3):

        for r in range(10,30):

            d = np.load(
                NPZ_DIR /
                f"C{c}_R{r}.npz"
            )

            truth = d["truth"]
            gamma = d["gamma"]

            omega_m = float(
                d["omega_m"]
            )

            cfg = (
                CFG_DIR /
                f"C{c}.ini"
            )

            write_config(
                cfg,
                omega_m
            )

            for p in range(8):

                counter += 1

                if (c,r,p) in done:

                    print(
                        f"[{counter}/{total}] "
                        f"C{c} R{r} P{p}: cached"
                    )
                    continue

                r0 = 176*p
                r1 = 176*(p+1)

                t = truth[r0:r1, :]
                g = gamma[r0:r1, :]
                m = MASK[r0:r1, :]

                cat = (
                    CAT_DIR /
                    f"C{c}_R{r}_P{p}.fits"
                )

                recfile = (
                    REC_DIR /
                    f"C{c}_R{r}_P{p}.fits"
                )

                log = (
                    LOG_DIR /
                    f"C{c}_R{r}_P{p}.log"
                )

                write_catalog(
                    cat,
                    g,
                    m
                )

                runtime = np.nan

                if not recfile.exists():

                    env = os.environ.copy()

                    # Same seed convention as validation:
                    # fixed for each map/patch.
                    env["GSL_RNG_SEED"] = str(
                        c*10000
                        + r*100
                        + p
                    )

                    cmd = [
                        str(GLIMPSE),
                        "--config",
                        str(cfg),
                        "--data",
                        str(cat),
                        "--output",
                        str(recfile),
                    ]

                    print(
                        f"[{counter}/{total}] "
                        f"C{c} R{r} P{p}"
                    )

                    t0 = time.perf_counter()

                    with open(
                        log, "w"
                    ) as lf:

                        proc = subprocess.run(
                            cmd,
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            env=env
                        )

                    runtime = (
                        time.perf_counter()
                        - t0
                    )

                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"GLIMPSE failed: {log}"
                        )

                rec = fits.getdata(
                    recfile
                ).astype(np.float64)

                nmse, pcc, sr = metrics(
                    rec,
                    t,
                    m
                )

                writer.writerow({
                    "method": "GLIMPSE",
                    "c": c,
                    "r": r,
                    "p": p,
                    "omega_m": omega_m,
                    "f_obs": float(
                        m.mean()
                    ),
                    "n_obs": int(
                        m.sum()
                    ),
                    "lambda": LAMBDA,
                    "nmse": nmse,
                    "pcc": pcc,
                    "std_ratio": sr,
                    "runtime_s": runtime,
                })

                fout.flush()

                done.add(
                    (c,r,p)
                )

finally:
    fout.close()


rows = []

with open(RESULTS) as f:
    rows = list(
        csv.DictReader(f)
    )

nmse = np.array([
    float(x["nmse"])
    for x in rows
])

pcc = np.array([
    float(x["pcc"])
    for x in rows
])

print()
print("="*70)
print("GLIMPSE FINAL TEST")
print("="*70)
print("N         =", len(rows))
print("NMSE mean =", nmse.mean())
print("median    =", np.median(nmse))
print("PCC mean  =", pcc.mean())
