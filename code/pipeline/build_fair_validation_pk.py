from project_paths import PATHS
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import map_coordinates

# ============================================================
# CONFIG
# ============================================================
BASE = Path(str(PATHS.work_root / 'validation/fair_npz'))

DATA_DIR = Path(str(PATHS.data_root))
MASK_FILE = DATA_DIR / "WIDE12H_bin2_2arcmin_mask.npy"

OUT = Path(str(PATHS.work_root / 'validation/mcalens'))
OUT.mkdir(parents=True, exist_ok=True)

COSMOS = range(3)
REALS = range(10)       # VALIDATION ONLY: R0...R9
PATCHES = range(8)

N = 176

# Same square decomposition used in GLIMPSE benchmark:
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
# LOAD GLOBAL FAIR MASK
# ============================================================
mask_full = np.load(MASK_FILE).astype(bool)

print("Global mask:")
print(" shape =", mask_full.shape)
print(" f_obs =", mask_full.mean())

assert mask_full.shape == (1424, 176)

mask_patches = np.stack(
    [
        mask_full[r0:r1, :]
        for r0, r1 in PATCH_ROWS
    ],
    axis=0,
)

assert mask_patches.shape == (8, 176, 176)

print()
print("Patch mask fractions:")
for p in PATCHES:
    print(
        f"P{p}: "
        f"{mask_patches[p].mean():.6f}"
    )


# ============================================================
# HELPERS
# ============================================================
def extract_truth_patches(truth, filename):
    """
    Support either:
      (8,176,176) cached patches
      (1424,176) full FAIR strip
    """
    truth = np.asarray(truth)

    if truth.shape == (8, 176, 176):
        return truth.astype(np.float64)

    if truth.shape == (1424, 176):
        return np.stack(
            [
                truth[r0:r1, :]
                for r0, r1 in PATCH_ROWS
            ],
            axis=0,
        ).astype(np.float64)

    raise ValueError(
        f"{filename}: unsupported truth shape {truth.shape}"
    )


def isotropic_spectrum_cosmostat_like(image):
    """
    Reproduce the normalization/convention used by:

        pycs.misc.im_isospec.im_isospec()

    based on the Sparse2D source:

      1. zero-pad image by nx//2 on every side
      2. centered FFT
      3. power = |FFT|^2 / Npix
      4. isotropic sampling on circles
      5. multiply spectrum by 4
      6. average adjacent radial bins

    The original Sparse2D implementation uses cubic B-spline
    interpolation of the 2-D power image. scipy map_coordinates
    with order=3 is used here as the Python equivalent.
    """
    image = np.asarray(
        image,
        dtype=np.float64,
    )

    if image.ndim != 2:
        raise ValueError(image.shape)

    nx, ny = image.shape

    if nx != ny:
        raise ValueError(
            f"Expected square image, got {image.shape}"
        )

    # --------------------------------------------------------
    # CosmoStat wrapper:
    #
    # pad_ke = np.pad(ima, nx // 2)
    # --------------------------------------------------------
    padded = np.pad(
        image,
        nx // 2,
        mode="constant",
        constant_values=0.0,
    )

    nl, nc = padded.shape
    npix = nl * nc

    # Sparse2D convention: zero-frequency at center.
    ft = np.fft.fftshift(
        np.fft.fft2(padded)
    )

    power2d = (
        np.abs(ft) ** 2
        / float(npix)
    )

    # Sparse2D:
    #
    # N2 = (Nl+1)/2 * sqrt(2)
    # NpSpec = int((N2 - 1)/Resol)
    #
    # Resol = 1.
    n2 = (
        ((nl + 1) // 2)
        * np.sqrt(2.0)
    )

    npspec = int(
        n2 - 1.0
    )

    spec = np.zeros(
        npspec,
        dtype=np.float64,
    )

    cy = nl / 2.0
    cx = nc / 2.0

    spec[0] = power2d[
        int(cy),
        int(cx),
    ]

    # --------------------------------------------------------
    # Circular interpolation
    # --------------------------------------------------------
    for r in range(1, npspec):

        npoints = max(
            1,
            int(
                2.0 * np.pi * r
                + 0.5
            ),
        )

        theta = (
            2.0
            * np.pi
            * np.arange(npoints)
            / npoints
        )

        x = (
            cx
            + r * np.cos(theta)
        )

        y = (
            cy
            + r * np.sin(theta)
        )

        good = (
            (x >= 0.0)
            & (x < nc)
            & (y >= 0.0)
            & (y < nl)
        )

        if not np.any(good):
            continue

        vals = map_coordinates(
            power2d,
            [
                y[good],
                x[good],
            ],
            order=3,
            mode="nearest",
            prefilter=True,
        )

        # Sparse2D:
        # MAX(0, interpolated_value)
        vals = np.maximum(
            vals,
            0.0,
        )

        spec[r] = vals.mean()

    # --------------------------------------------------------
    # Python CosmoStat wrapper:
    #
    # p1 = p[1,:] * 4
    # npix = len(p1)//2
    # p2 = rebin1d(p1[:2*npix],[npix])
    #
    # rebin1d => average adjacent bins
    # --------------------------------------------------------
    p1 = 4.0 * spec

    nout = len(p1) // 2

    p2 = (
        p1[: 2 * nout]
        .reshape(nout, 2)
        .mean(axis=1)
    )

    return p2


# ============================================================
# BUILD VALIDATION SPECTRA
# ============================================================
all_ps = []

first = True

for c in COSMOS:

    for r in REALS:

        fn = (
            BASE
            / f"C{c}_R{r}.npz"
        )

        if not fn.exists():
            raise FileNotFoundError(fn)

        with np.load(fn) as z:

            if first:
                print()
                print("=" * 72)
                print("CACHE CHECK")
                print("=" * 72)
                print("file      :", fn)
                print("keys      :", z.files)

                for key in z.files:
                    arr = np.asarray(z[key])
                    print(
                        f"{key:12s}: "
                        f"shape={arr.shape} "
                        f"dtype={arr.dtype}"
                    )

                first = False

            if "truth" not in z.files:
                raise KeyError(
                    f"{fn}: no 'truth' array"
                )

            truth_patches = (
                extract_truth_patches(
                    z["truth"],
                    fn,
                )
            )

        assert (
            truth_patches.shape
            == (8, 176, 176)
        )

        # ----------------------------------------------------
        # Patches
        # ----------------------------------------------------
        for p in PATCHES:

            truth = np.asarray(
                truth_patches[p],
                dtype=np.float64,
            )

            mask = np.asarray(
                mask_patches[p],
                dtype=bool,
            )

            fobs = mask.mean()

            if fobs <= 0:
                raise RuntimeError(
                    f"C{c} R{r} P{p}: empty mask"
                )

            # ------------------------------------------------
            # Mass-sheet invariant truth
            # ------------------------------------------------
            kappa = truth.copy()

            mean_obs = (
                kappa[mask].mean()
            )

            kappa[mask] -= mean_obs

            # FAIR truth is only defined/used on observed
            # footprint in this controlled benchmark.
            kappa[~mask] = 0.0

            # ------------------------------------------------
            # Isotropic signal spectrum
            # ------------------------------------------------
            ps = (
                isotropic_spectrum_cosmostat_like(
                    kappa
                )
            )

            # Same f_sky correction supported by
            # CosmoStat's im_isospec(..., mask=mask).
            ps = (
                ps
                / fobs
            )

            all_ps.append(ps)

# ============================================================
# STACK
# ============================================================
all_ps = np.asarray(
    all_ps,
    dtype=np.float64,
)

assert all_ps.shape[0] == 240

if not np.isfinite(all_ps).all():
    raise RuntimeError(
        "Non-finite values in spectra."
    )

# ============================================================
# POOLED VALIDATION PRIOR
# ============================================================
pk_mean = np.mean(
    all_ps,
    axis=0,
)

pk_median = np.median(
    all_ps,
    axis=0,
)

# Mass-sheet / DC mode.
pk_mean[0] = 0.0
pk_median[0] = 0.0

pk_mean = np.maximum(
    pk_mean,
    0.0,
)

pk_median = np.maximum(
    pk_median,
    0.0,
)

# ============================================================
# SAVE
# ============================================================
np.save(
    OUT
    / "fair_validation_pk_mean.npy",
    pk_mean,
)

np.save(
    OUT
    / "fair_validation_pk_median.npy",
    pk_median,
)

np.save(
    OUT
    / "fair_validation_pk_all.npy",
    all_ps,
)

meta = {
    "source_cache": str(BASE),
    "mask_file": str(MASK_FILE),
    "cosmologies": [0, 1, 2],
    "realizations": list(range(10)),
    "patches": list(range(8)),
    "patch_rows": PATCH_ROWS,
    "n_validation_maps": 240,
    "patch_shape": [176, 176],
    "truth_usage": "validation only",
    "test_truth_used": False,
    "mass_sheet": (
        "mean removed over observed pixels "
        "before spectrum estimation"
    ),
    "outside_mask": (
        "set explicitly to zero"
    ),
    "mask_correction": (
        "each patch spectrum divided by f_obs"
    ),
    "aggregation_primary": (
        "mean over all 240 validation patches"
    ),
    "spectrum_convention": (
        "Python reproduction of CosmoStat/Sparse2D "
        "im_isospec normalization"
    ),
}

with open(
    OUT
    / "fair_validation_pk_config.json",
    "w",
) as f:

    json.dump(
        meta,
        f,
        indent=2,
    )

# ============================================================
# DIAGNOSTICS
# ============================================================
print()
print("=" * 72)
print("FAIR VALIDATION POWER PRIOR")
print("=" * 72)

print(
    "individual spectra :",
    all_ps.shape,
)

print(
    "pooled spectrum     :",
    pk_mean.shape,
)

print(
    "finite              :",
    np.isfinite(pk_mean).all(),
)

print(
    "min                 :",
    pk_mean.min(),
)

print(
    "max                 :",
    pk_mean.max(),
)

print(
    "mean                :",
    pk_mean.mean(),
)

print(
    "median              :",
    np.median(pk_mean),
)

print()
print("first 20 bins:")

for i, v in enumerate(
    pk_mean[:20]
):
    print(
        f"{i:3d}  "
        f"{v:.12e}"
    )

print()
print("last 10 bins:")

for i in range(
    max(0, len(pk_mean)-10),
    len(pk_mean),
):
    print(
        f"{i:3d}  "
        f"{pk_mean[i]:.12e}"
    )

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

for fn in [
    "fair_validation_pk_mean.npy",
    "fair_validation_pk_median.npy",
    "fair_validation_pk_all.npy",
    "fair_validation_pk_config.json",
]:
    print(OUT / fn)
