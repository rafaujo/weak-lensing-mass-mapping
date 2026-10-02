from project_paths import PATHS
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(PATHS.code_root))


# =============================================================================
# CONFIGURATION
# =============================================================================
N = 176
J = 4
PIXEL_ARCMIN = 2.0

BASE = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

# Frozen Gaussian+Starlet validation reconstruction.
GS_FILE = Path(
    str(PATHS.work_root / 'validation/reweighted_gaussian_starlet/stage1_gaussian_starlet_rec.npy')
)

# Root under which we try to locate MCALens validation maps.
MC_ROOT = Path(
    str(PATHS.work_root / 'validation/mcalens')
)

OUT = Path(
    str(PATHS.work_root / 'validation/multiscale_fusion')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
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

MAP_SHAPE_GRID = (
    3,
    10,
    8,
    N,
    N,
)

MAP_SHAPE_FLAT = (
    240,
    N,
    N,
)


# =============================================================================
# HYBRIDS
# =============================================================================
#
# Starlet convention:
#
#   j=0 : finest detail
#   j=1
#   j=2
#   j=3 : coarsest penalised detail
#   cJ  : very-low-frequency residual
#
# Every hybrid keeps cJ from Gaussian+Starlet.
#
# H1:
#   GS: j0 + cJ
#   MC: j1,j2,j3
#
# H2:
#   GS: j0,j1 + cJ
#   MC: j2,j3
#
# H3:
#   GS: j0,j1,j2 + cJ
#   MC: j3
#
# H4:
#   GS: j0,j3 + cJ
#   MC: j1,j2
#
HYBRID_DEFINITION = {
    "H1": {
        "GS": [0],
        "MC": [1, 2, 3],
    },
    "H2": {
        "GS": [0, 1],
        "MC": [2, 3],
    },
    "H3": {
        "GS": [0, 1, 2],
        "MC": [3],
    },
    "H4": {
        "GS": [0, 3],
        "MC": [1, 2],
    },
}

METHODS = [
    "GAUSSIAN_STARLET",
    "MCALENS",
    "H1",
    "H2",
    "H3",
    "H4",
]


# =============================================================================
# FROZEN REFERENCE VALUES
# =============================================================================
EXPECTED_GS_NMSE = 0.6469240253547854
EXPECTED_GS_PCC = 0.5927316056309856

EXPECTED_MC_NMSE = 0.658007
EXPECTED_MC_PCC = 0.591189


# =============================================================================
# SPECTRA
# =============================================================================
N_ELL_BINS = 12
ELL_MIN = 90.0
ELL_MAX = 3000.0


# =============================================================================
# PEAKS
# =============================================================================
SMOOTH_ARCMIN = [
    2.0,
    4.0,
    6.0,
]

TRUTH_NU_THRESHOLDS = [
    2.0,
    2.5,
    3.0,
]

REC_NU_THRESHOLDS = np.arange(
    0.5,
    3.5001,
    0.25,
)

MATCH_RADIUS_ARCMIN = 4.0

MATCH_RADIUS_PIX = (
    MATCH_RADIUS_ARCMIN
    / PIXEL_ARCMIN
)

MIN_SMOOTH_SUPPORT = 0.95
EDGE_SIGMA = 3.0


# =============================================================================
# BASIC METRICS
# =============================================================================
def ms_metrics(rec, truth, mask):
    m = mask.astype(bool)

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[m]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[m]

    # Mass-sheet invariant.
    r -= r.mean()
    t -= t.mean()

    mse = np.mean(
        (r - t) ** 2
    )

    nmse = (
        mse
        / np.mean(t**2)
    )

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


# =============================================================================
# EXACT FOURIER STARLET FILTERS
# =============================================================================
def build_starlet_filters(
    H,
    W,
    J,
):
    """
    Same periodic B3-spline Starlet used in the
    Gaussian+Starlet solver.

    The J detail transfer functions plus the final
    coarse residual add exactly to one.
    """

    wy = (
        2.0
        * np.pi
        * np.fft.fftfreq(H)
    )

    wx = (
        2.0
        * np.pi
        * np.fft.fftfreq(W)
    )

    WY, WX = np.meshgrid(
        wy,
        wx,
        indexing="ij",
    )

    cumulative = np.ones(
        (H, W),
        dtype=np.float64,
    )

    bands = []

    for j in range(J):

        step = 2**j

        hy = (
            6.0
            + 8.0
            * np.cos(
                step * WY
            )
            + 2.0
            * np.cos(
                2.0
                * step
                * WY
            )
        ) / 16.0

        hx = (
            6.0
            + 8.0
            * np.cos(
                step * WX
            )
            + 2.0
            * np.cos(
                2.0
                * step
                * WX
            )
        ) / 16.0

        h2 = hy * hx

        bands.append(
            cumulative
            * (
                1.0
                - h2
            )
        )

        cumulative = (
            cumulative
            * h2
        )

    bands = np.asarray(
        bands,
        dtype=np.float64,
    )

    coarse = cumulative

    closure = (
        np.sum(
            bands,
            axis=0,
        )
        + coarse
    )

    closure_error = float(
        np.max(
            np.abs(
                closure - 1.0
            )
        )
    )

    return (
        bands,
        coarse,
        closure_error,
    )


# =============================================================================
# INSTANTIATE STARLET FILTERS
# =============================================================================
BANDS_F, COARSE_F, CLOSURE_ERROR = build_starlet_filters(
    N,
    N,
    J,
)

print(
    "Starlet closure error:",
    f"{CLOSURE_ERROR:.3e}",
)

if CLOSURE_ERROR > 1e-12:
    raise RuntimeError(
        f"Starlet filters do not close exactly: "
        f"{CLOSURE_ERROR:.3e}"
    )


def starlet_decompose_batch(
    maps,
):
    """
    maps:
        (B,H,W)

    returns:
        details : (B,J,H,W)
        coarse  : (B,H,W)
    """

    X = np.fft.fft2(
        maps,
        axes=(-2, -1),
        norm="ortho",
    )

    details = []

    for j in range(J):

        dj = np.fft.ifft2(
            X
            * BANDS_F[j][
                None,
                :,
                :,
            ],
            axes=(-2, -1),
            norm="ortho",
        ).real

        details.append(
            dj.astype(
                np.float32
            )
        )

    details = np.stack(
        details,
        axis=1,
    )

    coarse = np.fft.ifft2(
        X
        * COARSE_F[
            None,
            :,
            :,
        ],
        axes=(-2, -1),
        norm="ortho",
    ).real.astype(
        np.float32
    )

    return (
        details,
        coarse,
    )


# =============================================================================
# LOAD FAIR VALIDATION TRUTH/MASK
# =============================================================================
print()
print("=" * 110)
print("MULTISCALE FUSION — VALIDATION")
print("=" * 110)

mask_full = np.load(
    MASK_FILE
).astype(bool)

truths = []
masks = []
meta = []

for c in range(3):
    for r in range(10):

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
                dtype=np.float32,
            )

        for p, (
            r0,
            r1,
        ) in enumerate(
            PATCH_ROWS
        ):

            truths.append(
                truth_full[
                    r0:r1,
                    :
                ].copy()
            )

            masks.append(
                mask_full[
                    r0:r1,
                    :
                ].copy()
            )

            meta.append(
                (
                    c,
                    r,
                    p,
                )
            )

truths = np.asarray(
    truths,
    dtype=np.float32,
)

masks = np.asarray(
    masks,
    dtype=bool,
)

assert truths.shape == (
    240,
    N,
    N,
)

print(
    "truths         :",
    truths.shape,
)

print(
    "Starlet closure:",
    CLOSURE_ERROR,
)


# =============================================================================
# LOAD GAUSSIAN + STARLET VALIDATION MAPS
# =============================================================================
if not GS_FILE.exists():
    raise FileNotFoundError(
        GS_FILE
    )

gs_grid = np.load(
    GS_FILE,
    mmap_mode="r",
)

if gs_grid.shape != MAP_SHAPE_GRID:
    raise RuntimeError(
        f"Unexpected G+S shape: "
        f"{gs_grid.shape}"
    )

gs = np.empty(
    MAP_SHAPE_FLAT,
    dtype=np.float32,
)

for i, (
    c,
    r,
    p,
) in enumerate(meta):
    gs[i] = gs_grid[
        c,
        r,
        p,
    ]

del gs_grid


# =============================================================================
# LOCATE / LOAD MCALENS VALIDATION MAPS
# =============================================================================
def try_load_5d_mcalens():
    """
    Try to find an already consolidated MCALens reconstruction
    array under the validation/mcalens tree.
    """

    candidates = []

    for fn in MC_ROOT.rglob(
        "*.npy"
    ):
        name = fn.name.lower()

        # Ignore known 1-D spectra and obvious metrics.
        if (
            "pk_" in name
            or "spectrum" in name
            or "metric" in name
            or "summary" in name
        ):
            continue

        candidates.append(
            fn
        )

    for fn in candidates:
        try:
            arr = np.load(
                fn,
                mmap_mode="r",
            )
        except Exception:
            continue

        if arr.shape == MAP_SHAPE_GRID:

            print(
                "Found consolidated MCALens maps:",
                fn,
            )

            out = np.empty(
                MAP_SHAPE_FLAT,
                dtype=np.float32,
            )

            for i, (
                c,
                r,
                p,
            ) in enumerate(meta):

                out[i] = arr[
                    c,
                    r,
                    p,
                ]

            return out

        if arr.shape == MAP_SHAPE_FLAT:

            print(
                "Found flat MCALens maps:",
                fn,
            )

            return np.asarray(
                arr,
                dtype=np.float32,
            ).copy()

    return None


def load_npz_reconstruction(
    fn,
):
    preferred_keys = [
        "mcalens",
        "reconstruction",
        "rec",
        "kappa",
        "map",
    ]

    with np.load(fn) as z:

        for key in preferred_keys:

            if key in z:

                arr = np.asarray(
                    z[key]
                )

                if arr.shape == (
                    N,
                    N,
                ):
                    return arr.astype(
                        np.float32
                    )

        # Fallback: first 176x176 numeric array.
        for key in z.files:

            arr = np.asarray(
                z[key]
            )

            if (
                arr.shape
                == (
                    N,
                    N,
                )
                and np.issubdtype(
                    arr.dtype,
                    np.number,
                )
            ):
                return arr.astype(
                    np.float32
                )

    raise RuntimeError(
        f"No 176x176 reconstruction found in {fn}"
    )


def locate_case_file(
    c,
    r,
    p,
):
    direct = [
        MC_ROOT
        / "recon"
        / f"C{c}_R{r}_P{p}.npz",

        MC_ROOT
        / f"C{c}_R{r}_P{p}.npz",
    ]

    for fn in direct:
        if fn.exists():
            return fn

    patterns = [
        f"C{c}_R{r}_P{p}.npz",
        f"C{c}_R{r}_P{p}_*.npz",
    ]

    hits = []

    for pat in patterns:
        hits.extend(
            MC_ROOT.rglob(
                pat
            )
        )

    # Remove duplicates.
    hits = list(
        dict.fromkeys(
            hits
        )
    )

    if len(hits) == 1:
        return hits[0]

    if len(hits) > 1:

        # Prefer a path containing "recon".
        recon_hits = [
            x
            for x in hits
            if "recon" in str(x).lower()
        ]

        if len(recon_hits) == 1:
            return recon_hits[0]

        raise RuntimeError(
            f"Ambiguous MCALens files for "
            f"C{c} R{r} P{p}:\n"
            + "\n".join(
                str(x)
                for x in hits
            )
        )

    return None


mc = try_load_5d_mcalens()

if mc is None:

    print()
    print(
        "No consolidated MCALens map array found."
    )

    print(
        "Trying per-patch NPZ reconstructions..."
    )

    mc = np.empty(
        MAP_SHAPE_FLAT,
        dtype=np.float32,
    )

    first_file = None

    for i, (
        c,
        r,
        p,
    ) in enumerate(meta):

        fn = locate_case_file(
            c,
            r,
            p,
        )

        if fn is None:

            # Helpful diagnostic.
            print()
            print(
                "Could not find MCALens map for:",
                f"C{c}_R{r}_P{p}",
            )

            print(
                "Files currently under:",
                MC_ROOT,
            )

            for x in list(
                MC_ROOT.rglob("*")
            )[:50]:
                print(
                    "   ",
                    x,
                )

            raise FileNotFoundError(
                f"MCALens reconstruction missing "
                f"for C{c}_R{r}_P{p}"
            )

        if first_file is None:
            first_file = fn

            print(
                "First MCALens reconstruction:",
                first_file,
            )

        mc[i] = (
            load_npz_reconstruction(
                fn
            )
        )

        if (
            (i + 1) % 40
            == 0
        ):
            print(
                f"MCALens {i+1:3d}/240",
                flush=True,
            )


# =============================================================================
# BASELINE SANITY
# =============================================================================
def summarize_maps(
    name,
    arr,
):
    vals = []

    for i in range(240):
        vals.append(
            ms_metrics(
                arr[i],
                truths[i],
                masks[i],
            )
        )

    vals = np.asarray(
        vals,
        dtype=np.float64,
    )

    return {
        "method": name,
        "mse": float(
            vals[:, 0].mean()
        ),
        "nmse": float(
            vals[:, 1].mean()
        ),
        "pcc": float(
            vals[:, 2].mean()
        ),
        "std_ratio": float(
            vals[:, 3].mean()
        ),
    }


print()
print("=" * 110)
print("BASELINE SANITY")
print("=" * 110)

gs_check = summarize_maps(
    "GAUSSIAN_STARLET",
    gs,
)

mc_check = summarize_maps(
    "MCALENS",
    mc,
)

for x in [
    gs_check,
    mc_check,
]:
    print(
        f"{x['method']:20s} "
        f"NMSE={x['nmse']:.9f} "
        f"PCC={x['pcc']:.6f} "
        f"std={x['std_ratio']:.6f}"
    )

if abs(
    gs_check["nmse"]
    - EXPECTED_GS_NMSE
) > 2e-3:

    raise RuntimeError(
        "Gaussian+Starlet maps do not reproduce "
        "the frozen validation benchmark."
    )

if abs(
    mc_check["nmse"]
    - EXPECTED_MC_NMSE
) > 3e-3:

    raise RuntimeError(
        "\nMCALens maps do not reproduce the frozen "
        "validation benchmark closely enough.\n"
        f"Obtained: {mc_check['nmse']:.9f}\n"
        f"Expected: {EXPECTED_MC_NMSE:.9f}\n"
        "Do not continue with mismatched maps."
    )


# =============================================================================
# BUILD EXACT MULTISCALE HYBRIDS
# =============================================================================
print()
print("=" * 110)
print("BUILDING EXACT STARLET HYBRIDS")
print("=" * 110)

hybrids = {
    name: np.empty(
        MAP_SHAPE_FLAT,
        dtype=np.float32,
    )
    for name in HYBRID_DEFINITION
}

BATCH = 16

max_reconstruction_error_gs = 0.0
max_reconstruction_error_mc = 0.0

for ib in range(
    0,
    240,
    BATCH,
):

    ie = min(
        ib + BATCH,
        240,
    )

    gs_d, gs_c = (
        starlet_decompose_batch(
            gs[
                ib:ie
            ]
        )
    )

    mc_d, mc_c = (
        starlet_decompose_batch(
            mc[
                ib:ie
            ]
        )
    )

    # ---------------------------------------------------------
    # Verify exact decomposition/reconstruction.
    # ---------------------------------------------------------
    gs_back = (
        gs_c
        + np.sum(
            gs_d,
            axis=1,
        )
    )

    mc_back = (
        mc_c
        + np.sum(
            mc_d,
            axis=1,
        )
    )

    egs = np.max(
        np.abs(
            gs_back
            - gs[
                ib:ie
            ]
        )
    )

    emc = np.max(
        np.abs(
            mc_back
            - mc[
                ib:ie
            ]
        )
    )

    max_reconstruction_error_gs = max(
        max_reconstruction_error_gs,
        float(egs),
    )

    max_reconstruction_error_mc = max(
        max_reconstruction_error_mc,
        float(emc),
    )

    # ---------------------------------------------------------
    # Hybrid candidates.
    #
    # All use the G+S very-low-frequency residual.
    # ---------------------------------------------------------
    for name, definition in (
        HYBRID_DEFINITION.items()
    ):

        h = gs_c.copy()

        for j in definition[
            "GS"
        ]:
            h += gs_d[
                :,
                j,
            ]

        for j in definition[
            "MC"
        ]:
            h += mc_d[
                :,
                j,
            ]

        hybrids[
            name
        ][
            ib:ie
        ] = h.astype(
            np.float32
        )

    print(
        f"{ie:3d}/240",
        flush=True,
    )

print()
print(
    "max GS Starlet reconstruction error:",
    f"{max_reconstruction_error_gs:.3e}",
)

print(
    "max MC Starlet reconstruction error:",
    f"{max_reconstruction_error_mc:.3e}",
)

if (
    max_reconstruction_error_gs
    > 1e-5
    or
    max_reconstruction_error_mc
    > 1e-5
):
    raise RuntimeError(
        "Starlet decomposition is not sufficiently exact."
    )


# =============================================================================
# SAVE HYBRID MAPS
# =============================================================================
for name, arr in (
    hybrids.items()
):
    np.save(
        OUT
        / f"{name}_validation_maps.npy",
        arr,
    )


# =============================================================================
# METHOD DICTIONARY
# =============================================================================
maps = {
    "GAUSSIAN_STARLET": gs,
    "MCALENS": mc,
    **hybrids,
}


# =============================================================================
# FIELD-LEVEL METRICS
# =============================================================================
print()
print("=" * 120)
print("FIELD-LEVEL VALIDATION METRICS")
print("=" * 120)

metric_rows = []

for method in METHODS:

    for i, (
        c,
        r,
        p,
    ) in enumerate(meta):

        (
            mse,
            nmse,
            pcc,
            stdr,
        ) = ms_metrics(
            maps[
                method
            ][i],
            truths[i],
            masks[i],
        )

        metric_rows.append(
            {
                "method": method,
                "c": c,
                "r": r,
                "p": p,
                "mse": mse,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": stdr,
            }
        )

metrics = pd.DataFrame(
    metric_rows
)

metrics.to_csv(
    OUT
    / "field_metrics_all.csv",
    index=False,
)

field_summary = (
    metrics
    .groupby(
        "method"
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
    )
    .reset_index()
    .sort_values(
        "nmse_mean"
    )
)

field_summary.to_csv(
    OUT
    / "field_metrics_summary.csv",
    index=False,
)

print(
    field_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.8f}"
        ),
    )
)


# =============================================================================
# BY COSMOLOGY
# =============================================================================
by_cosmo = (
    metrics
    .groupby(
        [
            "method",
            "c",
        ]
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

by_cosmo.to_csv(
    OUT
    / "field_metrics_by_cosmology.csv",
    index=False,
)


# =============================================================================
# REALIZATION-BLOCK BOOTSTRAP FOR FIELD METRICS
# =============================================================================
def field_bootstrap_vs_baseline(
    candidate,
    baseline,
    nboot=20000,
    seed=12345,
):
    a = metrics[
        metrics.method
        == candidate
    ][
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    b = metrics[
        metrics.method
        == baseline
    ][
        [
            "c",
            "r",
            "p",
            "nmse",
            "pcc",
        ]
    ].copy()

    a = a.rename(
        columns={
            "nmse": "nmse_a",
            "pcc": "pcc_a",
        }
    )

    b = b.rename(
        columns={
            "nmse": "nmse_b",
            "pcc": "pcc_b",
        }
    )

    q = a.merge(
        b,
        on=[
            "c",
            "r",
            "p",
        ],
        validate="one_to_one",
    )

    block = (
        q
        .groupby(
            "r"
        )
        .agg(
            nmse_a=(
                "nmse_a",
                "mean",
            ),
            nmse_b=(
                "nmse_b",
                "mean",
            ),
            pcc_a=(
                "pcc_a",
                "mean",
            ),
            pcc_b=(
                "pcc_b",
                "mean",
            ),
        )
        .sort_index()
    )

    na = block[
        "nmse_a"
    ].to_numpy()

    nb = block[
        "nmse_b"
    ].to_numpy()

    pa = block[
        "pcc_a"
    ].to_numpy()

    pb = block[
        "pcc_b"
    ].to_numpy()

    rng = np.random.default_rng(
        seed
    )

    inds = rng.integers(
        0,
        len(block),
        size=(
            nboot,
            len(block),
        ),
    )

    # Positive = candidate better.
    gain_nmse = (
        100.0
        * (
            nb[inds].mean(axis=1)
            - na[inds].mean(axis=1)
        )
        / nb[inds].mean(axis=1)
    )

    gain_pcc = (
        pa[inds].mean(axis=1)
        - pb[inds].mean(axis=1)
    )

    obs_nmse = (
        100.0
        * (
            nb.mean()
            - na.mean()
        )
        / nb.mean()
    )

    obs_pcc = (
        pa.mean()
        - pb.mean()
    )

    lo_n, hi_n = np.percentile(
        gain_nmse,
        [
            2.5,
            97.5,
        ],
    )

    lo_p, hi_p = np.percentile(
        gain_pcc,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "candidate": candidate,
        "baseline": baseline,

        "nmse_gain_pct": obs_nmse,
        "nmse_gain_ci_lo": lo_n,
        "nmse_gain_ci_hi": hi_n,

        "pcc_delta": obs_pcc,
        "pcc_ci_lo": lo_p,
        "pcc_ci_hi": hi_p,

        "block_wins_nmse": int(
            np.sum(
                na < nb
            )
        ),

        "block_total": len(
            block
        ),
    }


boot_rows = []

for candidate in [
    "H1",
    "H2",
    "H3",
    "H4",
]:
    for baseline in [
        "GAUSSIAN_STARLET",
        "MCALENS",
    ]:
        boot_rows.append(
            field_bootstrap_vs_baseline(
                candidate,
                baseline,
            )
        )

field_boot = pd.DataFrame(
    boot_rows
)

field_boot.to_csv(
    OUT
    / "field_bootstrap.csv",
    index=False,
)

print()
print("=" * 135)
print("FIELD BOOTSTRAP")
print("=" * 135)

print(
    field_boot.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


# =============================================================================
# SPECTRAL GRID
# =============================================================================
pix_rad = (
    PIXEL_ARCMIN
    / 60.0
    * np.pi
    / 180.0
)

ky = (
    2.0
    * np.pi
    * np.fft.fftfreq(
        N,
        d=pix_rad,
    )
)

kx = (
    2.0
    * np.pi
    * np.fft.fftfreq(
        N,
        d=pix_rad,
    )
)

KY, KX = np.meshgrid(
    ky,
    kx,
    indexing="ij",
)

ELL = np.sqrt(
    KX**2
    + KY**2
)

ELL_EDGES = np.geomspace(
    ELL_MIN,
    ELL_MAX,
    N_ELL_BINS + 1,
)


def centered_masked_map(
    x,
    mask,
):
    out = np.zeros_like(
        x,
        dtype=np.float64,
    )

    m = mask.astype(bool)

    out[m] = (
        x[m]
        - np.mean(
            x[m]
        )
    )

    return out


def spectral_case(
    truth,
    rec,
    mask,
):
    t = centered_masked_map(
        truth,
        mask,
    )

    r = centered_masked_map(
        rec,
        mask,
    )

    T = np.fft.fft2(
        t,
        norm="ortho",
    )

    R = np.fft.fft2(
        r,
        norm="ortho",
    )

    Ptt = np.abs(T) ** 2
    Prr = np.abs(R) ** 2

    Prt = np.real(
        R
        * np.conj(T)
    )

    result = []

    for b in range(
        N_ELL_BINS
    ):

        sel = (
            (ELL >= ELL_EDGES[b])
            &
            (ELL < ELL_EDGES[b + 1])
        )

        if not np.any(sel):
            continue

        tt = float(
            np.mean(
                Ptt[sel]
            )
        )

        rr = float(
            np.mean(
                Prr[sel]
            )
        )

        rt = float(
            np.mean(
                Prt[sel]
            )
        )

        if tt <= 0:
            continue

        tcross = (
            rt / tt
        )

        aauto = (
            rr / tt
        )

        rell = (
            rt
            / np.sqrt(
                max(
                    rr * tt,
                    1e-30,
                )
            )
        )

        result.append(
            {
                "bin": b,

                "ell": float(
                    np.mean(
                        ELL[sel]
                    )
                ),

                "T_cross": tcross,

                "A_auto": aauto,

                "r_ell": rell,
            }
        )

    return result


# =============================================================================
# RUN SPECTRA
# =============================================================================
print()
print("=" * 110)
print("PSEUDO-C_ELL")
print("=" * 110)

spec_rows = []

for i, (
    c,
    r,
    p,
) in enumerate(meta):

    for method in METHODS:

        ss = spectral_case(
            truths[i],
            maps[method][i],
            masks[i],
        )

        for x in ss:

            x.update(
                {
                    "method": method,
                    "c": c,
                    "r": r,
                    "p": p,
                }
            )

            spec_rows.append(
                x
            )

    if (
        (i + 1)
        % 40
        == 0
    ):
        print(
            f"{i+1:3d}/240",
            flush=True,
        )

spec = pd.DataFrame(
    spec_rows
)

spec.to_csv(
    OUT
    / "spectral_all.csv",
    index=False,
)

spec_summary = (
    spec
    .groupby(
        [
            "method",
            "bin",
        ]
    )
    .agg(
        ell=(
            "ell",
            "mean",
        ),

        T_cross_mean=(
            "T_cross",
            "mean",
        ),

        A_auto_mean=(
            "A_auto",
            "mean",
        ),

        r_ell_mean=(
            "r_ell",
            "mean",
        ),
    )
    .reset_index()
)

spec_summary.to_csv(
    OUT
    / "spectral_summary.csv",
    index=False,
)


# =============================================================================
# SPECTRAL ERROR SUMMARY
# =============================================================================
spectral_score_rows = []

for method in METHODS:

    q = spec_summary[
        spec_summary.method
        == method
    ]

    spectral_score_rows.append(
        {
            "method": method,

            "mean_abs_T_error": float(
                np.mean(
                    np.abs(
                        q.T_cross_mean
                        - 1.0
                    )
                )
            ),

            "mean_abs_A_error": float(
                np.mean(
                    np.abs(
                        q.A_auto_mean
                        - 1.0
                    )
                )
            ),

            "mean_r_ell": float(
                q.r_ell_mean.mean()
            ),
        }
    )

spectral_score = pd.DataFrame(
    spectral_score_rows
)

spectral_score.to_csv(
    OUT
    / "spectral_score_summary.csv",
    index=False,
)

print()
print(
    spectral_score.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


# =============================================================================
# PEAK UTILITIES
# =============================================================================
def mask_normalized_smooth(
    image,
    mask,
    sigma_pix,
):
    m = mask.astype(
        np.float64
    )

    centered = np.zeros_like(
        image,
        dtype=np.float64,
    )

    obs = mask.astype(bool)

    centered[obs] = (
        image[obs]
        - np.mean(
            image[obs]
        )
    )

    numerator = gaussian_filter(
        centered * m,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    denominator = gaussian_filter(
        m,
        sigma=sigma_pix,
        mode="constant",
        cval=0.0,
    )

    smoothed = np.zeros_like(
        numerator
    )

    good = (
        denominator > 1e-12
    )

    smoothed[good] = (
        numerator[good]
        / denominator[good]
    )

    valid = (
        denominator
        >= MIN_SMOOTH_SUPPORT
    )

    margin = int(
        np.ceil(
            EDGE_SIGMA
            * sigma_pix
        )
    )

    if margin > 0:

        valid[
            :margin,
            :
        ] = False

        valid[
            -margin:,
            :
        ] = False

        valid[
            :,
            :margin
        ] = False

        valid[
            :,
            -margin:
        ] = False

    return (
        smoothed,
        valid,
    )


def make_nu_map(
    image,
    mask,
    sigma_pix,
):
    sm, valid = (
        mask_normalized_smooth(
            image,
            mask,
            sigma_pix,
        )
    )

    sigma_map = np.std(
        sm[valid]
    )

    if sigma_map <= 0:
        raise RuntimeError(
            "Invalid smoothed-map sigma."
        )

    return (
        sm,
        sm / sigma_map,
        valid,
    )


def peak_catalog(
    nu,
    valid,
    threshold,
):
    mx = maximum_filter(
        nu,
        size=3,
        mode="constant",
        cval=-np.inf,
    )

    is_peak = (
        valid
        &
        (nu == mx)
        &
        (nu >= threshold)
    )

    coords = np.argwhere(
        is_peak
    )

    if len(coords) == 0:

        return (
            np.empty(
                (0, 2),
                dtype=np.int32,
            ),
            np.empty(
                (0,),
                dtype=np.float64,
            ),
        )

    vals = nu[
        coords[:, 0],
        coords[:, 1],
    ]

    return (
        coords.astype(
            np.int32
        ),
        vals.astype(
            np.float64
        ),
    )


def match_catalogs(
    truth_coords,
    rec_coords,
    radius_pix,
):
    if (
        len(truth_coords) == 0
        or len(rec_coords) == 0
    ):
        return []

    tree = cKDTree(
        rec_coords
    )

    candidates = []

    for it, coord in enumerate(
        truth_coords
    ):

        inds = tree.query_ball_point(
            coord,
            radius_pix,
        )

        for ir in inds:

            d = float(
                np.linalg.norm(
                    coord
                    - rec_coords[
                        ir
                    ]
                )
            )

            candidates.append(
                (
                    d,
                    it,
                    ir,
                )
            )

    candidates.sort(
        key=lambda x: x[0]
    )

    used_t = set()
    used_r = set()
    matches = []

    for d, it, ir in candidates:

        if (
            it in used_t
            or ir in used_r
        ):
            continue

        used_t.add(
            it
        )

        used_r.add(
            ir
        )

        matches.append(
            (
                it,
                ir,
                d,
            )
        )

    return matches


def local_max_value(
    image,
    coord,
    valid,
    radius_pix,
):
    y0, x0 = coord

    rr = int(
        np.ceil(
            radius_pix
        )
    )

    y1 = max(
        0,
        y0 - rr,
    )

    y2 = min(
        N,
        y0 + rr + 1,
    )

    x1 = max(
        0,
        x0 - rr,
    )

    x2 = min(
        N,
        x0 + rr + 1,
    )

    yy, xx = np.mgrid[
        y1:y2,
        x1:x2
    ]

    dist = np.sqrt(
        (yy - y0) ** 2
        + (xx - x0) ** 2
    )

    ok = (
        valid[
            y1:y2,
            x1:x2
        ]
        &
        (
            dist
            <= radius_pix
        )
    )

    if not np.any(ok):
        return np.nan

    return float(
        np.max(
            image[
                y1:y2,
                x1:x2
            ][ok]
        )
    )


# =============================================================================
# PEAK ANALYSIS
# =============================================================================
print()
print("=" * 110)
print("PEAK ANALYSIS")
print("=" * 110)

peak_pr_rows = []
peak_height_rows = []

for smooth_arcmin in (
    SMOOTH_ARCMIN
):

    sigma_pix = (
        smooth_arcmin
        / PIXEL_ARCMIN
    )

    print()
    print(
        f"Smoothing = "
        f"{smooth_arcmin:g} arcmin"
    )

    for i, (
        c,
        r,
        p,
    ) in enumerate(meta):

        (
            truth_sm,
            truth_nu,
            truth_valid,
        ) = make_nu_map(
            truths[i],
            masks[i],
            sigma_pix,
        )

        rec_cache = {}

        for method in METHODS:

            (
                rec_sm,
                rec_nu,
                rec_valid,
            ) = make_nu_map(
                maps[
                    method
                ][i],
                masks[i],
                sigma_pix,
            )

            rec_cache[
                method
            ] = (
                rec_sm,
                rec_nu,
                (
                    truth_valid
                    &
                    rec_valid
                ),
            )

        for truth_thr in (
            TRUTH_NU_THRESHOLDS
        ):

            (
                truth_coords,
                truth_nu_vals,
            ) = peak_catalog(
                truth_nu,
                truth_valid,
                truth_thr,
            )

            # -------------------------------------------------
            # Peak-height response
            # -------------------------------------------------
            for method in METHODS:

                (
                    rec_sm,
                    rec_nu,
                    valid,
                ) = rec_cache[
                    method
                ]

                for ip, coord in enumerate(
                    truth_coords
                ):

                    y0, x0 = coord

                    truth_amp = float(
                        truth_sm[
                            y0,
                            x0,
                        ]
                    )

                    local_amp = (
                        local_max_value(
                            rec_sm,
                            coord,
                            valid,
                            MATCH_RADIUS_PIX,
                        )
                    )

                    if abs(
                        truth_amp
                    ) > 1e-15:

                        ratio = (
                            local_amp
                            / truth_amp
                        )

                    else:
                        ratio = np.nan

                    peak_height_rows.append(
                        {
                            "method": method,

                            "c": c,
                            "r": r,
                            "p": p,

                            "smooth_arcmin": (
                                smooth_arcmin
                            ),

                            "truth_nu_threshold": (
                                truth_thr
                            ),

                            "truth_nu": float(
                                truth_nu_vals[
                                    ip
                                ]
                            ),

                            "truth_amp": (
                                truth_amp
                            ),

                            "rec_local_amp": (
                                local_amp
                            ),

                            "local_ratio": (
                                ratio
                            ),
                        }
                    )

            # -------------------------------------------------
            # PR curves
            # -------------------------------------------------
            for method in METHODS:

                (
                    rec_sm,
                    rec_nu,
                    valid,
                ) = rec_cache[
                    method
                ]

                if len(
                    truth_coords
                ):

                    keep_truth = valid[
                        truth_coords[
                            :,
                            0,
                        ],
                        truth_coords[
                            :,
                            1,
                        ],
                    ]

                    tc = truth_coords[
                        keep_truth
                    ]

                else:
                    tc = truth_coords

                for rec_thr in (
                    REC_NU_THRESHOLDS
                ):

                    (
                        rec_coords,
                        _,
                    ) = peak_catalog(
                        rec_nu,
                        valid,
                        rec_thr,
                    )

                    matches = (
                        match_catalogs(
                            tc,
                            rec_coords,
                            MATCH_RADIUS_PIX,
                        )
                    )

                    nm = len(
                        matches
                    )

                    nt = len(
                        tc
                    )

                    nr = len(
                        rec_coords
                    )

                    peak_pr_rows.append(
                        {
                            "method": method,

                            "c": c,
                            "r": r,
                            "p": p,

                            "smooth_arcmin": (
                                smooth_arcmin
                            ),

                            "truth_nu_threshold": (
                                truth_thr
                            ),

                            "rec_nu_threshold": (
                                float(
                                    rec_thr
                                )
                            ),

                            "n_truth": nt,
                            "n_rec": nr,
                            "n_match": nm,
                        }
                    )

        if (
            (i + 1)
            % 80
            == 0
        ):
            print(
                f"  {i+1:3d}/240",
                flush=True,
            )


# =============================================================================
# SAVE RAW PEAK TABLES
# =============================================================================
peak_pr = pd.DataFrame(
    peak_pr_rows
)

peak_height = pd.DataFrame(
    peak_height_rows
)

peak_pr.to_csv(
    OUT
    / "peak_PR_all.csv",
    index=False,
)

peak_height.to_csv(
    OUT
    / "peak_height_all.csv",
    index=False,
)


# =============================================================================
# AGGREGATE PR CURVES
# =============================================================================
pr_summary = (
    peak_pr
    .groupby(
        [
            "method",
            "smooth_arcmin",
            "truth_nu_threshold",
            "rec_nu_threshold",
        ]
    )
    .agg(
        n_truth=(
            "n_truth",
            "sum",
        ),

        n_rec=(
            "n_rec",
            "sum",
        ),

        n_match=(
            "n_match",
            "sum",
        ),
    )
    .reset_index()
)

pr_summary[
    "completeness"
] = (
    pr_summary.n_match
    / pr_summary.n_truth.replace(
        0,
        np.nan,
    )
)

pr_summary[
    "purity"
] = (
    pr_summary.n_match
    / pr_summary.n_rec.replace(
        0,
        np.nan,
    )
)

pr_summary.to_csv(
    OUT
    / "peak_PR_summary.csv",
    index=False,
)


# =============================================================================
# PEAK HEIGHT SUMMARY
# =============================================================================
height_summary = (
    peak_height
    .groupby(
        [
            "method",
            "smooth_arcmin",
            "truth_nu_threshold",
        ]
    )
    .agg(
        n=(
            "local_ratio",
            "size",
        ),

        local_ratio_median=(
            "local_ratio",
            "median",
        ),

        local_ratio_mean=(
            "local_ratio",
            "mean",
        ),
    )
    .reset_index()
)

height_summary.to_csv(
    OUT
    / "peak_height_summary.csv",
    index=False,
)

print()
print("=" * 120)
print("PEAK HEIGHT SUMMARY")
print("=" * 120)

print(
    height_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# PR CURVE PREPARATION
# =============================================================================
def prepare_pr_curve(q):

    q = q[
        np.isfinite(
            q.completeness
        )
        &
        np.isfinite(
            q.purity
        )
    ].copy()

    if len(q) < 2:
        return None

    q = (
        q
        .groupby(
            "completeness",
            as_index=False,
        )["purity"]
        .max()
        .sort_values(
            "completeness"
        )
    )

    return (
        q.completeness.to_numpy(),
        q.purity.to_numpy(),
    )


# =============================================================================
# COMMON-RANGE PR AUC
# =============================================================================
auc_rows = []

for smooth in (
    SMOOTH_ARCMIN
):

    for truth_thr in (
        TRUTH_NU_THRESHOLDS
    ):

        curves = {}

        for method in METHODS:

            q = pr_summary[
                (
                    pr_summary.method
                    == method
                )
                &
                (
                    pr_summary.smooth_arcmin
                    == smooth
                )
                &
                (
                    pr_summary.truth_nu_threshold
                    == truth_thr
                )
            ]

            curves[
                method
            ] = prepare_pr_curve(
                q
            )

        valid_curves = [
            x
            for x in curves.values()
            if x is not None
        ]

        cmin = max(
            np.min(c)
            for c, _
            in valid_curves
        )

        cmax = min(
            np.max(c)
            for c, _
            in valid_curves
        )

        if cmax <= cmin:
            continue

        grid = np.linspace(
            cmin,
            cmax,
            500,
        )

        for method in METHODS:

            curve = curves[
                method
            ]

            if curve is None:
                continue

            c, p = curve

            interp = np.interp(
                grid,
                c,
                p,
            )

            auc = (
                np.trapz(
                    interp,
                    grid,
                )
                / (
                    cmax
                    - cmin
                )
            )

            auc_rows.append(
                {
                    "method": method,

                    "smooth_arcmin": (
                        smooth
                    ),

                    "truth_nu_threshold": (
                        truth_thr
                    ),

                    "common_cmin": cmin,
                    "common_cmax": cmax,

                    "normalized_common_auc": (
                        auc
                    ),
                }
            )

auc = pd.DataFrame(
    auc_rows
)

auc.to_csv(
    OUT
    / "peak_PR_common_range_AUC.csv",
    index=False,
)

print()
print("=" * 125)
print("COMMON-RANGE PR AUC")
print("=" * 125)

print(
    auc.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# COMPACT SCIENCE SUMMARY
# =============================================================================
science_rows = []

for method in METHODS:

    f = field_summary[
        field_summary.method
        == method
    ].iloc[0]

    s = spectral_score[
        spectral_score.method
        == method
    ].iloc[0]

    row = {
        "method": method,

        "nmse": float(
            f.nmse_mean
        ),

        "pcc": float(
            f.pcc_mean
        ),

        "std_ratio": float(
            f.std_ratio_mean
        ),

        "mean_abs_T_error": float(
            s.mean_abs_T_error
        ),

        "mean_abs_A_error": float(
            s.mean_abs_A_error
        ),

        "mean_r_ell": float(
            s.mean_r_ell
        ),
    }

    # PR AUC at nu_true >= 2.5.
    for smooth in [
        2.0,
        4.0,
        6.0,
    ]:

        q = auc[
            (
                auc.method
                == method
            )
            &
            (
                auc.smooth_arcmin
                == smooth
            )
            &
            (
                auc.truth_nu_threshold
                == 2.5
            )
        ]

        row[
            f"PR_AUC_{int(smooth)}arc"
        ] = (
            float(
                q.normalized_common_auc.iloc[
                    0
                ]
            )
            if len(q)
            else np.nan
        )

    # Peak-height response for strong peaks nu>=3.
    for smooth in [
        2.0,
        4.0,
        6.0,
    ]:

        q = height_summary[
            (
                height_summary.method
                == method
            )
            &
            (
                height_summary.smooth_arcmin
                == smooth
            )
            &
            (
                height_summary.truth_nu_threshold
                == 3.0
            )
        ]

        row[
            f"peak_amp_{int(smooth)}arc"
        ] = (
            float(
                q.local_ratio_median.iloc[
                    0
                ]
            )
            if len(q)
            else np.nan
        )

    science_rows.append(
        row
    )


science = pd.DataFrame(
    science_rows
)

science.to_csv(
    OUT
    / "SCIENCE_SUMMARY.csv",
    index=False,
)

print()
print("=" * 170)
print("SCIENCE SUMMARY")
print("=" * 170)

print(
    science.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.5f}"
        ),
    )
)


# =============================================================================
# SIMPLE COMPLEMENTARITY CHECK
# =============================================================================
print()
print("=" * 110)
print("HYBRID DEFINITIONS")
print("=" * 110)

for name, d in (
    HYBRID_DEFINITION.items()
):

    print(
        f"{name}: "
        f"GS bands={d['GS']} + "
        f"MC bands={d['MC']} + "
        f"GS coarse residual"
    )


print()
print("=" * 110)
print("FILES")
print("=" * 110)

for fn in [
    "field_metrics_all.csv",
    "field_metrics_summary.csv",
    "field_metrics_by_cosmology.csv",
    "field_bootstrap.csv",
    "spectral_all.csv",
    "spectral_summary.csv",
    "spectral_score_summary.csv",
    "peak_PR_all.csv",
    "peak_PR_summary.csv",
    "peak_PR_common_range_AUC.csv",
    "peak_height_all.csv",
    "peak_height_summary.csv",
    "SCIENCE_SUMMARY.csv",
]:
    print(
        OUT / fn
    )

for name in [
    "H1",
    "H2",
    "H3",
    "H4",
]:
    print(
        OUT
        / f"{name}_validation_maps.npy"
    )

print()
print(
    "IMPORTANT: these candidates were evaluated "
    "ONLY on validation R0-R9."
)

print()
print(
    "MULTISCALE FUSION VALIDATION COMPLETE"
)
