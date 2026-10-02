from project_paths import PATHS
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import weak_lensing_pipeline as wl


# ============================================================
# FROZEN SCIENTIFIC SETTINGS
# ============================================================
ROOT = Path(str(PATHS.code_root))
DATA = PATHS.data_root
GLROOT = Path(str(PATHS.work_root))

MASK_FILE = DATA / "WIDE12H_bin2_2arcmin_mask.npy"
KAPPA_FILE = DATA / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
LABEL_FILE = DATA / "sampled_label.npy"

N = 176
J = 4

SIGMA_N = 0.4 / np.sqrt(2.0 * 30.0 * 4.0)

KS_SIGMA_FINAL = 1.5
KS_SIGMA_INIT = 1.0

LAMBDA = 0.05

ALPHA = np.array([
    1.0,
    0.225148,
    0.096029,
    0.046286,
], dtype=np.float32)

# Same production budget as final Starlet experiments.
NITER = 6000

# Chambolle-Pock / PDHG:
# ||A||^2 <= 1 and ||K||^2 <= 1.
# tau*sigma*(||A||^2+||K||^2) <= .98 < 1.
TAU = 0.7
SIGMA_PD = 0.7
THETA = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PATCHES = [
    (p, 176*p, 176*(p+1))
    for p in range(8)
]


# ============================================================
# FOURIER KS OPERATOR
# ============================================================
def build_ks_kernel(N, device):

    fy = torch.fft.fftfreq(
        N, d=1.0, device=device
    )
    fx = torch.fft.fftfreq(
        N, d=1.0, device=device
    )

    ky, kx = torch.meshgrid(
        fy, fx, indexing="ij"
    )

    k2 = kx*kx + ky*ky

    D = torch.zeros(
        (N, N),
        dtype=torch.complex64,
        device=device
    )

    nz = k2 > 0

    D[nz] = (
        (kx[nz]*kx[nz] - ky[nz]*ky[nz])
        + 2j*kx[nz]*ky[nz]
    ) / k2[nz]

    return D


D = build_ks_kernel(N, DEVICE)


def A_forward(x):
    """
    x: real [B,N,N]
    return complex shear [B,N,N]
    """
    return torch.fft.ifft2(
        torch.fft.fft2(x) * D
    )


def A_adjoint(g):
    """
    g: complex [B,N,N]
    return real convergence [B,N,N]
    """
    return torch.fft.ifft2(
        torch.fft.fft2(g) * torch.conj(D)
    ).real


# ============================================================
# PERIODIC GAUSSIAN SMOOTHING
# ============================================================
fy = torch.fft.fftfreq(
    N, d=1.0, device=DEVICE
)
fx = torch.fft.fftfreq(
    N, d=1.0, device=DEVICE
)

KY, KX = torch.meshgrid(
    fy, fx, indexing="ij"
)


def gaussian_filter(sigma):
    return torch.exp(
        -2.0 * np.pi**2 * sigma**2
        * (KX*KX + KY*KY)
    )


G1 = gaussian_filter(KS_SIGMA_INIT)
G15 = gaussian_filter(KS_SIGMA_FINAL)


def smooth_real(x, kernel):
    return torch.fft.ifft2(
        torch.fft.fft2(x) * kernel
    ).real


def ks_reconstruct(y_obs, kernel):
    return smooth_real(
        A_adjoint(y_obs),
        kernel
    )


# ============================================================
# EXACT FOURIER STARLET (B3 spline, a trous)
# ============================================================
#
# h_hat(f) = cos^4(pi f)
#
# c_j = H_j c_{j-1}
# w_j = c_{j-1} - c_j
#
# Analysis transfer functions:
# W_j = C_{j-1}(1-H_j)
#
# ============================================================
C = torch.ones(
    (N, N),
    dtype=torch.float32,
    device=DEVICE
)

STARLET_W = []

for j in range(J):

    dilation = 2**j

    Hj = (
        torch.cos(
            np.pi*KX*dilation
        )**4
        *
        torch.cos(
            np.pi*KY*dilation
        )**4
    )

    Cnext = C * Hj

    STARLET_W.append(
        C - Cnext
    )

    C = Cnext

STARLET_W = torch.stack(
    STARLET_W, dim=0
).to(torch.float32)

STARLET_COARSE = C

# Frame diagnostic including unpenalized coarse band.
frame_power = (
    torch.sum(
        STARLET_W**2,
        dim=0
    )
    + STARLET_COARSE**2
)

print("device =", DEVICE)
print(
    "Starlet frame power min/max =",
    float(frame_power.min()),
    float(frame_power.max())
)


def K_analysis(x):
    """
    x: [B,N,N]
    returns detail bands [B,J,N,N]
    """

    X = torch.fft.fft2(x)

    out = []

    for j in range(J):
        out.append(
            torch.fft.ifft2(
                X * STARLET_W[j]
            ).real
        )

    return torch.stack(
        out, dim=1
    )


def K_adjoint(p):
    """
    p: [B,J,N,N]
    exact adjoint of detail-only analysis operator.
    """

    acc = torch.zeros(
        p.shape[0],
        N,
        N,
        dtype=torch.complex64,
        device=p.device
    )

    for j in range(J):

        acc = acc + (
            torch.fft.fft2(
                p[:, j]
            )
            * STARLET_W[j]
        )

    return torch.fft.ifft2(
        acc
    ).real


# ============================================================
# EXACT PDHG SOLVER
# ============================================================
@torch.no_grad()
def starlet_pdhg(
    y_obs,
    data_weight,
    x0,
):
    """
    Solve

      min_x 0.5 ||sqrt(W)(Ax-y)||^2
            + lambda Sum_j alpha_j ||(Kx)_j||_1

    y_obs      complex [B,N,N]
    data_weight real [B,N,N], binary
    x0         real [B,N,N]
    """

    B = x0.shape[0]

    x = x0.clone()
    xbar = x.clone()

    # dual variable of data fidelity
    q = torch.zeros_like(
        y_obs,
        dtype=torch.complex64
    )

    # dual variables of Starlet detail bands
    p = torch.zeros(
        B, J, N, N,
        dtype=torch.float32,
        device=DEVICE
    )

    bounds = (
        LAMBDA
        * torch.tensor(
            ALPHA,
            device=DEVICE
        )
    ).view(1, J, 1, 1)

    for it in range(NITER):

        # -----------------------------
        # Data dual:
        # prox_{sigma F*}
        # -----------------------------
        Aq = A_forward(xbar)

        v = q + SIGMA_PD*Aq

        q = data_weight * (
            v - SIGMA_PD*y_obs
        ) / (1.0 + SIGMA_PD)

        # -----------------------------
        # Sparse dual:
        # projection onto |p_j| <= lam*alpha_j
        # -----------------------------
        p = (
            p
            + SIGMA_PD*K_analysis(xbar)
        )

        p = torch.maximum(
            torch.minimum(
                p,
                bounds
            ),
            -bounds
        )

        # -----------------------------
        # Primal
        # -----------------------------
        grad = (
            A_adjoint(q)
            + K_adjoint(p)
        )

        xnew = x - TAU*grad

        xbar = (
            xnew
            + THETA*(xnew-x)
        )

        x = xnew

    return x


# ============================================================
# METRICS
# ============================================================
def metrics_ms(rec, truth, mask):

    r = np.asarray(
        rec[mask],
        dtype=np.float64
    ).copy()

    t = np.asarray(
        truth[mask],
        dtype=np.float64
    ).copy()

    r -= r.mean()
    t -= t.mean()

    nmse = (
        np.dot(r-t, r-t)
        / np.dot(t, t)
    )

    rr = np.dot(r, r)
    tt = np.dot(t, t)

    pcc = (
        np.dot(r, t)
        / np.sqrt(rr*tt)
    )

    sr = r.std() / t.std()

    return (
        float(nmse),
        float(pcc),
        float(sr)
    )


# ============================================================
# DATA
# ============================================================
mask_full = np.load(
    MASK_FILE
).astype(bool)

sampled = np.load(
    KAPPA_FILE,
    mmap_mode="r"
)

labels = np.load(
    LABEL_FILE
)

assert mask_full.shape == (1424, 176)


def generate_full_case(c, r):

    truth = np.zeros(
        mask_full.shape,
        dtype=np.float32
    )

    truth[mask_full] = (
        sampled[c, r]
        .astype(np.float32)
    )

    op_full = wl.WeakLensingOperator(
        1424,
        176,
        device=DEVICE
    )

    kt = torch.from_numpy(
        truth
    ).to(
        DEVICE,
        dtype=torch.float32
    )

    with torch.no_grad():

        gamma_clean = op_full.forward(kt)

        gamma = wl.add_shape_noise(
            gamma_clean,
            sigma_n=SIGMA_N,
            seed=r
        )

    return (
        truth,
        gamma.detach()
        .cpu()
        .numpy()
        .astype(np.complex64)
    )


def load_or_generate_case(
    c,
    r,
    cache_dir,
):

    fn = cache_dir / f"C{c}_R{r}.npz"

    if fn.exists():

        d = np.load(fn)

        return (
            d["truth"],
            d["gamma"],
            float(d["omega_m"]),
        )

    truth, gamma = generate_full_case(
        c, r
    )

    omega_m = float(
        labels[c, r, 0]
    )

    np.savez_compressed(
        fn,
        truth=truth,
        gamma=gamma,
        omega_m=np.float64(omega_m),
        cosmology=np.int32(c),
        realization=np.int32(r),
        sigma_n=np.float64(SIGMA_N),
    )

    return (
        truth,
        gamma,
        omega_m
    )


# ============================================================
# RUN
# ============================================================
def run(split):

    if split == "validation":

        rs = range(0, 10)

        BASE = (
            GLROOT /
            "validation"
        )

        # Reuse the exact FAIR observations already
        # used for GLIMPSE validation.
        cache_dir = (
            BASE /
            "fair_npz"
        )

    elif split == "test":

        rs = range(10, 30)

        BASE = (
            GLROOT /
            "test"
        )

        cache_dir = (
            BASE /
            "fair_npz"
        )

        BASE.mkdir(
            parents=True,
            exist_ok=True
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    else:
        raise ValueError(split)

    rec_dir = (
        BASE /
        "our_recon"
    )

    rec_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_file = (
        BASE /
        "square_ours_metrics.csv"
    )

    fields = [
        "method",
        "c", "r", "p",
        "row0", "row1",
        "omega_m",
        "f_obs",
        "n_obs",
        "nmse",
        "pcc",
        "std_ratio",
    ]

    # Resume existing rows.
    done = set()

    if csv_file.exists():

        with open(csv_file) as f:

            rd = csv.DictReader(f)

            for row in rd:
                done.add((
                    row["method"],
                    int(row["c"]),
                    int(row["r"]),
                    int(row["p"]),
                ))

        write_header = False

    else:
        write_header = True

    fout = open(
        csv_file,
        "a",
        newline=""
    )

    writer = csv.DictWriter(
        fout,
        fieldnames=fields
    )

    if write_header:
        writer.writeheader()
        fout.flush()

    total_maps = 3*len(rs)
    imap = 0

    try:

        for c in range(3):

            for r in rs:

                imap += 1

                print()
                print("="*70)
                print(
                    f"{split.upper()} "
                    f"[{imap}/{total_maps}] "
                    f"C{c} R{r}"
                )
                print("="*70)

                truth_full, gamma_full, omega_m = (
                    load_or_generate_case(
                        c,
                        r,
                        cache_dir,
                    )
                )

                # ----------------------------------------
                # Stack all 8 square patches.
                # ----------------------------------------
                truth_stack = []
                gamma_stack = []
                mask_stack = []

                for p, r0, r1 in PATCHES:

                    truth_stack.append(
                        truth_full[r0:r1, :]
                    )

                    gamma_stack.append(
                        gamma_full[r0:r1, :]
                    )

                    mask_stack.append(
                        mask_full[r0:r1, :]
                    )

                truth_np = np.stack(
                    truth_stack
                ).astype(np.float32)

                gamma_np = np.stack(
                    gamma_stack
                ).astype(np.complex64)

                mask_np = np.stack(
                    mask_stack
                ).astype(bool)

                truth_t = torch.from_numpy(
                    truth_np
                ).to(DEVICE)

                gamma_t = torch.from_numpy(
                    gamma_np
                ).to(DEVICE)

                mask_t = torch.from_numpy(
                    mask_np.astype(
                        np.float32
                    )
                ).to(DEVICE)

                # Correct observed field:
                # y = M(Ak+n)
                y_obs = (
                    gamma_t
                    * mask_t
                )

                # ----------------------------------------
                # KS baseline sigma=1.5
                # ----------------------------------------
                ks = ks_reconstruct(
                    y_obs,
                    G15
                )

                # ----------------------------------------
                # Frozen initialization sigma=1
                # ----------------------------------------
                x0 = ks_reconstruct(
                    y_obs,
                    G1
                )

                # ----------------------------------------
                # Solve NO_MASK and MASK in one batch.
                #
                # First 8:
                # W=I -> zero-filled holes treated as data.
                #
                # Last 8:
                # W=M -> holes excluded from fidelity.
                # ----------------------------------------
                y16 = torch.cat(
                    [y_obs, y_obs],
                    dim=0
                )

                w16 = torch.cat(
                    [
                        torch.ones_like(
                            mask_t
                        ),
                        mask_t
                    ],
                    dim=0
                )

                x016 = torch.cat(
                    [x0, x0],
                    dim=0
                )

                star16 = starlet_pdhg(
                    y16,
                    w16,
                    x016,
                )

                nomask = star16[:8]
                masked = star16[8:]

                ks_np = (
                    ks.detach()
                    .cpu()
                    .numpy()
                )

                nomask_np = (
                    nomask.detach()
                    .cpu()
                    .numpy()
                )

                masked_np = (
                    masked.detach()
                    .cpu()
                    .numpy()
                )

                # ----------------------------------------
                # Save maps for later spectra / peaks.
                # ----------------------------------------
                map_file = (
                    rec_dir /
                    f"C{c}_R{r}.npz"
                )

                np.savez_compressed(
                    map_file,
                    truth=truth_np,
                    mask=mask_np,
                    ks=ks_np.astype(
                        np.float32
                    ),
                    nomask=nomask_np.astype(
                        np.float32
                    ),
                    masked=masked_np.astype(
                        np.float32
                    ),
                )

                # ----------------------------------------
                # Metrics patch by patch
                # ----------------------------------------
                methods = {
                    "KS": ks_np,
                    "STARLET_NOMASK": nomask_np,
                    "STARLET_MASK": masked_np,
                }

                for p, r0, r1 in PATCHES:

                    for method, arr in methods.items():

                        key = (
                            method,
                            c, r, p
                        )

                        if key in done:
                            continue

                        nmse, pcc, sr = (
                            metrics_ms(
                                arr[p],
                                truth_np[p],
                                mask_np[p]
                            )
                        )

                        row = {
                            "method": method,
                            "c": c,
                            "r": r,
                            "p": p,
                            "row0": r0,
                            "row1": r1,
                            "omega_m": omega_m,
                            "f_obs": float(
                                mask_np[p].mean()
                            ),
                            "n_obs": int(
                                mask_np[p].sum()
                            ),
                            "nmse": nmse,
                            "pcc": pcc,
                            "std_ratio": sr,
                        }

                        writer.writerow(row)
                        fout.flush()

                        done.add(key)

                for method, arr in methods.items():

                    vals = []

                    for p in range(8):

                        nmse, pcc, sr = metrics_ms(
                            arr[p],
                            truth_np[p],
                            mask_np[p]
                        )

                        vals.append(
                            (nmse, pcc)
                        )

                    print(
                        f"{method:15s} "
                        f"NMSE={np.mean([v[0] for v in vals]):.6f} "
                        f"PCC={np.mean([v[1] for v in vals]):.6f}"
                    )

                del (
                    truth_t,
                    gamma_t,
                    mask_t,
                    y_obs,
                    ks,
                    x0,
                    star16,
                    nomask,
                    masked,
                )

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    finally:
        fout.close()

    # ========================================================
    # Summary
    # ========================================================
    rows = []

    with open(csv_file) as f:

        rd = csv.DictReader(f)

        for x in rd:
            rows.append(x)

    print()
    print("="*78)
    print(split.upper(), "SUMMARY")
    print("="*78)

    for method in [
        "KS",
        "STARLET_NOMASK",
        "STARLET_MASK",
    ]:

        rr = [
            x for x in rows
            if x["method"] == method
        ]

        nmse = np.array([
            float(x["nmse"])
            for x in rr
        ])

        pcc = np.array([
            float(x["pcc"])
            for x in rr
        ])

        print(
            f"{method:15s} "
            f"N={len(rr):3d} "
            f"NMSE={nmse.mean():.6f} "
            f"PCC={pcc.mean():.6f}"
        )

    # Validation self-check against the KS file
    # already generated for GLIMPSE tuning.
    if split == "validation":

        ks_ref = (
            GLROOT /
            "validation" /
            "ks_validation_metrics.csv"
        )

        if ks_ref.exists():

            ref_nmse = []

            with open(ks_ref) as f:

                rd = csv.DictReader(f)

                for x in rd:
                    ref_nmse.append(
                        float(x["nmse"])
                    )

            our_ks = [
                float(x["nmse"])
                for x in rows
                if x["method"] == "KS"
            ]

            print()
            print("KS IMPLEMENTATION CHECK")
            print(
                "existing mean =",
                np.mean(ref_nmse)
            )
            print(
                "new mean      =",
                np.mean(our_ks)
            )
            print(
                "difference    =",
                np.mean(our_ks)
                - np.mean(ref_nmse)
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test"
        ],
        required=True
    )

    args = parser.parse_args()

    run(args.split)
