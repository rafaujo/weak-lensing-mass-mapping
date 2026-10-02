from project_paths import PATHS
from pathlib import Path
import csv
import numpy as np
import torch
import weak_lensing_pipeline as wl

# ============================================================
# Frozen benchmark
# ============================================================
ROOT = Path(str(PATHS.code_root))
DATA = PATHS.data_root

OUT = Path(str(PATHS.work_root / 'validation'))
NPZ_DIR = OUT / "fair_npz"

OUT.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

MASK_FILE   = DATA / "WIDE12H_bin2_2arcmin_mask.npy"
KAPPA_FILE  = DATA / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
LABEL_FILE  = DATA / "sampled_label.npy"

SIGMA_N = 0.4 / np.sqrt(2.0 * 30.0 * 4.0)

# Frozen split
COSMOS = range(3)
REALIZATIONS = range(10)     # VALIDATION ONLY: R0...R9

# Fixed square benchmark:
# 8 non-overlapping 176x176 patches.
PATCH = 176
PATCHES = [
    (p, p * PATCH, (p + 1) * PATCH)
    for p in range(8)
]

KS_SIGMA = 1.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device   =", DEVICE)
print("sigma_n  =", SIGMA_N)
print("KS sigma =", KS_SIGMA)

# ============================================================
# Data
# ============================================================
mask = np.load(MASK_FILE).astype(bool)
sampled = np.load(KAPPA_FILE, mmap_mode="r")
labels = np.load(LABEL_FILE)

assert mask.shape == (1424, 176)
assert sampled.shape == (3, 30, mask.sum())
assert labels.shape == (3, 30, 5)

# ============================================================
# Operators
# ============================================================
op_full = wl.WeakLensingOperator(
    1424, 176, device=DEVICE
)

op_patch = wl.WeakLensingOperator(
    176, 176, device=DEVICE
)

# ============================================================
# Periodic Gaussian smoothing
#
# Equivalent to Gaussian smoothing in Fourier space.
# sigma is in pixels.
# ============================================================
fy = torch.fft.fftfreq(PATCH, d=1.0, device=DEVICE)
fx = torch.fft.fftfreq(PATCH, d=1.0, device=DEVICE)

KY, KX = torch.meshgrid(fy, fx, indexing="ij")

gaussian_kernel = torch.exp(
    -2.0 * np.pi**2 * KS_SIGMA**2 * (KX*KX + KY*KY)
)

# ============================================================
# Metrics
# ============================================================
def metrics_ms(rec, truth, m):
    r = np.asarray(rec[m], dtype=np.float64).copy()
    t = np.asarray(truth[m], dtype=np.float64).copy()

    r -= r.mean()
    t -= t.mean()

    den = np.dot(t, t)

    nmse = np.dot(r - t, r - t) / den

    rr = np.dot(r, r)
    tt = np.dot(t, t)

    if rr > 0 and tt > 0:
        pcc = np.dot(r, t) / np.sqrt(rr * tt)
    else:
        pcc = np.nan

    std_ratio = r.std() / t.std()

    return float(nmse), float(pcc), float(std_ratio)

# ============================================================
# KS reconstruction on the SAME square input as GLIMPSE
#
# gamma_obs = M gamma_noisy
# kappa_KS = A* gamma_obs
# followed by frozen sigma=1.5 smoothing.
# ============================================================
@torch.no_grad()
def reconstruct_ks(gamma_patch_np, mask_patch_np):
    gamma = torch.from_numpy(
        gamma_patch_np
    ).to(DEVICE)

    m = torch.from_numpy(
        mask_patch_np.astype(np.float32)
    ).to(DEVICE)

    gamma_obs = gamma * m

    kappa = op_patch.adjoint(gamma_obs)

    if torch.is_complex(kappa):
        kappa = kappa.real

    kh = torch.fft.fft2(kappa)

    ks = torch.fft.ifft2(
        kh * gaussian_kernel
    ).real

    return ks.detach().cpu().numpy()

# ============================================================
# Main
# ============================================================
ks_rows = []

total = 3 * 10
icase = 0

for c in COSMOS:
    for r in REALIZATIONS:

        icase += 1

        case_file = NPZ_DIR / f"C{c}_R{r}.npz"

        # ----------------------------------------------------
        # Resume: if full FAIR observation already exists,
        # don't regenerate it.
        # ----------------------------------------------------
        if case_file.exists():

            d = np.load(case_file)

            truth_full = d["truth"]
            gamma_full = d["gamma"]

            omega_m = float(d["omega_m"])

            print(
                f"[{icase:02d}/{total}] "
                f"C{c} R{r}: loading cached observation"
            )

        else:

            print(
                f"[{icase:02d}/{total}] "
                f"C{c} R{r}: generating full FAIR observation"
            )

            truth_full = np.zeros(
                mask.shape,
                dtype=np.float32
            )

            truth_full[mask] = sampled[
                c, r
            ].astype(np.float32)

            kt = torch.from_numpy(
                truth_full
            ).to(
                DEVICE,
                dtype=torch.float32
            )

            with torch.no_grad():

                gamma_clean = op_full.forward(kt)

                # SAME frozen rule as FAIR:
                # seed = realization index.
                gamma_noisy = wl.add_shape_noise(
                    gamma_clean,
                    sigma_n=SIGMA_N,
                    seed=r
                )

            gamma_full = (
                gamma_noisy
                .detach()
                .cpu()
                .numpy()
                .astype(np.complex64)
            )

            omega_m = float(labels[c, r, 0])

            np.savez_compressed(
                case_file,
                truth=truth_full,
                gamma=gamma_full,
                omega_m=np.float64(omega_m),
                cosmology=np.int32(c),
                realization=np.int32(r),
                sigma_n=np.float64(SIGMA_N),
            )

        # ----------------------------------------------------
        # 8 fixed patches
        # ----------------------------------------------------
        for p, r0, r1 in PATCHES:

            truth_p = truth_full[r0:r1, :]
            gamma_p = gamma_full[r0:r1, :]
            mask_p  = mask[r0:r1, :]

            assert truth_p.shape == (176, 176)
            assert gamma_p.shape == (176, 176)

            if mask_p.sum() == 0:
                print(
                    f"WARNING: C{c} R{r} P{p} has no observed pixels"
                )
                continue

            ks = reconstruct_ks(
                gamma_p,
                mask_p
            )

            nmse, pcc, sr = metrics_ms(
                ks,
                truth_p,
                mask_p
            )

            ks_rows.append({
                "method": "KS",
                "c": c,
                "r": r,
                "p": p,
                "row0": r0,
                "row1": r1,
                "omega_m": omega_m,
                "f_obs": float(mask_p.mean()),
                "n_obs": int(mask_p.sum()),
                "sigma": KS_SIGMA,
                "nmse": nmse,
                "pcc": pcc,
                "std_ratio": sr,
            })

# ============================================================
# Save KS validation metrics
# ============================================================
ks_csv = OUT / "ks_validation_metrics.csv"

fields = [
    "method",
    "c", "r", "p",
    "row0", "row1",
    "omega_m",
    "f_obs", "n_obs",
    "sigma",
    "nmse", "pcc", "std_ratio"
]

with open(ks_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(ks_rows)

print()
print("=" * 72)
print("PREPARATION COMPLETE")
print("=" * 72)
print("FAIR cases :", len(list(NPZ_DIR.glob("C*_R*.npz"))))
print("KS rows    :", len(ks_rows))
print("KS CSV     :", ks_csv)

nmse = np.array([x["nmse"] for x in ks_rows])
pcc  = np.array([x["pcc"] for x in ks_rows])

print()
print("KS square-patch validation:")
print("  NMSE mean   =", nmse.mean())
print("  NMSE median =", np.median(nmse))
print("  PCC mean    =", pcc.mean())
