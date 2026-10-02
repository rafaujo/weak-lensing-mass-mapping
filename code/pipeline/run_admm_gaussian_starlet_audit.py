from project_paths import PATHS
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(PATHS.code_root))

from weak_lensing_pipeline import (
    WeakLensingOperator,
    KaiserSquires,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N = 176
J = 4

LAMBDA = 0.05
BETA = 0.001

ALPHA = np.array(
    [
        1.0,
        0.225148,
        0.096029,
        0.046286,
    ],
    dtype=np.float32,
)

BASE = Path(
    str(PATHS.work_root / 'validation/fair_npz')
)

MASK_FILE = Path(
    str(PATHS.data_root / 'WIDE12H_bin2_2arcmin_mask.npy')
)

PK_FILE = Path(
    str(PATHS.prior_file)
)

PDHG_ROOT = Path(
    str(PATHS.work_root / 'validation/reweighted_gaussian_starlet')
)

PDHG_REC_FILE = (
    PDHG_ROOT
    / "stage1_gaussian_starlet_rec.npy"
)

PDHG_G_FILE = (
    PDHG_ROOT
    / "stage1_gaussian_component.npy"
)

PDHG_S_FILE = (
    PDHG_ROOT
    / "stage1_sparse_component.npy"
)

OUT = Path(
    str(PATHS.work_root / 'validation/admm_solver_audit')
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CSV = (
    OUT
    / "admm_vs_pdhg_cases.csv"
)

SUMMARY_CSV = (
    OUT
    / "admm_vs_pdhg_summary.csv"
)

RECON_DIR = (
    OUT / "recon"
)

RECON_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# AUDIT CASES
#
# 3 cosmologies x 10 validation realizations = 30 cases.
# Patch index cycles deterministically so all mask geometries appear.
# =============================================================================
CASES = []

for c in range(3):
    for r in range(10):
        p = (
            r
            + 2 * c
        ) % 8

        CASES.append(
            (
                c,
                r,
                p,
            )
        )

assert len(CASES) == 30


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
# ADMM PARAMETERS
# =============================================================================
#
# Common rho for the product-space split.
# It may be adapted by residual balancing.
#
RHO_INIT = 100.0

ADMM_MAXITER = 2500

# Boyd et al. absolute + relative stopping tolerances.
#
# Constraint:
#
#     Bx - z = 0
#
# with scaled dual variable u.
#
# eps_pri  = sqrt(m) eps_abs
#            + eps_rel max(||Bx||, ||z||)
#
# eps_dual = sqrt(n) eps_abs
#            + eps_rel ||rho B^T u||
#
ADMM_EPS_ABS = 1.0e-6
ADMM_EPS_REL = 1.0e-4

ADAPT_RHO = True
ADAPT_EVERY = 25

RESIDUAL_MU = 10.0
RHO_FACTOR = 2.0

RHO_MIN = 1.0e-2
RHO_MAX = 1.0e5

# Inner block-CG.
CG_MAXITER = 40
CG_TOL = 1.0e-7

# Run several audit maps together.
BATCH = 10

PRINT_EVERY = 50


# =============================================================================
# BOYD STOPPING-CRITERION DIMENSIONS
# =============================================================================
#
# Primal variable:
#
#     x = (x_G, x_S)
#
# Both are real H x W maps:
#
#     n = 2 H W.
#
# Product-space split:
#
#     z1 = M A(x_G+x_S)    complex H x W -> 2 H W real dof
#     z2 = x_G             real H x W
#     z3 = K x_S           J real H x W bands
#     z4 = x_S             real H x W
#
# Therefore:
#
#     m = (2 + 1 + J + 1) H W
#       = (J + 4) H W.
#
PRIMAL_DIM_REAL = (
    2
    * N
    * N
)

SPLIT_DIM_REAL = (
    (J + 4)
    * N
    * N
)

SQRT_PRIMAL_DIM = float(
    np.sqrt(
        PRIMAL_DIM_REAL
    )
)

SQRT_SPLIT_DIM = float(
    np.sqrt(
        SPLIT_DIM_REAL
    )
)


# =============================================================================
# BATCH ADAPTERS FOR CANONICAL 2-D OPERATOR
# =============================================================================
def op_forward_batch(
    op,
    x,
):
    if x.ndim == 2:
        return op.forward(x)

    return torch.vmap(
        op.forward
    )(x)


def op_adjoint_batch(
    op,
    y,
):
    if y.ndim == 2:
        return op.adjoint(y)

    return torch.vmap(
        op.adjoint
    )(y)


def ks_reconstruct_batch(
    ks,
    y,
    smooth_sigma,
):
    if y.ndim == 2:
        return ks.reconstruct(
            y,
            smooth_sigma=smooth_sigma,
        )

    return torch.vmap(
        lambda yy: ks.reconstruct(
            yy,
            smooth_sigma=smooth_sigma,
        )
    )(y)


# =============================================================================
# FOURIER STARLET
# =============================================================================
class FourierStarlet:
    """
    Same B3-spline Fourier Starlet used by the frozen
    Gaussian+Starlet PDHG experiment.

    K contains J detail bands only.
    """

    def __init__(
        self,
        H,
        W,
        J,
        device,
    ):
        wy = (
            2.0
            * np.pi
            * torch.fft.fftfreq(
                H,
                device=device,
            )
        )

        wx = (
            2.0
            * np.pi
            * torch.fft.fftfreq(
                W,
                device=device,
            )
        )

        WY, WX = torch.meshgrid(
            wy,
            wx,
            indexing="ij",
        )

        cumulative = torch.ones(
            (H, W),
            dtype=torch.float32,
            device=device,
        )

        bands = []

        for j in range(J):

            step = 2**j

            hy = (
                6.0
                + 8.0
                * torch.cos(
                    step * WY
                )
                + 2.0
                * torch.cos(
                    2.0 * step * WY
                )
            ) / 16.0

            hx = (
                6.0
                + 8.0
                * torch.cos(
                    step * WX
                )
                + 2.0
                * torch.cos(
                    2.0 * step * WX
                )
            ) / 16.0

            h2 = (
                hy * hx
            )

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

        self.B = torch.stack(
            bands,
            dim=0,
        ).unsqueeze(0)

        self.norm2 = float(
            torch.max(
                torch.sum(
                    self.B[0] ** 2,
                    dim=0,
                )
            ).item()
        )

    def analysis(
        self,
        x,
    ):
        X = torch.fft.fft2(
            x,
            norm="ortho",
        )

        return torch.fft.ifft2(
            self.B
            * X.unsqueeze(1),
            norm="ortho",
        ).real

    def adjoint(
        self,
        coeff,
    ):
        C = torch.fft.fft2(
            coeff,
            norm="ortho",
        )

        X = torch.sum(
            self.B
            * C,
            dim=1,
        )

        return torch.fft.ifft2(
            X,
            norm="ortho",
        ).real


# =============================================================================
# P_KAPPA MAP
# =============================================================================
def make_pk_map(
    pk,
    H,
    W,
):
    pk = np.asarray(
        pk,
        dtype=np.float64,
    ).copy()

    pk[0] = 0.0

    ext = np.concatenate(
        [
            pk,
            np.zeros(
                1,
                dtype=np.float64,
            ),
        ]
    )

    npk = len(pk)

    ky = (
        np.arange(H)
        - H / 2.0
    )

    kx = (
        np.arange(W)
        - W / 2.0
    )

    KY, KX = np.meshgrid(
        ky,
        kx,
        indexing="ij",
    )

    ir = np.sqrt(
        KX**2
        + KY**2
    ).astype(
        np.int64
    )

    ir[
        ir > npk
    ] = npk

    centered = ext[
        ir
    ]

    return np.fft.ifftshift(
        centered
    )


# =============================================================================
# PROX OPERATORS
# =============================================================================
def prox_gaussian(
    v,
    pk,
    beta_over_rho,
):
    """
    Prox of the frozen Gaussian prior.

    For P_kappa > 0:
        prox = P / (P + beta/rho)

    For P_kappa == 0:
        the frozen PDHG implementation sets the Fourier
        coefficient to zero. We reproduce that convention
        exactly here.
    """

    V = torch.fft.fft2(
        v,
        norm="ortho",
    )

    factor = torch.where(
        pk > 0,
        pk
        / (
            pk
            + beta_over_rho
        ),
        torch.zeros_like(
            pk
        ),
    )

    return torch.fft.ifft2(
        factor
        * V,
        norm="ortho",
    ).real


def soft_threshold(
    x,
    threshold,
):
    return torch.sign(x) * torch.clamp(
        torch.abs(x)
        - threshold,
        min=0.0,
    )


# =============================================================================
# PRODUCT-SPACE OPERATOR B
#
# B(x_G,x_S) =
#
#   1) M A(x_G+x_S)        data term
#   2) x_G                 Gaussian prox
#   3) K x_S               Starlet l1 prox
#   4) x_S                 redundant zero-cost consensus block
#
# Block 4 makes B^T B strongly positive definite and is an
# exact redundant splitting: its associated function is zero.
# =============================================================================
def apply_B(
    op,
    star,
    mask,
    xg,
    xs,
):
    b1 = (
        mask
        * op_forward_batch(
            op,
            xg + xs,
        )
    )

    b2 = xg

    b3 = star.analysis(
        xs
    )

    b4 = xs

    return (
        b1,
        b2,
        b3,
        b4,
    )


def apply_Bt(
    op,
    star,
    mask,
    d1,
    d2,
    d3,
    d4,
):
    common = op_adjoint_batch(
        op,
        mask * d1,
    )

    if torch.is_complex(
        common
    ):
        common = (
            common.real
        )

    out_g = (
        common
        + d2
    )

    out_s = (
        common
        + star.adjoint(
            d3
        )
        + d4
    )

    return (
        out_g,
        out_s,
    )


# =============================================================================
# BLOCK LINEAR SYSTEM H = B^T B
# =============================================================================
def apply_H(
    op,
    star,
    mask,
    vg,
    vs,
):
    common = op_adjoint_batch(
        op,
        mask
        * op_forward_batch(
            op,
            vg + vs,
        ),
    )

    if torch.is_complex(
        common
    ):
        common = common.real

    hg = (
        common
        + vg
    )

    hs = (
        common
        + star.adjoint(
            star.analysis(
                vs
            )
        )
        + vs
    )

    return (
        hg,
        hs,
    )


def block_dot(
    ag,
    ass,
    bg,
    bs,
):
    return (
        torch.sum(
            ag * bg,
            dim=(-2, -1),
        )
        +
        torch.sum(
            ass * bs,
            dim=(-2, -1),
        )
    )


def block_norm(
    ag,
    ass,
):
    return torch.sqrt(
        torch.clamp(
            block_dot(
                ag,
                ass,
                ag,
                ass,
            ),
            min=0.0,
        )
    )


# =============================================================================
# BATCHED BLOCK CG
# =============================================================================
@torch.no_grad()
def block_cg(
    op,
    star,
    mask,
    rhs_g,
    rhs_s,
    xg0,
    xs0,
):
    xg = xg0.clone()
    xs = xs0.clone()

    hg, hs = apply_H(
        op,
        star,
        mask,
        xg,
        xs,
    )

    rg = (
        rhs_g
        - hg
    )

    rs = (
        rhs_s
        - hs
    )

    pg = rg.clone()
    ps = rs.clone()

    rr = block_dot(
        rg,
        rs,
        rg,
        rs,
    )

    bnorm = torch.clamp(
        block_norm(
            rhs_g,
            rhs_s,
        ),
        min=1.0e-20,
    )

    last_rel = (
        torch.sqrt(
            torch.clamp(
                rr,
                min=0.0,
            )
        )
        / bnorm
    )

    nit = 0

    for it in range(
        1,
        CG_MAXITER + 1,
    ):
        hpg, hps = apply_H(
            op,
            star,
            mask,
            pg,
            ps,
        )

        denom = block_dot(
            pg,
            ps,
            hpg,
            hps,
        )

        alpha = (
            rr
            / torch.clamp(
                denom,
                min=1.0e-30,
            )
        )

        xg = (
            xg
            + alpha[
                :,
                None,
                None,
            ]
            * pg
        )

        xs = (
            xs
            + alpha[
                :,
                None,
                None,
            ]
            * ps
        )

        rg = (
            rg
            - alpha[
                :,
                None,
                None,
            ]
            * hpg
        )

        rs = (
            rs
            - alpha[
                :,
                None,
                None,
            ]
            * hps
        )

        rr_new = block_dot(
            rg,
            rs,
            rg,
            rs,
        )

        last_rel = (
            torch.sqrt(
                torch.clamp(
                    rr_new,
                    min=0.0,
                )
            )
            / bnorm
        )

        nit = it

        if float(
            torch.max(
                last_rel
            ).item()
        ) < CG_TOL:
            rr = rr_new
            break

        beta_cg = (
            rr_new
            / torch.clamp(
                rr,
                min=1.0e-30,
            )
        )

        pg = (
            rg
            + beta_cg[
                :,
                None,
                None,
            ]
            * pg
        )

        ps = (
            rs
            + beta_cg[
                :,
                None,
                None,
            ]
            * ps
        )

        rr = rr_new

    return (
        xg,
        xs,
        nit,
        last_rel,
    )


# =============================================================================
# BLOCK NORMS IN PRODUCT SPACE
# =============================================================================
def product_norm(
    a1,
    a2,
    a3,
    a4,
):
    n2 = (
        torch.sum(
            torch.abs(a1) ** 2,
            dim=(-2, -1),
        )
        +
        torch.sum(
            a2**2,
            dim=(-2, -1),
        )
        +
        torch.sum(
            a3**2,
            dim=(-3, -2, -1),
        )
        +
        torch.sum(
            a4**2,
            dim=(-2, -1),
        )
    )

    return torch.sqrt(
        torch.clamp(
            n2,
            min=0.0,
        )
    )


# =============================================================================
# ADMM
# =============================================================================
@torch.no_grad()
def solve_admm(
    op,
    ks,
    star,
    y,
    mask,
    pk,
    alpha,
):
    """
    Exact product-space ADMM for the same frozen
    Gaussian+Starlet objective solved by PDHG.

    Functions on the split variables:

      g1(z1) = 1/2 ||z1-y||^2

      g2(z2) = beta/2 Sum |F z2|^2/P
               + Fourier-support constraint

      g3(z3) = lambda Sum_j alpha_j |z3_j|

      g4(z4) = 0

    subject to z = B(x_G,x_S).
    """

    # ---------------------------------------------------------
    # Independent initialization:
    # mildly smoothed KS, NOT the PDHG solution.
    # ---------------------------------------------------------
    xg = ks_reconstruct_batch(
        ks,
        y,
        smooth_sigma=1.0,
    )

    if torch.is_complex(
        xg
    ):
        xg = xg.real

    xg = xg.float()

    xs = torch.zeros_like(
        xg
    )

    (
        z1,
        z2,
        z3,
        z4,
    ) = apply_B(
        op,
        star,
        mask,
        xg,
        xs,
    )

    u1 = torch.zeros_like(
        z1
    )

    u2 = torch.zeros_like(
        z2
    )

    u3 = torch.zeros_like(
        z3
    )

    u4 = torch.zeros_like(
        z4
    )

    rho = float(
        RHO_INIT
    )

    converged = False

    primal_rel_max = np.inf
    dual_rel_max = np.inf

    primal_ratio_max = np.inf
    dual_ratio_max = np.inf

    primal_norm_max = np.inf
    dual_norm_max = np.inf

    eps_pri_max = np.inf
    eps_dual_max = np.inf

    cg_nit_last = 0
    cg_rel_last = np.inf

    t0 = (
        time.perf_counter()
    )

    for it in range(
        1,
        ADMM_MAXITER + 1,
    ):
        # =====================================================
        # X UPDATE
        #
        # min_x ||Bx - z + u||^2
        #
        # B^T B x = B^T(z-u)
        # =====================================================
        rhs_g, rhs_s = apply_Bt(
            op,
            star,
            mask,
            z1 - u1,
            z2 - u2,
            z3 - u3,
            z4 - u4,
        )

        (
            xg,
            xs,
            cg_nit,
            cg_rel,
        ) = block_cg(
            op,
            star,
            mask,
            rhs_g,
            rhs_s,
            xg,
            xs,
        )

        cg_nit_last = (
            cg_nit
        )

        cg_rel_last = float(
            torch.max(
                cg_rel
            ).item()
        )

        # =====================================================
        # Bx
        # =====================================================
        (
            b1,
            b2,
            b3,
            b4,
        ) = apply_B(
            op,
            star,
            mask,
            xg,
            xs,
        )

        # Save old z for dual residual.
        z1_old = z1
        z2_old = z2
        z3_old = z3
        z4_old = z4

        # =====================================================
        # Z1: DATA PROX
        #
        # prox_{(1/rho) 0.5||.-y||^2}
        # =====================================================
        v1 = (
            b1
            + u1
        )

        z1 = (
            rho * v1
            + y
        ) / (
            rho + 1.0
        )

        # =====================================================
        # Z2: GAUSSIAN PRIOR PROX
        # =====================================================
        v2 = (
            b2
            + u2
        )

        z2 = prox_gaussian(
            v2,
            pk,
            BETA / rho,
        )

        # =====================================================
        # Z3: WEIGHTED STARLET L1 PROX
        # =====================================================
        v3 = (
            b3
            + u3
        )

        threshold = (
            LAMBDA
            * alpha
            / rho
        )

        z3 = soft_threshold(
            v3,
            threshold,
        )

        # =====================================================
        # Z4: ZERO-FUNCTION PROX
        # =====================================================
        v4 = (
            b4
            + u4
        )

        z4 = v4

        # =====================================================
        # PRIMAL RESIDUAL
        # =====================================================
        r1 = (
            b1 - z1
        )

        r2 = (
            b2 - z2
        )

        r3 = (
            b3 - z3
        )

        r4 = (
            b4 - z4
        )

        rnorm = product_norm(
            r1,
            r2,
            r3,
            r4,
        )

        bnorm = product_norm(
            b1,
            b2,
            b3,
            b4,
        )

        znorm = product_norm(
            z1,
            z2,
            z3,
            z4,
        )

        # =====================================================
        # BOYD PRIMAL TOLERANCE
        #
        # eps_pri =
        #   sqrt(m) eps_abs
        #   + eps_rel max(||Bx||, ||z||)
        # =====================================================
        primal_scale = torch.maximum(
            bnorm,
            znorm,
        )

        eps_pri = (
            SQRT_SPLIT_DIM
            * ADMM_EPS_ABS
            +
            ADMM_EPS_REL
            * primal_scale
        )

        # Retain a conventional relative residual for reporting.
        primal_rel = (
            rnorm
            / torch.clamp(
                primal_scale,
                min=1.0e-12,
            )
        )

        # Ratio <= 1 means the Boyd primal criterion is met.
        primal_ratio = (
            rnorm
            / torch.clamp(
                eps_pri,
                min=1.0e-30,
            )
        )

        # =====================================================
        # DUAL RESIDUAL
        #
        # s = rho B^T(z^k-z^{k-1})
        # =====================================================
        sg, ss = apply_Bt(
            op,
            star,
            mask,
            z1 - z1_old,
            z2 - z2_old,
            z3 - z3_old,
            z4 - z4_old,
        )

        sg = (
            rho * sg
        )

        ss = (
            rho * ss
        )

        snorm = block_norm(
            sg,
            ss,
        )

        # Scale by current dual magnitude.
        dug, dus = apply_Bt(
            op,
            star,
            mask,
            u1,
            u2,
            u3,
            u4,
        )

        dual_scale = (
            rho
            * block_norm(
                dug,
                dus,
            )
        )

        # =====================================================
        # BOYD DUAL TOLERANCE
        #
        # With scaled dual u, y_dual = rho u:
        #
        # eps_dual =
        #   sqrt(n) eps_abs
        #   + eps_rel ||rho B^T u||
        # =====================================================
        eps_dual = (
            SQRT_PRIMAL_DIM
            * ADMM_EPS_ABS
            +
            ADMM_EPS_REL
            * dual_scale
        )

        # Conventional relative quantity retained only for
        # diagnostics. It is NOT used for stopping.
        dual_rel = (
            snorm
            / torch.clamp(
                dual_scale,
                min=1.0e-12,
            )
        )

        # Ratio <= 1 means the Boyd dual criterion is met.
        dual_ratio = (
            snorm
            / torch.clamp(
                eps_dual,
                min=1.0e-30,
            )
        )

        # =====================================================
        # DUAL UPDATE
        # =====================================================
        u1 = (
            u1 + r1
        )

        u2 = (
            u2 + r2
        )

        u3 = (
            u3 + r3
        )

        u4 = (
            u4 + r4
        )

        primal_rel_max = float(
            torch.max(
                primal_rel
            ).item()
        )

        dual_rel_max = float(
            torch.max(
                dual_rel
            ).item()
        )

        primal_ratio_max = float(
            torch.max(
                primal_ratio
            ).item()
        )

        dual_ratio_max = float(
            torch.max(
                dual_ratio
            ).item()
        )

        primal_norm_max = float(
            torch.max(
                rnorm
            ).item()
        )

        dual_norm_max = float(
            torch.max(
                snorm
            ).item()
        )

        eps_pri_max = float(
            torch.max(
                eps_pri
            ).item()
        )

        eps_dual_max = float(
            torch.max(
                eps_dual
            ).item()
        )

        # =====================================================
        # REPORT
        # =====================================================
        if (
            it == 1
            or it % PRINT_EVERY == 0
        ):
            print(
                f"  ADMM {it:4d}  "
                f"rho={rho:10.3e}  "
                f"r/eps={primal_ratio_max:9.3e}  "
                f"s/eps={dual_ratio_max:9.3e}  "
                f"rrel={primal_rel_max:9.3e}  "
                f"srel={dual_rel_max:9.3e}  "
                f"CG={cg_nit:2d}  "
                f"CGrel={cg_rel_last:9.3e}",
                flush=True,
            )

        # =====================================================
        # STOPPING TEST
        # =====================================================
        # =====================================================
        # STANDARD BOYD STOPPING TEST
        #
        # Require every case in the current batch to satisfy
        # both absolute+relative tolerances.
        # =====================================================
        if (
            primal_ratio_max <= 1.0
            and
            dual_ratio_max <= 1.0
        ):
            converged = True
            break

        # =====================================================
        # RHO ADAPTATION
        # =====================================================
        if (
            ADAPT_RHO
            and
            it % ADAPT_EVERY == 0
        ):
            r_scalar = float(
                torch.mean(
                    rnorm
                ).item()
            )

            s_scalar = float(
                torch.mean(
                    snorm
                ).item()
            )

            old_rho = (
                rho
            )

            if (
                r_scalar
                >
                RESIDUAL_MU
                * s_scalar
            ):
                rho = min(
                    rho
                    * RHO_FACTOR,
                    RHO_MAX,
                )

            elif (
                s_scalar
                >
                RESIDUAL_MU
                * r_scalar
            ):
                rho = max(
                    rho
                    / RHO_FACTOR,
                    RHO_MIN,
                )

            if rho != old_rho:
                # Scaled dual u = y_dual/rho.
                scale = (
                    old_rho
                    / rho
                )

                u1 = (
                    u1 * scale
                )

                u2 = (
                    u2 * scale
                )

                u3 = (
                    u3 * scale
                )

                u4 = (
                    u4 * scale
                )

    elapsed = (
        time.perf_counter()
        - t0
    )

    rec = (
        xg + xs
    )

    return {
        "rec": rec,
        "xg": xg,
        "xs": xs,

        "iterations": it,

        "converged": converged,

        "rho_final": rho,

        "primal_rel_max": (
            primal_rel_max
        ),

        "dual_rel_max": (
            dual_rel_max
        ),

        "primal_ratio_max": (
            primal_ratio_max
        ),

        "dual_ratio_max": (
            dual_ratio_max
        ),

        "primal_norm_max": (
            primal_norm_max
        ),

        "dual_norm_max": (
            dual_norm_max
        ),

        "eps_pri_max": (
            eps_pri_max
        ),

        "eps_dual_max": (
            eps_dual_max
        ),

        "cg_nit_last": (
            cg_nit_last
        ),

        "cg_rel_last": (
            cg_rel_last
        ),

        "elapsed": elapsed,
    }


# =============================================================================
# OBJECTIVE
# =============================================================================
@torch.no_grad()
def objective_terms(
    op,
    star,
    y,
    mask,
    pk,
    alpha,
    xg,
    xs,
):
    rec = (
        xg + xs
    )

    pred = (
        mask
        * op_forward_batch(
            op,
            rec,
        )
    )

    res = (
        pred - y
    )

    data = (
        0.5
        * torch.sum(
            torch.abs(
                res
            ) ** 2,
            dim=(-2, -1),
        )
    )

    X = torch.fft.fft2(
        xg,
        norm="ortho",
    )

    pos = (
        pk > 0
    )

    denom = torch.where(
        pos,
        pk,
        torch.ones_like(
            pk
        ),
    )

    g_density = torch.where(
        pos,
        torch.abs(X) ** 2
        / denom,
        torch.zeros_like(
            denom
        ),
    )

    gauss = (
        0.5
        * BETA
        * torch.sum(
            g_density,
            dim=(-2, -1),
        )
    )

    coeff = star.analysis(
        xs
    )

    sparse = (
        LAMBDA
        * torch.sum(
            alpha
            * torch.abs(
                coeff
            ),
            dim=(-3, -2, -1),
        )
    )

    total = (
        data
        + gauss
        + sparse
    )

    # Fourier-support violation of the Gaussian component.
    neg = (
        ~pos
    )

    zero_power_energy = torch.sum(
        torch.where(
            neg,
            torch.abs(X) ** 2,
            torch.zeros_like(
                torch.abs(X) ** 2
            ),
        ),
        dim=(-2, -1),
    )

    total_fourier_energy = torch.sum(
        torch.abs(X) ** 2,
        dim=(-2, -1),
    )

    support_leak = torch.sqrt(
        zero_power_energy
        / torch.clamp(
            total_fourier_energy,
            min=1.0e-30,
        )
    )

    return {
        "data": data,
        "gauss": gauss,
        "sparse": sparse,
        "total": total,
        "support_leak": (
            support_leak
        ),
    }


# =============================================================================
# MAP METRICS
# =============================================================================
def ms_metrics(
    rec,
    truth,
    mask,
):
    m = mask.astype(
        bool
    )

    r = np.asarray(
        rec,
        dtype=np.float64,
    )[m]

    t = np.asarray(
        truth,
        dtype=np.float64,
    )[m]

    r -= r.mean()
    t -= t.mean()

    mse = np.mean(
        (
            r - t
        ) ** 2
    )

    nmse = (
        mse
        / np.mean(
            t**2
        )
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


def relative_map_difference(
    a,
    b,
):
    return float(
        np.linalg.norm(
            a - b
        )
        / max(
            np.linalg.norm(
                b
            ),
            1.0e-30,
        )
    )


def relative_centered_observed_difference(
    a,
    b,
    mask,
):
    m = mask.astype(
        bool
    )

    aa = np.asarray(
        a,
        dtype=np.float64,
    )[m]

    bb = np.asarray(
        b,
        dtype=np.float64,
    )[m]

    aa = (
        aa - aa.mean()
    )

    bb = (
        bb - bb.mean()
    )

    return float(
        np.linalg.norm(
            aa - bb
        )
        / max(
            np.linalg.norm(
                bb
            ),
            1.0e-30,
        )
    )


# =============================================================================
# LOAD INPUT DATA
# =============================================================================
mask_full = np.load(
    MASK_FILE
).astype(
    bool
)

pk1d = np.load(
    PK_FILE
).astype(
    np.float64
)

pk2d = make_pk_map(
    pk1d,
    N,
    N,
)

pk = torch.as_tensor(
    pk2d,
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(0)

alpha_t = torch.as_tensor(
    ALPHA,
    dtype=torch.float32,
    device=DEVICE,
).view(
    1,
    J,
    1,
    1,
)


# =============================================================================
# LOAD FROZEN PDHG MAPS / COMPONENTS
# =============================================================================
for fn in [
    PDHG_REC_FILE,
    PDHG_G_FILE,
    PDHG_S_FILE,
]:
    if not fn.exists():
        raise FileNotFoundError(
            fn
        )

pdhg_rec_grid = np.load(
    PDHG_REC_FILE,
    mmap_mode="r",
)

pdhg_g_grid = np.load(
    PDHG_G_FILE,
    mmap_mode="r",
)

pdhg_s_grid = np.load(
    PDHG_S_FILE,
    mmap_mode="r",
)


# =============================================================================
# OPERATOR
# =============================================================================
op = WeakLensingOperator(
    N,
    N,
    DEVICE,
)

ks = KaiserSquires(
    op
)

star = FourierStarlet(
    N,
    N,
    J,
    DEVICE,
)

print()
print("=" * 120)
print(
    "GAUSSIAN + STARLET SOLVER AUDIT"
)
print("=" * 120)

print(
    "device       :",
    DEVICE,
)

print(
    "cases        :",
    len(CASES),
)

print(
    "lambda       :",
    LAMBDA,
)

print(
    "beta         :",
    BETA,
)

print(
    "rho init     :",
    RHO_INIT,
)

print(
    "ADMM maxiter :",
    ADMM_MAXITER,
)

print(
    "Boyd eps_abs :",
    ADMM_EPS_ABS,
)

print(
    "Boyd eps_rel :",
    ADMM_EPS_REL,
)

print(
    "primal dim   :",
    PRIMAL_DIM_REAL,
)

print(
    "split dim    :",
    SPLIT_DIM_REAL,
)

print(
    "CG tol/max   :",
    CG_TOL,
    CG_MAXITER,
)

print(
    "Starlet ||K||²:",
    star.norm2,
)


# =============================================================================
# RUN AUDIT
# =============================================================================
rows = []

for ib in range(
    0,
    len(CASES),
    BATCH,
):
    batch_cases = (
        CASES[
            ib:
            ib + BATCH
        ]
    )

    truths = []
    gammas = []
    masks = []

    pdhg_rec_list = []
    pdhg_g_list = []
    pdhg_s_list = []

    for (
        c,
        r,
        p,
    ) in batch_cases:

        fn = (
            BASE
            / f"C{c}_R{r}.npz"
        )

        with np.load(fn) as z:

            truth_full = np.asarray(
                z["truth"],
                dtype=np.float32,
            )

            gamma_full = np.asarray(
                z["gamma"],
                dtype=np.complex64,
            )

        r0, r1 = (
            PATCH_ROWS[
                p
            ]
        )

        mask = mask_full[
            r0:r1,
            :
        ].copy()

        truth = truth_full[
            r0:r1,
            :
        ].copy()

        gamma = (
            gamma_full[
                r0:r1,
                :
            ].copy()
            * mask
        )

        truths.append(
            truth
        )

        gammas.append(
            gamma
        )

        masks.append(
            mask
        )

        pdhg_rec_list.append(
            np.asarray(
                pdhg_rec_grid[
                    c,
                    r,
                    p,
                ],
                dtype=np.float32,
            )
        )

        pdhg_g_list.append(
            np.asarray(
                pdhg_g_grid[
                    c,
                    r,
                    p,
                ],
                dtype=np.float32,
            )
        )

        pdhg_s_list.append(
            np.asarray(
                pdhg_s_grid[
                    c,
                    r,
                    p,
                ],
                dtype=np.float32,
            )
        )

    truths = np.asarray(
        truths,
        dtype=np.float32,
    )

    gammas = np.asarray(
        gammas,
        dtype=np.complex64,
    )

    masks = np.asarray(
        masks,
        dtype=np.float32,
    )

    pdhg_rec_np = np.asarray(
        pdhg_rec_list,
        dtype=np.float32,
    )

    pdhg_g_np = np.asarray(
        pdhg_g_list,
        dtype=np.float32,
    )

    pdhg_s_np = np.asarray(
        pdhg_s_list,
        dtype=np.float32,
    )

    y = torch.as_tensor(
        gammas,
        dtype=torch.complex64,
        device=DEVICE,
    )

    mask_t = torch.as_tensor(
        masks,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("=" * 120)
    print(
        f"BATCH {ib+1}-"
        f"{ib+len(batch_cases)}"
    )
    print("=" * 120)

    result = solve_admm(
        op=op,
        ks=ks,
        star=star,
        y=y,
        mask=mask_t,
        pk=pk,
        alpha=alpha_t,
    )

    admm_rec = (
        result[
            "rec"
        ]
        .cpu()
        .numpy()
    )

    admm_g = (
        result[
            "xg"
        ]
        .cpu()
        .numpy()
    )

    admm_s = (
        result[
            "xs"
        ]
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # OBJECTIVES: ADMM
    # ---------------------------------------------------------
    obj_admm = objective_terms(
        op,
        star,
        y,
        mask_t,
        pk,
        alpha_t,
        result["xg"],
        result["xs"],
    )

    # ---------------------------------------------------------
    # OBJECTIVES: FROZEN PDHG
    # ---------------------------------------------------------
    pdhg_g_t = torch.as_tensor(
        pdhg_g_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    pdhg_s_t = torch.as_tensor(
        pdhg_s_np,
        dtype=torch.float32,
        device=DEVICE,
    )

    obj_pdhg = objective_terms(
        op,
        star,
        y,
        mask_t,
        pk,
        alpha_t,
        pdhg_g_t,
        pdhg_s_t,
    )

    for j, (
        c,
        r,
        p,
    ) in enumerate(
        batch_cases
    ):
        admm_m = ms_metrics(
            admm_rec[j],
            truths[j],
            masks[j],
        )

        pdhg_m = ms_metrics(
            pdhg_rec_np[j],
            truths[j],
            masks[j],
        )

        obj_a = float(
            obj_admm[
                "total"
            ][j].item()
        )

        obj_p = float(
            obj_pdhg[
                "total"
            ][j].item()
        )

        obj_rel = (
            abs(
                obj_a - obj_p
            )
            / max(
                abs(obj_p),
                1.0e-30,
            )
        )

        map_rel = (
            relative_map_difference(
                admm_rec[j],
                pdhg_rec_np[j],
            )
        )

        map_rel_obs = (
            relative_centered_observed_difference(
                admm_rec[j],
                pdhg_rec_np[j],
                masks[j],
            )
        )

        g_rel = (
            relative_map_difference(
                admm_g[j],
                pdhg_g_np[j],
            )
        )

        s_rel = (
            relative_map_difference(
                admm_s[j],
                pdhg_s_np[j],
            )
        )

        out_fn = (
            RECON_DIR
            / f"C{c}_R{r}_P{p}.npz"
        )

        np.savez_compressed(
            out_fn,

            admm=np.asarray(
                admm_rec[j],
                dtype=np.float32,
            ),

            admm_gaussian=np.asarray(
                admm_g[j],
                dtype=np.float32,
            ),

            admm_sparse=np.asarray(
                admm_s[j],
                dtype=np.float32,
            ),

            pdhg=np.asarray(
                pdhg_rec_np[j],
                dtype=np.float32,
            ),

            pdhg_gaussian=np.asarray(
                pdhg_g_np[j],
                dtype=np.float32,
            ),

            pdhg_sparse=np.asarray(
                pdhg_s_np[j],
                dtype=np.float32,
            ),

            truth=np.asarray(
                truths[j],
                dtype=np.float32,
            ),

            mask=np.asarray(
                masks[j],
                dtype=np.uint8,
            ),
        )

        row = {
            "c": c,
            "r": r,
            "p": p,

            "admm_converged": (
                result[
                    "converged"
                ]
            ),

            "admm_iterations": (
                result[
                    "iterations"
                ]
            ),

            "rho_final": (
                result[
                    "rho_final"
                ]
            ),

            "primal_rel": (
                result[
                    "primal_rel_max"
                ]
            ),

            "dual_rel": (
                result[
                    "dual_rel_max"
                ]
            ),

            "primal_ratio_to_eps": (
                result[
                    "primal_ratio_max"
                ]
            ),

            "dual_ratio_to_eps": (
                result[
                    "dual_ratio_max"
                ]
            ),

            "primal_norm": (
                result[
                    "primal_norm_max"
                ]
            ),

            "dual_norm": (
                result[
                    "dual_norm_max"
                ]
            ),

            "eps_primal": (
                result[
                    "eps_pri_max"
                ]
            ),

            "eps_dual": (
                result[
                    "eps_dual_max"
                ]
            ),

            "cg_last_iterations": (
                result[
                    "cg_nit_last"
                ]
            ),

            "cg_last_rel": (
                result[
                    "cg_rel_last"
                ]
            ),

            "batch_elapsed": (
                result[
                    "elapsed"
                ]
            ),

            # Objective components.
            "obj_admm": obj_a,

            "obj_pdhg": obj_p,

            "obj_rel_diff": (
                obj_rel
            ),

            "data_admm": float(
                obj_admm[
                    "data"
                ][j].item()
            ),

            "data_pdhg": float(
                obj_pdhg[
                    "data"
                ][j].item()
            ),

            "gauss_admm": float(
                obj_admm[
                    "gauss"
                ][j].item()
            ),

            "gauss_pdhg": float(
                obj_pdhg[
                    "gauss"
                ][j].item()
            ),

            "sparse_admm": float(
                obj_admm[
                    "sparse"
                ][j].item()
            ),

            "sparse_pdhg": float(
                obj_pdhg[
                    "sparse"
                ][j].item()
            ),

            "gaussian_support_leak_admm": float(
                obj_admm[
                    "support_leak"
                ][j].item()
            ),

            "gaussian_support_leak_pdhg": float(
                obj_pdhg[
                    "support_leak"
                ][j].item()
            ),

            # Solution differences.
            "map_rel_diff_full": (
                map_rel
            ),

            "map_rel_diff_observed_centered": (
                map_rel_obs
            ),

            "gaussian_component_rel_diff": (
                g_rel
            ),

            "sparse_component_rel_diff": (
                s_rel
            ),

            # ADMM metrics.
            "nmse_admm": (
                admm_m[1]
            ),

            "pcc_admm": (
                admm_m[2]
            ),

            "std_admm": (
                admm_m[3]
            ),

            # PDHG metrics.
            "nmse_pdhg": (
                pdhg_m[1]
            ),

            "pcc_pdhg": (
                pdhg_m[2]
            ),

            "std_pdhg": (
                pdhg_m[3]
            ),

            # Metric differences.
            "nmse_delta_admm_minus_pdhg": (
                admm_m[1]
                - pdhg_m[1]
            ),

            "pcc_delta_admm_minus_pdhg": (
                admm_m[2]
                - pdhg_m[2]
            ),

            "std_delta_admm_minus_pdhg": (
                admm_m[3]
                - pdhg_m[3]
            ),
        }

        rows.append(
            row
        )

        print(
            f"C{c} R{r} P{p}  "
            f"Jrel={obj_rel:.3e}  "
            f"xrel={map_rel:.3e}  "
            f"xrel_obs={map_rel_obs:.3e}  "
            f"NMSE A/P="
            f"{admm_m[1]:.8f}/"
            f"{pdhg_m[1]:.8f}",
            flush=True,
        )

    # Resume-safe save.
    pd.DataFrame(
        rows
    ).to_csv(
        CSV,
        index=False,
    )


# =============================================================================
# FINAL CASE TABLE
# =============================================================================
tab = pd.DataFrame(
    rows
)

tab.to_csv(
    CSV,
    index=False,
)


# =============================================================================
# SUMMARY
# =============================================================================
summary = pd.DataFrame(
    [
        {
            "n_cases": len(tab),

            "admm_converged_cases": int(
                tab[
                    "admm_converged"
                ].sum()
            ),

            "admm_iterations": float(
                tab[
                    "admm_iterations"
                ].mean()
            ),

            "rho_final": float(
                tab[
                    "rho_final"
                ].mean()
            ),

            "primal_ratio_to_eps_max": float(
                tab[
                    "primal_ratio_to_eps"
                ].max()
            ),

            "dual_ratio_to_eps_max": float(
                tab[
                    "dual_ratio_to_eps"
                ].max()
            ),

            "primal_ratio_to_eps_median": float(
                tab[
                    "primal_ratio_to_eps"
                ].median()
            ),

            "dual_ratio_to_eps_median": float(
                tab[
                    "dual_ratio_to_eps"
                ].median()
            ),

            # Objective agreement.
            "obj_rel_median": float(
                tab[
                    "obj_rel_diff"
                ].median()
            ),

            "obj_rel_mean": float(
                tab[
                    "obj_rel_diff"
                ].mean()
            ),

            "obj_rel_max": float(
                tab[
                    "obj_rel_diff"
                ].max()
            ),

            # Map agreement.
            "map_rel_full_median": float(
                tab[
                    "map_rel_diff_full"
                ].median()
            ),

            "map_rel_full_mean": float(
                tab[
                    "map_rel_diff_full"
                ].mean()
            ),

            "map_rel_full_max": float(
                tab[
                    "map_rel_diff_full"
                ].max()
            ),

            "map_rel_obs_median": float(
                tab[
                    "map_rel_diff_observed_centered"
                ].median()
            ),

            "map_rel_obs_max": float(
                tab[
                    "map_rel_diff_observed_centered"
                ].max()
            ),

            # Component agreement.
            "gaussian_rel_median": float(
                tab[
                    "gaussian_component_rel_diff"
                ].median()
            ),

            "sparse_rel_median": float(
                tab[
                    "sparse_component_rel_diff"
                ].median()
            ),

            # Metrics.
            "nmse_admm_mean": float(
                tab[
                    "nmse_admm"
                ].mean()
            ),

            "nmse_pdhg_mean": float(
                tab[
                    "nmse_pdhg"
                ].mean()
            ),

            "nmse_delta_mean": float(
                tab[
                    "nmse_delta_admm_minus_pdhg"
                ].mean()
            ),

            "pcc_admm_mean": float(
                tab[
                    "pcc_admm"
                ].mean()
            ),

            "pcc_pdhg_mean": float(
                tab[
                    "pcc_pdhg"
                ].mean()
            ),

            "pcc_delta_mean": float(
                tab[
                    "pcc_delta_admm_minus_pdhg"
                ].mean()
            ),

            "std_admm_mean": float(
                tab[
                    "std_admm"
                ].mean()
            ),

            "std_pdhg_mean": float(
                tab[
                    "std_pdhg"
                ].mean()
            ),

            "std_delta_mean": float(
                tab[
                    "std_delta_admm_minus_pdhg"
                ].mean()
            ),

            "gaussian_support_leak_admm_max": float(
                tab[
                    "gaussian_support_leak_admm"
                ].max()
            ),

            "gaussian_support_leak_pdhg_max": float(
                tab[
                    "gaussian_support_leak_pdhg"
                ].max()
            ),
        }
    ]
)

summary.to_csv(
    SUMMARY_CSV,
    index=False,
)

print()
print("=" * 170)
print(
    "ADMM vs PDHG — SOLVER AUDIT SUMMARY"
)
print("=" * 170)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.10e}"
        ),
    )
)


# =============================================================================
# HUMAN-READABLE VERDICT
# =============================================================================
s = summary.iloc[0]

print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print(
    "Objective relative difference:"
)

print(
    f"  median = "
    f"{s.obj_rel_median:.3e}"
)

print(
    f"  max    = "
    f"{s.obj_rel_max:.3e}"
)

print()

print(
    "Reconstruction relative difference:"
)

print(
    f"  median full map = "
    f"{s.map_rel_full_median:.3e}"
)

print(
    f"  max full map    = "
    f"{s.map_rel_full_max:.3e}"
)

print(
    f"  median observed/centered = "
    f"{s.map_rel_obs_median:.3e}"
)

print(
    f"  max observed/centered    = "
    f"{s.map_rel_obs_max:.3e}"
)

print()

print(
    "Mean map metrics:"
)

print(
    f"  NMSE ADMM/PDHG = "
    f"{s.nmse_admm_mean:.9f} / "
    f"{s.nmse_pdhg_mean:.9f}"
)

print(
    f"  PCC  ADMM/PDHG = "
    f"{s.pcc_admm_mean:.9f} / "
    f"{s.pcc_pdhg_mean:.9f}"
)

print(
    f"  std  ADMM/PDHG = "
    f"{s.std_admm_mean:.9f} / "
    f"{s.std_pdhg_mean:.9f}"
)

print()

# Boyd convergence and solver-agreement verdict.
all_boyd = (
    int(
        s.admm_converged_cases
    )
    == int(
        s.n_cases
    )
)

if (
    all_boyd
    and
    s.obj_rel_max < 1e-4
    and
    s.map_rel_obs_max < 5e-3
):
    print(
        "VERDICT: all audit cases satisfy the standard "
        "Boyd stopping criteria, and ADMM/PDHG agree "
        "to numerical precision relevant to the manuscript."
    )

elif (
    s.obj_rel_max < 1e-4
    and
    s.map_rel_obs_max < 5e-3
):
    print(
        "VERDICT: ADMM/PDHG reconstruction agreement is "
        "already excellent, but not every batch satisfied "
        "the requested Boyd tolerances before maxiter."
    )

else:
    print(
        "VERDICT: inspect the solver diagnostics before "
        "claiming solver independence."
    )

print()
print("=" * 120)
print("FILES")
print("=" * 120)

print(CSV)
print(SUMMARY_CSV)
print(RECON_DIR)

print()
print(
    "SOLVER AUDIT COMPLETE"
)
