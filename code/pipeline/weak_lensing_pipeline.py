#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
weak_lensing_pipeline.py
========================

Reference weak-lensing reconstruction pipeline.

Objective
---------
    min_kappa
        1/2 || W^(1/2)(A kappa - gamma_obs) ||_2^2
        + lambda R(kappa)

where R(kappa) is the L1 norm of the DETAIL coefficients of an
orthonormal decimated 2-D DWT.

Scientific baseline
-------------------
    wavelet = db4
    mode    = periodization
    J       = 4

The coarsest approximation cA_J is NOT penalized.

Important
---------
- W is used literally.
- No normalization by f_obs or mean(W).
- PyWavelets is retained as the reference scientific DWT.
- Primary errors can be evaluated invariantly to the additive
  mass-sheet mode.
- A common proximal-gradient mapping residual is available for
  solver-independent convergence assessment.
"""

from __future__ import annotations

from typing import Optional, Callable, Tuple

import numpy as np
import torch
import pywt


# ============================================================
# HELPERS
# ============================================================

def _as_float_tensor(x, device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(
            device=device,
            dtype=torch.float32,
        )

    return torch.as_tensor(
        x,
        device=device,
        dtype=torch.float32,
    )


def _as_complex_tensor(x, device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(
            device=device,
            dtype=torch.complex64,
        )

    return torch.as_tensor(
        x,
        device=device,
        dtype=torch.complex64,
    )


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


# ============================================================
# 1. WEAK-LENSING OPERATOR
# ============================================================

class WeakLensingOperator:
    """
    Rectangular Kaiser-Squires shear operator.

    Forward:
        kappa -> gamma

    Fourier kernel:
                  (kx^2 - ky^2) + 2 i kx ky
        D(k) =   ----------------------------
                         kx^2 + ky^2

    with D(0)=0.
    """

    def __init__(
        self,
        H: int,
        W=None,
        device=None,
    ):
        # Backward-compatible:
        # WeakLensingOperator(N, device)

        if (
            device is None
            and W is not None
            and not isinstance(W, (int, np.integer))
        ):
            device = W
            W = H

        if W is None:
            W = H

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.H = int(H)
        self.W = int(W)
        self.device = torch.device(device)

        self._build_kernel()

    def _build_kernel(self):
        fy = torch.fft.fftfreq(
            self.H,
            d=1.0,
            device=self.device,
        )

        fx = torch.fft.fftfreq(
            self.W,
            d=1.0,
            device=self.device,
        )

        ky, kx = torch.meshgrid(
            fy,
            fx,
            indexing="ij",
        )

        denom = (
            kx**2 +
            ky**2
        )

        D = torch.zeros(
            self.H,
            self.W,
            device=self.device,
            dtype=torch.complex64,
        )

        nz = (
            denom > 0
        )

        D[nz] = torch.complex(
            (
                kx[nz]**2 -
                ky[nz]**2
            ) /
            denom[nz],

            (
                2.0 *
                kx[nz] *
                ky[nz]
            ) /
            denom[nz],
        )

        # Mass-sheet / DC mode.
        D[0, 0] = 0.0 + 0.0j

        self.D = D

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    def forward(
        self,
        kappa: torch.Tensor,
    ) -> torch.Tensor:

        kappa = _as_float_tensor(
            kappa,
            self.device,
        )

        if tuple(kappa.shape) != (
            self.H,
            self.W,
        ):
            raise ValueError(
                f"kappa shape mismatch: "
                f"{tuple(kappa.shape)} != "
                f"{(self.H, self.W)}"
            )

        kappa_hat = torch.fft.fft2(
            kappa
        )

        gamma_hat = (
            self.D *
            kappa_hat
        )

        gamma = torch.fft.ifft2(
            gamma_hat
        )

        return gamma.to(
            torch.complex64
        )

    apply_A = forward

    # --------------------------------------------------------
    # A^T
    # --------------------------------------------------------

    def adjoint(
        self,
        gamma: torch.Tensor,
    ) -> torch.Tensor:

        gamma = _as_complex_tensor(
            gamma,
            self.device,
        )

        if tuple(gamma.shape) != (
            self.H,
            self.W,
        ):
            raise ValueError(
                f"gamma shape mismatch: "
                f"{tuple(gamma.shape)} != "
                f"{(self.H, self.W)}"
            )

        gamma_hat = torch.fft.fft2(
            gamma
        )

        kappa_hat = (
            torch.conj(self.D) *
            gamma_hat
        )

        return (
            torch.fft.ifft2(
                kappa_hat
            )
            .real
            .to(torch.float32)
        )

    apply_AT = adjoint

    def normal(
        self,
        kappa: torch.Tensor,
    ) -> torch.Tensor:

        return self.adjoint(
            self.forward(kappa)
        )

    # --------------------------------------------------------
    # Robust dot test
    # --------------------------------------------------------

    def dot_test(
        self,
        seed: int = 12345,
    ) -> float:
        """
        Verify

            Re <A x, y> = <x, A^T y>

        using

            |lhs-rhs| / (||Ax|| ||y||).

        This avoids artificial inflation when lhs is small due to
        cancellation.
        """

        gen = torch.Generator(
            device=self.device
        )

        gen.manual_seed(seed)

        x = torch.randn(
            self.H,
            self.W,
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        yr = torch.randn(
            self.H,
            self.W,
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        yi = torch.randn(
            self.H,
            self.W,
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        y = torch.complex(
            yr,
            yi,
        )

        Ax = self.forward(x)
        ATy = self.adjoint(y)

        lhs = torch.real(
            torch.sum(
                torch.conj(Ax) *
                y
            )
        )

        rhs = torch.sum(
            x *
            ATy
        )

        abs_err = torch.abs(
            lhs -
            rhs
        )

        denom = (
            Ax.norm() *
            y.norm() +
            1e-30
        )

        return float(
            (
                abs_err /
                denom
            ).item()
        )


# ============================================================
# 2. KAISER-SQUIRES
# ============================================================

class KaiserSquires:

    def __init__(
        self,
        operator: WeakLensingOperator,
    ):
        self.op = operator
        self.device = operator.device

    def reconstruct(
        self,
        gamma_obs: torch.Tensor,
        smooth_sigma: Optional[float] = None,
    ) -> torch.Tensor:

        gamma_obs = _as_complex_tensor(
            gamma_obs,
            self.device,
        )

        gamma_hat = torch.fft.fft2(
            gamma_obs
        )

        if (
            smooth_sigma is not None
            and
            float(smooth_sigma) > 0.0
        ):
            sigma = float(
                smooth_sigma
            )

            fy = torch.fft.fftfreq(
                self.op.H,
                d=1.0,
                device=self.device,
            )

            fx = torch.fft.fftfreq(
                self.op.W,
                d=1.0,
                device=self.device,
            )

            ky, kx = torch.meshgrid(
                fy,
                fx,
                indexing="ij",
            )

            wy = (
                2.0 *
                np.pi *
                ky
            )

            wx = (
                2.0 *
                np.pi *
                kx
            )

            gaussian = torch.exp(
                -0.5 *
                sigma**2 *
                (
                    wx**2 +
                    wy**2
                )
            )

            gamma_hat = (
                gamma_hat *
                gaussian
            )

        kappa_hat = (
            torch.conj(self.op.D) *
            gamma_hat
        )

        return (
            torch.fft.ifft2(
                kappa_hat
            )
            .real
            .to(torch.float32)
        )


# ============================================================
# 3. ORTHONORMAL DWT — PYWAVELETS REFERENCE
# ============================================================

class WaveletTransform:
    """
    Orthonormal decimated DWT.

    Baseline:
        db4
        periodization
        J=4

    Coefficients:
        approx = cA_J

        details = [
            (cH_J,cV_J,cD_J),
            ...
            (cH_1,cV_1,cD_1)
        ]

    Only DETAILS are penalized.
    """

    def __init__(
        self,
        J: int,
        device,
        wavelet_name: str = "db4",
        mode: str = "periodization",
    ):
        self.J = int(J)
        self.device = torch.device(device)

        self.wavelet_name = str(
            wavelet_name
        )

        self.mode = str(
            mode
        )

        self.wavelet = pywt.Wavelet(
            self.wavelet_name
        )

        if not self.wavelet.orthogonal:
            raise ValueError(
                f"{self.wavelet_name} is not orthogonal."
            )

        if self.mode != "periodization":
            raise ValueError(
                "Revision baseline requires "
                "mode='periodization'."
            )

    @property
    def frame_norm_sq(
        self,
    ) -> float:

        return 1.0

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    def decompose(
        self,
        kappa: torch.Tensor,
    ) -> dict:

        if kappa.ndim != 2:
            raise ValueError(
                "WaveletTransform expects a 2-D tensor."
            )

        H, W = map(
            int,
            kappa.shape,
        )

        original_device = (
            kappa.device
        )

        x_np = (
            kappa
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        coeffs_np = pywt.wavedec2(
            x_np,
            wavelet=self.wavelet_name,
            mode=self.mode,
            level=self.J,
        )

        approx = torch.from_numpy(
            np.asarray(
                coeffs_np[0],
                dtype=np.float32,
            )
        ).to(
            original_device
        )

        details = []

        for (
            cH_np,
            cV_np,
            cD_np,
        ) in coeffs_np[1:]:

            cH = torch.from_numpy(
                np.asarray(
                    cH_np,
                    dtype=np.float32,
                )
            ).to(
                original_device
            )

            cV = torch.from_numpy(
                np.asarray(
                    cV_np,
                    dtype=np.float32,
                )
            ).to(
                original_device
            )

            cD = torch.from_numpy(
                np.asarray(
                    cD_np,
                    dtype=np.float32,
                )
            ).to(
                original_device
            )

            details.append(
                (
                    cH,
                    cV,
                    cD,
                )
            )

        return {
            "approx":
                approx,

            "details":
                details,

            "device":
                original_device,

            "shape":
                (
                    H,
                    W,
                ),
        }

    # --------------------------------------------------------
    # Synthesis
    # --------------------------------------------------------

    def reconstruct(
        self,
        coeffs: dict,
    ) -> torch.Tensor:

        target_device = torch.device(
            coeffs.get(
                "device",
                self.device,
            )
        )

        H, W = (
            coeffs["shape"]
        )

        coeffs_np = [
            coeffs[
                "approx"
            ]
            .detach()
            .float()
            .cpu()
            .numpy()
        ]

        for (
            cH,
            cV,
            cD,
        ) in coeffs[
            "details"
        ]:

            coeffs_np.append(
                (
                    cH
                    .detach()
                    .float()
                    .cpu()
                    .numpy(),

                    cV
                    .detach()
                    .float()
                    .cpu()
                    .numpy(),

                    cD
                    .detach()
                    .float()
                    .cpu()
                    .numpy(),
                )
            )

        rec_np = pywt.waverec2(
            coeffs_np,
            wavelet=self.wavelet_name,
            mode=self.mode,
        )

        rec_np = rec_np[
            :H,
            :W,
        ]

        return torch.from_numpy(
            np.asarray(
                rec_np,
                dtype=np.float32,
            )
        ).to(
            target_device
        )

    # --------------------------------------------------------
    # Coefficient algebra
    # --------------------------------------------------------

    @staticmethod
    def clone_coeffs(
        coeffs: dict,
    ) -> dict:

        return {
            "approx":
                coeffs[
                    "approx"
                ].clone(),

            "details": [
                (
                    h.clone(),
                    v.clone(),
                    d.clone(),
                )
                for (
                    h,
                    v,
                    d,
                )
                in coeffs[
                    "details"
                ]
            ],

            "device":
                coeffs[
                    "device"
                ],

            "shape":
                coeffs[
                    "shape"
                ],
        }

    @staticmethod
    def zeros_like_coeffs(
        coeffs: dict,
    ) -> dict:

        return {
            "approx":
                torch.zeros_like(
                    coeffs[
                        "approx"
                    ]
                ),

            "details": [
                (
                    torch.zeros_like(h),
                    torch.zeros_like(v),
                    torch.zeros_like(d),
                )
                for (
                    h,
                    v,
                    d,
                )
                in coeffs[
                    "details"
                ]
            ],

            "device":
                coeffs[
                    "device"
                ],

            "shape":
                coeffs[
                    "shape"
                ],
        }

    @staticmethod
    def add_coeffs(
        a: dict,
        b: dict,
    ) -> dict:

        return {
            "approx":
                a["approx"] +
                b["approx"],

            "details": [
                (
                    ah + bh,
                    av + bv,
                    ad + bd,
                )
                for (
                    (ah, av, ad),
                    (bh, bv, bd),
                )
                in zip(
                    a["details"],
                    b["details"],
                )
            ],

            "device":
                a["device"],

            "shape":
                a["shape"],
        }

    @staticmethod
    def sub_coeffs(
        a: dict,
        b: dict,
    ) -> dict:

        return {
            "approx":
                a["approx"] -
                b["approx"],

            "details": [
                (
                    ah - bh,
                    av - bv,
                    ad - bd,
                )
                for (
                    (ah, av, ad),
                    (bh, bv, bd),
                )
                in zip(
                    a["details"],
                    b["details"],
                )
            ],

            "device":
                a["device"],

            "shape":
                a["shape"],
        }

    @staticmethod
    def scale_coeffs(
        a: dict,
        scalar: float,
    ) -> dict:

        scalar = float(
            scalar
        )

        return {
            "approx":
                scalar *
                a["approx"],

            "details": [
                (
                    scalar * h,
                    scalar * v,
                    scalar * d,
                )
                for (
                    h,
                    v,
                    d,
                )
                in a[
                    "details"
                ]
            ],

            "device":
                a["device"],

            "shape":
                a["shape"],
        }

    @staticmethod
    def coeff_norm(
        coeffs: dict,
    ) -> float:

        s = (
            coeffs[
                "approx"
            ]
            .pow(2)
            .sum()
        )

        for (
            h,
            v,
            d,
        ) in coeffs[
            "details"
        ]:

            s = (
                s +
                h.pow(2).sum() +
                v.pow(2).sum() +
                d.pow(2).sum()
            )

        return float(
            torch.sqrt(
                s
            ).item()
        )

    # --------------------------------------------------------
    # Regularizer
    # --------------------------------------------------------

    @staticmethod
    def regularizer(
        coeffs: dict,
    ) -> float:
        """
        L1 norm of DETAILS only.
        """

        reg = torch.zeros(
            (),
            device=coeffs[
                "approx"
            ].device,
            dtype=coeffs[
                "approx"
            ].dtype,
        )

        for (
            h,
            v,
            d,
        ) in coeffs[
            "details"
        ]:

            reg = (
                reg +
                h.abs().sum() +
                v.abs().sum() +
                d.abs().sum()
            )

        return float(
            reg.item()
        )

    # --------------------------------------------------------
    # Thresholding
    # --------------------------------------------------------

    @staticmethod
    def soft_threshold_tensor(
        x: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:

        return (
            torch.sign(x) *
            torch.clamp(
                torch.abs(x) -
                float(tau),
                min=0.0,
            )
        )

    def soft_threshold_details(
        self,
        coeffs: dict,
        tau: float,
    ) -> dict:

        out = self.clone_coeffs(
            coeffs
        )

        out["details"] = [
            (
                self.soft_threshold_tensor(
                    h,
                    tau,
                ),
                self.soft_threshold_tensor(
                    v,
                    tau,
                ),
                self.soft_threshold_tensor(
                    d,
                    tau,
                ),
            )
            for (
                h,
                v,
                d,
            )
            in coeffs[
                "details"
            ]
        ]

        # cA_J unchanged.

        return out

    def prox_details_l1(
        self,
        x: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """
        Exact proximal map for details-only L1 under an
        orthonormal DWT.
        """

        coeffs = self.decompose(
            x
        )

        coeffs = self.soft_threshold_details(
            coeffs,
            tau,
        )

        return self.reconstruct(
            coeffs
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate_orthonormality(
        self,
        shape,
        seed: int = 1234,
    ) -> dict:

        H, W = map(
            int,
            shape,
        )

        gen = torch.Generator(
            device=self.device
        )

        gen.manual_seed(
            seed
        )

        x = torch.randn(
            H,
            W,
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        y = torch.randn(
            H,
            W,
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        cx = self.decompose(
            x
        )

        cy = self.decompose(
            y
        )

        xr = self.reconstruct(
            cx
        )

        roundtrip_rel = float(
            (
                torch.norm(
                    xr -
                    x
                ) /
                (
                    torch.norm(x) +
                    1e-30
                )
            ).item()
        )

        def coeff_energy(
            c,
        ):
            e = (
                c["approx"]
                .pow(2)
                .sum()
            )

            for (
                h,
                v,
                d,
            ) in c[
                "details"
            ]:
                e = (
                    e +
                    h.pow(2).sum() +
                    v.pow(2).sum() +
                    d.pow(2).sum()
                )

            return e

        image_energy = (
            x.pow(2)
            .sum()
        )

        wave_energy = coeff_energy(
            cx
        )

        energy_rel = float(
            (
                torch.abs(
                    wave_energy -
                    image_energy
                ) /
                (
                    torch.abs(
                        image_energy
                    ) +
                    1e-30
                )
            ).item()
        )

        def coeff_inner(
            a,
            b,
        ):
            value = (
                a["approx"] *
                b["approx"]
            ).sum()

            for (
                (ah, av, ad),
                (bh, bv, bd),
            ) in zip(
                a["details"],
                b["details"],
            ):
                value = (
                    value +
                    (ah * bh).sum() +
                    (av * bv).sum() +
                    (ad * bd).sum()
                )

            return value

        lhs = (
            x *
            y
        ).sum()

        rhs = coeff_inner(
            cx,
            cy,
        )

        denom = max(
            abs(
                float(lhs.item())
            ),
            abs(
                float(rhs.item())
            ),
            1e-30,
        )

        inner_product_rel = float(
            abs(
                float(lhs.item()) -
                float(rhs.item())
            ) /
            denom
        )

        return {
            "roundtrip_rel":
                roundtrip_rel,

            "energy_rel":
                energy_rel,

            "inner_product_rel":
                inner_product_rel,
        }


# ============================================================
# 4. COMMON OPTIMALITY RESIDUAL
# ============================================================

def composite_gradient_mapping_residual(
    operator: WeakLensingOperator,
    wavelet: WaveletTransform,
    kappa: torch.Tensor,
    gamma_obs: torch.Tensor,
    weight_map: torch.Tensor,
    lam: float,
    step: Optional[float] = None,
) -> float:
    """
    Relative proximal-gradient mapping residual.

    Let

        f(x) = 1/2 ||W^(1/2)(Ax-y)||^2

    and

        F(x) = f(x) + lambda R(x).

    Define

        G_t(x)
            =
        [x - prox_{t lambda R}(x - t grad f(x))] / t.

    Return

        ||G_t(x)|| / max(||x||,1).

    This is solver-independent and can therefore be used for
    both FISTA and ADMM outputs.
    """

    device = (
        operator.device
    )

    kappa = _as_float_tensor(
        kappa,
        device,
    )

    gamma_obs = _as_complex_tensor(
        gamma_obs,
        device,
    )

    W = _as_float_tensor(
        weight_map,
        device,
    )

    max_w = float(
        W.max().item()
    )

    if max_w <= 0:
        raise ValueError(
            "weight_map must contain positive weights."
        )

    if step is None:
        step = (
            0.99 /
            max_w
        )

    step = float(
        step
    )

    residual = (
        operator.forward(kappa) -
        gamma_obs
    )

    weighted_residual = torch.complex(
        W *
        residual.real,

        W *
        residual.imag,
    )

    grad = operator.adjoint(
        weighted_residual
    )

    trial = (
        kappa -
        step *
        grad
    )

    prox = wavelet.prox_details_l1(
        trial,
        tau=(
            step *
            float(lam)
        ),
    )

    G = (
        kappa -
        prox
    ) / step

    denom = max(
        float(
            kappa.norm().item()
        ),
        1.0,
    )

    return float(
        G.norm().item() /
        denom
    )


# ============================================================
# 5. FISTA
# ============================================================

class FISTA:
    """
    Solve

        min_kappa
            1/2 ||W^(1/2)(A kappa - gamma_obs)||^2
            + lambda R(kappa)

    using exact proximal FISTA.

    Optional convergence criterion:
        relative proximal-gradient mapping residual.
    """

    def __init__(
        self,
        operator: WeakLensingOperator,
        wavelet: WaveletTransform,
        lam: float,
        n_iter: int = 300,
        weight_map: Optional[torch.Tensor] = None,
        step: Optional[float] = None,
        power_iters: int = 50,
        safety_factor: float = 0.99,
        tol: Optional[float] = None,
        check_every: int = 5,
        min_iter: int = 10,
        optimality_step: Optional[float] = None,
        verbose: bool = False,
    ):
        self.op = operator
        self.wav = wavelet

        self.lam = float(
            lam
        )

        self.n_iter = int(
            n_iter
        )

        self.verbose = bool(
            verbose
        )

        self.tol = (
            None
            if tol is None
            else float(tol)
        )

        self.check_every = max(
            int(check_every),
            1,
        )

        self.min_iter = max(
            int(min_iter),
            1,
        )

        self.optimality_step = (
            None
            if optimality_step is None
            else float(
                optimality_step
            )
        )

        H, W = (
            operator.H,
            operator.W,
        )

        device = (
            operator.device
        )

        if weight_map is None:
            self.W = torch.ones(
                H,
                W,
                device=device,
                dtype=torch.float32,
            )

        else:
            if tuple(
                weight_map.shape
            ) != (
                H,
                W,
            ):
                raise ValueError(
                    "weight_map shape mismatch."
                )

            self.W = weight_map.to(
                device=device,
                dtype=torch.float32,
            )

        if torch.any(
            self.W < 0
        ):
            raise ValueError(
                "Weights must be non-negative."
            )

        if (
            self.W.max().item()
            <= 0
        ):
            raise ValueError(
                "At least one weight must be positive."
            )

        # ----------------------------------------------------
        # Step size
        # --------------------------------------------------------

        if step is None:
            # Constant W.
            if (
                self.W.max() -
                self.W.min()
            ).abs().item() < 1e-10:

                L = max(
                    float(
                        self.W.mean().item()
                    ),
                    1e-12,
                )

            else:
                L = self._estimate_lipschitz(
                    n_iter=power_iters
                )

            self.L = float(
                L
            )

            self.step = (
                float(
                    safety_factor
                ) /
                (
                    self.L +
                    1e-12
                )
            )

        else:
            self.step = float(
                step
            )

            self.L = (
                1.0 /
                self.step
            )

    # --------------------------------------------------------
    # Weighted A^T W A
    # --------------------------------------------------------

    def _apply_normal(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        Ax = self.op.forward(
            x
        )

        WAx = torch.complex(
            self.W *
            Ax.real,

            self.W *
            Ax.imag,
        )

        return self.op.adjoint(
            WAx
        )

    # --------------------------------------------------------
    # Power iteration
    # --------------------------------------------------------

    def _estimate_lipschitz(
        self,
        n_iter: int = 50,
    ) -> float:

        device = (
            self.op.device
        )

        gen = torch.Generator(
            device=device
        )

        gen.manual_seed(
            98765
        )

        v = torch.randn(
            self.op.H,
            self.op.W,
            generator=gen,
            device=device,
            dtype=torch.float32,
        )

        v /= (
            v.norm() +
            1e-30
        )

        for _ in range(
            int(n_iter)
        ):
            Bv = self._apply_normal(
                v
            )

            nrm = Bv.norm()

            if nrm.item() < 1e-20:
                return 1e-12

            v = (
                Bv /
                nrm
            )

        Bv = self._apply_normal(
            v
        )

        L = torch.sum(
            v *
            Bv
        ).item()

        return max(
            float(L),
            1e-12,
        )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    def _gradient(
        self,
        kappa: torch.Tensor,
        gamma_obs: torch.Tensor,
    ) -> torch.Tensor:

        residual = (
            self.op.forward(kappa) -
            gamma_obs
        )

        weighted_residual = torch.complex(
            self.W *
            residual.real,

            self.W *
            residual.imag,
        )

        return self.op.adjoint(
            weighted_residual
        )

    # --------------------------------------------------------
    # Data term
    # --------------------------------------------------------

    def data_fidelity(
        self,
        kappa: torch.Tensor,
        gamma_obs: torch.Tensor,
    ) -> float:

        residual = (
            self.op.forward(kappa) -
            gamma_obs
        )

        value = (
            0.5 *
            torch.sum(
                self.W *
                (
                    residual.real**2 +
                    residual.imag**2
                )
            )
        )

        return float(
            value.item()
        )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(
        self,
        kappa: torch.Tensor,
        gamma_obs: torch.Tensor,
    ) -> float:

        data = self.data_fidelity(
            kappa,
            gamma_obs,
        )

        reg = self.wav.regularizer(
            self.wav.decompose(
                kappa
            )
        )

        return float(
            data +
            self.lam *
            reg
        )

    # --------------------------------------------------------
    # Optimality
    # --------------------------------------------------------

    def optimality_residual(
        self,
        kappa: torch.Tensor,
        gamma_obs: torch.Tensor,
    ) -> float:

        return composite_gradient_mapping_residual(
            operator=self.op,
            wavelet=self.wav,
            kappa=kappa,
            gamma_obs=gamma_obs,
            weight_map=self.W,
            lam=self.lam,
            step=self.optimality_step,
        )

    # Backward-compatible diagnostic name.
    def gradient_mapping_residual(
        self,
        kappa,
        gamma_obs,
    ) -> float:

        return self.optimality_residual(
            kappa,
            gamma_obs,
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    def run(
        self,
        gamma_obs: torch.Tensor,
        kappa_init: Optional[torch.Tensor] = None,
        callback: Optional[Callable] = None,
    ) -> Tuple[
        torch.Tensor,
        list,
    ]:

        device = (
            self.op.device
        )

        gamma_obs = _as_complex_tensor(
            gamma_obs,
            device,
        )

        if kappa_init is None:
            x = torch.zeros(
                self.op.H,
                self.op.W,
                device=device,
                dtype=torch.float32,
            )

        else:
            x = (
                kappa_init
                .clone()
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

        y = x.clone()

        t = 1.0

        history = []

        converged = False

        for iteration in range(
            1,
            self.n_iter + 1,
        ):
            # Prox-gradient step.
            grad = self._gradient(
                y,
                gamma_obs,
            )

            trial = (
                y -
                self.step *
                grad
            )

            x_new = self.wav.prox_details_l1(
                trial,
                tau=(
                    self.step *
                    self.lam
                ),
            )

            # Nesterov acceleration.
            t_new = (
                1.0 +
                np.sqrt(
                    1.0 +
                    4.0 *
                    t**2
                )
            ) / 2.0

            momentum = (
                (t - 1.0) /
                t_new
            )

            y = (
                x_new +
                momentum *
                (
                    x_new -
                    x
                )
            )

            x = x_new
            t = t_new

            # Objective.
            data = self.data_fidelity(
                x,
                gamma_obs,
            )

            reg = self.wav.regularizer(
                self.wav.decompose(
                    x
                )
            )

            obj = (
                data +
                self.lam *
                reg
            )

            # Optimality.
            opt_res = None

            should_check = (
                iteration >=
                self.min_iter
                and
                (
                    iteration %
                    self.check_every
                    == 0
                    or
                    iteration ==
                    self.n_iter
                )
            )

            if should_check:
                opt_res = self.optimality_residual(
                    x,
                    gamma_obs,
                )

                if (
                    self.tol is not None
                    and
                    opt_res <
                    self.tol
                ):
                    converged = True

            record = {
                "iter":
                    int(iteration),

                "objective":
                    float(obj),

                "data_term":
                    float(data),

                "regularizer":
                    float(reg),

                "step":
                    float(self.step),

                "L":
                    float(self.L),

                "optimality_residual":
                    opt_res,

                "converged":
                    bool(converged),
            }

            history.append(
                record
            )

            if self.verbose:
                opt_txt = (
                    ""
                    if opt_res is None
                    else
                    f" | opt={opt_res:.3e}"
                )

                if (
                    iteration == 1
                    or
                    iteration % 10 == 0
                    or
                    converged
                ):
                    print(
                        f"FISTA {iteration:4d} | "
                        f"obj={obj:.6e} | "
                        f"data={data:.6e} | "
                        f"reg={reg:.6e}"
                        f"{opt_txt}"
                    )

            if callback is not None:
                callback(
                    iteration,
                    x,
                    record,
                )

            if converged:
                break

        return (
            x,
            history,
        )


# ============================================================
# 6. NOISE / OBSERVATION
# ============================================================

def add_shape_noise(
    gamma: torch.Tensor,
    sigma_n: float,
    seed: int = 0,
) -> torch.Tensor:
    """
    Independent Gaussian noise in each shear component.
    """

    sigma_n = float(
        sigma_n
    )

    if sigma_n < 0:
        raise ValueError(
            "sigma_n must be non-negative."
        )

    device = (
        gamma.device
    )

    gen = torch.Generator(
        device=device
    )

    gen.manual_seed(
        int(seed)
    )

    nr = torch.randn(
        gamma.shape,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )

    ni = torch.randn(
        gamma.shape,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )

    noise = torch.complex(
        sigma_n *
        nr,

        sigma_n *
        ni,
    )

    return (
        gamma +
        noise
    )


def apply_mask_to_complex(
    field: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    mask = mask.to(
        field.device,
        dtype=torch.float32,
    )

    return torch.complex(
        mask *
        field.real,

        mask *
        field.imag,
    )


def generate_lensing_observation(
    kappa: torch.Tensor,
    operator: WeakLensingOperator,
    mask: Optional[torch.Tensor] = None,
    sigma_n: float = 0.0,
    seed: int = 0,
    mode: str = "shear",
):
    """
    Generate controlled weak-lensing observations.

    mode='shear':
        gamma = A kappa

    mode='reduced_shear':
        g = gamma / (1-kappa)

    Returns
    -------
    observed, noiseless_signal
    """

    kappa = _as_float_tensor(
        kappa,
        operator.device,
    )

    gamma = operator.forward(
        kappa
    )

    mode = str(
        mode
    ).lower()

    if mode == "shear":
        signal = gamma

    elif mode == "reduced_shear":
        denom = (
            1.0 -
            kappa
        )

        eps = 1e-6

        sign = torch.where(
            denom >= 0,
            torch.ones_like(
                denom
            ),
            -torch.ones_like(
                denom
            ),
        )

        denom = torch.where(
            torch.abs(
                denom
            ) < eps,
            sign *
            eps,
            denom,
        )

        signal = (
            gamma /
            denom
        )

    else:
        raise ValueError(
            "mode must be 'shear' or 'reduced_shear'."
        )

    if float(
        sigma_n
    ) > 0:
        observed = add_shape_noise(
            signal,
            sigma_n=sigma_n,
            seed=seed,
        )

    else:
        observed = signal.clone()

    if mask is not None:
        observed = apply_mask_to_complex(
            observed,
            mask,
        )

        signal = apply_mask_to_complex(
            signal,
            mask,
        )

    return (
        observed,
        signal,
    )


# ============================================================
# 7. METRICS
# ============================================================

def _valid_mask(
    pred,
    truth,
    mask,
):
    p = _to_numpy(
        pred
    )

    t = _to_numpy(
        truth
    )

    m = _to_numpy(
        mask
    ).astype(
        bool
    )

    valid = (
        m &
        np.isfinite(p) &
        np.isfinite(t)
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "No valid pixels inside evaluation mask."
        )

    return (
        p,
        t,
        valid,
    )


def masked_mse(
    pred,
    truth,
    mask,
) -> float:
    """
    Raw masked MSE.
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    return float(
        np.mean(
            (
                p[valid] -
                t[valid]
            )**2
        )
    )


def masked_nmse(
    pred,
    truth,
    mask,
) -> float:
    """
    Raw masked NMSE.
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    err = (
        p[valid] -
        t[valid]
    )

    denom = np.sum(
        t[valid]**2
    )

    if denom <= 0:
        return np.nan

    return float(
        np.sum(
            err**2
        ) /
        denom
    )


# ============================================================
# 8. MASS-SHEET-INVARIANT METRICS
# ============================================================

def remove_mass_sheet(
    field,
    mask,
):
    """
    Remove the observed-footprint mean.
    """

    arr = np.array(
        _to_numpy(field),
        copy=True,
        dtype=np.float64,
    )

    m = _to_numpy(
        mask
    ).astype(
        bool
    )

    valid = (
        m &
        np.isfinite(arr)
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "No valid pixels inside mask."
        )

    arr[valid] -= float(
        np.mean(
            arr[valid]
        )
    )

    return arr


def align_mass_sheet(
    pred,
    truth,
    mask,
):
    """
    Optimal additive alignment:

        min_c ||pred + c - truth||^2_mask
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    aligned = np.array(
        p,
        copy=True,
        dtype=np.float64,
    )

    offset = float(
        np.mean(
            t[valid] -
            p[valid]
        )
    )

    aligned[valid] += (
        offset
    )

    return (
        aligned,
        offset,
    )


def masked_mse_ms(
    pred,
    truth,
    mask,
) -> float:
    """
    Mass-sheet-invariant MSE.
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    pv = (
        p[valid] -
        np.mean(
            p[valid]
        )
    )

    tv = (
        t[valid] -
        np.mean(
            t[valid]
        )
    )

    return float(
        np.mean(
            (
                pv -
                tv
            )**2
        )
    )


def masked_nmse_ms(
    pred,
    truth,
    mask,
) -> float:
    """
    Mass-sheet-invariant NMSE.
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    pv = (
        p[valid] -
        np.mean(
            p[valid]
        )
    )

    tv = (
        t[valid] -
        np.mean(
            t[valid]
        )
    )

    numerator = np.sum(
        (
            pv -
            tv
        )**2
    )

    denominator = np.sum(
        tv**2
    )

    if denominator <= 0:
        return np.nan

    return float(
        numerator /
        denominator
    )


masked_mse_mass_sheet_invariant = (
    masked_mse_ms
)

masked_nmse_mass_sheet_invariant = (
    masked_nmse_ms
)


def masked_pcc(
    pred,
    truth,
    mask,
) -> float:
    """
    Pearson correlation over observed pixels.
    """

    p, t, valid = _valid_mask(
        pred,
        truth,
        mask,
    )

    pv = p[valid].astype(
        np.float64
    )

    tv = t[valid].astype(
        np.float64
    )

    pv -= pv.mean()
    tv -= tv.mean()

    denom = (
        np.sqrt(
            np.sum(
                pv**2
            )
        ) *
        np.sqrt(
            np.sum(
                tv**2
            )
        )
    )

    if denom <= 0:
        return np.nan

    return float(
        np.sum(
            pv *
            tv
        ) /
        denom
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "WeakLensingOperator",
    "KaiserSquires",
    "WaveletTransform",
    "FISTA",

    "composite_gradient_mapping_residual",

    "add_shape_noise",
    "apply_mask_to_complex",
    "generate_lensing_observation",

    "masked_mse",
    "masked_nmse",

    "remove_mass_sheet",
    "align_mass_sheet",

    "masked_mse_ms",
    "masked_nmse_ms",

    "masked_mse_mass_sheet_invariant",
    "masked_nmse_mass_sheet_invariant",

    "masked_pcc",
]


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    H = 176
    W = 176

    print("=" * 72)
    print("weak_lensing_pipeline.py")
    print("=" * 72)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Operator
    # --------------------------------------------------------

    op = WeakLensingOperator(
        H,
        W,
        DEVICE,
    )

    dot_err = op.dot_test()

    print(
        f"\nOperator dot-test: "
        f"{dot_err:.3e}"
    )

    # --------------------------------------------------------
    # Wavelet
    # --------------------------------------------------------

    wav = WaveletTransform(
        J=4,
        device=DEVICE,
        wavelet_name="db4",
        mode="periodization",
    )

    tests = wav.validate_orthonormality(
        (
            H,
            W,
        )
    )

    print(
        "\nWavelet:"
    )

    for key, value in tests.items():
        print(
            f"  {key:24s}"
            f"{value:.3e}"
        )

    # --------------------------------------------------------
    # Mass-sheet metrics
    # --------------------------------------------------------

    rng = np.random.default_rng(
        123
    )

    truth = rng.normal(
        size=(
            H,
            W,
        )
    )

    pred = (
        truth +
        3.5
    )

    mask = (
        rng.random(
            (
                H,
                W,
            )
        ) >
        0.3
    )

    print(
        "\nMetric test:"
    )

    print(
        "raw MSE:",
        masked_mse(
            pred,
            truth,
            mask,
        )
    )

    print(
        "MS MSE :",
        masked_mse_ms(
            pred,
            truth,
            mask,
        )
    )

    print(
        "MS NMSE:",
        masked_nmse_ms(
            pred,
            truth,
            mask,
        )
    )

    print(
        "PCC    :",
        masked_pcc(
            pred,
            truth,
            mask,
        )
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert dot_err < 1e-8

    assert (
        tests[
            "roundtrip_rel"
        ] <
        1e-5
    )

    assert (
        tests[
            "energy_rel"
        ] <
        1e-5
    )

    assert (
        tests[
            "inner_product_rel"
        ] <
        1e-5
    )

    assert abs(
        masked_mse(
            pred,
            truth,
            mask,
        ) -
        12.25
    ) < 1e-10

    assert (
        masked_mse_ms(
            pred,
            truth,
            mask,
        ) <
        1e-20
    )

    assert (
        masked_nmse_ms(
            pred,
            truth,
            mask,
        ) <
        1e-20
    )

    assert abs(
        masked_pcc(
            pred,
            truth,
            mask,
        ) -
        1.0
    ) < 1e-12

    print(
        "\n" +
        "=" * 72
    )

    print(
        "ALL SMOKE TESTS PASSED"
    )

    print(
        "=" * 72
    )