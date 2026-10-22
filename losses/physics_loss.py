
"""
====================================================================
Physics-Informed 3D Acoustic Wave-Equation Loss
====================================================================

PhD framework:
Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings.

Governing equation
------------------
    (1 / v^2) * d2p/dt2 - Laplacian(p) = s

Residual
--------
    R_acoustic = (1 / v^2) * d2p/dt2 - Laplacian(p) - s

Physics loss
------------
    L_acoustic = mean(R_acoustic ** 2)

Tensor conventions
------------------
Pressure and source:
    [B, 1, T, D, H, W]

Velocity:
    [B, 1, D, H, W]

where:
    B = batch
    T = time
    D = depth
    H = crossline
    W = inline

Physical sampling:
    dt = temporal sample interval
    dz = depth sample interval
    dy = crossline sample interval
    dx = inline sample interval

Important
---------
The pressure input must be a physically defined, time-dependent
acoustic pressure field. A static reconstructed seismic cube must
not automatically be interpreted as such a field.

The source term must use units consistent with the stated
differential equation. Supply an explicit zero tensor only when
a source-free region or homogeneous equation is intended.

Spatial and temporal boundary points are excluded from the
central-difference residual. Appropriate initial and boundary
conditions must be handled by the forward model or additional
validated objectives.

Author: Ormin Joseph
====================================================================
"""

import math

import torch
import torch.nn as nn


class AcousticWaveEquationLoss(nn.Module):
    """
    Acoustic wave-equation residual loss for a 3D scalar pressure
    wavefield.

    Expected pressure shape:
        [B, 1, T, D, H, W]

    Expected velocity shape:
        [B, 1, D, H, W]

    Expected source shape:
        [B, 1, T, D, H, W]

    All sampling intervals must use consistent physical units.
    For example, use seconds for dt and metres for dx, dy, dz,
    with velocity expressed in metres per second.
    """

    def __init__(
        self,
        dt,
        dx,
        dy,
        dz,
    ):
        super().__init__()

        self._validate_spacing(dt, "dt")
        self._validate_spacing(dx, "dx")
        self._validate_spacing(dy, "dy")
        self._validate_spacing(dz, "dz")

        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

    @staticmethod
    def _validate_spacing(value, name):
        """Require a positive, finite physical sampling interval."""

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"{name} must be a numeric scalar."
            )

        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(
                f"{name} must be positive and finite."
            )

    @staticmethod
    def _validate_tensor(tensor, name, ndim):
        """Validate tensor type, dimensionality and finiteness."""

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"{name} must have a floating-point dtype."
            )

        if tensor.ndim != ndim:
            raise ValueError(
                f"{name} must have {ndim} dimensions; "
                f"received shape {tuple(tensor.shape)}."
            )

        if any(size < 1 for size in tensor.shape):
            raise ValueError(
                f"{name} contains an empty dimension."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    def _second_difference(self, field, spacing, dim):
        """
        Second-order central finite difference.

        The output contains only interior points along `dim`.
        Boundary points are deliberately excluded.
        """

        if field.shape[dim] < 3:
            raise ValueError(
                f"Dimension {dim} needs at least three samples "
                "for a central second derivative."
            )

        center = [slice(None)] * field.ndim
        before = [slice(None)] * field.ndim
        after = [slice(None)] * field.ndim

        center[dim] = slice(1, -1)
        before[dim] = slice(None, -2)
        after[dim] = slice(2, None)

        return (
            field[tuple(after)]
            - 2.0 * field[tuple(center)]
            + field[tuple(before)]
        ) / (spacing ** 2)

    def acoustic_residual(
        self,
        pressure,
        velocity,
        source,
    ):
        """
        Calculate the acoustic PDE residual on interior points.

        pressure: [B, 1, T, D, H, W]
        velocity: [B, 1, D, H, W]
        source:   [B, 1, T, D, H, W]

        Returns:
            Residual tensor with shape
            [B, 1, T-2, D-2, H-2, W-2].
        """

        self._validate_tensor(
            pressure, "pressure", ndim=6
        )
        self._validate_tensor(
            velocity, "velocity", ndim=5
        )
        self._validate_tensor(
            source, "source", ndim=6
        )

        if pressure.shape[1] != 1:
            raise ValueError(
                "Pressure must have one scalar channel."
            )

        if velocity.shape != (
            pressure.shape[0],
            1,
            pressure.shape[3],
            pressure.shape[4],
            pressure.shape[5],
        ):
            raise ValueError(
                "Velocity must have shape [B,1,D,H,W] "
                "matching the pressure batch and spatial axes."
            )

        if source.shape != pressure.shape:
            raise ValueError(
                "Source and pressure must have identical shapes."
            )

        if pressure.device != velocity.device:
            raise ValueError(
                "Pressure and velocity must be on the same device."
            )

        if pressure.device != source.device:
            raise ValueError(
                "Pressure and source must be on the same device."
            )

        if pressure.dtype != velocity.dtype:
            raise ValueError(
                "Pressure and velocity must have the same dtype."
            )

        if pressure.dtype != source.dtype:
            raise ValueError(
                "Pressure and source must have the same dtype."
            )

        if torch.any(velocity <= 0):
            raise ValueError(
                "Velocity must be strictly positive."
            )

        # Require sufficient samples in all five field axes.
        # Tensor dimensions are [B, C, T, D, H, W].
        for dim in (2, 3, 4, 5):
            if pressure.shape[dim] < 3:
                raise ValueError(
                    f"Pressure dimension {dim} needs at least "
                    "three samples."
                )

        # ----------------------------------------------------
        # Temporal second derivative
        # ----------------------------------------------------
        p_tt = self._second_difference(
            pressure, self.dt, dim=2
        )

        # ----------------------------------------------------
        # Spatial second derivatives
        # Each derivative excludes its corresponding boundary.
        # Crop the other derivatives to a common interior.
        # ----------------------------------------------------
        p_xx = self._second_difference(
            pressure, self.dx, dim=5
        )[:, :, 1:-1, 1:-1, 1:-1, :]

        p_yy = self._second_difference(
            pressure, self.dy, dim=4
        )[:, :, 1:-1, 1:-1, :, 1:-1]

        p_zz = self._second_difference(
            pressure, self.dz, dim=3
        )[:, :, 1:-1, :, 1:-1, 1:-1]

        # p_tt already excludes the first and last time samples.
        # Crop all derivative fields to the common spatial interior.
        p_tt = p_tt[:, :, :, 1:-1, 1:-1, 1:-1]

        # Laplacian on the common interior.
        laplacian = (
            p_xx + p_yy + p_zz
        )

        # Crop velocity to the common spatial interior and
        # broadcast it across the retained time samples.
        velocity_interior = velocity[
            :, :, 1:-1, 1:-1, 1:-1
        ].unsqueeze(2)

        # The source must be evaluated at exactly the same
        # interior space-time locations as the derivatives.
        source_interior = source[
            :, :, 1:-1, 1:-1, 1:-1, 1:-1
        ]

        # ----------------------------------------------------
        # Acoustic wave-equation residual
        # ----------------------------------------------------
        residual = (
            p_tt / velocity_interior.square()
            - laplacian
            - source_interior
        )

        if not torch.isfinite(residual).all():
            raise ValueError(
                "Acoustic residual contains NaN or infinite values."
            )

        return residual

    def forward(
        self,
        pressure,
        velocity,
        source,
    ):
        """
        Compute the acoustic physics loss and diagnostics.

        Returns a dictionary compatible with a composite
        training objective once the trainer is adapted.
        """

        residual = self.acoustic_residual(
            pressure=pressure,
            velocity=velocity,
            source=source,
        )

        acoustic_loss = residual.square().mean()
        residual_rms = torch.sqrt(
            residual.square().mean()
        )

        if not torch.isfinite(acoustic_loss):
            raise ValueError(
                "Acoustic wave-equation loss is not finite."
            )

        return {
            "physics_loss": acoustic_loss,
            "acoustic_wave_equation_loss": acoustic_loss,
            "acoustic_residual_rms": residual_rms,
        }
