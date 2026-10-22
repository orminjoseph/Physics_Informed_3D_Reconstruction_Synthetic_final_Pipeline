"""
====================================================================
Total Loss
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Composite objective
-------------------
    L_total = w_mae * L_MAE
            + w_physics * L_acoustic
            + w_uncertainty * L_aleatoric
            + w_ssim * L_SSIM

The physics term is the acoustic wave-equation residual supplied by
AcousticWaveEquationLoss. Eikonal, source-condition travel-time, and
travel-time-supervision terms are deliberately not used in this module.

Important physical distinction
------------------------------
The acoustic loss requires a time-dependent pressure wavefield
p(t, z, y, x), not merely a static seismic reconstruction cube.
The static reconstruction prediction/target remain [B, C, D, H, W].
The pressure wavefield and source are expected by the acoustic loss
to be [B, 1, T, D, H, W], and velocity [B, 1, D, H, W], following
the AcousticWaveEquationLoss implementation.

The pressure field used by the physics term must be physically
meaningful and consistently sampled. Do not pass a static reconstruction
cube as pressure_wavefield.

Author: Ormin Joseph
====================================================================
"""

import math

import torch
import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import AcousticWaveEquationLoss
from losses.ssim_loss import SSIMLoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    HeteroscedasticAleatoricUncertaintyLoss,
)

from utils.config import LOSS_WEIGHTS, SEISMIC_DATA_RANGE


class TotalLoss(nn.Module):
    """
    Composite reconstruction, acoustic-physics, and uncertainty loss.

    Parameters
    ----------
    loss_weights : dict, optional
        Required keys: "mae", "physics", "uncertainty", and "ssim".
        Defaults to LOSS_WEIGHTS from utils.config.

    data_range : float, optional
        Dynamic range passed to SSIMLoss. Defaults to SEISMIC_DATA_RANGE.

    acoustic_loss : AcousticWaveEquationLoss, optional
        An already-configured acoustic wave-equation loss instance.
        Supplying the instance avoids duplicating dt/dx/dy/dz settings
        that belong to the physics-loss module.

    log_variance_min, log_variance_max : float, optional
        Optional limits passed to the uncertainty-loss module.
    """

    def __init__(
        self,
        loss_weights=None,
        data_range=None,
        acoustic_loss=None,
        log_variance_min=None,
        log_variance_max=None,
    ):
        super().__init__()

        if loss_weights is None:
            loss_weights = LOSS_WEIGHTS
        if data_range is None:
            data_range = SEISMIC_DATA_RANGE

        self._validate_weight_dictionary(
            loss_weights,
            required_keys=("mae", "physics", "uncertainty", "ssim"),
            name="loss_weights",
        )
        self._validate_positive_finite(data_range, "data_range")

        self.loss_weights = {
            key: float(loss_weights[key])
            for key in ("mae", "physics", "uncertainty", "ssim")
        }
        self.data_range = float(data_range)

        if acoustic_loss is not None and not isinstance(
            acoustic_loss, AcousticWaveEquationLoss
        ):
            raise TypeError(
                "acoustic_loss must be an AcousticWaveEquationLoss instance."
            )
        self.acoustic_loss = acoustic_loss

        if log_variance_min is not None:
            self._validate_finite_number(
                log_variance_min, "log_variance_min"
            )
        if log_variance_max is not None:
            self._validate_finite_number(
                log_variance_max, "log_variance_max"
            )
        if (
            log_variance_min is not None
            and log_variance_max is not None
            and float(log_variance_min) >= float(log_variance_max)
        ):
            raise ValueError(
                "log_variance_min must be smaller than log_variance_max."
            )

        uncertainty_kwargs = {}
        if log_variance_min is not None:
            uncertainty_kwargs["log_variance_min"] = float(log_variance_min)
        if log_variance_max is not None:
            uncertainty_kwargs["log_variance_max"] = float(log_variance_max)

        self.mae_loss = MAELoss()
        self.ssim_loss = SSIMLoss(data_range=self.data_range)
        self.uncertainty_loss = HeteroscedasticAleatoricUncertaintyLoss(
            **uncertainty_kwargs
        )

    @staticmethod
    def _validate_finite_number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a numeric value.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    @classmethod
    def _validate_positive_finite(cls, value, name):
        cls._validate_finite_number(value, name)
        if float(value) <= 0:
            raise ValueError(f"{name} must be greater than zero.")

    @classmethod
    def _validate_weight_dictionary(cls, weights, required_keys, name):
        if not isinstance(weights, dict):
            raise TypeError(f"{name} must be a dictionary.")

        missing = [key for key in required_keys if key not in weights]
        if missing:
            raise KeyError(f"{name} is missing required keys: {missing}")

        for key in required_keys:
            value = weights[key]
            cls._validate_finite_number(value, f"{name}['{key}']")
            if float(value) < 0:
                raise ValueError(f"{name}['{key}'] cannot be negative.")

    @staticmethod
    def _validate_floating_tensor(tensor, name, ndim=None):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must be a floating-point tensor.")
        if ndim is not None and tensor.ndim != ndim:
            raise ValueError(
                f"{name} must be {ndim}D; received shape {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf values.")

    def forward(
        self,
        prediction,
        target,
        pressure_wavefield=None,
        velocity=None,
        source=None,
        log_variance=None,
    ):
        """
        Compute the composite loss.

        Parameters
        ----------
        prediction, target : torch.Tensor
            Reconstructed and ground-truth seismic cubes, identical
            [B, C, D, H, W] shapes.

        pressure_wavefield : torch.Tensor, optional
            Time-dependent pressure field expected by the acoustic loss,
            conventionally [B, 1, T, D, H, W].

        velocity : torch.Tensor, optional
            Acoustic velocity model expected by AcousticWaveEquationLoss,
            conventionally [B, 1, D, H, W].

        source : torch.Tensor, optional
            Source term with the same shape as pressure_wavefield.

        log_variance : torch.Tensor, optional
            Predicted log aleatoric variance with the same shape as
            prediction and target.

        Returns
        -------
        dict
            Scalar loss tensors, including acoustic physics diagnostics.
        """
        self._validate_floating_tensor(prediction, "prediction", ndim=5)
        self._validate_floating_tensor(target, "target", ndim=5)

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes; "
                f"received {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.dtype != target.dtype:
            raise TypeError("prediction and target must have the same dtype.")

        if log_variance is not None:
            self._validate_floating_tensor(
                log_variance, "log_variance", ndim=5
            )
            if log_variance.shape != prediction.shape:
                raise ValueError(
                    "log_variance must have the same shape as prediction."
                )

        mae_value = self.mae_loss(prediction, target)
        ssim_value = self.ssim_loss(prediction, target)

        # Aleatoric uncertainty is optional; if the model does not produce
        # log variance, its contribution is exactly zero.
        if log_variance is not None:
            uncertainty_value = self.uncertainty_loss(
                prediction=prediction,
                target=target,
                log_variance=log_variance,
            )
        else:
            uncertainty_value = prediction.new_zeros(())

        # Acoustic physics is evaluated only when all physical inputs are
        # supplied. A missing physical input is never silently replaced by
        # the reconstructed static seismic cube.
        physics_inputs = (
            pressure_wavefield is not None,
            velocity is not None,
            source is not None,
        )
        if any(physics_inputs) and not all(physics_inputs):
            raise ValueError(
                "Provide pressure_wavefield, velocity, and source together "
                "to evaluate the acoustic wave-equation loss."
            )

        if all(physics_inputs):
            if self.acoustic_loss is None:
                raise RuntimeError(
                    "Acoustic inputs were supplied, but TotalLoss has no "
                    "configured AcousticWaveEquationLoss. Pass an instance "
                    "using TotalLoss(acoustic_loss=...)."
                )

            physics_output = self.acoustic_loss(
                pressure=pressure_wavefield,
                velocity=velocity,
                source=source,
            )
            if not isinstance(physics_output, dict):
                raise TypeError(
                    "AcousticWaveEquationLoss must return a dictionary."
                )
            if "physics_loss" not in physics_output:
                raise KeyError(
                    "AcousticWaveEquationLoss output must contain "
                    "'physics_loss'."
                )

            physics_value = physics_output["physics_loss"]
            acoustic_residual_rms = physics_output.get(
                "acoustic_residual_rms",
                physics_value.new_zeros(()),
            )
            acoustic_wave_equation_value = physics_output.get(
                "acoustic_wave_equation_loss",
                physics_value,
            )
        else:
            physics_value = prediction.new_zeros(())
            acoustic_residual_rms = prediction.new_zeros(())
            acoustic_wave_equation_value = prediction.new_zeros(())

        for name, value in (
            ("physics_loss", physics_value),
            ("acoustic_residual_rms", acoustic_residual_rms),
            ("acoustic_wave_equation_loss", acoustic_wave_equation_value),
            ("mae_loss", mae_value),
            ("uncertainty_loss", uncertainty_value),
            ("ssim_loss", ssim_value),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or Inf values.")

        weighted_mae = self.loss_weights["mae"] * mae_value
        weighted_physics = self.loss_weights["physics"] * physics_value
        weighted_uncertainty = (
            self.loss_weights["uncertainty"] * uncertainty_value
        )
        weighted_ssim = self.loss_weights["ssim"] * ssim_value

        total_loss = (
            weighted_mae
            + weighted_physics
            + weighted_uncertainty
            + weighted_ssim
        )

        if not torch.isfinite(total_loss).all():
            raise ValueError("total_loss contains NaN or Inf values.")

        return {
            "total_loss": total_loss,
            "mae_loss": mae_value,
            "physics_loss": physics_value,
            "acoustic_wave_equation_loss": acoustic_wave_equation_value,
            "acoustic_residual_rms": acoustic_residual_rms,
            "uncertainty_loss": uncertainty_value,
            "ssim_loss": ssim_value,
            "weighted_mae_loss": weighted_mae,
            "weighted_physics_loss": weighted_physics,
            "weighted_uncertainty_loss": weighted_uncertainty,
            "weighted_ssim_loss": weighted_ssim,
        }
