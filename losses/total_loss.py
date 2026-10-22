"""
=========================================================
Total Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Composite objective:

    Total Loss =
        w_mae * MAE
        + w_physics * Physics Loss
        + w_uncertainty * Heteroscedastic Aleatoric
          Uncertainty Loss
        + w_ssim * SSIM Loss

The physics loss may contain:

    - Eikonal equation constraint
    - Source-condition constraint
    - Travel-time supervision

The reconstruction loss (MAE) remains the primary
reconstruction objective.

Input convention:

    [B, C, D, H, W]

where:

    B = batch
    C = seismic channels
    D = depth
    H = crossline
    W = inline

Author: Ormin Joseph
=========================================================
"""

import math

import torch
import torch.nn as nn

from losses.mae_loss import MAELoss
from losses.physics_loss import PhysicsLoss
from losses.ssim_loss import SSIMLoss
from losses.Heteroscedastic_Aleatoric_uncertainty_loss import (
    HeteroscedasticAleatoricUncertaintyLoss,
)

from utils.config import (
    LOSS_WEIGHTS,
    PHYSICS_LOSS_WEIGHTS,
    SEISMIC_DATA_RANGE,
)


class TotalLoss(nn.Module):
    """
    Composite loss for the physics-informed 3D seismic
    reconstruction framework.

    Parameters
    ----------
    loss_weights : dict, optional
        Weights for the major loss components:

            {
                "mae": ...,
                "physics": ...,
                "uncertainty": ...,
                "ssim": ...
            }

    physics_loss_weights : dict, optional
        Weights for physics sub-components:

            {
                "eikonal": ...,
                "source": ...,
                "travel_time": ...
            }

    data_range : float, optional
        Dynamic range of normalized seismic amplitudes.
        This is passed to the SSIM loss.

    log_variance_min : float, optional
        Lower bound for predicted log variance.

    log_variance_max : float, optional
        Upper bound for predicted log variance.
    """

    def __init__(
        self,
        loss_weights=None,
        physics_loss_weights=None,
        data_range=None,
        log_variance_min=None,
        log_variance_max=None,
    ):
        super().__init__()

        # -------------------------------------------------
        # Use configuration values when explicit values
        # are not supplied.
        # -------------------------------------------------
        if loss_weights is None:
            loss_weights = LOSS_WEIGHTS

        if physics_loss_weights is None:
            physics_loss_weights = PHYSICS_LOSS_WEIGHTS

        if data_range is None:
            data_range = SEISMIC_DATA_RANGE

        # -------------------------------------------------
        # Validate the loss-weight dictionaries.
        # -------------------------------------------------
        self._validate_weight_dictionary(
            loss_weights,
            required_keys=("mae", "physics", "uncertainty", "ssim"),
            name="loss_weights",
        )

        self._validate_weight_dictionary(
            physics_loss_weights,
            required_keys=("eikonal", "source", "travel_time"),
            name="physics_loss_weights",
        )

        # -------------------------------------------------
        # Validate seismic data range.
        # -------------------------------------------------
        if isinstance(data_range, bool):
            raise TypeError(
                "data_range must be a positive finite number."
            )

        if not isinstance(data_range, (int, float)):
            raise TypeError(
                "data_range must be a positive finite number."
            )

        if not math.isfinite(float(data_range)):
            raise ValueError(
                "data_range must be finite."
            )

        if float(data_range) <= 0.0:
            raise ValueError(
                "data_range must be greater than zero."
            )

        # -------------------------------------------------
        # Store configuration.
        # -------------------------------------------------
        self.loss_weights = {
            key: float(value)
            for key, value in loss_weights.items()
        }

        self.physics_loss_weights = {
            key: float(value)
            for key, value in physics_loss_weights.items()
        }

        self.data_range = float(data_range)

        # -------------------------------------------------
        # Validate optional log-variance limits.
        #
        # These are passed to the uncertainty loss so that
        # the uncertainty branch uses the same numerical
        # limits as the project configuration.
        # -------------------------------------------------
        if log_variance_min is not None:
            if isinstance(log_variance_min, bool):
                raise TypeError(
                    "log_variance_min must be numeric."
                )

            if not isinstance(log_variance_min, (int, float)):
                raise TypeError(
                    "log_variance_min must be numeric."
                )

            if not math.isfinite(float(log_variance_min)):
                raise ValueError(
                    "log_variance_min must be finite."
                )

        if log_variance_max is not None:
            if isinstance(log_variance_max, bool):
                raise TypeError(
                    "log_variance_max must be numeric."
                )

            if not isinstance(log_variance_max, (int, float)):
                raise TypeError(
                    "log_variance_max must be numeric."
                )

            if not math.isfinite(float(log_variance_max)):
                raise ValueError(
                    "log_variance_max must be finite."
                )

        if (
            log_variance_min is not None
            and log_variance_max is not None
            and float(log_variance_min) >= float(log_variance_max)
        ):
            raise ValueError(
                "log_variance_min must be smaller than "
                "log_variance_max."
            )

        # -------------------------------------------------
        # Store uncertainty limits when supplied.
        # They are intentionally optional here because the
        # uncertainty-loss module itself obtains the
        # project defaults from config.py.
        # -------------------------------------------------
        self.log_variance_min = log_variance_min
        self.log_variance_max = log_variance_max

        # -------------------------------------------------
        # Instantiate individual loss functions.
        # -------------------------------------------------
        self.mae_loss = MAELoss()

        self.physics_loss = PhysicsLoss(
            eikonal_weight=self.physics_loss_weights["eikonal"],
            source_weight=self.physics_loss_weights["source"],
            travel_time_weight=self.physics_loss_weights["travel_time"],
        )

        self.ssim_loss = SSIMLoss(
            data_range=self.data_range
        )

        # -------------------------------------------------
        # Instantiate the heteroscedastic aleatoric
        # uncertainty loss.
        #
        # Explicit bounds are supplied only when the caller
        # provides them. Otherwise the uncertainty-loss
        # module uses the centralized project configuration.
        # -------------------------------------------------
        uncertainty_kwargs = {}

        if log_variance_min is not None:
            uncertainty_kwargs["log_variance_min"] = (
                float(log_variance_min)
            )

        if log_variance_max is not None:
            uncertainty_kwargs["log_variance_max"] = (
                float(log_variance_max)
            )

        self.uncertainty_loss = (
            HeteroscedasticAleatoricUncertaintyLoss(
                **uncertainty_kwargs
            )
        )

    # =====================================================
    # Validation helpers
    # =====================================================

    @staticmethod
    def _validate_weight_dictionary(
        weights,
        required_keys,
        name,
    ):
        """
        Validate a loss-weight dictionary.
        """

        if not isinstance(weights, dict):
            raise TypeError(
                f"{name} must be a dictionary."
            )

        missing_keys = [
            key for key in required_keys
            if key not in weights
        ]

        if missing_keys:
            raise KeyError(
                f"{name} is missing required keys: "
                f"{missing_keys}"
            )

        for key in required_keys:

            value = weights[key]

            if isinstance(value, bool):
                raise TypeError(
                    f"{name}['{key}'] must be numeric."
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name}['{key}'] must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{name}['{key}'] must be finite."
                )

            if float(value) < 0.0:
                raise ValueError(
                    f"{name}['{key}'] cannot be negative."
                )

    @staticmethod
    def _validate_tensor(
        tensor,
        name,
    ):
        """
        Validate that an input is a finite PyTorch tensor.
        """

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"{name} must be a floating-point tensor."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    @staticmethod
    def _validate_5d_tensor(
        tensor,
        name,
    ):
        """
        Validate a tensor using the project's canonical
        seismic tensor convention:

            [B, C, D, H, W]
        """

        TotalLoss._validate_tensor(
            tensor=tensor,
            name=name,
        )

        if tensor.ndim != 5:
            raise ValueError(
                f"{name} must have shape [B, C, D, H, W]. "
                f"Received shape {tuple(tensor.shape)}."
            )

    # =====================================================
    # Forward pass
    # =====================================================

    def forward(
        self,
        prediction,
        target,
        velocity=None,
        source_indices=None,
        travel_time=None,
        travel_time_target=None,
        log_variance=None,
    ):
        """
        Calculate the complete composite loss.

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.

        target : torch.Tensor
            Ground-truth seismic volume.

        velocity : torch.Tensor, optional
            Velocity model.

        source_indices : torch.Tensor, optional
            Source coordinates in:

                [depth, crossline, inline]

            with shape [B, 3].

        travel_time : torch.Tensor, optional
            Predicted travel-time field.

        travel_time_target : torch.Tensor, optional
            Ground-truth travel-time field.

        log_variance : torch.Tensor, optional
            Predicted logarithmic aleatoric variance.

        Returns
        -------
        dict
            Dictionary containing:

            - total_loss
            - mae_loss
            - physics_loss
            - uncertainty_loss
            - ssim_loss
            - weighted_mae_loss
            - weighted_physics_loss
            - weighted_uncertainty_loss
            - weighted_ssim_loss
            - eikonal_loss
            - source_loss
            - travel_time_loss
        """

        # -------------------------------------------------
        # Validate reconstruction tensors.
        # -------------------------------------------------
        self._validate_5d_tensor(
            prediction,
            "prediction",
        )

        self._validate_5d_tensor(
            target,
            "target",
        )

        # -------------------------------------------------
        # Prediction and target must have identical shapes.
        # -------------------------------------------------
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical "
                f"shapes. Received "
                f"{tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        # -------------------------------------------------
        # Prediction and target must use the same numerical
        # dtype.
        # -------------------------------------------------
        if prediction.dtype != target.dtype:
            raise TypeError(
                "prediction and target must have the same "
                f"dtype. Received {prediction.dtype} and "
                f"{target.dtype}."
            )

        # -------------------------------------------------
        # Validate optional velocity field.
        # -------------------------------------------------
        if velocity is not None:

            self._validate_5d_tensor(
                velocity,
                "velocity",
            )

            if velocity.shape != target.shape:
                raise ValueError(
                    "velocity and target must have identical "
                    f"shapes. Received "
                    f"{tuple(velocity.shape)} and "
                    f"{tuple(target.shape)}."
                )

        # -------------------------------------------------
        # Validate optional travel-time prediction.
        # -------------------------------------------------
        if travel_time is not None:

            self._validate_5d_tensor(
                travel_time,
                "travel_time",
            )

            if travel_time.shape != target.shape:
                raise ValueError(
                    "travel_time and target must have "
                    "identical shapes. Received "
                    f"{tuple(travel_time.shape)} and "
                    f"{tuple(target.shape)}."
                )

        # -------------------------------------------------
        # Validate optional travel-time target.
        # -------------------------------------------------
        if travel_time_target is not None:

            self._validate_5d_tensor(
                travel_time_target,
                "travel_time_target",
            )

            if travel_time_target.shape != target.shape:
                raise ValueError(
                    "travel_time_target and target must have "
                    "identical shapes. Received "
                    f"{tuple(travel_time_target.shape)} and "
                    f"{tuple(target.shape)}."
                )

        # -------------------------------------------------
        # Validate optional logarithmic variance.
        # -------------------------------------------------
        if log_variance is not None:

            self._validate_5d_tensor(
                log_variance,
                "log_variance",
            )

            if log_variance.shape != target.shape:
                raise ValueError(
                    "log_variance and target must have "
                    "identical shapes. Received "
                    f"{tuple(log_variance.shape)} and "
                    f"{tuple(target.shape)}."
                )

        # -------------------------------------------------
        # Validate source indices.
        #
        # Canonical ordering:
        #
        #     [depth, crossline, inline]
        #
        # Expected shape:
        #
        #     [B, 3]
        # -------------------------------------------------
        if source_indices is not None:

            if not isinstance(
                source_indices,
                torch.Tensor,
            ):
                raise TypeError(
                    "source_indices must be a "
                    "torch.Tensor."
                )

            if source_indices.ndim != 2:
                raise ValueError(
                    "source_indices must have shape [B, 3]. "
                    f"Received "
                    f"{tuple(source_indices.shape)}."
                )

            if source_indices.shape[0] != prediction.shape[0]:
                raise ValueError(
                    "The batch dimension of source_indices "
                    "must match prediction."
                )

            if source_indices.shape[1] != 3:
                raise ValueError(
                    "source_indices must contain exactly "
                    "three coordinates in the order "
                    "[depth, crossline, inline]."
                )

            if not torch.isfinite(
                source_indices.to(torch.float32)
            ).all():
                raise ValueError(
                    "source_indices contains NaN or Inf."
                )

        # =================================================
        # 1. MAE reconstruction loss
        # =================================================

        mae_value = self.mae_loss(
            prediction,
            target,
        )

        # =================================================
        # 2. Physics-informed loss
        # =================================================
        #
        # PhysicsLoss is responsible for evaluating the
        # configured physics terms.
        #
        # We do NOT arbitrarily normalize the physics loss
        # here. Its numerical scale must remain traceable to
        # the Eikonal equation, velocity units, grid spacing,
        # and any travel-time representation used by the
        # network.
        # =================================================

        if velocity is not None:

            physics_output = self.physics_loss(
                velocity=velocity,
                travel_time=travel_time,
                source_indices=source_indices,
                travel_time_target=travel_time_target,
            )

            # -------------------------------------------------
            # PhysicsLoss is expected to return either:
            #
            #     total, eikonal, source, travel_time
            #
            # or a dictionary containing those quantities.
            #
            # The final implementation below accepts the
            # dictionary form used by the framework.
            # -------------------------------------------------
            if isinstance(physics_output, dict):

                physics_value = physics_output[
                    "physics_loss"
                ]

                eikonal_value = physics_output.get(
                    "eikonal_loss",
                    torch.zeros_like(physics_value),
                )

                source_value = physics_output.get(
                    "source_loss",
                    torch.zeros_like(physics_value),
                )

                travel_time_value = physics_output.get(
                    "travel_time_loss",
                    torch.zeros_like(physics_value),
                )

            elif isinstance(physics_output, tuple):

                if len(physics_output) != 4:
                    raise ValueError(
                        "PhysicsLoss tuple output must contain "
                        "exactly four values: "
                        "(physics_loss, eikonal_loss, "
                        "source_loss, travel_time_loss)."
                    )

                (
                    physics_value,
                    eikonal_value,
                    source_value,
                    travel_time_value,
                ) = physics_output

            else:

                raise TypeError(
                    "PhysicsLoss must return either a "
                    "dictionary or a four-element tuple."
                )

        else:

            # -------------------------------------------------
            # Physics cannot be evaluated without velocity.
            # A zero physics contribution is therefore used.
            # -------------------------------------------------
            zero = prediction.new_zeros(())

            physics_value = zero
            eikonal_value = zero
            source_value = zero
            travel_time_value = zero

        # =================================================
        # 3. Heteroscedastic aleatoric uncertainty loss
        # =================================================

        if log_variance is not None:

            uncertainty_value = self.uncertainty_loss(
                prediction=prediction,
                target=target,
                log_variance=log_variance,
            )

        else:

            uncertainty_value = prediction.new_zeros(())

        # =================================================
        # 4. SSIM loss
        # =================================================

        ssim_value = self.ssim_loss(
            prediction,
            target,
        )

        # =================================================
        # 5. Apply configured top-level weights
        # =================================================

        weighted_mae = (
            self.loss_weights["mae"]
            * mae_value
        )

        weighted_physics = (
            self.loss_weights["physics"]
            * physics_value
        )

        weighted_uncertainty = (
            self.loss_weights["uncertainty"]
            * uncertainty_value
        )

        weighted_ssim = (
            self.loss_weights["ssim"]
            * ssim_value
        )

        # =================================================
        # 6. Composite objective
        # =================================================

        total_loss = (
            weighted_mae
            + weighted_physics
            + weighted_uncertainty
            + weighted_ssim
        )

        # -------------------------------------------------
        # Final numerical validation.
        # -------------------------------------------------
        loss_components = {
            "total_loss": total_loss,
            "mae_loss": mae_value,
            "physics_loss": physics_value,
            "uncertainty_loss": uncertainty_value,
            "ssim_loss": ssim_value,
            "weighted_mae_loss": weighted_mae,
            "weighted_physics_loss": weighted_physics,
            "weighted_uncertainty_loss": weighted_uncertainty,
            "weighted_ssim_loss": weighted_ssim,
            "eikonal_loss": eikonal_value,
            "source_loss": source_value,
            "travel_time_loss": travel_time_value,
        }

        for name, value in loss_components.items():

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

            if not torch.isfinite(value).all():
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

        # -------------------------------------------------
        # Return all components for:
        #
        # - training diagnostics
        # - TensorBoard
        # - reports
        # - ablation studies
        # - physics auditing
        # - uncertainty auditing
        # -------------------------------------------------
        return loss_components