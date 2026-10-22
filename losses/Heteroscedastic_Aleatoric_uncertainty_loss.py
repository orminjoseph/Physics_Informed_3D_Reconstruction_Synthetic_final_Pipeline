"""
=========================================================
Heteroscedastic Aleatoric Uncertainty Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
This module implements the heteroscedastic Gaussian negative
log-likelihood used to learn aleatoric uncertainty during
seismic data reconstruction.

The neural network predicts the logarithm of the aleatoric
variance:

    s = log(sigma_a^2)

Therefore:

    sigma_a^2 = exp(s)

The heteroscedastic Gaussian negative log-likelihood is:

    L_aleatoric =
        0.5 * exp(-s) * (y_hat - y)^2
        + 0.5 * s

where:

    y_hat = reconstructed seismic amplitude
    y     = target seismic amplitude
    s     = predicted log variance

The constant Gaussian normalization term is omitted because
it does not affect optimization.

Aleatoric uncertainty
---------------------

Aleatoric uncertainty represents uncertainty associated with
the data or observation process, such as:

    - measurement noise
    - acquisition noise
    - unresolved observational variability
    - locally variable reconstruction difficulty

Epistemic uncertainty
---------------------

Epistemic uncertainty is NOT learned by this loss.

It is estimated separately during inference, for example
using Monte Carlo Dropout.

Predictive variance may subsequently be represented as:

    sigma_predictive^2 =
        sigma_aleatoric^2
        +
        sigma_epistemic^2

Input convention
----------------

Prediction, target, and log_variance must have the same
5-D tensor shape:

    [B, C, D, H, W]

where:

    B = batch size
    C = number of channels
    D = depth
    H = crossline
    W = inline

Numerical stabilization
-----------------------

The predicted log variance is clamped to the project-defined
range:

    LOG_VARIANCE_MIN
    LOG_VARIANCE_MAX

before exponentiation.

This prevents excessively large or small precision values
from destabilizing optimization.

Author: Ormin Joseph
=========================================================
"""

import math

import torch
import torch.nn as nn

from utils.config import (
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
)


class HeteroscedasticAleatoricUncertaintyLoss(
    nn.Module
):
    """
    Heteroscedastic Gaussian negative log-likelihood
    for learning aleatoric uncertainty.

    The neural network predicts:

        log_variance = log(sigma_a^2)

    and the loss is:

        L =
            0.5 * exp(-log_variance)
            * (prediction - target)^2
            +
            0.5 * log_variance

    Epistemic uncertainty is estimated separately during
    inference and is not included in this loss.
    """

    def __init__(
        self,
        log_variance_min=None,
        log_variance_max=None,
    ):
        """
        Initialize the heteroscedastic aleatoric
        uncertainty loss.

        Parameters
        ----------
        log_variance_min : float, optional
            Lower numerical bound for predicted log variance.

            If None, LOG_VARIANCE_MIN from config.py
            is used.

        log_variance_max : float, optional
            Upper numerical bound for predicted log variance.

            If None, LOG_VARIANCE_MAX from config.py
            is used.
        """

        super().__init__()

        # =================================================
        # USE CENTRALIZED CONFIGURATION BY DEFAULT
        # =================================================

        if log_variance_min is None:
            log_variance_min = LOG_VARIANCE_MIN

        if log_variance_max is None:
            log_variance_max = LOG_VARIANCE_MAX

        # =================================================
        # VALIDATE LOWER BOUND
        # =================================================

        self._validate_finite_scalar(
            log_variance_min,
            "log_variance_min",
        )

        # =================================================
        # VALIDATE UPPER BOUND
        # =================================================

        self._validate_finite_scalar(
            log_variance_max,
            "log_variance_max",
        )

        # =================================================
        # VALIDATE ORDERING
        # =================================================

        if (
            float(log_variance_min)
            >=
            float(log_variance_max)
        ):
            raise ValueError(
                "log_variance_min must be smaller than "
                "log_variance_max."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.log_variance_min = float(
            log_variance_min
        )

        self.log_variance_max = float(
            log_variance_max
        )

    # =====================================================
    # SCALAR VALIDATION
    # =====================================================

    @staticmethod
    def _validate_finite_scalar(
        value,
        name,
    ):
        """
        Validate that a configuration value is a finite
        numeric scalar.
        """

        # -------------------------------------------------
        # bool is technically an int in Python, therefore
        # it must be explicitly rejected.
        # -------------------------------------------------

        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a numeric scalar."
            )

        # -------------------------------------------------
        # Accept ordinary Python numeric values.
        # -------------------------------------------------

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a numeric scalar."
            )

        # -------------------------------------------------
        # Reject NaN and Inf.
        # -------------------------------------------------

        if not math.isfinite(
            float(value)
        ):
            raise ValueError(
                f"{name} must be finite."
            )

    # =====================================================
    # TENSOR VALIDATION
    # =====================================================

    @staticmethod
    def _validate_tensor(
        tensor,
        name,
    ):
        """
        Validate a seismic tensor.

        Required shape:

            [B, C, D, H, W]
        """

        # -------------------------------------------------
        # Tensor type
        # -------------------------------------------------

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        # -------------------------------------------------
        # Floating-point requirement
        #
        # The loss performs subtraction, squaring,
        # exponentiation, and gradient-based optimization.
        # -------------------------------------------------

        if not torch.is_floating_point(
            tensor
        ):
            raise TypeError(
                f"{name} must be a floating-point tensor."
            )

        # -------------------------------------------------
        # Five-dimensional seismic-volume convention
        # -------------------------------------------------

        if tensor.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B, C, D, H, W]. "
                f"Received shape: "
                f"{tuple(tensor.shape)}."
            )

        # -------------------------------------------------
        # Every dimension must be positive.
        # -------------------------------------------------

        if any(
            int(size) < 1
            for size in tensor.shape
        ):
            raise ValueError(
                f"{name} contains an invalid "
                "zero-sized dimension."
            )

        # -------------------------------------------------
        # Numerical validation
        # -------------------------------------------------

        if not torch.isfinite(
            tensor
        ).all():
            raise FloatingPointError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the heteroscedastic Gaussian negative
        log-likelihood.

        Parameters
        ----------
        prediction : torch.Tensor
            Reconstructed seismic volume.

            Shape:

                [B, C, D, H, W]

        target : torch.Tensor
            Ground-truth seismic volume.

            Shape:

                [B, C, D, H, W]

        log_variance : torch.Tensor
            Predicted logarithm of aleatoric variance.

            Shape:

                [B, C, D, H, W]

        Returns
        -------
        torch.Tensor
            Scalar aleatoric uncertainty loss.
        """

        # =================================================
        # 1. VALIDATE INPUT TENSORS
        # =================================================

        self._validate_tensor(
            prediction,
            "prediction",
        )

        self._validate_tensor(
            target,
            "target",
        )

        self._validate_tensor(
            log_variance,
            "log_variance",
        )

        # =================================================
        # 2. VALIDATE SHAPES
        # =================================================

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have "
                "identical shapes. "
                f"Received prediction "
                f"{tuple(prediction.shape)} and "
                f"target "
                f"{tuple(target.shape)}."
            )

        if prediction.shape != log_variance.shape:
            raise ValueError(
                "prediction and log_variance must have "
                "identical shapes. "
                f"Received prediction "
                f"{tuple(prediction.shape)} and "
                f"log_variance "
                f"{tuple(log_variance.shape)}."
            )

        # =================================================
        # 3. VALIDATE DTYPE CONSISTENCY
        # =================================================

        if prediction.dtype != target.dtype:
            raise TypeError(
                "prediction and target must have "
                f"the same dtype. Received "
                f"{prediction.dtype} and "
                f"{target.dtype}."
            )

        if prediction.dtype != log_variance.dtype:
            raise TypeError(
                "prediction and log_variance must have "
                f"the same dtype. Received "
                f"{prediction.dtype} and "
                f"{log_variance.dtype}."
            )

        # =================================================
        # 4. COMPUTE RECONSTRUCTION ERROR
        # =================================================

        residual = (
            prediction
            -
            target
        )

        # -------------------------------------------------
        # Squared reconstruction error:
        #
        #     (y_hat - y)^2
        # -------------------------------------------------

        squared_error = residual.pow(2)

        # -------------------------------------------------
        # Validate the squared error.
        # -------------------------------------------------

        if not torch.isfinite(
            squared_error
        ).all():
            raise FloatingPointError(
                "Squared reconstruction error contains "
                "NaN or Inf values."
            )

        # =================================================
        # 5. CLAMP PREDICTED LOG VARIANCE
        # =================================================
        #
        # The neural network may produce values outside
        # the numerically safe operating range.
        #
        # Clamping is performed before exp(-s).
        # =================================================

        bounded_log_variance = torch.clamp(
            log_variance,
            min=self.log_variance_min,
            max=self.log_variance_max,
        )

        # =================================================
        # 6. COMPUTE PRECISION
        # =================================================
        #
        # Variance:
        #
        #     sigma_a^2 = exp(s)
        #
        # Precision:
        #
        #     1 / sigma_a^2 = exp(-s)
        # =================================================

        precision = torch.exp(
            -bounded_log_variance
        )

        # -------------------------------------------------
        # Validate precision.
        # -------------------------------------------------

        if not torch.isfinite(
            precision
        ).all():
            raise FloatingPointError(
                "Aleatoric precision contains "
                "NaN or Inf values."
            )

        # =================================================
        # 7. HETEROSCEDASTIC GAUSSIAN NLL
        # =================================================
        #
        #     L =
        #
        #       0.5 * exp(-s)
        #       * (prediction - target)^2
        #
        #       + 0.5 * s
        #
        # The constant Gaussian normalization term is
        # intentionally omitted because it does not affect
        # optimization.
        # =================================================

        loss = (
            0.5
            *
            precision
            *
            squared_error
            +
            0.5
            *
            bounded_log_variance
        )

        # =================================================
        # 8. REDUCE TO SCALAR
        # =================================================

        loss = loss.mean()

        # =================================================
        # 9. FINAL NUMERICAL VALIDATION
        # =================================================

        if not torch.isfinite(
            loss
        ):
            raise FloatingPointError(
                "Heteroscedastic aleatoric uncertainty "
                "loss became NaN or Inf."
            )

        # =================================================
        # 10. RETURN LOSS
        # =================================================

        return loss


# =========================================================
# BACKWARD-COMPATIBILITY ALIAS
# =========================================================
#
# Existing code may still import:
#
#     UncertaintyLoss
#
# Keeping this alias avoids unnecessary breakage while the
# project transitions to the more explicit class name.
#
# The actual implementation remains:
#
#     HeteroscedasticAleatoricUncertaintyLoss
# =========================================================

UncertaintyLoss = (
    HeteroscedasticAleatoricUncertaintyLoss
)