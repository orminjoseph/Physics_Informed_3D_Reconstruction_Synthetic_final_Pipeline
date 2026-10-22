"""
======================================================================
f-x Prediction Controlled-Matrix Baseline
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Classical seismic interpolation baseline based on prediction-error
filtering in the frequency-space (f-x) domain.

This file contains BOTH:

    1. The f-x prediction reconstruction algorithm.
    2. The controlled-matrix experimental driver.

Controlled Experimental Design
------------------------------
The full factorial matrix contains:

    6 geological settings
    x 5 missing-data mechanisms
    x 5 missing rates
    x 5 random seeds

Therefore:

    6 x 5 x 5 x 5 = 750 cases

The number of cases executed can be controlled through:

    CONTROLLED_MATRIX_CASE_LIMIT

from:

    utils/config.py

For smoke testing:

    CONTROLLED_MATRIX_CASE_LIMIT = 10

For the final experiment:

    CONTROLLED_MATRIX_CASE_LIMIT = None

Tensor convention
-----------------
Input seismic cube:

    [C, D, H, W]

where:

    C = seismic channel
    D = temporal/depth dimension
    H = spatial dimension 1
    W = spatial dimension 2

Mask convention
---------------
    1 = observed
    0 = missing

Important methodological constraint
-----------------------------------
The f-x reconstruction algorithm receives ONLY:

    corrupted_cube
    mask

Ground-truth data are NEVER passed to the reconstruction algorithm.

Ground truth is used only AFTER reconstruction for evaluation metrics.

Observed-data consistency
-------------------------
All originally observed samples are restored exactly after every
iteration and once again before the final output is returned.

Evaluation metrics
------------------
The canonical metrics implementation is used:

    metrics/reconstruction_metrics.py

Metrics include:

    MAE
    MSE
    RMSE
    Relative L2
    Relative Error
    PSNR
    SNR
    SSIM
    Missing-region MAE
    Missing-region RMSE
    Observed-region MAE
    Observed-region RMSE

Author: Ormin Joseph
======================================================================
"""

from __future__ import annotations

import csv
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

from utils.config import (
    BASELINE_CUBE_SIZE,
    BASELINE_MISSING_RATE,
    BASELINE_NUM_SAMPLES,
    BASELINE_SEED,
    CONTROLLED_MATRIX_CASE_LIMIT,
    OBSERVED_PRESERVATION_TOLERANCE,
    REPORT_DIR,
)


# ======================================================================
# CANONICAL METRICS
# ======================================================================

from metrics.reconstruction_metrics import (
    calculate_reconstruction_metrics,
)


# ======================================================================
# DATASET
# ======================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset,
)


# ======================================================================
# F-X CONFIGURATION
# ======================================================================

# ----------------------------------------------------------------------
# These are the default algorithmic parameters.
#
# The controlled experiment uses these parameters consistently for
# every experimental case.
#
# If dedicated f-x configuration constants already exist in
# utils/config.py, those values should be imported here instead.
# ----------------------------------------------------------------------

DEFAULT_PREDICTION_ORDER = 4

DEFAULT_ITERATIONS = 2

DEFAULT_MIN_TRACE_OBSERVED_FRACTION = 0.80

DEFAULT_CONVERGENCE_TOLERANCE = 1.0e-5


# ======================================================================
# CONTROLLED EXPERIMENTAL FACTORS
# ======================================================================

# ----------------------------------------------------------------------
# Geological settings.
# ----------------------------------------------------------------------

GEOLOGICAL_MODES = (
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
)


# ----------------------------------------------------------------------
# Missing-data mechanisms.
# ----------------------------------------------------------------------

MASK_MODES = (
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
)


# ----------------------------------------------------------------------
# Missing-data rates.
# ----------------------------------------------------------------------

MISSING_RATES = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)


# ----------------------------------------------------------------------
# Random seeds.
# ----------------------------------------------------------------------

RANDOM_SEEDS = (
    42,
    43,
    44,
    45,
    46,
)


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def _validate_inputs(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and standardize the seismic cube and observation mask.

    Parameters
    ----------
    corrupted_cube:
        Seismic cube with shape (C, D, H, W).

    mask:
        Binary observation mask with shape (C, D, H, W).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Validated floating-point tensors.
    """

    # --------------------------------------------------------------
    # Convert NumPy arrays or array-like objects to tensors.
    # --------------------------------------------------------------

    if not isinstance(
        corrupted_cube,
        torch.Tensor,
    ):

        corrupted_cube = torch.as_tensor(
            corrupted_cube,
            dtype=torch.float32,
        )

    if not isinstance(
        mask,
        torch.Tensor,
    ):

        mask = torch.as_tensor(
            mask,
            dtype=torch.float32,
        )

    # --------------------------------------------------------------
    # Convert the seismic cube to floating point.
    # --------------------------------------------------------------

    corrupted_cube = corrupted_cube.float()

    # --------------------------------------------------------------
    # Move the mask to the same device as the seismic cube.
    # --------------------------------------------------------------

    mask = mask.to(
        device=corrupted_cube.device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------------
    # Verify dimensionality.
    # --------------------------------------------------------------

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "corrupted_cube must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(corrupted_cube.shape)}"
        )

    if mask.ndim != 4:

        raise ValueError(
            "mask must have shape "
            "(C, D, H, W). "
            f"Received: {tuple(mask.shape)}"
        )

    # --------------------------------------------------------------
    # Verify identical shapes.
    # --------------------------------------------------------------

    if corrupted_cube.shape != mask.shape:

        raise ValueError(
            "corrupted_cube and mask must have identical shapes. "
            f"Cube: {tuple(corrupted_cube.shape)}, "
            f"Mask: {tuple(mask.shape)}"
        )

    # --------------------------------------------------------------
    # Check finite values.
    # --------------------------------------------------------------

    if not torch.isfinite(
        corrupted_cube
    ).all():

        raise ValueError(
            "corrupted_cube contains NaN or Inf values."
        )

    if not torch.isfinite(
        mask
    ).all():

        raise ValueError(
            "mask contains NaN or Inf values."
        )

    # --------------------------------------------------------------
    # Check binary observation mask.
    # --------------------------------------------------------------

    if not torch.all(
        (mask == 0.0)
        |
        (mask == 1.0)
    ):

        unique_values = torch.unique(mask)

        raise ValueError(
            "mask must contain only 0.0 and 1.0. "
            f"Found values: {unique_values.tolist()}"
        )

    # --------------------------------------------------------------
    # At least one observed sample is required.
    # --------------------------------------------------------------

    if not (
        mask == 1.0
    ).any():

        raise ValueError(
            "The mask contains no observed seismic samples."
        )

    return (
        corrupted_cube,
        mask,
    )


# ======================================================================
# PARAMETER VALIDATION
# ======================================================================

def _validate_parameters(
    prediction_order: int,
    iterations: int,
    min_trace_observed_fraction: float,
    convergence_tolerance: float,
) -> None:
    """
    Validate f-x reconstruction parameters.
    """

    # --------------------------------------------------------------
    # Prediction order.
    # --------------------------------------------------------------

    if not isinstance(
        prediction_order,
        int,
    ):

        raise TypeError(
            "prediction_order must be an integer."
        )

    if prediction_order < 1:

        raise ValueError(
            "prediction_order must be >= 1."
        )

    # --------------------------------------------------------------
    # Number of iterations.
    # --------------------------------------------------------------

    if not isinstance(
        iterations,
        int,
    ):

        raise TypeError(
            "iterations must be an integer."
        )

    if iterations < 1:

        raise ValueError(
            "iterations must be >= 1."
        )

    # --------------------------------------------------------------
    # Minimum observed trace fraction.
    # --------------------------------------------------------------

    if not (
        0.0
        <
        min_trace_observed_fraction
        <=
        1.0
    ):

        raise ValueError(
            "min_trace_observed_fraction must satisfy "
            "0 < value <= 1."
        )

    # --------------------------------------------------------------
    # Convergence tolerance.
    # --------------------------------------------------------------

    if convergence_tolerance <= 0.0:

        raise ValueError(
            "convergence_tolerance must be > 0."
        )


# ======================================================================
# TRACE OBSERVATION ANALYSIS
# ======================================================================

def _trace_observation_fraction(
    observation_mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate the observed fraction of every spatial trace.

    Parameters
    ----------
    observation_mask:
        Boolean mask with shape (D, H, W).

    Returns
    -------
    ndarray
        Observation fractions with shape (H, W).
    """

    return np.mean(
        observation_mask,
        axis=0,
    )


# ======================================================================
# PREDICTION COEFFICIENT ESTIMATION
# ======================================================================


def _estimate_prediction_coefficients(
    trace: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Estimate numerically stabilized complex-valued
    linear prediction coefficients for f-x reconstruction.

    Ridge regularization reduces sensitivity to ill-conditioned
    prediction systems. An L1 coefficient constraint limits
    recursive prediction growth.
    """

    trace = np.asarray(trace, dtype=np.complex128)

    # Always provide a safe zero-coefficient fallback.
    zero_coefficients = np.zeros(
        order,
        dtype=np.complex128,
    )

    # Reject non-finite input.
    if not np.isfinite(trace).all():
        return zero_coefficients

    # A prediction filter requires more samples than its order.
    if trace.size <= order:
        return zero_coefficients

    # Normalize the trace to improve numerical conditioning.
    trace_scale = float(np.max(np.abs(trace)))

    if not np.isfinite(trace_scale) or trace_scale <= 1e-12:
        return zero_coefficients

    normalized_trace = trace / trace_scale

    # Construct the least-squares prediction system.
    number_of_rows = trace.size - order

    A = np.empty(
        (number_of_rows, order),
        dtype=np.complex128,
    )

    b = np.empty(
        number_of_rows,
        dtype=np.complex128,
    )

    for row, index in enumerate(range(order, trace.size)):
        A[row, :] = normalized_trace[
            index - order:index
        ][::-1]

        b[row] = normalized_trace[index]

    # Reject an invalid prediction system.
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        return zero_coefficients

    try:
        # Form the regularized complex least-squares system.
        A_h = A.conj().T
        gram = A_h @ A
        rhs = A_h @ b

        average_energy = float(
            np.real(np.trace(gram)) / order
        )

        if not np.isfinite(average_energy):
            return zero_coefficients

        # Scale regularization to the prediction system's energy.
        ridge = 1e-6 * max(average_energy, 1e-12)

        regularized_gram = gram + ridge * np.eye(
            order,
            dtype=np.complex128,
        )

        coefficients = np.linalg.solve(
            regularized_gram,
            rhs,
        )

    except (np.linalg.LinAlgError, FloatingPointError):
        return zero_coefficients

    # Reject non-finite estimated coefficients.
    if not np.isfinite(coefficients).all():
        return zero_coefficients

    # ----------------------------------------------------------
    # Stability constraint
    # ----------------------------------------------------------
    # Limit the sum of coefficient magnitudes. This constrains
    # recursive predictions and reduces the risk of runaway
    # amplitudes during forward and reverse prediction.
    coefficient_l1 = float(np.sum(np.abs(coefficients)))

    maximum_coefficient_l1 = 0.98

    if not np.isfinite(coefficient_l1):
        return zero_coefficients

    if coefficient_l1 > maximum_coefficient_l1:
        coefficients = (
            coefficients
            * (maximum_coefficient_l1 / coefficient_l1)
        )

    if not np.isfinite(coefficients).all():
        return zero_coefficients

    return coefficients

# ======================================================================
# ONE-DIMENSIONAL F-X PREDICTION
# ======================================================================

def _fx_predict_1d(
    spatial_trace: np.ndarray,
    observed: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Perform forward and reverse f-x prediction on one spatial trace.
    """

    # --------------------------------------------------------------
    # Create independent working copies.
    # --------------------------------------------------------------

    values = np.asarray(
        spatial_trace,
        dtype=np.complex128,
    ).copy()

    observed = np.asarray(
        observed,
        dtype=bool,
    )

    # --------------------------------------------------------------
    # No missing spatial samples.
    # --------------------------------------------------------------

    if observed.all():

        return values

    # --------------------------------------------------------------
    # No observed spatial samples.
    # --------------------------------------------------------------

    if not observed.any():

        return values

    # --------------------------------------------------------------
    # Extract observed values for coefficient estimation.
    # --------------------------------------------------------------

    observed_values = values[
        observed
    ]

    # --------------------------------------------------------------
    # Insufficient observations.
    # --------------------------------------------------------------

    if observed_values.size <= order:

        return values

    # --------------------------------------------------------------
    # Estimate prediction coefficients.
    # --------------------------------------------------------------

    coefficients = (
        _estimate_prediction_coefficients(
            observed_values,
            order,
        )
    )

    # --------------------------------------------------------------
    # Forward prediction.
    # --------------------------------------------------------------

    forward = values.copy()

    for index in range(
        order,
        forward.size,
    ):

        if observed[index]:

            continue

        previous = forward[
            index - order:index
        ][::-1]

        prediction = np.dot(
            coefficients,
            previous,
        )

        if np.isfinite(
            prediction
        ):

            forward[index] = (
                prediction
            )

    # --------------------------------------------------------------
    # Reverse prediction.
    # --------------------------------------------------------------

    backward = values[
        ::-1
    ].copy()

    reversed_observed = observed[
        ::-1
    ]

    for index in range(
        order,
        backward.size,
    ):

        if reversed_observed[index]:

            continue

        previous = backward[
            index - order:index
        ][::-1]

        prediction = np.dot(
            coefficients,
            previous,
        )

        if np.isfinite(
            prediction
        ):

            backward[index] = (
                prediction
            )

    backward = backward[
        ::-1
    ]

    # --------------------------------------------------------------
    # Identify missing positions.
    # --------------------------------------------------------------

    missing = ~observed

    forward_values = (
        forward[missing]
    )

    backward_values = (
        backward[missing]
    )

    valid_forward = np.isfinite(
        forward_values
    )

    valid_backward = np.isfinite(
        backward_values
    )

    # --------------------------------------------------------------
    # Begin reconstruction from original values.
    # --------------------------------------------------------------

    reconstructed = values.copy()

    reconstructed_missing = (
        reconstructed[missing]
    )

    # --------------------------------------------------------------
    # Both forward and reverse predictions valid.
    # --------------------------------------------------------------

    both_valid = (
        valid_forward
        &
        valid_backward
    )

    reconstructed_missing[
        both_valid
    ] = (
        0.5
        *
        forward_values[both_valid]
        +
        0.5
        *
        backward_values[both_valid]
    )

    # --------------------------------------------------------------
    # Forward prediction only.
    # --------------------------------------------------------------

    forward_only = (
        valid_forward
        &
        ~valid_backward
    )

    reconstructed_missing[
        forward_only
    ] = (
        forward_values[
            forward_only
        ]
    )

    # --------------------------------------------------------------
    # Reverse prediction only.
    # --------------------------------------------------------------

    backward_only = (
        ~valid_forward
        &
        valid_backward
    )

    reconstructed_missing[
        backward_only
    ] = (
        backward_values[
            backward_only
        ]
    )

    # --------------------------------------------------------------
    # Insert reconstructed missing values.
    # --------------------------------------------------------------

    reconstructed[
        missing
    ] = reconstructed_missing

    # --------------------------------------------------------------
    # Restore observed values exactly.
    # --------------------------------------------------------------

    reconstructed[
        observed
    ] = values[
        observed
    ]

    return reconstructed


# ======================================================================
# FREQUENCY-DOMAIN SPATIAL PREDICTION
# ======================================================================

def _predict_frequency_plane(
    frequency_plane: np.ndarray,
    trace_observed_h: np.ndarray,
    trace_observed_w: np.ndarray,
    prediction_order: int,
) -> np.ndarray:
    """
    Apply f-x prediction to one frequency plane.
    """

    # --------------------------------------------------------------
    # Working copy.
    # --------------------------------------------------------------

    plane = np.asarray(
        frequency_plane,
        dtype=np.complex128,
    ).copy()

    original_plane = plane.copy()

    # --------------------------------------------------------------
    # Prediction along H direction.
    # --------------------------------------------------------------

    for width_index in range(
        plane.shape[1]
    ):

        trace = plane[
            :,
            width_index
        ]

        observed = (
            trace_observed_h[
                :,
                width_index
            ]
        )

        plane[
            :,
            width_index
        ] = _fx_predict_1d(
            trace,
            observed,
            prediction_order,
        )

    # --------------------------------------------------------------
    # Prediction along W direction.
    # --------------------------------------------------------------

    for height_index in range(
        plane.shape[0]
    ):

        trace = plane[
            height_index,
            :
        ]

        observed = (
            trace_observed_w[
                height_index,
                :
            ]
        )

        plane[
            height_index,
            :
        ] = _fx_predict_1d(
            trace,
            observed,
            prediction_order,
        )

    # --------------------------------------------------------------
    # Restore any non-finite values generated numerically.
    # --------------------------------------------------------------

    invalid = ~np.isfinite(
        plane
    )

    if invalid.any():

        plane[
            invalid
        ] = original_plane[
            invalid
        ]

    return plane


# ======================================================================
# PUBLIC F-X RECONSTRUCTION
# ======================================================================

def fx_prediction_reconstruction(
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
    prediction_order: int = DEFAULT_PREDICTION_ORDER,
    iterations: int = DEFAULT_ITERATIONS,
    min_trace_observed_fraction: float = (
        DEFAULT_MIN_TRACE_OBSERVED_FRACTION
    ),
    convergence_tolerance: float = (
        DEFAULT_CONVERGENCE_TOLERANCE
    ),
) -> torch.Tensor:
    """
    Reconstruct a 3-D seismic cube using f-x prediction.

    Ground truth is NOT used.

    Parameters
    ----------
    corrupted_cube:
        Incomplete seismic cube:

            (C, D, H, W)

    mask:
        Binary observation mask:

            1 = observed
            0 = missing

    Returns
    -------
    torch.Tensor
        Reconstructed seismic cube.
    """

    # ==============================================================
    # VALIDATION
    # ==============================================================

    corrupted_cube, mask = _validate_inputs(
        corrupted_cube,
        mask,
    )

    _validate_parameters(
        prediction_order=prediction_order,
        iterations=iterations,
        min_trace_observed_fraction=(
            min_trace_observed_fraction
        ),
        convergence_tolerance=(
            convergence_tolerance
        ),
    )

    # ==============================================================
    # PRESERVE INPUT PROPERTIES
    # ==============================================================

    original_device = (
        corrupted_cube.device
    )

    original_dtype = (
        corrupted_cube.dtype
    )

    # ==============================================================
    # MOVE TO NUMPY
    # ==============================================================

    cube = (
        corrupted_cube.detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )

    observation_mask = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(
            bool,
            copy=False,
        )
    )

    # ==============================================================
    # INITIAL RECONSTRUCTION
    # ==============================================================

    reconstructed = cube.copy()

    # ==============================================================
    # PROCESS EACH CHANNEL
    # ==============================================================

    for channel in range(
        cube.shape[0]
    ):

        channel_mask = (
            observation_mask[
                channel
            ]
        )

        channel_reconstruction = (
            reconstructed[
                channel
            ]
        )

        # ----------------------------------------------------------
        # Determine spatial trace observation fractions.
        # ----------------------------------------------------------

        trace_fraction = (
            _trace_observation_fraction(
                channel_mask
            )
        )

        # ----------------------------------------------------------
        # Spatial traces satisfying the observation criterion.
        # ----------------------------------------------------------

        trace_observed_h = (
            trace_fraction
            >=
            min_trace_observed_fraction
        )

        trace_observed_w = (
            trace_fraction
            >=
            min_trace_observed_fraction
        )

        # ==========================================================
        # ITERATIVE RECONSTRUCTION
        # ==========================================================

        for _ in range(
            iterations
        ):

            previous_iteration = (
                channel_reconstruction.copy()
            )

            # Validate the current reconstruction before the FFT.
            if not np.isfinite(channel_reconstruction).all():
                raise FloatingPointError(
                    "Non-finite values detected before the f-x "
                    f"Fourier transform in channel {channel}."
                )

            # ------------------------------------------------------
            # Fourier transform along temporal/depth axis.
            # ------------------------------------------------------

            frequency_cube = np.fft.rfft(
                channel_reconstruction,
                axis=0,
            )

            # ------------------------------------------------------
            # Process every positive frequency.
            # ------------------------------------------------------

            for frequency in range(
                frequency_cube.shape[0]
            ):

                frequency_cube[
                    frequency
                ] = _predict_frequency_plane(
                    frequency_plane=(
                        frequency_cube[
                            frequency
                        ]
                    ),
                    trace_observed_h=(
                        trace_observed_h
                    ),
                    trace_observed_w=(
                        trace_observed_w
                    ),
                    prediction_order=(
                        prediction_order
                    ),
                )

            # ------------------------------------------------------
            # Inverse Fourier transform.
            # ------------------------------------------------------

            updated = np.fft.irfft(
                frequency_cube,
                n=(
                    channel_reconstruction.shape[
                        0
                    ]
                ),
                axis=0,
            )

            updated = updated.astype(
                np.float64,
                copy=False,
            )

            # ------------------------------------------------------
            # Numerical validation.
            # ------------------------------------------------------

            if not np.isfinite(
                updated
            ).all():

                raise FloatingPointError(
                    "f-x prediction produced NaN or Inf "
                    f"values in channel {channel}."
                )

            # ------------------------------------------------------
            # CRITICAL:
            #
            # Restore all observed samples exactly.
            # ------------------------------------------------------

            updated[
                channel_mask
            ] = cube[
                channel
            ][
                channel_mask
            ]

            # ------------------------------------------------------
            # Calculate relative change.
            # ------------------------------------------------------

            difference = np.linalg.norm(
                updated
                -
                previous_iteration
            )

            reference = max(
                np.linalg.norm(
                    previous_iteration
                ),
                1.0e-12,
            )

            relative_change = (
                difference
                /
                reference
            )

            # ------------------------------------------------------
            # Accept reconstruction.
            # ------------------------------------------------------

            channel_reconstruction = (
                updated
            )

            # ------------------------------------------------------
            # Convergence criterion.
            # ------------------------------------------------------

            if (
                relative_change
                <
                convergence_tolerance
            ):

                break

        # ----------------------------------------------------------
        # Final channel-level observed-data restoration.
        # ----------------------------------------------------------

        channel_reconstruction[
            channel_mask
        ] = cube[
            channel
        ][
            channel_mask
        ]

        reconstructed[
            channel
        ] = channel_reconstruction

    # ==============================================================
    # GLOBAL FINITE-VALUE CHECK
    # ==============================================================

    if not np.isfinite(
        reconstructed
    ).all():

        raise FloatingPointError(
            "Final f-x reconstruction contains "
            "NaN or Inf values."
        )

    # ==============================================================
    # FINAL OBSERVED-DATA RESTORATION
    # ==============================================================

    reconstructed[
        observation_mask
    ] = cube[
        observation_mask
    ]

    # ==============================================================
    # CONVERT BACK TO TORCH
    # ==============================================================

    result = torch.from_numpy(
        reconstructed
    ).to(
        device=original_device,
        dtype=original_dtype,
    )

    # ==============================================================
    # SHAPE CHECK
    # ==============================================================

    if result.shape != corrupted_cube.shape:

        raise RuntimeError(
            "f-x prediction changed the input shape. "
            f"Input: {tuple(corrupted_cube.shape)}, "
            f"Output: {tuple(result.shape)}"
        )

    # ==============================================================
    # FINITE-VALUE CHECK
    # ==============================================================

    if not torch.isfinite(
        result
    ).all():

        raise RuntimeError(
            "f-x prediction produced NaN or Inf values."
        )

    # ==============================================================
    # EXACT OBSERVED-DATA PRESERVATION
    # ==============================================================

    observed_values = (
        mask == 1.0
    )

    if observed_values.any():

        observed_difference = torch.max(
            torch.abs(
                result[
                    observed_values
                ]
                -
                corrupted_cube[
                    observed_values
                ]
            )
        )

        if (
            observed_difference.item()
            >
            OBSERVED_PRESERVATION_TOLERANCE
        ):

            raise RuntimeError(
                "f-x prediction modified observed "
                "seismic samples. "
                f"Maximum difference: "
                f"{observed_difference.item():.6e}"
            )

    return result


# ======================================================================
# CONTROLLED MATRIX CONSTRUCTION
# ======================================================================

def build_controlled_cases() -> list[dict[str, Any]]:
    """
    Build the complete 750-case factorial experimental matrix.

    Returns
    -------
    list of dictionaries
        One dictionary for each controlled experimental case.
    """

    cases: list[dict[str, Any]] = []

    case_id = 1

    # ------------------------------------------------------------------
    # Cartesian product of all experimental factors.
    # ------------------------------------------------------------------

    for (
        geology,
        mask_mode,
        missing_rate,
        seed,
    ) in product(
        GEOLOGICAL_MODES,
        MASK_MODES,
        MISSING_RATES,
        RANDOM_SEEDS,
    ):

        cases.append(
            {
                "case_id": case_id,
                "geological_mode": geology,
                "mask_mode": mask_mode,
                "missing_rate": missing_rate,
                "seed": seed,
            }
        )

        case_id += 1

    # ------------------------------------------------------------------
    # Verify the factorial design.
    # ------------------------------------------------------------------

    expected_cases = (
        len(GEOLOGICAL_MODES)
        *
        len(MASK_MODES)
        *
        len(MISSING_RATES)
        *
        len(RANDOM_SEEDS)
    )

    if len(cases) != expected_cases:

        raise RuntimeError(
            "Controlled matrix construction error. "
            f"Expected {expected_cases} cases but "
            f"constructed {len(cases)}."
        )

    if len(cases) != 750:

        raise RuntimeError(
            "The f-x controlled matrix must contain "
            f"750 cases. Constructed {len(cases)}."
        )

    return cases


# ======================================================================
# CONTROLLED CASE SELECTION
# ======================================================================

def select_cases_for_execution(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Select the cases to execute.

    CONTROLLED_MATRIX_CASE_LIMIT = None
        -> all 750 cases

    CONTROLLED_MATRIX_CASE_LIMIT = 10
        -> first 10 deterministic cases
    """

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):

        return cases

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        < 1
    ):

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "None or an integer >= 1."
        )

    return cases[
        :CONTROLLED_MATRIX_CASE_LIMIT
    ]


# ======================================================================
# DATASET SAMPLE EXTRACTION
# ======================================================================

def _extract_sample(
    sample: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract target, corrupted cube and mask from a dataset sample.

    The function supports the dictionary conventions used by the
    synthetic seismic dataset and provides clear errors if the
    expected fields are unavailable.
    """

    # ------------------------------------------------------------------
    # Dictionary-based dataset output.
    # ------------------------------------------------------------------

    if isinstance(
        sample,
        dict,
    ):

        # --------------------------------------------------------------
        # Possible target/clean field names.
        # --------------------------------------------------------------

        target = None

        for key in (
            "target",
            "clean",
            "clean_cube",
            "ground_truth",
            "ground_truth_cube",
        ):

            if key in sample:

                target = sample[
                    key
                ]

                break

        # --------------------------------------------------------------
        # Possible corrupted/input field names.
        # --------------------------------------------------------------

        corrupted = None

        for key in (
            "corrupted",
            "corrupted_cube",
            "input",
            "input_cube",
        ):

            if key in sample:

                corrupted = sample[
                    key
                ]

                break

        # --------------------------------------------------------------
        # Mask.
        # --------------------------------------------------------------

        mask = None

        for key in (
            "mask",
            "observation_mask",
        ):

            if key in sample:

                mask = sample[
                    key
                ]

                break

        if (
            target is None
            or corrupted is None
            or mask is None
        ):

            raise KeyError(
                "Could not identify target, corrupted cube, "
                "and mask in the synthetic dataset sample. "
                f"Available keys: {list(sample.keys())}"
            )

        return (
            torch.as_tensor(
                target,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                corrupted,
                dtype=torch.float32,
            ),
            torch.as_tensor(
                mask,
                dtype=torch.float32,
            ),
        )

    # ------------------------------------------------------------------
    # Tuple/list dataset output.
    # ------------------------------------------------------------------

    if isinstance(
        sample,
        (tuple, list),
    ):

        if len(sample) < 3:

            raise ValueError(
                "Dataset sample must contain at least "
                "three elements: target, corrupted, mask."
            )

        return (
            torch.as_tensor(
                sample[0],
                dtype=torch.float32,
            ),
            torch.as_tensor(
                sample[1],
                dtype=torch.float32,
            ),
            torch.as_tensor(
                sample[2],
                dtype=torch.float32,
            ),
        )

    # ------------------------------------------------------------------
    # Unsupported sample format.
    # ------------------------------------------------------------------

    raise TypeError(
        "Unsupported synthetic dataset sample type: "
        f"{type(sample)}"
    )


# ======================================================================
# OBSERVED-DATA PRESERVATION CHECK
# ======================================================================

def _calculate_observed_preservation_error(
    reconstruction: torch.Tensor,
    corrupted_cube: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Calculate the maximum absolute difference on observed samples.
    """

    observed = (
        mask == 1.0
    )

    if not observed.any():

        return 0.0

    difference = torch.abs(
        reconstruction[
            observed
        ]
        -
        corrupted_cube[
            observed
        ]
    )

    return float(
        torch.max(
            difference
        ).item()
    )


# ======================================================================
# MISSING-REGION METRIC FALLBACK
# ======================================================================

def _calculate_missing_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """
    Calculate missing-region MAE and RMSE directly.

    This is a safeguard so the controlled baseline remains compatible
    with the canonical metric convention.
    """

    missing = (
        mask == 0.0
    )

    if not missing.any():

        return (
            0.0,
            0.0,
        )

    difference = (
        reconstruction[
            missing
        ]
        -
        target[
            missing
        ]
    )

    mae = torch.mean(
        torch.abs(
            difference
        )
    ).item()

    rmse = torch.sqrt(
        torch.mean(
            difference
            ** 2
        )
    ).item()

    return (
        float(mae),
        float(rmse),
    )


# ======================================================================
# SINGLE CONTROLLED EXPERIMENT
# ======================================================================

def run_single_experiment(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one f-x controlled-matrix case.

    Ground truth is used ONLY for evaluation.

    The reconstruction algorithm receives:

        corrupted_cube
        mask
    """

    # ------------------------------------------------------------------
    # Read case factors.
    # ------------------------------------------------------------------

    case_id = int(
        case["case_id"]
    )

    geology = str(
        case["geological_mode"]
    )

    mask_mode = str(
        case["mask_mode"]
    )

    missing_rate = float(
        case["missing_rate"]
    )

    seed = int(
        case["seed"]
    )

    # ------------------------------------------------------------------
    # Create the exact synthetic dataset for this case.
    # ------------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=BASELINE_NUM_SAMPLES,
        cube_size=BASELINE_CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geology,
        mask_mode=mask_mode,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Obtain the first deterministic sample.
    # ------------------------------------------------------------------

    sample = dataset[0]

    (
        target,
        corrupted_cube,
        mask,
    ) = _extract_sample(
        sample
    )

    # ------------------------------------------------------------------
    # Remove unnecessary batch dimensions if present.
    #
    # The f-x algorithm expects:
    #
    #     [C,D,H,W]
    # ------------------------------------------------------------------

    if target.ndim == 5 and target.shape[0] == 1:

        target = target.squeeze(
            0
        )

    if (
        corrupted_cube.ndim == 5
        and
        corrupted_cube.shape[0] == 1
    ):

        corrupted_cube = (
            corrupted_cube.squeeze(
                0
            )
        )

    if mask.ndim == 5 and mask.shape[0] == 1:

        mask = mask.squeeze(
            0
        )

    # ------------------------------------------------------------------
    # Validate tensor shapes.
    # ------------------------------------------------------------------

    if corrupted_cube.ndim != 4:

        raise ValueError(
            "Expected corrupted cube with shape "
            f"[C,D,H,W]. Received "
            f"{tuple(corrupted_cube.shape)}."
        )

    if target.shape != corrupted_cube.shape:

        raise ValueError(
            "Target and corrupted cube shapes differ. "
            f"Target: {tuple(target.shape)}, "
            f"Corrupted: {tuple(corrupted_cube.shape)}"
        )

    if mask.shape != corrupted_cube.shape:

        raise ValueError(
            "Mask and corrupted cube shapes differ. "
            f"Mask: {tuple(mask.shape)}, "
            f"Corrupted: {tuple(corrupted_cube.shape)}"
        )

    # ------------------------------------------------------------------
    # Run f-x reconstruction.
    #
    # IMPORTANT:
    # target is deliberately NOT supplied here.
    # ------------------------------------------------------------------

    start_time = time.perf_counter()

    reconstruction = (
        fx_prediction_reconstruction(
            corrupted_cube=corrupted_cube,
            mask=mask,
            prediction_order=(
                DEFAULT_PREDICTION_ORDER
            ),
            iterations=(
                DEFAULT_ITERATIONS
            ),
            min_trace_observed_fraction=(
                DEFAULT_MIN_TRACE_OBSERVED_FRACTION
            ),
            convergence_tolerance=(
                DEFAULT_CONVERGENCE_TOLERANCE
            ),
        )
    )

    runtime_seconds = (
        time.perf_counter()
        -
        start_time
    )

    # ------------------------------------------------------------------
    # Calculate canonical reconstruction metrics.
    #
    # Canonical API expects:
    #
    #     [B,C,D,H,W]
    # ------------------------------------------------------------------

    prediction_batch = (
        reconstruction.unsqueeze(
            0
        )
    )

    target_batch = (
        target.unsqueeze(
            0
        )
    )

    mask_batch = (
        mask.unsqueeze(
            0
        )
    )

    metrics = (
        calculate_reconstruction_metrics(
            prediction=prediction_batch,
            target=target_batch,
            mask=mask_batch,
        )
    )

    # ------------------------------------------------------------------
    # Missing-region metrics.
    # ------------------------------------------------------------------

    missing_mae, missing_rmse = (
        _calculate_missing_metrics(
            reconstruction,
            target,
            mask,
        )
    )

    # ------------------------------------------------------------------
    # Observed-data preservation.
    # ------------------------------------------------------------------

    observed_error = (
        _calculate_observed_preservation_error(
            reconstruction,
            corrupted_cube,
            mask,
        )
    )

    observed_pass = (
        observed_error
        <=
        OBSERVED_PRESERVATION_TOLERANCE
    )

    # ------------------------------------------------------------------
    # Determine whether all numerical outputs are valid.
    # ------------------------------------------------------------------

    finite_reconstruction = bool(
        torch.isfinite(
            reconstruction
        ).all().item()
    )

    # ------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------

    status = (
        "PASS"
        if (
            finite_reconstruction
            and
            observed_pass
        )
        else
        "FAIL"
    )

    # ------------------------------------------------------------------
    # Return standardized result record.
    # ------------------------------------------------------------------

    return {
        "case_id": case_id,
        "geological_mode": geology,
        "mask_mode": mask_mode,
        "missing_rate": missing_rate,
        "seed": seed,
        "prediction_order": (
            DEFAULT_PREDICTION_ORDER
        ),
        "iterations": (
            DEFAULT_ITERATIONS
        ),
        "min_trace_observed_fraction": (
            DEFAULT_MIN_TRACE_OBSERVED_FRACTION
        ),
        "convergence_tolerance": (
            DEFAULT_CONVERGENCE_TOLERANCE
        ),
        "mae": float(
            metrics["mae"]
        ),
        "mse": float(
            metrics["mse"]
        ),
        "rmse": float(
            metrics["rmse"]
        ),
        "relative_l2": float(
            metrics["relative_l2"]
        ),
        "relative_error": float(
            metrics["relative_error"]
        ),
        "psnr": float(
            metrics["psnr"]
        ),
        "snr": float(
            metrics["snr"]
        ),
        "ssim": float(
            metrics["ssim"]
        ),
        "missing_mae": float(
            metrics.get(
                "missing_mae",
                missing_mae,
            )
        ),
        "missing_rmse": float(
            metrics.get(
                "missing_rmse",
                missing_rmse,
            )
        ),
        "observed_mae": float(
            metrics.get(
                "observed_mae",
                0.0,
            )
        ),
        "observed_rmse": float(
            metrics.get(
                "observed_rmse",
                0.0,
            )
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
        "observed_preservation_error": float(
            observed_error
        ),
        "observed_preservation": (
            "PASS"
            if observed_pass
            else "FAIL"
        ),
        "finite_reconstruction": (
            "PASS"
            if finite_reconstruction
            else "FAIL"
        ),
        "status": status,
    }


# ======================================================================
# SUMMARY CALCULATION
# ======================================================================

def calculate_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate summary statistics for successful controlled cases.
    """

    if not results:

        return {
            "number_of_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
        }

    successful = [
        result
        for result in results
        if result["status"] == "PASS"
    ]

    failed = [
        result
        for result in results
        if result["status"] == "FAIL"
    ]

    summary: dict[str, Any] = {
        "number_of_cases": len(results),
        "successful_cases": len(successful),
        "failed_cases": len(failed),
    }

    # ------------------------------------------------------------------
    # Metrics to summarize.
    # ------------------------------------------------------------------

    metric_names = (
        "mae",
        "mse",
        "rmse",
        "relative_l2",
        "relative_error",
        "psnr",
        "snr",
        "ssim",
        "missing_mae",
        "missing_rmse",
        "observed_mae",
        "observed_rmse",
        "runtime_seconds",
        "observed_preservation_error",
    )

    # ------------------------------------------------------------------
    # Calculate mean/std/min/max for each metric.
    # ------------------------------------------------------------------

    for metric_name in metric_names:

        if not successful:

            summary[
                f"{metric_name}_mean"
            ] = float("nan")

            summary[
                f"{metric_name}_std"
            ] = float("nan")

            summary[
                f"{metric_name}_min"
            ] = float("nan")

            summary[
                f"{metric_name}_max"
            ] = float("nan")

            continue

        values = np.asarray(
            [
                result[
                    metric_name
                ]
                for result in successful
            ],
            dtype=np.float64,
        )

        summary[
            f"{metric_name}_mean"
        ] = float(
            np.mean(
                values
            )
        )

        summary[
            f"{metric_name}_std"
        ] = float(
            np.std(
                values
            )
        )

        summary[
            f"{metric_name}_min"
        ] = float(
            np.min(
                values
            )
        )

        summary[
            f"{metric_name}_max"
        ] = float(
            np.max(
                values
            )
        )

    return summary


# ======================================================================
# CSV WRITING
# ======================================================================

def write_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """
    Write raw controlled-matrix results and summary CSV files.
    """

    # ------------------------------------------------------------------
    # Ensure report directory exists.
    # ------------------------------------------------------------------

    report_directory = Path(
        REPORT_DIR
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Output files.
    # ------------------------------------------------------------------

    results_path = (
        report_directory
        /
        "fx_controlled_matrix.csv"
    )

    summary_path = (
        report_directory
        /
        "fx_controlled_summary.csv"
    )

    # ------------------------------------------------------------------
    # Write raw results.
    # ------------------------------------------------------------------

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with results_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # ------------------------------------------------------------------
    # Write summary.
    # ------------------------------------------------------------------

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            summary
        )

    return (
        results_path,
        summary_path,
    )


# ======================================================================
# CONTROLLED MATRIX MAIN
# ======================================================================

def main() -> None:
    """
    Execute the f-x controlled-matrix experiment.
    """

    # ==============================================================
    # HEADER
    # ==============================================================

    print()
    print(
        "=" * 72
    )
    print(
        "F-X PREDICTION CONTROLLED-MATRIX EXPERIMENT"
    )
    print(
        "=" * 72
    )

    # ==============================================================
    # BUILD COMPLETE MATRIX
    # ==============================================================

    cases = build_controlled_cases()

    cases_to_execute = (
        select_cases_for_execution(
            cases
        )
    )

    # ==============================================================
    # EXPERIMENT INFORMATION
    # ==============================================================

    print(
        f"Full controlled matrix: "
        f"{len(cases)} cases"
    )

    print(
        f"Cases to execute: "
        f"{len(cases_to_execute)}"
    )

    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        is None
    ):

        print(
            "Execution mode: FULL EXPERIMENT"
        )

    else:

        print(
            "Execution mode: SMOKE TEST"
        )

    print()
    print(
        "-" * 72
    )

    print(
        "Prediction order:",
        DEFAULT_PREDICTION_ORDER,
    )

    print(
        "Iterations:",
        DEFAULT_ITERATIONS,
    )

    print(
        "Minimum trace observation fraction:",
        DEFAULT_MIN_TRACE_OBSERVED_FRACTION,
    )

    print(
        "Convergence tolerance:",
        DEFAULT_CONVERGENCE_TOLERANCE,
    )

    print(
        "-" * 72
    )

    # ==============================================================
    # EXECUTION
    # ==============================================================

    results: list[
        dict[str, Any]
    ] = []

    failures: list[
        dict[str, Any]
    ] = []

    experiment_start = (
        time.perf_counter()
    )

    # ------------------------------------------------------------------
    # Execute each selected case.
    # ------------------------------------------------------------------

    for index, case in enumerate(
        cases_to_execute,
        start=1,
    ):

        print()
        print(
            "-" * 72
        )

        print(
            f"Case {index}/"
            f"{len(cases_to_execute)}"
        )

        print(
            f"Case_ID: "
            f"{case['case_id']}"
        )

        print(
            f"Geology: "
            f"{case['geological_mode']}"
        )

        print(
            f"Mask: "
            f"{case['mask_mode']}"
        )

        print(
            f"Missing rate: "
            f"{case['missing_rate']:.2f}"
        )

        print(
            f"Seed: "
            f"{case['seed']}"
        )

        # --------------------------------------------------------------
        # Execute case.
        # --------------------------------------------------------------

        try:

            result = (
                run_single_experiment(
                    case
                )
            )

            results.append(
                result
            )

            # ----------------------------------------------------------
            # Print primary metrics.
            # ----------------------------------------------------------

            print(
                f"MAE: "
                f"{result['mae']:.6f}"
            )

            print(
                f"RMSE: "
                f"{result['rmse']:.6f}"
            )

            print(
                f"PSNR: "
                f"{result['psnr']:.4f}"
            )

            print(
                f"SSIM: "
                f"{result['ssim']:.6f}"
            )

            print(
                f"Missing MAE: "
                f"{result['missing_mae']:.6f}"
            )

            print(
                f"Runtime: "
                f"{result['runtime_seconds']:.3f} s"
            )

            print(
                "Observed preservation:",
                result[
                    "observed_preservation"
                ],
            )

            # ----------------------------------------------------------
            # Report case failure if any validation failed.
            # ----------------------------------------------------------

            if result[
                "status"
            ] != "PASS":

                failures.append(
                    {
                        "case": case,
                        "error": (
                            "Numerical or observed-data "
                            "preservation validation failed."
                        ),
                    }
                )

        except Exception as exc:

            # ----------------------------------------------------------
            # Preserve failed case information.
            # ----------------------------------------------------------

            failure = {
                "case": case,
                "error": str(
                    exc
                ),
            }

            failures.append(
                failure
            )

            print(
                "Status: FAIL"
            )

            print(
                "Error:",
                exc,
            )

    # ==============================================================
    # EXPERIMENT SUMMARY
    # ==============================================================

    total_runtime = (
        time.perf_counter()
        -
        experiment_start
    )

    summary = (
        calculate_summary(
            results
        )
    )

    # Add experiment-level information.
    summary[
        "full_matrix_cases"
    ] = len(cases)

    summary[
        "executed_cases"
    ] = len(
        cases_to_execute
    )

    summary[
        "execution_mode"
    ] = (
        "FULL"
        if CONTROLLED_MATRIX_CASE_LIMIT
        is None
        else
        "SMOKE_TEST"
    )

    summary[
        "experiment_runtime_seconds"
    ] = float(
        total_runtime
    )

    # ==============================================================
    # WRITE OUTPUTS
    # ==============================================================

    results_path, summary_path = (
        write_results(
            results,
            summary,
        )
    )

    print()
    print(
        "Raw results saved to:"
    )

    print(
        results_path
    )

    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )

    # ==============================================================
    # FINAL STATUS
    # ==============================================================

    successful_cases = sum(
        1
        for result in results
        if result[
            "status"
        ] == "PASS"
    )

    failed_cases = (
        len(
            cases_to_execute
        )
        -
        successful_cases
    )

    print()
    print(
        "=" * 72
    )

    if (
        failed_cases == 0
        and
        len(results)
        ==
        len(cases_to_execute)
    ):

        print(
            "F-X PREDICTION CONTROLLED MATRIX: PASS"
        )

    else:

        print(
            "F-X PREDICTION CONTROLLED MATRIX: FAIL"
        )

    print(
        f"Successful cases: "
        f"{successful_cases}/"
        f"{len(cases_to_execute)}"
    )

    print(
        f"Failed cases: "
        f"{failed_cases}"
    )

    print(
        f"Total experiment runtime: "
        f"{total_runtime:.3f} s"
    )

    print(
        "=" * 72
    )

    # ==============================================================
    # FAILURE DETAILS
    # ==============================================================

    if failures:

        print()
        print(
            "FAILED CASE DETAILS"
        )

        print(
            "-" * 72
        )

        for failure in failures:

            failed_case = (
                failure[
                    "case"
                ]
            )

            print(
                f"Case {failed_case['case_id']}: "
                f"{failure['error']}"
            )

    # ==============================================================
    # DO NOT SILENTLY HIDE FAILURES
    # ==============================================================

    if failed_cases > 0:

        raise RuntimeError(
            f"f-x controlled-matrix experiment "
            f"completed with {failed_cases} failed case(s)."
        )


# ======================================================================
# MODULE ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    main()