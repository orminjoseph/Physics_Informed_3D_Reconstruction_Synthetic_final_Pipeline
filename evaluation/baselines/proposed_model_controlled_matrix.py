"""
====================================================================
PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the FROZEN proposed model over the same controlled
experimental matrix used for the six classical baseline methods.

Experimental matrix
-------------------
Geological modes:
    horizontal
    dipping
    faulted
    folded
    complex
    highly_complex

Missing-data mechanisms:
    random_voxels
    missing_traces
    missing_inlines
    missing_crosslines
    missing_blocks

Missing rates:
    10%
    20%
    30%
    40%
    50%

Seeds:
    42
    43
    44
    45
    46

Total:
    6 × 5 × 5 × 5 = 750 cases

Execution control
-----------------
The complete experimental matrix always contains 750 cases.

For smoke testing, the execution can be limited through:

    CONTROLLED_MATRIX_CASE_LIMIT

defined centrally in:

    utils/config.py

Examples:

    CONTROLLED_MATRIX_CASE_LIMIT = 10
        -> run the first 10 deterministic cases

    CONTROLLED_MATRIX_CASE_LIMIT = None
        -> run all 750 cases

Important
---------
The neural network is NOT retrained for every case.

A single frozen best_model.pth is evaluated across all controlled
experimental cases.

The reconstruction is made data-consistent by restoring the
observed samples exactly after neural-network inference.

Predictive uncertainty:
    Aleatoric variance
    Epistemic variance
    Predictive variance
    Predictive standard deviation

Author: Ormin Joseph
====================================================================
"""


# ====================================================================
# STANDARD LIBRARY IMPORTS
# ====================================================================

import csv
import random
import time
from pathlib import Path


# ====================================================================
# THIRD-PARTY IMPORTS
# ====================================================================

import numpy as np
import torch


# ====================================================================
# PROJECT IMPORTS
# ====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset
)

from models.network import (
    Network3D
)

from models.mc_dropout import (
    MCDropout3D
)

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim
)

from utils.config import (
    MC_DROPOUT_SAMPLES,
    CONTROLLED_MATRIX_CASE_LIMIT,
    BASELINE_CUBE_SIZE,
    BASELINE_NUM_SAMPLES,
    OBSERVED_PRESERVATION_TOLERANCE,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX
)


# ====================================================================
# PROJECT PATHS
# ====================================================================

CHECKPOINT = (
    Path(CHECKPOINT_DIR)
    / "best_model.pth"
)


# ====================================================================
# CONTROLLED EXPERIMENTAL MATRIX
# ====================================================================

GEOLOGICAL_MODES = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


MASK_MODES = [
    "random_voxels",
    "missing_traces",
    "missing_inlines",
    "missing_crosslines",
    "missing_blocks",
]


MISSING_RATES = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


# ====================================================================
# CUBE CONFIGURATION
# ====================================================================

CUBE_SIZE = tuple(
    BASELINE_CUBE_SIZE
)


# ====================================================================
# NUMBER OF DATASET SAMPLES
# ====================================================================

NUM_SAMPLES = int(
    BASELINE_NUM_SAMPLES
)


# ====================================================================
# MC-DROPOUT CONFIGURATION
# ====================================================================

MC_SAMPLES = int(
    MC_DROPOUT_SAMPLES
)


# ====================================================================
# OUTPUT DIRECTORY
# ====================================================================

REPORT_DIR = Path(
    REPORT_DIR
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ====================================================================
# OUTPUT FILES
# ====================================================================

RESULTS_FILE = (
    REPORT_DIR
    /
    "proposed_model_controlled_matrix.csv"
)


SUMMARY_FILE = (
    REPORT_DIR
    /
    "proposed_model_controlled_matrix_summary.csv"
)


# ====================================================================
# NUMERICAL TOLERANCE
# ====================================================================

PRESERVATION_TOLERANCE = float(
    OBSERVED_PRESERVATION_TOLERANCE
)


# ====================================================================
# DEVICE
# ====================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """
    Set deterministic random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ====================================================================
# DEVICE COMPARISON
# ====================================================================

def devices_match(
    tensor_device,
    expected_device
):
    """
    Compare two PyTorch devices correctly.

    Important:
    -----------
    PyTorch may represent the same first CUDA device as:

        cuda

    or:

        cuda:0

    These must be treated as equivalent.

    Examples
    --------
    cuda     == cuda:0    -> True
    cuda:0   == cuda:0    -> True
    cuda:1   == cuda:0    -> False
    cpu      == cuda      -> False
    """

    actual = torch.device(
        tensor_device
    )

    expected = torch.device(
        expected_device
    )


    # ---------------------------------------------------------------
    # Device types must match.
    # ---------------------------------------------------------------

    if actual.type != expected.type:

        return False


    # ---------------------------------------------------------------
    # CPU has no device index.
    # ---------------------------------------------------------------

    if actual.type == "cpu":

        return True


    # ---------------------------------------------------------------
    # CUDA device comparison.
    #
    # torch.device("cuda") has no explicit index.
    #
    # A tensor normally reports:
    #
    #     cuda:0
    #
    # Therefore, resolve an unspecified CUDA index to the current
    # CUDA device before comparing.
    # ---------------------------------------------------------------

    if actual.type == "cuda":

        actual_index = (

            torch.cuda.current_device()

            if actual.index is None

            else actual.index

        )


        expected_index = (

            torch.cuda.current_device()

            if expected.index is None

            else expected.index

        )


        return (
            actual_index
            ==
            expected_index
        )


    # ---------------------------------------------------------------
    # Fallback for other device types.
    # ---------------------------------------------------------------

    return actual == expected


# ====================================================================
# METRIC CONVERSION
# ====================================================================

def metric_to_float(value):
    """
    Convert a metric output to a Python float.
    """

    if isinstance(value, torch.Tensor):

        return float(
            value.detach()
            .cpu()
            .item()
        )

    if isinstance(value, np.ndarray):

        return float(
            value.item()
        )

    return float(value)


# ====================================================================
# FINITE CHECK
# ====================================================================

def tensor_is_finite(tensor):
    """
    Return True when all tensor values are finite.
    """

    return bool(
        torch.isfinite(tensor)
        .all()
        .item()
    )


# ====================================================================
# LOAD FROZEN MODEL
# ====================================================================

def load_model():
    """
    Load the frozen production Network3D checkpoint.
    """

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(

            "\nProposed-model checkpoint was not found:\n"

            f"{CHECKPOINT}\n\n"

            "Expected project-level checkpoint directory:\n"

            f"{CHECKPOINT.parent}\n\n"

            "The trained best_model.pth must exist before "
            "running the controlled matrix."

        )


    print()
    print("=" * 70)
    print("LOADING FROZEN PROPOSED MODEL")
    print("=" * 70)

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Device:")
    print(DEVICE)


    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY
    )


    model = model.to(
        DEVICE
    )


    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )


    if not isinstance(
        checkpoint,
        dict
    ):

        raise TypeError(
            "Unexpected checkpoint format. "
            "Expected a dictionary containing "
            "'model_state_dict'."
        )


    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    print()
    print("Proposed model loaded successfully.")


    if "best_epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint["best_epoch"]
        )


    if "best_validation_loss" in checkpoint:

        print(
            "Best validation loss:",
            checkpoint["best_validation_loss"]
        )


    return model


# ====================================================================
# RUN ONE CONTROLLED EXPERIMENT
# ====================================================================

def run_single_experiment(
    model,
    geological_mode,
    mask_mode,
    missing_rate,
    seed
):
    """
    Run one controlled proposed-model experiment.
    """

    # ================================================================
    # SET RANDOM SEED
    # ================================================================

    set_seed(seed)


    # ================================================================
    # CREATE CONTROLLED DATASET
    # ================================================================

    dataset = SyntheticSeismicDataset(

        num_samples=NUM_SAMPLES,

        cube_size=CUBE_SIZE,

        missing_probability=missing_rate,

        geological_mode=geological_mode,

        mask_mode=mask_mode,

        seed=seed

    )


    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode
    ) = dataset[0]


    # ================================================================
    # METADATA VALIDATION
    # ================================================================

    if actual_geological_mode != geological_mode:

        raise RuntimeError(

            "Geological mode mismatch: "

            f"expected={geological_mode}, "

            f"received={actual_geological_mode}"

        )


    if actual_mask_mode != mask_mode:

        raise RuntimeError(

            "Mask mode mismatch: "

            f"expected={mask_mode}, "

            f"received={actual_mask_mode}"

        )


    # ================================================================
    # SHAPE VALIDATION
    # ================================================================

    expected_shape = (
        1,
        *CUBE_SIZE
    )


    if tuple(corrupted.shape) != expected_shape:

        raise RuntimeError(

            "Unexpected corrupted cube shape: "

            f"expected={expected_shape}, "

            f"received={tuple(corrupted.shape)}"

        )


    if corrupted.shape != target.shape:

        raise RuntimeError(
            "Corrupted and target shapes differ."
        )


    if corrupted.shape != mask.shape:

        raise RuntimeError(
            "Corrupted and mask shapes differ."
        )


    if corrupted.shape != velocity.shape:

        raise RuntimeError(
            "Corrupted and velocity shapes differ."
        )


    # ================================================================
    # FINITE CHECKS
    # ================================================================

    tensors_to_validate = {

        "corrupted":
            corrupted,

        "target":
            target,

        "mask":
            mask,

        "velocity":
            velocity

    }


    for name, tensor in (
        tensors_to_validate.items()
    ):

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )


    # ================================================================
    # MASK VALIDATION
    # ================================================================

    unique_values = torch.unique(mask)


    for value in unique_values.tolist():

        if float(value) not in {
            0.0,
            1.0
        }:

            raise RuntimeError(
                "Mask contains values other than "
                "0 and 1."
            )


    # ================================================================
    # INPUT CONSISTENCY
    # ================================================================

    expected_input = (
        target
        *
        mask
    )


    input_difference = torch.max(

        torch.abs(
            corrupted
            -
            expected_input
        )

    ).item()


    if input_difference > PRESERVATION_TOLERANCE:

        raise RuntimeError(

            "Input consistency check failed: "

            f"{input_difference:.6e}"

        )


    # ================================================================
    # SAMPLE COUNTS
    # ================================================================

    observed_samples = int(

        torch.sum(
            mask == 1
        ).item()

    )


    missing_samples = int(

        torch.sum(
            mask == 0
        ).item()

    )


    if missing_samples == 0:

        raise RuntimeError(
            "No missing samples detected."
        )


    total_samples = (
        observed_samples
        +
        missing_samples
    )


    measured_missing_rate = (
        missing_samples
        /
        total_samples
    )


    # ================================================================
    # MOVE CONTROLLED-EXPERIMENT TENSORS TO COMPUTATION DEVICE
    # ================================================================
    #
    # Dataset tensors are initially created on CPU.
    #
    # All dataset validation above is deliberately performed before
    # this transfer.
    #
    # From this point onward, tensors participating in inference,
    # reconstruction, uncertainty evaluation, and metric calculation
    # are placed on the same computation device as the model.
    #
    # Dataset convention:
    #
    #     [C,D,H,W]
    #
    # Network convention:
    #
    #     [B,C,D,H,W]
    # ================================================================

    corrupted = corrupted.to(
        DEVICE
    )

    target = target.to(
        DEVICE
    )

    mask = mask.to(
        DEVICE
    )

    velocity = velocity.to(
        DEVICE
    )


    # ================================================================
    # DEVICE CONSISTENCY VALIDATION
    # ================================================================
    #
    # IMPORTANT:
    #
    # Do NOT use:
    #
    #     tensor.device != DEVICE
    #
    # because:
    #
    #     torch.device("cuda")
    #
    # and:
    #
    #     torch.device("cuda:0")
    #
    # can represent the same CUDA device.
    #
    # devices_match() explicitly resolves the CUDA index.
    # ================================================================

    experiment_tensors = {

        "corrupted":
            corrupted,

        "target":
            target,

        "mask":
            mask,

        "velocity":
            velocity

    }


    for name, tensor in (
        experiment_tensors.items()
    ):

        if not devices_match(
            tensor.device,
            DEVICE
        ):

            raise RuntimeError(

                f"Device mismatch for {name}: "

                f"expected={DEVICE}, "

                f"received={tensor.device}"

            )


    # ================================================================
    # PREPARE NETWORK INPUT
    # ================================================================

    input_batch = (
        corrupted.unsqueeze(0)
    )


    # ================================================================
    # MC-DROPOUT PREDICTOR
    # ================================================================

    predictor = MCDropout3D(

        model=model,

        num_samples=MC_SAMPLES

    )


    # ================================================================
    # STOCHASTIC INFERENCE
    # ================================================================

    start_time = time.perf_counter()


    predictions = predictor.predict(
        input_batch
    )


    end_time = time.perf_counter()


    runtime_seconds = (
        end_time
        -
        start_time
    )


    # ================================================================
    # REQUIRED MC OUTPUTS
    # ================================================================

    required_keys = {

        "reconstruction_samples",

        "log_variance_samples"

    }


    missing_keys = (
        required_keys
        -
        predictions.keys()
    )


    if missing_keys:

        raise KeyError(

            "Missing MC outputs: "

            f"{missing_keys}"

        )


    reconstruction_samples = (
        predictions[
            "reconstruction_samples"
        ]
    )


    log_variance_samples = (
        predictions[
            "log_variance_samples"
        ]
    )


    # ================================================================
    # MC OUTPUT VALIDATION
    # ================================================================

    if not isinstance(
        reconstruction_samples,
        torch.Tensor
    ):

        raise TypeError(
            "reconstruction_samples must be "
            "a torch.Tensor."
        )


    if not isinstance(
        log_variance_samples,
        torch.Tensor
    ):

        raise TypeError(
            "log_variance_samples must be "
            "a torch.Tensor."
        )


    if not tensor_is_finite(
        reconstruction_samples
    ):

        raise RuntimeError(
            "reconstruction_samples contains "
            "NaN or Inf."
        )


    if not tensor_is_finite(
        log_variance_samples
    ):

        raise RuntimeError(
            "log_variance_samples contains "
            "NaN or Inf."
        )


    # ================================================================
    # VERIFY MC SAMPLE COUNT
    # ================================================================

    if (
        reconstruction_samples.shape[0]
        !=
        MC_SAMPLES
    ):

        raise RuntimeError(

            "Unexpected number of reconstruction MC samples: "

            f"expected={MC_SAMPLES}, "

            f"received={reconstruction_samples.shape[0]}"

        )


    if (
        log_variance_samples.shape[0]
        !=
        MC_SAMPLES
    ):

        raise RuntimeError(

            "Unexpected number of log-variance MC samples: "

            f"expected={MC_SAMPLES}, "

            f"received={log_variance_samples.shape[0]}"

        )


    # ================================================================
    # MC RECONSTRUCTION MEAN
    # ================================================================

    reconstruction_mean = (
        reconstruction_samples.mean(
            dim=0
        )
    )


    # ================================================================
    # PREDICTIVE UNCERTAINTY ESTIMATOR
    # ================================================================

    uncertainty_estimator = (
        PredictiveUncertaintyEstimator(
            min_log_variance=LOG_VARIANCE_MIN,
            max_log_variance=LOG_VARIANCE_MAX
        )
    )


    uncertainty_results = (
        uncertainty_estimator(
            log_variance_samples,
            reconstruction_samples
        )
    )


    # ================================================================
    # EXTRACT UNCERTAINTY COMPONENTS
    # ================================================================

    required_uncertainty_keys = {

        "aleatoric_variance",

        "epistemic_variance",

        "predictive_variance",

        "predictive_std"

    }


    missing_uncertainty_keys = (
        required_uncertainty_keys
        -
        uncertainty_results.keys()
    )


    if missing_uncertainty_keys:

        raise KeyError(

            "Missing predictive-uncertainty outputs: "

            f"{missing_uncertainty_keys}"

        )


    aleatoric_variance = (
        uncertainty_results[
            "aleatoric_variance"
        ]
    )


    epistemic_variance = (
        uncertainty_results[
            "epistemic_variance"
        ]
    )


    predictive_variance = (
        uncertainty_results[
            "predictive_variance"
        ]
    )


    predictive_std = (
        uncertainty_results[
            "predictive_std"
        ]
    )


    # ================================================================
    # UNCERTAINTY SHAPE VALIDATION
    # ================================================================

    expected_uncertainty_shape = (
        reconstruction_mean.shape
    )


    uncertainty_tensors = {

        "aleatoric_variance":
            aleatoric_variance,

        "epistemic_variance":
            epistemic_variance,

        "predictive_variance":
            predictive_variance,

        "predictive_std":
            predictive_std

    }


    for name, tensor in (
        uncertainty_tensors.items()
    ):

        if tuple(tensor.shape) != tuple(
            expected_uncertainty_shape
        ):

            raise RuntimeError(

                f"{name} shape mismatch: "

                f"expected={tuple(expected_uncertainty_shape)}, "

                f"received={tuple(tensor.shape)}"

            )


    # ================================================================
    # FINITE UNCERTAINTY CHECKS
    # ================================================================

    for name, tensor in (
        uncertainty_tensors.items()
    ):

        if not tensor_is_finite(tensor):

            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )


    # ================================================================
    # NON-NEGATIVE VARIANCE CHECKS
    # ================================================================

    if (
        aleatoric_variance < 0
    ).any():

        raise RuntimeError(
            "Negative aleatoric variance detected."
        )


    if (
        epistemic_variance < 0
    ).any():

        raise RuntimeError(
            "Negative epistemic variance detected."
        )


    if (
        predictive_variance < 0
    ).any():

        raise RuntimeError(
            "Negative predictive variance detected."
        )


    if (
        predictive_std < 0
    ).any():

        raise RuntimeError(
            "Negative predictive standard deviation detected."
        )


    # ================================================================
    # PREDICTIVE VARIANCE DECOMPOSITION CHECK
    # ================================================================

    decomposition_error = torch.max(

        torch.abs(

            predictive_variance
            -
            (
                aleatoric_variance
                +
                epistemic_variance
            )

        )

    ).item()


    if decomposition_error > 1.0e-5:

        raise RuntimeError(

            "Predictive uncertainty decomposition failed: "

            f"maximum difference={decomposition_error:.6e}"

        )


    # ================================================================
    # DATA CONSISTENCY PROJECTION
    # ================================================================

    mask_batch = (
        mask.unsqueeze(0)
    )


    reconstruction = (

        reconstruction_mean

        *
        (
            1.0
            -
            mask_batch
        )

        +

        corrupted.unsqueeze(0)

        *
        mask_batch

    )


    # ================================================================
    # RECONSTRUCTION FINITE CHECK
    # ================================================================

    if not tensor_is_finite(
        reconstruction
    ):

        raise RuntimeError(
            "Final reconstruction contains "
            "NaN or Inf."
        )


    # ================================================================
    # OBSERVED SAMPLE PRESERVATION
    # ================================================================

    observed_selector = (
        mask_batch == 1
    )


    observed_difference = torch.max(

        torch.abs(

            reconstruction[
                observed_selector
            ]

            -

            corrupted.unsqueeze(0)[
                observed_selector
            ]

        )

    ).item()


    if observed_difference > PRESERVATION_TOLERANCE:

        raise RuntimeError(

            "Observed-data preservation failed: "

            f"{observed_difference:.6e}"

        )


    # ================================================================
    # MISSING-DATA SELECTOR
    # ================================================================

    missing_selector = (
        mask_batch == 0
    )


    # ================================================================
    # MISSING-DATA METRICS
    # ================================================================

    missing_prediction = (
        reconstruction[
            missing_selector
        ]
    )


    missing_target = (
        target.unsqueeze(0)[
            missing_selector
        ]
    )


    if missing_prediction.numel() == 0:

        raise RuntimeError(
            "Missing-region metric selector "
            "contains zero elements."
        )


    missing_error = (
        missing_prediction
        -
        missing_target
    )


    missing_mae = float(

        torch.mean(

            torch.abs(
                missing_error
            )

        ).item()

    )


    missing_rmse = float(

        torch.sqrt(

            torch.mean(
                missing_error ** 2
            )

        ).item()

    )


    # ================================================================
    # GLOBAL METRICS
    # ================================================================

    target_batch = (
        target.unsqueeze(0)
    )


    reconstruction_batch = reconstruction


    metric_mae = metric_to_float(

        mae(
            reconstruction_batch,
            target_batch
        )

    )


    metric_rmse = metric_to_float(

        rmse(
            reconstruction_batch,
            target_batch
        )

    )


    metric_psnr = metric_to_float(

        psnr(
            reconstruction_batch,
            target_batch
        )

    )


    metric_snr = metric_to_float(

        snr(
            reconstruction_batch,
            target_batch
        )

    )


    metric_ssim = metric_to_float(

        ssim(
            reconstruction_batch,
            target_batch
        )

    )


    # ================================================================
    # GLOBAL UNCERTAINTY SUMMARY
    # ================================================================

    mean_aleatoric = float(
        aleatoric_variance
        .mean()
        .item()
    )


    mean_epistemic = float(
        epistemic_variance
        .mean()
        .item()
    )


    mean_predictive = float(
        predictive_variance
        .mean()
        .item()
    )


    mean_predictive_std = float(
        predictive_std
        .mean()
        .item()
    )


    # ================================================================
    # UNCERTAINTY IN MISSING REGION
    # ================================================================

    missing_aleatoric = float(
        aleatoric_variance[
            missing_selector
        ]
        .mean()
        .item()
    )


    missing_epistemic = float(
        epistemic_variance[
            missing_selector
        ]
        .mean()
        .item()
    )


    missing_predictive = float(
        predictive_variance[
            missing_selector
        ]
        .mean()
        .item()
    )


    missing_predictive_std = float(
        predictive_std[
            missing_selector
        ]
        .mean()
        .item()
    )


    # ================================================================
    # RESULT RECORD
    # ================================================================

    return {

        "method":
            "proposed_physics_informed_3d",

        "geological_mode":
            geological_mode,

        "mask_mode":
            mask_mode,

        "seed":
            int(seed),

        "requested_missing_rate":
            float(missing_rate),

        "measured_missing_rate":
            float(measured_missing_rate),

        "cube_depth":
            int(CUBE_SIZE[0]),

        "cube_height":
            int(CUBE_SIZE[1]),

        "cube_width":
            int(CUBE_SIZE[2]),

        "observed_samples":
            int(observed_samples),

        "missing_samples":
            int(missing_samples),

        "input_consistency_error":
            float(input_difference),

        "observed_preservation_error":
            float(observed_difference),

        "uncertainty_decomposition_error":
            float(decomposition_error),

        "runtime_seconds":
            float(runtime_seconds),

        "missing_mae":
            missing_mae,

        "missing_rmse":
            missing_rmse,

        "MAE":
            metric_mae,

        "RMSE":
            metric_rmse,

        "PSNR":
            metric_psnr,

        "SNR":
            metric_snr,

        "SSIM":
            metric_ssim,

        "aleatoric_variance_mean":
            mean_aleatoric,

        "epistemic_variance_mean":
            mean_epistemic,

        "predictive_variance_mean":
            mean_predictive,

        "predictive_std_mean":
            mean_predictive_std,

        "missing_aleatoric_variance_mean":
            missing_aleatoric,

        "missing_epistemic_variance_mean":
            missing_epistemic,

        "missing_predictive_variance_mean":
            missing_predictive,

        "missing_predictive_std_mean":
            missing_predictive_std,

        "status":
            "PASS",

        "error":
            ""

    }


# ====================================================================
# SUMMARY STATISTICS
# ====================================================================

def calculate_summary(records):
    """
    Calculate mean and standard deviation grouped by:

        geological_mode
        mask_mode
        requested_missing_rate
    """

    grouped = {}


    for record in records:

        key = (

            record[
                "geological_mode"
            ],

            record[
                "mask_mode"
            ],

            record[
                "requested_missing_rate"
            ]

        )


        grouped.setdefault(
            key,
            []
        ).append(
            record
        )


    metric_names = [

        "missing_mae",

        "missing_rmse",

        "MAE",

        "RMSE",

        "PSNR",

        "SNR",

        "SSIM",

        "runtime_seconds",

        "uncertainty_decomposition_error",

        "aleatoric_variance_mean",

        "epistemic_variance_mean",

        "predictive_variance_mean",

        "predictive_std_mean",

        "missing_aleatoric_variance_mean",

        "missing_epistemic_variance_mean",

        "missing_predictive_variance_mean",

        "missing_predictive_std_mean"

    ]


    summaries = []


    for key, group in grouped.items():

        (
            geological_mode,
            mask_mode,
            missing_rate
        ) = key


        summary = {

            "geological_mode":
                geological_mode,

            "mask_mode":
                mask_mode,

            "requested_missing_rate":
                missing_rate,

            "n_seeds":
                len(group)

        }


        for metric_name in metric_names:

            values = np.asarray(

                [

                    float(
                        record[
                            metric_name
                        ]
                    )

                    for record in group

                ],

                dtype=np.float64

            )


            summary[
                f"{metric_name}_mean"
            ] = float(
                values.mean()
            )


            summary[
                f"{metric_name}_std"
            ] = float(

                values.std(
                    ddof=1
                )

                if len(values) > 1
                else 0.0

            )


        summaries.append(
            summary
        )


    return summaries


# ====================================================================
# WRITE CSV
# ====================================================================

def write_csv(
    filename,
    records
):
    """
    Write records to a CSV file.
    """

    if not records:
        return


    fieldnames = list(
        records[0].keys()
    )


    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        writer.writeheader()


        writer.writerows(
            records
        )


# ====================================================================
# BUILD COMPLETE CONTROLLED CASE LIST
# ====================================================================

def build_controlled_cases():
    """
    Construct the complete deterministic controlled experimental
    matrix.
    """

    cases = []


    for geological_mode in GEOLOGICAL_MODES:

        for mask_mode in MASK_MODES:

            for missing_rate in MISSING_RATES:

                for seed in SEEDS:

                    cases.append(

                        (
                            geological_mode,
                            mask_mode,
                            missing_rate,
                            seed
                        )

                    )


    return cases


# ====================================================================
# SELECT CASES FOR EXECUTION
# ====================================================================

def select_cases_for_execution(
    cases
):
    """
    Select the controlled cases that will actually be executed.
    """

    full_matrix_size = len(cases)


    if CONTROLLED_MATRIX_CASE_LIMIT is None:

        return cases


    if not isinstance(
        CONTROLLED_MATRIX_CASE_LIMIT,
        int
    ):

        raise TypeError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "an integer or None."
        )


    if CONTROLLED_MATRIX_CASE_LIMIT < 1:

        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "greater than or equal to 1."
        )


    if (
        CONTROLLED_MATRIX_CASE_LIMIT
        >
        full_matrix_size
    ):

        raise ValueError(

            "CONTROLLED_MATRIX_CASE_LIMIT cannot exceed "
            f"the full matrix size of {full_matrix_size}."

        )


    return cases[
        :CONTROLLED_MATRIX_CASE_LIMIT
    ]


# ====================================================================
# MAIN CONTROLLED MATRIX
# ====================================================================

def main():

    print()
    print("=" * 78)
    print("PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX")
    print("=" * 78)


    # ================================================================
    # BUILD COMPLETE MATRIX
    # ================================================================

    cases = build_controlled_cases()


    expected_cases = (

        len(GEOLOGICAL_MODES)
        *
        len(MASK_MODES)
        *
        len(MISSING_RATES)
        *
        len(SEEDS)

    )


    # ================================================================
    # VERIFY COMPLETE MATRIX
    # ================================================================

    if len(cases) != expected_cases:

        raise RuntimeError(

            "Controlled matrix generation error: "

            f"expected {expected_cases} cases, "

            f"generated {len(cases)} cases."

        )


    # ================================================================
    # EXPLICIT SCIENTIFIC DESIGN CHECK
    # ================================================================

    if expected_cases != 750:

        raise RuntimeError(

            "The proposed-model controlled matrix must contain "
            f"exactly 750 cases, but {expected_cases} were generated."

        )


    # ================================================================
    # SELECT EXECUTION CASES
    # ================================================================

    cases_to_run = select_cases_for_execution(
        cases
    )


    cases_to_execute = len(
        cases_to_run
    )


    # ================================================================
    # EXECUTION MODE
    # ================================================================

    if CONTROLLED_MATRIX_CASE_LIMIT is None:

        execution_mode = (
            "FULL CONTROLLED MATRIX"
        )

    else:

        execution_mode = (
            "SMOKE TEST"
        )


    # ================================================================
    # EXPERIMENTAL CONFIGURATION REPORT
    # ================================================================

    print()
    print(
        f"Full matrix cases : "
        f"{expected_cases}"
    )

    print(
        f"Cases to run      : "
        f"{cases_to_execute}"
    )

    print(
        f"Execution mode    : "
        f"{execution_mode}"
    )

    print()
    print(
        f"Cube size         : "
        f"{CUBE_SIZE}"
    )

    print(
        f"Device            : "
        f"{DEVICE}"
    )

    print(
        f"MC samples        : "
        f"{MC_SAMPLES}"
    )

    print()
    print("Checkpoint:")
    print(CHECKPOINT)

    print()
    print("Results directory:")
    print(REPORT_DIR)


    # ================================================================
    # VERIFY CHECKPOINT BEFORE STARTING CASES
    # ================================================================

    if not CHECKPOINT.is_file():

        raise FileNotFoundError(

            "Required proposed-model checkpoint was not found:\n"

            f"{CHECKPOINT}"

        )


    # ================================================================
    # LOAD MODEL ONCE
    # ================================================================

    model = load_model()


    # ================================================================
    # RESULTS
    # ================================================================

    records = []

    completed_cases = 0

    successful_cases = 0

    failed_cases = 0


    # ================================================================
    # CONTROLLED CASE EXECUTION
    # ================================================================

    for (
        geological_mode,
        mask_mode,
        missing_rate,
        seed
    ) in cases_to_run:

        completed_cases += 1


        print()
        print("-" * 78)


        print(
            f"Case "
            f"{completed_cases}/"
            f"{cases_to_execute}"
        )


        print(
            f"Geology       : "
            f"{geological_mode}"
        )


        print(
            f"Mask          : "
            f"{mask_mode}"
        )


        print(
            f"Missing rate  : "
            f"{missing_rate:.0%}"
        )


        print(
            f"Seed          : "
            f"{seed}"
        )


        # ============================================================
        # RUN CASE
        # ============================================================

        try:

            result = run_single_experiment(

                model=model,

                geological_mode=
                    geological_mode,

                mask_mode=
                    mask_mode,

                missing_rate=
                    missing_rate,

                seed=
                    seed

            )


            successful_cases += 1


            print()
            print("Status        : SUCCESS")


            print(
                f"Missing MAE   : "
                f"{result['missing_mae']:.6f}"
            )


            print(
                f"Missing RMSE  : "
                f"{result['missing_rmse']:.6f}"
            )


            print(
                f"MAE           : "
                f"{result['MAE']:.6f}"
            )


            print(
                f"RMSE          : "
                f"{result['RMSE']:.6f}"
            )


            print(
                f"PSNR          : "
                f"{result['PSNR']:.6f} dB"
            )


            print(
                f"SNR           : "
                f"{result['SNR']:.6f} dB"
            )


            print(
                f"SSIM          : "
                f"{result['SSIM']:.6f}"
            )


            print(
                f"Aleatoric var : "
                f"{result['aleatoric_variance_mean']:.6e}"
            )


            print(
                f"Epistemic var : "
                f"{result['epistemic_variance_mean']:.6e}"
            )


            print(
                f"Predictive var: "
                f"{result['predictive_variance_mean']:.6e}"
            )


            print(
                f"Predictive σ  : "
                f"{result['predictive_std_mean']:.6e}"
            )


            print(
                f"Runtime       : "
                f"{result['runtime_seconds']:.4f} s"
            )


            print(
                f"Observed err  : "
                f"{result['observed_preservation_error']:.6e}"
            )


        # ============================================================
        # CASE FAILURE
        # ============================================================

        except Exception as exc:

            failed_cases += 1


            print()
            print("Status        : FAILED")


            print(
                f"Error         : "
                f"{exc}"
            )


            result = {

                "method":
                    "proposed_physics_informed_3d",

                "geological_mode":
                    geological_mode,

                "mask_mode":
                    mask_mode,

                "seed":
                    int(seed),

                "requested_missing_rate":
                    float(missing_rate),

                "measured_missing_rate":
                    np.nan,

                "cube_depth":
                    int(CUBE_SIZE[0]),

                "cube_height":
                    int(CUBE_SIZE[1]),

                "cube_width":
                    int(CUBE_SIZE[2]),

                "observed_samples":
                    0,

                "missing_samples":
                    0,

                "input_consistency_error":
                    np.nan,

                "observed_preservation_error":
                    np.nan,

                "uncertainty_decomposition_error":
                    np.nan,

                "runtime_seconds":
                    np.nan,

                "missing_mae":
                    np.nan,

                "missing_rmse":
                    np.nan,

                "MAE":
                    np.nan,

                "RMSE":
                    np.nan,

                "PSNR":
                    np.nan,

                "SNR":
                    np.nan,

                "SSIM":
                    np.nan,

                "aleatoric_variance_mean":
                    np.nan,

                "epistemic_variance_mean":
                    np.nan,

                "predictive_variance_mean":
                    np.nan,

                "predictive_std_mean":
                    np.nan,

                "missing_aleatoric_variance_mean":
                    np.nan,

                "missing_epistemic_variance_mean":
                    np.nan,

                "missing_predictive_variance_mean":
                    np.nan,

                "missing_predictive_std_mean":
                    np.nan,

                "status":
                    "FAILED",

                "error":
                    str(exc)

            }


        # ============================================================
        # STORE RESULT
        # ============================================================

        records.append(
            result
        )


    # ================================================================
    # WRITE RAW RESULTS
    # ================================================================

    write_csv(
        RESULTS_FILE,
        records
    )


    # ================================================================
    # SUCCESSFUL RECORDS
    # ================================================================

    successful_records = [

        record

        for record in records

        if record["status"] == "PASS"

    ]


    # ================================================================
    # SUMMARY
    # ================================================================

    summaries = calculate_summary(
        successful_records
    )


    write_csv(
        SUMMARY_FILE,
        summaries
    )


    # ================================================================
    # FINAL REPORT
    # ================================================================

    print()
    print("=" * 78)
    print("PROPOSED MODEL CONTROLLED MATRIX COMPLETE")
    print("=" * 78)


    print()
    print(
        f"Full matrix cases : "
        f"{expected_cases}"
    )


    print(
        f"Cases to run      : "
        f"{cases_to_execute}"
    )


    print(
        f"Completed cases   : "
        f"{completed_cases}"
    )


    print(
        f"Successful        : "
        f"{successful_cases}"
    )


    print(
        f"Failed            : "
        f"{failed_cases}"
    )


    print()
    print(
        f"Execution mode    : "
        f"{execution_mode}"
    )


    print()
    print("Raw results:")
    print(RESULTS_FILE)


    print()
    print("Summary results:")
    print(SUMMARY_FILE)


    # ================================================================
    # FINAL STATUS
    # ================================================================

    if (
        completed_cases == cases_to_execute
        and
        successful_cases == cases_to_execute
        and
        failed_cases == 0
    ):

        print()
        print("OVERALL STATUS: PASS")


        if CONTROLLED_MATRIX_CASE_LIMIT is None:

            print(
                "All controlled proposed-model "
                "experiments completed successfully."
            )

        else:

            print(
                "All smoke-test controlled proposed-model "
                "experiments completed successfully."
            )


    else:

        print()
        print("OVERALL STATUS: FAIL")


        print(
            "One or more controlled "
            "experiments failed."
        )


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()