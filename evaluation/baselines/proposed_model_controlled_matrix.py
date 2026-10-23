"""
====================================================================
PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate the FROZEN proposed acoustic-wave model over the same
controlled experimental matrix used for the six classical baselines.

Experimental matrix:
    6 geological modes
    5 missing-data mechanisms
    5 missing rates
    5 seeds

Total: 750 cases.

The complete matrix always contains 750 cases. Execution can be
limited through CONTROLLED_MATRIX_CASE_LIMIT in utils/config.py.

The neural network is not retrained for each case.

The trained best_model.pth is evaluated across all selected cases.
Observed samples are restored exactly after inference.

Predictive uncertainty:
    Aleatoric variance
    Epistemic variance
    Predictive variance
    Predictive standard deviation

The acoustic pressure wavefield is not retained across MC samples,
avoiding unnecessary memory consumption.

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

from dataset.synthetic_dataset import SyntheticSeismicDataset

from models.network import Network3D

from models.mc_dropout import MCDropout3D

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
    ACOUSTIC_TIME_STEPS,
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

CHECKPOINT = Path(CHECKPOINT_DIR) / "best_model.pth"


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
# CONFIGURATION
# ====================================================================

CUBE_SIZE = tuple(BASELINE_CUBE_SIZE)

NUM_SAMPLES = int(BASELINE_NUM_SAMPLES)

MC_SAMPLES = int(MC_DROPOUT_SAMPLES)

REPORT_DIR = Path(REPORT_DIR)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = (
    REPORT_DIR / "proposed_model_controlled_matrix.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "proposed_model_controlled_matrix_summary.csv"
)

PRESERVATION_TOLERANCE = float(
    OBSERVED_PRESERVATION_TOLERANCE
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ====================================================================
# REPRODUCIBILITY
# ====================================================================

def set_seed(seed):
    """Set random seeds for reproducible controlled experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================
# DEVICE COMPARISON
# ====================================================================

def devices_match(tensor_device, expected_device):
    """
    Compare PyTorch devices while treating cuda and cuda:0 as
    equivalent when they refer to the same current CUDA device.
    """

    actual = torch.device(tensor_device)
    expected = torch.device(expected_device)

    if actual.type != expected.type:
        return False

    if actual.type == "cpu":
        return True

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

        return actual_index == expected_index

    return actual == expected


# ====================================================================
# METRIC CONVERSION
# ====================================================================

def metric_to_float(value):
    """Convert a scalar tensor, NumPy scalar, or numeric value to float."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        return float(value.item())

    return float(value)


# ====================================================================
# FINITE CHECK
# ====================================================================

def tensor_is_finite(tensor):
    """Return True if every tensor value is finite."""

    return bool(torch.isfinite(tensor).all().item())


# ====================================================================
# LOAD FROZEN ACOUSTIC MODEL
# ====================================================================

def load_model():
    """
    Load the trained acoustic-wave Network3D checkpoint.

    IMPORTANT:
    ACOUSTIC_TIME_STEPS must match the architecture used during
    training so that the checkpoint can be loaded consistently.
    """

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "\nProposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}\n\n"
            "The trained best_model.pth must exist before running "
            "the controlled experimental matrix."
        )

    print()
    print("=" * 70)
    print("LOADING FROZEN ACOUSTIC PROPOSED MODEL")
    print("=" * 70)

    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print(f"Acoustic time steps: {ACOUSTIC_TIME_STEPS}")

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
        pressure_time_steps=ACOUSTIC_TIME_STEPS,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unexpected checkpoint format. Expected a dictionary "
            "containing 'model_state_dict'."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nProposed acoustic model loaded successfully.")

    if "best_epoch" in checkpoint:
        print("Best epoch:", checkpoint["best_epoch"])

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
    """Run one frozen-model controlled reconstruction experiment."""

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------

    set_seed(seed)

    # ---------------------------------------------------------------
    # Create controlled synthetic dataset
    # ---------------------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=missing_rate,
        geological_mode=geological_mode,
        mask_mode=mask_mode,
        seed=seed,
    )

    (
        corrupted,
        target,
        mask,
        velocity,
        pressure_wavefield,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    # ---------------------------------------------------------------
    # Metadata validation
    # ---------------------------------------------------------------

    if actual_geological_mode != geological_mode:
        raise RuntimeError(
            "Geological mode mismatch: "
            f"expected={geological_mode}, "
            f"received={actual_geological_mode}"
        )

    if actual_mask_mode != mask_mode:
        raise RuntimeError(
            "Mask mode mismatch: "
            f"expected={mask_mode}, received={actual_mask_mode}"
        )

    # ---------------------------------------------------------------
    # Shape validation
    # Dataset convention: [C,D,H,W]
    # Network convention: [B,C,D,H,W]
    # ---------------------------------------------------------------

    expected_shape = (1, *CUBE_SIZE)

    for name, tensor in {
        "corrupted": corrupted,
        "target": target,
        "mask": mask,
        "velocity": velocity,
    }.items():

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected {name} shape: "
                f"expected={expected_shape}, "
                f"received={tuple(tensor.shape)}"
            )

        if not tensor_is_finite(tensor):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    # ---------------------------------------------------------------
    # Mask validation
    # ---------------------------------------------------------------

    unique_values = torch.unique(mask)

    if not torch.all(
        (unique_values == 0) | (unique_values == 1)
    ):
        raise RuntimeError("Mask must contain only 0 and 1.")

    # ---------------------------------------------------------------
    # Input consistency
    # ---------------------------------------------------------------

    expected_input = target * mask

    input_difference = torch.max(
        torch.abs(corrupted - expected_input)
    ).item()

    if input_difference > PRESERVATION_TOLERANCE:
        raise RuntimeError(
            "Input consistency check failed: "
            f"{input_difference:.6e}"
        )

    # ---------------------------------------------------------------
    # Sample counts
    # ---------------------------------------------------------------

    observed_samples = int((mask == 1).sum().item())
    missing_samples = int((mask == 0).sum().item())

    if missing_samples == 0:
        raise RuntimeError("No missing samples detected.")

    total_samples = observed_samples + missing_samples

    measured_missing_rate = missing_samples / total_samples

    # ---------------------------------------------------------------
    # Move tensors to the model device
    # ---------------------------------------------------------------

    corrupted = corrupted.to(DEVICE)
    target = target.to(DEVICE)
    mask = mask.to(DEVICE)
    velocity = velocity.to(DEVICE)

    for name, tensor in {
        "corrupted": corrupted,
        "target": target,
        "mask": mask,
        "velocity": velocity,
    }.items():

        if not devices_match(tensor.device, DEVICE):
            raise RuntimeError(
                f"Device mismatch for {name}: "
                f"expected={DEVICE}, received={tensor.device}"
            )

    # ---------------------------------------------------------------
    # Prepare network input
    # ---------------------------------------------------------------

    input_batch = corrupted.unsqueeze(0)

    # ---------------------------------------------------------------
    # MC-dropout inference
    #
    # The updated MCDropout3D does not retain pressure-wavefield
    # samples by default. Only reconstruction and log-variance
    # samples are required for the metrics below.
    # ---------------------------------------------------------------

    predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    start_time = time.perf_counter()

    with torch.no_grad():
        predictions = predictor.predict(input_batch)

    runtime_seconds = time.perf_counter() - start_time

    # ---------------------------------------------------------------
    # Required MC outputs
    # ---------------------------------------------------------------

    required_keys = {
        "reconstruction_samples",
        "log_variance_samples",
    }

    missing_keys = required_keys - predictions.keys()

    if missing_keys:
        raise KeyError(f"Missing MC outputs: {missing_keys}")

    reconstruction_samples = predictions["reconstruction_samples"]
    log_variance_samples = predictions["log_variance_samples"]

    if not isinstance(reconstruction_samples, torch.Tensor):
        raise TypeError("reconstruction_samples must be a torch.Tensor.")

    if not isinstance(log_variance_samples, torch.Tensor):
        raise TypeError("log_variance_samples must be a torch.Tensor.")

    # ---------------------------------------------------------------
    # Expected MC tensor shape:
    # [MC_SAMPLES, B, C, D, H, W]
    # ---------------------------------------------------------------

    expected_mc_shape = (
        MC_SAMPLES,
        1,
        1,
        *CUBE_SIZE,
    )

    if tuple(reconstruction_samples.shape) != expected_mc_shape:
        raise RuntimeError(
            "Reconstruction MC shape mismatch: "
            f"expected={expected_mc_shape}, "
            f"received={tuple(reconstruction_samples.shape)}"
        )

    if tuple(log_variance_samples.shape) != expected_mc_shape:
        raise RuntimeError(
            "Log-variance MC shape mismatch: "
            f"expected={expected_mc_shape}, "
            f"received={tuple(log_variance_samples.shape)}"
        )

    for name, tensor in {
        "reconstruction_samples": reconstruction_samples,
        "log_variance_samples": log_variance_samples,
    }.items():

        if not devices_match(tensor.device, DEVICE):
            raise RuntimeError(
                f"{name} is on {tensor.device}; expected {DEVICE}."
            )

        if not tensor_is_finite(tensor):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    # ---------------------------------------------------------------
    # MC reconstruction mean
    # Result shape: [B,C,D,H,W]
    # ---------------------------------------------------------------

    reconstruction_mean = reconstruction_samples.mean(dim=0)

    # ---------------------------------------------------------------
    # Predictive uncertainty
    # ---------------------------------------------------------------

    uncertainty_estimator = PredictiveUncertaintyEstimator(
        min_log_variance=LOG_VARIANCE_MIN,
        max_log_variance=LOG_VARIANCE_MAX,
    )

    uncertainty_results = uncertainty_estimator(
        log_variance_samples,
        reconstruction_samples,
    )

    required_uncertainty_keys = {
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
    }

    missing_uncertainty_keys = (
        required_uncertainty_keys - uncertainty_results.keys()
    )

    if missing_uncertainty_keys:
        raise KeyError(
            "Missing predictive-uncertainty outputs: "
            f"{missing_uncertainty_keys}"
        )

    aleatoric_variance = uncertainty_results["aleatoric_variance"]
    epistemic_variance = uncertainty_results["epistemic_variance"]
    predictive_variance = uncertainty_results["predictive_variance"]
    predictive_std = uncertainty_results["predictive_std"]

    # ---------------------------------------------------------------
    # Uncertainty validation
    # ---------------------------------------------------------------

    expected_uncertainty_shape = tuple(reconstruction_mean.shape)

    uncertainty_tensors = {
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "predictive_variance": predictive_variance,
        "predictive_std": predictive_std,
    }

    for name, tensor in uncertainty_tensors.items():

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if tuple(tensor.shape) != expected_uncertainty_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected={expected_uncertainty_shape}, "
                f"received={tuple(tensor.shape)}"
            )

        if not devices_match(tensor.device, DEVICE):
            raise RuntimeError(
                f"{name} is on {tensor.device}; expected {DEVICE}."
            )

        if not tensor_is_finite(tensor):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    # ---------------------------------------------------------------
    # Non-negative uncertainty checks
    # ---------------------------------------------------------------

    for name in (
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "predictive_std",
    ):

        if (uncertainty_tensors[name] < 0).any():
            raise RuntimeError(f"Negative values detected in {name}.")

    # ---------------------------------------------------------------
    # Predictive variance decomposition:
    # predictive variance = aleatoric + epistemic variance
    # ---------------------------------------------------------------

    decomposition_error = torch.max(
        torch.abs(
            predictive_variance
            - (aleatoric_variance + epistemic_variance)
        )
    ).item()

    if decomposition_error > 1.0e-5:
        raise RuntimeError(
            "Predictive uncertainty decomposition failed: "
            f"maximum difference={decomposition_error:.6e}"
        )

    # ---------------------------------------------------------------
    # Exact observed-data preservation
    #
    # mask = 1: observed samples
    # mask = 0: missing samples
    # ---------------------------------------------------------------

    mask_batch = mask.unsqueeze(0)

    reconstruction = (
        reconstruction_mean * (1.0 - mask_batch)
        + corrupted.unsqueeze(0) * mask_batch
    )

    if not tensor_is_finite(reconstruction):
        raise RuntimeError("Final reconstruction contains NaN or Inf.")

    observed_selector = mask_batch == 1

    observed_difference = torch.max(
        torch.abs(
            reconstruction[observed_selector]
            - corrupted.unsqueeze(0)[observed_selector]
        )
    ).item()

    if observed_difference > PRESERVATION_TOLERANCE:
        raise RuntimeError(
            "Observed-data preservation failed: "
            f"{observed_difference:.6e}"
        )

    # ---------------------------------------------------------------
    # Missing-region metrics
    # ---------------------------------------------------------------

    missing_selector = mask_batch == 0

    missing_prediction = reconstruction[missing_selector]
    missing_target = target.unsqueeze(0)[missing_selector]

    if missing_prediction.numel() == 0:
        raise RuntimeError("Missing-region selector contains no elements.")

    missing_error = missing_prediction - missing_target

    missing_mae = float(missing_error.abs().mean().item())

    missing_rmse = float(
        torch.sqrt((missing_error ** 2).mean()).item()
    )

    # ---------------------------------------------------------------
    # Global reconstruction metrics
    # ---------------------------------------------------------------

    target_batch = target.unsqueeze(0)
    reconstruction_batch = reconstruction

    metric_mae = metric_to_float(
        mae(reconstruction_batch, target_batch)
    )

    metric_rmse = metric_to_float(
        rmse(reconstruction_batch, target_batch)
    )

    metric_psnr = metric_to_float(
        psnr(reconstruction_batch, target_batch)
    )

    metric_snr = metric_to_float(
        snr(reconstruction_batch, target_batch)
    )

    metric_ssim = metric_to_float(
        ssim(reconstruction_batch, target_batch)
    )

    # ---------------------------------------------------------------
    # Global uncertainty summaries
    # ---------------------------------------------------------------

    mean_aleatoric = float(aleatoric_variance.mean().item())
    mean_epistemic = float(epistemic_variance.mean().item())
    mean_predictive = float(predictive_variance.mean().item())
    mean_predictive_std = float(predictive_std.mean().item())

    # ---------------------------------------------------------------
    # Missing-region uncertainty summaries
    # ---------------------------------------------------------------

    missing_aleatoric = float(
        aleatoric_variance[missing_selector].mean().item()
    )

    missing_epistemic = float(
        epistemic_variance[missing_selector].mean().item()
    )

    missing_predictive = float(
        predictive_variance[missing_selector].mean().item()
    )

    missing_predictive_std = float(
        predictive_std[missing_selector].mean().item()
    )

    # ---------------------------------------------------------------
    # Return result record
    # ---------------------------------------------------------------

    return {
        "method": "proposed_physics_informed_3d",
        "geological_mode": geological_mode,
        "mask_mode": mask_mode,
        "seed": int(seed),
        "requested_missing_rate": float(missing_rate),
        "measured_missing_rate": float(measured_missing_rate),
        "cube_depth": int(CUBE_SIZE[0]),
        "cube_height": int(CUBE_SIZE[1]),
        "cube_width": int(CUBE_SIZE[2]),
        "observed_samples": int(observed_samples),
        "missing_samples": int(missing_samples),
        "input_consistency_error": float(input_difference),
        "observed_preservation_error": float(observed_difference),
        "uncertainty_decomposition_error": float(decomposition_error),
        "runtime_seconds": float(runtime_seconds),
        "missing_mae": missing_mae,
        "missing_rmse": missing_rmse,
        "MAE": metric_mae,
        "RMSE": metric_rmse,
        "PSNR": metric_psnr,
        "SNR": metric_snr,
        "SSIM": metric_ssim,
        "aleatoric_variance_mean": mean_aleatoric,
        "epistemic_variance_mean": mean_epistemic,
        "predictive_variance_mean": mean_predictive,
        "predictive_std_mean": mean_predictive_std,
        "missing_aleatoric_variance_mean": missing_aleatoric,
        "missing_epistemic_variance_mean": missing_epistemic,
        "missing_predictive_variance_mean": missing_predictive,
        "missing_predictive_std_mean": missing_predictive_std,
        "status": "PASS",
        "error": "",
    }


# ====================================================================
# SUMMARY STATISTICS
# ====================================================================

def calculate_summary(records):
    """
    Calculate means and standard deviations grouped by geology,
    mask mode, and requested missing rate.
    """

    grouped = {}

    for record in records:
        key = (
            record["geological_mode"],
            record["mask_mode"],
            record["requested_missing_rate"],
        )

        grouped.setdefault(key, []).append(record)

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
        "missing_predictive_std_mean",
    ]

    summaries = []

    for key, group in grouped.items():

        geological_mode, mask_mode, missing_rate = key

        summary = {
            "geological_mode": geological_mode,
            "mask_mode": mask_mode,
            "requested_missing_rate": missing_rate,
            "n_seeds": len(group),
        }

        for metric_name in metric_names:

            values = np.asarray(
                [float(record[metric_name]) for record in group],
                dtype=np.float64,
            )

            summary[f"{metric_name}_mean"] = float(values.mean())

            summary[f"{metric_name}_std"] = float(
                values.std(ddof=1) if len(values) > 1 else 0.0
            )

        summaries.append(summary)

    return summaries


# ====================================================================
# WRITE CSV
# ====================================================================

def write_csv(filename, records):
    """Write records to CSV without creating an empty file."""

    if not records:
        print(f"No records to write to {filename}")
        return

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)


# ====================================================================
# BUILD COMPLETE CONTROLLED CASE LIST
# ====================================================================

def build_controlled_cases():
    """Construct all 750 deterministic controlled cases."""

    return [
        (
            geological_mode,
            mask_mode,
            missing_rate,
            seed,
        )
        for geological_mode in GEOLOGICAL_MODES
        for mask_mode in MASK_MODES
        for missing_rate in MISSING_RATES
        for seed in SEEDS
    ]


# ====================================================================
# SELECT CASES FOR EXECUTION
# ====================================================================

def select_cases_for_execution(cases):
    """
    Apply the centrally configured execution limit.

    None means all cases. A positive integer runs the first N cases.
    """

    full_matrix_size = len(cases)
    limit = CONTROLLED_MATRIX_CASE_LIMIT

    if limit is None:
        return cases

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be an integer or None."
        )

    if limit < 1:
        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be at least 1."
        )

    if limit > full_matrix_size:
        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT cannot exceed "
            f"{full_matrix_size}."
        )

    return cases[:limit]


# ====================================================================
# FAILED CASE RECORD
# ====================================================================

def make_failed_record(
    geological_mode,
    mask_mode,
    missing_rate,
    seed,
    error,
):
    """Create a failed-case record with the same columns as a success."""

    return {
        "method": "proposed_physics_informed_3d",
        "geological_mode": geological_mode,
        "mask_mode": mask_mode,
        "seed": int(seed),
        "requested_missing_rate": float(missing_rate),
        "measured_missing_rate": np.nan,
        "cube_depth": int(CUBE_SIZE[0]),
        "cube_height": int(CUBE_SIZE[1]),
        "cube_width": int(CUBE_SIZE[2]),
        "observed_samples": 0,
        "missing_samples": 0,
        "input_consistency_error": np.nan,
        "observed_preservation_error": np.nan,
        "uncertainty_decomposition_error": np.nan,
        "runtime_seconds": np.nan,
        "missing_mae": np.nan,
        "missing_rmse": np.nan,
        "MAE": np.nan,
        "RMSE": np.nan,
        "PSNR": np.nan,
        "SNR": np.nan,
        "SSIM": np.nan,
        "aleatoric_variance_mean": np.nan,
        "epistemic_variance_mean": np.nan,
        "predictive_variance_mean": np.nan,
        "predictive_std_mean": np.nan,
        "missing_aleatoric_variance_mean": np.nan,
        "missing_epistemic_variance_mean": np.nan,
        "missing_predictive_variance_mean": np.nan,
        "missing_predictive_std_mean": np.nan,
        "status": "FAILED",
        "error": str(error),
    }


# ====================================================================
# MAIN CONTROLLED MATRIX
# ====================================================================

def main():

    print()
    print("=" * 78)
    print("PROPOSED MODEL CONTROLLED EXPERIMENTAL MATRIX")
    print("=" * 78)

    # ---------------------------------------------------------------
    # Build and verify the complete matrix
    # ---------------------------------------------------------------

    cases = build_controlled_cases()

    expected_cases = (
        len(GEOLOGICAL_MODES)
        * len(MASK_MODES)
        * len(MISSING_RATES)
        * len(SEEDS)
    )

    if len(cases) != expected_cases:
        raise RuntimeError(
            f"Expected {expected_cases} cases; generated {len(cases)}."
        )

    if expected_cases != 750:
        raise RuntimeError(
            "The complete proposed-model matrix must contain "
            f"exactly 750 cases, but {expected_cases} were generated."
        )

    # ---------------------------------------------------------------
    # Apply central execution limit
    # ---------------------------------------------------------------

    cases_to_run = select_cases_for_execution(cases)
    cases_to_execute = len(cases_to_run)

    execution_mode = (
        "FULL CONTROLLED MATRIX"
        if CONTROLLED_MATRIX_CASE_LIMIT is None
        else "SMOKE TEST"
    )

    print(f"\nFull matrix cases : {expected_cases}")
    print(f"Cases to run      : {cases_to_execute}")
    print(f"Execution mode    : {execution_mode}")
    print(f"Cube size         : {CUBE_SIZE}")
    print(f"Device            : {DEVICE}")
    print(f"MC samples        : {MC_SAMPLES}")
    print(f"Acoustic steps    : {ACOUSTIC_TIME_STEPS}")
    print(f"\nCheckpoint:\n{CHECKPOINT}")
    print(f"\nResults directory:\n{REPORT_DIR}")

    if MC_SAMPLES < 1:
        raise ValueError("MC_DROPOUT_SAMPLES must be at least 1.")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Required proposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}"
        )

    # ---------------------------------------------------------------
    # Load the frozen model once
    # ---------------------------------------------------------------

    model = load_model()

    records = []
    completed_cases = 0
    successful_cases = 0
    failed_cases = 0

    # ---------------------------------------------------------------
    # Execute selected cases
    # ---------------------------------------------------------------

    for (
        geological_mode,
        mask_mode,
        missing_rate,
        seed,
    ) in cases_to_run:

        completed_cases += 1

        print()
        print("-" * 78)
        print(f"Case          : {completed_cases}/{cases_to_execute}")
        print(f"Geology       : {geological_mode}")
        print(f"Mask          : {mask_mode}")
        print(f"Missing rate  : {missing_rate:.0%}")
        print(f"Seed          : {seed}")

        try:

            result = run_single_experiment(
                model=model,
                geological_mode=geological_mode,
                mask_mode=mask_mode,
                missing_rate=missing_rate,
                seed=seed,
            )

            successful_cases += 1

            print("\nStatus        : SUCCESS")
            print(f"Missing MAE   : {result['missing_mae']:.6f}")
            print(f"Missing RMSE  : {result['missing_rmse']:.6f}")
            print(f"MAE           : {result['MAE']:.6f}")
            print(f"RMSE          : {result['RMSE']:.6f}")
            print(f"PSNR          : {result['PSNR']:.6f} dB")
            print(f"SNR           : {result['SNR']:.6f} dB")
            print(f"SSIM          : {result['SSIM']:.6f}")
            print(
                "Aleatoric var : "
                f"{result['aleatoric_variance_mean']:.6e}"
            )
            print(
                "Epistemic var : "
                f"{result['epistemic_variance_mean']:.6e}"
            )
            print(
                "Predictive var: "
                f"{result['predictive_variance_mean']:.6e}"
            )
            print(
                "Predictive std: "
                f"{result['predictive_std_mean']:.6e}"
            )
            print(f"Runtime       : {result['runtime_seconds']:.4f} s")
            print(
                "Observed err  : "
                f"{result['observed_preservation_error']:.6e}"
            )

        except Exception as exc:

            failed_cases += 1

            print("\nStatus        : FAILED")
            print(f"Error         : {exc}")

            result = make_failed_record(
                geological_mode=geological_mode,
                mask_mode=mask_mode,
                missing_rate=missing_rate,
                seed=seed,
                error=exc,
            )

        records.append(result)

        # Save progress after each case. If a later case fails
        # unexpectedly, completed results remain available.
        write_csv(RESULTS_FILE, records)

    # ---------------------------------------------------------------
    # Write final raw results and successful-case summary
    # ---------------------------------------------------------------

    write_csv(RESULTS_FILE, records)

    successful_records = [
        record for record in records
        if record["status"] == "PASS"
    ]

    summaries = calculate_summary(successful_records)
    write_csv(SUMMARY_FILE, summaries)

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print()
    print("=" * 78)
    print("PROPOSED MODEL CONTROLLED MATRIX COMPLETE")
    print("=" * 78)

    print(f"\nFull matrix cases : {expected_cases}")
    print(f"Cases to run      : {cases_to_execute}")
    print(f"Completed cases   : {completed_cases}")
    print(f"Successful        : {successful_cases}")
    print(f"Failed            : {failed_cases}")
    print(f"Execution mode    : {execution_mode}")

    print(f"\nRaw results:\n{RESULTS_FILE}")
    print(f"\nSummary results:\n{SUMMARY_FILE}")

    if (
        completed_cases == cases_to_execute
        and successful_cases == cases_to_execute
        and failed_cases == 0
    ):

        print("\nOVERALL STATUS: PASS")

        if CONTROLLED_MATRIX_CASE_LIMIT is None:
            print(
                "All 750 controlled proposed-model experiments "
                "completed successfully."
            )
        else:
            print(
                "All selected smoke-test experiments "
                "completed successfully."
            )

    else:

        print("\nOVERALL STATUS: FAIL")
        print("One or more controlled experiments failed.")


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()