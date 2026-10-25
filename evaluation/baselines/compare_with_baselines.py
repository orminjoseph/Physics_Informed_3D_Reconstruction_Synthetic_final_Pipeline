"""
======================================================================
COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Perform a common side-by-side comparison of six classical
reconstruction baselines and the proposed Physics-Informed
3D Encoder-Decoder model.

Methods
-------
1. Nearest Neighbor
2. Linear Interpolation
3. f-x Prediction
4. Compressive Sensing
5. Curvelet POCS
6. Dictionary Learning
7. Proposed Physics-Informed 3D Encoder-Decoder

Important
---------
This is ONE common comparison case, NOT the 750-case
controlled experiment.

All seven methods receive the same corrupted cube and mask.
Ground truth is used only for post-reconstruction evaluation.

The proposed acoustic model returns:
    reconstructed_cube
    pressure_wavefield
    log_variance

MC-Dropout retains reconstruction and log-variance samples only.
Full pressure-wavefield samples are not stacked or retained.

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch


# =====================================================================
# DATASET
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset,
)


# =====================================================================
# CLASSICAL BASELINES
# =====================================================================

from evaluation.baselines.baseline_nearest_neighbor_controlled_matrix import (
    nearest_neighbor_reconstruction,
)

from evaluation.baselines.baseline_linear_interpolation_controlled_matrix import (
    linear_interpolation_reconstruction,
)

from evaluation.baselines.fx_controlled_matrix import (
    fx_prediction_reconstruction,
)

from evaluation.baselines.compressive_sensing_controlled_matrix import (
    compressive_sensing_reconstruction,
)

from evaluation.baselines.curvelet_pocs_controlled_matrix import (
    curvelet_pocs_reconstruction,
)

from evaluation.baselines.dictionary_learning_controlled_matrix import (
    dictionary_learning_reconstruction,
)


# =====================================================================
# PROPOSED MODEL
# =====================================================================

from models.network import Network3D

from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)


# =====================================================================
# CENTRALIZED METRICS
# =====================================================================

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)


# =====================================================================
# CENTRALIZED CONFIGURATION
# =====================================================================

from utils import config

from utils.config import (
    BASELINE_NUM_SAMPLES,
    BASELINE_CUBE_SIZE,
    BASELINE_MISSING_RATE,
    BASELINE_GEOLOGICAL_MODE,
    BASELINE_MASK_MODE,
    BASELINE_SEED,
    OBSERVED_PRESERVATION_TOLERANCE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MC_DROPOUT_SAMPLES,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
    DEVICE as CONFIG_DEVICE,
    CHECKPOINT_DIR,
    REPORT_DIR as CONFIG_REPORT_DIR,
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

# This script is located in evaluation/.
# parents[0] = evaluation/
# parents[1] = project root/
PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# =====================================================================
# COMMON COMPARISON CONFIGURATION
# =====================================================================

CUBE_SIZE = tuple(BASELINE_CUBE_SIZE)

NUM_SAMPLES = BASELINE_NUM_SAMPLES

MISSING_RATE = BASELINE_MISSING_RATE

GEOLOGICAL_MODE = BASELINE_GEOLOGICAL_MODE

MASK_MODE = BASELINE_MASK_MODE

SEED = BASELINE_SEED


# =====================================================================
# PROPOSED MODEL CHECKPOINT
# =====================================================================

CHECKPOINT = (
    PROJECT_ROOT
    / Path(CHECKPOINT_DIR)
    / "best_model.pth"
)


# =====================================================================
# MC-DROPOUT CONFIGURATION
# =====================================================================

MC_SAMPLES = MC_DROPOUT_SAMPLES


# =====================================================================
# OUTPUT DIRECTORY
# =====================================================================

REPORT_DIR = (
    PROJECT_ROOT
    / Path(CONFIG_REPORT_DIR)
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# OUTPUT FILES
# =====================================================================

RESULTS_FILE = (
    REPORT_DIR
    / "baseline_comparison.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "baseline_comparison_summary.csv"
)


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device(configured_device) -> torch.device:
    """
    Resolve the centralized DEVICE configuration.

    Supported values:
        auto
        cpu
        cuda
    """

    configured_device = str(configured_device).lower()

    if configured_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if configured_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' is configured, but CUDA is unavailable."
            )

        return torch.device("cuda")

    if configured_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Invalid DEVICE configuration: {configured_device!r}. "
        "Expected 'auto', 'cpu', or 'cuda'."
    )


DEVICE = resolve_device(CONFIG_DEVICE)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# FINITE TENSOR CHECK
# =====================================================================

def tensor_is_finite(tensor: torch.Tensor) -> bool:
    """Return True when every tensor element is finite."""

    return bool(torch.isfinite(tensor).all().item())


# =====================================================================
# METRIC CONVERSION
# =====================================================================

def metric_to_float(value) -> float:
    """Convert a tensor, NumPy scalar, or Python number to float."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "Expected a scalar tensor for metric conversion."
            )

        return float(value.detach().cpu().item())

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                "Expected a scalar NumPy array for metric conversion."
            )

        return float(value.item())

    return float(value)


# =====================================================================
# INPUT VALIDATION
# =====================================================================

def validate_common_input(
    corrupted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    velocity: torch.Tensor,
) -> None:
    """
    Validate the common comparison dataset.

    Dataset convention:
        (C, D, H, W)
    """

    tensors = {
        "corrupted": corrupted,
        "target": target,
        "mask": mask,
        "velocity": velocity,
    }

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape (C, D, H, W). "
                f"Received: {tuple(tensor.shape)}"
            )

    if corrupted.shape != target.shape:
        raise RuntimeError(
            "Corrupted and target shapes differ. "
            f"Corrupted: {tuple(corrupted.shape)}; "
            f"Target: {tuple(target.shape)}"
        )

    if corrupted.shape != mask.shape:
        raise RuntimeError(
            "Corrupted and mask shapes differ."
        )

    if corrupted.shape != velocity.shape:
        raise RuntimeError(
            "Corrupted and velocity shapes differ."
        )

    expected_shape = (1, *CUBE_SIZE)

    if tuple(corrupted.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected benchmark cube shape. "
            f"Expected: {expected_shape}; "
            f"Received: {tuple(corrupted.shape)}"
        )

    for name, tensor in tensors.items():
        if not tensor_is_finite(tensor):
            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

    unique_mask = torch.unique(mask)

    if not bool(
        ((unique_mask == 0) | (unique_mask == 1)).all().item()
    ):
        raise RuntimeError(
            "Mask must contain only 0 and 1. "
            f"Received: {unique_mask.tolist()}"
        )

    observed_count = int((mask == 1).sum().item())
    missing_count = int((mask == 0).sum().item())

    if observed_count == 0:
        raise RuntimeError(
            "The common comparison sample contains no observed samples."
        )

    if missing_count == 0:
        raise RuntimeError(
            "The common comparison sample contains no missing samples."
        )


# =====================================================================
# OBSERVED-DATA PRESERVATION
# =====================================================================

def validate_reconstruction(
    reconstruction: torch.Tensor,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Validate a reconstructed cube.

    Checks tensor type, shape, finite values, and preservation
    of the observed input samples.

    Returns
    -------
    float
        Maximum observed-data preservation error.
    """

    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError(
            "Reconstruction must be a torch.Tensor."
        )

    if reconstruction.shape != corrupted.shape:
        raise RuntimeError(
            "Reconstruction shape mismatch. "
            f"Expected: {tuple(corrupted.shape)}; "
            f"Received: {tuple(reconstruction.shape)}"
        )

    if not tensor_is_finite(reconstruction):
        raise RuntimeError(
            "Reconstruction contains NaN or Inf."
        )

    observed_selector = mask == 1

    observed_difference = torch.max(
        torch.abs(
            reconstruction[observed_selector]
            - corrupted[observed_selector]
        )
    ).item()

    if observed_difference > OBSERVED_PRESERVATION_TOLERANCE:
        raise RuntimeError(
            "Observed-data preservation failed. "
            f"Maximum difference: {observed_difference:.6e}; "
            f"Tolerance: {OBSERVED_PRESERVATION_TOLERANCE:.6e}"
        )

    return float(observed_difference)


# =====================================================================
# COMMON METRICS
# =====================================================================

def calculate_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """
    Calculate global and missing-region reconstruction metrics.

    Input convention:
        (C, D, H, W)
    """

    missing_selector = mask == 0

    missing_prediction = reconstruction[missing_selector]
    missing_target = target[missing_selector]

    if missing_prediction.numel() == 0:
        raise RuntimeError(
            "No missing samples available for missing-region metrics."
        )

    missing_mae = float(
        torch.mean(
            torch.abs(missing_prediction - missing_target)
        ).item()
    )

    missing_rmse = float(
        torch.sqrt(
            torch.mean(
                (missing_prediction - missing_target) ** 2
            )
        ).item()
    )

    reconstruction_batch = reconstruction.unsqueeze(0).to(DEVICE)
    target_batch = target.unsqueeze(0).to(DEVICE)

    return {
        "MAE": metric_to_float(
            mae(reconstruction_batch, target_batch)
        ),
        "RMSE": metric_to_float(
            rmse(reconstruction_batch, target_batch)
        ),
        "PSNR": metric_to_float(
            psnr(reconstruction_batch, target_batch)
        ),
        "SNR": metric_to_float(
            snr(reconstruction_batch, target_batch)
        ),
        "SSIM": metric_to_float(
            ssim(reconstruction_batch, target_batch)
        ),
        "missing_mae": missing_mae,
        "missing_rmse": missing_rmse,
    }


# =====================================================================
# LOAD PROPOSED MODEL
# =====================================================================

def load_proposed_model():
    """
    Load the configured acoustic Physics-Informed 3-D model.

    The model architecture must match the trained checkpoint,
    including the configured acoustic pressure-wavefield time steps.
    """

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Proposed-model checkpoint was not found:\n"
            f"{CHECKPOINT}"
        )

    print()
    print("=" * 78)
    print("LOADING PROPOSED MODEL")
    print("=" * 78)
    print(f"Checkpoint          : {CHECKPOINT}")
    print(f"Device              : {DEVICE}")
    print(
        "Acoustic time steps :",
        config.ACOUSTIC_TIME_STEPS,
    )

    # Construct the same architecture as the trained acoustic model.
    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
        pressure_time_steps=config.ACOUSTIC_TIME_STEPS,
    ).to(DEVICE)

    # Load the checkpoint on the selected device.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("Proposed model loaded successfully.")

    if "best_epoch" in checkpoint:
        print(f"Best epoch           : {checkpoint['best_epoch']}")

    if "best_val_loss" in checkpoint:
        print(
            "Best validation loss :",
            checkpoint["best_val_loss"],
        )

    return model


# =====================================================================
# RUN ONE CLASSICAL BASELINE
# =====================================================================

def run_classical_method(
    method_name: str,
    reconstruction_function,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Run one classical baseline.

    Ground truth is deliberately not passed to the reconstruction
    function. It is used only after reconstruction for evaluation.
    """

    print()
    print("-" * 78)
    print(f"METHOD: {method_name}")
    print("-" * 78)

    start_time = time.perf_counter()

    reconstruction = reconstruction_function(
        corrupted,
        mask,
    )

    runtime_seconds = time.perf_counter() - start_time

    preservation_error = validate_reconstruction(
        reconstruction,
        corrupted,
        mask,
    )

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    result = {
        "method": method_name,
        "geological_mode": GEOLOGICAL_MODE,
        "mask_mode": MASK_MODE,
        "seed": int(SEED),
        "requested_missing_rate": float(MISSING_RATE),
        "cube_depth": int(CUBE_SIZE[0]),
        "cube_height": int(CUBE_SIZE[1]),
        "cube_width": int(CUBE_SIZE[2]),
        "runtime_seconds": float(runtime_seconds),
        "observed_preservation_error": float(preservation_error),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "PSNR": metrics["PSNR"],
        "SNR": metrics["SNR"],
        "SSIM": metrics["SSIM"],
        "missing_mae": metrics["missing_mae"],
        "missing_rmse": metrics["missing_rmse"],
        "aleatoric_variance_mean": np.nan,
        "epistemic_variance_mean": np.nan,
        "predictive_variance_mean": np.nan,
        "predictive_std_mean": np.nan,
        "missing_aleatoric_variance_mean": np.nan,
        "missing_epistemic_variance_mean": np.nan,
        "missing_predictive_variance_mean": np.nan,
        "missing_predictive_std_mean": np.nan,
        "status": "PASS",
        "error": "",
    }

    print(f"MAE          : {result['MAE']:.6f}")
    print(f"RMSE         : {result['RMSE']:.6f}")
    print(f"PSNR         : {result['PSNR']:.6f} dB")
    print(f"SNR          : {result['SNR']:.6f} dB")
    print(f"SSIM         : {result['SSIM']:.6f}")
    print(f"Missing MAE  : {result['missing_mae']:.6f}")
    print(f"Missing RMSE : {result['missing_rmse']:.6f}")
    print(f"Runtime      : {result['runtime_seconds']:.4f} s")
    print(f"Observed err : {result['observed_preservation_error']:.6e}")

    return result


# =====================================================================
# RUN PROPOSED MODEL
# =====================================================================

def run_proposed_model(
    model,
    corrupted: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Run the proposed acoustic Physics-Informed 3-D model.

    Input:
        corrupted: (C,D,H,W)
        mask:      (C,D,H,W)

    Network input:
        (B,C,D,H,W)

    The pressure wavefield is generated by the network but is not
    retained as a collection of MC samples. Reconstruction uncertainty
    is calculated from reconstruction and log-variance samples.
    """

    print()
    print("-" * 78)
    print("METHOD: proposed_physics_informed_3d")
    print("-" * 78)

    input_batch = corrupted.unsqueeze(0).to(DEVICE)
    corrupted_batch = corrupted.unsqueeze(0).to(DEVICE)
    mask_batch = mask.unsqueeze(0).to(DEVICE)

    expected_network_shape = (1, 1, *CUBE_SIZE)

    if tuple(input_batch.shape) != expected_network_shape:
        raise RuntimeError(
            "Unexpected proposed-model input shape. "
            f"Expected: {expected_network_shape}; "
            f"Received: {tuple(input_batch.shape)}"
        )

    # MC-Dropout defaults to not storing pressure-wavefield samples.
    predictor = MCDropout3D(
        model=model,
        num_samples=MC_SAMPLES,
    )

    start_time = time.perf_counter()

    predictions = predictor.predict(input_batch)

    runtime_seconds = time.perf_counter() - start_time

    required_keys = {
        "reconstruction_samples",
        "log_variance_samples",
    }

    missing_keys = required_keys - predictions.keys()

    if missing_keys:
        raise KeyError(
            "MC-Dropout predictor did not return required outputs: "
            f"{missing_keys}"
        )

    reconstruction_samples = predictions["reconstruction_samples"]
    log_variance_samples = predictions["log_variance_samples"]

    expected_mc_shape = (
        MC_SAMPLES,
        1,
        1,
        *CUBE_SIZE,
    )

    if tuple(reconstruction_samples.shape) != expected_mc_shape:
        raise RuntimeError(
            "Unexpected reconstruction MC-sample shape. "
            f"Expected: {expected_mc_shape}; "
            f"Received: {tuple(reconstruction_samples.shape)}"
        )

    if tuple(log_variance_samples.shape) != expected_mc_shape:
        raise RuntimeError(
            "Unexpected log-variance MC-sample shape. "
            f"Expected: {expected_mc_shape}; "
            f"Received: {tuple(log_variance_samples.shape)}"
        )

    if not tensor_is_finite(reconstruction_samples):
        raise RuntimeError(
            "MC reconstruction samples contain NaN or Inf."
        )

    if not tensor_is_finite(log_variance_samples):
        raise RuntimeError(
            "MC log-variance samples contain NaN or Inf."
        )

    # Average stochastic reconstruction samples.
    reconstruction_mean = reconstruction_samples.mean(dim=0)

    # Estimate aleatoric, epistemic, and predictive uncertainty.
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
            "Predictive uncertainty estimator did not return: "
            f"{missing_uncertainty_keys}"
        )

    aleatoric_variance = uncertainty_results["aleatoric_variance"]
    epistemic_variance = uncertainty_results["epistemic_variance"]
    predictive_variance = uncertainty_results["predictive_variance"]
    predictive_std = uncertainty_results["predictive_std"]

    expected_uncertainty_shape = (1, 1, *CUBE_SIZE)

    uncertainty_tensors = {
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "predictive_variance": predictive_variance,
        "predictive_std": predictive_std,
    }

    for name, tensor in uncertainty_tensors.items():
        if tuple(tensor.shape) != expected_uncertainty_shape:
            raise RuntimeError(
                f"Unexpected {name} shape. "
                f"Expected: {expected_uncertainty_shape}; "
                f"Received: {tuple(tensor.shape)}"
            )

        if not tensor_is_finite(tensor):
            raise RuntimeError(
                f"{name} contains NaN or Inf."
            )

        if "variance" in name and bool((tensor < 0).any().item()):
            raise RuntimeError(
                f"Negative values detected in {name}."
            )

    # Verify the predictive-variance decomposition.
    decomposition_difference = torch.max(
        torch.abs(
            predictive_variance
            - (aleatoric_variance + epistemic_variance)
        )
    ).item()

    if decomposition_difference > 1.0e-5:
        raise RuntimeError(
            "Predictive variance decomposition failed. "
            f"Maximum difference: {decomposition_difference:.6e}"
        )

    # Restore observed data exactly; predict only missing samples.
    reconstruction_batch = (
        (1.0 - mask_batch) * reconstruction_mean
        + mask_batch * corrupted_batch
    )

    if tuple(reconstruction_batch.shape) != expected_network_shape:
        raise RuntimeError(
            "Unexpected final reconstruction shape. "
            f"Expected: {expected_network_shape}; "
            f"Received: {tuple(reconstruction_batch.shape)}"
        )

    reconstruction = (
        reconstruction_batch.squeeze(0).detach().cpu()
    )

    preservation_error = validate_reconstruction(
        reconstruction,
        corrupted,
        mask,
    )

    metrics = calculate_metrics(
        reconstruction,
        target,
        mask,
    )

    missing_selector = mask_batch == 0

    if not bool(missing_selector.any().item()):
        raise RuntimeError(
            "No missing samples available for missing-region uncertainty."
        )

    # Overall uncertainty statistics.
    mean_aleatoric = float(aleatoric_variance.mean().item())
    mean_epistemic = float(epistemic_variance.mean().item())
    mean_predictive = float(predictive_variance.mean().item())
    mean_predictive_std = float(predictive_std.mean().item())

    # Missing-region uncertainty statistics.
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

    result = {
        "method": "proposed_physics_informed_3d",
        "geological_mode": GEOLOGICAL_MODE,
        "mask_mode": MASK_MODE,
        "seed": int(SEED),
        "requested_missing_rate": float(MISSING_RATE),
        "cube_depth": int(CUBE_SIZE[0]),
        "cube_height": int(CUBE_SIZE[1]),
        "cube_width": int(CUBE_SIZE[2]),
        "runtime_seconds": float(runtime_seconds),
        "observed_preservation_error": float(preservation_error),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "PSNR": metrics["PSNR"],
        "SNR": metrics["SNR"],
        "SSIM": metrics["SSIM"],
        "missing_mae": metrics["missing_mae"],
        "missing_rmse": metrics["missing_rmse"],
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

    print(f"MAE          : {result['MAE']:.6f}")
    print(f"RMSE         : {result['RMSE']:.6f}")
    print(f"PSNR         : {result['PSNR']:.6f} dB")
    print(f"SNR          : {result['SNR']:.6f} dB")
    print(f"SSIM         : {result['SSIM']:.6f}")
    print(f"Missing MAE  : {result['missing_mae']:.6f}")
    print(f"Missing RMSE : {result['missing_rmse']:.6f}")
    print(f"Runtime      : {result['runtime_seconds']:.4f} s")
    print(f"Aleatoric σ² : {mean_aleatoric:.6e}")
    print(f"Epistemic σ² : {mean_epistemic:.6e}")
    print(f"Predictive σ²: {mean_predictive:.6e}")
    print(f"Predictive σ : {mean_predictive_std:.6e}")
    print(f"Observed err : {preservation_error:.6e}")

    # Release references to large MC tensors as soon as practical.
    del predictions
    del reconstruction_samples
    del log_variance_samples
    del uncertainty_results
    del reconstruction_mean

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


# =====================================================================
# FAILED RESULT RECORD
# =====================================================================

def build_failed_result(
    method_name: str,
    error: Exception,
) -> dict:
    """Create a standardized failed-result record."""

    return {
        "method": method_name,
        "geological_mode": GEOLOGICAL_MODE,
        "mask_mode": MASK_MODE,
        "seed": int(SEED),
        "requested_missing_rate": float(MISSING_RATE),
        "cube_depth": int(CUBE_SIZE[0]),
        "cube_height": int(CUBE_SIZE[1]),
        "cube_width": int(CUBE_SIZE[2]),
        "runtime_seconds": np.nan,
        "observed_preservation_error": np.nan,
        "MAE": np.nan,
        "RMSE": np.nan,
        "PSNR": np.nan,
        "SNR": np.nan,
        "SSIM": np.nan,
        "missing_mae": np.nan,
        "missing_rmse": np.nan,
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


# =====================================================================
# WRITE CSV
# =====================================================================

def write_csv(
    output_file: Path,
    records: list[dict],
) -> None:
    """Write result records to CSV."""

    if not records:
        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(records[0].keys())

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


# =====================================================================
# SUMMARY
# =====================================================================

def calculate_summary(
    records: list[dict],
) -> list[dict]:
    """
    Create the side-by-side summary.

    This script evaluates one common case per method.
    Statistical analysis for the 750-case controlled experiment
    remains in the dedicated controlled-experiment scripts.
    """

    summaries = []

    for record in records:
        summaries.append({
            "method": record["method"],
            "n_cases": 1 if record["status"] == "PASS" else 0,
            "MAE_mean": record["MAE"],
            "RMSE_mean": record["RMSE"],
            "PSNR_mean": record["PSNR"],
            "SNR_mean": record["SNR"],
            "SSIM_mean": record["SSIM"],
            "missing_mae_mean": record["missing_mae"],
            "missing_rmse_mean": record["missing_rmse"],
            "runtime_seconds_mean": record["runtime_seconds"],
            "observed_preservation_error_max": (
                record["observed_preservation_error"]
            ),
            "aleatoric_variance_mean": (
                record["aleatoric_variance_mean"]
            ),
            "epistemic_variance_mean": (
                record["epistemic_variance_mean"]
            ),
            "predictive_variance_mean": (
                record["predictive_variance_mean"]
            ),
            "predictive_std_mean": (
                record["predictive_std_mean"]
            ),
            "missing_aleatoric_variance_mean": (
                record["missing_aleatoric_variance_mean"]
            ),
            "missing_epistemic_variance_mean": (
                record["missing_epistemic_variance_mean"]
            ),
            "missing_predictive_variance_mean": (
                record["missing_predictive_variance_mean"]
            ),
            "missing_predictive_std_mean": (
                record["missing_predictive_std_mean"]
            ),
            "status": record["status"],
        })

    return summaries


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """Execute the common seven-method comparison."""

    print()
    print("=" * 78)
    print("COMMON SEVEN-METHOD RECONSTRUCTION COMPARISON")
    print("=" * 78)

    print(f"\nCube size          : {CUBE_SIZE}")
    print(f"Number of samples  : {NUM_SAMPLES}")
    print(f"Geological mode    : {GEOLOGICAL_MODE}")
    print(f"Mask mode          : {MASK_MODE}")
    print(f"Missing rate       : {MISSING_RATE:.0%}")
    print(f"Seed               : {SEED}")
    print(f"Configured device  : {CONFIG_DEVICE}")
    print(f"Resolved device    : {DEVICE}")
    print(f"Attention          : {USE_ATTENTION}")
    print(f"Residual           : {USE_RESIDUAL}")
    print(f"Uncertainty        : {USE_UNCERTAINTY}")
    print(f"Acoustic time steps: {config.ACOUSTIC_TIME_STEPS}")
    print(f"MC samples         : {MC_SAMPLES}")

    print("\nThis script evaluates ONE common test case.")
    print(
        "The 750-case controlled experiments are handled separately."
    )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------

    set_seed(SEED)

    # -----------------------------------------------------------------
    # Create the common test dataset
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("CREATING COMMON TEST SAMPLE")
    print("=" * 78)

    dataset = SyntheticSeismicDataset(
        num_samples=NUM_SAMPLES,
        cube_size=CUBE_SIZE,
        missing_probability=MISSING_RATE,
        geological_mode=GEOLOGICAL_MODE,
        mask_mode=MASK_MODE,
        seed=SEED,
    )

    (
        corrupted,
        target,
        mask,
        velocity,
        actual_mask_mode,
        actual_geological_mode,
    ) = dataset[0]

    if actual_geological_mode != GEOLOGICAL_MODE:
        raise RuntimeError(
            "Geological mode mismatch. "
            f"Requested: {GEOLOGICAL_MODE}; "
            f"Received: {actual_geological_mode}"
        )

    if actual_mask_mode != MASK_MODE:
        raise RuntimeError(
            "Mask mode mismatch. "
            f"Requested: {MASK_MODE}; "
            f"Received: {actual_mask_mode}"
        )

    validate_common_input(
        corrupted,
        target,
        mask,
        velocity,
    )

    # -----------------------------------------------------------------
    # Verify that corrupted input contains only observed target values
    # -----------------------------------------------------------------

    expected_corrupted = target * mask

    input_difference = torch.max(
        torch.abs(corrupted - expected_corrupted)
    ).item()

    if input_difference > OBSERVED_PRESERVATION_TOLERANCE:
        raise RuntimeError(
            "Common input consistency failed. "
            f"Maximum difference: {input_difference:.6e}"
        )

    # -----------------------------------------------------------------
    # Sample statistics
    # -----------------------------------------------------------------

    observed_samples = int((mask == 1).sum().item())
    missing_samples = int((mask == 0).sum().item())
    total_samples = observed_samples + missing_samples

    measured_missing_rate = missing_samples / total_samples

    print(f"\nObserved samples : {observed_samples}")
    print(f"Missing samples  : {missing_samples}")
    print(f"Measured missing : {measured_missing_rate:.6f}")

    # -----------------------------------------------------------------
    # Load the proposed model before running the methods
    # -----------------------------------------------------------------

    proposed_model = load_proposed_model()

    records = []

    # -----------------------------------------------------------------
    # Six classical methods
    # -----------------------------------------------------------------

    classical_methods = [
        (
            "nearest_neighbor",
            nearest_neighbor_reconstruction,
        ),
        (
            "linear_interpolation",
            linear_interpolation_reconstruction,
        ),
        (
            "fx_prediction",
            fx_prediction_reconstruction,
        ),
        (
            "compressive_sensing",
            compressive_sensing_reconstruction,
        ),
        (
            "curvelet_pocs",
            curvelet_pocs_reconstruction,
        ),
        (
            "dictionary_learning",
            dictionary_learning_reconstruction,
        ),
    ]

    for method_name, reconstruction_function in classical_methods:
        try:
            result = run_classical_method(
                method_name=method_name,
                reconstruction_function=reconstruction_function,
                corrupted=corrupted,
                mask=mask,
                target=target,
            )
        except Exception as error:
            print(f"\n{method_name} FAILED")
            print(f"Error: {error}")

            result = build_failed_result(
                method_name,
                error,
            )

        records.append(result)

    # -----------------------------------------------------------------
    # Proposed acoustic model
    # -----------------------------------------------------------------

    try:
        result = run_proposed_model(
            model=proposed_model,
            corrupted=corrupted,
            mask=mask,
            target=target,
        )
    except Exception as error:
        print("\nproposed_physics_informed_3d FAILED")
        print(f"Error: {error}")

        result = build_failed_result(
            "proposed_physics_informed_3d",
            error,
        )

    records.append(result)

    # -----------------------------------------------------------------
    # Save raw results and summary
    # -----------------------------------------------------------------

    write_csv(
        RESULTS_FILE,
        records,
    )

    summaries = calculate_summary(records)

    write_csv(
        SUMMARY_FILE,
        summaries,
    )

    successful = sum(
        record["status"] == "PASS"
        for record in records
    )

    failed = sum(
        record["status"] == "FAILED"
        for record in records
    )

    # -----------------------------------------------------------------
    # Console comparison table
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("COMMON RECONSTRUCTION METRICS")
    print("=" * 78)
    print()

    print(
        f"{'Method':<36}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'PSNR':>10}"
        f"{'SNR':>10}"
        f"{'SSIM':>10}"
    )

    print("-" * 86)

    for record in records:
        if record["status"] != "PASS":
            continue

        print(
            f"{record['method']:<36}"
            f"{record['MAE']:>10.6f}"
            f"{record['RMSE']:>10.6f}"
            f"{record['PSNR']:>10.4f}"
            f"{record['SNR']:>10.4f}"
            f"{record['SSIM']:>10.6f}"
        )

    # -----------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("SEVEN-METHOD COMPARISON COMPLETE")
    print("=" * 78)

    print(f"\nMethods evaluated : {len(records)}")
    print(f"Successful        : {successful}")
    print(f"Failed            : {failed}")

    print(f"\nRaw results:\n{RESULTS_FILE}")
    print(f"\nSummary results:\n{SUMMARY_FILE}")

    if failed == 0:
        print("\nOVERALL STATUS: PASS")
        print(
            "All seven methods completed successfully using the "
            "same corrupted cube and observation mask."
        )
    else:
        print("\nOVERALL STATUS: FAIL")
        print(
            "One or more methods failed. Inspect the error column "
            "in baseline_comparison.csv."
        )

    print()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()