
"""
====================================================================
Geological Complexity Robustness Evaluation
====================================================================

Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction

ACOUSTIC-WAVE-EQUATION PROJECT
-----------------------------
Network outputs:
    1. reconstructed_cube
    2. pressure_wavefield
    3. log_variance

Purpose
-------
Evaluate the trained reconstruction model across six geological
complexity levels without retraining.

Methodological principles
-------------------------
1. Load the trained model once and keep its parameters frozen.
2. Evaluate each explicitly requested geological mode.
3. Restore observed seismic samples exactly after inference.
4. Obtain reconstruction metrics from the canonical metrics module.
5. Calculate aleatoric uncertainty from predicted log-variance.
6. Do not fabricate epistemic or predictive uncertainty.
7. Use the configured acoustic pressure-wavefield time steps.
8. Do not interpret the pressure wavefield as Eikonal travel time.
9. Save results under the configured REPORT_DIR.

IMPORTANT
---------
dataset.build_dataset.build_dataset() must genuinely support the
geological_mode argument for this experiment to be scientifically
valid. This script deliberately does not silently substitute another
geological mode if the dataset builder does not support it.
====================================================================
"""

# ====================================================================
# 1. STANDARD LIBRARY IMPORTS
# ====================================================================

import csv
import json
import math
import random
from pathlib import Path

# ====================================================================
# 2. THIRD-PARTY IMPORTS
# ====================================================================

import numpy as np
import torch
from tqdm import tqdm

# ====================================================================
# 3. PROJECT IMPORTS
# ====================================================================

from utils.config import (
    DATASET_MODE,
    DEVICE,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
    ACOUSTIC_TIME_STEPS,
)

from models.network import Network3D

from dataset.build_dataset import build_dataset

from metrics.reconstruction_metrics import (
    calculate_reconstruction_metrics,
    missing_mae,
    missing_rmse,
    observed_mae,
    observed_rmse,
)


# ====================================================================
# 4. REPRODUCIBILITY
# ====================================================================

SEED = 42


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ====================================================================
# 5. DEVICE RESOLUTION
# ====================================================================

def resolve_device(device_setting):
    """Convert the configured device setting into torch.device."""

    if isinstance(device_setting, torch.device):
        device = device_setting

    else:
        device_setting = str(device_setting).lower()

        if device_setting == "cpu":
            device = torch.device("cpu")

        elif device_setting == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DEVICE='cuda' was requested, but CUDA is "
                    "not available."
                )

            device = torch.device("cuda")

        elif device_setting == "auto":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:
            raise ValueError(
                f"Unsupported DEVICE setting: {device_setting}"
            )

    return device


# ====================================================================
# 6. GEOLOGICAL COMPLEXITY LEVELS
# ====================================================================

GEOLOGICAL_COMPLEXITY_LEVELS = [
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
]


# ====================================================================
# 7. CHECKPOINT AND OUTPUT DIRECTORIES
# ====================================================================

CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "best_model.pth"

OUTPUT_DIR = Path(REPORT_DIR) / "geological_complexity"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ====================================================================
# 8. LOAD MODEL
# ====================================================================

def load_model(device):
    """
    Load the trained acoustic-wave-equation reconstruction model.

    The architecture must match the architecture used to create
    the saved checkpoint.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    # The acoustic model requires the configured pressure-wavefield
    # time dimension. Do not instantiate the old Eikonal version.
    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
        pressure_time_steps=ACOUSTIC_TIME_STEPS,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: expected a dictionary."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        # Allow a raw state dictionary when all values are tensors.
        if not all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            raise RuntimeError(
                "Checkpoint does not contain 'model_state_dict' "
                "or 'state_dict', and is not a raw state dictionary."
            )

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    # This evaluation must not update model parameters.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


# ====================================================================
# 9. MODEL OUTPUT EXTRACTION
# ====================================================================

def extract_model_outputs(model_output):
    """
    Extract the acoustic model outputs.

    Expected tuple/list order:
        reconstruction, pressure_wavefield, log_variance

    Dictionary outputs are also supported when the expected keys
    are present.
    """

    if isinstance(model_output, dict):
        reconstruction = model_output.get(
            "reconstructed_cube",
            model_output.get("reconstruction"),
        )

        pressure_wavefield = model_output.get(
            "pressure_wavefield"
        )

        log_variance = model_output.get(
            "log_variance"
        )

    elif isinstance(model_output, (tuple, list)):
        if len(model_output) != 3:
            raise RuntimeError(
                "The acoustic Network3D must return exactly three "
                "outputs: reconstruction, pressure_wavefield and "
                f"log_variance. Received {len(model_output)}."
            )

        reconstruction, pressure_wavefield, log_variance = (
            model_output
        )

    else:
        raise RuntimeError(
            "Unsupported Network3D output format. Expected a "
            "dictionary or a three-element tuple/list."
        )

    if not torch.is_tensor(reconstruction):
        raise RuntimeError(
            "The model did not return a valid reconstruction tensor."
        )

    if not torch.is_tensor(pressure_wavefield):
        raise RuntimeError(
            "The acoustic model did not return a valid "
            "pressure_wavefield tensor."
        )

    if log_variance is not None and not torch.is_tensor(log_variance):
        raise RuntimeError(
            "log_variance must be a tensor or None."
        )

    return (
        reconstruction,
        pressure_wavefield,
        log_variance,
    )


# ====================================================================
# 10. TENSOR STANDARDIZATION
# ====================================================================

def ensure_5d(tensor, name):
    """
    Standardize seismic cubes to [B,C,D,H,W].

    Accepted input shapes:
        [D,H,W]
        [C,D,H,W]
        [B,C,D,H,W]
    """

    if tensor is None:
        return None

    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(
            tensor,
            dtype=torch.float32,
        )

    if tensor.ndim == 3:
        return tensor.unsqueeze(0).unsqueeze(0)

    if tensor.ndim == 4:
        return tensor.unsqueeze(0)

    if tensor.ndim == 5:
        return tensor

    raise ValueError(
        f"{name} must have 3, 4 or 5 dimensions; "
        f"received {tuple(tensor.shape)}."
    )


# ====================================================================
# 11. ACOUSTIC PRESSURE-WAVEFIELD VALIDATION
# ====================================================================

def validate_pressure_wavefield(
    pressure_wavefield,
    reconstruction,
):
    """
    Validate the pressure-wavefield tensor.

    Expected convention:
        [B,C,T,D,H,W]

    The pressure field is not an Eikonal travel-time field. This
    function validates its shape and finiteness but does not claim
    to calculate an acoustic-equation residual.
    """

    if pressure_wavefield.ndim != 6:
        raise ValueError(
            "pressure_wavefield must have shape [B,C,T,D,H,W]. "
            f"Received {tuple(pressure_wavefield.shape)}."
        )

    expected_batch = reconstruction.shape[0]
    expected_channels = 1
    expected_time = ACOUSTIC_TIME_STEPS
    expected_spatial = tuple(reconstruction.shape[2:])

    expected_shape = (
        expected_batch,
        expected_channels,
        expected_time,
        *expected_spatial,
    )

    if tuple(pressure_wavefield.shape) != expected_shape:
        raise ValueError(
            "Unexpected pressure-wavefield shape. "
            f"Expected {expected_shape}, received "
            f"{tuple(pressure_wavefield.shape)}."
        )

    if not torch.isfinite(pressure_wavefield).all():
        raise ValueError(
            "The pressure wavefield contains NaN or Inf."
        )


def calculate_pressure_diagnostics(pressure_wavefield):
    """
    Calculate descriptive pressure-wavefield statistics.

    These statistics are diagnostics only. They are not a measure
    of acoustic-equation satisfaction.
    """

    return {
        "Pressure_Wavefield_Mean": float(
            pressure_wavefield.mean().detach().cpu()
        ),
        "Pressure_Wavefield_Std": float(
            pressure_wavefield.std(unbiased=False).detach().cpu()
        ),
        "Pressure_Wavefield_Time_Steps": int(
            pressure_wavefield.shape[2]
        ),
    }


# ====================================================================
# 12. DATA CONSISTENCY PROJECTION
# ====================================================================

def apply_data_consistency(
    reconstruction,
    observed_input,
    mask,
):
    """
    Restore observed samples exactly.

    Mask convention:
        1 = observed
        0 = missing
    """

    return (
        reconstruction * (1.0 - mask)
        + observed_input * mask
    )


# ====================================================================
# 13. OBSERVED-DATA PRESERVATION ERROR
# ====================================================================

def observed_preservation_error(
    reconstruction,
    observed_input,
    mask,
):
    """Calculate the maximum absolute error at observed locations."""

    observed = mask > 0.5

    if not torch.any(observed):
        return 0.0

    difference = torch.abs(
        reconstruction[observed] - observed_input[observed]
    )

    return float(difference.max().detach().cpu())


# ====================================================================
# 14. ALEATORIC UNCERTAINTY
# ====================================================================

def calculate_aleatoric_uncertainty(log_variance):
    """
    Convert predicted log-variance into variance and standard
    deviation.

    No epistemic uncertainty is inferred from this deterministic
    single-pass evaluation.
    """

    if log_variance is None:
        return None, None

    log_variance = torch.clamp(
        log_variance,
        min=LOG_VARIANCE_MIN,
        max=LOG_VARIANCE_MAX,
    )

    aleatoric_variance = torch.exp(log_variance)

    aleatoric_std = torch.sqrt(aleatoric_variance)

    return aleatoric_variance, aleatoric_std


# ====================================================================
# 15. CASE METRICS
# ====================================================================

def calculate_case_metrics(
    prediction,
    target,
    mask,
):
    """Calculate canonical global, missing-region and observed-region
    reconstruction metrics.
    """

    metrics = calculate_reconstruction_metrics(
        prediction=prediction,
        target=target,
    )

    metrics["Missing_MAE"] = float(
        missing_mae(prediction, target, mask)
    )

    metrics["Missing_RMSE"] = float(
        missing_rmse(prediction, target, mask)
    )

    metrics["Observed_MAE"] = float(
        observed_mae(prediction, target, mask)
    )

    metrics["Observed_RMSE"] = float(
        observed_rmse(prediction, target, mask)
    )

    return metrics


# ====================================================================
# 16. EXTRACT SAMPLE FROM DATASET
# ====================================================================

def extract_sample(dataset, index):
    """
    Extract a sample from the dataset.

    Dictionary samples must provide:
        input or corrupted
        target
        mask

    Tuple/list samples must provide:
        input, target, mask

    An optional fourth velocity item is accepted for compatibility,
    but is not used by this deterministic evaluation.
    """

    sample = dataset[index]

    if isinstance(sample, dict):
        corrupted = sample.get(
            "input",
            sample.get("corrupted"),
        )

        target = sample.get("target")
        mask = sample.get("mask")

    elif isinstance(sample, (tuple, list)):
        if len(sample) < 3:
            raise RuntimeError(
                "Dataset sample must contain input, target and mask."
            )

        corrupted, target, mask = sample[:3]

    else:
        raise RuntimeError(
            "Unsupported dataset sample format."
        )

    if corrupted is None:
        raise RuntimeError(
            "Dataset sample does not contain input/corrupted data."
        )

    if target is None:
        raise RuntimeError(
            "Dataset sample does not contain target data."
        )

    if mask is None:
        raise RuntimeError(
            "Dataset sample does not contain mask data."
        )

    return corrupted, target, mask


# ====================================================================
# 17. VALIDATE INPUT TENSORS
# ====================================================================

def validate_input_tensors(corrupted, target, mask, mode, index):
    """Validate cube shapes, finite values and binary observation mask."""

    if corrupted.shape != target.shape:
        raise ValueError(
            f"[{mode}, sample {index}] Input and target shapes differ: "
            f"{tuple(corrupted.shape)} vs {tuple(target.shape)}."
        )

    if corrupted.shape != mask.shape:
        raise ValueError(
            f"[{mode}, sample {index}] Input and mask shapes differ: "
            f"{tuple(corrupted.shape)} vs {tuple(mask.shape)}."
        )

    for tensor_name, tensor in (
        ("corrupted", corrupted),
        ("target", target),
        ("mask", mask),
    ):
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"[{mode}, sample {index}] {tensor_name} "
                "contains NaN or Inf."
            )

    unique_mask_values = torch.unique(mask)

    valid_values = (
        torch.isclose(
            unique_mask_values,
            torch.zeros_like(unique_mask_values),
        )
        |
        torch.isclose(
            unique_mask_values,
            torch.ones_like(unique_mask_values),
        )
    )

    if not torch.all(valid_values):
        raise ValueError(
            f"[{mode}, sample {index}] Mask must contain only 0 and 1. "
            f"Found: {unique_mask_values.detach().cpu().tolist()}."
        )


# ====================================================================
# 18. EVALUATE ONE GEOLOGICAL COMPLEXITY LEVEL
# ====================================================================

@torch.no_grad()
def evaluate_geological_level(
    model,
    device,
    geological_mode,
):
    """
    Evaluate one geological mode using the same frozen model.

    The dataset builder must explicitly implement geological_mode.
    """

    set_seed(SEED)

    try:
        dataset = build_dataset(
            geological_mode=geological_mode,
        )

    except TypeError as exc:
        raise TypeError(
            "build_dataset() does not accept geological_mode. "
            "The geological complexity experiment requires the "
            "dataset builder to explicitly support each requested "
            f"mode, including '{geological_mode}'. No fallback "
            "dataset will be substituted."
        ) from exc

    num_samples = len(dataset)

    if num_samples == 0:
        raise RuntimeError(
            f"No samples available for geological mode "
            f"'{geological_mode}'."
        )

    results = []

    for sample_index in tqdm(
        range(num_samples),
        desc=f"Geology: {geological_mode}",
    ):
        corrupted, target, mask = extract_sample(
            dataset,
            sample_index,
        )

        corrupted = ensure_5d(corrupted, "corrupted")
        target = ensure_5d(target, "target")
        mask = ensure_5d(mask, "mask")

        corrupted = corrupted.to(
            device=device,
            dtype=torch.float32,
        )

        target = target.to(
            device=device,
            dtype=torch.float32,
        )

        mask = mask.to(
            device=device,
            dtype=torch.float32,
        )

        validate_input_tensors(
            corrupted,
            target,
            mask,
            geological_mode,
            sample_index,
        )

        # -----------------------------------------------------------
        # Acoustic model inference.
        # -----------------------------------------------------------

        model_output = model(corrupted)

        (
            reconstruction,
            pressure_wavefield,
            log_variance,
        ) = extract_model_outputs(model_output)

        reconstruction = ensure_5d(
            reconstruction,
            "reconstruction",
        )

        if reconstruction.shape != target.shape:
            raise ValueError(
                f"[{geological_mode}, sample {sample_index}] "
                "Reconstruction and target shapes differ: "
                f"{tuple(reconstruction.shape)} vs "
                f"{tuple(target.shape)}."
            )

        if not torch.isfinite(reconstruction).all():
            raise ValueError(
                f"[{geological_mode}, sample {sample_index}] "
                "Reconstruction contains NaN or Inf."
            )

        # -----------------------------------------------------------
        # Validate the physical pressure-wavefield output.
        # -----------------------------------------------------------

        validate_pressure_wavefield(
            pressure_wavefield,
            reconstruction,
        )

        pressure_diagnostics = calculate_pressure_diagnostics(
            pressure_wavefield
        )

        # -----------------------------------------------------------
        # Predicted aleatoric uncertainty.
        # -----------------------------------------------------------

        if log_variance is not None:
            log_variance = ensure_5d(
                log_variance,
                "log_variance",
            )

            if log_variance.shape != reconstruction.shape:
                raise ValueError(
                    f"[{geological_mode}, sample {sample_index}] "
                    "log_variance and reconstruction shapes differ."
                )

            if not torch.isfinite(log_variance).all():
                raise ValueError(
                    f"[{geological_mode}, sample {sample_index}] "
                    "log_variance contains NaN or Inf."
                )

            aleatoric_variance, aleatoric_std = (
                calculate_aleatoric_uncertainty(log_variance)
            )

        else:
            aleatoric_variance = None
            aleatoric_std = None

        # -----------------------------------------------------------
        # Exact observed-data restoration.
        # -----------------------------------------------------------

        reconstruction = apply_data_consistency(
            reconstruction=reconstruction,
            observed_input=corrupted,
            mask=mask,
        )

        preservation_error = observed_preservation_error(
            reconstruction,
            corrupted,
            mask,
        )

        # -----------------------------------------------------------
        # Canonical reconstruction metrics.
        # -----------------------------------------------------------

        metrics = calculate_case_metrics(
            prediction=reconstruction,
            target=target,
            mask=mask,
        )

        # -----------------------------------------------------------
        # Aleatoric uncertainty summaries.
        # -----------------------------------------------------------

        global_aleatoric_variance = float("nan")
        global_aleatoric_std = float("nan")
        missing_region_aleatoric_std = float("nan")

        if aleatoric_variance is not None:
            global_aleatoric_variance = float(
                aleatoric_variance.mean().detach().cpu()
            )

        if aleatoric_std is not None:
            global_aleatoric_std = float(
                aleatoric_std.mean().detach().cpu()
            )

            missing_pixels = mask < 0.5

            if torch.any(missing_pixels):
                missing_region_aleatoric_std = float(
                    aleatoric_std[missing_pixels]
                    .mean()
                    .detach()
                    .cpu()
                )

        # -----------------------------------------------------------
        # Per-sample record.
        #
        # Deterministic inference does not estimate epistemic or
        # predictive uncertainty. NaN is intentional.
        # -----------------------------------------------------------

        result = {
            "Sample_ID": sample_index,
            "Geological_Mode": geological_mode,

            "MAE": float(metrics["MAE"]),
            "MSE": float(metrics["MSE"]),
            "RMSE": float(metrics["RMSE"]),
            "Relative_Error": float(metrics["Relative_Error"]),
            "PSNR": float(metrics["PSNR"]),
            "SNR": float(metrics["SNR"]),
            "SSIM": float(metrics["SSIM"]),

            "Missing_MAE": float(metrics["Missing_MAE"]),
            "Missing_RMSE": float(metrics["Missing_RMSE"]),
            "Observed_MAE": float(metrics["Observed_MAE"]),
            "Observed_RMSE": float(metrics["Observed_RMSE"]),

            "Aleatoric_Variance": global_aleatoric_variance,
            "Aleatoric_Std": global_aleatoric_std,
            "Missing_Aleatoric_Std": missing_region_aleatoric_std,

            "Epistemic_Variance": float("nan"),
            "Epistemic_Std": float("nan"),
            "Predictive_Variance": float("nan"),
            "Predictive_Std": float("nan"),

            "Observed_Preservation_Error": preservation_error,

            "Missing_Rate_Actual": float(
                (mask < 0.5).float().mean().detach().cpu()
            ),

            "Epistemic_Evaluation": "MC_Dropout_required",
            "Predictive_Evaluation": "MC_Dropout_required",

            **pressure_diagnostics,
        }

        results.append(result)

        # Explicitly release references to large per-sample outputs
        # before processing the next sample.
        del model_output
        del pressure_wavefield
        del reconstruction

        if log_variance is not None:
            del log_variance

        if aleatoric_variance is not None:
            del aleatoric_variance

        if aleatoric_std is not None:
            del aleatoric_std

    return results


# ====================================================================
# 19. SUMMARY STATISTICS
# ====================================================================

def calculate_summary(results):
    """Calculate per-mode means and population standard deviations."""

    summaries = []

    metric_names = [
        "MAE",
        "MSE",
        "RMSE",
        "Relative_Error",
        "PSNR",
        "SNR",
        "SSIM",
        "Missing_MAE",
        "Missing_RMSE",
        "Observed_MAE",
        "Observed_RMSE",
        "Aleatoric_Variance",
        "Aleatoric_Std",
        "Missing_Aleatoric_Std",
        "Epistemic_Variance",
        "Epistemic_Std",
        "Predictive_Variance",
        "Predictive_Std",
        "Observed_Preservation_Error",
        "Missing_Rate_Actual",
        "Pressure_Wavefield_Mean",
        "Pressure_Wavefield_Std",
        "Pressure_Wavefield_Time_Steps",
    ]

    for geological_mode in GEOLOGICAL_COMPLEXITY_LEVELS:
        mode_results = [
            result
            for result in results
            if result["Geological_Mode"] == geological_mode
        ]

        if not mode_results:
            continue

        summary = {
            "Geological_Mode": geological_mode,
            "Number_of_Samples": len(mode_results),
        }

        for metric_name in metric_names:
            values = np.asarray(
                [
                    result.get(metric_name, float("nan"))
                    for result in mode_results
                ],
                dtype=np.float64,
            )

            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                summary[f"Mean_{metric_name}"] = float("nan")
                summary[f"Std_{metric_name}"] = float("nan")

            else:
                summary[f"Mean_{metric_name}"] = float(
                    np.mean(finite_values)
                )

                summary[f"Std_{metric_name}"] = float(
                    np.std(finite_values, ddof=0)
                )

        summaries.append(summary)

    return summaries


# ====================================================================
# 20. SAVE CSV
# ====================================================================

def save_csv(results, path):
    """Save result dictionaries to a CSV file."""

    if not results:
        raise RuntimeError(
            f"No results were produced; cannot save {path}."
        )

    fieldnames = []

    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


# ====================================================================
# 21. SAVE JSON
# ====================================================================

def save_json(data, path):
    """Save JSON metadata, allowing NaN for unavailable metrics."""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            allow_nan=True,
        )


# ====================================================================
# 22. MAIN EVALUATION
# ====================================================================

def main():
    print()
    print("=" * 70)
    print("GEOLOGICAL COMPLEXITY ROBUSTNESS EVALUATION")
    print("=" * 70)

    set_seed(SEED)

    device = resolve_device(DEVICE)

    print(f"Dataset mode         : {DATASET_MODE}")
    print(f"Device               : {device}")
    print(f"Checkpoint           : {CHECKPOINT_PATH}")
    print(f"Output directory     : {OUTPUT_DIR}")
    print(f"Acoustic time steps  : {ACOUSTIC_TIME_STEPS}")
    print(
        f"Log-variance bounds  : "
        f"[{LOG_VARIANCE_MIN}, {LOG_VARIANCE_MAX}]"
    )

    print()
    print("Loading the trained model once...")

    model = load_model(device)

    all_results = []

    # ---------------------------------------------------------------
    # Evaluate all six geological modes using the same frozen model.
    # ---------------------------------------------------------------

    for geological_mode in GEOLOGICAL_COMPLEXITY_LEVELS:
        print()
        print("-" * 70)
        print(f"Evaluating geological mode: {geological_mode}")
        print("-" * 70)

        mode_results = evaluate_geological_level(
            model=model,
            device=device,
            geological_mode=geological_mode,
        )

        all_results.extend(mode_results)

        print(
            f"Completed {geological_mode}: "
            f"{len(mode_results)} samples"
        )

    if not all_results:
        raise RuntimeError(
            "The geological complexity evaluation produced no results."
        )

    # ---------------------------------------------------------------
    # Calculate summaries.
    # ---------------------------------------------------------------

    summaries = calculate_summary(all_results)

    # ---------------------------------------------------------------
    # Output paths.
    # ---------------------------------------------------------------

    detailed_csv = (
        OUTPUT_DIR / "geological_complexity_results.csv"
    )

    summary_csv = (
        OUTPUT_DIR / "geological_complexity_summary.csv"
    )

    metadata_json = (
        OUTPUT_DIR / "geological_complexity_metadata.json"
    )

    # ---------------------------------------------------------------
    # Save detailed results and summary.
    # ---------------------------------------------------------------

    save_csv(all_results, detailed_csv)
    save_csv(summaries, summary_csv)

    # ---------------------------------------------------------------
    # Metadata.
    # ---------------------------------------------------------------

    metadata = {
        "evaluation": "Geological Complexity Robustness",
        "dataset_mode": DATASET_MODE,
        "device": str(device),
        "checkpoint": str(CHECKPOINT_PATH),

        "geological_complexity_levels": (
            GEOLOGICAL_COMPLEXITY_LEVELS
        ),

        "number_of_levels_requested": len(
            GEOLOGICAL_COMPLEXITY_LEVELS
        ),

        "total_samples_evaluated": len(all_results),
        "seed": SEED,

        "use_attention": USE_ATTENTION,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": USE_UNCERTAINTY,

        "acoustic_time_steps": ACOUSTIC_TIME_STEPS,

        "model_output_convention": [
            "reconstructed_cube",
            "pressure_wavefield",
            "log_variance",
        ],

        "log_variance_min": LOG_VARIANCE_MIN,
        "log_variance_max": LOG_VARIANCE_MAX,

        "metric_source": "metrics.reconstruction_metrics",

        "uncertainty_method": (
            "Deterministic aleatoric uncertainty calculated "
            "from predicted log-variance."
        ),

        "epistemic_uncertainty": (
            "Not estimated in this deterministic evaluation; "
            "separate MC-Dropout evaluation is required."
        ),

        "predictive_uncertainty": (
            "Not estimated in this deterministic evaluation; "
            "separate MC-Dropout evaluation is required."
        ),

        "pressure_wavefield_diagnostics": (
            "Mean and standard deviation are descriptive diagnostics "
            "only. They are not an acoustic-wave-equation residual."
        ),

        "data_consistency": (
            "Observed seismic samples are restored after model "
            "inference using the binary observation mask."
        ),

        "detailed_results": str(detailed_csv),
        "summary_results": str(summary_csv),
    }

    save_json(metadata, metadata_json)

    # ---------------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------------

    expected_levels = set(GEOLOGICAL_COMPLEXITY_LEVELS)

    completed_levels = {
        result["Geological_Mode"]
        for result in all_results
    }

    missing_levels = expected_levels - completed_levels

    if missing_levels:
        raise RuntimeError(
            "Evaluation incomplete. Missing geological levels: "
            f"{sorted(missing_levels)}"
        )

    preservation_errors = np.asarray(
        [
            result["Observed_Preservation_Error"]
            for result in all_results
        ],
        dtype=np.float64,
    )

    if not np.isfinite(preservation_errors).all():
        raise RuntimeError(
            "Observed-data preservation errors contain NaN or Inf."
        )

    max_preservation_error = float(
        np.max(preservation_errors)
    )

    # Floating-point multiplication/addition may produce tiny
    # residuals. Exact preservation is expected to be zero or
    # within a small numerical tolerance.
    preservation_tolerance = 1e-6

    if max_preservation_error > preservation_tolerance:
        raise RuntimeError(
            "Observed-data preservation check failed. "
            f"Maximum error = {max_preservation_error:.6e}; "
            f"tolerance = {preservation_tolerance:.1e}."
        )

    # ---------------------------------------------------------------
    # Final report.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("GEOLOGICAL COMPLEXITY EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Total samples evaluated : {len(all_results)}")
    print(f"Geological levels       : {len(completed_levels)}")

    print(
        "Maximum observed-data preservation error: "
        f"{max_preservation_error:.6e}"
    )

    print()
    print(f"Detailed results : {detailed_csv}")
    print(f"Summary results  : {summary_csv}")
    print(f"Metadata         : {metadata_json}")

    print()
    print("STATUS: PASS")
    print("=" * 70)


# ====================================================================
# 23. SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()
