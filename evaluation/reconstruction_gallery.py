"""
======================================================================
FINAL PhD RECONSTRUCTION GALLERY
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generate publication-quality reconstruction figures for the trained
acoustic-wave-equation synthetic seismic reconstruction model.

Each figure contains five panels:
    1. Input / Corrupted Data
    2. Ground Truth
    3. Reconstructed Data
    4. Absolute Reconstruction Error
    5. Predictive Uncertainty

Predictive variance:
    predictive variance = aleatoric variance + epistemic variance

Predictive standard deviation:
    predictive std = sqrt(predictive variance)

Important
---------
- The trained model is not retrained during gallery generation.
- The checkpoint is loaded once.
- MC Dropout stores reconstruction and log-variance samples only.
- Pressure-wavefield samples are not retained by default because
  their memory requirements are substantial.
- Observed seismic samples are restored exactly after reconstruction.
- Acoustic pressure-wavefield output is retained in the model interface.
  It is not confused with Eikonal travel time.

Author:
Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from dataset.build_dataset import build_dataset
from models.network import Network3D
from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator
)

from utils import config

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    DEVICE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MC_DROPOUT_SAMPLES,
    OBSERVED_PRESERVATION_TOLERANCE,
    GALLERY_NUMBER_OF_SAMPLES,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
)


# =====================================================================
# GLOBAL SETTINGS
# =====================================================================

GALLERY_DIR = Path(REPORT_DIR) / "gallery"

CHECKPOINT_FILE = (
    Path(CHECKPOINT_DIR) / "best_model.pth"
)

FIGURE_DPI = 300

# Slice axis for a volume represented as [D, H, W].
# 0 = depth slices; 1 = H-axis slices; 2 = W-axis slices.
SLICE_AXIS = 0


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device():
    """Resolve DEVICE values: auto, cpu or cuda."""

    configured_device = str(DEVICE).strip().lower()

    if configured_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if configured_device == "cpu":
        return torch.device("cpu")

    if configured_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE is configured as 'cuda', but CUDA is "
                "not available in this environment."
            )
        return torch.device("cuda")

    raise ValueError(
        f"Unsupported DEVICE configuration: {DEVICE}"
    )


# =====================================================================
# TENSOR VALIDATION
# =====================================================================

def validate_tensor(tensor, name, expected_ndim=None):
    """Check tensor type, optional dimensionality and finite values."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have {expected_ndim} dimensions; "
            f"received shape {tuple(tensor.shape)}."
        )

    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or Inf values.")



# =====================================================================
# CONVERT TENSOR TO NUMPY
# =====================================================================

def tensor_to_numpy(tensor):
    """
    Convert a PyTorch tensor or NumPy array to a NumPy array.

    PyTorch tensors are detached from the computation graph and
    moved to the CPU before conversion.

    Accepted inputs:
        - torch.Tensor
        - numpy.ndarray

    Returns:
        numpy.ndarray
    """

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(tensor, np.ndarray):
        return tensor

    raise TypeError(
        "tensor_to_numpy() expected a torch.Tensor or "
        "numpy.ndarray, but received "
        f"{type(tensor).__name__}."
    )


# =====================================================================
# EXTRACT DATASET SAMPLE
# =====================================================================

def extract_dataset_sample(sample):
    """
    Extract corrupted input, ground truth and observation mask.

    Expected dataset convention:
        sample[0] = corrupted seismic input
        sample[1] = ground-truth seismic volume
        sample[2] = observation mask

    Returns:
        corrupted, target, mask
    """

    if not isinstance(sample, (tuple, list)):
        raise TypeError(
            "Dataset sample must be a tuple or list. "
            f"Received {type(sample).__name__}."
        )

    if len(sample) < 3:
        raise ValueError(
            "Dataset sample must contain at least three items: "
            "input, target and observation mask. "
            f"Received {len(sample)} items."
        )

    corrupted = sample[0]
    target = sample[1]
    mask = sample[2]

    return corrupted, target, mask

# =====================================================================
# PREPARE MODEL INPUT
# =====================================================================

def prepare_model_input(tensor, device):
    """
    Accepted shapes:
        [C, D, H, W]
        [B, C, D, H, W]

    Returned shape:
        [B, C, D, H, W]
    """

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    validate_tensor(tensor, "model input")

    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 5:
        raise ValueError(
            "Model input must have shape [C,D,H,W] or "
            "[B,C,D,H,W]. "
            f"Received {tuple(tensor.shape)}."
        )

    tensor = tensor.to(device=device, dtype=torch.float32)

    validate_tensor(tensor, "prepared model input")

    return tensor


# =====================================================================
# PREPARE OBSERVATION MASK
# =====================================================================

def prepare_mask(mask, device):
    """
    Convert an observation mask to [B,C,D,H,W].

    Convention:
        1 = observed
        0 = missing
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    validate_tensor(mask, "observation mask")

    if mask.ndim == 4:
        mask = mask.unsqueeze(0)
    elif mask.ndim != 5:
        raise ValueError(
            "Mask must have shape [C,D,H,W] or [B,C,D,H,W]. "
            f"Received {tuple(mask.shape)}."
        )

    mask = mask.to(device=device, dtype=torch.float32)

    # Reject non-binary masks.
    is_zero = torch.isclose(mask, torch.zeros_like(mask))
    is_one = torch.isclose(mask, torch.ones_like(mask))

    if not torch.all(is_zero | is_one).item():
        raise ValueError(
            "Observation mask must contain only binary 0 and 1 values."
        )

    return mask


# =====================================================================
# OBSERVED-DATA PRESERVATION
# =====================================================================

def enforce_observed_data_consistency(
    reconstruction,
    corrupted,
    mask
):
    """
    Restore the original observed values exactly.

    reconstruction_projected =
        reconstruction * (1 - mask) + corrupted * mask
    """

    validate_tensor(reconstruction, "reconstruction")
    validate_tensor(corrupted, "corrupted input")
    validate_tensor(mask, "observation mask")

    if reconstruction.shape != corrupted.shape:
        raise ValueError(
            "Reconstruction and corrupted input must have identical "
            "shapes."
        )

    if reconstruction.shape != mask.shape:
        raise ValueError(
            "Reconstruction and mask must have identical shapes."
        )

    return reconstruction * (1.0 - mask) + corrupted * mask


# =====================================================================
# OBSERVED-DATA PRESERVATION ERROR
# =====================================================================

def calculate_observed_preservation_error(
    reconstruction,
    corrupted,
    mask
):
    """Calculate the maximum absolute error at observed positions."""

    observed_difference = (
        torch.abs(reconstruction - corrupted) * mask
    )

    return float(observed_difference.max().item())


# =====================================================================
# EXTRACT MC OUTPUTS
# =====================================================================

def extract_mc_outputs(mc_results):
    """
    Extract the stochastic reconstruction and log-variance samples.

    The gallery deliberately does not require pressure-wavefield samples.
    """

    if not isinstance(mc_results, dict):
        raise TypeError(
            "MC Dropout prediction must return a dictionary."
        )

    required_keys = (
        "reconstruction_samples",
        "log_variance_samples",
    )

    for key in required_keys:
        if key not in mc_results:
            raise KeyError(
                f"MC Dropout output is missing required key: {key}"
            )

    reconstruction_samples = mc_results[
        "reconstruction_samples"
    ]

    log_variance_samples = mc_results[
        "log_variance_samples"
    ]

    return reconstruction_samples, log_variance_samples


# =====================================================================
# MC PREDICTION AND UNCERTAINTY
# =====================================================================

@torch.no_grad()
def run_mc_prediction(
    model,
    input_cube,
    mask,
    corrupted_cube,
    device
):
    """
    Run MC Dropout and calculate predictive uncertainty.

    Expected MC sample shapes:
        [S, B, C, D, H, W]

    Expected final volume shapes:
        [B, C, D, H, W]

    The acoustic pressure-wavefield samples are not stored by this
    function. MCDropout3D is called with its default
    store_pressure_wavefield_samples=False setting.
    """

    # ---------------------------------------------------------------
    # Validate the input tensors.
    # ---------------------------------------------------------------

    validate_tensor(input_cube, "input_cube", expected_ndim=5)
    validate_tensor(corrupted_cube, "corrupted_cube", expected_ndim=5)
    validate_tensor(mask, "observation mask", expected_ndim=5)

    if input_cube.shape != corrupted_cube.shape:
        raise ValueError(
            "input_cube and corrupted_cube must have identical shapes."
        )

    if input_cube.shape != mask.shape:
        raise ValueError(
            "input_cube and mask must have identical shapes."
        )

    # ---------------------------------------------------------------
    # Run MC Dropout.
    #
    # Do not set store_pressure_wavefield_samples=True here.
    # The pressure wavefield is not required for the five-panel
    # reconstruction gallery.
    # ---------------------------------------------------------------

    mc_predictor = MCDropout3D(
        model=model,
        num_samples=MC_DROPOUT_SAMPLES,
    )

    mc_results = mc_predictor.predict(input_cube)

    reconstruction_samples, log_variance_samples = (
        extract_mc_outputs(mc_results)
    )

    # ---------------------------------------------------------------
    # Validate the MC samples.
    #
    # The first dimension represents MC samples:
    # [S, B, C, D, H, W].
    # ---------------------------------------------------------------

    validate_tensor(
        reconstruction_samples,
        "reconstruction_samples",
        expected_ndim=6,
    )

    validate_tensor(
        log_variance_samples,
        "log_variance_samples",
        expected_ndim=6,
    )

    if reconstruction_samples.shape != log_variance_samples.shape:
        raise ValueError(
            "Reconstruction samples and log-variance samples must "
            "have identical shapes."
        )

    if reconstruction_samples.shape[0] != MC_DROPOUT_SAMPLES:
        raise ValueError(
            "The number of returned reconstruction samples does not "
            "match MC_DROPOUT_SAMPLES. "
            f"Expected {MC_DROPOUT_SAMPLES}; "
            f"received {reconstruction_samples.shape[0]}."
        )

    if reconstruction_samples.shape[1:] != input_cube.shape:
        raise ValueError(
            "MC reconstruction sample dimensions do not match the "
            "input dimensions. "
            f"Samples: {tuple(reconstruction_samples.shape)}; "
            f"input: {tuple(input_cube.shape)}."
        )

    # ---------------------------------------------------------------
    # Calculate the mean reconstruction.
    # ---------------------------------------------------------------

    reconstruction_mean = reconstruction_samples.mean(dim=0)

    # ---------------------------------------------------------------
    # Estimate aleatoric and epistemic uncertainty.
    #
    # The estimator receives stochastic log-variance and reconstruction
    # samples. Its returned variance/std tensors should correspond to
    # the volume shape [B,C,D,H,W].
    # ---------------------------------------------------------------

    uncertainty_estimator = PredictiveUncertaintyEstimator(
        min_log_variance=LOG_VARIANCE_MIN,
        max_log_variance=LOG_VARIANCE_MAX,
    ).to(device)

    uncertainty_results = uncertainty_estimator(
        log_variance_samples,
        reconstruction_samples,
    )

    required_uncertainty_keys = (
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "aleatoric_std",
        "epistemic_std",
        "predictive_std",
    )

    for key in required_uncertainty_keys:
        if key not in uncertainty_results:
            raise KeyError(
                "PredictiveUncertaintyEstimator output is missing "
                f"required key: {key}"
            )

    aleatoric_variance = uncertainty_results["aleatoric_variance"]
    epistemic_variance = uncertainty_results["epistemic_variance"]
    predictive_variance = uncertainty_results["predictive_variance"]

    aleatoric_std = uncertainty_results["aleatoric_std"]
    epistemic_std = uncertainty_results["epistemic_std"]
    predictive_std = uncertainty_results["predictive_std"]

    # ---------------------------------------------------------------
    # Validate uncertainty outputs.
    #
    # All these tensors must be reduced over the MC sample dimension
    # and match the reconstruction volume dimensions.
    # ---------------------------------------------------------------

    expected_volume_shape = reconstruction_mean.shape

    uncertainty_tensors = {
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "predictive_variance": predictive_variance,
        "aleatoric_std": aleatoric_std,
        "epistemic_std": epistemic_std,
        "predictive_std": predictive_std,
    }

    for name, tensor in uncertainty_tensors.items():
        validate_tensor(tensor, name)

        if tensor.shape != expected_volume_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{tuple(expected_volume_shape)}; "
                f"received {tuple(tensor.shape)}. "
                "Check PredictiveUncertaintyEstimator's output "
                "dimensions and MC sample reduction."
            )

    # Variances and standard deviations should not be negative.
    for name, tensor in (
        ("aleatoric_variance", aleatoric_variance),
        ("epistemic_variance", epistemic_variance),
        ("predictive_variance", predictive_variance),
        ("aleatoric_std", aleatoric_std),
        ("epistemic_std", epistemic_std),
        ("predictive_std", predictive_std),
    ):
        if torch.any(tensor < 0).item():
            raise ValueError(
                f"{name} contains negative values."
            )

    # ---------------------------------------------------------------
    # Enforce observed-data consistency on the mean reconstruction.
    #
    # This projection changes the final reconstruction at observed
    # positions but does not silently change the uncertainty estimates.
    # ---------------------------------------------------------------

    reconstruction = enforce_observed_data_consistency(
        reconstruction_mean,
        corrupted_cube,
        mask,
    )

    # ---------------------------------------------------------------
    # Verify observed-data preservation.
    # ---------------------------------------------------------------

    observed_error = calculate_observed_preservation_error(
        reconstruction,
        corrupted_cube,
        mask,
    )

    if observed_error > OBSERVED_PRESERVATION_TOLERANCE:
        raise RuntimeError(
            "Observed-data preservation error exceeds the configured "
            "tolerance. "
            f"Error = {observed_error:.12e}; "
            f"tolerance = {OBSERVED_PRESERVATION_TOLERANCE:.12e}."
        )

    # ---------------------------------------------------------------
    # Return results.
    # ---------------------------------------------------------------

    return {
        "reconstruction": reconstruction,
        "predictive_std": predictive_std,
        "predictive_variance": predictive_variance,
        "aleatoric_variance": aleatoric_variance,
        "epistemic_variance": epistemic_variance,
        "aleatoric_std": aleatoric_std,
        "epistemic_std": epistemic_std,
        "reconstruction_samples": reconstruction_samples,
        "log_variance_samples": log_variance_samples,
        "observed_preservation_error": observed_error,
    }


# =====================================================================
# PREPARE VISUALIZATION SLICE
# =====================================================================

def prepare_visualization_slice(data, slice_index):
    """
    Extract a 2D slice from a 3D volume [D,H,W].

    SLICE_AXIS:
        0 = [H,W] at a selected depth index
        1 = [D,W] at a selected H index
        2 = [D,H] at a selected W index
    """

    array = tensor_to_numpy(data)
    array = np.squeeze(array)

    if array.ndim != 3:
        raise ValueError(
            "Visualization data must be a 3D volume after squeezing. "
            f"Received shape: {array.shape}."
        )

    if SLICE_AXIS not in (0, 1, 2):
        raise ValueError(
            f"Unsupported SLICE_AXIS: {SLICE_AXIS}"
        )

    if not 0 <= slice_index < array.shape[SLICE_AXIS]:
        raise IndexError(
            f"Slice index {slice_index} is invalid for axis "
            f"{SLICE_AXIS} with size {array.shape[SLICE_AXIS]}."
        )

    if SLICE_AXIS == 0:
        return array[slice_index, :, :]

    if SLICE_AXIS == 1:
        return array[:, slice_index, :]

    return array[:, :, slice_index]


# =====================================================================
# CREATE GALLERY FIGURE
# =====================================================================

def create_gallery_figure(
    corrupted,
    target,
    reconstruction,
    predictive_std,
    sample_index,
):
    """Create the five-panel reconstruction and uncertainty figure."""

    # Convert [B,C,D,H,W] or [C,D,H,W] to [D,H,W].
    volumes = {
        "corrupted": np.squeeze(tensor_to_numpy(corrupted)),
        "target": np.squeeze(tensor_to_numpy(target)),
        "reconstruction": np.squeeze(
            tensor_to_numpy(reconstruction)
        ),
        "predictive_std": np.squeeze(
            tensor_to_numpy(predictive_std)
        ),
    }

    for name, array in volumes.items():
        if array.ndim != 3:
            raise ValueError(
                f"{name} must be a 3D volume after squeezing. "
                f"Received shape {array.shape}."
            )

    reference_shape = volumes["target"].shape

    for name, array in volumes.items():
        if array.shape != reference_shape:
            raise ValueError(
                f"{name} shape {array.shape} does not match target "
                f"shape {reference_shape}."
            )

    # Calculate absolute reconstruction error.
    absolute_error = np.abs(
        volumes["target"] - volumes["reconstruction"]
    )

    # Choose the central slice along the selected axis.
    slice_index = reference_shape[SLICE_AXIS] // 2

    corrupted_slice = prepare_visualization_slice(
        volumes["corrupted"], slice_index
    )
    target_slice = prepare_visualization_slice(
        volumes["target"], slice_index
    )
    reconstruction_slice = prepare_visualization_slice(
        volumes["reconstruction"], slice_index
    )
    error_slice = prepare_visualization_slice(
        absolute_error, slice_index
    )
    uncertainty_slice = prepare_visualization_slice(
        volumes["predictive_std"], slice_index
    )

    # Create five panels.
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )

    panels = [
        (
            corrupted_slice,
            "Input / Corrupted Data",
            "gray",
        ),
        (
            target_slice,
            "Ground Truth",
            "gray",
        ),
        (
            reconstruction_slice,
            "Reconstruction",
            "gray",
        ),
        (
            error_slice,
            "Absolute Error",
            "hot",
        ),
        (
            uncertainty_slice,
            "Predictive Uncertainty (Std.)",
            "hot",
        ),
    ]

    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(
            image,
            cmap=cmap,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Spatial axis")
        axis.set_ylabel(
            "Depth" if SLICE_AXIS == 0 else "Spatial axis"
        )

    figure.suptitle(
        "Physics-Informed 3D Seismic Reconstruction "
        f"— Sample {sample_index + 1}",
        fontsize=14,
    )

    figure.tight_layout()

    return figure


# =====================================================================
# GENERATE GALLERY
# =====================================================================

def generate_gallery():
    """Load the trained model and generate reconstruction figures."""

    print()
    print("=" * 70)
    print("FINAL PhD RECONSTRUCTION GALLERY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Resolve device.
    # ---------------------------------------------------------------

    device = resolve_device()

    print(f"\nExperiment       : {EXPERIMENT_NAME}")
    print(f"Configured device: {DEVICE}")
    print(f"Actual device    : {device}")
    print(f"MC samples       : {MC_DROPOUT_SAMPLES}")
    print(f"Gallery samples  : {GALLERY_NUMBER_OF_SAMPLES}")

    if MC_DROPOUT_SAMPLES <= 0:
        raise ValueError(
            "MC_DROPOUT_SAMPLES must be greater than zero."
        )

    # ---------------------------------------------------------------
    # Check checkpoint.
    # ---------------------------------------------------------------

    print(f"\nCheckpoint       : {CHECKPOINT_FILE}")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            "The required checkpoint does not exist:\n"
            f"{CHECKPOINT_FILE}"
        )

    # ---------------------------------------------------------------
    # Build configured dataset.
    # ---------------------------------------------------------------

    print("\nBuilding configured dataset...")
    dataset = build_dataset()

    print(f"Dataset samples  : {len(dataset)}")

    if len(dataset) <= 0:
        raise ValueError(
            "The configured dataset contains no samples."
        )

    number_of_samples = min(
        int(GALLERY_NUMBER_OF_SAMPLES),
        len(dataset),
    )

    if number_of_samples <= 0:
        raise ValueError(
            "GALLERY_NUMBER_OF_SAMPLES must be greater than zero."
        )

    # ---------------------------------------------------------------
    # Build the trained network architecture.
    #
    # The pressure-wavefield output is configured using the acoustic
    # time-step setting. Do not substitute an Eikonal travel-time
    # output here.
    # ---------------------------------------------------------------

    print("\nBuilding Network3D...")
    print(f"Attention        : {USE_ATTENTION}")
    print(f"Residual         : {USE_RESIDUAL}")
    print(f"Uncertainty head : {USE_UNCERTAINTY}")
    print(
        f"Acoustic time steps: {config.ACOUSTIC_TIME_STEPS}"
    )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
        pressure_time_steps=config.ACOUSTIC_TIME_STEPS,
    ).to(device)

    # ---------------------------------------------------------------
    # Load checkpoint once.
    # ---------------------------------------------------------------

    print("\nLoading best_model.pth...")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")

    # ---------------------------------------------------------------
    # Prepare output directory.
    # ---------------------------------------------------------------

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Gallery directory : {GALLERY_DIR}")

    # ---------------------------------------------------------------
    # Process requested dataset samples.
    # ---------------------------------------------------------------

    for sample_index in range(number_of_samples):

        print()
        print("-" * 70)
        print(
            f"Processing sample {sample_index + 1}/"
            f"{number_of_samples}"
        )
        print("-" * 70)

        sample = dataset[sample_index]

        corrupted, target, mask = extract_dataset_sample(sample)

        corrupted_input = prepare_model_input(
            corrupted, device
        )
        target_input = prepare_model_input(
            target, device
        )
        observation_mask = prepare_mask(
            mask, device
        )

        # Verify that all volumes have identical shapes.
        if corrupted_input.shape != target_input.shape:
            raise ValueError(
                "Corrupted input and ground truth must have "
                "identical shapes."
            )

        if corrupted_input.shape != observation_mask.shape:
            raise ValueError(
                "Corrupted input and observation mask must have "
                "identical shapes."
            )

        # -----------------------------------------------------------
        # Run MC Dropout and calculate predictive uncertainty.
        # -----------------------------------------------------------

        results = run_mc_prediction(
            model=model,
            input_cube=corrupted_input,
            mask=observation_mask,
            corrupted_cube=corrupted_input,
            device=device,
        )

        reconstruction = results["reconstruction"]
        predictive_std = results["predictive_std"]

        print(
            "Observed-data preservation error: "
            f"{results['observed_preservation_error']:.12e}"
        )

        # -----------------------------------------------------------
        # Generate the five-panel figure.
        # -----------------------------------------------------------

        figure = create_gallery_figure(
            corrupted=corrupted_input,
            target=target_input,
            reconstruction=reconstruction,
            predictive_std=predictive_std,
            sample_index=sample_index,
        )

        output_file = (
            GALLERY_DIR / f"sample_{sample_index:03d}.png"
        )

        try:
            figure.savefig(
                output_file,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )
        finally:
            plt.close(figure)

        print(f"Saved: {output_file}")

        # Explicitly release references to large sample tensors before
        # moving on to the next gallery item.
        del results
        del reconstruction
        del predictive_std
        del corrupted_input
        del target_input
        del observation_mask

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Completion.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("RECONSTRUCTION GALLERY COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput directory:\n{GALLERY_DIR}\n")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    generate_gallery()