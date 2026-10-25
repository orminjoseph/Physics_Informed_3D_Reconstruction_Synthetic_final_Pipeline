"""
======================================================================
UNCERTAINTY ANALYSIS
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Evaluate and visualize predictive uncertainty for the trained
synthetic acoustic-wave-equation reconstruction model.

Uncertainty decomposition
-------------------------
    Predictive Variance
        = Aleatoric Variance + Epistemic Variance

Aleatoric variance:
    Mean(exp(log_variance_samples))

Epistemic variance:
    Variance(reconstruction_samples)

Predictive standard deviation:
    Sqrt(predictive variance)

Outputs
-------
    uncertainty_analysis.png

The figure contains:
    1. Mean reconstruction
    2. Aleatoric uncertainty
    3. Epistemic uncertainty
    4. Predictive uncertainty

Important
---------
- The trained checkpoint is loaded without retraining.
- The acoustic pressure-wavefield output is not Eikonal travel time.
- Pressure-wavefield MC samples are not retained.
- Reconstruction and log-variance samples are used for uncertainty.
- This module does not perform formal uncertainty calibration,
  uncertainty-error correlation, ablation, robustness, or statistical
  significance analysis.

Author:
Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

from pathlib import Path

import torch
import matplotlib.pyplot as plt

from models.network import Network3D
from models.mc_dropout import MCDropout3D

from models.predictive_uncertainty import (
    PredictiveUncertaintyEstimator,
)

from dataset.build_dataset import build_dataset

from utils import config

from utils.config import (
    CHECKPOINT_DIR,
    EXPERIMENT_NAME,
    LOG_VARIANCE_MIN,
    LOG_VARIANCE_MAX,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    DEVICE as CONFIG_DEVICE,
    MC_DROPOUT_SAMPLES,
)


# =====================================================================
# OUTPUT PATHS
# =====================================================================

CHECKPOINT_PATH = (
    Path(CHECKPOINT_DIR) / "best_model.pth"
)

OUTPUT_DIRECTORY = (
    Path(REPORT_DIR) / "uncertainty"
)

FIGURE_PATH = (
    OUTPUT_DIRECTORY / "uncertainty_analysis.png"
)

FIGURE_DPI = 300


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device():
    """Resolve DEVICE: cpu, cuda, or auto."""

    configured_device = str(CONFIG_DEVICE).strip().lower()

    if configured_device == "cpu":
        return torch.device("cpu")

    if configured_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' was requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    if configured_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    raise ValueError(
        f"Invalid DEVICE configuration: {CONFIG_DEVICE!r}. "
        "Expected 'cpu', 'cuda', or 'auto'."
    )


DEVICE = resolve_device()


# =====================================================================
# CONFIGURATION VALIDATION
# =====================================================================

def validate_configuration():
    """Validate settings required for uncertainty analysis."""

    if MC_DROPOUT_SAMPLES < 2:
        raise ValueError(
            "MC_DROPOUT_SAMPLES must be at least 2 to estimate "
            "epistemic uncertainty."
        )

    if not USE_UNCERTAINTY:
        raise RuntimeError(
            "USE_UNCERTAINTY=False. Enable the uncertainty head "
            "before running uncertainty analysis."
        )

    if LOG_VARIANCE_MIN >= LOG_VARIANCE_MAX:
        raise ValueError(
            "LOG_VARIANCE_MIN must be smaller than LOG_VARIANCE_MAX."
        )

    if not hasattr(config, "ACOUSTIC_TIME_STEPS"):
        raise AttributeError(
            "utils.config must define ACOUSTIC_TIME_STEPS for the "
            "acoustic-wave model."
        )

    if int(config.ACOUSTIC_TIME_STEPS) <= 0:
        raise ValueError(
            "ACOUSTIC_TIME_STEPS must be a positive integer."
        )


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_model():
    """
    Load the trained acoustic-wave reconstruction network.

    The model architecture must match the architecture used to create
    the checkpoint, including ACOUSTIC_TIME_STEPS.
    """

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = Network3D(
        use_attention=USE_ATTENTION,
        use_residual=USE_RESIDUAL,
        use_uncertainty=USE_UNCERTAINTY,
        pressure_time_steps=config.ACOUSTIC_TIME_STEPS,
    ).to(DEVICE)

    # Load the existing checkpoint. No training is performed here.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format. Expected a dictionary "
            "containing model_state_dict, state_dict, or model weights."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Also support a raw state dictionary.
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


# =====================================================================
# INPUT VALIDATION AND PREPARATION
# =====================================================================

def prepare_input_cube(sample):
    """
    Extract a dataset input and return [B,C,D,H,W].

    Accepted input shapes:
        [D,H,W]
        [C,D,H,W]
        [B,C,D,H,W]
    """

    if not isinstance(sample, (tuple, list)) or len(sample) < 1:
        raise TypeError(
            "Expected a dataset sample containing the input cube."
        )

    x = sample[0]

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    x = x.to(dtype=torch.float32)

    if x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:
        x = x.unsqueeze(0)
    elif x.ndim != 5:
        raise ValueError(
            "Input must have shape [D,H,W], [C,D,H,W], or "
            "[B,C,D,H,W]. "
            f"Received {tuple(x.shape)}."
        )

    if not torch.isfinite(x).all().item():
        raise ValueError("Input cube contains NaN or Inf values.")

    return x.to(DEVICE)


# =====================================================================
# UNCERTAINTY OUTPUT VALIDATION
# =====================================================================

def validate_uncertainty_outputs(
    uncertainty_results,
    expected_shape,
):
    """Validate uncertainty keys, shapes, and numerical values."""

    required_keys = (
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "aleatoric_std",
        "epistemic_std",
        "predictive_std",
    )

    for key in required_keys:
        if key not in uncertainty_results:
            raise KeyError(
                "PredictiveUncertaintyEstimator did not return "
                f"the required key {key!r}."
            )

        value = uncertainty_results[key]

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Uncertainty output {key!r} must be a tensor."
            )

        if value.shape != expected_shape:
            raise ValueError(
                f"Uncertainty output {key!r} has shape "
                f"{tuple(value.shape)}; expected "
                f"{tuple(expected_shape)}."
            )

        if not torch.isfinite(value).all().item():
            raise ValueError(
                f"Uncertainty output {key!r} contains NaN or Inf."
            )

    for key in (
        "aleatoric_variance",
        "epistemic_variance",
        "predictive_variance",
        "aleatoric_std",
        "epistemic_std",
        "predictive_std",
    ):
        if torch.any(uncertainty_results[key] < 0).item():
            raise ValueError(
                f"Uncertainty output {key!r} contains negative values."
            )


# =====================================================================
# UNCERTAINTY COMPUTATION
# =====================================================================

@torch.no_grad()
def compute_uncertainty(
    model,
    x,
    num_mc_samples=None,
):
    """
    Compute MC reconstruction and predictive uncertainty.

    Parameters
    ----------
    model:
        Trained Network3D model.

    x:
        Input tensor with shape [B,C,D,H,W].

    num_mc_samples:
        Number of stochastic reconstruction passes. Defaults to
        MC_DROPOUT_SAMPLES from utils.config.

    Returns
    -------
    dict:
        Mean reconstruction and uncertainty components, returned
        on CPU for subsequent visualization.
    """

    if num_mc_samples is None:
        num_mc_samples = MC_DROPOUT_SAMPLES

    if not isinstance(num_mc_samples, int) or num_mc_samples < 2:
        raise ValueError(
            "num_mc_samples must be an integer of at least 2."
        )

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor.")

    if x.ndim != 5:
        raise ValueError(
            "Input must have shape [B,C,D,H,W]. "
            f"Received {tuple(x.shape)}."
        )

    if not torch.isfinite(x).all().item():
        raise ValueError("Input contains NaN or Inf values.")

    x = x.to(device=DEVICE, dtype=torch.float32)

    # ---------------------------------------------------------------
    # Run MC Dropout.
    #
    # The revised MCDropout3D defaults to
    # store_pressure_wavefield_samples=False.
    #
    # We need reconstruction and log-variance samples only.
    # Do not store the large acoustic pressure-wavefield samples.
    # ---------------------------------------------------------------

    mc_dropout = MCDropout3D(
        model=model,
        num_samples=num_mc_samples,
    )

    mc_results = mc_dropout.predict(x)

    if not isinstance(mc_results, dict):
        raise TypeError(
            "MCDropout3D.predict() must return a dictionary."
        )

    required_mc_keys = (
        "reconstruction_samples",
        "log_variance_samples",
    )

    for key in required_mc_keys:
        if key not in mc_results:
            raise KeyError(
                f"MCDropout3D output is missing {key!r}."
            )

    reconstruction_samples = mc_results[
        "reconstruction_samples"
    ]

    log_variance_samples = mc_results[
        "log_variance_samples"
    ]

    # ---------------------------------------------------------------
    # Validate MC sample dimensions.
    #
    # Expected:
    #     [N,B,C,D,H,W]
    # ---------------------------------------------------------------

    expected_shape = (
        num_mc_samples,
        *x.shape,
    )

    for name, tensor in (
        ("reconstruction_samples", reconstruction_samples),
        ("log_variance_samples", log_variance_samples),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Unexpected {name} shape. "
                f"Expected {expected_shape}; "
                f"received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"{name} contains NaN or Inf values."
            )

    # ---------------------------------------------------------------
    # Mean reconstruction.
    # ---------------------------------------------------------------

    reconstruction_mean = reconstruction_samples.mean(dim=0)

    # ---------------------------------------------------------------
    # Uncertainty decomposition.
    #
    # Use the project's existing estimator instead of independently
    # reimplementing the uncertainty formulas here.
    # ---------------------------------------------------------------

    uncertainty_estimator = PredictiveUncertaintyEstimator(
        min_log_variance=LOG_VARIANCE_MIN,
        max_log_variance=LOG_VARIANCE_MAX,
    ).to(DEVICE)

    uncertainty_results = uncertainty_estimator(
        log_variance_samples,
        reconstruction_samples,
    )

    validate_uncertainty_outputs(
        uncertainty_results,
        expected_shape=reconstruction_mean.shape,
    )

    # ---------------------------------------------------------------
    # Independently check the variance decomposition:
    #
    # predictive_variance =
    #     aleatoric_variance + epistemic_variance
    # ---------------------------------------------------------------

    decomposition_difference = torch.abs(
        uncertainty_results["predictive_variance"]
        - (
            uncertainty_results["aleatoric_variance"]
            + uncertainty_results["epistemic_variance"]
        )
    )

    maximum_decomposition_difference = float(
        decomposition_difference.max().item()
    )

    # ---------------------------------------------------------------
    # Return the outputs needed by the analysis.
    #
    # Do not return pressure-wavefield samples. They are not needed
    # for uncertainty decomposition or the four-panel visualization.
    # ---------------------------------------------------------------

    results = {
        "reconstruction_mean": reconstruction_mean,
        "aleatoric_variance": uncertainty_results[
            "aleatoric_variance"
        ],
        "epistemic_variance": uncertainty_results[
            "epistemic_variance"
        ],
        "predictive_variance": uncertainty_results[
            "predictive_variance"
        ],
        "aleatoric_std": uncertainty_results["aleatoric_std"],
        "epistemic_std": uncertainty_results["epistemic_std"],
        "predictive_std": uncertainty_results["predictive_std"],
        "maximum_decomposition_difference":
            maximum_decomposition_difference,
        "num_mc_samples": num_mc_samples,
    }

    # Transfer final analysis tensors to CPU. Intermediate MC samples
    # are not included in the returned dictionary.
    results = {
        key: value.detach().cpu()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in results.items()
    }

    # Release references to large MC tensors.
    del mc_results
    del reconstruction_samples
    del log_variance_samples
    del uncertainty_results
    del uncertainty_estimator
    del mc_dropout

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize_uncertainty(results, output_path=None):
    """
    Generate a four-panel central-slice visualization.

    Panels:
        1. Mean reconstruction
        2. Aleatoric uncertainty standard deviation
        3. Epistemic uncertainty standard deviation
        4. Predictive uncertainty standard deviation
    """

    required_keys = (
        "reconstruction_mean",
        "aleatoric_std",
        "epistemic_std",
        "predictive_std",
    )

    for key in required_keys:
        if key not in results:
            raise KeyError(
                f"Visualization results are missing {key!r}."
            )

    # ---------------------------------------------------------------
    # Extract the first batch and channel.
    # ---------------------------------------------------------------

    volumes = {}

    for key in required_keys:
        tensor = results[key]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{key} must be a torch.Tensor."
            )

        if tensor.ndim != 5:
            raise ValueError(
                f"{key} must have shape [B,C,D,H,W]. "
                f"Received {tuple(tensor.shape)}."
            )

        volumes[key] = tensor[0, 0].numpy()

    reference_shape = volumes["reconstruction_mean"].shape

    for key, volume in volumes.items():
        if volume.ndim != 3:
            raise ValueError(
                f"{key} must represent a 3D volume."
            )

        if volume.shape != reference_shape:
            raise ValueError(
                f"{key} has shape {volume.shape}; expected "
                f"{reference_shape}."
            )

    # ---------------------------------------------------------------
    # Central depth slice.
    # ---------------------------------------------------------------

    depth_index = reference_shape[0] // 2

    panels = (
        (
            volumes["reconstruction_mean"][depth_index],
            "Mean Reconstruction",
            "gray",
        ),
        (
            volumes["aleatoric_std"][depth_index],
            "Aleatoric Uncertainty (STD)",
            "hot",
        ),
        (
            volumes["epistemic_std"][depth_index],
            "Epistemic Uncertainty (STD)",
            "hot",
        ),
        (
            volumes["predictive_std"][depth_index],
            "Predictive Uncertainty (STD)",
            "hot",
        ),
    )

    # ---------------------------------------------------------------
    # Output path.
    # ---------------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:
        output_path = FIGURE_PATH
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------------
    # Create and save the figure.
    # ---------------------------------------------------------------

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    try:
        for axis, (image, title, cmap) in zip(
            axes.flat,
            panels,
        ):
            axis.imshow(
                image,
                cmap=cmap,
                aspect="auto",
            )
            axis.set_title(title)
            axis.set_xlabel("Inline / Crossline")
            axis.set_ylabel("Spatial sample")

        figure.suptitle(
            "Predictive Uncertainty Analysis",
            fontsize=14,
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    return str(output_path)


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

def analyze_uncertainty():
    """Run the complete uncertainty-analysis workflow."""

    validate_configuration()

    print()
    print("=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    print(f"Experiment          : {EXPERIMENT_NAME}")
    print(f"Device              : {DEVICE}")
    print(f"MC samples          : {MC_DROPOUT_SAMPLES}")
    print(f"Acoustic time steps : {config.ACOUSTIC_TIME_STEPS}")
    print(f"Checkpoint          : {CHECKPOINT_PATH}")
    print(f"Output directory    : {OUTPUT_DIRECTORY}")

    # ---------------------------------------------------------------
    # Build dataset.
    # ---------------------------------------------------------------

    print("\nBuilding dataset...")

    dataset = build_dataset()

    print(f"Dataset length      : {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("The configured dataset is empty.")

    # ---------------------------------------------------------------
    # Load trained model.
    # ---------------------------------------------------------------

    print("\nLoading trained model...")

    model = load_model()

    print("Model loaded successfully.")

    # ---------------------------------------------------------------
    # Select the first dataset sample.
    # ---------------------------------------------------------------

    print("\nPreparing representative sample...")

    input_cube = prepare_input_cube(dataset[0])

    print(f"Input shape         : {tuple(input_cube.shape)}")

    # ---------------------------------------------------------------
    # Compute uncertainty.
    # ---------------------------------------------------------------

    print("\nComputing MC Dropout uncertainty...")

    results = compute_uncertainty(
        model=model,
        x=input_cube,
        num_mc_samples=MC_DROPOUT_SAMPLES,
    )

    print("Uncertainty decomposition completed.")

    # ---------------------------------------------------------------
    # Generate visualization.
    # ---------------------------------------------------------------

    print("\nGenerating uncertainty visualization...")

    figure_path = visualize_uncertainty(results)

    print(f"Saved figure        : {figure_path}")

    # ---------------------------------------------------------------
    # Summary statistics.
    # ---------------------------------------------------------------

    summary_keys = (
        ("aleatoric_variance", "Aleatoric variance mean"),
        ("epistemic_variance", "Epistemic variance mean"),
        ("predictive_variance", "Predictive variance mean"),
        ("aleatoric_std", "Aleatoric STD mean"),
        ("epistemic_std", "Epistemic STD mean"),
        ("predictive_std", "Predictive STD mean"),
    )

    print()
    print("=" * 70)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("=" * 70)

    for key, label in summary_keys:
        mean_value = float(results[key].mean().item())
        print(f"{label:<28}: {mean_value:.8f}")

    print(
        "Maximum decomposition difference: "
        f"{results['maximum_decomposition_difference']:.10e}"
    )

    print(f"MC samples                       : {results['num_mc_samples']}")
    print(f"Figure                           : {figure_path}")

    print("=" * 70)

    return {
        "results": results,
        "figure_path": figure_path,
    }


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    analyze_uncertainty()