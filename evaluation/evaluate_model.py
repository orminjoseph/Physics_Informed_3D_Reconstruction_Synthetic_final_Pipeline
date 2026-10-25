"""
======================================================================
FINAL MODEL EVALUATION
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Final orchestration layer for evaluating the trained acoustic model.

Responsibilities
----------------
1. Use the active global configuration.
2. Build the evaluation dataset.
3. Adapt the dataset to the centralized Evaluator interface.
4. Construct the configured Network3D architecture.
5. Resolve the computational device from utils.config.
6. Load best_model.pth using the established Predictor.
7. Run the centralized Evaluator.
8. Save the complete evaluation results to CSV.

Checkpoint convention
---------------------
Evaluation:
    outputs/<EXPERIMENT_NAME>/checkpoints/best_model.pth

Training resume:
    outputs/<EXPERIMENT_NAME>/checkpoints/latest_checkpoint.pth

Important
---------
The configuration file remains the single source of truth for:
- model architecture
- acoustic pressure-wavefield time steps
- uncertainty configuration
- computational device
- DataLoader configuration
- experiment-specific output paths

The Predictor handles checkpoint loading.
The Evaluator performs metric calculations.

Author: Ormin Joseph
======================================================================
"""

import os

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from models.network import Network3D
from dataset.build_dataset import build_dataset
from evaluation.evaluator import Evaluator
from inference.predictor import Predictor

from utils import config

from utils.config import (
    EXPERIMENT_NAME,
    CHECKPOINT_DIR,
    REPORT_DIR,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    MC_DROPOUT_SAMPLES,
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    BATCH_SIZE,
)


# ======================================================================
# EVALUATION DATASET ADAPTER
# ======================================================================

class EvaluationDatasetAdapter(Dataset):
    """
    Adapt the project's tuple-based dataset to the dictionary-based
    interface expected by Evaluator.

    Original dataset sample:
        (
            input_cube,
            target_cube,
            mask,
            velocity_model,
            mask_type,
            geological_mode
        )

    Evaluator-compatible sample:
        {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
            "velocity": velocity_model,
            "mask_type": mask_type,
            "geological_mode": geological_mode
        }

    Optional metadata is preserved when supplied by the dataset.
    The underlying dataset is not modified.
    """

    def __init__(self, dataset):
        if dataset is None:
            raise ValueError("dataset cannot be None.")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        if not isinstance(sample, (tuple, list)):
            raise TypeError(
                "Expected dataset sample to be a tuple or list. "
                f"Received: {type(sample)}"
            )

        if len(sample) < 3:
            raise ValueError(
                "Dataset sample must contain at least "
                "input, target, and mask."
            )

        input_cube = sample[0]
        target_cube = sample[1]
        mask = sample[2]

        evaluation_sample = {
            "input": input_cube,
            "target": target_cube,
            "mask": mask,
        }

        if len(sample) >= 4:
            evaluation_sample["velocity"] = sample[3]

        if len(sample) >= 5:
            evaluation_sample["mask_type"] = sample[4]

        if len(sample) >= 6:
            evaluation_sample["geological_mode"] = sample[5]

        return evaluation_sample


# ======================================================================
# DEVICE RESOLUTION
# ======================================================================

def resolve_device():
    """
    Resolve the computational device according to utils.config.

    Supported modes:
        "cpu"  - Force CPU.
        "cuda" - Require CUDA.
        "auto" - Use CUDA when available; otherwise use CPU.
    """

    configured_device = str(DEVICE).lower()

    if configured_device == "cpu":
        return torch.device("cpu")

    if configured_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE='cuda' was requested, but CUDA is not "
                "available on this system."
            )

        return torch.device("cuda")

    if configured_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    raise ValueError(
        "Invalid DEVICE configuration: "
        f"{DEVICE!r}. Expected 'cpu', 'cuda', or 'auto'."
    )


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_positive_integer(value, name):
    """Validate an integer setting without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{name} must be a positive integer; received {value!r}."
        )


# ======================================================================
# MODEL EVALUATION
# ======================================================================

def evaluate(
    model_override=None,
    mc_samples=None,
    batch_size=None,
):
    """
    Evaluate the trained best model for the active experiment.

    Parameters
    ----------
    model_override:
        Optional externally supplied model. When None, construct
        Network3D using the active configuration.

    mc_samples:
        Number of stochastic forward passes. Defaults to
        MC_DROPOUT_SAMPLES.

    batch_size:
        Evaluation DataLoader batch size. Defaults to BATCH_SIZE.

    Returns
    -------
    dict
        Aggregated evaluation results returned by Evaluator.
    """

    print()
    print("=" * 80)
    print("FINAL MODEL EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Resolve evaluation settings
    # ------------------------------------------------------------------

    if mc_samples is None:
        mc_samples = MC_DROPOUT_SAMPLES

    if batch_size is None:
        batch_size = BATCH_SIZE

    validate_positive_integer(mc_samples, "mc_samples")
    validate_positive_integer(batch_size, "batch_size")

    if not isinstance(config.ACOUSTIC_TIME_STEPS, int):
        raise ValueError(
            "config.ACOUSTIC_TIME_STEPS must be an integer."
        )

    validate_positive_integer(
        config.ACOUSTIC_TIME_STEPS,
        "config.ACOUSTIC_TIME_STEPS",
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = resolve_device()

    print()
    print("Experiment          :", EXPERIMENT_NAME)
    print("Configured device   :", DEVICE)
    print("Using device        :", device)
    print("MC samples          :", mc_samples)
    print("Batch size          :", batch_size)
    print("Acoustic time steps :", config.ACOUSTIC_TIME_STEPS)

    if device.type == "cuda":
        print(
            "CUDA device         :",
            torch.cuda.get_device_name(device),
        )
        print(
            "CUDA device count   :",
            torch.cuda.device_count(),
        )

    # ------------------------------------------------------------------
    # Build evaluation dataset
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("BUILDING EVALUATION DATASET")
    print("-" * 80)

    dataset = build_dataset()

    if dataset is None:
        raise RuntimeError("build_dataset() returned None.")

    dataset_length = len(dataset)

    print("Dataset length      :", dataset_length)

    if dataset_length == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    # ------------------------------------------------------------------
    # Adapt dataset
    # ------------------------------------------------------------------

    evaluation_dataset = EvaluationDatasetAdapter(dataset)

    print(
        "Dataset adapter     :",
        type(evaluation_dataset).__name__,
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    dataloader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    number_of_batches = len(dataloader)

    print("DataLoader batches  :", number_of_batches)
    print("DataLoader workers  :", NUM_WORKERS)
    print("Pin memory          :", PIN_MEMORY)

    if number_of_batches == 0:
        raise RuntimeError("Evaluation DataLoader is empty.")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("BUILDING FINAL NETWORK")
    print("-" * 80)

    if model_override is None:
        model = Network3D(
            use_attention=USE_ATTENTION,
            use_residual=USE_RESIDUAL,
            use_uncertainty=USE_UNCERTAINTY,
            pressure_time_steps=config.ACOUSTIC_TIME_STEPS,
        )
    else:
        model = model_override

    print("Attention enabled   :", USE_ATTENTION)
    print("Residual enabled    :", USE_RESIDUAL)
    print("Uncertainty enabled :", USE_UNCERTAINTY)

    model = model.to(device)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth",
    )

    print()
    print("-" * 80)
    print("LOADING BEST MODEL CHECKPOINT")
    print("-" * 80)
    print("Checkpoint          :", checkpoint)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            "\nBest model checkpoint not found:\n"
            f"{checkpoint}\n\n"
            "Complete training and ensure best_model.pth exists "
            "in the configured checkpoint directory."
        )

    # Use the established Predictor checkpoint loader.
    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    # Retrieve the model instance after checkpoint loading.
    model = predictor.model

    print("Checkpoint loaded successfully.")
    print("Model is in evaluation mode:", not model.training)

    if model.training:
        raise RuntimeError(
            "Predictor did not leave the model in evaluation mode."
        )

    # ------------------------------------------------------------------
    # Centralized evaluator
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("INITIALIZING CENTRALIZED EVALUATOR")
    print("-" * 80)

    evaluator = Evaluator(
        model=model,
        device=device,
        mc_samples=mc_samples,
    )

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("RUNNING CENTRALIZED MODEL EVALUATION")
    print("-" * 80)

    results = evaluator.evaluate(dataloader)

    # ------------------------------------------------------------------
    # Validate results
    # ------------------------------------------------------------------

    if not isinstance(results, dict):
        raise TypeError(
            "Evaluator.evaluate() must return a dictionary. "
            f"Received: {type(results)}"
        )

    if not results:
        raise RuntimeError(
            "Evaluator returned an empty results dictionary."
        )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL MODEL EVALUATION RESULTS")
    print("=" * 80)

    for key, value in results.items():
        if isinstance(value, (float, int)):
            print(f"{key:<35}: {value:.6f}")
        else:
            print(f"{key:<35}: {value}")

    # ------------------------------------------------------------------
    # Create report directory
    # ------------------------------------------------------------------

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    output_file = os.path.join(
        REPORT_DIR,
        "evaluation_metrics.csv",
    )

    pd.DataFrame([results]).to_csv(
        output_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # Confirm output
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("EVALUATION OUTPUT")
    print("-" * 80)
    print("Evaluation results saved:")
    print(output_file)

    print()
    print("Experiment directory:")
    print(os.path.join("outputs", EXPERIMENT_NAME))

    print()
    print("=" * 80)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 80)

    return results


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    evaluate()