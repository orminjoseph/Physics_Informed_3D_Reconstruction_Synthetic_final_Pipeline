
"""
======================================================================
FINAL TRAINING ENTRY POINT
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This module performs final-model training only.

The independent controlled evaluation matrix is executed separately.

Physics constraint:
    Acoustic wave equation only.

No Eikonal implementation is used in this training entry point.

Author: Ormin Joseph
======================================================================
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset
from dataset.dataloader import create_dataloader

from models.network import Network3D

from losses.total_loss import TotalLoss

# If your acoustic loss file has a different name, adjust
# this import to match its actual filename.
from losses.physics_loss import (
    AcousticWaveEquationLoss,
)

from trainer.trainer import Trainer

from utils.experiment_manager import ExperimentManager
from utils import config


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE RESOLUTION
# ======================================================================

def resolve_device() -> torch.device:
    """Resolve the configured computational device."""

    configured_device = str(config.DEVICE).lower().strip()

    if configured_device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if configured_device == "cuda":
        if not torch.cuda.is_available():
            print(
                "WARNING: CUDA was requested but is unavailable. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

        return torch.device("cuda")

    if configured_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported DEVICE setting: {config.DEVICE!r}. "
        "Expected 'auto', 'cuda', or 'cpu'."
    )


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_training_configuration() -> None:
    """Validate the configuration required for training."""

    if hasattr(config, "validate_config"):
        result = config.validate_config()

        if result is False:
            raise ValueError(
                "The project configuration validation failed."
            )

    if config.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    if config.NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be at least 1.")

    if config.LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive.")

    if config.WEIGHT_DECAY < 0:
        raise ValueError("WEIGHT_DECAY cannot be negative.")

    if not 0.0 < config.VALIDATION_SPLIT < 1.0:
        raise ValueError(
            "VALIDATION_SPLIT must be between 0 and 1."
        )

    if config.DATASET_MODE not in {"synthetic", "f3"}:
        raise ValueError(
            f"Unsupported DATASET_MODE: {config.DATASET_MODE!r}."
        )

    # This project uses acoustic wave-equation physics only.
    if config.PHYSICS_TYPE != "acoustic_wave_equation":
        raise ValueError(
            "This training entry point requires "
            "PHYSICS_TYPE='acoustic_wave_equation'."
        )

    if not config.USE_PHYSICS_LOSS:
        raise ValueError(
            "USE_PHYSICS_LOSS must be True for this experiment."
        )

    if not config.USE_ACOUSTIC_WAVE_EQUATION_LOSS:
        raise ValueError(
            "USE_ACOUSTIC_WAVE_EQUATION_LOSS must be True."
        )

    if config.LOSS_WEIGHTS.get("physics", 0.0) <= 0:
        raise ValueError(
            "LOSS_WEIGHTS['physics'] must be positive."
        )


# ======================================================================
# TRAINING ENTRY POINT
# ======================================================================

def train_final_model():
    """Build the model and execute final-model training."""

    print()
    print("=" * 78)
    print("FINAL MODEL TRAINING")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Validate configuration and establish reproducibility.
    # ------------------------------------------------------------------

    validate_training_configuration()
    set_global_seed(config.SEED)

    # ------------------------------------------------------------------
    # 2. Resolve device.
    # ------------------------------------------------------------------

    device = resolve_device()

    print()
    print(f"Dataset mode        : {config.DATASET_MODE}")
    print(f"Experiment name     : {config.EXPERIMENT_NAME}")
    print(f"Batch size          : {config.BATCH_SIZE}")
    print(f"Number of epochs    : {config.NUM_EPOCHS}")
    print(f"Learning rate       : {config.LEARNING_RATE}")
    print(f"Weight decay        : {config.WEIGHT_DECAY}")
    print(f"Validation split    : {config.VALIDATION_SPLIT}")
    print(f"Resolved device     : {device}")
    print(f"Physics type        : {config.PHYSICS_TYPE}")

    # ------------------------------------------------------------------
    # 3. Build dataset.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING DATASET")
    print("-" * 78)

    dataset = build_dataset()

    if dataset is None or len(dataset) == 0:
        raise RuntimeError(
            "The dataset is empty or build_dataset() returned None."
        )

    print(f"Dataset size        : {len(dataset)}")

    # ------------------------------------------------------------------
    # 4. Create deterministic training/validation split.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("CREATING TRAINING / VALIDATION SPLIT")
    print("-" * 78)

    train_dataset, validation_dataset = split_dataset(dataset)

    if len(train_dataset) == 0:
        raise RuntimeError("The training split is empty.")

    if len(validation_dataset) == 0:
        raise RuntimeError("The validation split is empty.")

    print(f"Training samples    : {len(train_dataset)}")
    print(f"Validation samples  : {len(validation_dataset)}")

    # ------------------------------------------------------------------
    # 5. Create DataLoaders.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("CREATING DATALOADERS")
    print("-" * 78)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError("The training DataLoader has no batches.")

    if len(validation_loader) == 0:
        raise RuntimeError(
            "The validation DataLoader has no batches."
        )

    print(f"Training batches    : {len(train_loader)}")
    print(f"Validation batches  : {len(validation_loader)}")

    # ------------------------------------------------------------------
    # 6. Construct the 3D Encoder-Decoder model.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING 3D ENCODER-DECODER MODEL")
    print("-" * 78)


    model = Network3D(
        use_attention=config.USE_ATTENTION,
        use_residual=config.USE_RESIDUAL,
        use_uncertainty=config.USE_UNCERTAINTY,
        pressure_time_steps=config.ACOUSTIC_TIME_STEPS,
    ).to(device)
    print(f"Attention enabled   : {config.USE_ATTENTION}")
    print(f"Residual enabled    : {config.USE_RESIDUAL}")
    print(f"Uncertainty enabled : {config.USE_UNCERTAINTY}")

    print(
        f"Acoustic pressure time steps: "
        f"{model.pressure_time_steps}"
    )
    # ------------------------------------------------------------------
    # 7. Construct the acoustic wave-equation loss.
    #
    # The acoustic loss evaluates:
    #
    #     (1 / v^2) * d2p/dt2 - Laplacian(p) - s
    #
    # Required physical inputs:
    #     pressure : [B, 1, T, D, H, W]
    #     velocity : [B, 1, D, H, W]
    #     source   : [B, 1, T, D, H, W]
    #
    # The Trainer and dataset must supply physically compatible tensors.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING ACOUSTIC WAVE-EQUATION LOSS")
    print("-" * 78)

    acoustic_loss = AcousticWaveEquationLoss(
        dt=config.ACOUSTIC_DT,
        dx=config.ACOUSTIC_DX,
        dy=config.ACOUSTIC_DY,
        dz=config.ACOUSTIC_DZ,
    ).to(device)

    print(f"Acoustic dt         : {config.ACOUSTIC_DT}")
    print(f"Acoustic dx         : {config.ACOUSTIC_DX}")
    print(f"Acoustic dy         : {config.ACOUSTIC_DY}")
    print(f"Acoustic dz         : {config.ACOUSTIC_DZ}")

    # ------------------------------------------------------------------
    # 8. Construct TotalLoss and connect the acoustic loss.
    #
    # IMPORTANT:
    # TotalLoss receives acoustic_loss explicitly.
    # Do not pass physics_loss_weights unless the TotalLoss constructor
    # has specifically been updated to accept that argument.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING TOTAL LOSS")
    print("-" * 78)

    criterion = TotalLoss(
        loss_weights=config.LOSS_WEIGHTS,
        data_range=config.SEISMIC_DATA_RANGE,
        acoustic_loss=acoustic_loss,
        log_variance_min=config.LOG_VARIANCE_MIN,
        log_variance_max=config.LOG_VARIANCE_MAX,
    ).to(device)

    print(f"Loss weights        : {config.LOSS_WEIGHTS}")
    print("Physics constraint  : Acoustic wave equation")
    print("Acoustic loss linked: Yes")

    # ------------------------------------------------------------------
    # 9. Construct optimizer.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING OPTIMIZER")
    print("-" * 78)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    print("Optimizer           : Adam")
    print(f"Learning rate       : {config.LEARNING_RATE}")
    print(f"Weight decay        : {config.WEIGHT_DECAY}")

    # ------------------------------------------------------------------
    # 10. Initialize experiment manager.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("INITIALIZING EXPERIMENT MANAGER")
    print("-" * 78)

    experiment_manager = ExperimentManager()

    checkpoint_directory = Path(
        experiment_manager.checkpoints
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Checkpoint directory: {checkpoint_directory}")

    # ------------------------------------------------------------------
    # 11. Initialize Trainer.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("INITIALIZING TRAINER")
    print("-" * 78)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
    )

    # ------------------------------------------------------------------
    # 12. Start or resume training.
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STARTING TRAINING")
    print("=" * 78)

    trainer.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
        epochs=config.NUM_EPOCHS,
        resume=True,
    )

    # ------------------------------------------------------------------
    # 13. Verify the expected best checkpoint.
    # ------------------------------------------------------------------

    best_checkpoint = (
        checkpoint_directory / "best_model.pth"
    )

    print()
    print("-" * 78)
    print("VERIFYING TRAINING OUTPUT")
    print("-" * 78)

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            "Training completed but the expected best checkpoint "
            f"was not found:\n{best_checkpoint}"
        )

    if best_checkpoint.stat().st_size == 0:
        raise RuntimeError(
            "The best checkpoint exists but is empty:\n"
            f"{best_checkpoint}"
        )

    print(f"Best checkpoint     : {best_checkpoint}")

    print()
    print("=" * 78)
    print("FINAL MODEL TRAINING COMPLETED")
    print("=" * 78)

    return trainer


# ======================================================================
# DIRECT SCRIPT EXECUTION
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 78)
    print("TRAINING ENTRY POINT")
    print("=" * 78)

    # Master safety switch: prevent accidental training.
    if not config.RUN_TRAINING:

        print()
        print("RUN_TRAINING=False")
        print("Training has NOT been started.")
        print(
            "Set RUN_TRAINING=True only after the "
            "pre-training integration checks have passed."
        )

    else:
        train_final_model()
