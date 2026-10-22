"""
======================================================================
FINAL TRAINING ENTRY POINT
======================================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

This module is responsible for:

    1. Resolving the computational device.
    2. Building the configured training dataset.
    3. Creating the deterministic training/validation split.
    4. Creating training and validation DataLoaders.
    5. Constructing the final 3D Encoder–Decoder model.
    6. Constructing the complete TotalLoss.
    7. Constructing the optimizer.
    8. Creating the Trainer.
    9. Starting or resuming model training.
   10. Reporting the location of the resulting checkpoint.

IMPORTANT
---------
This module performs TRAINING only.

The controlled 750-case evaluation matrix is NOT generated here.

The intended project workflow is:

    Training dataset
          |
          v
    Train final model
          |
          v
    best_model.pth
          |
          v
    Independent evaluation
          |
          v
    750 controlled cases

The master project pipeline controls whether training is executed.

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

from trainer.trainer import Trainer

from utils.experiment_manager import ExperimentManager

from utils import config


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_global_seed(seed: int) -> None:
    """
    Set the random seeds used by the training entry point.

    The dataset itself is responsible for its own sample-level
    deterministic generation. This function establishes reproducibility
    for the training process and any remaining Python/NumPy/PyTorch
    randomness.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE RESOLUTION
# ======================================================================

def resolve_device() -> torch.device:
    """
    Resolve config.DEVICE into a valid torch.device.

    Supported configuration policies:

        "auto"
        "cuda"
        "cpu"

    "auto" selects CUDA when available and otherwise CPU.
    """

    configured_device = str(config.DEVICE).lower().strip()

    if configured_device == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if configured_device == "cuda":

        if not torch.cuda.is_available():
            print(
                "WARNING: DEVICE='cuda' was requested, "
                "but CUDA is unavailable."
            )

            print("Falling back to CPU.")

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
    """
    Validate the minimum configuration required by the training stage.

    The complete configuration validator remains in utils/config.py.
    This function performs only training-stage checks.
    """

    if config.BATCH_SIZE < 1:
        raise ValueError(
            "BATCH_SIZE must be greater than or equal to 1."
        )

    if config.NUM_EPOCHS < 1:
        raise ValueError(
            "NUM_EPOCHS must be greater than or equal to 1."
        )

    if config.LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE must be greater than 0."
        )

    if config.WEIGHT_DECAY < 0:
        raise ValueError(
            "WEIGHT_DECAY cannot be negative."
        )

    if not 0.0 < config.VALIDATION_SPLIT < 1.0:
        raise ValueError(
            "VALIDATION_SPLIT must be between 0 and 1."
        )

    if config.DATASET_MODE not in {
        "synthetic",
        "f3",
    }:
        raise ValueError(
            f"Unsupported DATASET_MODE: {config.DATASET_MODE!r}."
        )


# ======================================================================
# TRAINING ENTRY POINT
# ======================================================================

def train_final_model():
    """
    Execute the complete final-model training stage.

    Returns
    -------
    Trainer
        The completed Trainer instance.

    Notes
    -----
    This function does not run the independent controlled evaluation
    matrix. Evaluation is performed after the best checkpoint has been
    produced.
    """

    print()
    print("=" * 78)
    print("FINAL MODEL TRAINING")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Validate the training configuration.
    # ------------------------------------------------------------------

    validate_training_configuration()

    # ------------------------------------------------------------------
    # 2. Establish reproducibility.
    # ------------------------------------------------------------------

    set_global_seed(config.SEED)

    # ------------------------------------------------------------------
    # 3. Resolve computational device.
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
    print(f"Configured device   : {config.DEVICE}")
    print(f"Resolved device     : {device}")

    # ------------------------------------------------------------------
    # 4. Build the configured dataset.
    #
    # build_dataset() is responsible for routing to the appropriate
    # dataset implementation according to DATASET_MODE.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING DATASET")
    print("-" * 78)

    dataset = build_dataset()

    if dataset is None:
        raise RuntimeError(
            "build_dataset() returned None. "
            "Training cannot continue."
        )

    if len(dataset) == 0:
        raise RuntimeError(
            "The training dataset is empty."
        )

    print(f"Dataset size        : {len(dataset)}")

    # ------------------------------------------------------------------
    # 5. Create the deterministic training/validation split.
    #
    # split_dataset() contains the project's coverage-aware splitting
    # logic and is therefore used instead of implementing another split
    # here.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("CREATING TRAINING / VALIDATION SPLIT")
    print("-" * 78)

    train_dataset, validation_dataset = split_dataset(
        dataset
    )


    if len(train_dataset) == 0:
        raise RuntimeError(
            "The training split is empty."
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "The validation split is empty."
        )

    print(f"Training samples    : {len(train_dataset)}")
    print(f"Validation samples  : {len(validation_dataset)}")

    # ------------------------------------------------------------------
    # 6. Create DataLoaders.
    #
    # Training data is shuffled.
    #
    # Validation data is deterministic and is not shuffled.
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
        raise RuntimeError(
            "The training DataLoader contains no batches."
        )

    if len(validation_loader) == 0:
        raise RuntimeError(
            "The validation DataLoader contains no batches."
        )

    print(f"Training batches    : {len(train_loader)}")
    print(f"Validation batches  : {len(validation_loader)}")

    # ------------------------------------------------------------------
    # 7. Construct the final 3D Encoder–Decoder model.
    #
    # All architectural switches come from utils.config.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING 3D ENCODER–DECODER MODEL")
    print("-" * 78)

    model = Network3D(
        use_attention=config.USE_ATTENTION,
        use_residual=config.USE_RESIDUAL,
        use_uncertainty=config.USE_UNCERTAINTY,
    )

    model = model.to(device)

    print(f"Attention enabled  : {config.USE_ATTENTION}")
    print(f"Residual enabled   : {config.USE_RESIDUAL}")
    print(f"Uncertainty enabled: {config.USE_UNCERTAINTY}")

    # ------------------------------------------------------------------
    # 8. Construct the complete TotalLoss.
    #
    # IMPORTANT:
    #
    # The actual TotalLoss constructor accepts:
    #
    #     loss_weights
    #     physics_loss_weights
    #     data_range
    #     log_variance_min
    #     log_variance_max
    #
    # It does NOT accept dx, dy, or dz as constructor arguments.
    #
    # Grid spacing is already handled by the physics-loss implementation
    # through the appropriate configuration/interface.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("BUILDING TOTAL LOSS")
    print("-" * 78)

    criterion = TotalLoss(
        loss_weights=config.LOSS_WEIGHTS,
        physics_loss_weights=config.PHYSICS_LOSS_WEIGHTS,
        data_range=config.SEISMIC_DATA_RANGE,
        log_variance_min=config.LOG_VARIANCE_MIN,
        log_variance_max=config.LOG_VARIANCE_MAX,
    )

    criterion = criterion.to(device)

    print(f"Loss weights        : {config.LOSS_WEIGHTS}")
    print(
        "Physics loss weights: "
        f"{config.PHYSICS_LOSS_WEIGHTS}"
    )

    # ------------------------------------------------------------------
    # 9. Construct the optimizer.
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
    # 10. Create the ExperimentManager.
    #
    # The ExperimentManager owns the experiment-level output structure
    # and checkpoint/report management.
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("INITIALIZING EXPERIMENT MANAGER")
    print("-" * 78)

    experiment_manager = ExperimentManager()

    # ------------------------------------------------------------------
    # 11. Verify the experiment checkpoint directory.
    # ------------------------------------------------------------------

    checkpoint_directory = Path(
        experiment_manager.checkpoints
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Checkpoint directory: "
        f"{checkpoint_directory}"
    )

    # ------------------------------------------------------------------
    # 12. Create the Trainer.
    #
    # The Trainer owns:
    #
    #     - training loop
    #     - validation loop
    #     - checkpoint creation
    #     - history
    #     - gradient handling
    #     - early stopping
    #     - TensorBoard/logging
    #
    # Therefore these responsibilities are deliberately not duplicated
    # here.
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
    # 13. Start or resume training.
    #
    # RESUME_EVALUATION is intentionally unrelated to training resume.
    # Training resume is controlled here by the existence of a previous
    # training checkpoint and the Trainer's resume mechanism.
    #
    # The master pipeline decides whether this function is called at
    # all through RUN_TRAINING.
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
    # 14. Verify that the expected best checkpoint exists.
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

    print(
        f"Best checkpoint    : {best_checkpoint}"
    )

    print()
    print("=" * 78)
    print("FINAL MODEL TRAINING COMPLETED")
    print("=" * 78)

    print(
        f"Best checkpoint:\n{best_checkpoint}"
    )

    print()

    return trainer


# ======================================================================
# DIRECT SCRIPT EXECUTION
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 78)
    print("TRAINING ENTRY POINT")
    print("=" * 78)

    # --------------------------------------------------------------
    # RUN_TRAINING is the master safety switch.
    #
    # When False, executing this module must NOT accidentally retrain
    # the model.
    # --------------------------------------------------------------

    if not config.RUN_TRAINING:

        print()
        print(
            "RUN_TRAINING=False"
        )

        print(
            "Training has NOT been started."
        )

        print(
            "Set RUN_TRAINING=True only when the "
            "pre-training integration checks have passed."
        )

    else:

        train_final_model()