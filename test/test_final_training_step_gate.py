"""
=======================================================================
FINAL TRAINING-STEP GATE
=======================================================================

Physics-Informed 3D Encoder-Decoder Framework with Predictive Uncertainty
for 3D Seismic Data Reconstruction

Purpose
-------
Final pre-training integration gate.

This script verifies that the actual production training pathway can
successfully execute:

    SyntheticSeismicDataset
            ↓
    DataLoader
            ↓
    Network3D
            ↓
    TotalLoss
            ↓
    Backward propagation
            ↓
    Gradient inspection
            ↓
    Gradient clipping
            ↓
    Adam optimizer update
            ↓
    Second forward/loss evaluation

Checks
------
1. Synthetic dataset creation
2. Dataset sample interface
3. DataLoader batch construction
4. Input/target/mask/velocity validation
5. Network3D forward pass
6. Reconstruction output
7. Travel-time output
8. Log-variance output
9. TotalLoss calculation
10. Individual loss-component finiteness
11. Aleatoric variance positivity
12. Backward propagation
13. Gradient finiteness
14. Gradient clipping
15. Adam optimizer update
16. Parameter finiteness
17. Post-update forward pass
18. Post-update loss calculation

Important
---------
This is a diagnostic gate only.

It does NOT:
    - modify Network3D
    - modify TotalLoss
    - modify config.py
    - save a checkpoint
    - overwrite the production model
    - start the actual training process

Run from the project root:

    python -m test.test_final_training_step_gate

=======================================================================
"""


# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import random
from pathlib import Path
import sys


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# THIRD-PARTY IMPORTS
# =====================================================================

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset,
)

from models.network import (
    Network3D,
)

from losses.total_loss import (
    TotalLoss,
)

from utils import config


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):
    """
    Set Python, NumPy and PyTorch random seeds.

    Parameters
    ----------
    seed : int
        Reproducibility seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DEVICE RESOLUTION
# =====================================================================

def resolve_device(device_setting):
    """
    Resolve the project device configuration into a valid
    torch.device.

    Supported configuration values:

        "auto"
        "cpu"
        "cuda"

    Parameters
    ----------
    device_setting : str or torch.device
        Device setting from utils.config.

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """

    if isinstance(
        device_setting,
        torch.device,
    ):

        return device_setting

    setting = str(
        device_setting
    ).strip().lower()

    if setting == "auto":

        if torch.cuda.is_available():

            return torch.device("cuda")

        return torch.device("cpu")

    if setting == "cpu":

        return torch.device("cpu")

    if setting == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE is configured as 'cuda', but CUDA is "
                "not available in the current PyTorch environment."
            )

        return torch.device("cuda")

    raise ValueError(
        "Unsupported DEVICE configuration: "
        f"{device_setting!r}. "
        "Expected 'auto', 'cpu', 'cuda', or torch.device."
    )


# =====================================================================
# FINITE-TENSOR CHECK
# =====================================================================

def check_finite(
    name,
    tensor,
):
    """
    Verify that a tensor contains only finite values.
    """

    if not torch.is_tensor(tensor):

        raise TypeError(
            f"{name} is not a torch.Tensor."
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise RuntimeError(
            f"{name} contains NaN or infinite values."
        )


# =====================================================================
# TENSOR STATUS
# =====================================================================

def print_tensor_status(
    name,
    tensor,
):
    """
    Print tensor shape and finite-value status.
    """

    check_finite(
        name,
        tensor,
    )

    print(
        f"{name:<24}: "
        f"shape={tuple(tensor.shape)} "
        f"finite=PASS"
    )


# =====================================================================
# PARAMETER VALIDATION
# =====================================================================

def check_model_parameters_finite(
    model,
):
    """
    Verify that all model parameters remain finite.
    """

    for name, parameter in model.named_parameters():

        if not torch.isfinite(
            parameter
        ).all():

            raise RuntimeError(
                "Non-finite model parameter detected: "
                f"{name}"
            )


# =====================================================================
# GRADIENT VALIDATION
# =====================================================================

def inspect_gradients(
    model,
):
    """
    Inspect model gradients.

    Returns
    -------
    tuple
        (
            global_gradient_norm,
            maximum_absolute_gradient,
            gradient_parameter_count
        )
    """

    squared_sum = 0.0

    maximum_gradient = 0.0

    gradient_parameter_count = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is None:

            continue

        gradient = parameter.grad.detach()

        check_finite(
            f"gradient:{name}",
            gradient,
        )

        squared_sum += (
            gradient
            .float()
            .square()
            .sum()
            .item()
        )

        maximum_gradient = max(
            maximum_gradient,
            gradient.abs().max().item(),
        )

        gradient_parameter_count += 1

    gradient_norm = (
        squared_sum ** 0.5
    )

    return (
        gradient_norm,
        maximum_gradient,
        gradient_parameter_count,
    )


# =====================================================================
# MASK VALIDATION
# =====================================================================

def validate_mask(
    mask,
):
    """
    Validate the seismic observation mask.

    Project convention:

        1 = observed
        0 = missing
    """

    check_finite(
        "mask",
        mask,
    )

    unique_values = torch.unique(
        mask.detach()
    )

    valid_values = (
        (unique_values == config.MASK_OBSERVED_VALUE)
        |
        (unique_values == config.MASK_MISSING_VALUE)
    )

    if not bool(
        valid_values.all().item()
    ):

        raise RuntimeError(
            "Mask contains values other than the configured "
            "observed/missing values. "
            f"Observed={config.MASK_OBSERVED_VALUE}, "
            f"Missing={config.MASK_MISSING_VALUE}, "
            f"Found={unique_values.tolist()}"
        )


# =====================================================================
# VELOCITY VALIDATION
# =====================================================================

def validate_velocity(
    velocity,
):
    """
    Validate the velocity model.
    """

    check_finite(
        "velocity",
        velocity,
    )

    if bool(
        (velocity <= 0).any().item()
    ):

        raise RuntimeError(
            "Velocity model contains non-positive values."
        )

    if bool(
        (velocity < config.VELOCITY_MIN).any().item()
    ):

        raise RuntimeError(
            "Velocity model contains values below "
            f"VELOCITY_MIN={config.VELOCITY_MIN}."
        )

    if bool(
        (velocity > config.VELOCITY_MAX).any().item()
    ):

        raise RuntimeError(
            "Velocity model contains values above "
            f"VELOCITY_MAX={config.VELOCITY_MAX}."
        )


# =====================================================================
# LOSS DICTIONARY VALIDATION
# =====================================================================

def validate_loss_dictionary(
    losses,
):
    """
    Validate the dictionary returned by TotalLoss.

    The current production TotalLoss returns:

        total_loss
        mae_loss
        physics_loss
        uncertainty_loss
        ssim_loss
        weighted_mae_loss
        weighted_physics_loss
        weighted_uncertainty_loss
        weighted_ssim_loss
        eikonal_loss
        source_loss
        travel_time_loss
    """

    if not isinstance(
        losses,
        dict,
    ):

        raise TypeError(
            "TotalLoss must return a dictionary."
        )

    required_keys = (
        "total_loss",
        "mae_loss",
        "physics_loss",
        "uncertainty_loss",
        "ssim_loss",
        "weighted_mae_loss",
        "weighted_physics_loss",
        "weighted_uncertainty_loss",
        "weighted_ssim_loss",
        "eikonal_loss",
        "source_loss",
        "travel_time_loss",
    )

    missing_keys = [
        key
        for key in required_keys
        if key not in losses
    ]

    if missing_keys:

        raise KeyError(
            "TotalLoss output is missing required keys: "
            f"{missing_keys}"
        )

    for name, value in losses.items():

        if not torch.is_tensor(value):

            raise TypeError(
                f"TotalLoss component '{name}' "
                "must be a torch.Tensor."
            )

        check_finite(
            f"loss:{name}",
            value,
        )


# =====================================================================
# FORWARD PASS AND LOSS
# =====================================================================

def forward_and_loss(
    model,
    criterion,
    batch,
):
    """
    Execute one production forward pass and calculate
    the complete composite loss.

    Parameters
    ----------
    model : Network3D
        Production 3D encoder-decoder network.

    criterion : TotalLoss
        Production composite loss.

    batch : tuple
        Batch containing:

            inputs
            targets
            mask
            velocity

    Returns
    -------
    tuple
        reconstruction,
        travel_time,
        log_variance,
        losses
    """

    (
        inputs,
        targets,
        mask,
        velocity,
    ) = batch

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------

    (
        reconstruction,
        travel_time,
        log_variance,
    ) = model(
        inputs
    )

    # ---------------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------------

    if reconstruction.shape != targets.shape:

        raise RuntimeError(
            "Reconstruction/target shape mismatch.\n"
            f"Reconstruction: "
            f"{tuple(reconstruction.shape)}\n"
            f"Target: "
            f"{tuple(targets.shape)}"
        )

    if travel_time.shape != velocity.shape:

        raise RuntimeError(
            "Travel-time/velocity shape mismatch.\n"
            f"Travel-time: "
            f"{tuple(travel_time.shape)}\n"
            f"Velocity: "
            f"{tuple(velocity.shape)}"
        )

    if log_variance.shape != reconstruction.shape:

        raise RuntimeError(
            "Log-variance/reconstruction shape mismatch.\n"
            f"Log-variance: "
            f"{tuple(log_variance.shape)}\n"
            f"Reconstruction: "
            f"{tuple(reconstruction.shape)}"
        )

    if mask.shape != targets.shape:

        raise RuntimeError(
            "Mask/target shape mismatch.\n"
            f"Mask: "
            f"{tuple(mask.shape)}\n"
            f"Target: "
            f"{tuple(targets.shape)}"
        )

    # ---------------------------------------------------------------
    # Finite output checks
    # ---------------------------------------------------------------

    check_finite(
        "reconstruction",
        reconstruction,
    )

    check_finite(
        "travel_time",
        travel_time,
    )

    check_finite(
        "log_variance",
        log_variance,
    )

    # ---------------------------------------------------------------
    # Validate velocity and mask
    # ---------------------------------------------------------------

    validate_velocity(
        velocity
    )

    validate_mask(
        mask
    )

    # ---------------------------------------------------------------
    # Current production TotalLoss interface
    #
    # TotalLoss.forward():
    #
    #     prediction
    #     target
    #     velocity
    #     source_indices
    #     travel_time
    #     travel_time_target
    #     log_variance
    #
    # The current synthetic dataset does not provide:
    #
    #     source_indices
    #     travel_time_target
    #
    # Therefore they remain None.
    # ---------------------------------------------------------------

    losses = criterion(
        prediction=reconstruction,
        target=targets,
        velocity=velocity,
        source_indices=None,
        travel_time=travel_time,
        travel_time_target=None,
        log_variance=log_variance,
    )

    # ---------------------------------------------------------------
    # Validate complete TotalLoss output
    # ---------------------------------------------------------------

    validate_loss_dictionary(
        losses
    )

    # ---------------------------------------------------------------
    # Validate aleatoric variance
    #
    # log_variance is converted to variance using the same
    # centralized limits used throughout the project.
    # ---------------------------------------------------------------

    variance = torch.exp(
        torch.clamp(
            log_variance,
            min=config.LOG_VARIANCE_MIN,
            max=config.LOG_VARIANCE_MAX,
        )
    )

    check_finite(
        "aleatoric_variance",
        variance,
    )

    if bool(
        (variance <= 0).any().item()
    ):

        raise RuntimeError(
            "Aleatoric variance contains non-positive values."
        )

    return (
        reconstruction,
        travel_time,
        log_variance,
        losses,
    )


# =====================================================================
# PRINT LOSS COMPONENTS
# =====================================================================

def print_loss_components(
    losses,
):
    """
    Print all TotalLoss components.
    """

    for name, value in losses.items():

        print(
            f"{name:<28}: "
            f"{value.detach().item():.8e}"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 72)
    print("FINAL TRAINING-STEP GATE")
    print("=" * 72)

    # ===============================================================
    # DEVICE
    # ===============================================================

    device = resolve_device(
        config.DEVICE
    )

    print(
        f"Configured device      : "
        f"{config.DEVICE}"
    )

    print(
        f"Resolved PyTorch device: "
        f"{device}"
    )

    print(
        f"Dataset mode           : "
        f"{config.DATASET_MODE}"
    )

    print(
        f"Experiment             : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Synthetic samples      : "
        f"{config.SYNTHETIC_NUM_SAMPLES}"
    )

    print(
        f"Synthetic cube size    : "
        f"{config.SYNTHETIC_PATCH_SIZE}"
    )

    print(
        f"Missing probability    : "
        f"{config.SYNTHETIC_MISSING_PROBABILITY}"
    )

    print(
        f"Batch size             : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Learning rate          : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Weight decay           : "
        f"{config.WEIGHT_DECAY}"
    )

    print(
        f"Spatial spacing        : "
        f"DX={config.DX}, "
        f"DY={config.DY}, "
        f"DZ={config.DZ}"
    )

    # ===============================================================
    # REPRODUCIBILITY
    # ===============================================================

    set_seed(
        config.SEED
    )

    # ===============================================================
    # 1. DATASET
    # ===============================================================

    print()
    print("[1/7] Creating synthetic dataset...")
    print("-" * 72)

    if config.DATASET_MODE != "synthetic":

        raise RuntimeError(
            "This final training-step gate is designed "
            "for the current synthetic training pipeline. "
            f"DATASET_MODE={config.DATASET_MODE!r}."
        )

    dataset = SyntheticSeismicDataset(
        num_samples=config.SYNTHETIC_NUM_SAMPLES,
        cube_size=config.SYNTHETIC_PATCH_SIZE,
        missing_probability=(
            config.SYNTHETIC_MISSING_PROBABILITY
        ),
        geological_mode="random",
        mask_mode="random",
        seed=config.SEED,
    )

    if len(dataset) != config.SYNTHETIC_NUM_SAMPLES:

        raise RuntimeError(
            "Synthetic dataset length does not match "
            "SYNTHETIC_NUM_SAMPLES.\n"
            f"Expected: "
            f"{config.SYNTHETIC_NUM_SAMPLES}\n"
            f"Actual: "
            f"{len(dataset)}"
        )

    if len(dataset) == 0:

        raise RuntimeError(
            "Synthetic dataset is empty."
        )

    print(
        f"Dataset size: {len(dataset)}"
    )

    print(
        "Dataset creation: PASS"
    )

    # ===============================================================
    # 2. SAMPLE AND BATCH
    # ===============================================================

    print()
    print("[2/7] Reading dataset sample and constructing batch...")
    print("-" * 72)

    sample = dataset[0]

    if not isinstance(
        sample,
        (tuple, list),
    ):

        raise RuntimeError(
            "SyntheticSeismicDataset must return "
            "a tuple or list."
        )

    if len(sample) < 4:

        raise RuntimeError(
            "SyntheticSeismicDataset must return at least "
            "(input_cube, target_cube, mask, velocity_model)."
        )

    print(
        f"Dataset sample length: {len(sample)}"
    )

    # ---------------------------------------------------------------
    # Current dataset interface:
    #
    #     0 input_cube
    #     1 target_cube
    #     2 mask
    #     3 velocity_model
    #     4 mask_type
    #     5 geological_mode
    #
    # Only the first four are tensors required by the training
    # pathway.
    # ---------------------------------------------------------------

    input_cube = sample[0]
    target_cube = sample[1]
    mask = sample[2]
    velocity_model = sample[3]

    print_tensor_status(
        "input_cube",
        input_cube,
    )

    print_tensor_status(
        "target_cube",
        target_cube,
    )

    print_tensor_status(
        "mask",
        mask,
    )

    print_tensor_status(
        "velocity_model",
        velocity_model,
    )

    validate_mask(
        mask
    )

    validate_velocity(
        velocity_model
    )

    print(
        "Dataset sample validation: PASS"
    )

    # ---------------------------------------------------------------
    # Dataset sample:
    #
    #     [C,D,H,W]
    #
    # Training batch:
    #
    #     [B,C,D,H,W]
    # ---------------------------------------------------------------

    tensor_dataset = TensorDataset(
        input_cube.unsqueeze(0),
        target_cube.unsqueeze(0),
        mask.unsqueeze(0),
        velocity_model.unsqueeze(0),
    )

    loader = DataLoader(
        tensor_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=(
            config.PERSISTENT_WORKERS
            if config.NUM_WORKERS > 0
            else False
        ),
    )

    batch = next(
        iter(loader)
    )

    batch = tuple(
        tensor.to(
            device,
            non_blocking=(
                config.PIN_MEMORY
                and device.type == "cuda"
            ),
        )
        for tensor in batch
    )

    (
        inputs,
        targets,
        batch_mask,
        velocity,
    ) = batch

    print_tensor_status(
        "inputs",
        inputs,
    )

    print_tensor_status(
        "targets",
        targets,
    )

    print_tensor_status(
        "mask",
        batch_mask,
    )

    print_tensor_status(
        "velocity",
        velocity,
    )

    validate_mask(
        batch_mask
    )

    validate_velocity(
        velocity
    )

    print(
        "Training batch construction: PASS"
    )

    # ===============================================================
    # 3. MODEL
    # ===============================================================

    print()
    print("[3/7] Creating production model...")
    print("-" * 72)

    model = Network3D(
        use_attention=config.USE_ATTENTION,
        use_residual=config.USE_RESIDUAL,
        use_uncertainty=config.USE_UNCERTAINTY,
    ).to(
        device
    )

    model.train()

    check_model_parameters_finite(
        model
    )

    print(
        "Network3D creation: PASS"
    )

    print(
        "Model device: "
        f"{next(model.parameters()).device}"
    )

    # ===============================================================
    # 4. LOSS AND OPTIMIZER
    # ===============================================================

    print()
    print("[4/7] Creating production loss and optimizer...")
    print("-" * 72)

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # This matches the CURRENT TotalLoss constructor supplied by
    # the user:
    #
    # TotalLoss(
    #     loss_weights=None,
    #     physics_loss_weights=None,
    #     data_range=None,
    #     log_variance_min=None,
    #     log_variance_max=None,
    # )
    #
    # All values are supplied from centralized config.py.
    # ---------------------------------------------------------------

    criterion = TotalLoss(
        loss_weights=config.LOSS_WEIGHTS,
        physics_loss_weights=config.PHYSICS_LOSS_WEIGHTS,
        data_range=config.SEISMIC_DATA_RANGE,
        log_variance_min=config.LOG_VARIANCE_MIN,
        log_variance_max=config.LOG_VARIANCE_MAX,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    print(
        "TotalLoss creation: PASS"
    )

    print(
        "Adam optimizer creation: PASS"
    )

    # ===============================================================
    # 5. INITIAL FORWARD AND LOSS
    # ===============================================================

    print()
    print("[5/7] Running initial forward pass and loss...")
    print("-" * 72)

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    (
        reconstruction_before,
        travel_time_before,
        log_variance_before,
        losses_before,
    ) = forward_and_loss(
        model,
        criterion,
        batch,
    )

    total_loss_before = (
        losses_before["total_loss"]
    )

    print_tensor_status(
        "reconstruction_before",
        reconstruction_before,
    )

    print_tensor_status(
        "travel_time_before",
        travel_time_before,
    )

    print_tensor_status(
        "log_variance_before",
        log_variance_before,
    )

    check_finite(
        "total_loss_before",
        total_loss_before,
    )

    print()
    print(
        f"Initial total loss: "
        f"{total_loss_before.item():.8e}"
    )

    print()
    print("Initial loss components:")
    print_loss_components(
        losses_before
    )

    # ===============================================================
    # 6. BACKWARD AND OPTIMIZER UPDATE
    # ===============================================================

    print()
    print("[6/7] Running backward pass and optimizer update...")
    print("-" * 72)

    total_loss_before.backward()

    (
        gradient_norm_before,
        maximum_gradient_before,
        gradient_parameter_count,
    ) = inspect_gradients(
        model
    )

    if gradient_parameter_count == 0:

        raise RuntimeError(
            "No model parameters received gradients."
        )

    print(
        f"Parameters with gradients: "
        f"{gradient_parameter_count}"
    )

    print(
        f"Gradient norm before clip: "
        f"{gradient_norm_before:.8e}"
    )

    print(
        f"Maximum gradient before clip: "
        f"{maximum_gradient_before:.8e}"
    )

    # ---------------------------------------------------------------
    # Current Trainer gradient clipping threshold.
    #
    # This remains a diagnostic value for the current training
    # integration gate.
    # ---------------------------------------------------------------

    max_gradient_norm = 1.0

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
    )

    (
        gradient_norm_after,
        maximum_gradient_after,
        gradient_parameter_count_after,
    ) = inspect_gradients(
        model
    )

    if gradient_parameter_count_after == 0:

        raise RuntimeError(
            "No gradients remain after clipping."
        )

    print(
        f"Gradient norm after clip: "
        f"{gradient_norm_after:.8e}"
    )

    print(
        f"Maximum gradient after clip: "
        f"{maximum_gradient_after:.8e}"
    )

    if gradient_norm_after > (
        max_gradient_norm + 1e-5
    ):

        raise RuntimeError(
            "Gradient clipping failed. "
            f"Observed norm={gradient_norm_after:.8e}, "
            f"allowed={max_gradient_norm:.8e}"
        )

    optimizer.step()

    check_model_parameters_finite(
        model
    )

    print(
        "Backward pass: PASS"
    )

    print(
        "Gradient validation: PASS"
    )

    print(
        "Gradient clipping: PASS"
    )

    print(
        "Optimizer update: PASS"
    )

    print(
        "Model parameter finiteness: PASS"
    )

    # ===============================================================
    # 7. POST-UPDATE FORWARD AND LOSS
    # ===============================================================

    print()
    print("[7/7] Running post-update forward pass...")
    print("-" * 72)

    # ---------------------------------------------------------------
    # The physics-informed Eikonal calculation may require
    # autograd for the travel-time field.
    #
    # Therefore, this evaluation intentionally remains inside
    # an enabled-gradient context.
    # ---------------------------------------------------------------

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.enable_grad():

        (
            reconstruction_after,
            travel_time_after,
            log_variance_after,
            losses_after,
        ) = forward_and_loss(
            model,
            criterion,
            batch,
        )

    total_loss_after = (
        losses_after["total_loss"]
    )

    print_tensor_status(
        "reconstruction_after",
        reconstruction_after,
    )

    print_tensor_status(
        "travel_time_after",
        travel_time_after,
    )

    print_tensor_status(
        "log_variance_after",
        log_variance_after,
    )

    check_finite(
        "total_loss_after",
        total_loss_after,
    )

    print()
    print(
        f"Post-update total loss: "
        f"{total_loss_after.item():.8e}"
    )

    print()
    print("Post-update loss components:")
    print_loss_components(
        losses_after
    )

    # ---------------------------------------------------------------
    # The gate does NOT require the loss to decrease after one
    # optimizer step.
    #
    # A single stochastic/nonlinear optimization step is not a
    # scientifically valid basis for requiring monotonic improvement.
    #
    # Numerical and computational stability are what matter here.
    # ---------------------------------------------------------------

    loss_before_value = float(
        total_loss_before.detach().item()
    )

    loss_after_value = float(
        total_loss_after.detach().item()
    )

    if not np.isfinite(
        loss_before_value
    ):

        raise RuntimeError(
            "Initial loss is non-finite."
        )

    if not np.isfinite(
        loss_after_value
    ):

        raise RuntimeError(
            "Post-update loss is non-finite."
        )

    print(
        "Post-update loss finiteness: PASS"
    )

    # ===============================================================
    # FINAL DECISION
    # ===============================================================

    print()
    print("=" * 72)
    print("FINAL TRAINING-STEP GATE RESULT")
    print("=" * 72)

    print(
        "Dataset pathway             : PASS"
    )

    print(
        "Batch construction          : PASS"
    )

    print(
        "Network3D forward           : PASS"
    )

    print(
        "TotalLoss calculation       : PASS"
    )

    print(
        "Backward propagation        : PASS"
    )

    print(
        "Gradient validation         : PASS"
    )

    print(
        "Gradient clipping           : PASS"
    )

    print(
        "Optimizer update            : PASS"
    )

    print(
        "Post-update forward         : PASS"
    )

    print(
        "Post-update loss            : PASS"
    )

    print()
    print(
        f"Initial total loss          : "
        f"{loss_before_value:.8e}"
    )

    print(
        f"Post-update total loss      : "
        f"{loss_after_value:.8e}"
    )

    print()
    print("=" * 72)
    print("FINAL TRAINING-STEP GATE: PASS")
    print("=" * 72)

    print(
        "The production dataset, model, composite loss, "
        "backward pass and optimizer update are numerically "
        "integrated and ready for the next training-stage check."
    )


# =====================================================================
# SCRIPT ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()