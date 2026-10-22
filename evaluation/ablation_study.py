"""
=====================================================================
FINAL PhD ABLATION STUDY
=====================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

PURPOSE
-------
Controlled component ablation study for the proposed framework.

The five configurations are:

    1. Full_Model
    2. No_Attention
    3. No_Residual
    4. No_Uncertainty
    5. Plain_UNet

EXPERIMENT MANAGEMENT
---------------------

This module supports three execution modes:

    AUTO
        Automatically determines whether the experiment is:

            * a new experiment,
            * a repair of an old-protocol experiment,
            * an already-completed current experiment,
            * or a partially incomplete experiment.

    REPAIR
        Forces the legacy uncertainty-loss repair protocol.

    FULL
        Forces all five configurations to be retrained.

The experiment condition is identified independently from the
ablation protocol version.

This is important because a previous experiment may have been
generated with an older ablation protocol but still represent the
same scientific experimental condition.

Therefore:

    Same experiment + old protocol
        -> repair affected configurations only.

    Same experiment + current protocol
        -> preserve valid configurations.

    Different experiment
        -> evaluate all five configurations.

The uncertainty-loss correction is:

    use_uncertainty=True
        uncertainty loss included when log_variance is supplied
        by the model.

    use_uncertainty=False
        model does not provide log_variance and therefore the
        uncertainty loss contribution is zero.

Therefore:

    No_Uncertainty
        uncertainty loss = 0

    Plain_UNet
        uncertainty loss = 0

Physics and SSIM remain active because the ablation isolates
architectural uncertainty rather than removing the common
reconstruction objectives.

=====================================================================
OUTPUT
=====================================================================

outputs/
    <EXPERIMENT_NAME>/
        ablation/
            Full_Model/
            No_Attention/
            No_Residual/
            No_Uncertainty/
            Plain_UNet/

        reports/
            ablation_study.csv
            ablation_summary.csv
            ablation_metadata.json

=====================================================================
"""

# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import hashlib
import json
import os
import random
from datetime import datetime


# =====================================================================
# THIRD-PARTY LIBRARIES
# =====================================================================

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# =====================================================================
# PROJECT MODULES
# =====================================================================

from models.network import Network3D

from dataset.build_dataset import build_dataset
from dataset.split_dataset import split_dataset

from trainer.trainer import Trainer

from inference.predictor import Predictor

from losses.total_loss import TotalLoss

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.experiment_manager import ExperimentManager

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DX,
    DY,
    DZ,
    DEVICE as CONFIG_DEVICE,
    USE_ATTENTION,
    USE_RESIDUAL,
    USE_UNCERTAINTY,
    VALIDATION_SPLIT,
    SEED,
)


# =====================================================================
# 1. ABLATION CONFIGURATIONS
# =====================================================================

ABLATION_MODELS = {

    "Full_Model": {
        "use_attention": USE_ATTENTION,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Attention": {
        "use_attention": False,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Residual": {
        "use_attention": USE_ATTENTION,
        "use_residual": False,
        "use_uncertainty": USE_UNCERTAINTY,
    },

    "No_Uncertainty": {
        "use_attention": USE_ATTENTION,
        "use_residual": USE_RESIDUAL,
        "use_uncertainty": False,
    },

    "Plain_UNet": {
        "use_attention": False,
        "use_residual": False,
        "use_uncertainty": False,
    },
}


# =====================================================================
# 2. ABLATION EXECUTION CONTROL
# =====================================================================

"""
Allowed execution modes:

    auto
        Normal final-pipeline operation.

    repair
        Force the old-protocol repair logic.

    full
        Force complete retraining of all five configurations.
"""

ABLATION_EXECUTION_MODE = "auto"

FORCE_FULL_ABLATION = False


# =====================================================================
# 3. ABLATION PROTOCOL
# =====================================================================

"""
Protocol 2.0 defines the corrected uncertainty-loss behaviour.

For uncertainty-enabled models:

    uncertainty loss is active when the model supplies
    log_variance.

For uncertainty-disabled models:

    uncertainty loss is exactly zero because no log_variance
    output is supplied.
"""

ABLATION_PROTOCOL_VERSION = "2.0"


# =====================================================================
# 4. LEGACY ABLATION CONTROL
# =====================================================================

"""
These three configurations were not affected by the previous
uncertainty-loss implementation problem.
"""

VALID_LEGACY_MODELS = {
    "Full_Model",
    "No_Attention",
    "No_Residual",
}


"""
These configurations must be retrained when repairing results
generated under the previous uncertainty-loss protocol.
"""

LEGACY_AFFECTED_MODELS = {
    "No_Uncertainty",
    "Plain_UNet",
}


# =====================================================================
# 5. GLOBAL REPRODUCIBILITY
# =====================================================================

def set_global_seed(seed):
    """
    Set all relevant random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# =====================================================================
# 6. DEVICE RESOLUTION
# =====================================================================

def get_device():
    """
    Resolve the configured device.
    """

    if CONFIG_DEVICE == "cpu":

        return torch.device("cpu")

    if CONFIG_DEVICE == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "DEVICE='cuda' is configured, but CUDA "
                "is not available."
            )

        return torch.device("cuda")

    if CONFIG_DEVICE == "auto":

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    raise ValueError(
        f"Unsupported DEVICE configuration: {CONFIG_DEVICE}"
    )


# =====================================================================
# 7. DATALOADER FACTORY
# =====================================================================

def create_ablation_dataloader(
    dataset,
    shuffle=False,
):
    """
    Create the DataLoader used by the ablation study.
    """

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# =====================================================================
# 8. MODEL FACTORY
# =====================================================================

def build_ablation_model(
    settings,
    device,
):
    """
    Construct an independent Network3D instance.
    """

    model = Network3D(
        use_attention=settings["use_attention"],
        use_residual=settings["use_residual"],
        use_uncertainty=settings["use_uncertainty"],
    )

    model = model.to(device)

    return model


# =====================================================================
# 9. MODEL DEVICE VALIDATION
# =====================================================================

def validate_model_device(
    model,
    device,
):
    """
    Verify that parameters and buffers are on the expected device.

    CUDA NOTE
    ---------
    torch.device("cuda") and torch.device("cuda:0") refer to the
    same active CUDA device when CUDA device index 0 is active, but
    they are represented differently by PyTorch.

    Therefore, when the expected device is CUDA without an explicit
    index, resolve it to the currently active CUDA device before
    comparing model parameters and buffers.

    This prevents a false device-mismatch error such as:

        expected: cuda
        actual:   cuda:0

    while still detecting genuine CPU/CUDA or CUDA-device mismatches.
    """

    expected_device = torch.device(device)

    # -------------------------------------------------------------
    # Normalize an unspecified CUDA device.
    #
    # Example:
    #
    #     cuda   -> cuda:0
    #
    # This is the critical correction.
    # -------------------------------------------------------------

    if expected_device.type == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA device validation requested, but CUDA "
                "is not available."
            )

        if expected_device.index is None:

            expected_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
            )

    # -------------------------------------------------------------
    # Validate model parameters.
    # -------------------------------------------------------------

    for name, parameter in model.named_parameters():

        actual_device = parameter.device

        if actual_device.type == "cuda":

            if actual_device.index is None:

                actual_device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )

        if actual_device != expected_device:

            raise RuntimeError(
                f"Model parameter '{name}' is on "
                f"{actual_device}, expected "
                f"{expected_device}."
            )

    # -------------------------------------------------------------
    # Validate model buffers.
    # -------------------------------------------------------------

    for name, buffer in model.named_buffers():

        actual_device = buffer.device

        if actual_device.type == "cuda":

            if actual_device.index is None:

                actual_device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )

        if actual_device != expected_device:

            raise RuntimeError(
                f"Model buffer '{name}' is on "
                f"{actual_device}, expected "
                f"{expected_device}."
            )

    return True


# =====================================================================
# 10. SAMPLE-ID EXTRACTION
# =====================================================================

def get_sample_id(
    dataset,
    index,
):
    """
    Return a stable original dataset Sample_ID.
    """

    if hasattr(dataset, "indices"):

        return int(
            dataset.indices[index]
        )

    return int(index)


# =====================================================================
# 11. PREDICTION OUTPUT NORMALIZATION
# =====================================================================

def extract_reconstruction(
    prediction,
):
    """
    Extract the reconstruction tensor from Predictor output.

    The current model interface is expected to return:

        reconstruction,
        log_variance,
        auxiliary_output

    or an equivalent tuple in which the reconstruction is the
    first element.
    """

    if not isinstance(prediction, tuple):

        raise TypeError(
            "Predictor.predict() must return a tuple."
        )

    if len(prediction) == 0:

        raise ValueError(
            "Predictor.predict() returned an empty tuple."
        )

    reconstruction = prediction[0]

    if not torch.is_tensor(reconstruction):

        raise TypeError(
            "The first Predictor output must be a tensor."
        )

    return reconstruction


# =====================================================================
# 12. TENSOR SHAPE NORMALIZATION
# =====================================================================

def ensure_batched_tensor(
    tensor,
    name,
):
    """
    Ensure tensor has shape:

        [B, C, D, H, W]
    """

    if not torch.is_tensor(tensor):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if tensor.ndim == 4:

        return tensor.unsqueeze(0)

    if tensor.ndim == 5:

        return tensor

    raise ValueError(
        f"{name} must have 4 or 5 dimensions. "
        f"Received {tuple(tensor.shape)}."
    )


# =====================================================================
# 13. METRIC VALIDATION
# =====================================================================

def validate_metric_value(
    metric_name,
    value,
    sample_id,
):
    """
    Ensure a metric is finite.
    """

    value = float(value)

    if not np.isfinite(value):

        raise RuntimeError(
            f"Non-finite {metric_name} detected for "
            f"Sample_ID={sample_id}: {value}"
        )

    return value


# =====================================================================
# 14. PER-SAMPLE EVALUATION
# =====================================================================

def evaluate_checkpoint(
    model,
    checkpoint,
    dataset,
    device,
):
    """
    Evaluate one trained ablation model on the validation dataset.

    One row is generated per validation sample.
    """

    if len(dataset) == 0:

        raise RuntimeError(
            "Cannot evaluate an empty validation dataset."
        )

    model = model.to(device)

    validate_model_device(
        model,
        device,
    )

    predictor = Predictor(
        model=model,
        checkpoint=checkpoint,
        device=device,
    )

    sample_results = []

    for index in range(len(dataset)):

        sample_id = get_sample_id(
            dataset,
            index,
        )

        print(
            f"Evaluating validation sample "
            f"{index + 1}/{len(dataset)} "
            f"(Sample_ID={sample_id})"
        )

        sample = dataset[index]

        if len(sample) < 3:

            raise ValueError(
                "Dataset sample must contain at least "
                "(input_cube, target_cube, mask)."
            )

        input_cube = ensure_batched_tensor(
            sample[0],
            "input_cube",
        )

        target_cube = ensure_batched_tensor(
            sample[1],
            "target_cube",
        )

        mask = ensure_batched_tensor(
            sample[2],
            "mask",
        )

        # -------------------------------------------------------------
        # Explicitly place all evaluation tensors on the resolved
        # device.
        # -------------------------------------------------------------

        input_cube = input_cube.to(
            device=device,
            non_blocking=False,
        )

        target_cube = target_cube.to(
            device=device,
            non_blocking=False,
        )

        mask = mask.to(
            device=device,
            non_blocking=False,
        )

        if input_cube.shape != target_cube.shape:

            raise ValueError(
                "Input and target shapes do not match.\n"
                f"Input : {tuple(input_cube.shape)}\n"
                f"Target: {tuple(target_cube.shape)}"
            )

        if mask.shape != input_cube.shape:

            raise ValueError(
                "Mask and input shapes do not match.\n"
                f"Mask : {tuple(mask.shape)}\n"
                f"Input: {tuple(input_cube.shape)}"
            )

        if not torch.isfinite(input_cube).all():

            raise RuntimeError(
                f"Input contains NaN/Inf for Sample_ID={sample_id}."
            )

        if not torch.isfinite(target_cube).all():

            raise RuntimeError(
                f"Target contains NaN/Inf for Sample_ID={sample_id}."
            )

        model.eval()

        with torch.no_grad():

            prediction = predictor.predict(
                input_cube
            )

        reconstruction = extract_reconstruction(
            prediction
        )

        reconstruction = ensure_batched_tensor(
            reconstruction,
            "reconstruction",
        )

        reconstruction = reconstruction.to(
            device=device,
            non_blocking=False,
        )

        if reconstruction.shape != target_cube.shape:

            raise ValueError(
                "Reconstruction and target shapes do not match.\n"
                f"Reconstruction: {tuple(reconstruction.shape)}\n"
                f"Target: {tuple(target_cube.shape)}"
            )

        if not torch.isfinite(reconstruction).all():

            raise RuntimeError(
                f"Reconstruction contains NaN/Inf "
                f"for Sample_ID={sample_id}."
            )

        # -------------------------------------------------------------
        # Canonical reconstruction metrics
        # -------------------------------------------------------------

        sample_mae = mae(
            reconstruction,
            target_cube,
        )

        sample_rmse = rmse(
            reconstruction,
            target_cube,
        )

        sample_psnr = psnr(
            reconstruction,
            target_cube,
        )

        sample_snr = snr(
            reconstruction,
            target_cube,
        )

        sample_ssim = ssim(
            reconstruction,
            target_cube,
        )

        sample_results.append(
            {
                "Sample_ID": sample_id,

                "MAE": validate_metric_value(
                    "MAE",
                    sample_mae.item(),
                    sample_id,
                ),

                "RMSE": validate_metric_value(
                    "RMSE",
                    sample_rmse.item(),
                    sample_id,
                ),

                "PSNR": validate_metric_value(
                    "PSNR",
                    sample_psnr.item(),
                    sample_id,
                ),

                "SNR": validate_metric_value(
                    "SNR",
                    sample_snr.item(),
                    sample_id,
                ),

                "SSIM": validate_metric_value(
                    "SSIM",
                    sample_ssim.item(),
                    sample_id,
                ),
            }
        )

    return sample_results


# =====================================================================
# 15. TRAIN ONE ABLATION CONFIGURATION
# =====================================================================

def train_ablation_model(
    model_name,
    settings,
    train_loader,
    val_loader,
    device,
    experiment_root,
    seed,
):
    """
    Train one ablation configuration completely from scratch.
    """

    print()
    print("=" * 70)
    print(
        f"TRAINING ABLATION MODEL: {model_name}"
    )
    print("=" * 70)

    set_global_seed(seed)

    print()
    print(
        "Attention   :",
        settings["use_attention"],
    )

    print(
        "Residual    :",
        settings["use_residual"],
    )

    print(
        "Uncertainty :",
        settings["use_uncertainty"],
    )

    print(
        "Seed        :",
        seed,
    )

    print(
        "Device      :",
        device,
    )

    print(
        "Experiment  :",
        experiment_root,
    )

    model = build_ablation_model(
        settings,
        device,
    )

    validate_model_device(
        model,
        device,
    )

    # -------------------------------------------------------------
    # Current composite loss interface
    #
    # TotalLoss obtains the centralized loss configuration from
    # utils.config.py.
    #
    # DX, DY and DZ are handled internally by PhysicsLoss.
    #
    # use_uncertainty is controlled by the model architecture.
    # When uncertainty is disabled, the model does not provide
    # log_variance and the uncertainty loss contribution is zero.
    # -------------------------------------------------------------

    criterion = TotalLoss()

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------
    # Experiment manager
    # -------------------------------------------------------------

    experiment_manager = ExperimentManager(
        root=experiment_root,
    )

    # -------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
    )

    # -------------------------------------------------------------
    # Train from scratch
    # -------------------------------------------------------------

    trainer.fit(
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
        epochs=NUM_EPOCHS,
        resume=False,
    )

    # -------------------------------------------------------------
    # Best checkpoint
    # -------------------------------------------------------------

    checkpoint = os.path.join(
        experiment_manager.checkpoints,
        "best_model.pth",
    )

    if not os.path.isfile(checkpoint):

        raise FileNotFoundError(
            f"Best checkpoint was not created for "
            f"{model_name}:\n{checkpoint}"
        )

    print()
    print("Best checkpoint:")
    print(checkpoint)

    return checkpoint


# =====================================================================
# 16. CURRENT EXPERIMENT INPUTS
# =====================================================================

def current_experiment_inputs(
    dataset,
    train_dataset,
    val_dataset,
    validation_sample_ids,
):
    """
    Build the scientific conditions that define one ablation
    experiment.

    Protocol version is deliberately excluded.

    This allows an old-protocol experiment to be recognised as the
    same scientific experiment and repaired rather than incorrectly
    treated as a completely new experiment.
    """

    return {

        "experiment_name":
            EXPERIMENT_NAME,

        "dataset_size":
            len(dataset),

        "training_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "validation_sample_ids": [
            int(value)
            for value in validation_sample_ids
        ],

        "batch_size":
            BATCH_SIZE,

        "epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "validation_split":
            VALIDATION_SPLIT,

        "seed":
            SEED,

        "dx":
            DX,

        "dy":
            DY,

        "dz":
            DZ,

        "ablation_configurations":
            ABLATION_MODELS,
    }


# =====================================================================
# 17. EXPERIMENT SIGNATURE
# =====================================================================

def build_experiment_signature(
    experiment_inputs,
):
    """
    Generate a deterministic SHA-256 signature from the scientific
    experimental conditions.
    """

    serialized = json.dumps(
        experiment_inputs,
        sort_keys=True,
        default=str,
    )

    signature = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return signature


# =====================================================================
# 18. PREVIOUS EXPERIMENT COMPATIBILITY
# =====================================================================

def previous_experiment_matches_current(
    previous_metadata,
    current_inputs,
):
    """
    Determine whether previous metadata describes the same scientific
    experimental condition.

    This function is backward-compatible with metadata produced by
    earlier versions of the ablation script.
    """

    if not previous_metadata:

        return False

    previous_inputs = previous_metadata.get(
        "experiment_signature_inputs"
    )

    if not isinstance(
        previous_inputs,
        dict,
    ):

        return False

    # -------------------------------------------------------------
    # Compare only the scientific inputs.
    #
    # Protocol version and execution environment are deliberately
    # ignored here.
    # -------------------------------------------------------------

    scientific_keys = [
        "experiment_name",
        "dataset_size",
        "training_samples",
        "validation_samples",
        "validation_sample_ids",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "validation_split",
        "seed",
        "dx",
        "dy",
        "dz",
        "ablation_configurations",
    ]

    for key in scientific_keys:

        if previous_inputs.get(key) != current_inputs.get(key):

            return False

    return True


# =====================================================================
# 19. LOAD PREVIOUS EXPERIMENT
# =====================================================================

def load_previous_experiment(
    output_file,
    metadata_file,
):
    """
    Load previous results and metadata.
    """

    if not os.path.isfile(output_file):

        return None, None

    if not os.path.isfile(metadata_file):

        return None, None

    try:

        dataframe = pd.read_csv(
            output_file
        )

        with open(
            metadata_file,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as error:

        print()
        print(
            "WARNING: Existing ablation results could not "
            "be loaded."
        )

        print(
            f"Reason: {error}"
        )

        return None, None

    return dataframe, metadata


# =====================================================================
# 20. VALIDATE ONE REUSABLE MODEL
# =====================================================================

def validate_reusable_model(
    dataframe,
    model_name,
    expected_sample_ids,
):
    """
    Determine whether one model's previous results are complete,
    correctly paired and numerically valid.

    Returns:

        True
            if reusable.

        False
            if missing, incomplete or invalid.
    """

    required_columns = [
        "Model",
        "Sample_ID",
        "Attention",
        "Residual",
        "Uncertainty",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "Checkpoint",
    ]

    if dataframe is None:

        return False

    if any(
        column not in dataframe.columns
        for column in required_columns
    ):

        return False

    model_rows = dataframe[
        dataframe["Model"] == model_name
    ].copy()

    if model_rows.empty:

        return False

    expected_ids = {
        int(value)
        for value in expected_sample_ids
    }

    actual_ids = {
        int(value)
        for value in model_rows["Sample_ID"]
    }

    if actual_ids != expected_ids:

        return False

    if model_rows.duplicated(
        subset=["Model", "Sample_ID"]
    ).any():

        return False

    for metric in [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]:

        try:

            values = model_rows[metric].astype(float)

        except Exception:

            return False

        if not np.isfinite(values).all():

            return False

    # -------------------------------------------------------------
    # Verify checkpoint paths exist.
    # -------------------------------------------------------------

    for checkpoint in model_rows["Checkpoint"]:

        if not isinstance(
            checkpoint,
            str,
        ):

            return False

        if not os.path.isfile(checkpoint):

            return False

    return True


# =====================================================================
# 21. EXTRACT REUSABLE MODEL RESULTS
# =====================================================================

def get_reusable_models(
    dataframe,
    expected_sample_ids,
):
    """
    Return all configurations whose previous results are valid.
    """

    reusable = set()

    for model_name in ABLATION_MODELS:

        if validate_reusable_model(
            dataframe,
            model_name,
            expected_sample_ids,
        ):

            reusable.add(model_name)

    return reusable


# =====================================================================
# 22. DETERMINE EVALUATION PLAN
# =====================================================================

def determine_evaluation_plan(
    previous_dataframe,
    previous_metadata,
    current_signature,
    current_inputs,
    expected_sample_ids,
):
    """
    Determine exactly which configurations should be preserved
    and which must be evaluated.

    Priority:

        1. Force full
        2. Explicit full mode
        3. No previous experiment
        4. Different scientific experiment
        5. Same experiment + old protocol
        6. Same experiment + current protocol
        7. Repair incomplete current results
    """

    all_models = set(
        ABLATION_MODELS.keys()
    )

    mode = (
        ABLATION_EXECUTION_MODE
        .strip()
        .lower()
    )

    if mode not in {
        "auto",
        "repair",
        "full",
    }:

        raise ValueError(
            "ABLATION_EXECUTION_MODE must be "
            "'auto', 'repair', or 'full'. "
            f"Received: {ABLATION_EXECUTION_MODE}"
        )

    # -------------------------------------------------------------
    # Explicit full execution
    # -------------------------------------------------------------

    if (
        FORCE_FULL_ABLATION
        or mode == "full"
    ):

        return {
            "mode": "full",
            "preserve": set(),
            "evaluate": all_models,
        }

    # -------------------------------------------------------------
    # No previous results
    # -------------------------------------------------------------

    if (
        previous_dataframe is None
        or previous_metadata is None
    ):

        return {
            "mode": "new_experiment",
            "preserve": set(),
            "evaluate": all_models,
        }

    # -------------------------------------------------------------
    # Verify scientific experimental condition.
    # -------------------------------------------------------------

    same_experiment = (
        previous_experiment_matches_current(
            previous_metadata,
            current_inputs,
        )
    )

    if not same_experiment:

        return {
            "mode": "new_experiment",
            "preserve": set(),
            "evaluate": all_models,
        }

    # -------------------------------------------------------------
    # Determine which old results are actually reusable.
    # -------------------------------------------------------------

    reusable_models = get_reusable_models(
        previous_dataframe,
        expected_sample_ids,
    )

    previous_protocol = previous_metadata.get(
        "ablation_protocol_version"
    )

    # -------------------------------------------------------------
    # Explicit repair mode.
    # -------------------------------------------------------------

    if mode == "repair":

        preserve = (
            reusable_models
            & VALID_LEGACY_MODELS
        )

        evaluate = (
            all_models
            - preserve
        )

        return {
            "mode": "repair_current_experiment",
            "preserve": preserve,
            "evaluate": evaluate,
        }

    # -------------------------------------------------------------
    # Old protocol -> repair affected configurations.
    # -------------------------------------------------------------

    if previous_protocol != ABLATION_PROTOCOL_VERSION:

        preserve = (
            reusable_models
            & VALID_LEGACY_MODELS
        )

        evaluate = (
            all_models
            - preserve
        )

        return {
            "mode": "repair_current_experiment",
            "preserve": preserve,
            "evaluate": evaluate,
        }

    # -------------------------------------------------------------
    # Current protocol.
    #
    # Preserve every valid configuration and rerun only missing or
    # invalid configurations.
    # -------------------------------------------------------------

    preserve = reusable_models

    evaluate = (
        all_models
        - preserve
    )

    if not evaluate:

        plan_mode = "already_current"

    else:

        plan_mode = "repair_incomplete_current_experiment"

    return {
        "mode": plan_mode,
        "preserve": preserve,
        "evaluate": evaluate,
    }


# =====================================================================
# 23. EXTRACT PRESERVED RESULTS
# =====================================================================

def extract_preserved_results(
    dataframe,
    preserve_models,
):
    """
    Extract only the validated preserved configurations.
    """

    if not preserve_models:

        return pd.DataFrame()

    preserved = dataframe[
        dataframe["Model"].isin(
            preserve_models
        )
    ].copy()

    return preserved


# =====================================================================
# 24. VALIDATE COMPLETE ABLATION PAIRING
# =====================================================================

def validate_ablation_pairing(
    dataframe,
    expected_sample_ids,
):
    """
    Strictly validate the final paired ablation dataset.
    """

    expected_rows = (
        len(expected_sample_ids)
        * len(ABLATION_MODELS)
    )

    if len(dataframe) != expected_rows:

        raise RuntimeError(
            "Unexpected number of ablation rows.\n"
            f"Expected: {expected_rows}\n"
            f"Received: {len(dataframe)}"
        )

    expected_ids = {
        int(value)
        for value in expected_sample_ids
    }

    actual_ids = {
        int(value)
        for value in dataframe["Sample_ID"]
    }

    if actual_ids != expected_ids:

        raise RuntimeError(
            "Validation Sample_ID mismatch.\n"
            f"Expected: {expected_ids}\n"
            f"Received: {actual_ids}"
        )

    expected_models = set(
        ABLATION_MODELS.keys()
    )

    actual_models = set(
        dataframe["Model"].unique()
    )

    if actual_models != expected_models:

        raise RuntimeError(
            "Ablation model set mismatch.\n"
            f"Expected: {expected_models}\n"
            f"Received: {actual_models}"
        )

    duplicates = dataframe[
        dataframe.duplicated(
            subset=[
                "Model",
                "Sample_ID",
            ],
            keep=False,
        )
    ]

    if not duplicates.empty:

        raise RuntimeError(
            "Duplicate Model + Sample_ID pairs detected."
        )

    for model_name in expected_models:

        model_ids = set(
            dataframe.loc[
                dataframe["Model"] == model_name,
                "Sample_ID",
            ]
        )

        if model_ids != expected_ids:

            raise RuntimeError(
                f"Sample pairing failure for {model_name}."
            )

    for metric in [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]:

        values = dataframe[metric].astype(float)

        if not np.isfinite(values).all():

            raise RuntimeError(
                f"Non-finite values detected in {metric}."
            )

    return True


# =====================================================================
# 25. BUILD ABLATION SUMMARY
# =====================================================================

def build_ablation_summary(
    dataframe,
):
    """
    Calculate descriptive model-level means and percentage changes
    relative to Full_Model.

    These are descriptive comparisons and are not statistical
    significance tests.
    """

    metric_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    summary = (
        dataframe
        .groupby(
            "Model",
            as_index=False,
        )[metric_columns]
        .mean()
    )

    configuration = (
        dataframe[
            [
                "Model",
                "Attention",
                "Residual",
                "Uncertainty",
            ]
        ]
        .drop_duplicates(
            subset=["Model"]
        )
    )

    summary = configuration.merge(
        summary,
        on="Model",
        how="left",
    )

    full_rows = summary[
        summary["Model"] == "Full_Model"
    ]

    if full_rows.empty:

        raise RuntimeError(
            "Full_Model is missing from ablation summary."
        )

    full_model = full_rows.iloc[0]

    for metric in metric_columns:

        reference = float(
            full_model[metric]
        )

        change_column = (
            f"{metric}_Change_Percent"
        )

        if (
            np.isfinite(reference)
            and reference != 0.0
        ):

            summary[change_column] = (
                (
                    summary[metric]
                    - reference
                )
                / reference
            ) * 100.0

        else:

            summary[change_column] = np.nan

    return summary[
        [
            "Model",
            "Attention",
            "Residual",
            "Uncertainty",
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM",
            "MAE_Change_Percent",
            "RMSE_Change_Percent",
            "PSNR_Change_Percent",
            "SNR_Change_Percent",
            "SSIM_Change_Percent",
        ]
    ]


# =====================================================================
# 26. MAIN ABLATION PROCEDURE
# =====================================================================

def run_ablation():

    set_global_seed(SEED)

    # =================================================================
    # HEADER
    # =================================================================

    print()
    print("=" * 70)
    print("FINAL PhD ABLATION STUDY")
    print("=" * 70)

    print()
    print(
        "Experiment Name:",
        EXPERIMENT_NAME,
    )

    print(
        "Report Directory:",
        REPORT_DIR,
    )

    print(
        "Checkpoint Directory:",
        CHECKPOINT_DIR,
    )

    print(
        "Validation Split:",
        VALIDATION_SPLIT,
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Ablation Protocol:",
        ABLATION_PROTOCOL_VERSION,
    )

    print(
        "Execution Mode:",
        ABLATION_EXECUTION_MODE,
    )

    print(
        "Force Full Run:",
        FORCE_FULL_ABLATION,
    )

    # =================================================================
    # DEVICE
    # =================================================================

    device = get_device()

    print()
    print(
        "Configured Device:",
        CONFIG_DEVICE,
    )

    print(
        "Resolved Device:",
        device,
    )

    # Display the concrete CUDA device when applicable.
    if device.type == "cuda":

        print(
            "Active CUDA Device:",
            torch.cuda.current_device(),
        )

    # =================================================================
    # BUILD DATASET
    # =================================================================

    print()
    print("=" * 70)
    print("BUILDING DATASET")
    print("=" * 70)

    dataset = build_dataset()

    if len(dataset) == 0:

        raise RuntimeError(
            "Ablation dataset is empty."
        )

    print()
    print(
        "Total Dataset Samples:",
        len(dataset),
    )

    # =================================================================
    # SPLIT DATASET
    # =================================================================

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    print()
    print(
        "Training Samples:",
        len(train_dataset),
    )

    print(
        "Validation Samples:",
        len(val_dataset),
    )

    # =================================================================
    # VALIDATION SAMPLE IDS
    # =================================================================

    validation_sample_ids = [
        get_sample_id(
            val_dataset,
            index,
        )
        for index in range(
            len(val_dataset)
        )
    ]

    print()
    print(
        "Validation Sample IDs:"
    )

    print(
        validation_sample_ids
    )

    # =================================================================
    # BUILD SCIENTIFIC EXPERIMENT INPUTS
    # =================================================================

    experiment_inputs = current_experiment_inputs(
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        validation_sample_ids=validation_sample_ids,
    )

    # =================================================================
    # BUILD EXPERIMENT SIGNATURE
    # =================================================================

    experiment_signature = (
        build_experiment_signature(
            experiment_inputs
        )
    )

    print()
    print(
        "Experiment Signature:"
    )

    print(
        experiment_signature
    )

    # =================================================================
    # DATA LOADERS
    # =================================================================

    train_loader = create_ablation_dataloader(
        train_dataset,
        shuffle=True,
    )

    val_loader = create_ablation_dataloader(
        val_dataset,
        shuffle=False,
    )

    # =================================================================
    # REPORT PATHS
    # =================================================================

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    output_file = os.path.join(
        REPORT_DIR,
        "ablation_study.csv",
    )

    summary_file = os.path.join(
        REPORT_DIR,
        "ablation_summary.csv",
    )

    metadata_file = os.path.join(
        REPORT_DIR,
        "ablation_metadata.json",
    )

    # =================================================================
    # LOAD PREVIOUS RESULTS
    # =================================================================

    previous_dataframe, previous_metadata = (
        load_previous_experiment(
            output_file,
            metadata_file,
        )
    )

    # =================================================================
    # DETERMINE PLAN
    # =================================================================

    plan = determine_evaluation_plan(
        previous_dataframe=previous_dataframe,
        previous_metadata=previous_metadata,
        current_signature=experiment_signature,
        current_inputs=experiment_inputs,
        expected_sample_ids=validation_sample_ids,
    )

    preserve_models = plan["preserve"]
    evaluate_models = plan["evaluate"]

    print()
    print("=" * 70)
    print("ABLATION EVALUATION PLAN")
    print("=" * 70)

    print()
    print(
        "Plan:",
        plan["mode"],
    )

    print()
    print(
        "Configurations to preserve:"
    )

    if preserve_models:

        for model_name in ABLATION_MODELS:

            if model_name in preserve_models:

                print(
                    f"    PRESERVE: {model_name}"
                )

    else:

        print(
            "    None"
        )

    print()
    print(
        "Configurations to evaluate:"
    )

    if evaluate_models:

        for model_name in ABLATION_MODELS:

            if model_name in evaluate_models:

                print(
                    f"    RUN:      {model_name}"
                )

    else:

        print(
            "    None"
        )

    # =================================================================
    # PRESERVE VALID RESULTS
    # =================================================================

    all_result_frames = []

    if preserve_models:

        if previous_dataframe is None:

            raise RuntimeError(
                "Preservation requested but previous "
                "results are unavailable."
            )

        print()
        print("=" * 70)
        print("PRESERVING VALID RESULTS")
        print("=" * 70)

        preserved_dataframe = extract_preserved_results(
            previous_dataframe,
            preserve_models,
        )

        all_result_frames.append(
            preserved_dataframe
        )

        for model_name in ABLATION_MODELS:

            if model_name in preserve_models:

                print(
                    f"    {model_name}"
                )

    # =================================================================
    # TRAIN AND EVALUATE REQUIRED CONFIGURATIONS
    # =================================================================

    for model_name, settings in ABLATION_MODELS.items():

        if model_name not in evaluate_models:

            continue

        print()
        print("=" * 70)
        print(
            f"ABLATION CONFIGURATION: {model_name}"
        )
        print("=" * 70)

        set_global_seed(SEED)

        # -------------------------------------------------------------
        # Isolated output directory
        # -------------------------------------------------------------

        experiment_root = os.path.abspath(
            os.path.join(
                CHECKPOINT_DIR,
                "..",
                "ablation",
                model_name,
            )
        )

        os.makedirs(
            experiment_root,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # Train
        # -------------------------------------------------------------

        checkpoint = train_ablation_model(
            model_name=model_name,
            settings=settings,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            experiment_root=experiment_root,
            seed=SEED,
        )

        # -------------------------------------------------------------
        # Independent model instance for evaluation
        # -------------------------------------------------------------

        evaluation_model = build_ablation_model(
            settings,
            device,
        )

        validate_model_device(
            evaluation_model,
            device,
        )

        # -------------------------------------------------------------
        # Evaluate validation set
        # -------------------------------------------------------------

        sample_metrics = evaluate_checkpoint(
            model=evaluation_model,
            checkpoint=checkpoint,
            dataset=val_dataset,
            device=device,
        )

        # -------------------------------------------------------------
        # Add configuration information
        # -------------------------------------------------------------

        for result in sample_metrics:

            result["Model"] = model_name

            result["Attention"] = (
                settings["use_attention"]
            )

            result["Residual"] = (
                settings["use_residual"]
            )

            result["Uncertainty"] = (
                settings["use_uncertainty"]
            )

            result["Checkpoint"] = checkpoint

        model_dataframe = pd.DataFrame(
            sample_metrics
        )

        all_result_frames.append(
            model_dataframe
        )

        # -------------------------------------------------------------
        # Display means
        # -------------------------------------------------------------

        print()
        print(
            f"{model_name} validation means"
        )

        print("-" * 50)

        for metric in [
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM",
        ]:

            print(
                f"{metric:<6}: "
                f"{model_dataframe[metric].mean():.6f}"
            )

    # =================================================================
    # COMBINE RESULTS
    # =================================================================

    if not all_result_frames:

        raise RuntimeError(
            "No ablation results were produced."
        )

    dataframe = pd.concat(
        all_result_frames,
        ignore_index=True,
    )

    # =================================================================
    # VALIDATE COMPLETE RESULTS
    # =================================================================

    validate_ablation_pairing(
        dataframe,
        validation_sample_ids,
    )

    # =================================================================
    # SORT RESULTS
    # =================================================================

    model_order = {
        name: index
        for index, name
        in enumerate(
            ABLATION_MODELS.keys()
        )
    }

    dataframe["_Model_Order"] = (
        dataframe["Model"]
        .map(model_order)
    )

    dataframe = (
        dataframe
        .sort_values(
            by=[
                "Sample_ID",
                "_Model_Order",
            ]
        )
        .drop(
            columns=["_Model_Order"]
        )
        .reset_index(
            drop=True
        )
    )

    # =================================================================
    # SAVE PER-SAMPLE RESULTS
    # =================================================================

    dataframe.to_csv(
        output_file,
        index=False,
    )

    # =================================================================
    # SUMMARY
    # =================================================================

    summary_dataframe = build_ablation_summary(
        dataframe
    )

    summary_dataframe.to_csv(
        summary_file,
        index=False,
    )

    # =================================================================
    # METADATA
    # =================================================================

    metadata = {

        "study":
            "Controlled PhD Ablation Study",

        "experiment_name":
            EXPERIMENT_NAME,

        "timestamp":
            datetime.now().isoformat(),

        "ablation_protocol_version":
            ABLATION_PROTOCOL_VERSION,

        "experiment_signature":
            experiment_signature,

        "experiment_signature_inputs":
            experiment_inputs,

        "execution_plan":
            plan["mode"],

        "preserved_configurations":
            sorted(
                list(preserve_models)
            ),

        "evaluated_configurations":
            sorted(
                list(evaluate_models)
            ),

        "seed":
            SEED,

        "configured_device":
            CONFIG_DEVICE,

        "resolved_device":
            str(device),

        "dataset_size":
            len(dataset),

        "training_samples":
            len(train_dataset),

        "validation_samples":
            len(val_dataset),

        "validation_sample_ids":
            validation_sample_ids,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "validation_split":
            VALIDATION_SPLIT,

        "dx":
            DX,

        "dy":
            DY,

        "dz":
            DZ,

        "ablation_configurations":
            ABLATION_MODELS,

        "uncertainty_loss_protocol":
            {
                "Full_Model": True,
                "No_Attention": True,
                "No_Residual": True,
                "No_Uncertainty": False,
                "Plain_UNet": False,
            },

        "per_sample_output":
            output_file,

        "summary_output":
            summary_file,

        "metadata_output":
            metadata_file,
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # =================================================================
    # DISPLAY RESULTS
    # =================================================================

    print()
    print("=" * 70)
    print("PER-SAMPLE ABLATION RESULTS")
    print("=" * 70)

    print()

    print(
        dataframe.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print()

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    # =================================================================
    # OUTPUT FILES
    # =================================================================

    print()
    print("=" * 70)
    print("ABLATION OUTPUT FILES")
    print("=" * 70)

    print()
    print(
        "Per-sample results:"
    )

    print(
        output_file
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_file
    )

    print()
    print(
        "Metadata:"
    )

    print(
        metadata_file
    )

    # =================================================================
    # COMPLETION
    # =================================================================

    print()
    print("=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total configurations: "
        f"{len(ABLATION_MODELS)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_sample_ids)}"
    )

    print(
        f"Total paired observations: "
        f"{len(dataframe)}"
    )

    return (
        dataframe,
        summary_dataframe,
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    run_ablation()