"""
=========================================================
Trainer
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Training pipeline:

    Input seismic cube
          |
          v
    Physics-Informed 3D Network
          |
          +---- Reconstruction
          |
          +---- Travel-Time Field
          |
          +---- Log Variance
          |
          v
    Composite Total Loss
          |
          +---- MAE
          +---- Eikonal Physics
          +---- Uncertainty
          +---- SSIM

Dataset batch convention:

    inputs
    targets
    mask
    velocity_model

Tensor convention:

    [B, C, D, H, W]

Important numerical-stability policy:

    Non-finite gradients
        -> FATAL ERROR

    Large finite gradients
        -> gradient clipping
        -> optimizer step

The Trainer does not modify the physics equation.

Author: Ormin Joseph
=========================================================
"""

import csv
import json
import os
import time

import matplotlib.pyplot as plt
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from metrics.reconstruction_metrics import (
    mae,
    rmse,
    psnr,
    snr,
    ssim,
)

from utils.experiment_manager import ExperimentManager
from utils import config


# =========================================================
# DEBUG CONFIGURATION
# =========================================================

DEBUG_VALIDATION = False


# =========================================================
# TRAINER
# =========================================================

class Trainer:
    """
    Trainer for the Physics-Informed 3D Encoder-Decoder
    seismic reconstruction framework.
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        experiment_manager=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Physics-informed 3D encoder-decoder network.

        criterion : torch.nn.Module
            Composite TotalLoss.

        optimizer : torch.optim.Optimizer
            Optimizer used for training.

        device : torch.device or str
            Training device.

        experiment_manager : ExperimentManager, optional
            Experiment directory manager.
        """

        # -------------------------------------------------
        # Validate required objects
        # -------------------------------------------------

        if model is None:
            raise ValueError(
                "model cannot be None."
            )

        if criterion is None:
            raise ValueError(
                "criterion cannot be None."
            )

        if optimizer is None:
            raise ValueError(
                "optimizer cannot be None."
            )

        # -------------------------------------------------
        # Store core components
        # -------------------------------------------------

        self.model = model

        self.criterion = criterion

        self.optimizer = optimizer

        self.device = torch.device(
            device
        )

        # -------------------------------------------------
        # Experiment manager
        # -------------------------------------------------

        self.experiment = (
            experiment_manager
            if experiment_manager is not None
            else ExperimentManager()
        )

        # -------------------------------------------------
        # Move model to device
        # -------------------------------------------------

        self.model.to(
            self.device
        )

        # -------------------------------------------------
        # Checkpoint directory
        # -------------------------------------------------

        self.checkpoint_directory = (
            self.experiment.global_checkpoints
        )

        os.makedirs(
            self.checkpoint_directory,
            exist_ok=True
        )

        # -------------------------------------------------
        # TensorBoard
        # -------------------------------------------------

        self.writer = SummaryWriter(
            log_dir=self.experiment.tensorboard
        )

        # -------------------------------------------------
        # Training history CSV
        # -------------------------------------------------

        self.log_file = os.path.join(
            self.experiment.logs,
            "training_history.csv"
        )

        os.makedirs(
            self.experiment.logs,
            exist_ok=True
        )

        self._create_history_file()

        # -------------------------------------------------
        # Training history in memory
        # -------------------------------------------------

        self.history = {
            "epoch": [],
            "train_total": [],
            "validation_total": [],
            "train_mae": [],
            "validation_mae": [],
            "train_physics": [],
            "validation_physics": [],
            "train_uncertainty": [],
            "validation_uncertainty": [],
            "train_ssim": [],
            "validation_ssim": [],
            "metric_mae": [],
            "metric_rmse": [],
            "metric_psnr": [],
            "metric_snr": [],
            "metric_ssim": [],
            "learning_rate": [],
        }

        # -------------------------------------------------
        # Restore existing CSV history.
        #
        # IMPORTANT:
        #
        # Validation MAE is restored from:
        #
        #     Validation_MAE
        #
        # and NOT from:
        #
        #     RMSE
        # -------------------------------------------------

        self._restore_history_from_csv()

        # -------------------------------------------------
        # Best-model state
        # -------------------------------------------------

        self.best_validation_loss = float(
            "inf"
        )

        self.best_epoch = 0

        self.best_metrics = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "PSNR": 0.0,
            "SNR": 0.0,
            "SSIM": 0.0,
        }

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        self.patience = int(
            config.PATIENCE
        )

        self.wait = 0

        # -------------------------------------------------
        # Learning-rate scheduler
        #
        # No scheduler-specific project parameters are
        # present in utils.config.py.
        #
        # Therefore PyTorch's default scheduler parameters
        # are used rather than introducing undocumented
        # project-level configuration values.
        # -------------------------------------------------

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
        )

        # -------------------------------------------------
        # Current epoch
        # -------------------------------------------------

        self.current_epoch = 0


    # =====================================================
    # CREATE HISTORY FILE
    # =====================================================

    def _create_history_file(self):
        """
        Create the training-history CSV file when it does
        not already exist.
        """

        if os.path.exists(
            self.log_file
        ):
            return

        with open(
            self.log_file,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                "Epoch",
                "Train_Total",
                "Validation_Total",
                "Train_MAE",
                "Validation_MAE",
                "Train_Physics",
                "Validation_Physics",
                "Train_Uncertainty",
                "Validation_Uncertainty",
                "Train_SSIM",
                "Validation_SSIM",
                "RMSE",
                "PSNR",
                "SNR",
                "Metric_SSIM",
                "Learning_Rate",
            ])


    # =====================================================
    # RESTORE CSV HISTORY
    # =====================================================

    def _restore_history_from_csv(self):
        """
        Restore training history from the existing CSV file.

        This method deliberately maps every metric to its
        correct CSV column.

        In particular:

            metric_mae <- Validation_MAE

        and never:

            metric_mae <- RMSE
        """

        if not os.path.exists(
            self.log_file
        ):
            return

        try:

            with open(
                self.log_file,
                "r",
                newline="",
            ) as file:

                reader = csv.DictReader(
                    file
                )

                required_columns = {
                    "Epoch",
                    "Train_Total",
                    "Validation_Total",
                    "Train_MAE",
                    "Validation_MAE",
                    "Train_Physics",
                    "Validation_Physics",
                    "Train_Uncertainty",
                    "Validation_Uncertainty",
                    "Train_SSIM",
                    "Validation_SSIM",
                    "RMSE",
                    "PSNR",
                    "SNR",
                    "Metric_SSIM",
                    "Learning_Rate",
                }

                if reader.fieldnames is None:

                    return

                missing_columns = (
                    required_columns
                    - set(reader.fieldnames)
                )

                if missing_columns:

                    print(
                        "Training-history CSV does not "
                        "contain all expected columns."
                    )

                    return

                for row in reader:

                    self.history["epoch"].append(
                        int(row["Epoch"])
                    )

                    self.history["train_total"].append(
                        float(row["Train_Total"])
                    )

                    self.history["validation_total"].append(
                        float(row["Validation_Total"])
                    )

                    self.history["train_mae"].append(
                        float(row["Train_MAE"])
                    )

                    # -------------------------------------------------
                    # CRITICAL CORRECTION
                    # -------------------------------------------------

                    self.history["validation_mae"].append(
                        float(row["Validation_MAE"])
                    )

                    self.history["train_physics"].append(
                        float(row["Train_Physics"])
                    )

                    self.history["validation_physics"].append(
                        float(row["Validation_Physics"])
                    )

                    self.history["train_uncertainty"].append(
                        float(row["Train_Uncertainty"])
                    )

                    self.history["validation_uncertainty"].append(
                        float(row["Validation_Uncertainty"])
                    )

                    self.history["train_ssim"].append(
                        float(row["Train_SSIM"])
                    )

                    self.history["validation_ssim"].append(
                        float(row["Validation_SSIM"])
                    )

                    self.history["metric_mae"].append(
                        float(row["Validation_MAE"])
                    )

                    self.history["metric_rmse"].append(
                        float(row["RMSE"])
                    )

                    self.history["metric_psnr"].append(
                        float(row["PSNR"])
                    )

                    self.history["metric_snr"].append(
                        float(row["SNR"])
                    )

                    self.history["metric_ssim"].append(
                        float(row["Metric_SSIM"])
                    )

                    self.history["learning_rate"].append(
                        float(row["Learning_Rate"])
                    )

        except (
            OSError,
            ValueError,
            KeyError,
        ) as error:

            print(
                "Warning: existing training history "
                f"could not be restored: {error}"
            )


    # =====================================================
    # DATA BATCH VALIDATION
    # =====================================================

    @staticmethod
    def _validate_batch(batch):
        """
        Validate the standard dataset batch.

        Required:

            (
                inputs,
                targets,
                mask,
                velocity_model,
                mask_type,
                geological_mode
            )

        The first four elements are tensors used for training.
        The final two elements are dataset metadata.
        """

        if not isinstance(
                batch,
                (tuple, list)
        ):
            raise TypeError(
                "Dataset batch must be a tuple or list."
            )

        if len(batch) != 6:
            raise ValueError(
                "Expected dataset batch to contain exactly "
                "six elements: "
                "(inputs, targets, mask, velocity_model, "
                "mask_type, geological_mode)."
            )

        (
            inputs,
            targets,
            mask,
            velocity_model,
            mask_type,
            geological_mode,
        ) = batch

        # =================================================
        # VALIDATE TENSORS
        # =================================================

        tensors = (
            inputs,
            targets,
            mask,
            velocity_model,
        )

        tensor_names = (
            "inputs",
            "targets",
            "mask",
            "velocity_model",
        )

        for tensor, name in zip(
                tensors,
                tensor_names,
        ):

            if not isinstance(
                    tensor,
                    torch.Tensor
            ):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"but received "
                    f"{type(tensor).__name__}."
                )

        # ---------------------------------------------------------
        # Validate metadata after DataLoader collation
        # ---------------------------------------------------------
        #
        # PyTorch's default DataLoader collation converts string
        # metadata into a sequence of strings.
        #
        # Example for batch_size=1:
        #
        #     "missing_crosslines"
        #
        # becomes:
        #
        #     ["missing_crosslines"]
        #
        # Therefore, the Trainer should validate the metadata as
        # a list/tuple of strings rather than requiring a single
        # string.
        # ---------------------------------------------------------

        if not isinstance(
                mask_type,
                (list, tuple)
        ):
            raise TypeError(
                "mask_type must be a list or tuple of strings "
                "after DataLoader collation."
            )

        if not all(
                isinstance(item, str)
                for item in mask_type
        ):
            raise TypeError(
                "Every mask_type metadata value must be a string."
            )

        if not isinstance(
                geological_mode,
                (list, tuple)
        ):
            raise TypeError(
                "geological_mode must be a list or tuple of strings "
                "after DataLoader collation."
            )

        if not all(
                isinstance(item, str)
                for item in geological_mode
        ):
            raise TypeError(
                "Every geological_mode metadata value must be a string."
            )

    # =====================================================
    # TENSOR VALIDATION
    # =====================================================

    @staticmethod
    def _validate_tensor(
        tensor,
        name,
    ):
        """
        Validate tensor type and finite values.
        """

        if not isinstance(
            tensor,
            torch.Tensor
        ):

            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if tensor.numel() == 0:

            raise ValueError(
                f"{name} is empty."
            )

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or infinite values."
            )


    # =====================================================
    # TRAINING EPOCH
    # =====================================================

    def train_epoch(
        self,
        dataloader,
    ):
        """
        Train the model for one epoch.
        """

        self.model.train()

        running_total = 0.0
        running_mae = 0.0
        running_physics = 0.0
        running_uncertainty = 0.0
        running_ssim = 0.0

        num_batches = len(
            dataloader
        )

        if num_batches == 0:

            raise RuntimeError(
                "Training DataLoader contains no batches."
            )

        for batch_index, batch in enumerate(
            dataloader
        ):

            self._validate_batch(
                batch
            )

            (
                inputs,
                targets,
                mask,
                velocity_model,
                mask_type,
                geological_mode,
            ) = batch

            # -------------------------------------------------
            # Move data to device
            # -------------------------------------------------

            inputs = inputs.to(
                self.device,
                non_blocking=True
            )

            targets = targets.to(
                self.device,
                non_blocking=True
            )

            mask = mask.to(
                self.device,
                non_blocking=True
            )

            velocity_model = velocity_model.to(
                self.device,
                non_blocking=True
            )

            # -------------------------------------------------
            # Validate tensors
            # -------------------------------------------------

            self._validate_tensor(
                inputs,
                "inputs"
            )

            self._validate_tensor(
                targets,
                "targets"
            )

            self._validate_tensor(
                mask,
                "mask"
            )

            self._validate_tensor(
                velocity_model,
                "velocity_model"
            )

            # -------------------------------------------------
            # Clear gradients
            # -------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # =================================================
            # FORWARD PASS
            # =================================================

            (
                reconstruction,
                travel_time,
                log_variance,
            ) = self.model(
                inputs
            )

            # -------------------------------------------------
            # Validate output shapes
            # -------------------------------------------------

            if reconstruction.shape != targets.shape:

                raise RuntimeError(
                    "Reconstruction and target shapes "
                    "do not match.\n"
                    f"Reconstruction: "
                    f"{tuple(reconstruction.shape)}\n"
                    f"Target: "
                    f"{tuple(targets.shape)}"
                )

            if travel_time.shape != velocity_model.shape:

                raise RuntimeError(
                    "Travel-time and velocity-model shapes "
                    "do not match.\n"
                    f"Travel-time: "
                    f"{tuple(travel_time.shape)}\n"
                    f"Velocity: "
                    f"{tuple(velocity_model.shape)}"
                )

            if log_variance.shape != reconstruction.shape:

                raise RuntimeError(
                    "Log-variance and reconstruction shapes "
                    "do not match.\n"
                    f"Log-variance: "
                    f"{tuple(log_variance.shape)}\n"
                    f"Reconstruction: "
                    f"{tuple(reconstruction.shape)}"
                )

            # -------------------------------------------------
            # Validate network outputs
            # -------------------------------------------------

            self._validate_tensor(
                reconstruction,
                "reconstruction"
            )

            self._validate_tensor(
                travel_time,
                "travel_time"
            )

            self._validate_tensor(
                log_variance,
                "log_variance"
            )

            # =================================================
            # COMPOSITE LOSS
            # =================================================

            losses = self.criterion(
                prediction=reconstruction,
                target=targets,
                velocity=velocity_model,
                travel_time=travel_time,
                log_variance=log_variance,
            )

            loss = losses["total_loss"]

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite training loss detected at "
                    f"batch {batch_index + 1}."
                )

            # =================================================
            # BACKPROPAGATION
            # =================================================

            loss.backward()

            # -------------------------------------------------
            # Gradient validation
            # -------------------------------------------------

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            if not torch.isfinite(
                torch.as_tensor(
                    gradient_norm,
                    device=self.device,
                )
            ):

                raise RuntimeError(
                    "Non-finite gradient norm detected at "
                    f"batch {batch_index + 1}."
                )

            # -------------------------------------------------
            # Optimizer update
            # -------------------------------------------------

            self.optimizer.step()

            # =================================================
            # ACCUMULATE LOSS COMPONENTS
            # =================================================

            running_total += (
                losses["total_loss"]
                .detach()
                .item()
            )

            running_mae += (
                losses["mae_loss"]
                .detach()
                .item()
            )

            running_physics += (
                losses["physics_loss"]
                .detach()
                .item()
            )

            running_uncertainty += (
                losses["uncertainty_loss"]
                .detach()
                .item()
            )

            running_ssim += (
                losses["ssim_loss"]
                .detach()
                .item()
            )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                batch_index == 0
                or (batch_index + 1) == num_batches
            ):

                print(
                    f"Training batch "
                    f"{batch_index + 1}/{num_batches}",
                    flush=True
                )

        # =====================================================
        # RETURN AVERAGES
        # =====================================================

        return {

            "total":
                running_total / num_batches,

            "mae":
                running_mae / num_batches,

            "physics":
                running_physics / num_batches,

            "uncertainty":
                running_uncertainty / num_batches,

            "ssim":
                running_ssim / num_batches,
        }


    # =====================================================
    # VALIDATION EPOCH
    # =====================================================

    def validate_epoch(
            self,
            dataloader,
    ):
        """
        Validate the model for one epoch.

        Validation is deterministic because the model is
        placed in evaluation mode and gradients are disabled.

        Dataset batch format:

            (
                inputs,
                targets,
                mask,
                velocity_model,
                mask_type,
                geological_mode
            )

        The first four elements are tensors used by the
        validation pipeline.

        The final two elements are metadata produced by the
        synthetic dataset and collated by PyTorch's DataLoader.
        """

        # =========================================================
        # EVALUATION MODE
        # =========================================================

        self.model.eval()

        # ---------------------------------------------------------
        # Running composite-loss components
        # ---------------------------------------------------------

        running_total = 0.0
        running_mae = 0.0
        running_physics = 0.0
        running_uncertainty = 0.0
        running_ssim = 0.0

        # ---------------------------------------------------------
        # Running reconstruction metrics
        # ---------------------------------------------------------

        running_metric_mae = 0.0
        running_metric_rmse = 0.0
        running_metric_psnr = 0.0
        running_metric_snr = 0.0
        running_metric_ssim = 0.0

        # =========================================================
        # VALIDATION DATALOADER CHECK
        # =========================================================

        num_batches = len(
            dataloader
        )

        if num_batches == 0:
            raise RuntimeError(
                "Validation DataLoader contains no batches."
            )

        # =========================================================
        # VALIDATION LOOP
        # =========================================================

        with torch.no_grad():

            for batch_index, batch in enumerate(
                    dataloader
            ):

                # -------------------------------------------------
                # Validate complete six-element dataset batch
                # -------------------------------------------------

                self._validate_batch(
                    batch
                )

                (
                    inputs,
                    targets,
                    mask,
                    velocity_model,
                    mask_type,
                    geological_mode,
                ) = batch

                # -------------------------------------------------
                # Move tensor data to the selected device
                # -------------------------------------------------

                inputs = inputs.to(
                    self.device,
                    non_blocking=True
                )

                targets = targets.to(
                    self.device,
                    non_blocking=True
                )

                mask = mask.to(
                    self.device,
                    non_blocking=True
                )

                velocity_model = velocity_model.to(
                    self.device,
                    non_blocking=True
                )

                # =================================================
                # MODEL FORWARD PASS
                # =================================================

                (
                    reconstruction,
                    travel_time,
                    log_variance,
                ) = self.model(
                    inputs
                )

                # =================================================
                # SHAPE VALIDATION
                # =================================================

                if reconstruction.shape != targets.shape:
                    raise RuntimeError(
                        "Validation reconstruction and target "
                        "shapes do not match."
                    )

                if travel_time.shape != velocity_model.shape:
                    raise RuntimeError(
                        "Validation travel-time and velocity "
                        "shapes do not match."
                    )

                if log_variance.shape != reconstruction.shape:
                    raise RuntimeError(
                        "Validation log-variance and "
                        "reconstruction shapes do not match."
                    )

                # =================================================
                # COMPOSITE VALIDATION LOSS
                # =================================================
                #
                # Use explicit keyword arguments because the
                # TotalLoss interface is:
                #
                # prediction
                # target
                # velocity
                # source_indices
                # travel_time
                # travel_time_target
                # log_variance
                #
                # Source and travel-time target are not being
                # supplied in the current synthetic configuration.
                # =================================================

                losses = self.criterion(
                    prediction=reconstruction,
                    target=targets,
                    velocity=velocity_model,
                    travel_time=travel_time,
                    log_variance=log_variance,
                )

                # =================================================
                # RECONSTRUCTION METRICS
                # =================================================

                metric_mae = mae(
                    reconstruction,
                    targets
                )

                metric_rmse = rmse(
                    reconstruction,
                    targets
                )

                metric_psnr = psnr(
                    reconstruction,
                    targets
                )

                metric_snr = snr(
                    reconstruction,
                    targets
                )

                metric_ssim = ssim(
                    reconstruction,
                    targets
                )

                # =================================================
                # FINITE CHECKS
                # =================================================

                values_to_check = {

                    "validation_total":
                        losses["total_loss"],

                    "validation_mae":
                        metric_mae,

                    "validation_rmse":
                        metric_rmse,

                    "validation_psnr":
                        metric_psnr,

                    "validation_snr":
                        metric_snr,

                    "validation_ssim":
                        metric_ssim,
                }

                for name, value in values_to_check.items():

                    if not torch.isfinite(
                            torch.as_tensor(value)
                    ).all():
                        raise RuntimeError(
                            f"Non-finite {name} detected."
                        )

                # =================================================
                # ACCUMULATE COMPOSITE LOSS COMPONENTS
                # =================================================

                running_total += (
                    losses["total_loss"]
                    .detach()
                    .item()
                )

                running_mae += (
                    losses["mae_loss"]
                    .detach()
                    .item()
                )

                running_physics += (
                    losses["physics_loss"]
                    .detach()
                    .item()
                )

                running_uncertainty += (
                    losses["uncertainty_loss"]
                    .detach()
                    .item()
                )

                running_ssim += (
                    losses["ssim_loss"]
                    .detach()
                    .item()
                )

                # =================================================
                # ACCUMULATE RECONSTRUCTION METRICS
                # =================================================

                running_metric_mae += (
                    metric_mae
                    .detach()
                    .item()
                )

                running_metric_rmse += (
                    metric_rmse
                    .detach()
                    .item()
                )

                running_metric_psnr += (
                    metric_psnr
                    .detach()
                    .item()
                )

                running_metric_snr += (
                    metric_snr
                    .detach()
                    .item()
                )

                running_metric_ssim += (
                    metric_ssim
                    .detach()
                    .item()
                )

                # =================================================
                # FIRST VALIDATION VISUALIZATION
                # =================================================

                if batch_index == 0:
                    uncertainty = torch.exp(
                        0.5 * log_variance
                    )

                    self.save_validation_visualization(
                        inputs,
                        targets,
                        reconstruction,
                        travel_time,
                        uncertainty,
                        self.current_epoch,
                    )

                # =================================================
                # OPTIONAL DEBUG INFORMATION
                # =================================================

                if (
                        DEBUG_VALIDATION
                        and batch_index == 0
                ):
                    print()
                    print(
                        "Validation Batch Statistics"
                    )

                    print(
                        "-" * 40
                    )

                    print(
                        f"Prediction min : "
                        f"{reconstruction.min().item():.6f}"
                    )

                    print(
                        f"Prediction max : "
                        f"{reconstruction.max().item():.6f}"
                    )

                    print(
                        f"Target min     : "
                        f"{targets.min().item():.6f}"
                    )

                    print(
                        f"Target max     : "
                        f"{targets.max().item():.6f}"
                    )

                    print(
                        f"Travel-time min: "
                        f"{travel_time.min().item():.6e}"
                    )

                    print(
                        f"Travel-time max: "
                        f"{travel_time.max().item():.6e}"
                    )

                    print(
                        f"Log-var min    : "
                        f"{log_variance.min().item():.6f}"
                    )

                    print(
                        f"Log-var max    : "
                        f"{log_variance.max().item():.6f}"
                    )

        # =========================================================
        # RETURN VALIDATION RESULTS
        # =========================================================

        return {

            "total":
                running_total / num_batches,

            "mae":
                running_mae / num_batches,

            "physics":
                running_physics / num_batches,

            "uncertainty":
                running_uncertainty / num_batches,

            "ssim":
                running_ssim / num_batches,

            "metric_mae":
                running_metric_mae / num_batches,

            "metric_rmse":
                running_metric_rmse / num_batches,

            "metric_psnr":
                running_metric_psnr / num_batches,

            "metric_snr":
                running_metric_snr / num_batches,

            "metric_ssim":
                running_metric_ssim / num_batches,
        }

    # =====================================================
    # VALIDATION VISUALIZATION
    # =====================================================

    def save_validation_visualization(
        self,
        inputs,
        targets,
        reconstruction,
        travel_time,
        uncertainty,
        epoch,
    ):
        """
        Save a validation visualization.

        The figure contains:

            1. Incomplete input
            2. Ground truth
            3. Reconstruction
            4. Travel-time field
            5. Predictive uncertainty
            6. Absolute reconstruction error
        """

        output_directory = (
            self.experiment.training_progress
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        # -------------------------------------------------
        # First sample / first channel
        # -------------------------------------------------

        inputs_np = (
            inputs[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        targets_np = (
            targets[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        reconstruction_np = (
            reconstruction[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        travel_time_np = (
            travel_time[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        uncertainty_np = (
            uncertainty[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------
        # Middle depth slice
        # -------------------------------------------------

        middle = (
            inputs_np.shape[0] // 2
        )

        # -------------------------------------------------
        # Figure
        # -------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9)
        )

        # -------------------------------------------------
        # Input
        # -------------------------------------------------

        axes[0, 0].imshow(
            inputs_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 0].set_title(
            "Incomplete Input"
        )

        # -------------------------------------------------
        # Target
        # -------------------------------------------------

        axes[0, 1].imshow(
            targets_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 1].set_title(
            "Ground Truth"
        )

        # -------------------------------------------------
        # Reconstruction
        # -------------------------------------------------

        axes[0, 2].imshow(
            reconstruction_np[middle],
            cmap="gray",
            aspect="auto"
        )

        axes[0, 2].set_title(
            "Reconstruction"
        )

        # -------------------------------------------------
        # Travel time
        # -------------------------------------------------

        travel_image = axes[1, 0].imshow(
            travel_time_np[middle],
            cmap="viridis",
            aspect="auto"
        )

        axes[1, 0].set_title(
            "Predicted Travel Time"
        )

        fig.colorbar(
            travel_image,
            ax=axes[1, 0],
            fraction=0.046,
            pad=0.04
        )

        # -------------------------------------------------
        # Predictive uncertainty
        # -------------------------------------------------

        uncertainty_image = axes[1, 1].imshow(
            uncertainty_np[middle],
            cmap="hot",
            aspect="auto"
        )

        axes[1, 1].set_title(
            "Predictive Uncertainty"
        )

        fig.colorbar(
            uncertainty_image,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04
        )

        # -------------------------------------------------
        # Absolute reconstruction error
        # -------------------------------------------------

        absolute_error = (
            abs(
                reconstruction_np
                - targets_np
            )
        )

        error_image = axes[1, 2].imshow(
            absolute_error[middle],
            cmap="magma",
            aspect="auto"
        )

        axes[1, 2].set_title(
            "Absolute Reconstruction Error"
        )

        fig.colorbar(
            error_image,
            ax=axes[1, 2],
            fraction=0.046,
            pad=0.04
        )

        # -------------------------------------------------
        # Remove axes
        # -------------------------------------------------

        for axis in axes.flat:

            axis.axis("off")

        plt.tight_layout()

        # -------------------------------------------------
        # Save
        # -------------------------------------------------

        output_file = os.path.join(
            output_directory,
            f"epoch_{epoch:03d}.png"
        )

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )


    # =====================================================
    # CHECKPOINT METADATA
    # =====================================================

    @staticmethod
    def _configuration_snapshot():
        """
        Create a compact, JSON-serializable snapshot of the
        active project configuration.

        Only simple public configuration values are stored.
        """

        snapshot = {}

        for name in dir(config):

            if name.startswith("_"):
                continue

            value = getattr(
                config,
                name
            )

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                )
            ):

                snapshot[name] = value

            elif isinstance(
                value,
                (list, tuple)
            ):

                try:
                    json.dumps(
                        value
                    )

                    snapshot[name] = value

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

            elif isinstance(
                value,
                dict
            ):

                try:
                    json.dumps(
                        value
                    )

                    snapshot[name] = value

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        return snapshot


    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        epoch,
        loss,
        periodic=False,
    ):
        """
        Save the latest checkpoint.

        Parameters
        ----------
        epoch : int
            Completed epoch number.

        loss : float
            Validation loss.

        periodic : bool
            Whether this is a periodic epoch checkpoint.
        """

        checkpoint = {

            "epoch":
                int(epoch),

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_validation_loss":
                float(
                    self.best_validation_loss
                ),

            "best_epoch":
                int(
                    self.best_epoch
                ),

            "best_metrics":
                dict(
                    self.best_metrics
                ),

            "loss":
                float(loss),

            "experiment_metadata": {

                "dataset_mode":
                    getattr(
                        config,
                        "DATASET_MODE",
                        None
                    ),

                "experiment_name":
                    getattr(
                        config,
                        "EXPERIMENT_NAME",
                        None
                    ),

                "synthetic_num_samples":
                    getattr(
                        config,
                        "SYNTHETIC_NUM_SAMPLES",
                        None
                    ),

                "synthetic_patch_size":
                    getattr(
                        config,
                        "SYNTHETIC_PATCH_SIZE",
                        None
                    ),

                "validation_split":
                    getattr(
                        config,
                        "VALIDATION_SPLIT",
                        None
                    ),
            },

            "config_snapshot":
                self._configuration_snapshot(),
        }

        # -------------------------------------------------
        # Latest checkpoint
        # -------------------------------------------------

        latest_file = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth"
        )

        torch.save(
            checkpoint,
            latest_file
        )

        # -------------------------------------------------
        # Periodic checkpoint
        #
        # SAVE_EVERY is now actually honored.
        # -------------------------------------------------

        if periodic:

            epoch_directory = os.path.join(
                self.checkpoint_directory,
                "epoch_checkpoints"
            )

            os.makedirs(
                epoch_directory,
                exist_ok=True
            )

            epoch_file = os.path.join(
                epoch_directory,
                f"epoch_{epoch:04d}.pth"
            )

            torch.save(
                checkpoint,
                epoch_file
            )

            print(
                f"Periodic checkpoint saved: "
                f"{epoch_file}"
            )

        # -------------------------------------------------
        # Training state
        # -------------------------------------------------

        state_file = os.path.join(
            self.checkpoint_directory,
            "training_state.json"
        )

        with open(
            state_file,
            "w"
        ) as file:

            json.dump(
                {
                    "current_epoch":
                        int(epoch),

                    "best_epoch":
                        int(self.best_epoch),

                    "best_validation_loss":
                        float(
                            self.best_validation_loss
                        ),

                    "best_metrics":
                        self.best_metrics,

                    "dataset_mode":
                        getattr(
                            config,
                            "DATASET_MODE",
                            None
                        ),

                    "experiment_name":
                        getattr(
                            config,
                            "EXPERIMENT_NAME",
                            None
                        ),
                },
                file,
                indent=4
            )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    def save_best_model(
        self,
        epoch,
        validation_loss,
    ):
        """
        Save the best-performing model checkpoint.
        """

        checkpoint = {

            "epoch":
                int(epoch),

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_validation_loss":
                float(
                    self.best_validation_loss
                ),

            "best_epoch":
                int(
                    self.best_epoch
                ),

            "best_metrics":
                dict(
                    self.best_metrics
                ),

            "loss":
                float(validation_loss),

            "experiment_metadata": {

                "dataset_mode":
                    getattr(
                        config,
                        "DATASET_MODE",
                        None
                    ),

                "experiment_name":
                    getattr(
                        config,
                        "EXPERIMENT_NAME",
                        None
                    ),

                "synthetic_num_samples":
                    getattr(
                        config,
                        "SYNTHETIC_NUM_SAMPLES",
                        None
                    ),

                "synthetic_patch_size":
                    getattr(
                        config,
                        "SYNTHETIC_PATCH_SIZE",
                        None
                    ),
            },

            "config_snapshot":
                self._configuration_snapshot(),
        }

        best_file = os.path.join(
            self.checkpoint_directory,
            "best_model.pth"
        )

        torch.save(
            checkpoint,
            best_file
        )

        print()
        print(
            "NEW BEST MODEL SAVED"
        )
        print(
            f"Best epoch : {epoch}"
        )
        print(
            f"Best loss  : "
            f"{validation_loss:.6f}"
        )


    # =====================================================
    # CHECKPOINT COMPATIBILITY
    # =====================================================

    def _validate_checkpoint_metadata(
        self,
        checkpoint,
    ):
        """
        Validate important experiment identity fields.

        The model state itself is validated separately through
        strict load_state_dict().
        """

        metadata = checkpoint.get(
            "experiment_metadata"
        )

        if metadata is None:

            print(
                "Warning: checkpoint contains no experiment "
                "metadata. Model compatibility will still be "
                "checked through the state dictionary."
            )

            return

        current_dataset_mode = getattr(
            config,
            "DATASET_MODE",
            None
        )

        current_experiment_name = getattr(
            config,
            "EXPERIMENT_NAME",
            None
        )

        stored_dataset_mode = metadata.get(
            "dataset_mode"
        )

        stored_experiment_name = metadata.get(
            "experiment_name"
        )

        if (
            stored_dataset_mode is not None
            and current_dataset_mode is not None
            and stored_dataset_mode
            != current_dataset_mode
        ):

            raise RuntimeError(
                "Checkpoint dataset-mode mismatch.\n"
                f"Checkpoint: {stored_dataset_mode}\n"
                f"Current   : {current_dataset_mode}"
            )

        if (
            stored_experiment_name is not None
            and current_experiment_name is not None
            and stored_experiment_name
            != current_experiment_name
        ):

            raise RuntimeError(
                "Checkpoint experiment-name mismatch.\n"
                f"Checkpoint: {stored_experiment_name}\n"
                f"Current   : {current_experiment_name}"
            )


    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    def load_checkpoint(
        self,
        checkpoint_path=None,
    ):
        """
        Load a previously saved training checkpoint.

        Returns
        -------
        int
            Next epoch from which training should continue.
        """

        if checkpoint_path is None:

            checkpoint_path = os.path.join(
                self.checkpoint_directory,
                "latest_checkpoint.pth"
            )

        if not os.path.exists(
            checkpoint_path
        ):

            print()
            print(
                "=" * 60
            )
            print(
                "No checkpoint found."
            )
            print(
                "Starting training from scratch."
            )
            print(
                "=" * 60
            )

            return 1

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # -------------------------------------------------
        # Basic checkpoint validation
        # -------------------------------------------------

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
        }

        missing_keys = (
            required_keys
            - set(checkpoint.keys())
        )

        if missing_keys:

            raise RuntimeError(
                "Checkpoint is incomplete. Missing keys: "
                f"{sorted(missing_keys)}"
            )

        # -------------------------------------------------
        # Metadata validation
        # -------------------------------------------------

        self._validate_checkpoint_metadata(
            checkpoint
        )

        # -------------------------------------------------
        # Model compatibility
        # -------------------------------------------------

        try:

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )

        except RuntimeError as error:

            raise RuntimeError(
                "Checkpoint model architecture is not "
                "compatible with the current Network3D "
                "configuration.\n\n"
                f"{error}"
            ) from error

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        if (
            "scheduler_state_dict"
            in checkpoint
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        # -------------------------------------------------
        # Restore best state
        # -------------------------------------------------

        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get(
                    "loss",
                    float("inf")
                )
            )
        )

        self.best_epoch = int(
            checkpoint.get(
                "best_epoch",
                checkpoint["epoch"]
            )
        )

        self.best_metrics = dict(
            checkpoint.get(
                "best_metrics",
                self.best_metrics
            )
        )

        # -------------------------------------------------
        # Resume from the next epoch
        # -------------------------------------------------

        last_completed_epoch = int(
            checkpoint["epoch"]
        )

        start_epoch = (
            last_completed_epoch + 1
        )

        print()
        print(
            "=" * 60
        )
        print(
            "CHECKPOINT LOADED SUCCESSFULLY"
        )
        print(
            "=" * 60
        )
        print(
            f"Last completed epoch : "
            f"{last_completed_epoch}"
        )
        print(
            f"Best epoch           : "
            f"{self.best_epoch}"
        )
        print(
            f"Best validation loss : "
            f"{self.best_validation_loss:.6f}"
        )
        print(
            f"Next epoch           : "
            f"{start_epoch}"
        )
        print(
            "=" * 60
        )

        return start_epoch


    # =====================================================
    # RECORD HISTORY
    # =====================================================

    def _record_history(
        self,
        epoch,
        train_losses,
        validation_losses,
        learning_rate,
    ):
        """
        Store the current epoch results in memory and CSV.
        """

        epoch_number = int(
            epoch
        )

        # -------------------------------------------------
        # In-memory history
        # -------------------------------------------------

        self.history["epoch"].append(
            epoch_number
        )

        self.history["train_total"].append(
            train_losses["total"]
        )

        self.history["validation_total"].append(
            validation_losses["total"]
        )

        self.history["train_mae"].append(
            train_losses["mae"]
        )

        self.history["validation_mae"].append(
            validation_losses["mae"]
        )

        self.history["train_physics"].append(
            train_losses["physics"]
        )

        self.history["validation_physics"].append(
            validation_losses["physics"]
        )

        self.history["train_uncertainty"].append(
            train_losses["uncertainty"]
        )

        self.history["validation_uncertainty"].append(
            validation_losses["uncertainty"]
        )

        self.history["train_ssim"].append(
            train_losses["ssim"]
        )

        self.history["validation_ssim"].append(
            validation_losses["ssim"]
        )

        self.history["metric_mae"].append(
            validation_losses["metric_mae"]
        )

        self.history["metric_rmse"].append(
            validation_losses["metric_rmse"]
        )

        self.history["metric_psnr"].append(
            validation_losses["metric_psnr"]
        )

        self.history["metric_snr"].append(
            validation_losses["metric_snr"]
        )

        self.history["metric_ssim"].append(
            validation_losses["metric_ssim"]
        )

        self.history["learning_rate"].append(
            learning_rate
        )

        # -------------------------------------------------
        # CSV
        # -------------------------------------------------

        with open(
            self.log_file,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([

                epoch_number,

                train_losses["total"],

                validation_losses["total"],

                train_losses["mae"],

                validation_losses["mae"],

                train_losses["physics"],

                validation_losses["physics"],

                train_losses["uncertainty"],

                validation_losses["uncertainty"],

                train_losses["ssim"],

                validation_losses["ssim"],

                validation_losses["metric_rmse"],

                validation_losses["metric_psnr"],

                validation_losses["metric_snr"],

                validation_losses["metric_ssim"],

                learning_rate,
            ])


    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        train_dataloader,
        validation_dataloader,
        epochs=None,
        resume=True,
    ):
        """
        Complete training procedure.
        """

        if epochs is None:

            epochs = int(
                config.NUM_EPOCHS
            )

        if epochs <= 0:

            raise ValueError(
                "epochs must be greater than zero."
            )

        start_time = time.time()

        latest_checkpoint = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth"
        )

        # -------------------------------------------------
        # Resume
        # -------------------------------------------------

        if (
            resume
            and os.path.exists(
                latest_checkpoint
            )
        ):

            start_epoch = (
                self.load_checkpoint(
                    latest_checkpoint
                )
            )

        else:

            start_epoch = 1

        # -------------------------------------------------
        # Already complete
        # -------------------------------------------------

        if start_epoch > epochs:

            print(
                "Training is already complete according "
                "to the latest checkpoint."
            )

            self.writer.close()

            return

        # =================================================
        # EPOCH LOOP
        # =================================================

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):

            self.current_epoch = epoch

            print()
            print(
                "=" * 70
            )
            print(
                f"Epoch {epoch}/{epochs}"
            )
            print(
                "=" * 70
            )

            # =================================================
            # TRAIN
            # =================================================

            train_losses = self.train_epoch(
                train_dataloader
            )

            # =================================================
            # VALIDATION
            # =================================================

            validation_losses = (
                self.validate_epoch(
                    validation_dataloader
                )
            )

            # =================================================
            # SCHEDULER
            # =================================================

            self.scheduler.step(
                validation_losses["total"]
            )

            current_lr = float(
                self.optimizer
                .param_groups[0]["lr"]
            )

            # =================================================
            # TENSORBOARD
            # =================================================

            self.writer.add_scalar(
                "Loss/Train_Total",
                train_losses["total"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation_Total",
                validation_losses["total"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Train_MAE",
                train_losses["mae"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation_MAE",
                validation_losses["mae"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Train_Physics",
                train_losses["physics"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation_Physics",
                validation_losses["physics"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Train_Uncertainty",
                train_losses["uncertainty"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation_Uncertainty",
                validation_losses["uncertainty"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Train_SSIM",
                train_losses["ssim"],
                epoch,
            )

            self.writer.add_scalar(
                "Loss/Validation_SSIM",
                validation_losses["ssim"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/Validation_MAE",
                validation_losses["metric_mae"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/Validation_RMSE",
                validation_losses["metric_rmse"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/Validation_PSNR",
                validation_losses["metric_psnr"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/Validation_SNR",
                validation_losses["metric_snr"],
                epoch,
            )

            self.writer.add_scalar(
                "Metrics/Validation_SSIM",
                validation_losses["metric_ssim"],
                epoch,
            )

            self.writer.add_scalar(
                "Learning_Rate",
                current_lr,
                epoch,
            )

            # =================================================
            # CONSOLE REPORT
            # =================================================

            print()
            print(
                "TRAINING"
            )
            print(
                "-" * 40
            )

            print(
                f"Total Loss       : "
                f"{train_losses['total']:.6f}"
            )

            print(
                f"MAE Loss         : "
                f"{train_losses['mae']:.6f}"
            )

            print(
                f"Physics Loss     : "
                f"{train_losses['physics']:.6f}"
            )

            print(
                f"Uncertainty Loss : "
                f"{train_losses['uncertainty']:.6f}"
            )

            print(
                f"SSIM Loss        : "
                f"{train_losses['ssim']:.6f}"
            )

            print()
            print(
                "VALIDATION"
            )
            print(
                "-" * 40
            )

            print(
                f"Total Loss       : "
                f"{validation_losses['total']:.6f}"
            )

            print(
                f"MAE Loss         : "
                f"{validation_losses['mae']:.6f}"
            )

            print(
                f"Physics Loss     : "
                f"{validation_losses['physics']:.6f}"
            )

            print(
                f"Uncertainty Loss : "
                f"{validation_losses['uncertainty']:.6f}"
            )

            print(
                f"SSIM Loss        : "
                f"{validation_losses['ssim']:.6f}"
            )

            print()
            print(
                "RECONSTRUCTION METRICS"
            )
            print(
                "-" * 40
            )

            print(
                f"MAE  : "
                f"{validation_losses['metric_mae']:.6f}"
            )

            print(
                f"RMSE : "
                f"{validation_losses['metric_rmse']:.6f}"
            )

            print(
                f"PSNR : "
                f"{validation_losses['metric_psnr']:.3f} dB"
            )

            print(
                f"SNR  : "
                f"{validation_losses['metric_snr']:.3f} dB"
            )

            print(
                f"SSIM : "
                f"{validation_losses['metric_ssim']:.6f}"
            )

            print(
                f"Learning Rate : "
                f"{current_lr:.8f}"
            )

            # =================================================
            # RECORD HISTORY
            # =================================================

            self._record_history(
                epoch,
                train_losses,
                validation_losses,
                current_lr,
            )

            # =================================================
            # BEST MODEL
            # =================================================

            if (
                validation_losses["total"]
                < self.best_validation_loss
            ):

                self.best_validation_loss = (
                    validation_losses["total"]
                )

                self.best_epoch = epoch

                self.best_metrics = {

                    "MAE":
                        validation_losses[
                            "metric_mae"
                        ],

                    "RMSE":
                        validation_losses[
                            "metric_rmse"
                        ],

                    "PSNR":
                        validation_losses[
                            "metric_psnr"
                        ],

                    "SNR":
                        validation_losses[
                            "metric_snr"
                        ],

                    "SSIM":
                        validation_losses[
                            "metric_ssim"
                        ],
                }

                self.wait = 0

                self.save_best_model(
                    epoch,
                    validation_losses["total"],
                )

            else:

                self.wait += 1

                print(
                    f"Early stopping counter: "
                    f"{self.wait}/{self.patience}"
                )

            # =================================================
            # PERIODIC / LATEST CHECKPOINT
            # =================================================

            periodic_save = (
                epoch
                % int(config.SAVE_EVERY)
                == 0
            )

            self.save_checkpoint(
                epoch=epoch,
                loss=validation_losses["total"],
                periodic=periodic_save,
            )

            # =================================================
            # EARLY STOPPING
            # =================================================

            if self.wait >= self.patience:

                print()
                print(
                    "=" * 60
                )
                print(
                    "Early stopping triggered."
                )
                print(
                    "=" * 60
                )

                break

        # =================================================
        # TRAINING SUMMARY
        # =================================================

        training_time = (
            time.time()
            - start_time
        )

        self.save_training_summary(
            best_epoch=self.best_epoch,
            best_validation_loss=(
                self.best_validation_loss
            ),
            best_metrics=self.best_metrics,
            training_time=training_time,
        )

        self.writer.close()


    # =====================================================
    # TRAINING SUMMARY
    # =====================================================

    def save_training_summary(
        self,
        best_epoch,
        best_validation_loss,
        best_metrics,
        training_time,
    ):
        """
        Save final training summary.
        """

        report_directory = (
            self.experiment.reports
        )

        os.makedirs(
            report_directory,
            exist_ok=True
        )

        report_file = os.path.join(
            report_directory,
            "training_summary.txt"
        )

        with open(
            report_file,
            "w"
        ) as file:

            file.write(
                "Physics-Informed 3D Seismic "
                "Reconstruction Training Summary\n"
            )

            file.write(
                "=" * 60 + "\n\n"
            )

            file.write(
                f"Dataset Mode          : "
                f"{getattr(config, 'DATASET_MODE', 'N/A')}\n"
            )

            file.write(
                f"Experiment Name       : "
                f"{getattr(config, 'EXPERIMENT_NAME', 'N/A')}\n"
            )

            file.write(
                f"Best Epoch            : "
                f"{best_epoch}\n"
            )

            file.write(
                f"Best Validation Loss  : "
                f"{best_validation_loss:.6f}\n\n"
            )

            file.write(
                f"Best MAE              : "
                f"{best_metrics['MAE']:.6f}\n"
            )

            file.write(
                f"Best RMSE             : "
                f"{best_metrics['RMSE']:.6f}\n"
            )

            file.write(
                f"Best PSNR             : "
                f"{best_metrics['PSNR']:.3f} dB\n"
            )

            file.write(
                f"Best SNR              : "
                f"{best_metrics['SNR']:.3f} dB\n"
            )

            file.write(
                f"Best SSIM             : "
                f"{best_metrics['SSIM']:.6f}\n\n"
            )

            file.write(
                f"Training Time         : "
                f"{training_time:.2f} seconds\n"
            )

            file.write(
                f"Number of Epochs      : "
                f"{getattr(config, 'NUM_EPOCHS', 'N/A')}\n"
            )

            file.write(
                f"Batch Size            : "
                f"{getattr(config, 'BATCH_SIZE', 'N/A')}\n"
            )

            file.write(
                f"Validation Split      : "
                f"{getattr(config, 'VALIDATION_SPLIT', 'N/A')}\n"
            )

            file.write(
                f"Learning Rate         : "
                f"{getattr(config, 'LEARNING_RATE', 'N/A')}\n"
            )

            file.write(
                f"Weight Decay          : "
                f"{getattr(config, 'WEIGHT_DECAY', 'N/A')}\n"
            )

            file.write(
                f"Checkpoint Frequency  : "
                f"{getattr(config, 'SAVE_EVERY', 'N/A')}\n"
            )

            file.write(
                f"Early Stopping Patience: "
                f"{getattr(config, 'PATIENCE', 'N/A')}\n"
            )