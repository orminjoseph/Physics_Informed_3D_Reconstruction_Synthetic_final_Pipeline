
"""
====================================================================
Trainer — Acoustic Wave-Equation Version
====================================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Physics:
    Acoustic wave equation implemented by AcousticWaveEquationLoss.

    (1 / v^2) * d²p/dt² - Laplacian(p) = source

Network outputs:
    1. Reconstructed seismic cube
    2. Pressure wavefield p(t, z, y, x)
    3. Log variance for predictive uncertainty

Dataset batch:
    inputs, targets, mask, velocity_model, source,
    mask_type, geological_mode

Shapes:
    inputs, targets, mask, velocity_model:
        [B, C, D, H, W]

    pressure_wavefield, source:
        [B, 1, T, D, H, W]

Important:
    This Trainer does not implement or modify the acoustic
    equation. It passes the pressure, velocity, and source
    to TotalLoss, which delegates the physics calculation
    to AcousticWaveEquationLoss.

    Non-finite loss or gradients cause a fatal error.
    Finite gradients are clipped before the optimizer step.

Author: Ormin Joseph
====================================================================
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


DEBUG_VALIDATION = False


class Trainer:
    """Train and evaluate the acoustic-wave reconstruction model."""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        experiment_manager=None,
    ):
        if model is None:
            raise ValueError("model cannot be None.")
        if criterion is None:
            raise ValueError("criterion cannot be None.")
        if optimizer is None:
            raise ValueError("optimizer cannot be None.")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)

        self.experiment = (
            experiment_manager
            if experiment_manager is not None
            else ExperimentManager()
        )

        self.model.to(self.device)

        self.checkpoint_directory = self.experiment.global_checkpoints
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        os.makedirs(self.experiment.logs, exist_ok=True)
        os.makedirs(self.experiment.reports, exist_ok=True)
        os.makedirs(self.experiment.training_progress, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=self.experiment.tensorboard
        )

        self.log_file = os.path.join(
            self.experiment.logs,
            "training_history.csv",
        )

        self._create_history_file()

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

        self._restore_history_from_csv()

        self.best_validation_loss = float("inf")
        self.best_epoch = 0
        self.best_metrics = {
            "MAE": 0.0,
            "RMSE": 0.0,
            "PSNR": 0.0,
            "SNR": 0.0,
            "SSIM": 0.0,
        }

        self.patience = int(config.PATIENCE)
        self.wait = 0
        self.current_epoch = 0

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
        )

    # ================================================================
    # HISTORY
    # ================================================================

    def _create_history_file(self):
        if os.path.exists(self.log_file):
            return

        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
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

    def _restore_history_from_csv(self):
        """Restore history using the matching CSV column for each metric."""
        if not os.path.exists(self.log_file):
            return

        try:
            with open(self.log_file, "r", newline="") as file:
                reader = csv.DictReader(file)

                if not reader.fieldnames:
                    return

                required = {
                    "Epoch", "Train_Total", "Validation_Total",
                    "Train_MAE", "Validation_MAE",
                    "Train_Physics", "Validation_Physics",
                    "Train_Uncertainty", "Validation_Uncertainty",
                    "Train_SSIM", "Validation_SSIM",
                    "RMSE", "PSNR", "SNR", "Metric_SSIM",
                    "Learning_Rate",
                }

                missing = required - set(reader.fieldnames)
                if missing:
                    print(
                        "Warning: history CSV is missing columns: "
                        f"{sorted(missing)}. Existing history was not restored."
                    )
                    return

                for row in reader:
                    self.history["epoch"].append(int(row["Epoch"]))
                    self.history["train_total"].append(float(row["Train_Total"]))
                    self.history["validation_total"].append(float(row["Validation_Total"]))
                    self.history["train_mae"].append(float(row["Train_MAE"]))
                    self.history["validation_mae"].append(float(row["Validation_MAE"]))
                    self.history["train_physics"].append(float(row["Train_Physics"]))
                    self.history["validation_physics"].append(float(row["Validation_Physics"]))
                    self.history["train_uncertainty"].append(float(row["Train_Uncertainty"]))
                    self.history["validation_uncertainty"].append(float(row["Validation_Uncertainty"]))
                    self.history["train_ssim"].append(float(row["Train_SSIM"]))
                    self.history["validation_ssim"].append(float(row["Validation_SSIM"]))

                    # Validation MAE comes from Validation_MAE, not RMSE.
                    self.history["metric_mae"].append(float(row["Validation_MAE"]))
                    self.history["metric_rmse"].append(float(row["RMSE"]))
                    self.history["metric_psnr"].append(float(row["PSNR"]))
                    self.history["metric_snr"].append(float(row["SNR"]))
                    self.history["metric_ssim"].append(float(row["Metric_SSIM"]))
                    self.history["learning_rate"].append(float(row["Learning_Rate"]))

        except (OSError, ValueError, KeyError) as error:
            print(f"Warning: could not restore training history: {error}")

    # ================================================================
    # VALIDATION HELPERS
    # ================================================================

    @staticmethod
    def _validate_tensor(tensor, name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if tensor.numel() == 0:
            raise ValueError(f"{name} is empty.")

        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{name} contains NaN or infinite values.")

    @staticmethod
    def _scalar(value, name):
        """Convert a scalar tensor or numeric value to a finite Python float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{name} must be scalar, got shape {tuple(value.shape)}."
                )
            value = value.detach().item()

        value = float(value)

        if not torch.isfinite(torch.tensor(value)).item():
            raise RuntimeError(f"{name} is non-finite: {value}")

        return value

    @staticmethod
    def _validate_batch(batch):
        """
        Expected batch:
        inputs, targets, mask, velocity_model, source,
        mask_type, geological_mode.
        """
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Dataset batch must be a tuple or list.")

        if len(batch) != 7:
            raise ValueError(
                "Acoustic training requires a seven-element batch: "
                "(inputs, targets, mask, velocity_model, source, "
                "mask_type, geological_mode). "
                f"Received {len(batch)} elements."
            )

        inputs, targets, mask, velocity, source, mask_type, geological_mode = batch

        for tensor, name in (
            (inputs, "inputs"),
            (targets, "targets"),
            (mask, "mask"),
            (velocity, "velocity_model"),
            (source, "source"),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")

        for metadata, name in (
            (mask_type, "mask_type"),
            (geological_mode, "geological_mode"),
        ):
            if not isinstance(metadata, (list, tuple)):
                raise TypeError(
                    f"{name} must be a list or tuple after DataLoader collation."
                )
            if not all(isinstance(item, str) for item in metadata):
                raise TypeError(f"All {name} values must be strings.")

    def _prepare_batch(self, batch):
        """Validate a batch and move its tensors to the selected device."""
        self._validate_batch(batch)

        inputs, targets, mask, velocity, source, mask_type, geological_mode = batch

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        velocity = velocity.to(self.device, non_blocking=True)
        source = source.to(self.device, non_blocking=True)

        for tensor, name in (
            (inputs, "inputs"),
            (targets, "targets"),
            (mask, "mask"),
            (velocity, "velocity_model"),
            (source, "source"),
        ):
            self._validate_tensor(tensor, name)

        if inputs.ndim != 5 or targets.ndim != 5:
            raise ValueError(
                "inputs and targets must be five-dimensional "
                "[B,C,D,H,W] tensors."
            )

        if inputs.shape != targets.shape:
            raise ValueError(
                f"Input and target shapes differ: {tuple(inputs.shape)} "
                f"versus {tuple(targets.shape)}."
            )

        if mask.shape != targets.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match target "
                f"shape {tuple(targets.shape)}."
            )

        if velocity.ndim != 5:
            raise ValueError(
                "velocity_model must have shape [B,C,D,H,W]."
            )

        if velocity.shape[0] != targets.shape[0]:
            raise ValueError("Velocity and target batch sizes do not match.")

        if source.ndim != 6:
            raise ValueError(
                "Acoustic source must have shape [B,1,T,D,H,W]. "
                f"Received {tuple(source.shape)}."
            )

        if source.shape[0] != targets.shape[0]:
            raise ValueError("Source and target batch sizes do not match.")

        if velocity.shape[2:] != targets.shape[2:]:
            raise ValueError(
                "Velocity spatial dimensions must match the target cube. "
                f"Velocity: {tuple(velocity.shape)}, "
                f"target: {tuple(targets.shape)}."
            )

        if source.shape[3:] != targets.shape[2:]:
            raise ValueError(
                "Source spatial dimensions must match the target cube. "
                f"Source: {tuple(source.shape)}, "
                f"target: {tuple(targets.shape)}."
            )

        if (velocity <= 0).any().item():
            raise ValueError(
                "Acoustic velocity must be strictly positive."
            )

        return (
            inputs, targets, mask, velocity, source,
            mask_type, geological_mode,
        )

    # ================================================================
    # MODEL AND ACOUSTIC LOSS
    # ================================================================

    def _forward_and_loss(self, batch):
        """
        Execute the model and composite loss.

        Required model outputs:
            reconstruction:    [B,C,D,H,W]
            pressure_wavefield:[B,1,T,D,H,W]
            log_variance:      same shape as reconstruction

        TotalLoss is expected to accept:
            prediction, target, pressure_wavefield, velocity,
            source, log_variance.
        """
        (
            inputs, targets, mask, velocity, source,
            mask_type, geological_mode,
        ) = self._prepare_batch(batch)

        outputs = self.model(inputs)

        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise RuntimeError(
                "The acoustic model must return exactly three outputs: "
                "(reconstruction, pressure_wavefield, log_variance). "
                "Do not pass the old travel-time field as pressure."
            )

        reconstruction, pressure_wavefield, log_variance = outputs

        self._validate_tensor(reconstruction, "reconstruction")
        self._validate_tensor(pressure_wavefield, "pressure_wavefield")
        self._validate_tensor(log_variance, "log_variance")

        if reconstruction.shape != targets.shape:
            raise RuntimeError(
                "Reconstruction and target shapes do not match: "
                f"{tuple(reconstruction.shape)} versus {tuple(targets.shape)}."
            )

        if log_variance.shape != reconstruction.shape:
            raise RuntimeError(
                "Log variance must match the reconstruction shape: "
                f"{tuple(log_variance.shape)} versus "
                f"{tuple(reconstruction.shape)}."
            )

        if pressure_wavefield.ndim != 6:
            raise RuntimeError(
                "pressure_wavefield must have shape [B,1,T,D,H,W]. "
                f"Received {tuple(pressure_wavefield.shape)}."
            )

        if pressure_wavefield.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Pressure-wavefield and reconstruction batch sizes differ."
            )

        if pressure_wavefield.shape[1] != 1:
            raise RuntimeError(
                "This Trainer expects a single acoustic pressure channel."
            )

        if pressure_wavefield.shape[3:] != targets.shape[2:]:
            raise RuntimeError(
                "Pressure-wavefield spatial dimensions must match the "
                "reconstruction cube. "
                f"Pressure: {tuple(pressure_wavefield.shape)}, "
                f"reconstruction: {tuple(targets.shape)}."
            )

        if pressure_wavefield.shape[2] < 3:
            raise RuntimeError(
                "The pressure wavefield needs at least three time samples "
                "for a second-order temporal finite difference."
            )

        # Acoustic inputs are mandatory. Physics is not allowed to
        # silently become zero because an input was omitted.
        losses = self.criterion(
            prediction=reconstruction,
            target=targets,
            pressure_wavefield=pressure_wavefield,
            velocity=velocity,
            source=source,
            log_variance=log_variance,
        )

        required_loss_keys = {
            "total_loss",
            "mae_loss",
            "physics_loss",
            "uncertainty_loss",
            "ssim_loss",
        }

        if not isinstance(losses, dict):
            raise TypeError("TotalLoss must return a dictionary of losses.")

        missing = required_loss_keys - set(losses.keys())
        if missing:
            raise KeyError(
                f"TotalLoss is missing required keys: {sorted(missing)}."
            )

        for name in required_loss_keys:
            self._validate_tensor(
                losses[name],
                f"losses['{name}']",
            )

        # Verify that the acoustic physics term is connected to
        # the computation graph during training.
        if self.model.training and not losses["physics_loss"].requires_grad:
            raise RuntimeError(
                "physics_loss has no gradient connection. Check that the "
                "pressure wavefield and acoustic loss remain differentiable."
            )

        return (
            targets,
            reconstruction,
            pressure_wavefield,
            log_variance,
            losses,
        )

    @staticmethod
    def _loss_averages(totals, count):
        if count == 0:
            raise RuntimeError("No batches were processed.")

        return {key: value / count for key, value in totals.items()}

    # ================================================================
    # TRAINING EPOCH
    # ================================================================

    def train_epoch(self, dataloader):
        self.model.train()

        keys = ("total", "mae", "physics", "uncertainty", "ssim")
        totals = {key: 0.0 for key in keys}

        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Training DataLoader contains no batches.")

        for batch_index, batch in enumerate(dataloader):
            self.optimizer.zero_grad(set_to_none=True)

            _, _, _, _, losses = self._forward_and_loss(batch)

            loss = losses["total_loss"]
            if not loss.requires_grad:
                raise RuntimeError(
                    "Total loss has no gradient connection to the model."
                )

            loss.backward()

            # Fail immediately on NaN/Inf gradients before clipping.
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError(
                            f"Non-finite gradient detected in parameter "
                            f"'{name}' at batch {batch_index + 1}."
                        )

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            if not torch.isfinite(gradient_norm).item():
                raise RuntimeError(
                    f"Non-finite gradient norm at batch {batch_index + 1}."
                )

            self.optimizer.step()

            mapping = {
                "total": "total_loss",
                "mae": "mae_loss",
                "physics": "physics_loss",
                "uncertainty": "uncertainty_loss",
                "ssim": "ssim_loss",
            }

            for key, loss_key in mapping.items():
                totals[key] += self._scalar(losses[loss_key], loss_key)

            if batch_index == 0 or batch_index + 1 == num_batches:
                print(
                    f"Training batch {batch_index + 1}/{num_batches}",
                    flush=True,
                )

        return self._loss_averages(totals, num_batches)

    # ================================================================
    # VALIDATION EPOCH
    # ================================================================

    def validate_epoch(self, dataloader):
        self.model.eval()

        loss_keys = ("total", "mae", "physics", "uncertainty", "ssim")
        metric_keys = (
            "metric_mae", "metric_rmse", "metric_psnr",
            "metric_snr", "metric_ssim",
        )

        totals = {key: 0.0 for key in (*loss_keys, *metric_keys)}

        num_batches = len(dataloader)
        if num_batches == 0:
            raise RuntimeError("Validation DataLoader contains no batches.")

        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                (
                    targets,
                    reconstruction,
                    pressure_wavefield,
                    log_variance,
                    losses,
                ) = self._forward_and_loss(batch)

                metrics = {
                    "metric_mae": mae(reconstruction, targets),
                    "metric_rmse": rmse(reconstruction, targets),
                    "metric_psnr": psnr(reconstruction, targets),
                    "metric_snr": snr(reconstruction, targets),
                    "metric_ssim": ssim(reconstruction, targets),
                }

                loss_mapping = {
                    "total": "total_loss",
                    "mae": "mae_loss",
                    "physics": "physics_loss",
                    "uncertainty": "uncertainty_loss",
                    "ssim": "ssim_loss",
                }

                for key, loss_key in loss_mapping.items():
                    totals[key] += self._scalar(losses[loss_key], loss_key)

                for key, value in metrics.items():
                    totals[key] += self._scalar(value, key)

                if batch_index == 0:
                    uncertainty = torch.exp(0.5 * log_variance)

                    # Visualize the pressure wavefield at its middle time.
                    middle_time = pressure_wavefield.shape[2] // 2
                    pressure_snapshot = pressure_wavefield[
                        :, :, middle_time, :, :, :
                    ]

                    self.save_validation_visualization(
                        inputs=batch[0].to(self.device),
                        targets=targets,
                        reconstruction=reconstruction,
                        pressure_snapshot=pressure_snapshot,
                        uncertainty=uncertainty,
                        epoch=self.current_epoch,
                    )

                if DEBUG_VALIDATION and batch_index == 0:
                    print("\nAcoustic validation diagnostics")
                    print("-" * 45)
                    print(
                        "Pressure-wavefield shape:",
                        tuple(pressure_wavefield.shape),
                    )
                    print(
                        "Pressure min/max:",
                        pressure_wavefield.min().item(),
                        pressure_wavefield.max().item(),
                    )
                    print(
                        "Physics loss:",
                        losses["physics_loss"].item(),
                    )

        return self._loss_averages(totals, num_batches)

    # ================================================================
    # VALIDATION VISUALIZATION
    # ================================================================

    def save_validation_visualization(
        self,
        inputs,
        targets,
        reconstruction,
        pressure_snapshot,
        uncertainty,
        epoch,
    ):
        """
        Save a central-depth image for each 3D volume.

        The pressure panel shows a snapshot at the middle time sample.
        It is not a travel-time field.
        """
        output_directory = self.experiment.training_progress
        os.makedirs(output_directory, exist_ok=True)

        def to_numpy(tensor):
            return tensor[0, 0].detach().cpu().numpy()

        inputs_np = to_numpy(inputs)
        targets_np = to_numpy(targets)
        reconstruction_np = to_numpy(reconstruction)
        pressure_np = to_numpy(pressure_snapshot)
        uncertainty_np = to_numpy(uncertainty)

        middle_depth = targets_np.shape[0] // 2

        panels = [
            (inputs_np[middle_depth], "Incomplete Input", "gray"),
            (targets_np[middle_depth], "Ground Truth", "gray"),
            (reconstruction_np[middle_depth], "Reconstruction", "gray"),
            (pressure_np[middle_depth], "Acoustic Pressure Snapshot", "viridis"),
            (uncertainty_np[middle_depth], "Predictive Uncertainty", "hot"),
            (
                abs(reconstruction_np[middle_depth] - targets_np[middle_depth]),
                "Absolute Reconstruction Error",
                "magma",
            ),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        for axis, (image, title, cmap) in zip(axes.flat, panels):
            artist = axis.imshow(image, cmap=cmap, aspect="auto")
            axis.set_title(title)
            axis.axis("off")

            if title in (
                "Acoustic Pressure Snapshot",
                "Predictive Uncertainty",
                "Absolute Reconstruction Error",
            ):
                fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)

        fig.tight_layout()

        output_file = os.path.join(
            output_directory,
            f"epoch_{epoch:03d}.png",
        )
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ================================================================
    # CONFIGURATION AND CHECKPOINTS
    # ================================================================

    @staticmethod
    def _configuration_snapshot():
        snapshot = {}

        for name in dir(config):
            if name.startswith("_"):
                continue

            value = getattr(config, name)

            if isinstance(value, (str, int, float, bool, type(None))):
                snapshot[name] = value
            elif isinstance(value, (list, tuple, dict)):
                try:
                    json.dumps(value)
                    snapshot[name] = value
                except (TypeError, ValueError):
                    pass

        return snapshot

    @staticmethod
    def _experiment_metadata():
        return {
            "dataset_mode": getattr(config, "DATASET_MODE", None),
            "experiment_name": getattr(config, "EXPERIMENT_NAME", None),
            "physics_type": "acoustic_wave_equation",
            "synthetic_num_samples": getattr(
                config, "SYNTHETIC_NUM_SAMPLES", None
            ),
            "synthetic_patch_size": getattr(
                config, "SYNTHETIC_PATCH_SIZE", None
            ),
            "validation_split": getattr(config, "VALIDATION_SPLIT", None),
        }

    def _make_checkpoint(self, epoch, loss):
        return {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_validation_loss": float(self.best_validation_loss),
            "best_epoch": int(self.best_epoch),
            "best_metrics": dict(self.best_metrics),
            "loss": float(loss),
            "experiment_metadata": self._experiment_metadata(),
            "config_snapshot": self._configuration_snapshot(),
        }

    def save_checkpoint(self, epoch, loss, periodic=False):
        checkpoint = self._make_checkpoint(epoch, loss)

        latest_file = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth",
        )
        torch.save(checkpoint, latest_file)

        if periodic:
            epoch_directory = os.path.join(
                self.checkpoint_directory,
                "epoch_checkpoints",
            )
            os.makedirs(epoch_directory, exist_ok=True)

            epoch_file = os.path.join(
                epoch_directory,
                f"epoch_{epoch:04d}.pth",
            )
            torch.save(checkpoint, epoch_file)
            print(f"Periodic checkpoint saved: {epoch_file}")

        state_file = os.path.join(
            self.checkpoint_directory,
            "training_state.json",
        )

        with open(state_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "current_epoch": int(epoch),
                    "best_epoch": int(self.best_epoch),
                    "best_validation_loss": float(self.best_validation_loss),
                    "best_metrics": self.best_metrics,
                    "dataset_mode": getattr(config, "DATASET_MODE", None),
                    "experiment_name": getattr(config, "EXPERIMENT_NAME", None),
                    "physics_type": "acoustic_wave_equation",
                },
                file,
                indent=4,
            )

    def save_best_model(self, epoch, validation_loss):
        best_file = os.path.join(
            self.checkpoint_directory,
            "best_model.pth",
        )
        torch.save(self._make_checkpoint(epoch, validation_loss), best_file)

        print("\nNEW BEST MODEL SAVED")
        print(f"Best epoch : {epoch}")
        print(f"Best loss  : {validation_loss:.6f}")

    def _validate_checkpoint_metadata(self, checkpoint):
        metadata = checkpoint.get("experiment_metadata")

        if not metadata:
            print(
                "Warning: checkpoint has no experiment metadata; "
                "strict model-state validation will still be applied."
            )
            return

        stored_physics = metadata.get("physics_type")
        if stored_physics and stored_physics != "acoustic_wave_equation":
            raise RuntimeError(
                "Physics-type mismatch: this Trainer requires an acoustic "
                f"checkpoint, but the checkpoint declares '{stored_physics}'. "
                "Do not resume an Eikonal checkpoint in this experiment."
            )

        for key in ("dataset_mode", "experiment_name"):
            stored = metadata.get(key)
            current = getattr(config, key.upper(), None)

            if stored is not None and current is not None and stored != current:
                raise RuntimeError(
                    f"Checkpoint {key} mismatch: stored={stored!r}, "
                    f"current={current!r}."
                )

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.checkpoint_directory,
                "latest_checkpoint.pth",
            )

        if not os.path.exists(checkpoint_path):
            print("No checkpoint found. Starting training from scratch.")
            return 1

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        required = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
        }
        missing = required - set(checkpoint.keys())

        if missing:
            raise RuntimeError(
                f"Checkpoint is incomplete. Missing keys: {sorted(missing)}"
            )

        self._validate_checkpoint_metadata(checkpoint)

        try:
            self.model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "Checkpoint architecture is incompatible with the current "
                "acoustic model. A previous Eikonal model may have different "
                f"output heads.\n\n{error}"
            ) from error

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                checkpoint.get("loss", float("inf")),
            )
        )
        self.best_epoch = int(
            checkpoint.get("best_epoch", checkpoint["epoch"])
        )
        self.best_metrics = dict(
            checkpoint.get("best_metrics", self.best_metrics)
        )

        start_epoch = int(checkpoint["epoch"]) + 1

        print("\nCHECKPOINT LOADED")
        print(f"Last completed epoch : {start_epoch - 1}")
        print(f"Best epoch           : {self.best_epoch}")
        print(f"Best validation loss : {self.best_validation_loss:.6f}")
        print(f"Next epoch           : {start_epoch}")

        return start_epoch

    # ================================================================
    # HISTORY RECORDING
    # ================================================================

    def _record_history(
        self,
        epoch,
        train_losses,
        validation_losses,
        learning_rate,
    ):
        values = {
            "epoch": int(epoch),
            "train_total": train_losses["total"],
            "validation_total": validation_losses["total"],
            "train_mae": train_losses["mae"],
            "validation_mae": validation_losses["mae"],
            "train_physics": train_losses["physics"],
            "validation_physics": validation_losses["physics"],
            "train_uncertainty": train_losses["uncertainty"],
            "validation_uncertainty": validation_losses["uncertainty"],
            "train_ssim": train_losses["ssim"],
            "validation_ssim": validation_losses["ssim"],
            "metric_mae": validation_losses["metric_mae"],
            "metric_rmse": validation_losses["metric_rmse"],
            "metric_psnr": validation_losses["metric_psnr"],
            "metric_snr": validation_losses["metric_snr"],
            "metric_ssim": validation_losses["metric_ssim"],
            "learning_rate": learning_rate,
        }

        for key, value in values.items():
            self.history[key].append(value)

        with open(self.log_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
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

    # ================================================================
    # FIT
    # ================================================================

    def fit(
        self,
        train_dataloader,
        validation_dataloader,
        epochs=None,
        resume=True,
    ):
        if epochs is None:
            epochs = int(config.NUM_EPOCHS)

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.patience <= 0:
            raise ValueError("PATIENCE must be greater than zero.")

        save_every = int(getattr(config, "SAVE_EVERY", 1))
        if save_every <= 0:
            raise ValueError("SAVE_EVERY must be greater than zero.")

        start_time = time.time()

        latest_file = os.path.join(
            self.checkpoint_directory,
            "latest_checkpoint.pth",
        )

        if resume and os.path.exists(latest_file):
            start_epoch = self.load_checkpoint(latest_file)
        else:
            start_epoch = 1

        if start_epoch > epochs:
            print(
                "Training is already complete according to the latest "
                "checkpoint."
            )
            self.writer.close()
            return

        try:
            for epoch in range(start_epoch, epochs + 1):
                self.current_epoch = epoch

                print("\n" + "=" * 70)
                print(f"Epoch {epoch}/{epochs}")
                print("=" * 70)

                train_losses = self.train_epoch(train_dataloader)
                validation_losses = self.validate_epoch(validation_dataloader)

                self.scheduler.step(validation_losses["total"])

                current_lr = float(self.optimizer.param_groups[0]["lr"])

                # TensorBoard: composite loss components.
                for split, results in (
                    ("Train", train_losses),
                    ("Validation", validation_losses),
                ):
                    for key in ("total", "mae", "physics", "uncertainty", "ssim"):
                        self.writer.add_scalar(
                            f"Loss/{split}_{key.capitalize()}",
                            results[key],
                            epoch,
                        )

                # TensorBoard: reconstruction metrics.
                for key in (
                    "metric_mae", "metric_rmse", "metric_psnr",
                    "metric_snr", "metric_ssim",
                ):
                    self.writer.add_scalar(
                        f"Metrics/Validation_{key.removeprefix('metric_').upper()}",
                        validation_losses[key],
                        epoch,
                    )

                self.writer.add_scalar("Learning_Rate", current_lr, epoch)

                print("\nTRAINING")
                print("-" * 40)
                for key in ("total", "mae", "physics", "uncertainty", "ssim"):
                    print(f"{key.capitalize():18s}: {train_losses[key]:.6f}")

                print("\nVALIDATION")
                print("-" * 40)
                for key in ("total", "mae", "physics", "uncertainty", "ssim"):
                    print(f"{key.capitalize():18s}: {validation_losses[key]:.6f}")

                print("\nRECONSTRUCTION METRICS")
                print("-" * 40)
                print(f"MAE  : {validation_losses['metric_mae']:.6f}")
                print(f"RMSE : {validation_losses['metric_rmse']:.6f}")
                print(f"PSNR : {validation_losses['metric_psnr']:.3f} dB")
                print(f"SNR  : {validation_losses['metric_snr']:.3f} dB")
                print(f"SSIM : {validation_losses['metric_ssim']:.6f}")
                print(f"Learning rate: {current_lr:.8f}")

                self._record_history(
                    epoch,
                    train_losses,
                    validation_losses,
                    current_lr,
                )

                validation_total = validation_losses["total"]

                if validation_total < self.best_validation_loss:
                    self.best_validation_loss = validation_total
                    self.best_epoch = epoch
                    self.best_metrics = {
                        "MAE": validation_losses["metric_mae"],
                        "RMSE": validation_losses["metric_rmse"],
                        "PSNR": validation_losses["metric_psnr"],
                        "SNR": validation_losses["metric_snr"],
                        "SSIM": validation_losses["metric_ssim"],
                    }
                    self.wait = 0
                    self.save_best_model(epoch, validation_total)
                else:
                    self.wait += 1
                    print(
                        f"Early stopping counter: {self.wait}/{self.patience}"
                    )

                self.save_checkpoint(
                    epoch=epoch,
                    loss=validation_total,
                    periodic=(epoch % save_every == 0),
                )

                if self.wait >= self.patience:
                    print("\nEarly stopping triggered.")
                    break

            training_time = time.time() - start_time

            self.save_training_summary(
                best_epoch=self.best_epoch,
                best_validation_loss=self.best_validation_loss,
                best_metrics=self.best_metrics,
                training_time=training_time,
            )

        finally:
            self.writer.close()

    # ================================================================
    # TRAINING SUMMARY
    # ================================================================

    def save_training_summary(
        self,
        best_epoch,
        best_validation_loss,
        best_metrics,
        training_time,
    ):
        report_file = os.path.join(
            self.experiment.reports,
            "training_summary.txt",
        )

        with open(report_file, "w", encoding="utf-8") as file:
            file.write(
                "Physics-Informed 3D Seismic Reconstruction "
                "Training Summary\n"
            )
            file.write("=" * 65 + "\n\n")

            file.write(
                f"Dataset Mode           : "
                f"{getattr(config, 'DATASET_MODE', 'N/A')}\n"
            )
            file.write(
                f"Experiment Name        : "
                f"{getattr(config, 'EXPERIMENT_NAME', 'N/A')}\n"
            )
            file.write("Physics Type           : Acoustic wave equation\n")
            file.write(f"Best Epoch             : {best_epoch}\n")
            file.write(
                f"Best Validation Loss   : {best_validation_loss:.6f}\n\n"
            )

            file.write(f"Best MAE               : {best_metrics['MAE']:.6f}\n")
            file.write(f"Best RMSE              : {best_metrics['RMSE']:.6f}\n")
            file.write(f"Best PSNR              : {best_metrics['PSNR']:.3f} dB\n")
            file.write(f"Best SNR               : {best_metrics['SNR']:.3f} dB\n")
            file.write(f"Best SSIM              : {best_metrics['SSIM']:.6f}\n\n")

            file.write(f"Training Time          : {training_time:.2f} seconds\n")
            file.write(
                f"Configured Epochs      : "
                f"{getattr(config, 'NUM_EPOCHS', 'N/A')}\n"
            )
            file.write(
                f"Batch Size             : "
                f"{getattr(config, 'BATCH_SIZE', 'N/A')}\n"
            )
            file.write(
                f"Validation Split       : "
                f"{getattr(config, 'VALIDATION_SPLIT', 'N/A')}\n"
            )
            file.write(
                f"Learning Rate          : "
                f"{getattr(config, 'LEARNING_RATE', 'N/A')}\n"
            )
            file.write(
                f"Weight Decay           : "
                f"{getattr(config, 'WEIGHT_DECAY', 'N/A')}\n"
            )
            file.write(
                f"Checkpoint Frequency   : "
                f"{getattr(config, 'SAVE_EVERY', 'N/A')}\n"
            )
            file.write(
                f"Early Stopping Patience: "
                f"{getattr(config, 'PATIENCE', 'N/A')}\n"
            )
