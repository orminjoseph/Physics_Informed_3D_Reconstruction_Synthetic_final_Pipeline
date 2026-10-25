
"""
====================================================================
ACOUSTIC-WAVE SYNTHETIC SEISMIC DATASET
====================================================================

Project:
Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
1. Generate a geological model to define the synthetic model geometry.
2. Generate a corresponding velocity model.
3. Simulate acoustic pressure propagation using AcousticForwardSolver.
4. Derive the 3D reconstruction target from the simulated pressure field.
5. Generate a missing-data mask and corrupt the reconstruction target.
6. Return the full pressure wavefield and source for the acoustic
   wave-equation physics loss.

Dataset output
--------------
input_cube       : [1, D, H, W]
target_cube      : [1, D, H, W]
mask              : [1, D, H, W]
velocity          : [1, D, H, W]
source            : [1, T, D, H, W]
mask_type         : string
geological_mode   : string

The DataLoader adds the batch dimension.

IMPORTANT SCIENTIFIC NOTE
-------------------------
The target is a 3D pressure snapshot extracted from the simulated
time-dependent acoustic pressure field. It is not a conventional
recorded seismic survey cube.

The acoustic physics loss uses the complete pressure wavefield,
the velocity model, and the source term.

This dataset is intended for the separate acoustic synthetic
experiment. It does not modify the original Eikonal experiment.
====================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator
from dataset.mask_generator import SeismicMaskGenerator
from dataset.acoustic_forward_solver import AcousticForwardSolver


class SyntheticSeismicDataset(Dataset):
    """
    Synthetic dataset using acoustic-wave-equation simulations.

    Each sample is generated lazily and deterministically from its
    dataset index, allowing reproducible training and validation.
    """

    def __init__(
        self,
        num_samples=100,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
        acoustic_dt=0.0005,
        acoustic_dx=10.0,
        acoustic_dy=10.0,
        acoustic_dz=10.0,
        acoustic_time_steps=128,
        source_frequency=25.0,
        source_amplitude=1.0,
        snapshot_index=None,
    ):
        """
        Parameters
        ----------
        num_samples : int
            Number of synthetic samples.

        cube_size : tuple
            Spatial dimensions (D, H, W).

        missing_probability : float
            Fraction of target samples that are masked as missing.

        geological_mode : str
            Geological generator mode.

        mask_mode : str
            Mask generator mode.

        seed : int
            Base random seed.

        acoustic_dt : float
            Acoustic simulation time step in seconds.

        acoustic_dx, acoustic_dy, acoustic_dz : float
            Spatial grid spacing in metres.

        acoustic_time_steps : int
            Number of time steps in the acoustic simulation.

        source_frequency : float
            Dominant source frequency in Hz.

        source_amplitude : float
            Amplitude of the prescribed source.

        snapshot_index : int or None
            Optional time index used to extract the target snapshot.
            If None, the time step with the greatest spatial pressure
            energy is selected.
        """

        super().__init__()

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")

        if len(cube_size) != 3 or any(int(n) < 3 for n in cube_size):
            raise ValueError(
                "cube_size must contain three spatial dimensions, "
                "each at least 3."
            )

        if not 0.0 <= missing_probability <= 1.0:
            raise ValueError(
                "missing_probability must be between 0 and 1."
            )

        self.num_samples = int(num_samples)
        self.cube_size = tuple(int(n) for n in cube_size)
        self.missing_probability = float(missing_probability)
        self.geological_mode = geological_mode
        self.mask_mode = mask_mode
        self.seed = int(seed)
        self.snapshot_index = snapshot_index

        # ----------------------------------------------------------
        # Existing generators are retained.
        # ----------------------------------------------------------
        self.geological_generator = GeologicalGenerator()
        self.velocity_generator = VelocityGenerator()

        self.mask_generator = SeismicMaskGenerator(
            cube_size=self.cube_size,
            missing_probability=self.missing_probability,
        )

        # ----------------------------------------------------------
        # Acoustic forward solver.
        #
        # The solver generates pressure and source arrays from the
        # velocity model. The pressure wavefield is used by the
        # acoustic-wave-equation loss during training.
        # ----------------------------------------------------------
        self.acoustic_solver = AcousticForwardSolver(
            dt=acoustic_dt,
            dx=acoustic_dx,
            dy=acoustic_dy,
            dz=acoustic_dz,
            n_time_steps=acoustic_time_steps,
            source_frequency=source_frequency,
            source_amplitude=source_amplitude,
        )

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _to_float_tensor(array):
        """Convert a NumPy array or tensor to float32."""
        if isinstance(array, torch.Tensor):
            return array.detach().to(dtype=torch.float32)

        return torch.as_tensor(
            np.asarray(array),
            dtype=torch.float32,
        )

    def _prepare_spatial_cube(self, array, name):
        """
        Convert a spatial array into [1, D, H, W].

        This function does not silently resize data. A shape mismatch
        is reported because resizing a velocity or pressure model
        without a defined interpolation method can alter its physics.
        """
        tensor = self._to_float_tensor(array)

        # Accept a spatial array [D,H,W].
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Accept [1,D,H,W].
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"{name} must have shape [D,H,W] or [1,D,H,W]; "
                f"received {tuple(tensor.shape)}."
            )

        expected = (1, *self.cube_size)

        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, "
                f"but expected {expected}. Check the generator's "
                "cube dimensions and acoustic solver grid."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

        return tensor.contiguous()

    @staticmethod
    def _prepare_wavefield(array, name):
        """
        Convert pressure/source to [1, T, D, H, W].

        The DataLoader will add the batch dimension, giving
        [B, 1, T, D, H, W] for the trainer.
        """
        tensor = SyntheticSeismicDataset._to_float_tensor(array)

        if tensor.ndim == 4:
            # Accept [T,D,H,W].
            tensor = tensor.unsqueeze(0)

        if tensor.ndim != 5 or tensor.shape[0] != 1:
            raise ValueError(
                f"{name} must have shape [T,D,H,W] or [1,T,D,H,W]; "
                f"received {tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

        return tensor.contiguous()

    def _generate_mask(self, generator):
        """Generate and validate a binary observation mask."""
        mask_result = generator.generate(mask_type=self.mask_mode)

        # Support generators that return either a mask alone or
        # (mask, mask_type).
        if isinstance(mask_result, tuple):
            mask_array = mask_result[0]
            actual_mask_type = (
                mask_result[1]
                if len(mask_result) > 1
                else self.mask_mode
            )
        else:
            mask_array = mask_result
            actual_mask_type = self.mask_mode

        mask = self._prepare_spatial_cube(mask_array, "mask")

        # A mask must contain only observed (1) or missing (0) values.
        mask = (mask > 0.5).to(torch.float32)

        return mask, str(actual_mask_type)

    def _select_pressure_snapshot(self, pressure_wavefield):
        """
        Extract a spatial pressure snapshot for reconstruction.

        Input:
            [1, T, D, H, W]

        Output:
            [1, D, H, W]

        If snapshot_index is not supplied, choose the time sample
        with the largest spatial mean-square pressure. This avoids
        selecting a time step solely because it is the midpoint.
        """
        number_of_steps = pressure_wavefield.shape[1]

        if self.snapshot_index is not None:
            index = int(self.snapshot_index)

            if not 0 <= index < number_of_steps:
                raise ValueError(
                    f"snapshot_index={index} is outside the valid "
                    f"range 0 to {number_of_steps - 1}."
                )
        else:
            # Calculate one energy value per time step.
            energy = pressure_wavefield[0].square().mean(
                dim=(1, 2, 3)
            )
            index = int(torch.argmax(energy).item())

        # Select one simulated spatial pressure field.
        target_cube = pressure_wavefield[:, index, :, :, :]

        if target_cube.ndim != 4:
            raise RuntimeError(
                "Pressure snapshot must have shape [1,D,H,W]."
            )

        return target_cube.contiguous()

    def _generate_sample(self, idx):
        """
        Generate one deterministic acoustic synthetic sample.
        """

        # ----------------------------------------------------------
        # 1. Set up reproducible random streams for this sample.
        # ----------------------------------------------------------
        sample_seed = self.seed + int(idx)

        np_rng = np.random.default_rng(sample_seed)

        # Seed PyTorch operations used by the generators.
        torch_generator = torch.Generator()
        torch_generator.manual_seed(sample_seed)

        # ----------------------------------------------------------
        # 2. Generate the geological model.
        #
        # The geology generator supplies the geological mode/model
        # context. Its output is NOT used as the reconstruction
        # target in this acoustic experiment.
        # ----------------------------------------------------------
        modes = [
            "horizontal",
            "dipping",
            "faulted",
            "folded",
            "complex",
            "highly_complex",
        ]

        if self.geological_mode == "random":
            geological_mode = modes[idx % len(modes)]
        else:
            geological_mode = self.geological_mode

        geological_model = self.geological_generator.generate(
            mode=geological_mode
        )

        geological_model = self._prepare_spatial_cube(
            geological_model,
            "geological_model",
        )

        # ----------------------------------------------------------
        # 3. Generate the velocity model.
        #
        # The velocity model is supplied to the acoustic solver;
        # velocity is not invented inside the physics loss.
        # ----------------------------------------------------------
        velocity = self.velocity_generator.generate()
        velocity = self._prepare_spatial_cube(
            velocity,
            "velocity",
        )

        if torch.any(velocity <= 0):
            raise ValueError(
                "Acoustic velocity must be strictly positive."
            )

        # ----------------------------------------------------------
        # 4. Run the acoustic forward simulation.
        #
        # Expected solver API:
        #     pressure, source = solver.simulate(velocity_model)
        #
        # Expected returned shapes:
        #     pressure: [1,T,D,H,W]
        #     source:   [1,T,D,H,W]
        #
        # The velocity is converted to NumPy because the forward
        # solver uses NumPy finite-difference calculations.
        # ----------------------------------------------------------
        velocity_np = velocity[0].cpu().numpy()

        pressure_result, source_result = self.acoustic_solver.simulate(
            velocity=velocity_np
        )

        pressure_wavefield = self._prepare_wavefield(
            pressure_result,
            "pressure_wavefield",
        )

        source = self._prepare_wavefield(
            source_result,
            "source",
        )

        # Verify the solver used the same spatial grid.
        expected_spatial_shape = self.cube_size
        actual_pressure_shape = tuple(pressure_wavefield.shape[2:])
        actual_source_shape = tuple(source.shape[2:])

        if actual_pressure_shape != expected_spatial_shape:
            raise ValueError(
                "Acoustic pressure spatial dimensions do not match "
                f"cube_size: {actual_pressure_shape} versus "
                f"{expected_spatial_shape}."
            )

        if actual_source_shape != expected_spatial_shape:
            raise ValueError(
                "Acoustic source spatial dimensions do not match "
                f"cube_size: {actual_source_shape} versus "
                f"{expected_spatial_shape}."
            )

        if pressure_wavefield.shape[1] != source.shape[1]:
            raise ValueError(
                "Pressure and source must have the same number "
                "of time steps."
            )

        # ----------------------------------------------------------
        # 5. Build the reconstruction target from acoustic pressure.
        #
        # Target: [1,D,H,W]
        # Pressure supplied to physics loss: [1,T,D,H,W]
        # ----------------------------------------------------------
        target = self._select_pressure_snapshot(
            pressure_wavefield
        )

        # ----------------------------------------------------------
        # 6. Generate the observation mask.
        # ----------------------------------------------------------
        mask, mask_type = self._generate_mask(
            self.mask_generator
        )

        # ----------------------------------------------------------
        # 7. Corrupt the target by removing masked samples.
        #
        # mask=1 means observed.
        # mask=0 means missing.
        # ----------------------------------------------------------
        input_cube = target * mask

        # ----------------------------------------------------------
        # 8. Final integrity checks.
        # ----------------------------------------------------------
        spatial_expected = (1, *self.cube_size)

        for name, tensor in (
            ("input_cube", input_cube),
            ("target", target),
            ("mask", mask),
            ("velocity", velocity),
        ):
            if tuple(tensor.shape) != spatial_expected:
                raise RuntimeError(
                    f"{name} has unexpected shape "
                    f"{tuple(tensor.shape)}."
                )

        return (
            input_cube,
            target,
            mask,
            velocity,
            source,
            mask_type,
            geological_mode,
        )

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_samples

        if not 0 <= idx < self.num_samples:
            raise IndexError(
                f"Dataset index {idx} is out of range."
            )

        return self._generate_sample(idx)
