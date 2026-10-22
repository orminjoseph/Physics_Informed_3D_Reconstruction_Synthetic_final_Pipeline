"""
=========================================================
Synthetic 3D Seismic Dataset
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction
in Complex Geological Settings

Purpose
-------
Generates scientifically controlled synthetic 3D seismic
reconstruction samples containing:

    1. incomplete seismic input
    2. complete seismic target
    3. sampling mask
    4. geologically conditioned velocity model
    5. mask type
    6. geological mode

The dataset is designed for:

    - supervised seismic reconstruction
    - physics-informed learning
    - Eikonal-based physical constraints
    - uncertainty-aware reconstruction
    - geological-complexity experiments
    - missing-data robustness experiments
    - reproducible PhD-level numerical experiments

Tensor convention
-----------------
Individual sample:

    [C, D, H, W]

DataLoader batch:

    [B, C, D, H, W]

where:

    C = seismic channel
    D = depth/time-sample dimension
    H = crossline/spatial dimension
    W = inline/spatial dimension

Important
---------
The velocity model is supplied to the physics-informed loss.
It is NOT predicted by the neural network.

The velocity model is conditioned on the same geological
scenario used to generate the seismic target.

Samples are generated lazily inside __getitem__.
The complete dataset is therefore NOT stored in RAM.

Reproducibility
---------------
Every sample receives a deterministic seed derived from:

    dataset_seed + sample_index * 100003

Independent deterministic seed streams are used for:

    - geological generation
    - velocity generation
    - mask generation

Each geological sample receives a NEW GeologicalGenerator
instance initialized with its deterministic geological seed.
This prevents state carried by a persistent generator from
changing the result when the same sample is requested more
than once.

Balanced training coverage
--------------------------
When geological_mode="random", geological structures are
NOT selected using uncontrolled random sampling.

Instead, a deterministic balanced cycle is used:

    horizontal
    dipping
    faulted
    folded
    complex
    highly_complex
    horizontal
    dipping
    ...

Therefore, even a 10-sample smoke-test dataset contains
all six geological structures.

Likewise, when mask_mode="random", the five missing-data
mechanisms are selected using a deterministic balanced cycle.

This design scales naturally to larger datasets.

Author: Ormin Joseph
=========================================================
"""

# =========================================================
# STANDARD LIBRARY
# =========================================================

import random


# =========================================================
# NUMERICAL / DEEP LEARNING LIBRARIES
# =========================================================

import numpy as np
import torch

from torch.utils.data import Dataset


# =========================================================
# PROJECT MODULES
# =========================================================

from dataset.geological_generator import GeologicalGenerator
from dataset.velocity_generator import VelocityGenerator
from dataset.mask_generator import SeismicMaskGenerator


# =========================================================
# DATASET CLASS
# =========================================================

class SyntheticSeismicDataset(Dataset):
    """
    Lazy synthetic 3D seismic dataset.

    Each generated sample contains:

        input_cube
        target_cube
        mask
        velocity_model
        mask_type
        geological_mode

    Tensor shape:

        [C, D, H, W]

    where C is normally 1 for the present framework.

    The dataset does not retain complete seismic samples in
    memory. Samples are generated only when requested.

    Important design principle
    --------------------------
    When geological_mode="random", the word "random" means
    deterministic balanced assignment rather than uncontrolled
    random sampling.

    This guarantees that small smoke-test datasets still
    contain all geological structures.
    """

    # =====================================================
    # VALID GEOLOGICAL MODES
    # =====================================================

    VALID_GEOLOGICAL_MODES = (
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    )

    # =====================================================
    # VALID MISSING-DATA MODES
    # =====================================================

    VALID_MASK_TYPES = (
        "random_voxels",
        "missing_traces",
        "missing_inlines",
        "missing_crosslines",
        "missing_blocks",
    )

    # =====================================================
    # SAMPLE SEED MULTIPLIER
    # =====================================================

    SAMPLE_SEED_MULTIPLIER = 100003

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        num_samples=100,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    ):
        """
        Parameters
        ----------
        num_samples : int
            Number of synthetic seismic samples.

        cube_size : tuple
            3D seismic volume dimensions:

                (depth, height, width)

        missing_probability : float
            Fraction of seismic observations removed
            by the sampling mask.

        geological_mode : str
            Geological scenario.

            Options:

                horizontal
                dipping
                faulted
                folded
                complex
                highly_complex
                random

            When "random" is selected, the six geological
            structures are assigned using a deterministic
            balanced cycle.

        mask_mode : str
            Missing-data mechanism.

            Options:

                random_voxels
                missing_traces
                missing_inlines
                missing_crosslines
                missing_blocks
                random

            When "random" is selected, the five mechanisms
            are assigned using a deterministic balanced cycle.

        seed : int
            Base seed controlling reproducibility.
        """

        # -------------------------------------------------
        # Initialize Dataset parent class
        # -------------------------------------------------

        super().__init__()

        # =================================================
        # VALIDATE BASIC TYPES BEFORE CONVERSION
        # =================================================

        if not isinstance(num_samples, int):
            raise TypeError(
                "num_samples must be an integer."
            )

        if not isinstance(seed, int):
            raise TypeError(
                "seed must be an integer."
            )

        if not isinstance(cube_size, (tuple, list)):
            raise TypeError(
                "cube_size must be a tuple or list "
                "containing (depth, height, width)."
            )

        if len(cube_size) != 3:
            raise ValueError(
                "cube_size must contain exactly "
                "(depth, height, width)."
            )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.num_samples = num_samples

        self.cube_size = tuple(
            int(dimension)
            for dimension in cube_size
        )

        self.missing_probability = float(
            missing_probability
        )

        self.geological_mode = str(
            geological_mode
        ).lower()

        self.mask_mode = str(
            mask_mode
        ).lower()

        self.seed = seed

        # =================================================
        # EXPECTED TENSOR SHAPE
        # =================================================

        self.expected_shape = (
            1,
            *self.cube_size,
        )

        # =================================================
        # VALIDATE CONFIGURATION
        # =================================================

        self._validate_configuration()

        # =================================================
        # LIGHTWEIGHT METADATA
        # =================================================
        #
        # IMPORTANT:
        #
        # No persistent GeologicalGenerator is created here.
        #
        # A persistent generator may maintain internal random
        # state. If the same dataset sample is requested twice,
        # that state could cause the target to change.
        #
        # Instead, a fresh geological generator is created
        # inside _generate_sample() using the deterministic
        # geological seed.
        # =================================================

        self.mask_types = [
            None
            for _ in range(self.num_samples)
        ]

        self.geological_modes = [
            None
            for _ in range(self.num_samples)
        ]

        # =================================================
        # PRECOMPUTE DETERMINISTIC SCHEDULE
        # =================================================
        #
        # This is intentionally lightweight.
        #
        # It contains only the geological and mask labels
        # assigned to each sample index.
        #
        # It does NOT generate seismic data.
        # =================================================

        self._geological_schedule = [
            self._scheduled_geological_mode(index)
            for index in range(self.num_samples)
        ]

        self._mask_schedule = [
            self._scheduled_mask_type(index)
            for index in range(self.num_samples)
        ]

        # -------------------------------------------------
        # Populate metadata immediately.
        # -------------------------------------------------

        self.geological_modes = list(
            self._geological_schedule
        )

        self.mask_types = list(
            self._mask_schedule
        )

        # =================================================
        # DATASET INFORMATION
        # =================================================

        print()
        print("=" * 70)
        print("PHYSICS-INFORMED SYNTHETIC 3D SEISMIC DATASET")
        print("=" * 70)

        print(
            f"Number of Samples    : {self.num_samples}"
        )

        print(
            f"Cube Size            : {self.cube_size}"
        )

        print(
            f"Missing Probability  : "
            f"{self.missing_probability:.2f}"
        )

        print(
            f"Geological Mode      : "
            f"{self.geological_mode}"
        )

        print(
            f"Mask Mode            : "
            f"{self.mask_mode}"
        )

        print(
            f"Random Seed          : "
            f"{self.seed}"
        )

        print(
            "Generation Strategy  : "
            "On-demand / lazy"
        )

        print(
            "Sampling Strategy    : "
            "Deterministic balanced scheduling"
        )

        print(
            "Velocity Coupling    : "
            "Geological-mode conditioned"
        )

        # -------------------------------------------------
        # Print geology coverage.
        # -------------------------------------------------

        if self.geological_mode == "random":

            counts = self.get_geological_coverage()

            print()
            print("Geological Coverage")
            print("-" * 70)

            for mode in self.VALID_GEOLOGICAL_MODES:

                print(
                    f"{mode:<18}: {counts[mode]}"
                )

        # -------------------------------------------------
        # Print mask coverage.
        # -------------------------------------------------

        if self.mask_mode == "random":

            counts = self.get_mask_coverage()

            print()
            print("Mask-Mechanism Coverage")
            print("-" * 70)

            for mask_type in self.VALID_MASK_TYPES:

                print(
                    f"{mask_type:<22}: {counts[mask_type]}"
                )

        print("=" * 70)
        print()

    # =====================================================
    # CONFIGURATION VALIDATION
    # =====================================================

    def _validate_configuration(self):
        """
        Validate all dataset configuration parameters.
        """

        # -------------------------------------------------
        # Number of samples
        # -------------------------------------------------

        if self.num_samples <= 0:

            raise ValueError(
                "num_samples must be greater than zero."
            )

        # -------------------------------------------------
        # Cube dimensions
        # -------------------------------------------------

        if len(self.cube_size) != 3:

            raise ValueError(
                "cube_size must contain exactly "
                "(depth, height, width)."
            )

        # -------------------------------------------------
        # Positive dimensions
        # -------------------------------------------------

        if any(
            dimension <= 0
            for dimension in self.cube_size
        ):

            raise ValueError(
                "All cube dimensions must be positive."
            )

        # -------------------------------------------------
        # Missing-data probability
        # -------------------------------------------------

        if not (
            0.0
            <= self.missing_probability
            < 1.0
        ):

            raise ValueError(
                "missing_probability must be between "
                "0.0 and less than 1.0."
            )

        # -------------------------------------------------
        # Geological mode
        # -------------------------------------------------

        valid_geological_modes = (
            self.VALID_GEOLOGICAL_MODES
            + ("random",)
        )

        if self.geological_mode not in valid_geological_modes:

            raise ValueError(
                "Unsupported geological_mode: "
                f"{self.geological_mode}. "
                "Supported modes are: "
                f"{valid_geological_modes}"
            )

        # -------------------------------------------------
        # Mask mode
        # -------------------------------------------------

        valid_mask_modes = (
            self.VALID_MASK_TYPES
            + ("random",)
        )

        if self.mask_mode not in valid_mask_modes:

            raise ValueError(
                "Unsupported mask_mode: "
                f"{self.mask_mode}. "
                "Supported modes are: "
                f"{valid_mask_modes}"
            )

    # =====================================================
    # SAMPLE SEED
    # =====================================================

    def _get_sample_seed(self, idx):
        """
        Generate a deterministic seed for a sample.

        The seed depends only on:

            dataset seed
            sample index
        """

        return (
            self.seed
            + int(idx) * self.SAMPLE_SEED_MULTIPLIER
        )

    # =====================================================
    # SET SAMPLE RANDOM SEEDS
    # =====================================================

    def _set_sample_seed(self, idx):
        """
        Establish deterministic global random states.

        These seeds are retained for compatibility with
        components that may use global random states.
        """

        sample_seed = self._get_sample_seed(idx)

        random.seed(
            sample_seed
        )

        np.random.seed(
            sample_seed % (2**32 - 1)
        )

        torch.manual_seed(
            sample_seed
        )

        return sample_seed

    # =====================================================
    # DETERMINISTIC GEOLOGICAL SCHEDULER
    # =====================================================

    def _scheduled_geological_mode(self, idx):
        """
        Return the deterministic geological mode assigned
        to a sample index.

        If geological_mode is explicitly specified, that
        geological mode is returned for every sample.

        If geological_mode="random", the six geological
        structures are cycled deterministically.
        """

        if self.geological_mode != "random":

            return self.geological_mode

        return self.VALID_GEOLOGICAL_MODES[
            int(idx)
            % len(self.VALID_GEOLOGICAL_MODES)
        ]

    # =====================================================
    # DETERMINISTIC MASK SCHEDULER
    # =====================================================

    def _scheduled_mask_type(self, idx):
        """
        Return the deterministic missing-data mechanism
        assigned to a sample index.

        If mask_mode is explicitly specified, that mechanism
        is returned for every sample.

        If mask_mode="random", the five mechanisms are
        cycled deterministically.
        """

        if self.mask_mode != "random":

            return self.mask_mode

        return self.VALID_MASK_TYPES[
            int(idx)
            % len(self.VALID_MASK_TYPES)
        ]

    # =====================================================
    # GET SAMPLE METADATA
    # =====================================================

    def get_sample_metadata(self, idx):
        """
        Return deterministic metadata for a sample without
        generating the seismic cube.
        """

        idx = int(idx)

        # -------------------------------------------------
        # Support negative indexing.
        # -------------------------------------------------

        if idx < 0:

            idx += self.num_samples

        # -------------------------------------------------
        # Validate index.
        # -------------------------------------------------

        if idx < 0 or idx >= self.num_samples:

            raise IndexError(
                f"Dataset index {idx} is out of range "
                f"for dataset of size {self.num_samples}."
            )

        # -------------------------------------------------
        # Return lightweight metadata.
        # -------------------------------------------------

        return {
            "index": idx,
            "sample_seed": self._get_sample_seed(idx),
            "geological_mode": (
                self._geological_schedule[idx]
            ),
            "mask_type": (
                self._mask_schedule[idx]
            ),
            "missing_probability": (
                self.missing_probability
            ),
        }

    # =====================================================
    # GEOLOGICAL COVERAGE
    # =====================================================

    def get_geological_coverage(self):
        """
        Return the number of samples assigned to each
        geological structure.

        No seismic data are generated.
        """

        coverage = {
            mode: 0
            for mode in self.VALID_GEOLOGICAL_MODES
        }

        for mode in self._geological_schedule:

            coverage[mode] += 1

        return coverage

    # =====================================================
    # MASK COVERAGE
    # =====================================================

    def get_mask_coverage(self):
        """
        Return the number of samples assigned to each
        missing-data mechanism.

        No seismic data are generated.
        """

        coverage = {
            mask_type: 0
            for mask_type in self.VALID_MASK_TYPES
        }

        for mask_type in self._mask_schedule:

            coverage[mask_type] += 1

        return coverage

    # =====================================================
    # SET COMPONENT RANDOM STATES
    # =====================================================

    @staticmethod
    def _set_component_seed(seed):
        """
        Set deterministic Python, NumPy and PyTorch random
        states for a particular generation component.
        """

        random.seed(
            seed
        )

        np.random.seed(
            seed % (2**32 - 1)
        )

        torch.manual_seed(
            seed
        )

    # =====================================================
    # CONVERT DATA TO FLOAT32 TENSOR
    # =====================================================

    @staticmethod
    def _to_float_tensor(data):
        """
        Convert input data to a CPU float32 tensor.

        GPU transfer is intentionally NOT performed here.

        The DataLoader/training pipeline is responsible for
        transferring batches to the selected device.
        """

        if isinstance(
            data,
            torch.Tensor,
        ):

            return data.to(
                dtype=torch.float32,
                device="cpu",
            )

        return torch.as_tensor(
            data,
            dtype=torch.float32,
            device="cpu",
        )

    # =====================================================
    # GENERAL TENSOR VALIDATION
    # =====================================================

    def _validate_tensor(
        self,
        tensor,
        name,
    ):
        """
        Validate tensor type, shape, and numerical validity.
        """

        if not isinstance(
            tensor,
            torch.Tensor,
        ):

            raise RuntimeError(
                f"{name} must be a PyTorch tensor."
            )

        if tuple(tensor.shape) != self.expected_shape:

            raise RuntimeError(
                f"{name} has an unexpected shape.\n"
                f"Expected: {self.expected_shape}\n"
                f"Received: {tuple(tensor.shape)}"
            )

        if not torch.isfinite(
            tensor
        ).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf values."
            )

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    def _validate_velocity(
        self,
        velocity,
    ):
        """
        Validate the generated velocity model.

        Requirements:

            correct tensor shape
            finite values
            strictly positive velocity
        """

        self._validate_tensor(
            velocity,
            "Generated velocity model",
        )

        if torch.any(
            velocity <= 0
        ):

            raise RuntimeError(
                "Velocity model must contain "
                "strictly positive velocities."
            )

    # =====================================================
    # MASK VALIDATION
    # =====================================================

    def _validate_mask(
        self,
        mask,
    ):
        """
        Validate the seismic sampling mask.
        """

        self._validate_tensor(
            mask,
            "Generated sampling mask",
        )

        unique_values = torch.unique(
            mask
        )

        if not torch.all(
            (unique_values == 0.0)
            |
            (unique_values == 1.0)
        ):

            raise RuntimeError(
                "Sampling mask must contain "
                "only 0.0 and 1.0 values."
            )

    # =====================================================
    # GENERATE ONE SAMPLE
    # =====================================================

    def _generate_sample(
        self,
        idx,
    ):
        """
        Generate one complete synthetic training sample.

        Generation sequence
        -------------------
        1. establish deterministic sample seed
        2. create independent component seeds
        3. obtain scheduled geological scenario
        4. create sample-specific geological generator
        5. generate complete seismic target
        6. generate corresponding velocity model
        7. obtain scheduled missing-data mechanism
        8. generate sampling mask
        9. simulate incomplete acquisition
        10. validate all outputs
        11. return sample and metadata

        Returns
        -------
        tuple

            input_cube
            target_cube
            mask
            velocity_model
            mask_type
            geological_mode
        """

        # =================================================
        # STEP 1: SAMPLE SEED
        # =================================================

        sample_seed = self._set_sample_seed(
            idx
        )

        # =================================================
        # STEP 2: INDEPENDENT SEED STREAMS
        # =================================================

        geological_seed = (
            sample_seed + 1
        )

        velocity_seed = (
            sample_seed + 2
        )

        mask_seed = (
            sample_seed + 3
        )

        # =================================================
        # STEP 3: GEOLOGICAL SCENARIO
        # =================================================

        geological_mode = (
            self._geological_schedule[idx]
        )

        # =================================================
        # STEP 4: COMPLETE SEISMIC TARGET
        # =================================================
        #
        # CRITICAL REPRODUCIBILITY CORRECTION
        #
        # A NEW GeologicalGenerator is created for every
        # sample using the deterministic geological seed.
        #
        # We intentionally do NOT keep one generator in
        # self.generator.
        #
        # This prevents internal generator state from
        # advancing between repeated calls to the same
        # dataset index.
        # =================================================

        geological_generator = GeologicalGenerator(
            cube_size=self.cube_size,
            seed=geological_seed,
        )

        target = geological_generator.generate(
            mode=geological_mode
        )

        target = self._to_float_tensor(
            target
        )

        self._validate_tensor(
            target,
            "Generated seismic target",
        )

        # =================================================
        # STEP 5: GEOLOGICALLY CONDITIONED VELOCITY MODEL
        # =================================================

        velocity_generator = VelocityGenerator(
            cube_size=self.cube_size,
            seed=velocity_seed,
        )

        velocity = (
            velocity_generator.generate(
                mode=geological_mode
            )
        )

        velocity = self._to_float_tensor(
            velocity
        )

        self._validate_velocity(
            velocity
        )

        # =================================================
        # STEP 6: MISSING-DATA MECHANISM
        # =================================================

        mask_type = (
            self._mask_schedule[idx]
        )

        # =================================================
        # STEP 7: GENERATE SAMPLING MASK
        # =================================================

        self._set_component_seed(
            mask_seed
        )

        mask_generator = SeismicMaskGenerator(
            cube_size=self.cube_size,
            missing_probability=self.missing_probability,
            seed=mask_seed,
        )

        mask = (
            mask_generator.generate(
                mask_type=mask_type
            )
        )

        mask = self._to_float_tensor(
            mask
        )

        self._validate_mask(
            mask
        )

        # =================================================
        # STEP 8: SIMULATE INCOMPLETE ACQUISITION
        # =================================================

        input_cube = (
            target * mask
        )

        self._validate_tensor(
            input_cube,
            "Generated input seismic volume",
        )

        # =================================================
        # STEP 9: STORE LIGHTWEIGHT METADATA
        # =================================================

        self.mask_types[idx] = (
            mask_type
        )

        self.geological_modes[idx] = (
            geological_mode
        )

        # =================================================
        # STEP 10: RETURN SAMPLE
        # =================================================

        return (
            input_cube,
            target,
            mask,
            velocity,
            mask_type,
            geological_mode,
        )

    # =====================================================
    # DATASET LENGTH
    # =====================================================

    def __len__(self):
        """
        Return number of samples in the dataset.
        """

        return self.num_samples

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx,
    ):
        """
        Generate one sample on demand.

        Returns
        -------

        input_cube
            Incomplete seismic volume.

        target_cube
            Complete seismic volume.

        mask
            Binary sampling mask.

        velocity_model
            Geologically conditioned velocity model.

        mask_type
            Missing-data mechanism.

        geological_mode
            Geological scenario.

        Tensor shape for all four tensors:

            [C, D, H, W]
        """

        # =================================================
        # HANDLE TENSOR INDEX
        # =================================================

        if isinstance(
            idx,
            torch.Tensor,
        ):

            if idx.numel() != 1:

                raise ValueError(
                    "Dataset index tensor must contain "
                    "exactly one element."
                )

            idx = idx.item()

        # =================================================
        # CONVERT INDEX TO INTEGER
        # =================================================

        idx = int(idx)

        # =================================================
        # SUPPORT NEGATIVE INDEXING
        # =================================================

        if idx < 0:

            idx += self.num_samples

        # =================================================
        # VALIDATE INDEX
        # =================================================

        if idx < 0 or idx >= self.num_samples:

            raise IndexError(
                f"Dataset index {idx} is out of range "
                f"for dataset of size {self.num_samples}."
            )

        # =================================================
        # GENERATE SAMPLE LAZILY
        # =================================================

        return self._generate_sample(
            idx
        )


# =========================================================
# STANDALONE DATASET TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SYNTHETIC DATASET STANDARD VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Create a 10-sample smoke-test dataset.
    # -----------------------------------------------------

    dataset = SyntheticSeismicDataset(
        num_samples=10,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        geological_mode="random",
        mask_mode="random",
        seed=42,
    )

    # =====================================================
    # CHECK GEOLOGICAL COVERAGE
    # =====================================================

    geological_coverage = (
        dataset.get_geological_coverage()
    )

    print()
    print("=" * 70)
    print("GEOLOGICAL COVERAGE TEST")
    print("=" * 70)

    for mode in dataset.VALID_GEOLOGICAL_MODES:

        print(
            f"{mode:<18}: "
            f"{geological_coverage[mode]}"
        )

        assert (
            geological_coverage[mode] > 0
        ), (
            "Every geological structure must be represented "
            "in the 10-sample smoke test."
        )

    # =====================================================
    # CHECK MASK COVERAGE
    # =====================================================

    mask_coverage = (
        dataset.get_mask_coverage()
    )

    print()
    print("=" * 70)
    print("MASK COVERAGE TEST")
    print("=" * 70)

    for mask_type in dataset.VALID_MASK_TYPES:

        print(
            f"{mask_type:<22}: "
            f"{mask_coverage[mask_type]}"
        )

        assert (
            mask_coverage[mask_type] > 0
        ), (
            "Every missing-data mechanism must be represented "
            "in the 10-sample smoke test."
        )

    # =====================================================
    # DISPLAY SAMPLE SCHEDULE
    # =====================================================

    print()
    print("=" * 70)
    print("DETERMINISTIC SAMPLE SCHEDULE")
    print("=" * 70)

    for index in range(
        len(dataset)
    ):

        metadata = (
            dataset.get_sample_metadata(index)
        )

        print(
            f"Sample {index:02d} | "
            f"Geology: "
            f"{metadata['geological_mode']:<15} | "
            f"Mask: "
            f"{metadata['mask_type']}"
        )

    # =====================================================
    # GENERATE FIRST SAMPLE
    # =====================================================

    sample_a = dataset[0]

    input_a = sample_a[0]
    target_a = sample_a[1]
    mask_a = sample_a[2]
    velocity_a = sample_a[3]
    mask_type_a = sample_a[4]
    geological_mode_a = sample_a[5]

    # -----------------------------------------------------
    # Print sample information.
    # -----------------------------------------------------

    print()
    print("Sample 0")
    print("-" * 70)

    print(
        "Input shape       :",
        tuple(input_a.shape)
    )

    print(
        "Target shape      :",
        tuple(target_a.shape)
    )

    print(
        "Mask shape        :",
        tuple(mask_a.shape)
    )

    print(
        "Velocity shape    :",
        tuple(velocity_a.shape)
    )

    print(
        "Mask type         :",
        mask_type_a
    )

    print(
        "Geological mode   :",
        geological_mode_a
    )

    print(
        "Velocity minimum  :",
        float(velocity_a.min())
    )

    print(
        "Velocity maximum  :",
        float(velocity_a.max())
    )

    # =====================================================
    # REPRODUCIBILITY TEST
    # =====================================================

    print()
    print("=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)

    sample_b = dataset[0]

    input_identical = torch.equal(
        sample_a[0],
        sample_b[0],
    )

    target_identical = torch.equal(
        sample_a[1],
        sample_b[1],
    )

    mask_identical = torch.equal(
        sample_a[2],
        sample_b[2],
    )

    velocity_identical = torch.equal(
        sample_a[3],
        sample_b[3],
    )

    mask_type_identical = (
        sample_a[4] == sample_b[4]
    )

    geology_identical = (
        sample_a[5] == sample_b[5]
    )

    print(
        "Input identical    :",
        input_identical
    )

    print(
        "Target identical   :",
        target_identical
    )

    print(
        "Mask identical     :",
        mask_identical
    )

    print(
        "Velocity identical :",
        velocity_identical
    )

    print(
        "Mask type identical:",
        mask_type_identical
    )

    print(
        "Geology identical  :",
        geology_identical
    )

    # =====================================================
    # ASSERT REPRODUCIBILITY
    # =====================================================

    assert input_identical, (
        "Repeated generation of the same sample must "
        "produce an identical input cube."
    )

    assert target_identical, (
        "Repeated generation of the same sample must "
        "produce an identical seismic target."
    )

    assert mask_identical, (
        "Repeated generation of the same sample must "
        "produce an identical sampling mask."
    )

    assert velocity_identical, (
        "Repeated generation of the same sample must "
        "produce an identical velocity model."
    )

    assert mask_type_identical, (
        "Repeated generation of the same sample must "
        "produce the same mask type."
    )

    assert geology_identical, (
        "Repeated generation of the same sample must "
        "produce the same geological mode."
    )

    # =====================================================
    # TEST SAMPLE METADATA WITHOUT GENERATION
    # =====================================================

    metadata = dataset.get_sample_metadata(5)

    assert metadata["geological_mode"] == "highly_complex"

    # =====================================================
    # FINAL STATUS
    # =====================================================

    print()
    print("=" * 70)
    print("STATUS: SYNTHETIC DATASET VALIDATION PASSED")
    print("=" * 70)