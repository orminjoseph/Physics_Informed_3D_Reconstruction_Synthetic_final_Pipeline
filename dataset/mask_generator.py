"""
=========================================================
3D Seismic Sampling Mask Generator
=========================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generates binary sampling masks for simulating incomplete
3D seismic acquisition.

Mask convention
---------------

    1.0 = observed seismic sample
    0.0 = missing seismic sample

Tensor shape
------------

    [C, D, H, W]

where:

    C = channel
    D = depth
    H = crossline
    W = inline

Supported mask types
--------------------

    random_voxels
        Randomly removes individual seismic voxels.

    missing_traces
        Removes complete seismic traces.

    missing_inlines
        Removes complete inline sections.

    missing_crosslines
        Removes complete crossline sections.

    missing_blocks
        Removes contiguous 3D regions.

    random
        Randomly selects one of the above patterns.

Author
------
Ormin Joseph
=========================================================
"""

import random

import torch


class SeismicMaskGenerator:
    """
    Generate binary sampling masks for incomplete 3D
    seismic data.

    Canonical tensor shape:

        [C, D, H, W]

    where:

        C = channel
        D = depth
        H = crossline
        W = inline

    Mask convention:

        1.0 = observed
        0.0 = missing
    """

    # =====================================================
    # SUPPORTED MASK TYPES
    # =====================================================

    SUPPORTED_MASK_TYPES = (
        "random_voxels",
        "missing_traces",
        "missing_inlines",
        "missing_crosslines",
        "missing_blocks",
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        cube_size=(64, 128, 128),
        missing_probability=0.30,
        seed=None,
    ):
        """
        Parameters
        ----------
        cube_size : tuple
            Spatial dimensions:

                (depth, crossline, inline)

        missing_probability : float
            Requested proportion of data to remove.

        seed : int or None
            Optional seed for reproducible mask generation.
        """

        # =================================================
        # VALIDATE CUBE SIZE
        # =================================================

        if (
            not isinstance(cube_size, tuple)
            or len(cube_size) != 3
        ):
            raise ValueError(
                "cube_size must be a tuple "
                "(depth, crossline, inline)."
            )

        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in cube_size
        ):
            raise ValueError(
                "All cube dimensions must be positive integers."
            )

        # =================================================
        # VALIDATE MISSING PROBABILITY
        # =================================================

        if (
            not isinstance(
                missing_probability,
                (int, float),
            )
            or isinstance(
                missing_probability,
                bool,
            )
        ):
            raise TypeError(
                "missing_probability must be a real number."
            )

        missing_probability = float(
            missing_probability
        )

        if not (
            0.0
            <= missing_probability
            < 1.0
        ):
            raise ValueError(
                "missing_probability must be "
                "between 0.0 and 1.0."
            )

        # =================================================
        # VALIDATE SEED
        # =================================================

        if seed is not None:
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
            ):
                raise TypeError(
                    "seed must be an integer or None."
                )

        # =================================================
        # STORE CONFIGURATION
        # =================================================

        self.cube_size = cube_size

        self.depth = cube_size[0]
        self.height = cube_size[1]     # crossline
        self.width = cube_size[2]      # inline

        self.missing_probability = missing_probability

        self.seed = seed

        # =================================================
        # LOCAL PYTHON RANDOM-NUMBER GENERATOR
        # =================================================

        # This generator is independent of Python's global
        # random-number state.
        self.rng = random.Random(seed)

        # =================================================
        # LOCAL PYTORCH RANDOM-NUMBER GENERATOR
        # =================================================

        # This generator is independent of PyTorch's global
        # random-number state.
        self.torch_generator = torch.Generator()

        if seed is not None:
            self.torch_generator.manual_seed(seed)

    # =====================================================
    # SET SEED
    # =====================================================

    def set_seed(self, seed):
        """
        Reset the random-number generators.

        This method allows the dataset to assign a
        deterministic sample-specific seed without creating
        a new generator object.

        Parameters
        ----------
        seed : int
            Random seed.
        """

        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise TypeError(
                "seed must be an integer."
            )

        self.seed = seed

        self.rng = random.Random(seed)

        self.torch_generator.manual_seed(seed)

    # =====================================================
    # CREATE COMPLETE MASK
    # =====================================================

    def _ones_mask(self):
        """
        Create a mask representing completely observed data.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        return torch.ones(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
        )

    # =====================================================
    # RANDOM VOXEL MASK
    # =====================================================

    def random_voxels(self):
        """
        Randomly remove individual seismic voxels.

        Each voxel is independently retained with probability
        (1 - missing_probability).

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        random_values = torch.rand(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            generator=self.torch_generator,
            dtype=torch.float32,
        )

        mask = (
            random_values
            >= self.missing_probability
        )

        return mask.to(dtype=torch.float32)

    # =====================================================
    # MISSING SEISMIC TRACES
    # =====================================================

    def missing_traces(self):
        """
        Remove complete seismic traces.

        A seismic trace extends along the depth dimension.

        Therefore, a trace is identified by its:

            (crossline, inline)

        position.

        For a selected trace:

            mask[:, :, h, w] = 0

        where:

            h = crossline index
            w = inline index

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        total_traces = (
            self.height
            * self.width
        )

        number_missing = int(
            round(
                self.missing_probability
                * total_traces
            )
        )

        number_missing = min(
            number_missing,
            total_traces,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            total_traces,
            generator=self.torch_generator,
        )[:number_missing]

        crossline_indices = (
            selected
            // self.width
        )

        inline_indices = (
            selected
            % self.width
        )

        mask[
            :,
            :,
            crossline_indices,
            inline_indices,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING INLINE SECTIONS
    # =====================================================

    def missing_inlines(self):
        """
        Remove complete inline sections.

        Under the project-wide convention:

            H = crossline
            W = inline

        Therefore, a selected inline index is a W-axis
        index and the corresponding section is:

            mask[:, :, :, w] = 0

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        number_missing = int(
            round(
                self.missing_probability
                * self.width
            )
        )

        number_missing = min(
            number_missing,
            self.width,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            self.width,
            generator=self.torch_generator,
        )[:number_missing]

        mask[
            :,
            :,
            :,
            selected,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING CROSSLINE SECTIONS
    # =====================================================

    def missing_crosslines(self):
        """
        Remove complete crossline sections.

        Under the project-wide convention:

            H = crossline
            W = inline

        Therefore, a selected crossline index is an H-axis
        index and the corresponding section is:

            mask[:, :, h, :] = 0

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        number_missing = int(
            round(
                self.missing_probability
                * self.height
            )
        )

        number_missing = min(
            number_missing,
            self.height,
        )

        if number_missing == 0:
            return mask

        selected = torch.randperm(
            self.height,
            generator=self.torch_generator,
        )[:number_missing]

        mask[
            :,
            :,
            selected,
            :,
        ] = 0.0

        return mask

    # =====================================================
    # MISSING CONTIGUOUS BLOCK
    # =====================================================

    def missing_blocks(self):
        """
        Remove a contiguous approximately cubic 3D region.

        The block dimensions are determined from the requested
        missing probability. Because dimensions must be integer
        voxel counts, the realized missing fraction may differ
        slightly from the requested probability.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        mask = self._ones_mask()

        total_voxels = (
            self.depth
            * self.height
            * self.width
        )

        target_missing = int(
            round(
                self.missing_probability
                * total_voxels
            )
        )

        if target_missing <= 0:
            return mask

        # =================================================
        # DETERMINE APPROXIMATELY CUBIC BLOCK DIMENSIONS
        # =================================================

        scale = (
            self.missing_probability
            ** (1.0 / 3.0)
        )

        block_depth = max(
            1,
            min(
                self.depth,
                int(
                    round(
                        self.depth
                        * scale
                    )
                ),
            ),
        )

        block_height = max(
            1,
            min(
                self.height,
                int(
                    round(
                        self.height
                        * scale
                    )
                ),
            ),
        )

        block_width = max(
            1,
            min(
                self.width,
                int(
                    round(
                        self.width
                        * scale
                    )
                ),
            ),
        )

        # =================================================
        # RANDOM STARTING POSITION
        # =================================================

        if self.depth == block_depth:
            start_depth = 0
        else:
            start_depth = self.rng.randint(
                0,
                self.depth - block_depth,
            )

        if self.height == block_height:
            start_height = 0
        else:
            start_height = self.rng.randint(
                0,
                self.height - block_height,
            )

        if self.width == block_width:
            start_width = 0
        else:
            start_width = self.rng.randint(
                0,
                self.width - block_width,
            )

        # =================================================
        # REMOVE CONTIGUOUS BLOCK
        # =================================================

        mask[
            :,
            start_depth:
            start_depth + block_depth,
            start_height:
            start_height + block_height,
            start_width:
            start_width + block_width,
        ] = 0.0

        return mask

    # =====================================================
    # RANDOM MASK TYPE
    # =====================================================

    def generate(
        self,
        mask_type="random",
        seed=None,
    ):
        """
        Generate a binary seismic sampling mask.

        Parameters
        ----------
        mask_type : str
            Supported values:

                random_voxels
                missing_traces
                missing_inlines
                missing_crosslines
                missing_blocks
                random

        seed : int or None
            Optional sample-specific seed.

            When supplied, both the Python and PyTorch local
            random-number generators are reset before mask
            generation.

            This is useful for deterministic lazy dataset
            generation.

        Returns
        -------
        torch.Tensor
            Shape [1, D, H, W].
        """

        # =================================================
        # VALIDATE MASK TYPE
        # =================================================

        if not isinstance(
            mask_type,
            str,
        ):
            raise TypeError(
                "mask_type must be a string."
            )

        # =================================================
        # VALIDATE OPTIONAL GENERATION SEED
        # =================================================

        if seed is not None:
            self.set_seed(seed)

        # =================================================
        # RANDOMLY SELECT MASK TYPE
        # =================================================

        if mask_type == "random":

            mask_type = self.rng.choice(
                self.SUPPORTED_MASK_TYPES
            )

        # =================================================
        # GENERATE SELECTED MASK
        # =================================================

        if mask_type == "random_voxels":

            mask = self.random_voxels()

        elif mask_type == "missing_traces":

            mask = self.missing_traces()

        elif mask_type == "missing_inlines":

            mask = self.missing_inlines()

        elif mask_type == "missing_crosslines":

            mask = self.missing_crosslines()

        elif mask_type == "missing_blocks":

            mask = self.missing_blocks()

        else:

            raise ValueError(
                "Unsupported mask_type: "
                f"{mask_type}. "
                "Supported types are: "
                f"{list(self.SUPPORTED_MASK_TYPES) + ['random']}"
            )

        # =================================================
        # VALIDATE GENERATED MASK
        # =================================================

        self.validate_mask(mask)

        return mask

    # =====================================================
    # CALCULATE REALIZED MISSING FRACTION
    # =====================================================

    @staticmethod
    def missing_fraction(mask):
        """
        Calculate the realized missing fraction of a mask.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask with:

                1.0 = observed
                0.0 = missing

        Returns
        -------
        float
            Fraction of voxels that are missing.
        """

        SeismicMaskGenerator.validate_mask(mask)

        return float(
            (mask == 0.0)
            .to(dtype=torch.float32)
            .mean()
            .item()
        )

    # =====================================================
    # MASK VALIDATION
    # =====================================================

    @staticmethod
    def validate_mask(mask):
        """
        Validate a generated seismic sampling mask.

        Parameters
        ----------
        mask : torch.Tensor
            Expected shape:

                [C, D, H, W]

        Returns
        -------
        bool
            True when the mask is valid.
        """

        # =================================================
        # TYPE VALIDATION
        # =================================================

        if not isinstance(
            mask,
            torch.Tensor,
        ):
            raise TypeError(
                "mask must be a torch.Tensor."
            )

        # =================================================
        # DIMENSION VALIDATION
        # =================================================

        if mask.ndim != 4:
            raise ValueError(
                "mask must have shape "
                "[C,D,H,W]. "
                f"Received: {tuple(mask.shape)}."
            )

        # =================================================
        # SINGLE-CHANNEL VALIDATION
        # =================================================

        if mask.shape[0] != 1:
            raise ValueError(
                "mask must contain exactly one "
                f"channel. Received {mask.shape[0]}."
            )

        # =================================================
        # NUMERICAL VALIDATION
        # =================================================

        if not torch.isfinite(mask).all():
            raise ValueError(
                "Mask contains NaN or Inf."
            )

        # =================================================
        # BINARY-VALUE VALIDATION
        # =================================================

        if not torch.all(
            (mask == 0.0)
            | (mask == 1.0)
        ):
            raise ValueError(
                "Mask must contain only binary values "
                "0.0 and 1.0."
            )

        return True

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        """
        Return a concise representation of the generator.
        """

        return (
            "SeismicMaskGenerator("
            f"cube_size={self.cube_size}, "
            f"missing_probability="
            f"{self.missing_probability}, "
            f"seed={self.seed}"
            ")"
        )