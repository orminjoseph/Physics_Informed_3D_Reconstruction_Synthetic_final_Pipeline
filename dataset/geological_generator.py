"""
=========================================================
Geological Generator
=========================================================

Physics-Informed 3D Encoder-Decoder Framework with
Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generate controlled synthetic 3D geological structures
for seismic reconstruction experiments.

Supported geological structures
--------------------------------
1. horizontal
2. dipping
3. faulted
4. folded
5. complex
6. highly_complex

Tensor convention
-----------------
Internal geological volume:

    [D, H, W]

where:

    D = depth
    H = crossline
    W = inline

Returned volume:

    [C, D, H, W]

where:

    C = seismic/geological channel

For the present framework:

    C = 1

Important methodological note
-----------------------------
This module generates controlled geological structural
models. It is NOT a full acoustic or elastic wave-equation
forward simulator.

The generated structures provide synthetic ground-truth
volumes for controlled reconstruction experiments.

Reproducibility
---------------
The generator supports an explicit seed. A local NumPy
random generator is used so that generation does not
depend on uncontrolled global random-state changes.

Author: Ormin Joseph
=========================================================
"""

import math

import numpy as np
import torch


class GeologicalGenerator:
    """
    Generate controlled synthetic 3D geological structures.

    Parameters
    ----------
    cube_size : tuple
        Cube dimensions in the form:

            (depth, crossline, inline)

        Example:

            (64, 128, 128)

    num_layers : int
        Number of alternating geological layers.

    amplitude : float
        Base absolute amplitude assigned to alternating
        geological layers.

    seed : int or None
        Optional local random seed.

    Notes
    -----
    The generator deliberately produces structural
    abstractions rather than full physical seismic
    wavefields.

    The structures contain genuine variation along both
    lateral dimensions:

        crossline (H)
        inline   (W)

    This prevents the geological volume from being merely
    a 2D structure copied unchanged across one dimension.
    """

    # =====================================================
    # SUPPORTED GEOLOGICAL MODES
    # =====================================================

    VALID_MODES = (
        "horizontal",
        "dipping",
        "faulted",
        "folded",
        "complex",
        "highly_complex",
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        cube_size=(64, 128, 128),
        num_layers=8,
        amplitude=0.2,
        seed=None,
    ):
        """
        Initialize the geological generator.

        Parameters
        ----------
        cube_size : tuple
            (depth, crossline, inline)

        num_layers : int
            Number of geological layers.

        amplitude : float
            Base layer amplitude.

        seed : int or None
            Seed for reproducible structural variation.
        """

        # -------------------------------------------------
        # Validate cube size
        # -------------------------------------------------

        if len(cube_size) != 3:
            raise ValueError(
                "cube_size must contain exactly "
                "(depth, crossline, inline)."
            )

        # -------------------------------------------------
        # Store dimensions
        # -------------------------------------------------

        self.depth = int(cube_size[0])
        self.height = int(cube_size[1])
        self.width = int(cube_size[2])

        # -------------------------------------------------
        # Store geological parameters
        # -------------------------------------------------

        self.num_layers = int(num_layers)
        self.amplitude = float(amplitude)

        # -------------------------------------------------
        # Validate dimensions
        # -------------------------------------------------

        if self.depth <= 0:
            raise ValueError(
                "depth must be positive."
            )

        if self.height <= 0:
            raise ValueError(
                "crossline dimension must be positive."
            )

        if self.width <= 0:
            raise ValueError(
                "inline dimension must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.amplitude <= 0:
            raise ValueError(
                "amplitude must be positive."
            )

        # -------------------------------------------------
        # Store seed
        # -------------------------------------------------

        self.seed = seed

        # -------------------------------------------------
        # Create a local random-number generator
        # -------------------------------------------------

        self.rng = np.random.default_rng(seed)

    # =====================================================
    # RANDOM STATE CONTROL
    # =====================================================

    def set_seed(self, seed):
        """
        Reset the local random generator.

        This method allows the dataset layer to assign a
        deterministic seed to each synthetic sample without
        modifying NumPy's global random state.
        """

        self.seed = seed

        self.rng = np.random.default_rng(seed)

    # =====================================================
    # GRID CREATION
    # =====================================================

    def _create_grid(self):
        """
        Create normalized 3D coordinate grids.

        Returns
        -------
        z : ndarray
            Normalized depth coordinate.

        y : ndarray
            Normalized crossline coordinate.

        x : ndarray
            Normalized inline coordinate.

        Shape
        -----
        All arrays have shape:

            [D, H, W]
        """

        z = np.linspace(
            0.0,
            1.0,
            self.depth,
            dtype=np.float32,
        )

        y = np.linspace(
            -1.0,
            1.0,
            self.height,
            dtype=np.float32,
        )

        x = np.linspace(
            -1.0,
            1.0,
            self.width,
            dtype=np.float32,
        )

        z, y, x = np.meshgrid(
            z,
            y,
            x,
            indexing="ij",
        )

        return z, y, x

    # =====================================================
    # LAYER THICKNESS
    # =====================================================

    def _layer_boundaries(self):
        """
        Generate approximately uniform layer boundaries.

        Returns
        -------
        boundaries : ndarray
            Integer depth boundaries.
        """

        return np.linspace(
            0,
            self.depth,
            self.num_layers + 1,
            dtype=int,
        )

    # =====================================================
    # LAYER AMPLITUDE
    # =====================================================

    def _layer_amplitudes(self):
        """
        Generate alternating layer amplitudes.

        Small seeded amplitude perturbations are introduced
        so that different seeds can produce different
        realizations while preserving the controlled
        geological character.
        """

        amplitudes = []

        for layer in range(self.num_layers):

            sign = (
                1.0
                if layer % 2 == 0
                else -1.0
            )

            # ---------------------------------------------
            # Small reproducible amplitude variation
            # ---------------------------------------------

            variation = self.rng.uniform(
                0.90,
                1.10,
            )

            amplitudes.append(
                sign
                * self.amplitude
                * variation
            )

        return amplitudes

    # =====================================================
    # HORIZONTAL LAYERS
    # =====================================================

    def generate_horizontal_layers(self):
        """
        Generate horizontal geological layers.

        The layers are horizontally stratified in depth.
        A small reproducible lateral modulation is included
        so that the volume retains genuine 3D variation
        without changing the fundamental horizontal-layer
        character.
        """

        z, y, x = self._create_grid()

        boundaries = self._layer_boundaries()
        amplitudes = self._layer_amplitudes()

        cube = np.zeros(
            (
                self.depth,
                self.height,
                self.width,
            ),
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Construct alternating layers
        # -------------------------------------------------

        for layer in range(self.num_layers):

            start = boundaries[layer]
            end = boundaries[layer + 1]

            if start >= self.depth:
                break

            end = min(
                end,
                self.depth,
            )

            if start >= end:
                continue

            cube[
                start:end,
                :,
                :,
            ] = amplitudes[layer]

        # -------------------------------------------------
        # Very small lateral structural modulation
        # -------------------------------------------------

        modulation = (
            1.0
            + 0.02
            * np.sin(
                2.0 * np.pi * x
            )
            * np.cos(
                np.pi * y
            )
        )

        cube *= modulation

        return self._to_tensor(cube)

    # =====================================================
    # DIPPING LAYERS
    # =====================================================

    def generate_dipping_layers(
        self,
        dip=None,
    ):
        """
        Generate dipping geological layers.

        Parameters
        ----------
        dip : float or None
            Normalized dip magnitude.

            If None, a reproducible value is generated.
        """

        z, y, x = self._create_grid()

        # -------------------------------------------------
        # Seed-dependent dip
        # -------------------------------------------------

        if dip is None:
            dip = self.rng.uniform(
                0.12,
                0.25,
            )

        # -------------------------------------------------
        # Seed-dependent azimuthal contribution
        # -------------------------------------------------

        crossline_dip = self.rng.uniform(
            -0.04,
            0.04,
        )

        # -------------------------------------------------
        # Depth displacement
        # -------------------------------------------------

        displacement = (
            dip * x
            + crossline_dip * y
        )

        # -------------------------------------------------
        # Normalize depth coordinate
        # -------------------------------------------------

        shifted_depth = (
            z
            - displacement
        )

        # -------------------------------------------------
        # Layer boundaries
        # -------------------------------------------------

        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_layers + 1,
        )

        amplitudes = self._layer_amplitudes()

        cube = np.zeros_like(
            shifted_depth,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Populate dipping layers
        # -------------------------------------------------

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
            )

            cube[layer_mask] = amplitudes[layer]

        return self._to_tensor(cube)

    # =====================================================
    # FAULTED LAYERS
    # =====================================================

    def generate_faulted_layers(
        self,
        dip=None,
        fault_position=None,
        throw=None,
    ):
        """
        Generate dipping layers containing a lateral fault.

        The fault is implemented as a displacement of all
        structures on one side of a fault plane.

        Parameters
        ----------
        dip : float or None
            Layer dip.

        fault_position : float or None
            Normalized inline location of the fault.

        throw : float or None
            Normalized vertical displacement.
        """

        z, y, x = self._create_grid()

        # -------------------------------------------------
        # Random structural parameters
        # -------------------------------------------------

        if dip is None:
            dip = self.rng.uniform(
                0.12,
                0.22,
            )

        if fault_position is None:
            fault_position = self.rng.uniform(
                0.40,
                0.60,
            )

        if throw is None:
            throw = self.rng.uniform(
                0.06,
                0.14,
            )

        # -------------------------------------------------
        # Base dipping structure
        # -------------------------------------------------

        displacement = dip * x

        # -------------------------------------------------
        # Crossline contribution
        # -------------------------------------------------

        displacement += (
            self.rng.uniform(
                -0.03,
                0.03,
            )
            * y
        )

        # -------------------------------------------------
        # Fault displacement
        # -------------------------------------------------

        fault_mask = (
            x > (
                2.0
                * fault_position
                - 1.0
            )
        )

        displacement = (
            displacement
            + fault_mask * throw
        )

        shifted_depth = (
            z
            - displacement
        )

        # -------------------------------------------------
        # Layer construction
        # -------------------------------------------------

        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_layers + 1,
        )

        amplitudes = self._layer_amplitudes()

        cube = np.zeros_like(
            shifted_depth,
            dtype=np.float32,
        )

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
            )

            cube[layer_mask] = amplitudes[layer]

        return self._to_tensor(cube)

    # =====================================================
    # FOLDED LAYERS
    # =====================================================

    def generate_folded_layers(
        self,
        fold_amplitude=None,
        frequency=None,
    ):
        """
        Generate sinusoidally folded geological layers.

        Parameters
        ----------
        fold_amplitude : float or None
            Normalized fold amplitude.

        frequency : float or None
            Number of fold cycles across the inline
            direction.
        """

        z, y, x = self._create_grid()

        # -------------------------------------------------
        # Random fold parameters
        # -------------------------------------------------

        if fold_amplitude is None:
            fold_amplitude = self.rng.uniform(
                0.05,
                0.12,
            )

        if frequency is None:
            frequency = self.rng.uniform(
                1.0,
                2.0,
            )

        # -------------------------------------------------
        # Main fold
        # -------------------------------------------------

        fold = (
            fold_amplitude
            * np.sin(
                2.0
                * np.pi
                * frequency
                * x
            )
        )

        # -------------------------------------------------
        # Crossline variation
        # -------------------------------------------------

        crossline_fold = (
            0.02
            * np.cos(
                np.pi
                * y
            )
        )

        shifted_depth = (
            z
            - fold
            - crossline_fold
        )

        # -------------------------------------------------
        # Layer boundaries
        # -------------------------------------------------

        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_layers + 1,
        )

        amplitudes = self._layer_amplitudes()

        cube = np.zeros_like(
            shifted_depth,
            dtype=np.float32,
        )

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
            )

            cube[layer_mask] = amplitudes[layer]

        return self._to_tensor(cube)

    # =====================================================
    # COMPLEX STRUCTURE
    # =====================================================

    def generate_complex_structure(
        self,
    ):
        """
        Generate a combined folded + faulted structure.

        The structure contains:

            - dipping/folded layers
            - lateral crossline variation
            - one major fault
        """

        z, y, x = self._create_grid()

        # -------------------------------------------------
        # Random structural parameters
        # -------------------------------------------------

        dip = self.rng.uniform(
            0.10,
            0.18,
        )

        fold_amplitude = self.rng.uniform(
            0.06,
            0.12,
        )

        frequency = self.rng.uniform(
            1.0,
            2.0,
        )

        fault_position = self.rng.uniform(
            0.42,
            0.58,
        )

        throw = self.rng.uniform(
            0.08,
            0.16,
        )

        # -------------------------------------------------
        # Combined deformation
        # -------------------------------------------------

        folded = (
            fold_amplitude
            * np.sin(
                2.0
                * np.pi
                * frequency
                * x
            )
        )

        crossline_deformation = (
            0.03
            * np.sin(
                np.pi
                * y
            )
        )

        dipping = (
            dip
            * x
        )

        fault_boundary = (
            2.0
            * fault_position
            - 1.0
        )

        fault_mask = (
            x > fault_boundary
        )

        fault_displacement = (
            fault_mask
            * throw
        )

        shifted_depth = (
            z
            - dipping
            - folded
            - crossline_deformation
            - fault_displacement
        )

        # -------------------------------------------------
        # Layer construction
        # -------------------------------------------------

        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_layers + 1,
        )

        amplitudes = self._layer_amplitudes()

        cube = np.zeros_like(
            shifted_depth,
            dtype=np.float32,
        )

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
            )

            cube[layer_mask] = amplitudes[layer]

        return self._to_tensor(cube)

    # =====================================================
    # HIGHLY COMPLEX STRUCTURE
    # =====================================================

    def generate_highly_complex_structure(
        self,
    ):
        """
        Generate the highest-complexity synthetic structure.

        Components
        ----------
        1. Dipping layers
        2. Strong folding
        3. Two faults
        4. Salt-dome deformation
        5. Salt body

        The parameters are seeded so different seeds can
        produce different geological realizations.
        """

        z, y, x = self._create_grid()

        # -------------------------------------------------
        # Random structural parameters
        # -------------------------------------------------

        dip = self.rng.uniform(
            0.14,
            0.24,
        )

        fold_amplitude = self.rng.uniform(
            0.08,
            0.15,
        )

        frequency = self.rng.uniform(
            1.0,
            2.0,
        )

        fault1_position = self.rng.uniform(
            0.25,
            0.35,
        )

        fault2_position = self.rng.uniform(
            0.62,
            0.72,
        )

        throw1 = self.rng.uniform(
            0.08,
            0.14,
        )

        throw2 = self.rng.uniform(
            0.10,
            0.18,
        )

        # -------------------------------------------------
        # Dipping structure
        # -------------------------------------------------

        dipping = (
            dip
            * x
        )

        # -------------------------------------------------
        # Strong folding
        # -------------------------------------------------

        folding = (
            fold_amplitude
            * np.sin(
                2.0
                * np.pi
                * frequency
                * x
            )
        )

        # -------------------------------------------------
        # Crossline deformation
        # -------------------------------------------------

        lateral_deformation = (
            0.04
            * np.sin(
                np.pi
                * y
            )
            * np.cos(
                np.pi
                * x
            )
        )

        # -------------------------------------------------
        # Fault 1
        # -------------------------------------------------

        fault1_boundary = (
            2.0
            * fault1_position
            - 1.0
        )

        fault1_mask = (
            x > fault1_boundary
        )

        # -------------------------------------------------
        # Fault 2
        # -------------------------------------------------

        fault2_boundary = (
            2.0
            * fault2_position
            - 1.0
        )

        fault2_mask = (
            x > fault2_boundary
        )

        # -------------------------------------------------
        # Combined displacement
        # -------------------------------------------------

        fault_displacement = (
            fault1_mask * throw1
            + fault2_mask * throw2
        )

        shifted_depth = (
            z
            - dipping
            - folding
            - lateral_deformation
            - fault_displacement
        )

        # -------------------------------------------------
        # Layer construction
        # -------------------------------------------------

        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_layers + 1,
        )

        amplitudes = self._layer_amplitudes()

        cube = np.zeros_like(
            shifted_depth,
            dtype=np.float32,
        )

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
            )

            cube[layer_mask] = amplitudes[layer]

        # =================================================
        # SALT-DOME DEFORMATION
        # =================================================

        # -------------------------------------------------
        # Random salt centre
        # -------------------------------------------------

        salt_center_x = self.rng.uniform(
            -0.15,
            0.15,
        )

        salt_center_y = self.rng.uniform(
            -0.15,
            0.15,
        )

        salt_center_z = self.rng.uniform(
            0.45,
            0.60,
        )

        salt_radius_x = self.rng.uniform(
            0.16,
            0.24,
        )

        salt_radius_y = self.rng.uniform(
            0.14,
            0.22,
        )

        salt_radius_z = self.rng.uniform(
            0.16,
            0.24,
        )

        # -------------------------------------------------
        # Ellipsoidal salt body
        # -------------------------------------------------

        salt_distance = (
            (
                (x - salt_center_x)
                / salt_radius_x
            ) ** 2
            +
            (
                (y - salt_center_y)
                / salt_radius_y
            ) ** 2
            +
            (
                (z - salt_center_z)
                / salt_radius_z
            ) ** 2
        )

        salt_mask = (
            salt_distance <= 1.0
        )

        # =================================================
        # SALT-DOME UPLIFT
        # =================================================

        horizontal_distance = np.sqrt(
            (
                x
                - salt_center_x
            ) ** 2
            +
            (
                y
                - salt_center_y
            ) ** 2
        )

        uplift_radius = (
            max(
                salt_radius_x,
                salt_radius_y,
            )
            * 1.8
        )

        uplift = np.zeros_like(
            z,
            dtype=np.float32,
        )

        uplift_mask = (
            horizontal_distance
            < uplift_radius
        )

        normalized_distance = (
            horizontal_distance[
                uplift_mask
            ]
            / uplift_radius
        )

        uplift[
            uplift_mask
        ] = (
            0.10
            * (
                1.0
                - normalized_distance
            ) ** 2
        )

        # -------------------------------------------------
        # Apply uplift to geological depth
        # -------------------------------------------------

        shifted_depth = (
            shifted_depth
            - uplift
        )

        # -------------------------------------------------
        # Rebuild layers after salt deformation
        # -------------------------------------------------

        cube.fill(0.0)

        for layer in range(self.num_layers):

            lower = boundaries[layer]
            upper = boundaries[layer + 1]

            layer_mask = (
                (shifted_depth >= lower)
                &
                (shifted_depth < upper)
                &
                (~salt_mask)
            )

            cube[layer_mask] = amplitudes[layer]

        # =================================================
        # SALT BODY AMPLITUDE
        # =================================================

        cube[
            salt_mask
        ] = 0.35

        return self._to_tensor(cube)

    # =====================================================
    # MAIN GENERATION INTERFACE
    # =====================================================

    def generate(
        self,
        mode="horizontal",
        seed=None,
    ):
        """
        Generate a geological model.

        Parameters
        ----------
        mode : str
            One of:

                horizontal
                dipping
                faulted
                folded
                complex
                highly_complex

        seed : int or None
            Optional per-generation seed.

            If supplied, the generator's local random
            state is reset before generation.

        Returns
        -------
        torch.Tensor
            Geological volume with shape:

                [1, D, H, W]
        """

        # -------------------------------------------------
        # Validate mode
        # -------------------------------------------------

        mode = str(mode).lower()

        if mode not in self.VALID_MODES:
            raise ValueError(
                "Unknown geological mode: "
                f"{mode}. Supported modes are: "
                f"{self.VALID_MODES}"
            )

        # -------------------------------------------------
        # Optional deterministic seed
        # -------------------------------------------------

        if seed is not None:
            self.set_seed(seed)

        # -------------------------------------------------
        # Dispatch to selected generator
        # -------------------------------------------------

        if mode == "horizontal":

            return self.generate_horizontal_layers()

        if mode == "dipping":

            return self.generate_dipping_layers()

        if mode == "faulted":

            return self.generate_faulted_layers()

        if mode == "folded":

            return self.generate_folded_layers()

        if mode == "complex":

            return self.generate_complex_structure()

        if mode == "highly_complex":

            return self.generate_highly_complex_structure()

        # -------------------------------------------------
        # Defensive fallback
        # -------------------------------------------------

        raise RuntimeError(
            "Geological generation dispatch failed."
        )

    # =====================================================
    # OUTPUT VALIDATION
    # =====================================================

    def _validate_output(
        self,
        tensor,
    ):
        """
        Validate the generated geological tensor.
        """

        # -------------------------------------------------
        # Expected shape
        # -------------------------------------------------

        expected_shape = (
            1,
            self.depth,
            self.height,
            self.width,
        )

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                "Generated geological model has "
                f"incorrect shape: {tuple(tensor.shape)}. "
                f"Expected: {expected_shape}."
            )

        # -------------------------------------------------
        # Floating-point validation
        # -------------------------------------------------

        if not torch.is_floating_point(tensor):
            raise RuntimeError(
                "Generated geological model must "
                "contain floating-point values."
            )

        # -------------------------------------------------
        # Finite-value validation
        # -------------------------------------------------

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                "Generated geological model contains "
                "NaN or infinite values."
            )

        return tensor

    # =====================================================
    # NUMPY → TORCH CONVERSION
    # =====================================================

    def _to_tensor(
        self,
        cube,
    ):
        """
        Convert a NumPy [D,H,W] cube to a validated
        PyTorch [1,D,H,W] tensor.
        """

        # -------------------------------------------------
        # Ensure float32
        # -------------------------------------------------

        cube = np.asarray(
            cube,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Convert to tensor
        # -------------------------------------------------

        tensor = torch.from_numpy(
            cube.copy()
        )

        # -------------------------------------------------
        # Add channel dimension
        # -------------------------------------------------

        tensor = tensor.unsqueeze(0)

        # -------------------------------------------------
        # Validate output
        # -------------------------------------------------

        return self._validate_output(
            tensor
        )

    # =====================================================
    # REPRESENTATION
    # =====================================================

    def __repr__(self):
        """
        Return a concise representation of the generator.
        """

        return (
            "GeologicalGenerator("
            f"cube_size=("
            f"{self.depth}, "
            f"{self.height}, "
            f"{self.width}"
            "), "
            f"num_layers={self.num_layers}, "
            f"amplitude={self.amplitude}, "
            f"seed={self.seed}"
            ")"
        )