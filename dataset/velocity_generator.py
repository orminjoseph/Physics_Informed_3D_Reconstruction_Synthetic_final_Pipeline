"""
=========================================================
Geologically Conditioned 3D Velocity Model Generator
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction
in Complex Geological Settings

Purpose
-------
Generate physically plausible synthetic 3D seismic velocity
models for:

    1. Synthetic training
    2. Physics-informed training
    3. Controlled validation
    4. Geological-complexity experiments
    5. Eikonal physics-loss evaluation

The velocity model is generated using the SAME resolved
geological scenario used to generate the corresponding
synthetic seismic target.

Supported geological modes
--------------------------
    horizontal
    gradient
    dipping
    folded
    faulted
    complex
    highly_complex
    random

Tensor convention
-----------------
Velocity output:

    [C, D, H, W]

where:

    C = velocity channel
    D = depth
    H = crossline
    W = inline

Training adds the batch dimension:

    [B, C, D, H, W]

Physical units
--------------
Velocity     : m/s
Travel time  : s

Eikonal relation
----------------
For isotropic acoustic propagation:

    V^2 |grad(T)|^2 - 1 = 0

or equivalently:

    |grad(T)|^2 = 1 / V^2

Important
---------
The velocity model is supplied to the physics-informed loss.

The velocity model is NOT predicted by the neural network.

The final velocity field is constrained to:

    VELOCITY_MIN <= V <= VELOCITY_MAX

Reproducibility
---------------
A generation seed may be supplied directly to generate():

    velocity = generator.generate(
        mode="faulted",
        seed=velocity_seed,
    )

This allows the synthetic dataset to assign independent,
sample-specific velocity seeds.

Author
------
Ormin Joseph
=========================================================
"""

# =========================================================
# STANDARD LIBRARY
# =========================================================

import math
import random


# =========================================================
# DEEP LEARNING LIBRARY
# =========================================================

import torch


# =========================================================
# PROJECT CONFIGURATION
# =========================================================
#
# The project configuration is the authoritative source for
# the physical velocity limits.
#
# This avoids silently maintaining different velocity ranges
# in different modules.
# =========================================================

from utils.config import (
    VELOCITY_MIN,
    VELOCITY_MAX,
)


# =========================================================
# VELOCITY GENERATOR
# =========================================================

class VelocityGenerator:
    """
    Generate geologically conditioned synthetic 3D
    seismic velocity models.

    Parameters
    ----------
    cube_size : tuple
        Velocity cube dimensions:

            (D, H, W)

        where:

            D = depth
            H = crossline
            W = inline

    min_velocity : float
        Minimum physically allowed velocity in m/s.

    max_velocity : float
        Maximum physically allowed velocity in m/s.

    seed : int or None
        Optional random seed for reproducibility.

    Notes
    -----
    The generator returns one velocity channel:

        [1, D, H, W]

    The velocity model is supplied to the physics-informed
    loss and is not predicted by the neural network.
    """

    # =====================================================
    # VALID MODES
    # =====================================================

    VALID_MODES = (
        "horizontal",
        "gradient",
        "dipping",
        "folded",
        "faulted",
        "complex",
        "highly_complex",
        "random",
    )

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
        self,
        cube_size=(64, 128, 128),
        min_velocity=VELOCITY_MIN,
        max_velocity=VELOCITY_MAX,
        seed=None,
    ):
        """
        Initialize the velocity model generator.
        """

        # -------------------------------------------------
        # Validate cube_size type.
        # -------------------------------------------------

        if (
            not isinstance(cube_size, tuple)
            or len(cube_size) != 3
        ):
            raise ValueError(
                "cube_size must be a tuple "
                "(depth, height, width)."
            )

        # -------------------------------------------------
        # Validate cube dimensions.
        # -------------------------------------------------

        if any(
            (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            )
            for value in cube_size
        ):
            raise ValueError(
                "All cube dimensions must be "
                "positive integers."
            )

        # -------------------------------------------------
        # Validate minimum velocity.
        # -------------------------------------------------

        if not isinstance(
            min_velocity,
            (int, float),
        ):
            raise TypeError(
                "min_velocity must be a real number."
            )

        # -------------------------------------------------
        # Validate maximum velocity.
        # -------------------------------------------------

        if not isinstance(
            max_velocity,
            (int, float),
        ):
            raise TypeError(
                "max_velocity must be a real number."
            )

        # -------------------------------------------------
        # Convert velocity limits to floats.
        # -------------------------------------------------

        min_velocity = float(min_velocity)
        max_velocity = float(max_velocity)

        # -------------------------------------------------
        # Validate physical velocity limits.
        # -------------------------------------------------

        if not math.isfinite(min_velocity):
            raise ValueError(
                "min_velocity must be finite."
            )

        if not math.isfinite(max_velocity):
            raise ValueError(
                "max_velocity must be finite."
            )

        if min_velocity <= 0.0:
            raise ValueError(
                "min_velocity must be greater than zero."
            )

        if max_velocity <= min_velocity:
            raise ValueError(
                "max_velocity must be greater than "
                "min_velocity."
            )

        # -------------------------------------------------
        # Store cube dimensions.
        # -------------------------------------------------

        self.cube_size = tuple(cube_size)

        self.depth = cube_size[0]
        self.height = cube_size[1]
        self.width = cube_size[2]

        # -------------------------------------------------
        # Store physical velocity limits.
        # -------------------------------------------------

        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        # -------------------------------------------------
        # Store initial seed.
        # -------------------------------------------------

        self.seed = seed

        # -------------------------------------------------
        # Create an independent random-number generator.
        #
        # This prevents velocity generation from modifying
        # Python's global random state.
        # -------------------------------------------------

        self.rng = random.Random(seed)

    # =====================================================
    # SET GENERATION SEED
    # =====================================================

    def set_seed(self, seed):
        """
        Set the random seed used by this generator.

        Parameters
        ----------
        seed : int or None
            Seed for deterministic generation.

        Returns
        -------
        int or None
            The stored seed.
        """

        if seed is not None:

            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
            ):
                raise TypeError(
                    "seed must be an integer or None."
                )

        self.seed = seed

        self.rng = random.Random(seed)

        return self.seed

    # =====================================================
    # VALIDATE MODE
    # =====================================================

    @classmethod
    def _validate_mode(cls, mode):
        """
        Validate the requested velocity-generation mode.
        """

        if not isinstance(mode, str):

            raise TypeError(
                "mode must be a string."
            )

        if mode not in cls.VALID_MODES:

            raise ValueError(
                f"Invalid velocity mode '{mode}'. "
                f"Valid modes are: {cls.VALID_MODES}"
            )

    # =====================================================
    # VALIDATE NUMBER OF LAYERS
    # =====================================================

    def _validate_number_of_layers(
        self,
        number_of_layers,
    ):
        """
        Validate the requested number of velocity layers.
        """

        if (
            not isinstance(number_of_layers, int)
            or isinstance(number_of_layers, bool)
        ):
            raise TypeError(
                "number_of_layers must be an integer."
            )

        if number_of_layers < 1:

            raise ValueError(
                "number_of_layers must be at least 1."
            )

        if number_of_layers > self.depth:

            raise ValueError(
                "number_of_layers cannot exceed "
                "the depth of the velocity cube."
            )

    # =====================================================
    # PHYSICAL BOUND ENFORCEMENT
    # =====================================================

    def _enforce_physical_bounds(
        self,
        velocity,
    ):
        """
        Constrain velocity to the configured physical range.

        Parameters
        ----------
        velocity : torch.Tensor
            Velocity tensor.

        Returns
        -------
        torch.Tensor
            Physically bounded velocity tensor.
        """

        velocity = torch.clamp(
            velocity,
            min=self.min_velocity,
            max=self.max_velocity,
        )

        return velocity.contiguous()

    # =====================================================
    # COORDINATE GENERATION
    # =====================================================

    def _coordinates(self):
        """
        Create normalized 3-D coordinates.

        Returns
        -------
        tuple
            z, y, x coordinates.

        Shapes
        ------
        z : [D, 1, 1]
        y : [1, H, 1]
        x : [1, 1, W]

        Coordinate convention
        ---------------------
        z = depth
        y = crossline
        x = inline
        """

        # -------------------------------------------------
        # Normalized depth coordinate.
        # -------------------------------------------------

        z = torch.linspace(
            0.0,
            1.0,
            self.depth,
            dtype=torch.float32,
        ).view(
            self.depth,
            1,
            1,
        )

        # -------------------------------------------------
        # Normalized crossline coordinate.
        # -------------------------------------------------

        y = torch.linspace(
            0.0,
            1.0,
            self.height,
            dtype=torch.float32,
        ).view(
            1,
            self.height,
            1,
        )

        # -------------------------------------------------
        # Normalized inline coordinate.
        # -------------------------------------------------

        x = torch.linspace(
            0.0,
            1.0,
            self.width,
            dtype=torch.float32,
        ).view(
            1,
            1,
            self.width,
        )

        return z, y, x

    # =====================================================
    # EMPTY VELOCITY CUBE
    # =====================================================

    def _empty_velocity_cube(self):
        """
        Create a zero-initialized velocity cube.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].

        Important
        ---------
        zeros are deliberately used instead of torch.empty()
        so that no uninitialized numerical values can enter
        the velocity model.
        """

        return torch.zeros(
            (
                1,
                self.depth,
                self.height,
                self.width,
            ),
            dtype=torch.float32,
        )

    # =====================================================
    # LAYER BOUNDARIES
    # =====================================================

    def _generate_layer_boundaries(
        self,
        number_of_layers,
    ):
        """
        Generate valid layer boundaries.

        Every layer receives at least one depth sample.
        """

        self._validate_number_of_layers(
            number_of_layers
        )

        # -------------------------------------------------
        # Single layer.
        # -------------------------------------------------

        if number_of_layers == 1:

            return [
                0,
                self.depth,
            ]

        # -------------------------------------------------
        # Select unique internal boundaries.
        # -------------------------------------------------

        internal_boundaries = self.rng.sample(
            range(1, self.depth),
            number_of_layers - 1,
        )

        # -------------------------------------------------
        # Sort boundaries in increasing depth order.
        # -------------------------------------------------

        internal_boundaries.sort()

        # -------------------------------------------------
        # Add top and bottom boundaries.
        # -------------------------------------------------

        return (
            [0]
            + internal_boundaries
            + [self.depth]
        )

    # =====================================================
    # HORIZONTAL LAYERED MODEL
    # =====================================================

    def generate_layered_model(
        self,
        number_of_layers=5,
    ):
        """
        Generate a horizontally layered velocity model.

        Velocity increases discretely with depth.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        boundaries = (
            self._generate_layer_boundaries(
                number_of_layers
            )
        )

        velocity = (
            self._empty_velocity_cube()
        )

        # -------------------------------------------------
        # Single-layer model.
        # -------------------------------------------------

        if number_of_layers == 1:

            velocity.fill_(
                self.min_velocity
            )

            return self._enforce_physical_bounds(
                velocity
            )

        # -------------------------------------------------
        # Velocity increment between layers.
        # -------------------------------------------------

        velocity_increment = (
            self.max_velocity
            - self.min_velocity
        ) / (
            number_of_layers - 1
        )

        # -------------------------------------------------
        # Assign velocity to each layer.
        # -------------------------------------------------

        for layer_index in range(
            number_of_layers
        ):

            top = boundaries[
                layer_index
            ]

            bottom = boundaries[
                layer_index + 1
            ]

            layer_velocity = (
                self.min_velocity
                + layer_index
                * velocity_increment
            )

            velocity[
                :,
                top:bottom,
                :,
                :,
            ] = layer_velocity

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # LINEAR GRADIENT MODEL
    # =====================================================

    def generate_gradient_model(
        self,
    ):
        """
        Generate a continuous velocity gradient.

        The velocity increases continuously with depth:

            V(z) = Vmin + (Vmax - Vmin) z

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        # -------------------------------------------------
        # Obtain normalized depth coordinate.
        # -------------------------------------------------

        z, _, _ = self._coordinates()

        # -------------------------------------------------
        # Calculate velocity at every depth.
        # -------------------------------------------------

        velocity_profile = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * z
        )

        # -------------------------------------------------
        # Expand to the full 3-D volume.
        # -------------------------------------------------

        velocity = (
            velocity_profile
            .expand(
                self.depth,
                self.height,
                self.width,
            )
            .unsqueeze(0)
        )

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # DIPPING MODEL
    # =====================================================

    def generate_dipping_model(
        self,
        number_of_layers=5,
        dip=0.20,
        crossline_dip=0.05,
    ):
        """
        Generate a 3-D dipping velocity structure.

        Parameters
        ----------
        number_of_layers : int
            Number of velocity layers.

        dip : float
            Inline-directed normalized structural dip.

        crossline_dip : float
            Crossline-directed normalized structural dip.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        self._validate_number_of_layers(
            number_of_layers
        )

        if dip < 0.0:

            raise ValueError(
                "dip must be non-negative."
            )

        if crossline_dip < 0.0:

            raise ValueError(
                "crossline_dip must be non-negative."
            )

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Generate layer boundaries.
        # -------------------------------------------------

        boundaries = (
            self._generate_layer_boundaries(
                number_of_layers
            )
        )

        normalized_boundaries = [
            boundary / self.depth
            for boundary in boundaries
        ]

        # -------------------------------------------------
        # Construct a genuinely 3-D structural coordinate.
        #
        # Both inline and crossline position contribute to
        # the interface displacement.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
            - crossline_dip * (y - 0.5)
        )

        # -------------------------------------------------
        # Start with the deepest velocity value.
        #
        # This guarantees that every voxel has a valid
        # initialized value before torch.where() operations.
        # -------------------------------------------------

        velocity = torch.full(
            (
                self.depth,
                self.height,
                self.width,
            ),
            self.max_velocity,
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Assign each geological velocity layer.
        # -------------------------------------------------

        for layer_index in range(
            number_of_layers
        ):

            top = normalized_boundaries[
                layer_index
            ]

            bottom = normalized_boundaries[
                layer_index + 1
            ]

            layer_mask = (
                (structural_z >= top)
                & (structural_z < bottom)
            )

            layer_fraction = (
                layer_index
                / max(
                    number_of_layers - 1,
                    1,
                )
            )

            layer_velocity = (
                self.min_velocity
                + layer_fraction
                * (
                    self.max_velocity
                    - self.min_velocity
                )
            )

            velocity = torch.where(
                layer_mask,
                torch.full_like(
                    velocity,
                    layer_velocity,
                ),
                velocity,
            )

        # -------------------------------------------------
        # Convert to [C,D,H,W].
        # -------------------------------------------------

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # FOLDED MODEL
    # =====================================================

    def generate_folded_model(
        self,
        fold_amplitude=0.10,
        fold_frequency=2.0,
        crossline_fold_amplitude=0.04,
    ):
        """
        Generate a folded 3-D velocity structure.

        Parameters
        ----------
        fold_amplitude : float
            Inline fold amplitude.

        fold_frequency : float
            Number of fold cycles across inline.

        crossline_fold_amplitude : float
            Crossline fold contribution.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        if fold_amplitude < 0.0:

            raise ValueError(
                "fold_amplitude must be non-negative."
            )

        if fold_frequency <= 0.0:

            raise ValueError(
                "fold_frequency must be positive."
            )

        if crossline_fold_amplitude < 0.0:

            raise ValueError(
                "crossline_fold_amplitude must be "
                "non-negative."
            )

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Inline folding.
        # -------------------------------------------------

        inline_fold = (
            fold_amplitude
            * torch.sin(
                2.0
                * math.pi
                * fold_frequency
                * x
            )
        )

        # -------------------------------------------------
        # Crossline folding.
        # -------------------------------------------------

        crossline_fold = (
            crossline_fold_amplitude
            * torch.sin(
                2.0
                * math.pi
                * y
            )
        )

        # -------------------------------------------------
        # Combined structural coordinate.
        # -------------------------------------------------

        structural_z = (
            z
            - inline_fold
            - crossline_fold
        )

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Convert structure to velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        # -------------------------------------------------
        # Expand to full 3-D volume.
        # -------------------------------------------------

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        # -------------------------------------------------
        # Add channel dimension.
        # -------------------------------------------------

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # FAULTED MODEL
    # =====================================================

    def generate_faulted_model(
        self,
        fault_position=0.50,
        fault_throw=0.12,
        dip=0.10,
        crossline_dip=0.04,
    ):
        """
        Generate a 3-D faulted velocity structure.

        Parameters
        ----------
        fault_position : float
            Normalized inline location of the fault.

        fault_throw : float
            Normalized structural displacement.

        dip : float
            Inline-directed background dip.

        crossline_dip : float
            Crossline-directed background dip.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        if not 0.0 <= fault_position <= 1.0:

            raise ValueError(
                "fault_position must lie between 0 and 1."
            )

        if fault_throw < 0.0:

            raise ValueError(
                "fault_throw must be non-negative."
            )

        if dip < 0.0:

            raise ValueError(
                "dip must be non-negative."
            )

        if crossline_dip < 0.0:

            raise ValueError(
                "crossline_dip must be non-negative."
            )

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Background dipping structure.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
            - crossline_dip * (y - 0.5)
        )

        # -------------------------------------------------
        # Fault mask.
        # -------------------------------------------------

        fault_mask = (
            x >= fault_position
        )

        # -------------------------------------------------
        # Apply structural throw.
        # -------------------------------------------------

        structural_z = torch.where(
            fault_mask,
            structural_z + fault_throw,
            structural_z,
        )

        # -------------------------------------------------
        # Keep structural coordinate bounded.
        # -------------------------------------------------

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Convert structure to velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        # -------------------------------------------------
        # Expand to full volume.
        # -------------------------------------------------

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # LATERAL HETEROGENEITY
    # =====================================================

    def _add_lateral_heterogeneity(
        self,
        velocity,
        amplitude=0.05,
        frequency_x=1.0,
        frequency_y=1.0,
    ):
        """
        Add smooth 3-D lateral heterogeneity.

        The perturbation is spatially correlated rather than
        independent voxel noise.

        Parameters
        ----------
        velocity : torch.Tensor
            Input velocity model.

        amplitude : float
            Fraction of the velocity range used for the
            perturbation.

        frequency_x : float
            Inline spatial frequency.

        frequency_y : float
            Crossline spatial frequency.

        Returns
        -------
        torch.Tensor
            Perturbed velocity model.
        """

        if amplitude < 0.0:

            raise ValueError(
                "amplitude must be non-negative."
            )

        if frequency_x <= 0.0:

            raise ValueError(
                "frequency_x must be positive."
            )

        if frequency_y <= 0.0:

            raise ValueError(
                "frequency_y must be positive."
            )

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        _, y, x = self._coordinates()

        # -------------------------------------------------
        # Smooth lateral heterogeneity.
        # -------------------------------------------------

        heterogeneity = (
            torch.sin(
                2.0
                * math.pi
                * frequency_x
                * x
            )
            * torch.cos(
                2.0
                * math.pi
                * frequency_y
                * y
            )
        )

        # -------------------------------------------------
        # Velocity range.
        # -------------------------------------------------

        velocity_range = (
            self.max_velocity
            - self.min_velocity
        )

        # -------------------------------------------------
        # Perturbation amplitude.
        # -------------------------------------------------

        perturbation = (
            amplitude
            * velocity_range
            * heterogeneity
        )

        # -------------------------------------------------
        # Broadcast perturbation over depth.
        # -------------------------------------------------

        velocity = (
            velocity
            + perturbation.unsqueeze(0)
        )

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # COMPLEX MODEL
    # =====================================================

    def generate_complex_model(
        self,
        dip=0.10,
        fold_amplitude=0.08,
        fault_1_position=0.32,
        fault_1_throw=0.08,
        fault_2_position=0.68,
        fault_2_throw=0.10,
    ):
        """
        Generate a complex 3-D geological velocity model.

        Components:

            - dipping structure
            - folding
            - multiple faults
            - smooth lateral heterogeneity

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Background dipping structure.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
            - 0.04 * (y - 0.5)
        )

        # -------------------------------------------------
        # Fold structure.
        # -------------------------------------------------

        fold = (
            fold_amplitude
            * torch.sin(
                4.0
                * math.pi
                * x
            )
        )

        crossline_fold = (
            0.03
            * torch.sin(
                2.0
                * math.pi
                * y
            )
        )

        structural_z = (
            structural_z
            - fold
            - crossline_fold
        )

        # -------------------------------------------------
        # Fault 1.
        # -------------------------------------------------

        fault_1 = (
            x >= fault_1_position
        )

        structural_z = torch.where(
            fault_1,
            structural_z + fault_1_throw,
            structural_z,
        )

        # -------------------------------------------------
        # Fault 2.
        # -------------------------------------------------

        fault_2 = (
            x >= fault_2_position
        )

        structural_z = torch.where(
            fault_2,
            structural_z - fault_2_throw,
            structural_z,
        )

        # -------------------------------------------------
        # Keep structural coordinate physical.
        # -------------------------------------------------

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Convert structural coordinate to velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        # -------------------------------------------------
        # Expand to full 3-D cube.
        # -------------------------------------------------

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        # -------------------------------------------------
        # Add smooth lateral heterogeneity.
        # -------------------------------------------------

        velocity = (
            self._add_lateral_heterogeneity(
                velocity,
                amplitude=0.04,
                frequency_x=1.0,
                frequency_y=1.0,
            )
        )

        # -------------------------------------------------
        # Final physical bound.
        # -------------------------------------------------

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # HIGHLY COMPLEX MODEL
    # =====================================================

    def generate_highly_complex_model(
        self,
        dip=0.12,
        fold_amplitude=0.12,
        fault_1_position=0.30,
        fault_1_throw=0.10,
        fault_2_position=0.65,
        fault_2_throw=0.14,
    ):
        """
        Generate a highly complex 3-D geological velocity
        model.

        Components:

            - dipping structure
            - strong folding
            - multiple faults
            - smooth lateral heterogeneity
            - salt-like high-velocity body

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].
        """

        # -------------------------------------------------
        # Coordinates.
        # -------------------------------------------------

        z, y, x = self._coordinates()

        # -------------------------------------------------
        # Background dipping structure.
        # -------------------------------------------------

        structural_z = (
            z
            - dip * (x - 0.5)
            - 0.06 * (y - 0.5)
        )

        # -------------------------------------------------
        # Strong folding.
        # -------------------------------------------------

        inline_fold = (
            fold_amplitude
            * torch.sin(
                4.0
                * math.pi
                * x
            )
        )

        crossline_fold = (
            0.05
            * torch.sin(
                2.0
                * math.pi
                * y
            )
        )

        structural_z = (
            structural_z
            - inline_fold
            - crossline_fold
        )

        # -------------------------------------------------
        # Fault 1.
        # -------------------------------------------------

        fault_1 = (
            x >= fault_1_position
        )

        structural_z = torch.where(
            fault_1,
            structural_z + fault_1_throw,
            structural_z,
        )

        # -------------------------------------------------
        # Fault 2.
        # -------------------------------------------------

        fault_2 = (
            x >= fault_2_position
        )

        structural_z = torch.where(
            fault_2,
            structural_z - fault_2_throw,
            structural_z,
        )

        # -------------------------------------------------
        # Keep structural coordinate bounded.
        # -------------------------------------------------

        structural_z = torch.clamp(
            structural_z,
            0.0,
            1.0,
        )

        # -------------------------------------------------
        # Background velocity.
        # -------------------------------------------------

        velocity = (
            self.min_velocity
            + (
                self.max_velocity
                - self.min_velocity
            )
            * structural_z
        )

        # -------------------------------------------------
        # Expand to 3-D.
        # -------------------------------------------------

        velocity = velocity.expand(
            self.depth,
            self.height,
            self.width,
        )

        velocity = velocity.unsqueeze(0)

        # -------------------------------------------------
        # Smooth lateral heterogeneity.
        # -------------------------------------------------

        velocity = (
            self._add_lateral_heterogeneity(
                velocity,
                amplitude=0.05,
                frequency_x=1.0,
                frequency_y=1.0,
            )
        )

        # =================================================
        # SALT-LIKE HIGH-VELOCITY BODY
        # =================================================

        # -------------------------------------------------
        # Random but reproducible salt-body location.
        #
        # The values are bounded so the body remains inside
        # a reasonable geological region.
        # -------------------------------------------------

        salt_center_x = self.rng.uniform(
            0.40,
            0.60,
        )

        salt_center_y = self.rng.uniform(
            0.40,
            0.60,
        )

        salt_center_z = self.rng.uniform(
            0.45,
            0.65,
        )

        # -------------------------------------------------
        # Random but reproducible body dimensions.
        # -------------------------------------------------

        sigma_x = self.rng.uniform(
            0.08,
            0.14,
        )

        sigma_y = self.rng.uniform(
            0.10,
            0.18,
        )

        sigma_z = self.rng.uniform(
            0.14,
            0.22,
        )

        # -------------------------------------------------
        # Gaussian salt-body geometry.
        # -------------------------------------------------

        salt = torch.exp(
            -(
                (
                    (x - salt_center_x)
                    / sigma_x
                ) ** 2
                +
                (
                    (y - salt_center_y)
                    / sigma_y
                ) ** 2
                +
                (
                    (z - salt_center_z)
                    / sigma_z
                ) ** 2
            )
        )

        # -------------------------------------------------
        # Salt velocity enhancement.
        # -------------------------------------------------

        velocity_range = (
            self.max_velocity
            - self.min_velocity
        )

        salt_strength = (
            0.20
            * velocity_range
        )

        velocity = (
            velocity
            + salt_strength
            * salt.unsqueeze(0)
        )

        # -------------------------------------------------
        # Final physical bound.
        # -------------------------------------------------

        return self._enforce_physical_bounds(
            velocity
        )

    # =====================================================
    # RANDOM MODEL
    # =====================================================

    def generate_random_model(
        self,
        number_of_layers=5,
    ):
        """
        Randomly select one supported velocity model.

        The selection is reproducible when a seed is supplied.
        """

        modes = (
            "horizontal",
            "gradient",
            "dipping",
            "folded",
            "faulted",
            "complex",
            "highly_complex",
        )

        mode = self.rng.choice(
            modes
        )

        return self.generate(
            mode=mode,
            number_of_layers=number_of_layers,
        )

    # =====================================================
    # MAIN GENERATOR
    # =====================================================

    def generate(
        self,
        mode="random",
        number_of_layers=5,
        seed=None,
    ):
        """
        Generate a velocity model.

        Parameters
        ----------
        mode : str
            Velocity/geological mode.

        number_of_layers : int
            Number of layers for applicable models.

        seed : int or None
            Optional generation-specific seed.

            When supplied, the generator is reseeded before
            generation. This provides deterministic,
            sample-specific velocity generation.

        Returns
        -------
        torch.Tensor
            Shape [1,D,H,W].

        Units
        -----
        m/s
        """

        # -------------------------------------------------
        # Validate mode.
        # -------------------------------------------------

        self._validate_mode(
            mode
        )

        # -------------------------------------------------
        # Apply an explicit generation seed.
        #
        # This is particularly important for the synthetic
        # dataset, where each sample has an independent
        # velocity seed.
        # -------------------------------------------------

        if seed is not None:

            self.set_seed(
                seed
            )

        # =================================================
        # RANDOM MODE
        # =================================================

        if mode == "random":

            return self.generate_random_model(
                number_of_layers=number_of_layers
            )

        # =================================================
        # HORIZONTAL MODEL
        # =================================================

        if mode == "horizontal":

            return self.generate_layered_model(
                number_of_layers=number_of_layers
            )

        # =================================================
        # GRADIENT MODEL
        # =================================================

        if mode == "gradient":

            return self.generate_gradient_model()

        # =================================================
        # DIPPING MODEL
        # =================================================

        if mode == "dipping":

            return self.generate_dipping_model(
                number_of_layers=number_of_layers
            )

        # =================================================
        # FOLDED MODEL
        # =================================================

        if mode == "folded":

            return self.generate_folded_model()

        # =================================================
        # FAULTED MODEL
        # =================================================

        if mode == "faulted":

            return self.generate_faulted_model()

        # =================================================
        # COMPLEX MODEL
        # =================================================

        if mode == "complex":

            return self.generate_complex_model()

        # =================================================
        # HIGHLY COMPLEX MODEL
        # =================================================

        if mode == "highly_complex":

            return self.generate_highly_complex_model()

        # -------------------------------------------------
        # Defensive programming.
        # -------------------------------------------------

        raise RuntimeError(
            "Unhandled velocity generation mode."
        )

    # =====================================================
    # VELOCITY VALIDATION
    # =====================================================

    @staticmethod
    def validate_velocity(
        velocity,
        min_velocity=None,
        max_velocity=None,
        expected_cube_size=None,
    ):
        """
        Validate a generated velocity model.

        Parameters
        ----------
        velocity : torch.Tensor
            Expected shape:

                [C,D,H,W]

        min_velocity : float or None
            Optional lower physical bound.

        max_velocity : float or None
            Optional upper physical bound.

        expected_cube_size : tuple or None
            Optional expected spatial dimensions:

                (D,H,W)

        Returns
        -------
        bool
            True if validation succeeds.
        """

        # -------------------------------------------------
        # Tensor type.
        # -------------------------------------------------

        if not isinstance(
            velocity,
            torch.Tensor,
        ):

            raise TypeError(
                "velocity must be a torch.Tensor."
            )

        # -------------------------------------------------
        # Tensor dimensionality.
        # -------------------------------------------------

        if velocity.ndim != 4:

            raise ValueError(
                "velocity must have shape "
                "[C,D,H,W]. "
                f"Received: {tuple(velocity.shape)}."
            )

        # -------------------------------------------------
        # One-channel requirement.
        # -------------------------------------------------

        if velocity.shape[0] != 1:

            raise ValueError(
                "velocity must contain exactly "
                "one channel. "
                f"Received {velocity.shape[0]}."
            )

        # -------------------------------------------------
        # Optional spatial-shape validation.
        # -------------------------------------------------

        if expected_cube_size is not None:

            expected_shape = (
                1,
                *tuple(expected_cube_size),
            )

            if tuple(velocity.shape) != expected_shape:

                raise ValueError(
                    "Velocity has an unexpected shape. "
                    f"Expected {expected_shape}; "
                    f"received {tuple(velocity.shape)}."
                )

        # -------------------------------------------------
        # Numerical validity.
        # -------------------------------------------------

        if not torch.isfinite(
            velocity
        ).all():

            raise ValueError(
                "Velocity model contains NaN or Inf."
            )

        # -------------------------------------------------
        # Positive velocity requirement.
        # -------------------------------------------------

        if torch.any(
            velocity <= 0.0
        ):

            raise ValueError(
                "Velocity values must be strictly "
                "greater than zero."
            )

        # -------------------------------------------------
        # Optional minimum bound.
        # -------------------------------------------------

        if min_velocity is not None:

            if torch.any(
                velocity < min_velocity
            ):

                raise ValueError(
                    "Velocity contains values below "
                    "the specified minimum velocity."
                )

        # -------------------------------------------------
        # Optional maximum bound.
        # -------------------------------------------------

        if max_velocity is not None:

            if torch.any(
                velocity > max_velocity
            ):

                raise ValueError(
                    "Velocity contains values above "
                    "the specified maximum velocity."
                )

        return True

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self):
        """
        Return a concise generator description.
        """

        return {
            "cube_size": self.cube_size,

            "min_velocity_mps":
                self.min_velocity,

            "max_velocity_mps":
                self.max_velocity,

            "seed":
                self.seed,

            "output_shape": (
                1,
                self.depth,
                self.height,
                self.width,
            ),

            "tensor_convention":
                "[C,D,H,W]",

            "spatial_convention":
                "D=depth, H=crossline, W=inline",

            "supported_modes":
                list(self.VALID_MODES),
        }


# =========================================================
# STANDALONE VALIDATION TEST
# =========================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "GEOLOGICALLY CONDITIONED 3D VELOCITY GENERATOR TEST"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Create generator.
    # -----------------------------------------------------

    generator = VelocityGenerator(
        cube_size=(64, 128, 128),
        min_velocity=VELOCITY_MIN,
        max_velocity=VELOCITY_MAX,
        seed=42,
    )

    # -----------------------------------------------------
    # Print summary.
    # -----------------------------------------------------

    print(
        "\nGenerator summary:"
    )

    print(
        generator.summary()
    )

    # =====================================================
    # TEST ALL EXPLICIT MODES
    # =====================================================

    modes = [
        "horizontal",
        "gradient",
        "dipping",
        "folded",
        "faulted",
        "complex",
        "highly_complex",
    ]

    for mode_index, mode in enumerate(
        modes
    ):

        # -------------------------------------------------
        # Give every mode a deterministic test seed.
        # -------------------------------------------------

        test_seed = (
            42
            + mode_index
        )

        # -------------------------------------------------
        # Generate velocity.
        # -------------------------------------------------

        velocity = (
            generator.generate(
                mode=mode,
                seed=test_seed,
            )
        )

        # -------------------------------------------------
        # Validate velocity.
        # -------------------------------------------------

        VelocityGenerator.validate_velocity(
            velocity,
            min_velocity=VELOCITY_MIN,
            max_velocity=VELOCITY_MAX,
            expected_cube_size=(
                64,
                128,
                128,
            ),
        )

        # -------------------------------------------------
        # Print statistics.
        # -------------------------------------------------

        print(
            f"\nMode: {mode}"
        )

        print(
            f"Shape: {tuple(velocity.shape)}"
        )

        print(
            "Minimum velocity: "
            f"{velocity.min().item():.2f} m/s"
        )

        print(
            "Maximum velocity: "
            f"{velocity.max().item():.2f} m/s"
        )

        print(
            "Mean velocity: "
            f"{velocity.mean().item():.2f} m/s"
        )

        print(
            "Standard deviation: "
            f"{velocity.std().item():.2f} m/s"
        )

        print(
            "Validation: PASSED"
        )

    # =====================================================
    # REPRODUCIBILITY TEST
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REPRODUCIBILITY TEST"
    )

    print(
        "=" * 70
    )

    velocity_a = generator.generate(
        mode="highly_complex",
        seed=12345,
    )

    velocity_b = generator.generate(
        mode="highly_complex",
        seed=12345,
    )

    reproducible = torch.equal(
        velocity_a,
        velocity_b,
    )

    print(
        f"Same-seed reproducibility: {reproducible}"
    )

    if not reproducible:

        raise RuntimeError(
            "Velocity generator failed the "
            "same-seed reproducibility test."
        )

    # -----------------------------------------------------
    # Different-seed test.
    # -----------------------------------------------------

    velocity_c = generator.generate(
        mode="highly_complex",
        seed=12346,
    )

    different_seed_changes_output = not torch.equal(
        velocity_a,
        velocity_c,
    )

    print(
        "Different-seed variation: "
        f"{different_seed_changes_output}"
    )

    if not different_seed_changes_output:

        raise RuntimeError(
            "Velocity generator did not change "
            "when a different seed was supplied."
        )

    print(
        "\nReproducibility validation: PASSED"
    )

    # =====================================================
    # FINAL TEST RESULT
    # =====================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL VELOCITY GENERATOR TESTS PASSED"
    )

    print(
        "=" * 70
    )