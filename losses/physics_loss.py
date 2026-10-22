"""
=========================================================
Physics-Informed 3D Eikonal Loss
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Physics-informed objective
--------------------------

For seismic travel time T(x, y, z):

    |grad T| = 1 / V

where:

    T = seismic travel-time field [s]
    V = P-wave velocity field [m/s]

Equivalent classical form:

    |grad T|^2 = 1 / V^2

The numerically convenient positive-velocity form used here is:

    V |grad T| = 1

Therefore:

    R_eikonal = V |grad T| - 1

and:

    L_eikonal = mean(R_eikonal^2)

The implementation does NOT introduce an arbitrary numerical
normalization factor. The physical scaling of the Eikonal
residual must remain traceable to:

    - velocity units
    - spatial sampling
    - travel-time units
    - the network travel-time representation

Source condition
----------------

When source coordinates are supplied:

    T(x_s, y_s, z_s) = 0

Therefore:

    L_source = mean(T_source^2)

Optional travel-time supervision
--------------------------------

When an independently valid travel-time target is supplied:

    L_travel_time =
        mean((T_pred - T_target)^2)

Complete physics objective
--------------------------

    L_physics =
        lambda_eikonal * L_eikonal
        +
        lambda_source * L_source
        +
        lambda_travel_time * L_travel_time

Tensor convention
-----------------

All seismic fields use:

    [B, C, D, H, W]

where:

    B = batch
    C = channel
    D = depth
    H = crossline
    W = inline

Spatial derivatives are therefore:

    dT/dz -> dimension 2
    dT/dy -> dimension 3
    dT/dx -> dimension 4

Source coordinate convention:

    [depth, crossline, inline]

Author: Ormin Joseph
=========================================================
"""

import math

import torch
import torch.nn as nn

from utils.config import (
    DX,
    DY,
    DZ,
    PHYSICS_LOSS_WEIGHTS,
)


class PhysicsLoss(nn.Module):
    """
    Physics-informed 3D Eikonal loss.

    The Eikonal residual is:

        R_eikonal = V * |grad(T)| - 1

    rather than:

        R_eikonal = V^2 * |grad(T)|^2 - 1

    This avoids unnecessarily large intermediate numerical
    values when physical velocity is represented in m/s.

    Spatial sampling distances are obtained from the project
    configuration unless explicitly supplied.

    Parameters
    ----------
    dx : float, optional
        Inline spatial sampling distance.

    dy : float, optional
        Crossline spatial sampling distance.

    dz : float, optional
        Depth spatial sampling distance.

    eikonal_weight : float, optional
        Weight applied to the Eikonal loss.

    source_weight : float, optional
        Weight applied to the source-condition loss.

    travel_time_weight : float, optional
        Weight applied to supervised travel-time loss.

    eps : float, optional
        Small positive value used when calculating the
        gradient magnitude.
    """

    def __init__(
        self,
        dx=None,
        dy=None,
        dz=None,
        eikonal_weight=None,
        source_weight=None,
        travel_time_weight=None,
        eps=1.0e-12,
    ):
        super().__init__()

        # =================================================
        # USE CENTRALIZED CONFIGURATION BY DEFAULT
        # =================================================

        if dx is None:
            dx = DX

        if dy is None:
            dy = DY

        if dz is None:
            dz = DZ

        if eikonal_weight is None:
            eikonal_weight = PHYSICS_LOSS_WEIGHTS["eikonal"]

        if source_weight is None:
            source_weight = PHYSICS_LOSS_WEIGHTS["source"]

        if travel_time_weight is None:
            travel_time_weight = (
                PHYSICS_LOSS_WEIGHTS["travel_time"]
            )

        # =================================================
        # VALIDATE SPATIAL SAMPLING
        # =================================================

        self._validate_positive_finite(
            dx,
            "dx",
        )

        self._validate_positive_finite(
            dy,
            "dy",
        )

        self._validate_positive_finite(
            dz,
            "dz",
        )

        # =================================================
        # VALIDATE LOSS WEIGHTS
        # =================================================

        self._validate_nonnegative_finite(
            eikonal_weight,
            "eikonal_weight",
        )

        self._validate_nonnegative_finite(
            source_weight,
            "source_weight",
        )

        self._validate_nonnegative_finite(
            travel_time_weight,
            "travel_time_weight",
        )

        # =================================================
        # VALIDATE NUMERICAL STABILITY CONSTANT
        # =================================================

        self._validate_positive_finite(
            eps,
            "eps",
        )

        # =================================================
        # STORE PARAMETERS
        # =================================================

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        self.eikonal_weight = float(
            eikonal_weight
        )

        self.source_weight = float(
            source_weight
        )

        self.travel_time_weight = float(
            travel_time_weight
        )

        self.eps = float(eps)

    # =====================================================
    # NUMERICAL VALIDATION HELPERS
    # =====================================================

    @staticmethod
    def _validate_positive_finite(
        value,
        name,
    ):
        """
        Validate a strictly positive finite scalar.
        """

        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if float(value) <= 0.0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    @staticmethod
    def _validate_nonnegative_finite(
        value,
        name,
    ):
        """
        Validate a non-negative finite scalar.
        """

        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{name} must be finite."
            )

        if float(value) < 0.0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    # =====================================================
    # FIELD VALIDATION
    # =====================================================

    @staticmethod
    def _validate_field(
        field,
        name,
    ):
        """
        Validate a seismic field.

        Required convention:

            [B, C, D, H, W]
        """

        if not isinstance(
            field,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if not torch.is_floating_point(field):
            raise TypeError(
                f"{name} must be a floating-point tensor."
            )

        if field.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B, C, D, H, W]. "
                f"Received: {tuple(field.shape)}."
            )

        if any(
            int(size) < 1
            for size in field.shape
        ):
            raise ValueError(
                f"{name} contains an invalid dimension."
            )

        if not torch.isfinite(field).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # =====================================================
    # SPATIAL DERIVATIVE
    # =====================================================

    @staticmethod
    def _derivative(
        field,
        spacing,
        dimension,
    ):
        """
        Calculate a first-order spatial derivative.

        Interior points use the central difference:

            df/dx =
                [f(x+h) - f(x-h)] / (2h)

        First boundary uses a forward difference:

            df/dx =
                [f(x+h) - f(x)] / h

        Final boundary uses a backward difference:

            df/dx =
                [f(x) - f(x-h)] / h

        Parameters
        ----------
        field : torch.Tensor
            Tensor with shape [B,C,D,H,W].

        spacing : float
            Physical distance between adjacent samples.

        dimension : int
            Spatial dimension:

                2 -> depth
                3 -> crossline
                4 -> inline
        """

        # -------------------------------------------------
        # Validate tensor dimensionality.
        # -------------------------------------------------

        if not isinstance(
            field,
            torch.Tensor,
        ):
            raise TypeError(
                "field must be a torch.Tensor."
            )

        if field.ndim != 5:
            raise ValueError(
                "field must have shape "
                "[B,C,D,H,W]."
            )

        # -------------------------------------------------
        # Validate spacing.
        # -------------------------------------------------

        PhysicsLoss._validate_positive_finite(
            spacing,
            "spacing",
        )

        # -------------------------------------------------
        # Validate selected spatial dimension.
        # -------------------------------------------------

        if dimension not in (2, 3, 4):
            raise ValueError(
                "dimension must be 2, 3, or 4 "
                "for depth, crossline, or inline."
            )

        size = field.shape[dimension]

        if size < 2:
            raise ValueError(
                "The selected spatial dimension must "
                "contain at least two samples."
            )

        # -------------------------------------------------
        # Allocate derivative tensor.
        #
        # empty_like preserves:
        #
        # - device
        # - dtype
        # - shape
        # -------------------------------------------------

        derivative = torch.empty_like(field)

        # =================================================
        # INTERIOR: CENTRAL DIFFERENCE
        # =================================================

        if size > 2:

            center = [slice(None)] * 5
            forward = [slice(None)] * 5
            backward = [slice(None)] * 5

            center[dimension] = slice(1, -1)
            forward[dimension] = slice(2, None)
            backward[dimension] = slice(None, -2)

            derivative[tuple(center)] = (
                field[tuple(forward)]
                -
                field[tuple(backward)]
            ) / (
                2.0 * spacing
            )

        # =================================================
        # FIRST BOUNDARY: FORWARD DIFFERENCE
        # =================================================

        first = [slice(None)] * 5
        first_forward = [slice(None)] * 5

        first[dimension] = 0
        first_forward[dimension] = 1

        derivative[tuple(first)] = (
            field[tuple(first_forward)]
            -
            field[tuple(first)]
        ) / spacing

        # =================================================
        # FINAL BOUNDARY: BACKWARD DIFFERENCE
        # =================================================

        last = [slice(None)] * 5
        last_backward = [slice(None)] * 5

        last[dimension] = -1
        last_backward[dimension] = -2

        derivative[tuple(last)] = (
            field[tuple(last)]
            -
            field[tuple(last_backward)]
        ) / spacing

        # -------------------------------------------------
        # Validate derivative.
        # -------------------------------------------------

        if not torch.isfinite(derivative).all():
            raise ValueError(
                "Spatial derivative contains NaN "
                "or infinite values."
            )

        return derivative

    # =====================================================
    # TRAVEL-TIME GRADIENT
    # =====================================================

    def travel_time_gradient(
        self,
        travel_time,
    ):
        """
        Calculate the complete 3D spatial gradient.

        Returns
        -------

        dT_dz : torch.Tensor
            Depth derivative.

        dT_dy : torch.Tensor
            Crossline derivative.

        dT_dx : torch.Tensor
            Inline derivative.

        gradient_squared : torch.Tensor
            Squared gradient magnitude.

        gradient_magnitude : torch.Tensor
            Gradient magnitude.
        """

        # -------------------------------------------------
        # Validate travel-time field.
        # -------------------------------------------------

        self._validate_field(
            travel_time,
            "travel_time",
        )

        # =================================================
        # DEPTH DERIVATIVE
        # =================================================

        dT_dz = self._derivative(
            field=travel_time,
            spacing=self.dz,
            dimension=2,
        )

        # =================================================
        # CROSSLINE DERIVATIVE
        # =================================================

        dT_dy = self._derivative(
            field=travel_time,
            spacing=self.dy,
            dimension=3,
        )

        # =================================================
        # INLINE DERIVATIVE
        # =================================================

        dT_dx = self._derivative(
            field=travel_time,
            spacing=self.dx,
            dimension=4,
        )

        # =================================================
        # SQUARED GRADIENT MAGNITUDE
        # =================================================

        gradient_squared = (
            dT_dx.pow(2)
            +
            dT_dy.pow(2)
            +
            dT_dz.pow(2)
        )

        # -------------------------------------------------
        # Protect against tiny negative floating-point
        # round-off values before square root.
        # -------------------------------------------------

        gradient_squared = torch.clamp(
            gradient_squared,
            min=0.0,
        )

        # =================================================
        # GRADIENT MAGNITUDE
        # =================================================

        gradient_magnitude = torch.sqrt(
            gradient_squared
            +
            self.eps
        )

        # -------------------------------------------------
        # Validate outputs.
        # -------------------------------------------------

        if not torch.isfinite(
            gradient_squared
        ).all():
            raise ValueError(
                "gradient_squared contains NaN "
                "or infinite values."
            )

        if not torch.isfinite(
            gradient_magnitude
        ).all():
            raise ValueError(
                "gradient_magnitude contains NaN "
                "or infinite values."
            )

        return (
            dT_dz,
            dT_dy,
            dT_dx,
            gradient_squared,
            gradient_magnitude,
        )

    # =====================================================
    # EIKONAL RESIDUAL
    # =====================================================

    def eikonal_residual(
        self,
        travel_time,
        velocity,
    ):
        """
        Calculate the dimensionless stabilized
        Eikonal residual.

        Governing equation:

            |grad T| = 1 / V

        Equivalent form:

            V |grad T| = 1

        Residual:

            R_eikonal =
                V |grad T| - 1

        The velocity supplied to this function must be in
        physical units compatible with the spatial spacing
        and travel-time units.

        For example:

            velocity -> m/s
            spacing  -> m
            travel time -> s

        gives:

            V * |grad T|

        as a dimensionless quantity.
        """

        # =================================================
        # VALIDATE INPUTS
        # =================================================

        self._validate_field(
            travel_time,
            "travel_time",
        )

        self._validate_field(
            velocity,
            "velocity",
        )

        # =================================================
        # VALIDATE SHAPES
        # =================================================

        if travel_time.shape != velocity.shape:
            raise ValueError(
                "travel_time and velocity must have "
                "identical shapes. "
                f"Travel time: "
                f"{tuple(travel_time.shape)}, "
                f"Velocity: "
                f"{tuple(velocity.shape)}."
            )

        # =================================================
        # VALIDATE VELOCITY
        # =================================================

        if torch.any(velocity <= 0.0):
            raise ValueError(
                "Velocity must contain strictly positive "
                "P-wave velocity values."
            )

        # =================================================
        # COMPUTE TRAVEL-TIME GRADIENT
        # =================================================

        (
            _dT_dz,
            _dT_dy,
            _dT_dx,
            _gradient_squared,
            gradient_magnitude,
        ) = self.travel_time_gradient(
            travel_time
        )

        # =================================================
        # EIKONAL RESIDUAL
        # =================================================
        #
        # IMPORTANT:
        #
        # No arbitrary scaling factor is introduced here.
        #
        # The residual must remain physically interpretable.
        # =================================================

        residual = (
            velocity
            *
            gradient_magnitude
            -
            1.0
        )

        # =================================================
        # VALIDATE RESIDUAL
        # =================================================

        if not torch.isfinite(
            residual
        ).all():
            raise ValueError(
                "Eikonal residual contains NaN "
                "or infinite values."
            )

        return residual

    # =====================================================
    # EIKONAL LOSS
    # =====================================================

    def eikonal_loss(
        self,
        travel_time,
        velocity,
    ):
        """
        Compute the mean squared Eikonal residual.
        """

        residual = self.eikonal_residual(
            travel_time=travel_time,
            velocity=velocity,
        )

        loss = residual.pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Eikonal loss contains NaN "
                "or infinite values."
            )

        return loss

    # =====================================================
    # SOURCE CONDITION LOSS
    # =====================================================

    def source_condition_loss(
        self,
        travel_time,
        source_indices,
    ):
        """
        Enforce the source condition:

            T(x_s, y_s, z_s) = 0

        Source indices must use:

            [depth, crossline, inline]

        with shape:

            [B, 3]
        """

        # -------------------------------------------------
        # Validate travel-time field.
        # -------------------------------------------------

        self._validate_field(
            travel_time,
            "travel_time",
        )

        # -------------------------------------------------
        # If no source coordinates are supplied, there is
        # no source-condition contribution.
        # -------------------------------------------------

        if source_indices is None:
            return travel_time.new_zeros(())

        # -------------------------------------------------
        # Validate source tensor.
        # -------------------------------------------------

        if not isinstance(
            source_indices,
            torch.Tensor,
        ):
            raise TypeError(
                "source_indices must be a torch.Tensor."
            )

        if source_indices.ndim != 2:
            raise ValueError(
                "source_indices must have shape [B,3]. "
                f"Received: "
                f"{tuple(source_indices.shape)}."
            )

        if source_indices.shape[1] != 3:
            raise ValueError(
                "source_indices must contain exactly "
                "three coordinates in the order "
                "[depth, crossline, inline]."
            )

        # -------------------------------------------------
        # Validate batch size.
        # -------------------------------------------------

        if (
            source_indices.shape[0]
            !=
            travel_time.shape[0]
        ):
            raise ValueError(
                "Number of source locations must match "
                "the travel-time batch size."
            )

        # -------------------------------------------------
        # Validate source coordinates numerically before
        # converting to integer indices.
        # -------------------------------------------------

        source_coordinates = source_indices.to(
            device=travel_time.device,
            dtype=torch.float32,
        )

        if not torch.isfinite(
            source_coordinates
        ).all():
            raise ValueError(
                "source_indices contains NaN "
                "or infinite values."
            )

        # -------------------------------------------------
        # Source coordinates represent voxel indices.
        #
        # They must therefore be integer-valued.
        # -------------------------------------------------

        if not torch.equal(
            source_coordinates,
            source_coordinates.round(),
        ):
            raise ValueError(
                "source_indices must contain integer-valued "
                "voxel coordinates."
            )

        source_indices = source_coordinates.to(
            dtype=torch.long
        )

        # =================================================
        # EXTRACT SOURCE TRAVEL TIMES
        # =================================================

        source_values = []

        for batch_index in range(
            travel_time.shape[0]
        ):

            depth_index = int(
                source_indices[
                    batch_index,
                    0
                ].item()
            )

            crossline_index = int(
                source_indices[
                    batch_index,
                    1
                ].item()
            )

            inline_index = int(
                source_indices[
                    batch_index,
                    2
                ].item()
            )

            # -------------------------------------------------
            # Validate depth coordinate.
            # -------------------------------------------------

            if not (
                0
                <=
                depth_index
                <
                travel_time.shape[2]
            ):
                raise ValueError(
                    f"Source depth index "
                    f"{depth_index} is outside "
                    f"the valid range "
                    f"[0, {travel_time.shape[2] - 1}]."
                )

            # -------------------------------------------------
            # Validate crossline coordinate.
            # -------------------------------------------------

            if not (
                0
                <=
                crossline_index
                <
                travel_time.shape[3]
            ):
                raise ValueError(
                    f"Source crossline index "
                    f"{crossline_index} is outside "
                    f"the valid range "
                    f"[0, {travel_time.shape[3] - 1}]."
                )

            # -------------------------------------------------
            # Validate inline coordinate.
            # -------------------------------------------------

            if not (
                0
                <=
                inline_index
                <
                travel_time.shape[4]
            ):
                raise ValueError(
                    f"Source inline index "
                    f"{inline_index} is outside "
                    f"the valid range "
                    f"[0, {travel_time.shape[4] - 1}]."
                )

            # -------------------------------------------------
            # Extract all channels at the source voxel.
            # -------------------------------------------------

            source_values.append(
                travel_time[
                    batch_index,
                    :,
                    depth_index,
                    crossline_index,
                    inline_index,
                ]
            )

        # -------------------------------------------------
        # Combine source values.
        # -------------------------------------------------

        source_time = torch.stack(
            source_values,
            dim=0,
        )

        # =================================================
        # SOURCE CONDITION LOSS
        # =================================================

        loss = source_time.pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Source-condition loss contains "
                "NaN or infinite values."
            )

        return loss

    # =====================================================
    # TRAVEL-TIME SUPERVISION LOSS
    # =====================================================

    def travel_time_supervision_loss(
        self,
        predicted,
        target,
    ):
        """
        Compute optional supervised travel-time loss.

        This term is evaluated only when a valid
        travel-time target is supplied.
        """

        # -------------------------------------------------
        # Validate prediction.
        # -------------------------------------------------

        self._validate_field(
            predicted,
            "predicted travel_time",
        )

        # -------------------------------------------------
        # No target means no supervised travel-time term.
        # -------------------------------------------------

        if target is None:
            return predicted.new_zeros(())

        # -------------------------------------------------
        # Validate target.
        # -------------------------------------------------

        self._validate_field(
            target,
            "travel_time_target",
        )

        # -------------------------------------------------
        # Validate shape.
        # -------------------------------------------------

        if predicted.shape != target.shape:
            raise ValueError(
                "Predicted and target travel-time fields "
                "must have identical shapes. "
                f"Predicted: {tuple(predicted.shape)}, "
                f"Target: {tuple(target.shape)}."
            )

        # =================================================
        # MEAN SQUARED TRAVEL-TIME ERROR
        # =================================================

        loss = (
            predicted
            -
            target
        ).pow(2).mean()

        if not torch.isfinite(loss):
            raise ValueError(
                "Travel-time supervision loss contains "
                "NaN or infinite values."
            )

        return loss

    # =====================================================
    # COMPLETE PHYSICS LOSS
    # =====================================================

    def forward(
        self,
        velocity,
        travel_time=None,
        source_indices=None,
        travel_time_target=None,
    ):
        """
        Calculate the complete physics-informed loss.

        Parameters
        ----------
        velocity : torch.Tensor
            Physical P-wave velocity field.

            Shape:

                [B,C,D,H,W]

        travel_time : torch.Tensor
            Predicted travel-time field.

            Shape:

                [B,C,D,H,W]

        source_indices : torch.Tensor, optional
            Source coordinates.

            Shape:

                [B,3]

            Ordering:

                [depth, crossline, inline]

        travel_time_target : torch.Tensor, optional
            Independent travel-time target.

        Returns
        -------
        dict
            Dictionary containing:

                physics_loss
                eikonal_loss
                source_loss
                travel_time_loss

                weighted_eikonal_loss
                weighted_source_loss
                weighted_travel_time_loss
        """

        # =================================================
        # TRAVEL TIME IS REQUIRED FOR THE EIKONAL EQUATION
        # =================================================

        if travel_time is None:
            raise ValueError(
                "travel_time must be supplied to PhysicsLoss. "
                "The Eikonal equation cannot be evaluated "
                "without a predicted travel-time field."
            )

        # =================================================
        # VALIDATE VELOCITY
        # =================================================

        self._validate_field(
            velocity,
            "velocity",
        )

        # =================================================
        # VALIDATE TRAVEL TIME
        # =================================================

        self._validate_field(
            travel_time,
            "travel_time",
        )

        # =================================================
        # VALIDATE FIELD SHAPES
        # =================================================

        if velocity.shape != travel_time.shape:
            raise ValueError(
                "velocity and travel_time must have "
                "identical shapes. "
                f"Velocity: {tuple(velocity.shape)}, "
                f"Travel time: {tuple(travel_time.shape)}."
            )

        # =================================================
        # EIKONAL LOSS
        # =================================================

        eikonal = self.eikonal_loss(
            travel_time=travel_time,
            velocity=velocity,
        )

        # =================================================
        # SOURCE CONDITION LOSS
        # =================================================

        source = self.source_condition_loss(
            travel_time=travel_time,
            source_indices=source_indices,
        )

        # =================================================
        # OPTIONAL TRAVEL-TIME SUPERVISION
        # =================================================

        travel_time_supervision = (
            self.travel_time_supervision_loss(
                predicted=travel_time,
                target=travel_time_target,
            )
        )

        # =================================================
        # APPLY PHYSICS SUB-WEIGHTS
        # =================================================

        weighted_eikonal = (
            self.eikonal_weight
            *
            eikonal
        )

        weighted_source = (
            self.source_weight
            *
            source
        )

        weighted_travel_time = (
            self.travel_time_weight
            *
            travel_time_supervision
        )

        # =================================================
        # TOTAL PHYSICS LOSS
        # =================================================

        total = (
            weighted_eikonal
            +
            weighted_source
            +
            weighted_travel_time
        )

        # =================================================
        # NUMERICAL VALIDATION
        # =================================================

        physics_components = {
            "physics_loss": total,
            "eikonal_loss": eikonal,
            "source_loss": source,
            "travel_time_loss":
                travel_time_supervision,
            "weighted_eikonal_loss":
                weighted_eikonal,
            "weighted_source_loss":
                weighted_source,
            "weighted_travel_time_loss":
                weighted_travel_time,
        }

        for name, value in physics_components.items():

            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise TypeError(
                    f"{name} must be a torch.Tensor."
                )

            if not torch.isfinite(
                value
            ).all():
                raise ValueError(
                    f"{name} contains NaN or "
                    "infinite values."
                )

        # =================================================
        # RETURN COMPLETE PHYSICS DIAGNOSTICS
        # =================================================

        return physics_components