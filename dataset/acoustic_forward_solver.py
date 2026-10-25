
"""
====================================================================
3D Acoustic Forward Solver
====================================================================

PhD framework:
Physics-Informed 3D Encoder-Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings.

Purpose
-------
Numerically simulate scalar acoustic wave propagation in a
three-dimensional, heterogeneous velocity model.

Governing equation
------------------
    (1 / v**2) * d2p/dt2 - Laplacian(p) = s

where:
    p = time-dependent acoustic pressure field
    v = acoustic velocity model
    s = explicitly defined source term

Numerical method
----------------
Second-order finite differences in time and space.

The solver returns:
    pressure: [1, T, D, H, W]
    source:   [1, T, D, H, W]

These are single-sample tensors. Adding a batch dimension produces
the [B, 1, T, D, H, W] convention required by AcousticWaveEquationLoss.

Important limitations
---------------------
1. This is a scalar acoustic simulation, not an elastic-wave solver.
2. The current boundary condition is zero pressure at the outer
   boundaries. Reflections can therefore occur.
3. The source is a prescribed point-source Ricker wavelet.
4. Velocity and spatial sampling must use consistent physical units.
5. This solver does not infer a velocity model from seismic data.
6. The resulting pressure field must be validated before being used
   to train the proposed physics-informed reconstruction network.

Author: Ormin Joseph
====================================================================
"""

import math

import torch


class AcousticForwardSolver:
    """
    Finite-difference solver for the 3D acoustic wave equation.

    Spatial axes:
        D = depth
        H = crossline
        W = inline

    Input velocity:
        [D, H, W]
        [1, D, H, W]
        or [1, 1, D, H, W]

    Output pressure and source:
        [1, T, D, H, W]

    The leading dimension in the output is the scalar channel.
    The batch dimension is added by the dataset/DataLoader pipeline.
    """

    def __init__(
        self,
        dt=0.0005,
        dx=10.0,
        dy=10.0,
        dz=10.0,
        n_time_steps=128,
        source_frequency=25.0,
        source_amplitude=1.0,
        cfl_safety=0.9,
        device="cpu",
    ):
        """Configure physical sampling and numerical simulation."""

        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        self.n_time_steps = int(n_time_steps)
        self.source_frequency = float(source_frequency)
        self.source_amplitude = float(source_amplitude)
        self.cfl_safety = float(cfl_safety)
        self.device = torch.device(device)

        self._validate_configuration()

    def _validate_configuration(self):
        """Reject invalid numerical and physical settings."""

        positive_values = {
            "dt": self.dt,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "source_frequency": self.source_frequency,
            "cfl_safety": self.cfl_safety,
        }

        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite; "
                    f"received {value}."
                )

        if self.n_time_steps < 3:
            raise ValueError(
                "n_time_steps must be at least 3."
            )

        if not math.isfinite(self.source_amplitude):
            raise ValueError(
                "source_amplitude must be finite."
            )

        if self.cfl_safety > 1.0:
            raise ValueError(
                "cfl_safety must not exceed 1.0."
            )

        # The source wavelet must be adequately sampled in time.
        nyquist_frequency = 1.0 / (2.0 * self.dt)

        if self.source_frequency >= nyquist_frequency:
            raise ValueError(
                "source_frequency must be below the temporal "
                "Nyquist frequency. Reduce the source frequency "
                "or decrease dt."
            )

    @staticmethod
    def _prepare_velocity(velocity, device):
        """
        Convert velocity to [D, H, W] float32.

        The solver accepts common single-sample velocity shapes.
        """

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.as_tensor(velocity)

        velocity = velocity.to(
            device=device,
            dtype=torch.float32,
        )

        if velocity.ndim == 5:
            if velocity.shape[:2] != (1, 1):
                raise ValueError(
                    "A 5D velocity input must have shape "
                    "[1, 1, D, H, W]."
                )
            velocity = velocity[0, 0]

        elif velocity.ndim == 4:
            if velocity.shape[0] != 1:
                raise ValueError(
                    "A 4D velocity input must have shape "
                    "[1, D, H, W]."
                )
            velocity = velocity[0]

        elif velocity.ndim != 3:
            raise ValueError(
                "Velocity must have shape [D,H,W], "
                "[1,D,H,W], or [1,1,D,H,W]."
            )

        if any(size < 3 for size in velocity.shape):
            raise ValueError(
                "Each spatial dimension must contain at least "
                "three samples."
            )

        if not torch.isfinite(velocity).all():
            raise ValueError(
                "Velocity contains NaN or infinite values."
            )

        if torch.any(velocity <= 0):
            raise ValueError(
                "Velocity must be strictly positive."
            )

        return velocity.contiguous()

    def _check_cfl_condition(self, velocity):
        """
        Check the explicit finite-difference stability condition.

        For the standard 3D second-order spatial stencil, the
        theoretical condition is:

            vmax * dt * sqrt(1/dx**2 + 1/dy**2 + 1/dz**2) <= 1

        cfl_safety applies a stricter safety margin.
        """

        vmax = float(velocity.max().item())

        cfl_number = (
            vmax
            * self.dt
            * math.sqrt(
                1.0 / self.dx**2
                + 1.0 / self.dy**2
                + 1.0 / self.dz**2
            )
        )

        if cfl_number > self.cfl_safety:
            stable_dt = (
                self.cfl_safety
                / (
                    vmax
                    * math.sqrt(
                        1.0 / self.dx**2
                        + 1.0 / self.dy**2
                        + 1.0 / self.dz**2
                    )
                )
            )

            raise ValueError(
                "The requested time step fails the configured "
                f"CFL safety check. CFL={cfl_number:.5f}, "
                f"allowed={self.cfl_safety:.5f}. "
                f"Reduce dt to approximately {stable_dt:.8g} "
                "seconds or less."
            )

        return cfl_number

    def _laplacian(self, pressure):
        """
        Calculate the 3D spatial Laplacian on interior points.

        Input:
            pressure: [D, H, W]

        Output:
            Laplacian: [D-2, H-2, W-2]

        The returned tensor excludes the outer spatial boundaries.
        """

        center = pressure[1:-1, 1:-1, 1:-1]

        d2p_dz2 = (
            pressure[2:, 1:-1, 1:-1]
            - 2.0 * center
            + pressure[:-2, 1:-1, 1:-1]
        ) / self.dz**2

        d2p_dy2 = (
            pressure[1:-1, 2:, 1:-1]
            - 2.0 * center
            + pressure[1:-1, :-2, 1:-1]
        ) / self.dy**2

        d2p_dx2 = (
            pressure[1:-1, 1:-1, 2:]
            - 2.0 * center
            + pressure[1:-1, 1:-1, :-2]
        ) / self.dx**2

        return d2p_dx2 + d2p_dy2 + d2p_dz2

    def _create_ricker_source(
        self,
        spatial_shape,
        source_location=None,
    ):
        """
        Construct an explicit point source using a Ricker wavelet.

        source_location uses (depth, crossline, inline) indices.

        Returns:
            source: [T, D, H, W]

        The source is a prescribed forcing term for the PDE as
        written, not a recorded seismic trace.
        """

        depth, height, width = spatial_shape

        if source_location is None:
            # Place the source inside the model, away from boundaries.
            source_location = (
                depth // 2,
                height // 2,
                width // 2,
            )

        if len(source_location) != 3:
            raise ValueError(
                "source_location must be (depth, crossline, inline)."
            )

        iz, iy, ix = (
            int(source_location[0]),
            int(source_location[1]),
            int(source_location[2]),
        )

        if not (
            1 <= iz < depth - 1
            and 1 <= iy < height - 1
            and 1 <= ix < width - 1
        ):
            raise ValueError(
                "The source location must lie inside the spatial "
                "domain and not on its outer boundary."
            )

        # Time samples are expressed in seconds.
        time = (
            torch.arange(
                self.n_time_steps,
                device=self.device,
                dtype=torch.float32,
            )
            * self.dt
        )

        # Standard Ricker wavelet.
        # Its peak is delayed to allow a smooth initial source onset.
        t_peak = 1.5 / self.source_frequency

        argument = math.pi * self.source_frequency * (
            time - t_peak
        )

        argument_squared = argument.square()

        wavelet = self.source_amplitude * (
            1.0 - 2.0 * argument_squared
        ) * torch.exp(-argument_squared)

        # Allocate the source forcing tensor.
        source = torch.zeros(
            (
                self.n_time_steps,
                depth,
                height,
                width,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        # Apply the time-dependent source at one grid point.
        source[:, iz, iy, ix] = wavelet

        return source

    def simulate(
        self,
        velocity,
        source_location=None,
    ):
        """
        Simulate acoustic pressure propagation.

        Parameters
        ----------
        velocity:
            Positive velocity model in metres per second.

        source_location:
            Optional (depth, crossline, inline) source indices.

        Returns
        -------
        pressure:
            Float32 tensor [1, T, D, H, W].

        source:
            Float32 tensor [1, T, D, H, W].

        The output tensors are single-sample tensors. A DataLoader
        can add the batch dimension when samples are collated.
        """

        velocity = self._prepare_velocity(
            velocity,
            self.device,
        )

        cfl_number = self._check_cfl_condition(velocity)

        depth, height, width = velocity.shape

        source = self._create_ricker_source(
            spatial_shape=(depth, height, width),
            source_location=source_location,
        )

        # Store the full time-dependent pressure field.
        pressure = torch.zeros(
            (
                self.n_time_steps,
                depth,
                height,
                width,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        # Two consecutive time levels for the explicit time update.
        previous = torch.zeros_like(velocity)
        current = torch.zeros_like(velocity)

        # Precompute v^2 * dt^2.
        velocity_factor = velocity.square() * self.dt**2

        # Explicit second-order time stepping:
        #
        # p[t+1] = 2p[t] - p[t-1]
        #          + dt^2 * v^2 * (Laplacian(p[t]) + s[t])
        #
        # The equation is therefore consistent with:
        #
        # (1/v^2) * p_tt - Laplacian(p) = s
        #
        # No gradient tracking is required for this data-generation
        # solver. The generated pressure is a training target/input,
        # not a differentiable model prediction.
        with torch.no_grad():

            # Initial pressure is zero.
            pressure[0] = current

            for time_index in range(self.n_time_steps - 1):

                next_pressure = torch.zeros_like(current)

                laplacian = self._laplacian(current)

                source_interior = source[
                    time_index,
                    1:-1,
                    1:-1,
                    1:-1,
                ]

                next_pressure[
                    1:-1,
                    1:-1,
                    1:-1,
                ] = (
                    2.0
                    * current[1:-1, 1:-1, 1:-1]
                    - previous[1:-1, 1:-1, 1:-1]
                    + velocity_factor[
                        1:-1,
                        1:-1,
                        1:-1,
                    ]
                    * (laplacian + source_interior)
                )

                # Outer boundaries remain zero-pressure boundaries.
                # This is simple and explicit, but can cause reflections.

                if not torch.isfinite(next_pressure).all():
                    raise RuntimeError(
                        "The acoustic simulation became non-finite "
                        f"at time index {time_index}. Check the "
                        "velocity, sampling intervals, source, "
                        "and CFL condition."
                    )

                pressure[time_index + 1] = next_pressure

                previous = current
                current = next_pressure

        # Convert [T,D,H,W] to the single-sample convention
        # [channel,T,D,H,W].
        pressure = pressure.unsqueeze(0).contiguous()
        source = source.unsqueeze(0).contiguous()

        if not torch.isfinite(pressure).all():
            raise RuntimeError(
                "Generated pressure field contains non-finite values."
            )

        if not torch.isfinite(source).all():
            raise RuntimeError(
                "Generated source contains non-finite values."
            )

        # Retain the CFL value for diagnostic inspection.
        self.last_cfl_number = cfl_number

        return pressure, source
