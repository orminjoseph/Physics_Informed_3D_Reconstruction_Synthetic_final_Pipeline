
"""
====================================================================
Network3D
====================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

ACOUSTIC-WAVE VERSION
--------------------
This version replaces the old Eikonal/travel-time head with a
pressure-wavefield head.

Expected forward outputs:
    reconstructed_cube:
        [B, out_channels, D, H, W]

    pressure_wavefield:
        [B, 1, T, D, H, W]

    log_variance:
        [B, out_channels, D, H, W]

IMPORTANT SCIENTIFIC NOTE
-------------------------
The pressure head predicts a time-dependent pressure field from
the encoded seismic features. Its physical validity must be enforced
by the acoustic-wave-equation loss, which also requires a compatible
velocity model, source term, and discretization.

This network does not solve the acoustic wave equation by itself.
====================================================================
Author: Ormin Joseph
====================================================================
"""

import torch
import torch.nn as nn

from models.encoder import Encoder3D
from models.bottleneck import Bottleneck3D
from models.decoder import Decoder3D


class Network3D(nn.Module):
    """
    3D encoder-decoder network for seismic reconstruction.

    Parameters
    ----------
    in_channels : int
        Number of input seismic channels.

    out_channels : int
        Number of reconstructed seismic channels.

    use_attention : bool
        Enable attention gates in the decoder.

    use_residual : bool
        Enable residual connections where supported.

    use_uncertainty : bool
        Enable the aleatoric uncertainty head.

    pressure_time_steps : int
        Number of temporal samples predicted by the pressure head.

    Notes
    -----
    The forward method returns three outputs:

        reconstruction, pressure_wavefield, log_variance

    The pressure-wavefield head is separate from the reconstruction
    and uncertainty heads.

    The temporal dimension of pressure_wavefield is not assumed to
    equal any spatial dimension of the input. It is explicitly
    controlled by pressure_time_steps.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        use_attention=True,
        use_residual=True,
        use_uncertainty=True,
        pressure_time_steps=64,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Validate constructor arguments
        # ----------------------------------------------------------

        if not isinstance(in_channels, int) or isinstance(
            in_channels, bool
        ):
            raise TypeError("in_channels must be an integer.")

        if in_channels <= 0:
            raise ValueError("in_channels must be greater than zero.")

        if not isinstance(out_channels, int) or isinstance(
            out_channels, bool
        ):
            raise TypeError("out_channels must be an integer.")

        if out_channels <= 0:
            raise ValueError("out_channels must be greater than zero.")

        if not isinstance(use_attention, bool):
            raise TypeError("use_attention must be a boolean.")

        if not isinstance(use_residual, bool):
            raise TypeError("use_residual must be a boolean.")

        if not isinstance(use_uncertainty, bool):
            raise TypeError("use_uncertainty must be a boolean.")

        if not isinstance(pressure_time_steps, int) or isinstance(
            pressure_time_steps, bool
        ):
            raise TypeError(
                "pressure_time_steps must be an integer."
            )

        if pressure_time_steps < 3:
            raise ValueError(
                "pressure_time_steps must be at least 3 for "
                "second-order temporal differentiation."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_uncertainty = use_uncertainty
        self.pressure_time_steps = pressure_time_steps

        # ----------------------------------------------------------
        # 3D encoder
        # ----------------------------------------------------------

        self.encoder = Encoder3D(
            in_channels=in_channels,
            use_residual=use_residual,
        )

        # ----------------------------------------------------------
        # 3D bottleneck
        #
        # This assumes the existing encoder's deepest feature map
        # has 512 channels, as in the supplied network.
        # ----------------------------------------------------------

        self.bottleneck = Bottleneck3D(
            channels=512,
            use_residual=use_residual,
        )

        # ----------------------------------------------------------
        # 3D decoder
        # ----------------------------------------------------------

        self.decoder = Decoder3D(
            use_attention=use_attention,
            use_residual=use_residual,
        )

        # Decoder feature channels are assumed to be 32, consistent
        # with the supplied network implementation.
        decoder_channels = 32

        # ----------------------------------------------------------
        # Reconstruction head
        # ----------------------------------------------------------
        # No sigmoid/tanh: seismic amplitudes can be signed.

        self.reconstruction_head = nn.Conv3d(
            in_channels=decoder_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        # ----------------------------------------------------------
        # Pressure-wavefield head
        # ----------------------------------------------------------
        # Predicts pressure values at each requested time step.
        #
        # Output before reshaping:
        #     [B, pressure_time_steps, D, H, W]
        #
        # Final output:
        #     [B, 1, T, D, H, W]
        #
        # The output channel represents scalar acoustic pressure.
        # No positivity activation is applied because pressure
        # perturbations can be positive or negative.
        # ----------------------------------------------------------

        self.pressure_wavefield_head = nn.Conv3d(
            in_channels=decoder_channels,
            out_channels=pressure_time_steps,
            kernel_size=1,
        )

        nn.init.normal_(
            self.pressure_wavefield_head.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.pressure_wavefield_head.bias)

        # ----------------------------------------------------------
        # Aleatoric uncertainty head
        # ----------------------------------------------------------
        # Predicts log(sigma_a^2) for each reconstructed voxel.
        # The exponential is applied by the uncertainty loss or
        # uncertainty evaluation code, not in this head.
        # ----------------------------------------------------------

        self.uncertainty_head = nn.Conv3d(
            in_channels=decoder_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        # Start at log variance = 0, corresponding to variance = 1.
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input seismic cube with shape [B, C, D, H, W].

        Returns
        -------
        reconstructed_cube : torch.Tensor
            Shape [B, out_channels, D, H, W].

        pressure_wavefield : torch.Tensor
            Shape [B, 1, T, D, H, W].

        log_variance : torch.Tensor
            Shape [B, out_channels, D, H, W].
        """

        # ----------------------------------------------------------
        # Validate input
        # ----------------------------------------------------------

        if not isinstance(x, torch.Tensor):
            raise TypeError("Network input must be a torch.Tensor.")

        if x.ndim != 5:
            raise ValueError(
                "Network input must have shape [B, C, D, H, W]. "
                f"Received {tuple(x.shape)}."
            )

        if x.shape[0] < 1:
            raise ValueError("Input batch size must be positive.")

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channel(s), "
                f"but received {x.shape[1]}."
            )

        if min(x.shape[2:]) < 1:
            raise ValueError("Spatial dimensions must be positive.")

        if not torch.isfinite(x).all():
            raise ValueError("Network input contains NaN or Inf.")

        # ----------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------

        x1, x2, x3, x4, x5 = self.encoder(x)

        # ----------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------

        bottleneck_output = self.bottleneck(x5)

        # ----------------------------------------------------------
        # Decoder
        # ----------------------------------------------------------

        decoder_output = self.decoder(
            x1,
            x2,
            x3,
            x4,
            bottleneck_output,
        )

        # ----------------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------------

        reconstructed_cube = self.reconstruction_head(
            decoder_output
        )

        # ----------------------------------------------------------
        # Time-dependent pressure field
        # ----------------------------------------------------------

        pressure_sequence = self.pressure_wavefield_head(
            decoder_output
        )

        batch_size, _, depth, height, width = pressure_sequence.shape

        pressure_wavefield = pressure_sequence.reshape(
            batch_size,
            1,
            self.pressure_time_steps,
            depth,
            height,
            width,
        )

        # ----------------------------------------------------------
        # Aleatoric uncertainty
        # ----------------------------------------------------------

        if self.use_uncertainty:
            log_variance = self.uncertainty_head(decoder_output)
        else:
            log_variance = torch.zeros_like(reconstructed_cube)

        # ----------------------------------------------------------
        # Validate output shapes
        # ----------------------------------------------------------

        expected_reconstruction_shape = (
            x.shape[0],
            self.out_channels,
            *x.shape[2:],
        )

        if tuple(reconstructed_cube.shape) != (
            expected_reconstruction_shape
        ):
            raise RuntimeError(
                "Unexpected reconstruction shape. "
                f"Expected {expected_reconstruction_shape}; "
                f"received {tuple(reconstructed_cube.shape)}."
            )

        expected_pressure_shape = (
            x.shape[0],
            1,
            self.pressure_time_steps,
            *x.shape[2:],
        )

        if tuple(pressure_wavefield.shape) != expected_pressure_shape:
            raise RuntimeError(
                "Unexpected pressure-wavefield shape. "
                f"Expected {expected_pressure_shape}; "
                f"received {tuple(pressure_wavefield.shape)}."
            )

        if log_variance.shape != reconstructed_cube.shape:
            raise RuntimeError(
                "Log-variance shape must match reconstruction shape. "
                f"Received {tuple(log_variance.shape)} versus "
                f"{tuple(reconstructed_cube.shape)}."
            )

        # ----------------------------------------------------------
        # Validate finite outputs
        # ----------------------------------------------------------

        if not torch.isfinite(reconstructed_cube).all():
            raise RuntimeError(
                "Reconstruction contains NaN or Inf values."
            )

        if not torch.isfinite(pressure_wavefield).all():
            raise RuntimeError(
                "Pressure wavefield contains NaN or Inf values."
            )

        if not torch.isfinite(log_variance).all():
            raise RuntimeError(
                "Log variance contains NaN or Inf values."
            )

        return (
            reconstructed_cube,
            pressure_wavefield,
            log_variance,
        )
