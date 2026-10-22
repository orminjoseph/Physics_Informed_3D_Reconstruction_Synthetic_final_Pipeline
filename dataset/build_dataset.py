"""
======================================================================
DATASET BUILDER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Construct the seismic dataset selected by the global configuration.

Supported dataset modes
-----------------------

1. synthetic
2. f3

The active dataset mode is controlled by:

    DATASET_MODE

in:

    utils/config.py

The builder does not perform the train/validation split.

Dataset workflow
----------------

    Global Configuration
            |
            v
    build_dataset()
            |
            v
    Complete Dataset
            |
            v
    split_dataset()
            |
            +------------------+
            |                  |
            v                  v
       Training Set       Validation Set

Tensor convention
-----------------

Individual dataset samples are expected to follow:

    [C, D, H, W]

where:

    C = channel
    D = depth
    H = crossline
    W = inline

The DataLoader subsequently creates:

    [B, C, D, H, W]

where:

    B = batch size

Author
------
Ormin Joseph
======================================================================
"""

# =====================================================================
# STANDARD LIBRARY
# =====================================================================

import os


# =====================================================================
# PROJECT CONFIGURATION
# =====================================================================

from utils.config import (
    DATASET_MODE,

    # ---------------------------------------------------------------
    # Synthetic dataset configuration
    # ---------------------------------------------------------------

    SYNTHETIC_NUM_SAMPLES,
    SYNTHETIC_PATCH_SIZE,
    SYNTHETIC_MISSING_PROBABILITY,

    # ---------------------------------------------------------------
    # F3 dataset configuration
    # ---------------------------------------------------------------

    F3_PATH,
    F3_PATCH_SIZE,
    F3_STRIDE,
    F3_MISSING_PROBABILITY,
)


# =====================================================================
# SYNTHETIC DATASET
# =====================================================================

from dataset.synthetic_dataset import (
    SyntheticSeismicDataset,
)


# =====================================================================
# INTERNAL VALIDATION HELPERS
# =====================================================================

def _validate_spatial_size(
    value,
    name,
):
    """
    Validate a three-dimensional seismic spatial size.

    Parameters
    ----------
    value : tuple or list
        Expected form:

            (Depth, Crossline, Inline)

    name : str
        Configuration parameter name.

    Returns
    -------
    tuple
        Validated integer dimensions.
    """

    if not isinstance(
        value,
        (tuple, list),
    ):
        raise TypeError(
            f"{name} must be a tuple or list containing "
            "(Depth, Crossline, Inline)."
        )

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly three dimensions: "
            "(Depth, Crossline, Inline)."
        )

    if any(
        not isinstance(
            dimension,
            int,
        )
        or isinstance(
            dimension,
            bool,
        )
        or dimension <= 0
        for dimension in value
    ):
        raise ValueError(
            f"All dimensions in {name} must be "
            "positive integers."
        )

    return tuple(value)


def _validate_probability(
    value,
    name,
):
    """
    Validate a missing-data probability.

    The mask generator requires:

        0.0 <= probability < 1.0

    Therefore, the dataset builder uses the same convention.

    Parameters
    ----------
    value : float
        Missing-data probability.

    name : str
        Configuration parameter name.

    Returns
    -------
    float
        Validated probability.
    """

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{name} must be a real number."
        )

    value = float(value)

    if not (
        0.0
        <= value
        < 1.0
    ):
        raise ValueError(
            f"{name} must be between "
            "0.0 inclusive and 1.0 exclusive."
        )

    return value


def _validate_stride(
    stride,
    patch_size,
    name,
):
    """
    Validate a three-dimensional extraction stride.

    Parameters
    ----------
    stride : tuple or list
        Extraction stride:

            (Depth, Crossline, Inline)

    patch_size : tuple
        Corresponding patch dimensions.

    name : str
        Configuration parameter name.

    Returns
    -------
    tuple
        Validated integer stride.
    """

    stride = _validate_spatial_size(
        stride,
        name,
    )

    # ---------------------------------------------------------------
    # A stride greater than the corresponding patch dimension would
    # create spatial gaps between neighboring extracted patches.
    #
    # The present project uses non-overlapping or contiguous patch
    # extraction, so such a configuration is rejected explicitly.
    # ---------------------------------------------------------------

    for stride_value, patch_value, axis_name in zip(
        stride,
        patch_size,
        (
            "Depth",
            "Crossline",
            "Inline",
        ),
    ):

        if stride_value > patch_value:

            raise ValueError(
                f"{name} contains a stride larger than the "
                f"corresponding patch dimension along the "
                f"{axis_name} axis. "
                f"Received stride={stride_value}, "
                f"patch_size={patch_value}."
            )

    return stride


# =====================================================================
# DATASET BUILDER
# =====================================================================

def build_dataset():
    """
    Construct the complete dataset selected by DATASET_MODE.

    Returns
    -------
    torch.utils.data.Dataset
        Complete seismic dataset before train/validation splitting.

    Raises
    ------
    ValueError
        If DATASET_MODE is not supported.

    FileNotFoundError
        If DATASET_MODE is 'f3' and the configured SEG-Y file
        does not exist.

    Notes
    -----
    This function intentionally does not perform dataset splitting.

    The returned dataset should subsequently be passed to:

        split_dataset(dataset)

    so that training and validation data remain explicitly separated.
    """

    # =================================================================
    # DISPLAY DATASET CONFIGURATION
    # =================================================================

    print()
    print("=" * 60)
    print("BUILDING DATASET")
    print("=" * 60)

    print(
        f"Dataset mode: {DATASET_MODE}"
    )

    # =================================================================
    # VALIDATE DATASET MODE
    # =================================================================

    if not isinstance(
        DATASET_MODE,
        str,
    ):
        raise TypeError(
            "DATASET_MODE must be a string."
        )

    mode = DATASET_MODE.strip().lower()

    supported_modes = {
        "synthetic",
        "f3",
    }

    if mode not in supported_modes:

        raise ValueError(
            f"Unsupported DATASET_MODE: '{DATASET_MODE}'.\n"
            f"Supported modes are: "
            f"{sorted(supported_modes)}"
        )

    # =================================================================
    # 1. SYNTHETIC DATASET
    # =================================================================

    if mode == "synthetic":

        # -------------------------------------------------------------
        # Validate number of samples.
        # -------------------------------------------------------------

        if (
            not isinstance(
                SYNTHETIC_NUM_SAMPLES,
                int,
            )
            or isinstance(
                SYNTHETIC_NUM_SAMPLES,
                bool,
            )
        ):
            raise TypeError(
                "SYNTHETIC_NUM_SAMPLES must be "
                "a positive integer."
            )

        if SYNTHETIC_NUM_SAMPLES < 1:

            raise ValueError(
                "SYNTHETIC_NUM_SAMPLES must be at least 1."
            )

        # -------------------------------------------------------------
        # Validate synthetic cube dimensions.
        # -------------------------------------------------------------

        synthetic_patch_size = _validate_spatial_size(
            SYNTHETIC_PATCH_SIZE,
            "SYNTHETIC_PATCH_SIZE",
        )

        # -------------------------------------------------------------
        # Validate synthetic missing probability.
        # -------------------------------------------------------------

        synthetic_missing_probability = (
            _validate_probability(
                SYNTHETIC_MISSING_PROBABILITY,
                "SYNTHETIC_MISSING_PROBABILITY",
            )
        )

        # -------------------------------------------------------------
        # Display synthetic configuration.
        # -------------------------------------------------------------

        print()
        print("Synthetic dataset configuration")
        print("-" * 60)

        print(
            f"Number of samples       : "
            f"{SYNTHETIC_NUM_SAMPLES}"
        )

        print(
            f"Patch size              : "
            f"{synthetic_patch_size}"
        )

        print(
            f"Missing probability     : "
            f"{synthetic_missing_probability}"
        )

        # -------------------------------------------------------------
        # Construct synthetic dataset.
        # -------------------------------------------------------------

        dataset = SyntheticSeismicDataset(
            num_samples=SYNTHETIC_NUM_SAMPLES,
            cube_size=synthetic_patch_size,
            missing_probability=(
                synthetic_missing_probability
            ),
        )

        # -------------------------------------------------------------
        # Validate resulting dataset.
        # -------------------------------------------------------------

        if len(dataset) == 0:

            raise RuntimeError(
                "Synthetic dataset was created but contains "
                "zero samples."
            )

        print()
        print(
            "Synthetic dataset created successfully."
        )

        print(
            f"Total samples: {len(dataset)}"
        )

        print("=" * 60)
        print()

        return dataset

    # =================================================================
    # 2. F3 DATASET
    # =================================================================

    if mode == "f3":

        # -------------------------------------------------------------
        # Validate F3 SEG-Y path.
        # -------------------------------------------------------------

        if not isinstance(
            F3_PATH,
            str,
        ):
            raise TypeError(
                "F3_PATH must be a string."
            )

        f3_path = os.path.abspath(
            os.path.expanduser(
                F3_PATH
            )
        )

        if not os.path.isfile(
            f3_path
        ):
            raise FileNotFoundError(
                "F3 SEG-Y dataset was not found.\n"
                f"Configured path:\n"
                f"{F3_PATH}"
            )

        # -------------------------------------------------------------
        # Validate F3 patch size.
        # -------------------------------------------------------------

        f3_patch_size = _validate_spatial_size(
            F3_PATCH_SIZE,
            "F3_PATCH_SIZE",
        )

        # -------------------------------------------------------------
        # Validate F3 stride.
        # -------------------------------------------------------------

        f3_stride = _validate_stride(
            F3_STRIDE,
            f3_patch_size,
            "F3_STRIDE",
        )

        # -------------------------------------------------------------
        # Validate F3 missing probability.
        # -------------------------------------------------------------

        f3_missing_probability = _validate_probability(
            F3_MISSING_PROBABILITY,
            "F3_MISSING_PROBABILITY",
        )

        # -------------------------------------------------------------
        # Display F3 configuration.
        # -------------------------------------------------------------

        print()
        print("F3 dataset configuration")
        print("-" * 60)

        print(
            f"SEG-Y path              : "
            f"{f3_path}"
        )

        print(
            f"Patch size              : "
            f"{f3_patch_size}"
        )

        print(
            f"Stride                  : "
            f"{f3_stride}"
        )

        print(
            f"Missing probability     : "
            f"{f3_missing_probability}"
        )

        # -------------------------------------------------------------
        # Import F3 dataset only when required.
        # -------------------------------------------------------------
        #
        # This preserves lazy loading of F3-specific dependencies
        # when operating in synthetic mode.

        from dataset.f3_dataset import (
            F3Dataset,
        )

        # -------------------------------------------------------------
        # Construct F3 dataset.
        # -------------------------------------------------------------

        dataset = F3Dataset(
            segy_path=f3_path,
            patch_size=f3_patch_size,
            stride=f3_stride,
            missing_probability=(
                f3_missing_probability
            ),
        )

        # -------------------------------------------------------------
        # Validate resulting dataset.
        # -------------------------------------------------------------

        if len(dataset) == 0:

            raise RuntimeError(
                "F3 dataset was created but contains "
                "zero patches."
            )

        print()
        print(
            "F3 dataset created successfully."
        )

        print(
            f"Total patches: {len(dataset)}"
        )

        print("=" * 60)
        print()

        return dataset

    # =================================================================
    # SAFETY FALLBACK
    # =================================================================

    raise RuntimeError(
        f"Dataset mode '{mode}' reached an unexpected "
        "execution path."
    )


# =====================================================================
# END OF MODULE
# =====================================================================