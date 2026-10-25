
# =========================================================
# PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION
# GLOBAL CONFIGURATION FILE
# =========================================================

"""
Global configuration for the acoustic-wave-equation
synthetic seismic reconstruction experiment.

Research framework
------------------

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data
Reconstruction in Complex Geological Settings.

Physical constraint
-------------------

    (1 / v^2) * d^2p/dt^2 - Laplacian(p) = s

where:
    p = acoustic pressure wavefield
    v = P-wave velocity [m/s]
    t = physical time [s]
    s = source term

The physics-informed objective uses the acoustic wave equation
only. No Eikonal or travel-time physics loss is configured.

Tensor convention
-----------------

Seismic volumes:
    [B, C, D, H, W]

where:
    B = batch size
    C = channel
    D = depth
    H = crossline
    W = inline

Individual dataset samples:
    [C, D, H, W]

Pressure wavefield convention:
    [B, 1, T, D, H, W]

Author: Ormin Joseph
"""


# =========================================================
# 1. DATASET MODE
# =========================================================

DATASET_MODE = "synthetic"

# Supported modes:
# "synthetic"
# "f3"


# =========================================================
# 2. EXPERIMENT NAME
# =========================================================

EXPERIMENT_NAME = "synthetic_training"


# =========================================================
# 3. SYNTHETIC DATASET CONFIGURATION
# =========================================================

SYNTHETIC_NUM_SAMPLES = 10

# [Depth, Crossline, Inline]
SYNTHETIC_PATCH_SIZE = (64, 128, 128)

SYNTHETIC_MISSING_PROBABILITY = 0.30


# =========================================================
# 4. F3 DATASET CONFIGURATION
# =========================================================

F3_PATH = (
    r"C:\Users\ormin\Desktop"
    r"\SEG_FILES"
    r"\F3_Demo_2023 (1)"
    r"\F3_Demo_2023"
    r"\Rawdata"
    r"\Seismic_data.sgy"
)

# [Depth, Crossline, Inline]
F3_PATCH_SIZE = (64, 64, 64)

# [Depth, Crossline, Inline]
F3_STRIDE = (64, 64, 64)

F3_MISSING_PROBABILITY = 0.30


# =========================================================
# 5. COMMON BASELINE COMPARISON CONFIGURATION
# =========================================================

BASELINE_NUM_SAMPLES = 1

# [Depth, Crossline, Inline]
BASELINE_CUBE_SIZE = (64, 128, 128)

BASELINE_MISSING_RATE = 0.30

BASELINE_GEOLOGICAL_MODE = "folded"

BASELINE_MASK_MODE = "missing_crosslines"

BASELINE_SEED = 42

OBSERVED_PRESERVATION_TOLERANCE = 1.0e-6


# =========================================================
# 6. TRAINING CONFIGURATION
# =========================================================

BATCH_SIZE = 1

NUM_EPOCHS = 5

LEARNING_RATE = 1.0e-4

WEIGHT_DECAY = 1.0e-5


# =========================================================
# 7. VALIDATION CONFIGURATION
# =========================================================

VALIDATION_SPLIT = 0.20


# =========================================================
# 8. CHECKPOINT CONFIGURATION
# =========================================================

SAVE_EVERY = 5


# =========================================================
# 9. EARLY STOPPING CONFIGURATION
# =========================================================

PATIENCE = 15


# =========================================================
# 10. UNCERTAINTY EVALUATION CONFIGURATION
# =========================================================

# None means use all available samples.

UNCERTAINTY_EVALUATION_NUM_SAMPLES = None

UNCERTAINTY_EVALUATION_SEED = 42

# None means use all eligible voxels.
UNCERTAINTY_CORRELATION_MAX_VOXELS = 500000

EVALUATE_MISSING_REGION_UNCERTAINTY = True

EVALUATE_OBSERVED_REGION_UNCERTAINTY = True

# None means analyse all available patches.
UNCERTAINTY_ERROR_CORRELATION_NUM_PATCHES = None

UNCERTAINTY_ERROR_CORRELATION_SEED = 42


# =========================================================
# 11. NOISE ROBUSTNESS EVALUATION
# =========================================================

# Noise standard deviations in normalized amplitude units.

NOISE_ROBUSTNESS_LEVELS = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
)

NOISE_ROBUSTNESS_NUM_SAMPLES = 20

NOISE_ROBUSTNESS_SEED = 42


# =========================================================
# 12. MISSING-DATA ROBUSTNESS EVALUATION
# =========================================================

MISSING_DATA_ROBUSTNESS_LEVELS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
)

MISSING_DATA_ROBUSTNESS_NUM_SAMPLES = 20

MISSING_DATA_ROBUSTNESS_SEED = 42


# =========================================================
# 13. STATISTICAL SIGNIFICANCE CONFIGURATION
# =========================================================

STATISTICAL_ALPHA = 0.05

STATISTICAL_METRIC = "SSIM"

STATISTICAL_TEST = "paired_t_test"

MULTIPLE_COMPARISON_CORRECTION = "Holm-Bonferroni"

EFFECT_SIZE = "Cohen_dz"


# =========================================================
# 14. GEOLOGICAL COMPLEXITY ROBUSTNESS
# =========================================================

GEOLOGICAL_COMPLEXITY_LEVELS = (
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
)

GEOLOGICAL_COMPLEXITY_NUM_SAMPLES = 5

GEOLOGICAL_COMPLEXITY_MISSING_PROBABILITY = 0.30

GEOLOGICAL_COMPLEXITY_DATASET_SAMPLES = 20

GEOLOGICAL_COMPLEXITY_SEED = 42


# =========================================================
# 15. FINAL REPORT CONFIGURATION
# =========================================================

FINAL_REPORT_FILENAME = "final_report.txt"

FINAL_REPORT_METADATA_FILENAME = "final_report_metadata.json"


# =========================================================
# 16. EVALUATION SETTINGS
# =========================================================

GALLERY_NUMBER_OF_SAMPLES = 5


# =========================================================
# 17. MASTER PIPELINE CONTROL
# =========================================================

RUN_TRAINING = True

RESUME_EVALUATION = True

FORCE_RERUN_EVALUATION = False


# =========================================================
# 18. CONTROLLED EXPERIMENTAL MATRIX
# =========================================================

"""
Planned full experimental matrix:

    6 geological structures
    x 5 missing mechanisms
    x 5 missing-data rates
    x 5 random seeds

Total: 750 cases.

Use a finite case limit during development.
Set the limit to None for the full matrix.
"""

CONTROLLED_MATRIX_CASE_LIMIT = 10


# =========================================================
# 19. CURVELET POCS CONFIGURATION
# =========================================================

CURVELET_NUM_SCALES = 3

CURVELET_WEDGES_PER_DIRECTION = 3

CURVELET_ITERATIONS = 12

CURVELET_THRESHOLD = 0.05

CURVELET_THRESHOLD_DECAY = 0.90

CURVELET_TOLERANCE = 1.0e-5


# =========================================================
# 20. MODEL CONFIGURATION
# =========================================================

USE_ATTENTION = True

USE_RESIDUAL = True

USE_UNCERTAINTY = True

MC_DROPOUT_SAMPLES = 20


# =========================================================
# 21. UNCERTAINTY MODEL PARAMETERS
# =========================================================

LOG_VARIANCE_MIN = -10.0

LOG_VARIANCE_MAX = 10.0


# =========================================================
# 22. ACOUSTIC PHYSICS CONFIGURATION
# =========================================================

# Enable the acoustic wave-equation physics loss.
USE_PHYSICS_LOSS = True

# Acoustic wave equation:
#
#     (1/v^2) * d^2p/dt^2 - Laplacian(p) = s
#
# The pressure wavefield, velocity model, and source must
# be supplied by the acoustic synthetic data pipeline.

PHYSICS_TYPE = "acoustic_wave_equation"

USE_ACOUSTIC_WAVE_EQUATION_LOSS = True


# =========================================================
# 23. ACOUSTIC SIMULATION PARAMETERS
# =========================================================

# Temporal sampling interval [seconds].
ACOUSTIC_DT = 0.0005

# Spatial grid intervals [metres].
#
# Keep these separate from the general DX, DY, and DZ
# settings. These values must agree with the acoustic
# solver's grid convention.

ACOUSTIC_DX = 10.0

ACOUSTIC_DY = 10.0

ACOUSTIC_DZ = 10.0

# Number of pressure-wavefield time samples.
# The temporal second derivative requires at least 3.

ACOUSTIC_TIME_STEPS = 128

# Ricker-type source dominant frequency [Hz].
ACOUSTIC_SOURCE_FREQUENCY = 25.0

# Source amplitude in the synthetic solver's amplitude units.
ACOUSTIC_SOURCE_AMPLITUDE = 1.0

# CFL safety factor used by the acoustic solver, if supported.
# This does not replace a solver-specific stability check.

ACOUSTIC_CFL_SAFETY = 0.90

# None means use the solver's configured/default snapshot.
# Otherwise, a solver may use the specified time index
# for optional snapshot-based diagnostics.

ACOUSTIC_SNAPSHOT_INDEX = None


# =========================================================
# 24. COMPOSITE LOSS WEIGHTS
# =========================================================

"""
Composite training objective:

    L_total =
        lambda_mae         * L_mae
        + lambda_physics   * L_acoustic
        + lambda_uncertainty * L_uncertainty
        + lambda_ssim      * L_ssim

L_acoustic is exclusively the acoustic wave-equation loss.
"""

LOSS_WEIGHTS = {
    "mae": 1.0,
    "physics": 0.10,
    "uncertainty": 0.01,
    "ssim": 0.10,
}


# =========================================================
# 25. PHYSICS-LOSS COMPONENT WEIGHTS
# =========================================================

"""
The only configured physics component is:

    L_physics = L_acoustic_wave_equation
"""

PHYSICS_LOSS_WEIGHTS = {
    "acoustic_wave_equation": 1.0,
}


# =========================================================
# 26. GENERAL SPATIAL SAMPLING
# =========================================================

# Retained for other spatial operations that use these
# project-wide parameters. The acoustic solver should use
# ACOUSTIC_DX, ACOUSTIC_DY, and ACOUSTIC_DZ.

DX = 1.0

DY = 1.0

DZ = 1.0


# =========================================================
# 27. VELOCITY MODEL CONFIGURATION
# =========================================================

"""
Physical P-wave velocity:

    V [m/s]

For normalized velocity inputs:
    VELOCITY_NORMALIZED = True

For physical velocity inputs:
    VELOCITY_NORMALIZED = False

The acoustic equation requires physically meaningful
velocity values, or a correctly implemented conversion
from normalized values to physical velocity.
"""

VELOCITY_NORMALIZED = False

VELOCITY_MIN = 1500.0

VELOCITY_MAX = 5000.0


# =========================================================
# 28. SEISMIC AMPLITUDE NORMALIZATION
# =========================================================

SEISMIC_AMPLITUDE_MIN = -1.0

SEISMIC_AMPLITUDE_MAX = 1.0

SEISMIC_DATA_RANGE = (
    SEISMIC_AMPLITUDE_MAX - SEISMIC_AMPLITUDE_MIN
)


# =========================================================
# 29. MASK CONFIGURATION
# =========================================================

"""
Mask convention:

    1 = observed
    0 = missing
"""

MASK_OBSERVED_VALUE = 1.0

MASK_MISSING_VALUE = 0.0


# =========================================================
# 30. DATALOADER CONFIGURATION
# =========================================================

NUM_WORKERS = 0

PIN_MEMORY = False

PERSISTENT_WORKERS = False


# =========================================================
# 31. MIXED-PRECISION CONFIGURATION
# =========================================================

USE_AMP = False


# =========================================================
# 32. DEVICE CONFIGURATION
# =========================================================

"""
Available modes:

    "cpu"  - force CPU
    "cuda" - require CUDA
    "auto" - use CUDA when available, otherwise CPU
"""

DEVICE = "auto"


# =========================================================
# 33. OUTPUT CONFIGURATION
# =========================================================

OUTPUT_ROOT = "outputs"

CHECKPOINT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"checkpoints"
)

FIGURE_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"figures"
)

REPORT_DIR = (
    f"{OUTPUT_ROOT}/"
    f"{EXPERIMENT_NAME}/"
    f"reports"
)


# =========================================================
# 34. REPRODUCIBILITY
# =========================================================

SEED = 42


# =========================================================
# 35. CONFIGURATION VALIDATION HELPERS
# =========================================================

def _validate_positive_integer(value, name):
    """Require a positive integer, excluding booleans."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _validate_nonnegative_number(value, name):
    """Require a non-negative numeric value."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative number."
        )


def _validate_positive_number(value, name):
    """Require a strictly positive numeric value."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive number."
        )


def _validate_probability(value, name):
    """Require 0 <= value < 1."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{name} must be a real number."
        )

    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{name} must satisfy 0.0 <= value < 1.0."
        )


def _validate_spatial_size(value, name):
    """Validate (Depth, Crossline, Inline) dimensions."""

    if not isinstance(value, (tuple, list)):
        raise TypeError(
            f"{name} must be a tuple or list."
        )

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly three dimensions."
        )

    for dimension in value:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise ValueError(
                f"All dimensions in {name} must be "
                "positive integers."
            )


def _validate_stride(stride, patch_size, name):
    """Validate patch-extraction stride."""

    _validate_spatial_size(stride, name)

    for stride_value, patch_value in zip(stride, patch_size):
        if stride_value > patch_value:
            raise ValueError(
                f"{name} cannot exceed the corresponding "
                f"patch dimension: stride={stride_value}, "
                f"patch={patch_value}."
            )


# =========================================================
# 36. CONFIGURATION VALIDATION
# =========================================================

def validate_config():
    """Validate dataset, training, acoustic physics, and outputs."""

    # -----------------------------------------------------
    # Dataset mode and experiment name
    # -----------------------------------------------------

    if DATASET_MODE not in {"synthetic", "f3"}:
        raise ValueError(
            "DATASET_MODE must be 'synthetic' or 'f3'."
        )

    if (
        not isinstance(EXPERIMENT_NAME, str)
        or not EXPERIMENT_NAME.strip()
    ):
        raise ValueError(
            "EXPERIMENT_NAME must be a non-empty string."
        )

    # -----------------------------------------------------
    # Synthetic dataset
    # -----------------------------------------------------

    _validate_positive_integer(
        SYNTHETIC_NUM_SAMPLES,
        "SYNTHETIC_NUM_SAMPLES",
    )

    _validate_spatial_size(
        SYNTHETIC_PATCH_SIZE,
        "SYNTHETIC_PATCH_SIZE",
    )

    _validate_probability(
        SYNTHETIC_MISSING_PROBABILITY,
        "SYNTHETIC_MISSING_PROBABILITY",
    )

    # -----------------------------------------------------
    # F3 dataset
    # -----------------------------------------------------

    _validate_spatial_size(F3_PATCH_SIZE, "F3_PATCH_SIZE")

    _validate_stride(
        F3_STRIDE,
        F3_PATCH_SIZE,
        "F3_STRIDE",
    )

    _validate_probability(
        F3_MISSING_PROBABILITY,
        "F3_MISSING_PROBABILITY",
    )

    if not isinstance(F3_PATH, str) or not F3_PATH.strip():
        raise ValueError(
            "F3_PATH must be a non-empty string."
        )

    # -----------------------------------------------------
    # Baseline configuration
    # -----------------------------------------------------

    _validate_positive_integer(
        BASELINE_NUM_SAMPLES,
        "BASELINE_NUM_SAMPLES",
    )

    _validate_spatial_size(
        BASELINE_CUBE_SIZE,
        "BASELINE_CUBE_SIZE",
    )

    _validate_probability(
        BASELINE_MISSING_RATE,
        "BASELINE_MISSING_RATE",
    )

    if not isinstance(BASELINE_GEOLOGICAL_MODE, str):
        raise TypeError(
            "BASELINE_GEOLOGICAL_MODE must be a string."
        )

    if not isinstance(BASELINE_MASK_MODE, str):
        raise TypeError(
            "BASELINE_MASK_MODE must be a string."
        )

    _validate_nonnegative_number(
        OBSERVED_PRESERVATION_TOLERANCE,
        "OBSERVED_PRESERVATION_TOLERANCE",
    )

    # -----------------------------------------------------
    # Training and validation
    # -----------------------------------------------------

    _validate_positive_integer(BATCH_SIZE, "BATCH_SIZE")

    _validate_positive_integer(NUM_EPOCHS, "NUM_EPOCHS")

    _validate_positive_number(LEARNING_RATE, "LEARNING_RATE")

    _validate_nonnegative_number(WEIGHT_DECAY, "WEIGHT_DECAY")

    if not 0.0 < VALIDATION_SPLIT < 1.0:
        raise ValueError(
            "VALIDATION_SPLIT must be greater than 0 "
            "and less than 1."
        )

    _validate_positive_integer(SAVE_EVERY, "SAVE_EVERY")

    _validate_positive_integer(PATIENCE, "PATIENCE")

    # -----------------------------------------------------
    # Uncertainty
    # -----------------------------------------------------

    _validate_positive_integer(
        MC_DROPOUT_SAMPLES,
        "MC_DROPOUT_SAMPLES",
    )

    if LOG_VARIANCE_MAX <= LOG_VARIANCE_MIN:
        raise ValueError(
            "LOG_VARIANCE_MAX must exceed LOG_VARIANCE_MIN."
        )

    if UNCERTAINTY_EVALUATION_NUM_SAMPLES is not None:
        _validate_positive_integer(
            UNCERTAINTY_EVALUATION_NUM_SAMPLES,
            "UNCERTAINTY_EVALUATION_NUM_SAMPLES",
        )

    if UNCERTAINTY_CORRELATION_MAX_VOXELS is not None:
        _validate_positive_integer(
            UNCERTAINTY_CORRELATION_MAX_VOXELS,
            "UNCERTAINTY_CORRELATION_MAX_VOXELS",
        )

    if UNCERTAINTY_ERROR_CORRELATION_NUM_PATCHES is not None:
        _validate_positive_integer(
            UNCERTAINTY_ERROR_CORRELATION_NUM_PATCHES,
            "UNCERTAINTY_ERROR_CORRELATION_NUM_PATCHES",
        )

    # -----------------------------------------------------
    # Robustness configuration
    # -----------------------------------------------------

    for level in NOISE_ROBUSTNESS_LEVELS:
        _validate_nonnegative_number(
            level,
            "NOISE_ROBUSTNESS_LEVELS",
        )

    for level in MISSING_DATA_ROBUSTNESS_LEVELS:
        _validate_probability(
            level,
            "MISSING_DATA_ROBUSTNESS_LEVELS",
        )

    _validate_positive_integer(
        NOISE_ROBUSTNESS_NUM_SAMPLES,
        "NOISE_ROBUSTNESS_NUM_SAMPLES",
    )

    _validate_positive_integer(
        MISSING_DATA_ROBUSTNESS_NUM_SAMPLES,
        "MISSING_DATA_ROBUSTNESS_NUM_SAMPLES",
    )

    # -----------------------------------------------------
    # Geological complexity
    # -----------------------------------------------------

    if not GEOLOGICAL_COMPLEXITY_LEVELS:
        raise ValueError(
            "GEOLOGICAL_COMPLEXITY_LEVELS cannot be empty."
        )

    _validate_positive_integer(
        GEOLOGICAL_COMPLEXITY_NUM_SAMPLES,
        "GEOLOGICAL_COMPLEXITY_NUM_SAMPLES",
    )

    _validate_probability(
        GEOLOGICAL_COMPLEXITY_MISSING_PROBABILITY,
        "GEOLOGICAL_COMPLEXITY_MISSING_PROBABILITY",
    )

    _validate_positive_integer(
        GEOLOGICAL_COMPLEXITY_DATASET_SAMPLES,
        "GEOLOGICAL_COMPLEXITY_DATASET_SAMPLES",
    )

    # -----------------------------------------------------
    # Statistical settings
    # -----------------------------------------------------

    if not 0.0 < STATISTICAL_ALPHA < 1.0:
        raise ValueError(
            "STATISTICAL_ALPHA must be between 0 and 1."
        )

    if not isinstance(STATISTICAL_METRIC, str):
        raise TypeError(
            "STATISTICAL_METRIC must be a string."
        )

    if not isinstance(STATISTICAL_TEST, str):
        raise TypeError(
            "STATISTICAL_TEST must be a string."
        )

    # -----------------------------------------------------
    # Controlled matrix
    # -----------------------------------------------------

    if CONTROLLED_MATRIX_CASE_LIMIT is not None:
        _validate_positive_integer(
            CONTROLLED_MATRIX_CASE_LIMIT,
            "CONTROLLED_MATRIX_CASE_LIMIT",
        )

    # -----------------------------------------------------
    # Curvelet POCS
    # -----------------------------------------------------

    _validate_positive_integer(
        CURVELET_NUM_SCALES,
        "CURVELET_NUM_SCALES",
    )

    _validate_positive_integer(
        CURVELET_WEDGES_PER_DIRECTION,
        "CURVELET_WEDGES_PER_DIRECTION",
    )

    _validate_positive_integer(
        CURVELET_ITERATIONS,
        "CURVELET_ITERATIONS",
    )

    _validate_nonnegative_number(
        CURVELET_THRESHOLD,
        "CURVELET_THRESHOLD",
    )

    if not 0.0 < CURVELET_THRESHOLD_DECAY <= 1.0:
        raise ValueError(
            "CURVELET_THRESHOLD_DECAY must be greater "
            "than 0 and at most 1."
        )

    _validate_nonnegative_number(
        CURVELET_TOLERANCE,
        "CURVELET_TOLERANCE",
    )

    # -----------------------------------------------------
    # Strict acoustic-physics configuration
    # -----------------------------------------------------

    if not USE_PHYSICS_LOSS:
        raise ValueError(
            "This experiment requires USE_PHYSICS_LOSS=True."
        )

    if not USE_ACOUSTIC_WAVE_EQUATION_LOSS:
        raise ValueError(
            "The acoustic wave-equation loss must be enabled."
        )

    if PHYSICS_TYPE != "acoustic_wave_equation":
        raise ValueError(
            "PHYSICS_TYPE must be 'acoustic_wave_equation'."
        )

    if set(PHYSICS_LOSS_WEIGHTS) != {
        "acoustic_wave_equation"
    }:
        raise ValueError(
            "PHYSICS_LOSS_WEIGHTS must contain only "
            "'acoustic_wave_equation'."
        )

    for name, value in PHYSICS_LOSS_WEIGHTS.items():
        _validate_nonnegative_number(
            value,
            f"PHYSICS_LOSS_WEIGHTS['{name}']",
        )

    if PHYSICS_LOSS_WEIGHTS["acoustic_wave_equation"] <= 0:
        raise ValueError(
            "The acoustic wave-equation weight must be positive."
        )

    if LOSS_WEIGHTS.get("physics", 0.0) <= 0:
        raise ValueError(
            "LOSS_WEIGHTS['physics'] must be positive "
            "for acoustic physics training."
        )

    # -----------------------------------------------------
    # Acoustic simulation parameters
    # -----------------------------------------------------

    _validate_positive_number(ACOUSTIC_DT, "ACOUSTIC_DT")

    _validate_positive_number(ACOUSTIC_DX, "ACOUSTIC_DX")

    _validate_positive_number(ACOUSTIC_DY, "ACOUSTIC_DY")

    _validate_positive_number(ACOUSTIC_DZ, "ACOUSTIC_DZ")

    _validate_positive_integer(
        ACOUSTIC_TIME_STEPS,
        "ACOUSTIC_TIME_STEPS",
    )

    if ACOUSTIC_TIME_STEPS < 3:
        raise ValueError(
            "ACOUSTIC_TIME_STEPS must be at least 3 "
            "for the temporal second derivative."
        )

    _validate_positive_number(
        ACOUSTIC_SOURCE_FREQUENCY,
        "ACOUSTIC_SOURCE_FREQUENCY",
    )

    _validate_nonnegative_number(
        ACOUSTIC_SOURCE_AMPLITUDE,
        "ACOUSTIC_SOURCE_AMPLITUDE",
    )

    if not 0.0 < ACOUSTIC_CFL_SAFETY <= 1.0:
        raise ValueError(
            "ACOUSTIC_CFL_SAFETY must be in (0, 1]."
        )

    if ACOUSTIC_SNAPSHOT_INDEX is not None:
        if (
            not isinstance(ACOUSTIC_SNAPSHOT_INDEX, int)
            or isinstance(ACOUSTIC_SNAPSHOT_INDEX, bool)
            or not 0 <= ACOUSTIC_SNAPSHOT_INDEX
            < ACOUSTIC_TIME_STEPS
        ):
            raise ValueError(
                "ACOUSTIC_SNAPSHOT_INDEX must be None "
                "or an index within the simulated time range."
            )

    # -----------------------------------------------------
    # Composite-loss weights
    # -----------------------------------------------------

    required_loss_weights = {
        "mae",
        "physics",
        "uncertainty",
        "ssim",
    }

    if not required_loss_weights.issubset(LOSS_WEIGHTS):
        raise ValueError(
            "LOSS_WEIGHTS must include mae, physics, "
            "uncertainty, and ssim."
        )

    for name, value in LOSS_WEIGHTS.items():
        _validate_nonnegative_number(
            value,
            f"LOSS_WEIGHTS['{name}']",
        )

    # -----------------------------------------------------
    # General spatial sampling
    # -----------------------------------------------------

    _validate_positive_number(DX, "DX")
    _validate_positive_number(DY, "DY")
    _validate_positive_number(DZ, "DZ")

    # -----------------------------------------------------
    # Velocity
    # -----------------------------------------------------

    _validate_positive_number(VELOCITY_MIN, "VELOCITY_MIN")

    _validate_positive_number(VELOCITY_MAX, "VELOCITY_MAX")

    if VELOCITY_MAX <= VELOCITY_MIN:
        raise ValueError(
            "VELOCITY_MAX must exceed VELOCITY_MIN."
        )

    # -----------------------------------------------------
    # Seismic amplitude
    # -----------------------------------------------------

    if SEISMIC_AMPLITUDE_MAX <= SEISMIC_AMPLITUDE_MIN:
        raise ValueError(
            "SEISMIC_AMPLITUDE_MAX must exceed "
            "SEISMIC_AMPLITUDE_MIN."
        )

    if SEISMIC_DATA_RANGE <= 0:
        raise ValueError(
            "SEISMIC_DATA_RANGE must be positive."
        )

    # -----------------------------------------------------
    # Mask convention
    # -----------------------------------------------------

    if MASK_OBSERVED_VALUE == MASK_MISSING_VALUE:
        raise ValueError(
            "Observed and missing mask values must differ."
        )

    if (
        MASK_OBSERVED_VALUE != 1.0
        or MASK_MISSING_VALUE != 0.0
    ):
        raise ValueError(
            "Mask convention must remain 1.0=observed "
            "and 0.0=missing."
        )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    _validate_nonnegative_number(NUM_WORKERS, "NUM_WORKERS")

    if not isinstance(PIN_MEMORY, bool):
        raise TypeError("PIN_MEMORY must be boolean.")

    if not isinstance(PERSISTENT_WORKERS, bool):
        raise TypeError("PERSISTENT_WORKERS must be boolean.")

    if PERSISTENT_WORKERS and NUM_WORKERS == 0:
        raise ValueError(
            "PERSISTENT_WORKERS=True requires NUM_WORKERS > 0."
        )

    # -----------------------------------------------------
    # Device and mixed precision
    # -----------------------------------------------------

    if DEVICE not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            "DEVICE must be 'cpu', 'cuda', or 'auto'."
        )

    if not isinstance(USE_AMP, bool):
        raise TypeError("USE_AMP must be boolean.")

    # -----------------------------------------------------
    # Pipeline controls
    # -----------------------------------------------------

    for name, value in (
        ("RUN_TRAINING", RUN_TRAINING),
        ("RESUME_EVALUATION", RESUME_EVALUATION),
        ("FORCE_RERUN_EVALUATION", FORCE_RERUN_EVALUATION),
        ("USE_ATTENTION", USE_ATTENTION),
        ("USE_RESIDUAL", USE_RESIDUAL),
        ("USE_UNCERTAINTY", USE_UNCERTAINTY),
        ("VELOCITY_NORMALIZED", VELOCITY_NORMALIZED),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean.")

    # -----------------------------------------------------
    # Reproducibility and output
    # -----------------------------------------------------

    if not isinstance(SEED, int) or isinstance(SEED, bool):
        raise TypeError("SEED must be an integer.")

    if not isinstance(OUTPUT_ROOT, str) or not OUTPUT_ROOT.strip():
        raise ValueError(
            "OUTPUT_ROOT must be a non-empty string."
        )

    return True


# =========================================================
# 37. RUN CONFIGURATION VALIDATION
# =========================================================

validate_config()
