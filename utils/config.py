# =========================================================
# PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION
# GLOBAL CONFIGURATION FILE
# =========================================================

"""
Global configuration for the complete
Physics-Informed 3D Seismic Reconstruction project.

Research framework
------------------

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data
Reconstruction in Complex Geological Settings.

This file centralizes all major project parameters.

Configuration categories
------------------------

1. Dataset configuration
2. Synthetic dataset configuration
3. F3 dataset configuration
4. Common baseline configuration
5. Training configuration
6. Validation configuration
7. Checkpoint configuration
8. Early-stopping configuration
9. Uncertainty evaluation
10. Robustness evaluation
11. Statistical significance
12. Geological complexity evaluation
13. Final reporting
14. Master pipeline control
15. Controlled experimental matrix
16. Curvelet POCS configuration
17. Model configuration
18. Physics configuration
19. Composite-loss weights
20. Physics-loss weights
21. Physical seismic sampling
22. Travel-time configuration
23. Velocity-model configuration
24. Seismic amplitude normalization
25. Mask configuration
26. DataLoader configuration
27. Mixed-precision configuration
28. Device configuration
29. Output configuration
30. Reproducibility
31. Configuration validation

Tensor convention
-----------------

All seismic volumes use:

    [B, C, D, H, W]

where:

    B = batch size
    C = channel
    D = depth
    H = crossline
    W = inline

Individual dataset samples use:

    [C, D, H, W]

Important physics convention
----------------------------

The Eikonal equation is:

    |∇T|² = 1 / V²

where:

    T = seismic travel time [s]
    V = P-wave velocity [m/s]

The velocity model supplied to the physics loss must therefore
represent physically meaningful P-wave velocity.

Author: Ormin Joseph
"""


# =========================================================
# 1. DATASET MODE
# =========================================================

DATASET_MODE = "synthetic"

# Supported modes:
#
# DATASET_MODE = "synthetic"
# DATASET_MODE = "f3"


# =========================================================
# 2. EXPERIMENT NAME
# =========================================================

EXPERIMENT_NAME = "synthetic_training"


# =========================================================
# 3. SYNTHETIC DATASET CONFIGURATION
# =========================================================

# Number of synthetic seismic volumes.

SYNTHETIC_NUM_SAMPLES = 50


# Synthetic seismic cube dimensions:
#
#     [Depth, Crossline, Inline]

SYNTHETIC_PATCH_SIZE = (
    64,
    128,
    128,
)


# Probability that seismic samples/voxels are removed.

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


# F3 patch dimensions:
#
#     [Depth, Crossline, Inline]

F3_PATCH_SIZE = (
    64,
    64,
    64,
)


# F3 patch extraction stride:
#
#     [Depth, Crossline, Inline]

F3_STRIDE = (
    64,
    64,
    64,
)


# Missing-data probability for F3 experiments.

F3_MISSING_PROBABILITY = 0.30


# =========================================================
# 5. COMMON BASELINE COMPARISON CONFIGURATION
# =========================================================

# Number of samples used in the common seven-method
# reconstruction comparison.

BASELINE_NUM_SAMPLES = 1


# Controlled benchmark cube:
#
#     [Depth, Crossline, Inline]

BASELINE_CUBE_SIZE = (
    64,
    128,
    128,
)


# Missing-data probability.

BASELINE_MISSING_RATE = 0.30


# Geological structure.

BASELINE_GEOLOGICAL_MODE = "folded"


# Missing-data mechanism.

BASELINE_MASK_MODE = "missing_crosslines"


# Reproducibility seed.

BASELINE_SEED = 42


# Maximum acceptable observed-data preservation error.

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

# Number of patches/samples used for uncertainty evaluation.
#
# None = evaluate all available samples.
# Integer = evaluate the specified number.

UNCERTAINTY_EVALUATION_NUM_SAMPLES = None


# Deterministic uncertainty-evaluation seed.

UNCERTAINTY_EVALUATION_SEED = 42


# Maximum number of voxel pairs used for uncertainty/error
# correlation analysis.
#
# None = use all eligible voxels.

UNCERTAINTY_CORRELATION_MAX_VOXELS = 500000


# Evaluate uncertainty separately in missing regions.

EVALUATE_MISSING_REGION_UNCERTAINTY = True


# Evaluate uncertainty separately in observed regions.

EVALUATE_OBSERVED_REGION_UNCERTAINTY = True


# Number of dataset patches used for uncertainty-error
# correlation analysis.
#
# None = analyse all available patches.

UNCERTAINTY_ERROR_CORRELATION_NUM_PATCHES = None


# Reproducibility seed for uncertainty-error correlation.

UNCERTAINTY_ERROR_CORRELATION_SEED = 42


# =========================================================
# 11. NOISE ROBUSTNESS EVALUATION
# =========================================================

# Gaussian noise standard deviations in the normalized
# seismic amplitude scale.

NOISE_ROBUSTNESS_LEVELS = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
)


# Number of samples evaluated at each noise level.
#
# None = evaluate all available samples.

NOISE_ROBUSTNESS_NUM_SAMPLES = 20


# Reproducibility seed.

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


# Number of samples evaluated at each missing-data level.

MISSING_DATA_ROBUSTNESS_NUM_SAMPLES = 20


# Reproducibility seed.

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

# Increasing structural complexity.

GEOLOGICAL_COMPLEXITY_LEVELS = (
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
)


# Number of independently masked test cases per structure.

GEOLOGICAL_COMPLEXITY_NUM_SAMPLES = 5


# Missing-data probability.

GEOLOGICAL_COMPLEXITY_MISSING_PROBABILITY = 0.30


# Number of patches evaluated for non-synthetic datasets.

GEOLOGICAL_COMPLEXITY_DATASET_SAMPLES = 20


# Reproducibility seed.

GEOLOGICAL_COMPLEXITY_SEED = 42


# =========================================================
# 15. FINAL REPORT CONFIGURATION
# =========================================================

FINAL_REPORT_FILENAME = "final_report.txt"

FINAL_REPORT_METADATA_FILENAME = (
    "final_report_metadata.json"
)


# =========================================================
# 16. EVALUATION SETTINGS
# =========================================================

GALLERY_NUMBER_OF_SAMPLES = 5


# =========================================================
# 17. MASTER PIPELINE CONTROL
# =========================================================

# ---------------------------------------------------------
# TRAINING CONTROL
# ---------------------------------------------------------

RUN_TRAINING = True


# ---------------------------------------------------------
# EVALUATION CONTROL
# ---------------------------------------------------------

RESUME_EVALUATION = True

FORCE_RERUN_EVALUATION = False


# =========================================================
# 18. CONTROLLED EXPERIMENTAL MATRIX
# =========================================================

"""
Complete controlled experimental matrix:

    6 geological structures
    × 5 missing mechanisms
    × 5 missing-data rates
    × 5 random seeds

Total:

    6 × 5 × 5 × 5 = 750 cases

During development:

    CONTROLLED_MATRIX_CASE_LIMIT = 10

For the final PhD experiment:

    CONTROLLED_MATRIX_CASE_LIMIT = None
"""

CONTROLLED_MATRIX_CASE_LIMIT = 50


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

# Attention gates.

USE_ATTENTION = True


# Residual connections.

USE_RESIDUAL = True


# Predictive uncertainty.

USE_UNCERTAINTY = True


# Number of stochastic forward passes for MC Dropout.

MC_DROPOUT_SAMPLES = 20


# =========================================================
# 21. UNCERTAINTY MODEL PARAMETERS
# =========================================================

# Lower and upper limits for predicted log variance.

LOG_VARIANCE_MIN = -10.0

LOG_VARIANCE_MAX = 10.0


# =========================================================
# 22. PHYSICS-INFORMED CONFIGURATION
# =========================================================

USE_PHYSICS_LOSS = True


# Source coordinates are currently unavailable.

USE_SOURCE_LOSS = False


# Independent travel-time targets are currently unavailable.

USE_TRAVEL_TIME_LOSS = False


# Eikonal physics remains active.

USE_EIKONAL_LOSS = True


# =========================================================
# 23. COMPOSITE LOSS WEIGHTS
# =========================================================

"""
Global composite loss:

    L_total =
        λ_mae L_mae
        +
        λ_physics L_physics
        +
        λ_uncertainty L_uncertainty
        +
        λ_ssim L_ssim
"""

LOSS_WEIGHTS = {
    "mae": 1.0,
    "physics": 0.10,
    "uncertainty": 0.01,
    "ssim": 0.10,
}


# =========================================================
# 24. PHYSICS-LOSS COMPONENT WEIGHTS
# =========================================================

"""
Physics loss:

    L_physics =
        λ_eikonal L_eikonal
        +
        λ_source L_source
        +
        λ_travel_time L_travel_time
"""

PHYSICS_LOSS_WEIGHTS = {
    "eikonal": 1.0,
    "source": 1.0,
    "travel_time": 1.0,
}


# =========================================================
# 25. PHYSICAL SEISMIC SAMPLING
# =========================================================

# Inline spacing [m].

DX = 1.0


# Crossline spacing [m].

DY = 1.0


# Depth spacing [m].

DZ = 1.0


# =========================================================
# 26. TRAVEL-TIME CONFIGURATION
# =========================================================

TRAVEL_TIME_SCALE = 0.1


# =========================================================
# 27. VELOCITY MODEL CONFIGURATION
# =========================================================

"""
Physical P-wave velocity:

    V [m/s]

If velocity is normalized to [0,1]:

    VELOCITY_NORMALIZED = True

Otherwise:

    VELOCITY_NORMALIZED = False
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
    SEISMIC_AMPLITUDE_MAX
    - SEISMIC_AMPLITUDE_MIN
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

    "cpu"
        Force CPU.

    "cuda"
        Require CUDA.

    "auto"
        Use CUDA when available; otherwise CPU.
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
# 35. CONFIGURATION VALIDATION
# =========================================================

def _validate_positive_integer(
    value,
    name,
):
    """
    Validate a positive integer configuration value.
    """

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _validate_nonnegative_number(
    value,
    name,
):
    """
    Validate a non-negative numeric value.
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
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative number."
        )


def _validate_probability(
    value,
    name,
):
    """
    Validate probability using the project-wide convention:

        0.0 <= p < 1.0
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
            f"{name} must satisfy "
            "0.0 <= value < 1.0."
        )


def _validate_spatial_size(
    value,
    name,
):
    """
    Validate a spatial dimension tuple:

        (Depth, Crossline, Inline)
    """

    if not isinstance(
        value,
        (tuple, list),
    ):
        raise TypeError(
            f"{name} must be a tuple or list."
        )

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly "
            "three dimensions."
        )

    for dimension in value:

        if (
            not isinstance(
                dimension,
                int,
            )
            or isinstance(
                dimension,
                bool,
            )
            or dimension <= 0
        ):
            raise ValueError(
                f"All dimensions in {name} must "
                "be positive integers."
            )


def _validate_stride(
    stride,
    patch_size,
    name,
):
    """
    Validate patch extraction stride.

    The present project uses contiguous/non-overlapping
    extraction, so stride values larger than the corresponding
    patch dimensions are rejected.
    """

    _validate_spatial_size(
        stride,
        name,
    )

    for stride_value, patch_value in zip(
        stride,
        patch_size,
    ):

        if stride_value > patch_value:

            raise ValueError(
                f"{name} cannot exceed the corresponding "
                f"patch dimension. "
                f"Received stride={stride_value}, "
                f"patch={patch_value}."
            )


def validate_config():
    """
    Validate the complete global configuration.

    This function should be executed before training,
    evaluation, or controlled experiments.
    """

    # =====================================================
    # DATASET MODE
    # =====================================================

    if not isinstance(
        DATASET_MODE,
        str,
    ):
        raise TypeError(
            "DATASET_MODE must be a string."
        )

    if DATASET_MODE not in {
        "synthetic",
        "f3",
    }:
        raise ValueError(
            "DATASET_MODE must be either "
            "'synthetic' or 'f3'."
        )

    # =====================================================
    # EXPERIMENT NAME
    # =====================================================

    if not isinstance(
        EXPERIMENT_NAME,
        str,
    ) or not EXPERIMENT_NAME.strip():
        raise ValueError(
            "EXPERIMENT_NAME must be a non-empty string."
        )

    # =====================================================
    # SYNTHETIC DATASET
    # =====================================================

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

    # =====================================================
    # F3 DATASET
    # =====================================================

    _validate_spatial_size(
        F3_PATCH_SIZE,
        "F3_PATCH_SIZE",
    )

    _validate_stride(
        F3_STRIDE,
        F3_PATCH_SIZE,
        "F3_STRIDE",
    )

    _validate_probability(
        F3_MISSING_PROBABILITY,
        "F3_MISSING_PROBABILITY",
    )

    if not isinstance(
        F3_PATH,
        str,
    ) or not F3_PATH.strip():

        raise ValueError(
            "F3_PATH must be a non-empty string."
        )

    # =====================================================
    # BASELINE CONFIGURATION
    # =====================================================

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

    if not isinstance(
        BASELINE_GEOLOGICAL_MODE,
        str,
    ):
        raise TypeError(
            "BASELINE_GEOLOGICAL_MODE must be a string."
        )

    if not isinstance(
        BASELINE_MASK_MODE,
        str,
    ):
        raise TypeError(
            "BASELINE_MASK_MODE must be a string."
        )

    _validate_nonnegative_number(
        OBSERVED_PRESERVATION_TOLERANCE,
        "OBSERVED_PRESERVATION_TOLERANCE",
    )

    # =====================================================
    # TRAINING
    # =====================================================

    _validate_positive_integer(
        BATCH_SIZE,
        "BATCH_SIZE",
    )

    _validate_positive_integer(
        NUM_EPOCHS,
        "NUM_EPOCHS",
    )

    if LEARNING_RATE <= 0:
        raise ValueError(
            "LEARNING_RATE must be greater than zero."
        )

    _validate_nonnegative_number(
        WEIGHT_DECAY,
        "WEIGHT_DECAY",
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    if not (
        0.0
        < VALIDATION_SPLIT
        < 1.0
    ):
        raise ValueError(
            "VALIDATION_SPLIT must be greater than "
            "0 and less than 1."
        )

    # =====================================================
    # CHECKPOINTS / EARLY STOPPING
    # =====================================================

    _validate_positive_integer(
        SAVE_EVERY,
        "SAVE_EVERY",
    )

    _validate_positive_integer(
        PATIENCE,
        "PATIENCE",
    )

    # =====================================================
    # UNCERTAINTY
    # =====================================================

    _validate_positive_integer(
        MC_DROPOUT_SAMPLES,
        "MC_DROPOUT_SAMPLES",
    )

    if LOG_VARIANCE_MAX <= LOG_VARIANCE_MIN:
        raise ValueError(
            "LOG_VARIANCE_MAX must be greater than "
            "LOG_VARIANCE_MIN."
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

    # =====================================================
    # ROBUSTNESS LEVELS
    # =====================================================

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

    # =====================================================
    # GEOLOGICAL COMPLEXITY
    # =====================================================

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

    # =====================================================
    # STATISTICS
    # =====================================================

    if not (
        0.0
        < STATISTICAL_ALPHA
        < 1.0
    ):
        raise ValueError(
            "STATISTICAL_ALPHA must be between 0 and 1."
        )

    if not isinstance(
        STATISTICAL_METRIC,
        str,
    ):
        raise TypeError(
            "STATISTICAL_METRIC must be a string."
        )

    if not isinstance(
        STATISTICAL_TEST,
        str,
    ):
        raise TypeError(
            "STATISTICAL_TEST must be a string."
        )

    # =====================================================
    # CONTROLLED MATRIX
    # =====================================================

    if (
        CONTROLLED_MATRIX_CASE_LIMIT is not None
        and (
            not isinstance(
                CONTROLLED_MATRIX_CASE_LIMIT,
                int,
            )
            or isinstance(
                CONTROLLED_MATRIX_CASE_LIMIT,
                bool,
            )
            or CONTROLLED_MATRIX_CASE_LIMIT < 1
        )
    ):
        raise ValueError(
            "CONTROLLED_MATRIX_CASE_LIMIT must be "
            "None or a positive integer."
        )

    # =====================================================
    # CURVELET
    # =====================================================

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

    if not (
        0.0
        < CURVELET_THRESHOLD_DECAY
        <= 1.0
    ):
        raise ValueError(
            "CURVELET_THRESHOLD_DECAY must be "
            "greater than 0 and at most 1."
        )

    _validate_nonnegative_number(
        CURVELET_TOLERANCE,
        "CURVELET_TOLERANCE",
    )

    # =====================================================
    # PHYSICS
    # =====================================================

    if USE_SOURCE_LOSS:
        raise ValueError(
            "USE_SOURCE_LOSS=True is not supported by "
            "the current dataset because valid source "
            "coordinates are not available."
        )

    if USE_TRAVEL_TIME_LOSS:
        raise ValueError(
            "USE_TRAVEL_TIME_LOSS=True is not supported "
            "because independent travel-time targets are "
            "not currently available."
        )

    if not USE_PHYSICS_LOSS:

        if USE_EIKONAL_LOSS:
            raise ValueError(
                "USE_EIKONAL_LOSS cannot be True when "
                "USE_PHYSICS_LOSS is False."
            )

        if USE_SOURCE_LOSS:
            raise ValueError(
                "USE_SOURCE_LOSS cannot be True when "
                "USE_PHYSICS_LOSS is False."
            )

        if USE_TRAVEL_TIME_LOSS:
            raise ValueError(
                "USE_TRAVEL_TIME_LOSS cannot be True "
                "when USE_PHYSICS_LOSS is False."
            )

    # =====================================================
    # LOSS WEIGHTS
    # =====================================================

    for name, value in LOSS_WEIGHTS.items():

        _validate_nonnegative_number(
            value,
            f"LOSS_WEIGHTS['{name}']",
        )

    for name, value in PHYSICS_LOSS_WEIGHTS.items():

        _validate_nonnegative_number(
            value,
            f"PHYSICS_LOSS_WEIGHTS['{name}']",
        )

    # =====================================================
    # PHYSICAL SAMPLING
    # =====================================================

    if DX <= 0:
        raise ValueError(
            "DX must be greater than zero."
        )

    if DY <= 0:
        raise ValueError(
            "DY must be greater than zero."
        )

    if DZ <= 0:
        raise ValueError(
            "DZ must be greater than zero."
        )

    if TRAVEL_TIME_SCALE <= 0:
        raise ValueError(
            "TRAVEL_TIME_SCALE must be greater than zero."
        )

    # =====================================================
    # VELOCITY
    # =====================================================

    if VELOCITY_MIN <= 0:
        raise ValueError(
            "VELOCITY_MIN must be greater than zero."
        )

    if VELOCITY_MAX <= VELOCITY_MIN:
        raise ValueError(
            "VELOCITY_MAX must be greater than "
            "VELOCITY_MIN."
        )

    # =====================================================
    # AMPLITUDE
    # =====================================================

    if (
        SEISMIC_AMPLITUDE_MAX
        <= SEISMIC_AMPLITUDE_MIN
    ):
        raise ValueError(
            "SEISMIC_AMPLITUDE_MAX must be greater than "
            "SEISMIC_AMPLITUDE_MIN."
        )

    if SEISMIC_DATA_RANGE <= 0:
        raise ValueError(
            "SEISMIC_DATA_RANGE must be greater than zero."
        )

    # =====================================================
    # MASK
    # =====================================================

    if (
        MASK_OBSERVED_VALUE
        == MASK_MISSING_VALUE
    ):
        raise ValueError(
            "MASK_OBSERVED_VALUE and "
            "MASK_MISSING_VALUE must be different."
        )

    if (
        MASK_OBSERVED_VALUE != 1.0
        or MASK_MISSING_VALUE != 0.0
    ):
        raise ValueError(
            "The project-wide mask convention must remain: "
            "1.0 = observed and 0.0 = missing."
        )

    # =====================================================
    # DATALOADER
    # =====================================================

    _validate_nonnegative_number(
        NUM_WORKERS,
        "NUM_WORKERS",
    )

    if not isinstance(
        PIN_MEMORY,
        bool,
    ):
        raise TypeError(
            "PIN_MEMORY must be boolean."
        )

    if not isinstance(
        PERSISTENT_WORKERS,
        bool,
    ):
        raise TypeError(
            "PERSISTENT_WORKERS must be boolean."
        )

    if (
        PERSISTENT_WORKERS
        and NUM_WORKERS == 0
    ):
        raise ValueError(
            "PERSISTENT_WORKERS=True requires "
            "NUM_WORKERS > 0."
        )

    # =====================================================
    # DEVICE
    # =====================================================

    if DEVICE not in {
        "cpu",
        "cuda",
        "auto",
    }:
        raise ValueError(
            "DEVICE must be 'cpu', 'cuda', or 'auto'."
        )

    # =====================================================
    # AMP
    # =====================================================

    if not isinstance(
        USE_AMP,
        bool,
    ):
        raise TypeError(
            "USE_AMP must be boolean."
        )

    # IMPORTANT:
    #
    # DEVICE='auto' is allowed with AMP because the
    # runtime may resolve automatically to CUDA.
    #
    # The Trainer/device utility is responsible for
    # enabling AMP only when CUDA is actually active.

    # =====================================================
    # PIPELINE CONTROLS
    # =====================================================

    if not isinstance(
        RUN_TRAINING,
        bool,
    ):
        raise TypeError(
            "RUN_TRAINING must be boolean."
        )

    if not isinstance(
        RESUME_EVALUATION,
        bool,
    ):
        raise TypeError(
            "RESUME_EVALUATION must be boolean."
        )

    if not isinstance(
        FORCE_RERUN_EVALUATION,
        bool,
    ):
        raise TypeError(
            "FORCE_RERUN_EVALUATION must be boolean."
        )

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================

    if (
        not isinstance(
            SEED,
            int,
        )
        or isinstance(
            SEED,
            bool,
        )
    ):
        raise TypeError(
            "SEED must be an integer."
        )

    # =====================================================
    # OUTPUT DIRECTORIES
    # =====================================================

    if not isinstance(
        OUTPUT_ROOT,
        str,
    ) or not OUTPUT_ROOT.strip():

        raise ValueError(
            "OUTPUT_ROOT must be a non-empty string."
        )

    return True


# =========================================================
# RUN CONFIGURATION VALIDATION
# =========================================================

validate_config()