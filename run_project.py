"""
======================================================================
MASTER PROJECT PIPELINE
======================================================================

Physics-Informed 3D Encoder–Decoder Framework with Predictive
Uncertainty for Seismic Data Reconstruction in Complex Geological
Settings

MASTER PROJECT ORCHESTRATOR

Purpose
-------
This script is the top-level controller for the complete research
pipeline.

The master pipeline coordinates:

    1. Configuration validation
    2. Final model training / checkpoint resume
    3. Best-model checkpoint verification
    4. Complete model evaluation
    5. Reconstruction gallery generation
    6. Uncertainty analysis
    7. Uncertainty evaluation
    8. Uncertainty statistics
    9. Common seven-method reconstruction comparison
   10. Controlled classical-baseline experiments
   11. Controlled proposed-model experiment
   12. Ablation study
   13. Statistical significance analysis
   14. Thesis tables
   15. Final report

IMPORTANT
---------
This file is an ORCHESTRATOR.

Detailed scientific computation belongs to the appropriate module.

This file must NOT duplicate:

    - model architecture
    - loss calculations
    - dataset generation
    - baseline algorithms
    - uncertainty calculations
    - metric calculations
    - ablation calculations
    - statistical tests
    - thesis-table calculations

The master pipeline determines only:

    what should run,
    in what order,
    and whether required outputs exist.

TRAINING
--------
Only ONE final proposed model is trained.

The training module is responsible for:

    - dataset creation
    - dataset splitting
    - model construction
    - loss construction
    - optimizer
    - checkpoint saving
    - checkpoint resume
    - final best-model selection

The master pipeline does not duplicate any of these operations.

EVALUATION
----------
Evaluation begins only after a valid:

    best_model.pth

has been verified.

The controlled evaluation matrix is NOT used to train
750 different models.

Only ONE final model is trained.

The complete controlled matrix is defined by the
evaluation modules.

Current project structure:

    Training
        |
        v
    train_final_model()
        |
        v
    best_model.pth
        |
        v
    Complete evaluation
        |
        v
    Case-level results
        |
        v
    Statistical analysis
        |
        v
    Thesis tables
        |
        v
    Final report

DATASET MODE
------------
All output locations are derived from:

    config.DATASET_MODE
    config.EXPERIMENT_NAME
    config.CHECKPOINT_DIR
    config.FIGURE_DIR
    config.REPORT_DIR

Therefore this master pipeline does not hard-code:

    outputs/synthetic_training/

or any other experiment-specific path.

Author: Ormin Joseph
======================================================================
"""


# ======================================================================
# STANDARD LIBRARY
# ======================================================================

from pathlib import Path
import sys
import traceback


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

from utils import config


# ======================================================================
# PROJECT ROOT
# ======================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent
)


# ======================================================================
# BEST CHECKPOINT
# ======================================================================
#
# The checkpoint location is derived entirely from the centralized
# configuration.
#
# For the current synthetic experiment this resolves to:
#
#     outputs/synthetic_training/checkpoints/best_model.pth
#
# but the master pipeline does not hard-code that path.
# ======================================================================

BEST_CHECKPOINT = (
    Path(config.CHECKPOINT_DIR)
    / "best_model.pth"
)


# ======================================================================
# PIPELINE RESULTS
# ======================================================================

PIPELINE_RESULTS = {
    "configuration": False,
    "training": False,
    "checkpoint": False,
    "evaluation": False,
    "final_outputs": False,
}


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(
    title: str,
) -> None:
    """
    Print a clearly separated pipeline section header.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


def print_success(
    message: str,
) -> None:
    """
    Print a successful pipeline message.
    """

    print(
        f"[PASS] {message}"
    )


def print_warning(
    message: str,
) -> None:
    """
    Print a warning without terminating the pipeline.
    """

    print(
        f"[WARNING] {message}"
    )


def print_failure(
    message: str,
) -> None:
    """
    Print a pipeline failure message.
    """

    print(
        f"[FAIL] {message}"
    )


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_configuration() -> bool:
    """
    Validate the minimum configuration required by the
    master pipeline.

    Detailed scientific/configuration validation remains
    the responsibility of utils.config.
    """

    print_header(
        "MASTER PIPELINE CONFIGURATION VALIDATION"
    )

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------

    if not isinstance(
        config.EXPERIMENT_NAME,
        str,
    ):

        raise RuntimeError(
            "EXPERIMENT_NAME must be a string."
        )

    if not config.EXPERIMENT_NAME.strip():

        raise RuntimeError(
            "EXPERIMENT_NAME must not be empty."
        )

    # ------------------------------------------------------------------
    # Dataset mode
    # ------------------------------------------------------------------

    if not isinstance(
        config.DATASET_MODE,
        str,
    ):

        raise RuntimeError(
            "DATASET_MODE must be a string."
        )

    if not config.DATASET_MODE.strip():

        raise RuntimeError(
            "DATASET_MODE must not be empty."
        )

    # ------------------------------------------------------------------
    # Device policy
    # ------------------------------------------------------------------

    if not isinstance(
        config.DEVICE,
        str,
    ):

        raise RuntimeError(
            "DEVICE must be a string."
        )

    device_policy = (
        config.DEVICE
        .lower()
        .strip()
    )

    if device_policy not in {
        "auto",
        "cpu",
        "cuda",
    }:

        raise RuntimeError(
            "DEVICE must be one of: "
            "'auto', 'cpu', or 'cuda'."
        )

    # ------------------------------------------------------------------
    # Controlled evaluation case limit
    # ------------------------------------------------------------------
    #
    # None:
    #     complete controlled matrix.
    #
    # Integer:
    #     limited smoke-test run.
    #
    # The master pipeline does not define the experimental factors.
    # Those belong to the evaluation modules.
    # ------------------------------------------------------------------

    case_limit = (
        config.CONTROLLED_MATRIX_CASE_LIMIT
    )

    if case_limit is not None:

        if not isinstance(
            case_limit,
            int,
        ):

            raise RuntimeError(
                "CONTROLLED_MATRIX_CASE_LIMIT must "
                "be an integer or None."
            )

        if case_limit <= 0:

            raise RuntimeError(
                "CONTROLLED_MATRIX_CASE_LIMIT must "
                "be greater than zero or None."
            )

    # ------------------------------------------------------------------
    # Create the required project directories.
    # ------------------------------------------------------------------

    checkpoint_directory = Path(
        config.CHECKPOINT_DIR
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_directory = Path(
        config.FIGURE_DIR
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory = Path(
        config.REPORT_DIR
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Display configuration.
    # ------------------------------------------------------------------

    print(
        f"Experiment              : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode             : "
        f"{config.DATASET_MODE}"
    )

    print(
        f"Configured device        : "
        f"{config.DEVICE}"
    )

    print(
        f"Run training             : "
        f"{config.RUN_TRAINING}"
    )

    print(
        f"Resume evaluation        : "
        f"{config.RESUME_EVALUATION}"
    )

    print(
        f"Force rerun evaluation   : "
        f"{config.FORCE_RERUN_EVALUATION}"
    )

    print(
        "Controlled matrix limit  : "
        f"{case_limit}"
    )

    print(
        f"Best checkpoint          : "
        f"{BEST_CHECKPOINT}"
    )

    print_success(
        "Master pipeline configuration is valid."
    )

    PIPELINE_RESULTS[
        "configuration"
    ] = True

    return True


# ======================================================================
# CHECKPOINT VERIFICATION
# ======================================================================

def verify_checkpoint() -> bool:
    """
    Verify that the final best-model checkpoint exists.

    Evaluation always uses:

        best_model.pth

    rather than an arbitrary latest checkpoint.
    """

    print_header(
        "CHECKPOINT VERIFICATION"
    )

    checkpoint = Path(
        BEST_CHECKPOINT
    )

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    if not checkpoint.exists():

        raise FileNotFoundError(
            "Required best-model checkpoint was not found:\n"
            f"{checkpoint}\n\n"
            "Train the final model first by setting "
            "RUN_TRAINING=True."
        )

    # ------------------------------------------------------------------
    # File-size check
    # ------------------------------------------------------------------

    checkpoint_size = (
        checkpoint.stat().st_size
    )

    if checkpoint_size <= 0:

        raise RuntimeError(
            "The best-model checkpoint exists but is empty:\n"
            f"{checkpoint}"
        )

    # ------------------------------------------------------------------
    # Display checkpoint information.
    # ------------------------------------------------------------------

    print(
        "Checkpoint found:"
    )

    print(
        f"    {checkpoint}"
    )

    print(
        "Checkpoint size:"
    )

    print(
        f"    {checkpoint_size:,} bytes"
    )

    print_success(
        "Best-model checkpoint verified."
    )

    PIPELINE_RESULTS[
        "checkpoint"
    ] = True

    return True


# ======================================================================
# TRAINING STAGE
# ======================================================================

def run_training_stage() -> bool:
    """
    Execute the final proposed-model training stage.

    The actual training implementation remains entirely inside:

        train.train_model.train_final_model()

    This function does NOT duplicate:

        - dataset construction
        - model construction
        - loss construction
        - optimizer construction
        - checkpoint loading
        - checkpoint saving
        - resume logic
        - epoch handling

    The current Trainer already handles checkpoint resume correctly.

    Therefore, if training is interrupted after a valid checkpoint
    has been created, calling train_final_model() again allows the
    existing training/checkpoint mechanism to resume from that state.
    """

    print_header(
        "TRAINING STAGE"
    )

    # ------------------------------------------------------------------
    # Training disabled.
    # ------------------------------------------------------------------

    if not config.RUN_TRAINING:

        print(
            "RUN_TRAINING=False"
        )

        print(
            "Training stage skipped."
        )

        print(
            "The existing best_model.pth will be used."
        )

        PIPELINE_RESULTS[
            "training"
        ] = True

        return True

    # ------------------------------------------------------------------
    # Lazy import.
    #
    # This prevents unnecessary training-module initialization when
    # the master pipeline is being used only for evaluation.
    # ------------------------------------------------------------------

    from train.train_model import (
        train_final_model
    )

    # ------------------------------------------------------------------
    # Execute final model training.
    #
    # IMPORTANT:
    #
    # train_final_model() owns the resume mechanism.
    #
    # The master pipeline does not pass an epoch number or manually
    # manipulate checkpoint state.
    # ------------------------------------------------------------------

    train_final_model()

    print_success(
        "Final model training stage completed."
    )

    PIPELINE_RESULTS[
        "training"
    ] = True

    return True


# ======================================================================
# EVALUATION STAGE
# ======================================================================

def run_evaluation_stage() -> bool:
    """
    Execute the complete downstream evaluation pipeline.

    The detailed scientific sequence remains inside:

        evaluation.run_full_evaluation

    The master pipeline deliberately does not duplicate individual
    evaluation modules.
    """

    print_header(
        "COMPLETE EVALUATION STAGE"
    )

    # ------------------------------------------------------------------
    # Evaluation cannot begin without a valid best checkpoint.
    # ------------------------------------------------------------------

    verify_checkpoint()

    # ------------------------------------------------------------------
    # Lazy import of the evaluation controller.
    # ------------------------------------------------------------------

    from evaluation.run_full_evaluation import (
        run_full_evaluation
    )

    # ------------------------------------------------------------------
    # Execute the complete evaluation sequence.
    #
    # We deliberately do not pass RESUME_EVALUATION or
    # FORCE_RERUN_EVALUATION as arguments here because the exact
    # interface of run_full_evaluation() belongs to that module.
    #
    # Those configuration values should only be wired into this
    # function after its actual interface has been verified.
    # ------------------------------------------------------------------

    run_full_evaluation()

    print_success(
        "Complete evaluation pipeline finished."
    )

    PIPELINE_RESULTS[
        "evaluation"
    ] = True

    return True


# ======================================================================
# FINAL OUTPUT VERIFICATION
# ======================================================================

def verify_final_outputs() -> bool:
    """
    Perform final master-level output verification.

    Individual scientific evaluation modules remain responsible
    for validating their detailed result files.

    The master pipeline verifies only critical project-level
    artifacts:

        1. best_model.pth exists and is non-empty.
        2. evaluation completed successfully.
        3. figure directory exists.
        4. report directory exists.

    The master pipeline deliberately does not assume a particular
    final-report filename.
    """

    print_header(
        "FINAL OUTPUT VERIFICATION"
    )

    # ------------------------------------------------------------------
    # Verify checkpoint.
    # ------------------------------------------------------------------

    checkpoint = Path(
        BEST_CHECKPOINT
    )

    if not checkpoint.exists():

        raise RuntimeError(
            "Final best-model checkpoint is missing:\n"
            f"{checkpoint}"
        )

    if checkpoint.stat().st_size <= 0:

        raise RuntimeError(
            "Final best-model checkpoint is empty:\n"
            f"{checkpoint}"
        )

    # ------------------------------------------------------------------
    # Verify evaluation.
    # ------------------------------------------------------------------

    if not PIPELINE_RESULTS[
        "evaluation"
    ]:

        raise RuntimeError(
            "Evaluation stage did not complete successfully."
        )

    # ------------------------------------------------------------------
    # Verify figure directory.
    # ------------------------------------------------------------------

    figure_directory = Path(
        config.FIGURE_DIR
    )

    if not figure_directory.exists():

        raise RuntimeError(
            "Figure directory does not exist:\n"
            f"{figure_directory}"
        )

    # ------------------------------------------------------------------
    # Verify report directory.
    # ------------------------------------------------------------------

    report_directory = Path(
        config.REPORT_DIR
    )

    if not report_directory.exists():

        raise RuntimeError(
            "Report directory does not exist:\n"
            f"{report_directory}"
        )

    # ------------------------------------------------------------------
    # Display successful verification.
    # ------------------------------------------------------------------

    print(
        "Best-model checkpoint verified."
    )

    print(
        "Evaluation stage verified."
    )

    print(
        "Figure directory verified:"
    )

    print(
        f"    {figure_directory}"
    )

    print(
        "Report directory verified:"
    )

    print(
        f"    {report_directory}"
    )

    print_success(
        "Master-level final output verification passed."
    )

    PIPELINE_RESULTS[
        "final_outputs"
    ] = True

    return True


# ======================================================================
# PIPELINE SUMMARY
# ======================================================================

def print_pipeline_summary() -> None:
    """
    Print the final status of each master-pipeline stage.
    """

    print_header(
        "MASTER PIPELINE SUMMARY"
    )

    status_names = {
        "configuration":
            "Configuration",

        "training":
            "Training",

        "checkpoint":
            "Checkpoint",

        "evaluation":
            "Evaluation",

        "final_outputs":
            "Final outputs",
    }

    for key, label in status_names.items():

        status = PIPELINE_RESULTS.get(
            key,
            False,
        )

        if status:

            print(
                f"[PASS] {label}"
            )

        else:

            print(
                f"[FAIL] {label}"
            )

    print()

    print(
        f"Experiment : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset    : "
        f"{config.DATASET_MODE}"
    )

    print(
        f"Checkpoint : "
        f"{BEST_CHECKPOINT}"
    )

    print()


# ======================================================================
# MASTER PROJECT RUNNER
# ======================================================================

def run_project() -> bool:
    """
    Execute the complete research project pipeline.

    Sequence
    --------
    1. Validate configuration.
    2. Train final model if requested.
    3. Verify best checkpoint.
    4. Execute complete evaluation.
    5. Verify project-level outputs.
    6. Print final summary.

    Returns
    -------
    bool
        True when the complete pipeline succeeds.
        False when an exception occurs.
    """

    print()
    print("=" * 78)
    print(
        "PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION"
    )
    print(
        "MASTER PROJECT PIPELINE"
    )
    print("=" * 78)
    print()

    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Experiment   : "
        f"{config.EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode : "
        f"{config.DATASET_MODE}"
    )

    print()

    try:

        # ==============================================================
        # STAGE 1
        # CONFIGURATION
        # ==============================================================

        validate_configuration()

        # ==============================================================
        # STAGE 2
        # FINAL MODEL TRAINING
        # ==============================================================

        run_training_stage()

        # ==============================================================
        # STAGE 3
        # BEST CHECKPOINT VERIFICATION
        # ==============================================================

        verify_checkpoint()

        # ==============================================================
        # STAGE 4
        # COMPLETE SCIENTIFIC EVALUATION
        # ==============================================================

        run_evaluation_stage()

        # ==============================================================
        # STAGE 5
        # FINAL OUTPUT VERIFICATION
        # ==============================================================

        verify_final_outputs()

        # ==============================================================
        # FINAL SUMMARY
        # ==============================================================

        print_pipeline_summary()

        print()
        print("=" * 78)
        print(
            "MASTER PROJECT PIPELINE COMPLETED SUCCESSFULLY"
        )
        print("=" * 78)
        print()

        return True

    except Exception as error:

        print()
        print("=" * 78)
        print(
            "MASTER PROJECT PIPELINE FAILED"
        )
        print("=" * 78)
        print()

        print_failure(
            str(error)
        )

        print()
        print(
            "Traceback:"
        )

        traceback.print_exc()

        print()

        print_pipeline_summary()

        return False


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    success = run_project()

    # ------------------------------------------------------------------
    # Return a non-zero exit code when the pipeline fails.
    # ------------------------------------------------------------------

    if not success:

        sys.exit(1)

    sys.exit(0)