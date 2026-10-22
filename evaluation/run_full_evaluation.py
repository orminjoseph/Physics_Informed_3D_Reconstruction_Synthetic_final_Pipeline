"""
====================================================================
FULL EVALUATION PIPELINE
====================================================================

Physics-Informed 3D Encoder–Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Master orchestration script for the complete PhD evaluation pipeline.

The runner supports:

    1. Optional training
    2. Model evaluation
    3. Reconstruction gallery
    4. Uncertainty analysis
    5. Uncertainty evaluation
    6. Uncertainty statistics
    7. Controlled six-classical-baseline experiments
    8. Controlled proposed-model experiment
    9. Common seven-method representative comparison
   10. Ablation study
   11. Statistical significance
   12. Thesis tables
   13. Final report

IMPORTANT
---------
The common seven-method representative comparison is executed by
this master controller after the controlled proposed-model experiment
and before the ablation study, through:

    evaluation.baselines.compare_with_baselines

Controlled experimental design
------------------------------
The controlled reconstruction experiment contains:

    6 geological settings
    × 5 missing-data mechanisms
    × 5 missing-data rates
    × 5 random seeds

Total:

    6 × 5 × 5 × 5 = 750 cases

For development/smoke testing:

    CONTROLLED_MATRIX_CASE_LIMIT = 10

For the final PhD experiment:

    CONTROLLED_MATRIX_CASE_LIMIT = None

Resume behaviour
----------------
When RESUME_EVALUATION = True:

    - Valid completed stages are skipped.
    - Incomplete stages are executed.
    - Existing valid results are preserved.

When FORCE_RERUN_EVALUATION = True:

    - All evaluation stages are executed again.
    - FORCE_RERUN_EVALUATION takes priority over RESUME_EVALUATION.

Important
---------
This controller does NOT modify the scientific implementations of
the individual evaluation modules.

It only orchestrates them.

All configurable parameters remain centralized in utils/config.py.

Current experiment
------------------
DATASET_MODE = synthetic
EXPERIMENT_NAME = synthetic_training

Output paths are derived from utils/config.py.

Author: Ormin Joseph
====================================================================
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from utils.config import (
    DATASET_MODE,
    EXPERIMENT_NAME,
    OUTPUT_ROOT,
    CHECKPOINT_DIR,
    REPORT_DIR,
    RUN_TRAINING,
    RESUME_EVALUATION,
    FORCE_RERUN_EVALUATION,
    GALLERY_NUMBER_OF_SAMPLES,
    CONTROLLED_MATRIX_CASE_LIMIT,
)


# ====================================================================
# PATHS
# ====================================================================

REPORT_PATH = Path(REPORT_DIR)

CHECKPOINT_PATH = Path(CHECKPOINT_DIR)

BEST_MODEL_PATH = (
    CHECKPOINT_PATH
    / "best_model.pth"
)


# ====================================================================
# CONTROLLED EXPERIMENT SETTINGS
# ====================================================================

FULL_CONTROLLED_CASE_COUNT = (
    6
    * 5
    * 5
    * 5
)


def expected_controlled_case_count() -> int:
    """
    Return the number of controlled cases expected from the
    current configuration.

    CONTROLLED_MATRIX_CASE_LIMIT = 10
        -> expected 10 cases

    CONTROLLED_MATRIX_CASE_LIMIT = None
        -> expected 750 cases
    """

    if CONTROLLED_MATRIX_CASE_LIMIT is None:

        return FULL_CONTROLLED_CASE_COUNT

    return min(
        int(CONTROLLED_MATRIX_CASE_LIMIT),
        FULL_CONTROLLED_CASE_COUNT,
    )


EXPECTED_CONTROLLED_CASES = (
    expected_controlled_case_count()
)


# ====================================================================
# CONTROLLED CLASSICAL BASELINE DEFINITIONS
# ====================================================================

CONTROLLED_CLASSICAL_BASELINES = [

    {
        "name": "Nearest Neighbor",

        "module": (
            "evaluation.baselines."
            "baseline_nearest_neighbor_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "nearest_neighbor_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "nearest_neighbor_controlled_summary.csv"
        ),
    },

    {
        "name": "Linear Interpolation",

        "module": (
            "evaluation.baselines."
            "baseline_linear_interpolation_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "linear_interpolation_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "linear_interpolation_controlled_summary.csv"
        ),
    },

    {
        "name": "f-x Prediction",

        "module": (
            "evaluation.baselines."
            "fx_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "fx_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "fx_controlled_summary.csv"
        ),
    },

    {
        "name": "Compressive Sensing",

        "module": (
            "evaluation.baselines."
            "compressive_sensing_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "cs_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "cs_controlled_summary.csv"
        ),
    },

    {
        "name": "Curvelet POCS",

        "module": (
            "evaluation.baselines."
            "curvelet_pocs_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "curvelet_pocs_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "curvelet_pocs_controlled_matrix_summary.csv"
        ),
    },

    {
        "name": "Dictionary Learning",

        "module": (
            "evaluation.baselines."
            "dictionary_learning_controlled_matrix"
        ),

        "csv": (
            REPORT_PATH
            / "dictionary_learning_controlled_matrix.csv"
        ),

        "summary": (
            REPORT_PATH
            / "dictionary_learning_controlled_matrix_summary.csv"
        ),
    },
]


# ====================================================================
# PROPOSED MODEL CONTROLLED EXPERIMENT
# ====================================================================

PROPOSED_CONTROLLED_MODULE = (
    "evaluation.baselines."
    "proposed_model_controlled_matrix"
)


PROPOSED_CONTROLLED_CSV = (
    REPORT_PATH
    / "proposed_controlled_matrix.csv"
)

PROPOSED_CONTROLLED_SUMMARY = (
    REPORT_PATH
    / "proposed_controlled_summary.csv"
)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def print_header(
    title: str,
) -> None:

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_stage(
    stage_number: int,
    title: str,
) -> None:

    print()
    print("-" * 78)

    print(
        f"STAGE {stage_number}: {title}"
    )

    print("-" * 78)


def run_module(
    module_name: str,
    description: str,
) -> None:

    print()
    print(
        f"Running: {description}"
    )

    print(
        f"Module : python -m {module_name}"
    )

    print()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ],
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "\nEvaluation stage failed:\n"
            f"Module      : {module_name}\n"
            f"Return code : {result.returncode}\n"
        )


def file_is_valid(
    path: Path,
    minimum_size: int = 1,
) -> bool:

    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size >= minimum_size
    )


def csv_has_required_columns(
    path: Path,
    required_columns: list[str],
) -> bool:

    if not file_is_valid(path):

        return False

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:

                return False

            available_columns = set(
                reader.fieldnames
            )

            return set(
                required_columns
            ).issubset(
                available_columns
            )

    except Exception:

        return False


def csv_has_expected_rows(
    path: Path,
    expected_rows: int,
) -> bool:

    if not file_is_valid(path):

        return False

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:

                return False

            row_count = sum(
                1
                for _ in reader
            )

        return (
            row_count
            == expected_rows
        )

    except Exception:

        return False


def json_is_valid(
    path: Path,
) -> bool:

    if not file_is_valid(path):

        return False

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            json.load(file)

        return True

    except Exception:

        return False


def figure_is_valid(
    path: Path,
) -> bool:

    return file_is_valid(
        path,
        minimum_size=100,
    )


# ====================================================================
# STAGE 1 VALIDATION
# ====================================================================

def model_evaluation_complete() -> bool:

    path = (
        REPORT_PATH
        / "evaluation_metrics.csv"
    )

    return csv_has_required_columns(
        path,
        [
            "MAE",
            "RMSE",
            "PSNR",
            "SSIM",
        ],
    )


# ====================================================================
# STAGE 2 VALIDATION
# ====================================================================

def reconstruction_gallery_complete() -> bool:

    gallery_dir = (
        REPORT_PATH
        / "gallery"
    )

    if not gallery_dir.exists():

        return False

    number_of_samples = int(
        GALLERY_NUMBER_OF_SAMPLES
    )

    for index in range(
        number_of_samples
    ):

        figure_path = (
            gallery_dir
            / f"sample_{index:03d}.png"
        )

        if not figure_is_valid(
            figure_path
        ):

            return False

    return True


# ====================================================================
# STAGE 3 VALIDATION
# ====================================================================

def uncertainty_analysis_complete() -> bool:

    path = (
        REPORT_PATH
        / "uncertainty"
        / "uncertainty_analysis.png"
    )

    return figure_is_valid(path)


# ====================================================================
# STAGE 4 VALIDATION
# ====================================================================

def uncertainty_evaluation_complete() -> bool:

    evaluation_csv = (
        REPORT_PATH
        / "uncertainty_evaluation.csv"
    )

    correlation_csv = (
        REPORT_PATH
        / "uncertainty_error_correlation.csv"
    )

    metadata_json = (
        REPORT_PATH
        / "uncertainty_evaluation_metadata.json"
    )

    evaluation_valid = csv_has_required_columns(
        evaluation_csv,
        [
            "MAE",
            "RMSE",
            "Missing_MAE",
            "Missing_RMSE",
            "aleatoric_variance",
            "epistemic_variance",
            "predictive_variance",
            "predictive_std",
        ],
    )

    correlation_valid = file_is_valid(
        correlation_csv
    )

    metadata_valid = json_is_valid(
        metadata_json
    )

    return (
        evaluation_valid
        and correlation_valid
        and metadata_valid
    )


# ====================================================================
# STAGE 5 VALIDATION
# ====================================================================

def uncertainty_statistics_complete() -> bool:

    path = (
        REPORT_PATH
        / "uncertainty_statistics.csv"
    )

    return csv_has_required_columns(
        path,
        [
            "Experiment_Name",
            "Number_of_Patches",
            "MC_Samples",
            "Mean_MAE",
            "Mean_RMSE",
            "Mean_Missing_MAE",
            "Mean_Missing_RMSE",
            "Mean_Aleatoric_Variance",
            "Mean_Epistemic_Variance",
            "Mean_Predictive_Variance",
            "Mean_Predictive_Std",
            "Maximum_Predictive_Variance",
            "Maximum_Predictive_Std",
            "Mean_Observed_Preservation_Error",
            "Maximum_Observed_Preservation_Error",
            "Mean_Measured_Missing_Rate",
            "Maximum_Measured_Missing_Rate",
        ],
    )


# ====================================================================
# STAGE 6 VALIDATION
# ====================================================================

def controlled_classical_baselines_complete() -> bool:
    """
    Check whether all six classical controlled baseline matrices
    have completed with the expected number of cases.
    """

    required_columns = [
        "Case_ID",
        "Geological_Mode",
        "Mask_Mode",
        "Missing_Rate",
        "Seed",
        "MAE",
        "RMSE",
        "PSNR",
        "SSIM",
    ]

    for baseline in (
        CONTROLLED_CLASSICAL_BASELINES
    ):

        csv_path = baseline["csv"]

        summary_path = baseline["summary"]

        if not csv_has_required_columns(
            csv_path,
            required_columns,
        ):

            return False

        if not csv_has_expected_rows(
            csv_path,
            EXPECTED_CONTROLLED_CASES,
        ):

            return False

        if not file_is_valid(
            summary_path
        ):

            return False

    return True


# ====================================================================
# STAGE 7 VALIDATION
# ====================================================================

def controlled_proposed_model_complete() -> bool:

    required_columns = [
        "Case_ID",
        "Geological_Mode",
        "Mask_Mode",
        "Missing_Rate",
        "Seed",
        "MAE",
        "RMSE",
        "PSNR",
        "SSIM",
    ]

    if not csv_has_required_columns(
        PROPOSED_CONTROLLED_CSV,
        required_columns,
    ):

        return False

    if not csv_has_expected_rows(
        PROPOSED_CONTROLLED_CSV,
        EXPECTED_CONTROLLED_CASES,
    ):

        return False

    return file_is_valid(
        PROPOSED_CONTROLLED_SUMMARY
    )


# ====================================================================
# STAGE 8 VALIDATION — COMMON SEVEN-METHOD COMPARISON
# ====================================================================

def common_baseline_comparison_complete() -> bool:
    """Check whether the common seven-method comparison outputs exist."""

    results_path = REPORT_PATH / "baseline_comparison.csv"
    summary_path = REPORT_PATH / "baseline_comparison_summary.csv"

    required_columns = [
        "method",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "status",
    ]

    if not csv_has_required_columns(results_path, required_columns):
        return False

    if not file_is_valid(summary_path):
        return False

    try:
        with open(results_path, "r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except (OSError, csv.Error):
        return False

    expected_methods = {
        "nearest_neighbor",
        "linear_interpolation",
        "fx_prediction",
        "compressive_sensing",
        "curvelet_pocs",
        "dictionary_learning",
        "proposed_physics_informed_3d",
    }

    return (
        len(rows) == 7
        and {row.get("method", "") for row in rows} == expected_methods
        and all(row.get("status", "") == "PASS" for row in rows)
    )


# ====================================================================
# STAGE 9 VALIDATION
# ====================================================================

def ablation_complete() -> bool:

    path = (
        REPORT_PATH
        / "ablation_study.csv"
    )

    summary_path = (
        REPORT_PATH
        / "ablation_summary.csv"
    )

    metadata_path = (
        REPORT_PATH
        / "ablation_metadata.json"
    )

    return (
        csv_has_required_columns(
            path,
            [
                "Model",
                "MAE",
                "RMSE",
                "PSNR",
                "SSIM",
            ],
        )
        and file_is_valid(
            summary_path
        )
        and json_is_valid(
            metadata_path
        )
    )


# ====================================================================
# STAGE 10 VALIDATION
# ====================================================================

def statistical_significance_complete() -> bool:

    path = (
        REPORT_PATH
        / "statistical_significance.csv"
    )

    metadata_path = (
        REPORT_PATH
        / "statistical_significance_metadata.json"
    )

    return (
        csv_has_required_columns(
            path,
            [
                "Comparison",
                "Metric",
                "p_value",
            ],
        )
        and json_is_valid(
            metadata_path
        )
    )


# ====================================================================
# STAGE 11 VALIDATION
# ====================================================================

def thesis_tables_complete() -> bool:

    required_tables = [

        "Table_4_1_Main_Performance.csv",

        "Table_4_2_Ablation_Study.csv",

        "Table_4_3_Uncertainty_Statistics.csv",

        "Table_4_4_Statistical_Significance.csv",
    ]

    tables_dir = (
        REPORT_PATH
        / "thesis_tables"
    )

    if not tables_dir.exists():

        return False

    for table_name in required_tables:

        table_path = (
            tables_dir
            / table_name
        )

        if not file_is_valid(
            table_path
        ):

            return False

    return True


# ====================================================================
# STAGE 12 VALIDATION
# ====================================================================

def final_report_complete() -> bool:

    path = (
        REPORT_PATH
        / "final_report.txt"
    )

    return file_is_valid(path)


# ====================================================================
# STATUS REPORT
# ====================================================================

def display_pipeline_status() -> None:

    print_header(
        "CURRENT EVALUATION PIPELINE STATUS"
    )

    stages = [

        (
            "Model Evaluation",
            model_evaluation_complete(),
        ),

        (
            "Reconstruction Gallery",
            reconstruction_gallery_complete(),
        ),

        (
            "Uncertainty Analysis",
            uncertainty_analysis_complete(),
        ),

        (
            "Uncertainty Evaluation",
            uncertainty_evaluation_complete(),
        ),

        (
            "Uncertainty Statistics",
            uncertainty_statistics_complete(),
        ),

        (
            "Controlled Classical Baselines",
            controlled_classical_baselines_complete(),
        ),

        (
            "Controlled Proposed Model",
            controlled_proposed_model_complete(),
        ),

        (
            "Common Seven-Method Comparison",
            common_baseline_comparison_complete(),
        ),

        (
            "Ablation Study",
            ablation_complete(),
        ),

        (
            "Statistical Significance",
            statistical_significance_complete(),
        ),

        (
            "Thesis Tables",
            thesis_tables_complete(),
        ),

        (
            "Final Report",
            final_report_complete(),
        ),
    ]

    for number, (
        name,
        complete,
    ) in enumerate(
        stages,
        start=1,
    ):

        status = (
            "[COMPLETE]"
            if complete
            else "[PENDING]"
        )

        print(
            f"{number:02d}. "
            f"{status:<12} "
            f"{name}"
        )

    print()

    print(
        f"Controlled cases expected: "
        f"{EXPECTED_CONTROLLED_CASES}"
    )

    if CONTROLLED_MATRIX_CASE_LIMIT is None:

        print(
            "Controlled matrix mode: "
            "FULL 750-CASE EXPERIMENT"
        )

    else:

        print(
            "Controlled matrix mode: "
            f"SMOKE TEST "
            f"({CONTROLLED_MATRIX_CASE_LIMIT} cases)"
        )


# ====================================================================
# TRAINING
# ====================================================================

def run_training_if_required() -> None:

    if not RUN_TRAINING:

        print(
            "RUN_TRAINING = False"
        )

        print(
            "Training skipped."
        )

        return

    print_header(
        "TRAINING"
    )

    print(
        "RUN_TRAINING = True"
    )

    print(
        "Training module:"
    )

    print(
        "train.train_model"
    )

    run_module(
        "train.train_model",
        "Training pipeline",
    )


# ====================================================================
# CONTROLLED CLASSICAL BASELINES
# ====================================================================

def run_controlled_classical_baselines() -> None:

    for baseline in (
        CONTROLLED_CLASSICAL_BASELINES
    ):

        name = baseline["name"]

        module = baseline["module"]

        print()

        print(
            f"Controlled baseline: {name}"
        )

        run_module(
            module,
            f"{name} controlled matrix",
        )


# ====================================================================
# CONTROLLED PROPOSED MODEL
# ====================================================================

def run_controlled_proposed_model() -> None:

    run_module(
        PROPOSED_CONTROLLED_MODULE,
        "Proposed model controlled matrix",
    )


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main() -> None:

    print_header(
        "PHYSICS-INFORMED 3D SEISMIC RECONSTRUCTION"
    )

    print(
        "RESUME-ENABLED FULL EVALUATION PIPELINE"
    )

    print()

    print(
        f"Experiment        : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Dataset mode      : "
        f"{DATASET_MODE}"
    )

    print(
        f"Output root       : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Report directory  : "
        f"{REPORT_PATH}"
    )

    print(
        f"Checkpoint        : "
        f"{BEST_MODEL_PATH}"
    )

    print()

    print(
        f"RUN_TRAINING              : "
        f"{RUN_TRAINING}"
    )

    print(
        f"RESUME_EVALUATION         : "
        f"{RESUME_EVALUATION}"
    )

    print(
        f"FORCE_RERUN_EVALUATION    : "
        f"{FORCE_RERUN_EVALUATION}"
    )

    print(
        f"GALLERY_NUMBER_OF_SAMPLES : "
        f"{GALLERY_NUMBER_OF_SAMPLES}"
    )

    print(
        f"CONTROLLED_MATRIX_CASE_LIMIT : "
        f"{CONTROLLED_MATRIX_CASE_LIMIT}"
    )

    print(
        f"EXPECTED_CONTROLLED_CASES : "
        f"{EXPECTED_CONTROLLED_CASES}"
    )

    REPORT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_training_if_required()

    print_header(
        "CHECKPOINT VALIDATION"
    )

    if not file_is_valid(
        BEST_MODEL_PATH,
        minimum_size=1_000_000,
    ):

        raise FileNotFoundError(
            "\nRequired checkpoint was not found "
            "or is unexpectedly small:\n"
            f"{BEST_MODEL_PATH}\n\n"
            "The evaluation pipeline cannot continue "
            "without best_model.pth."
        )

    print(
        "[VALID] best_model.pth"
    )

    display_pipeline_status()

    resume = (
        RESUME_EVALUATION
        and not FORCE_RERUN_EVALUATION
    )

    if FORCE_RERUN_EVALUATION:

        print()
        print(
            "FORCE_RERUN_EVALUATION = True"
        )
        print(
            "All evaluation stages will be executed again."
        )

    elif RESUME_EVALUATION:

        print()
        print(
            "RESUME_EVALUATION = True"
        )
        print(
            "Valid completed stages will be skipped."
        )

    else:

        print()
        print(
            "RESUME_EVALUATION = False"
        )
        print(
            "Evaluation stages will be executed."
        )

    # =================================================================
    # STAGE 1 — MODEL EVALUATION
    # =================================================================

    print_stage(
        1,
        "MODEL EVALUATION",
    )

    if (
        resume
        and model_evaluation_complete()
    ):

        print(
            "[SKIPPED] Model evaluation already completed."
        )

    else:

        run_module(
            "evaluation.evaluate_model",
            "Model evaluation",
        )

    # =================================================================
    # STAGE 2 — RECONSTRUCTION GALLERY
    # =================================================================

    print_stage(
        2,
        "RECONSTRUCTION GALLERY",
    )

    if (
        resume
        and reconstruction_gallery_complete()
    ):

        print(
            "[SKIPPED] Reconstruction gallery already completed."
        )

    else:

        run_module(
            "evaluation.reconstruction_gallery",
            "Reconstruction gallery",
        )

    # =================================================================
    # STAGE 3 — UNCERTAINTY ANALYSIS
    # =================================================================

    print_stage(
        3,
        "UNCERTAINTY ANALYSIS",
    )

    if (
        resume
        and uncertainty_analysis_complete()
    ):

        print(
            "[SKIPPED] Uncertainty analysis already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_analysis",
            "Uncertainty analysis",
        )

    # =================================================================
    # STAGE 4 — UNCERTAINTY EVALUATION
    # =================================================================

    print_stage(
        4,
        "UNCERTAINTY EVALUATION",
    )

    if (
        resume
        and uncertainty_evaluation_complete()
    ):

        print(
            "[SKIPPED] Uncertainty evaluation already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_evaluation",
            "Uncertainty–reconstruction error evaluation",
        )

    # =================================================================
    # STAGE 5 — UNCERTAINTY STATISTICS
    # =================================================================

    print_stage(
        5,
        "UNCERTAINTY STATISTICS",
    )

    if (
        resume
        and uncertainty_statistics_complete()
    ):

        print(
            "[SKIPPED] Uncertainty statistics already completed."
        )

    else:

        run_module(
            "evaluation.uncertainty_statistics",
            "Final uncertainty statistics generation",
        )

    # =================================================================
    # STAGE 6 — CONTROLLED CLASSICAL BASELINES
    # =================================================================

    print_stage(
        6,
        "CONTROLLED CLASSICAL BASELINES",
    )

    if (
        resume
        and controlled_classical_baselines_complete()
    ):

        print(
            "[SKIPPED] All six controlled classical baseline "
            "experiments already completed."
        )

    else:

        run_controlled_classical_baselines()

    # =================================================================
    # STAGE 7 — CONTROLLED PROPOSED MODEL
    # =================================================================

    print_stage(
        7,
        "CONTROLLED PROPOSED MODEL",
    )

    if (
        resume
        and controlled_proposed_model_complete()
    ):

        print(
            "[SKIPPED] Proposed model controlled experiment "
            "already completed."
        )

    else:

        run_controlled_proposed_model()

    # =================================================================
    # STAGE 8 — COMMON SEVEN-METHOD COMPARISON
    # =================================================================

    print_stage(
        8,
        "COMMON SEVEN-METHOD COMPARISON",
    )

    if (
        resume
        and common_baseline_comparison_complete()
    ):

        print(
            "[SKIPPED] Common seven-method comparison "
            "already completed."
        )

    else:

        run_module(
            "evaluation.baselines.compare_with_baselines",
            "Common seven-method reconstruction comparison",
        )

    # =================================================================
    # STAGE 9 — ABLATION STUDY
    # =================================================================

    print_stage(
        9,
        "ABLATION STUDY",
    )

    if (
        resume
        and ablation_complete()
    ):

        print(
            "[SKIPPED] Ablation study already completed."
        )

    else:

        run_module(
            "evaluation.ablation_study",
            "Ablation study",
        )

    # =================================================================
    # STAGE 10 — STATISTICAL SIGNIFICANCE
    # =================================================================

    print_stage(
        10,
        "STATISTICAL SIGNIFICANCE",
    )

    if (
        resume
        and statistical_significance_complete()
    ):

        print(
            "[SKIPPED] Statistical significance analysis "
            "already completed."
        )

    else:

        run_module(
            "evaluation.statistical_significance",
            "Statistical significance analysis",
        )

    # =================================================================
    # STAGE 11 — THESIS TABLES
    # =================================================================

    print_stage(
        11,
        "THESIS TABLES",
    )

    if (
        resume
        and thesis_tables_complete()
    ):

        print(
            "[SKIPPED] Thesis tables already completed."
        )

    else:

        run_module(
            "evaluation.final_thesis_tables",
            "Thesis table generation",
        )

    # =================================================================
    # STAGE 12 — FINAL REPORT
    # =================================================================

    print_stage(
        12,
        "FINAL REPORT",
    )

    if (
        resume
        and final_report_complete()
    ):

        print(
            "[SKIPPED] Final report already completed."
        )

    else:

        run_module(
            "evaluation.final_report",
            "Final report generation",
        )

    display_pipeline_status()

    print_header(
        "FULL EVALUATION PIPELINE COMPLETE"
    )

    print(
        "All configured evaluation stages have "
        "completed successfully."
    )

    print()

    print(
        f"Experiment : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Dataset    : "
        f"{DATASET_MODE}"
    )

    print(
        f"Reports    : "
        f"{REPORT_PATH}"
    )


# ====================================================================
# PUBLIC PIPELINE FUNCTION
# ====================================================================

def run_full_evaluation() -> None:

    main()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    run_full_evaluation()
