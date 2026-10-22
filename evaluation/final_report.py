"""
=========================================================
FINAL REPORT GENERATOR
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Validate all completed evaluation outputs and generate
a final text report.

IMPORTANT
---------
This module consumes existing evaluation outputs.

It does NOT:
    - retrain the model
    - rerun baselines
    - modify canonical metrics
    - modify experimental results

It is strictly a reporting layer.

Author: Ormin Joseph
=========================================================
"""

from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# =========================================================
# PATHS
# =========================================================

REPORT_PATH = Path(
    REPORT_DIR
)

FINAL_REPORT_FILE = (
    REPORT_PATH /
    "final_report.txt"
)

FINAL_REPORT_METADATA_FILE = (
    REPORT_PATH /
    "final_report_metadata.json"
)


# =========================================================
# REQUIRED RESULT FILES
# =========================================================

RESULT_FILES = {
    "Evaluation Metrics":
        REPORT_PATH / "evaluation_metrics.csv",

    "Uncertainty Statistics":
        REPORT_PATH / "uncertainty_statistics.csv",

    "Baseline Comparison":
        REPORT_PATH / "baseline_comparison.csv",

    "Statistical Significance":
        REPORT_PATH / "statistical_significance.csv",

    "Ablation Study (Per Sample)":
        REPORT_PATH / "ablation_study.csv",

    "Ablation Summary":
        REPORT_PATH / "ablation_summary.csv",
}


# =========================================================
# METRIC ALIASES
# =========================================================

METRIC_ALIASES = {
    "MAE": ["MAE", "mae"],
    "RMSE": ["RMSE", "rmse"],
    "PSNR": ["PSNR", "psnr"],
    "SNR": ["SNR", "snr"],
    "SSIM": ["SSIM", "ssim"],
}


# =========================================================
# HELPERS
# =========================================================

def find_column(
    dataframe,
    aliases,
):
    """
    Find the first available column from aliases.
    """

    for column in aliases:

        if column in dataframe.columns:
            return column

    return None


def validate_evaluation_metrics():
    """
    Validate the canonical evaluation metrics file.

    Returns
    -------
    tuple
        (is_valid, dataframe, metric_columns)
    """

    file_path = RESULT_FILES[
        "Evaluation Metrics"
    ]

    dataframe = pd.read_csv(
        file_path
    )

    metric_columns = {}

    missing = []

    for metric, aliases in METRIC_ALIASES.items():

        source_column = find_column(
            dataframe,
            aliases,
        )

        if source_column is None:

            missing.append(
                metric
            )

        else:

            metric_columns[
                metric
            ] = source_column

    if missing:

        return (
            False,
            dataframe,
            metric_columns,
        )

    for metric, source_column in (
        metric_columns.items()
    ):

        if dataframe[source_column].isna().any():

            print(
                f"[WARNING] {metric} contains "
                "missing values."
            )

            return (
                False,
                dataframe,
                metric_columns,
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[source_column]
        ):

            print(
                f"[WARNING] {metric} is not numeric."
            )

            return (
                False,
                dataframe,
                metric_columns,
            )

    return (
        True,
        dataframe,
        metric_columns,
    )


def validate_baseline_comparison():
    """
    Validate the seven-method baseline comparison.
    """

    file_path = RESULT_FILES[
        "Baseline Comparison"
    ]

    dataframe = pd.read_csv(
        file_path
    )

    model_column = find_column(
        dataframe,
        [
            "Model",
            "Method",
            "model",
            "method",
        ],
    )

    if model_column is None:

        return False, dataframe, None

    metric_columns = {}

    missing = []

    for metric, aliases in METRIC_ALIASES.items():

        source_column = find_column(
            dataframe,
            aliases,
        )

        if source_column is None:
            missing.append(metric)

        else:
            metric_columns[
                metric
            ] = source_column

    if missing:

        return (
            False,
            dataframe,
            model_column,
        )

    return (
        True,
        dataframe,
        model_column,
    )


def validate_generic_file(
    label,
):
    """
    Validate that a result CSV exists and is non-empty.
    """

    file_path = RESULT_FILES[
        label
    ]

    dataframe = pd.read_csv(
        file_path
    )

    return (
        not dataframe.empty,
        dataframe,
    )


# =========================================================
# REPORT GENERATION
# =========================================================

def generate_report():

    print()
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    print()
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Report Dir : {REPORT_PATH}"
    )

    print(
        f"Report File: {FINAL_REPORT_FILE}"
    )

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING REQUIRED RESULTS")
    print("=" * 70)

    file_status = {}

    for label, file_path in (
        RESULT_FILES.items()
    ):

        if file_path.exists():

            try:

                dataframe = pd.read_csv(
                    file_path
                )

                file_status[label] = {
                    "exists": True,
                    "rows": len(dataframe),
                    "columns": len(
                        dataframe.columns
                    ),
                }

                print(
                    f"[FOUND] {label:<32} "
                    f"{file_path} "
                    f"({len(dataframe)} rows)"
                )

            except Exception as error:

                file_status[label] = {
                    "exists": False,
                    "error": str(error),
                }

                print(
                    f"[ERROR] {label:<32} "
                    f"{error}"
                )

        else:

            file_status[label] = {
                "exists": False,
            }

            print(
                f"[MISSING] {label:<32} "
                f"{file_path}"
            )

    missing_files = [
        label
        for label, status in file_status.items()
        if not status["exists"]
    ]

    if missing_files:

        print()
        print(
            "[ERROR] Required result files are missing."
        )

        for label in missing_files:

            print(
                f"  - {label}"
            )

        print()
        print(
            "FINAL REPORT NOT GENERATED"
        )

        return False

    # -----------------------------------------------------
    # Structural validation
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATING RESULT STRUCTURES")
    print("=" * 70)

    validation_errors = []

    # -----------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------

    (
        evaluation_valid,
        evaluation_df,
        evaluation_metric_columns,
    ) = validate_evaluation_metrics()

    if evaluation_valid:

        print(
            "[VALID] Evaluation Metrics"
        )

    else:

        print(
            "[INVALID] Evaluation Metrics"
        )

        missing_metrics = [
            metric
            for metric in METRIC_ALIASES
            if metric not in evaluation_metric_columns
        ]

        if missing_metrics:

            print(
                "         Missing:"
            )

            for metric in missing_metrics:

                print(
                    f"           - {metric}"
                )

            validation_errors.append(
                "Evaluation Metrics"
            )

    # -----------------------------------------------------
    # Baseline comparison
    # -----------------------------------------------------

    (
        baseline_valid,
        baseline_df,
        baseline_model_column,
    ) = validate_baseline_comparison()

    if baseline_valid:

        print(
            "[VALID] Baseline Comparison"
        )

    else:

        print(
            "[INVALID] Baseline Comparison"
        )

        if baseline_model_column is None:

            print(
                "         Could not identify "
                "Model/Method column."
            )

        validation_errors.append(
            "Baseline Comparison"
        )

    # -----------------------------------------------------
    # Generic result files
    # -----------------------------------------------------

    generic_labels = [
        "Uncertainty Statistics",
        "Statistical Significance",
        "Ablation Study (Per Sample)",
        "Ablation Summary",
    ]

    generic_dataframes = {}

    for label in generic_labels:

        valid, dataframe = (
            validate_generic_file(label)
        )

        generic_dataframes[
            label
        ] = dataframe

        if valid:

            print(
                f"[VALID] {label}"
            )

        else:

            print(
                f"[INVALID] {label}"
            )

            validation_errors.append(
                label
            )

    # -----------------------------------------------------
    # Stop if invalid
    # -----------------------------------------------------

    if validation_errors:

        print()
        print(
            "=" * 70
        )

        print(
            "FINAL REPORT NOT GENERATED"
        )

        print()

        print(
            "One or more required result files "
            "failed structural or numerical validation."
        )

        print()

        for error in validation_errors:

            print(
                f"  - {error}"
            )

        return False

    # =====================================================
    # GENERATE REPORT
    # =====================================================

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "FINAL EVALUATION REPORT"
    )

    lines.append(
        "Physics-Informed 3D Encoder-Decoder Framework "
        "with Predictive Uncertainty"
    )

    lines.append(
        "for Seismic Data Reconstruction"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    lines.append(
        f"Experiment: {EXPERIMENT_NAME}"
    )

    lines.append(
        f"Generated: {datetime.now().isoformat()}"
    )

    lines.append("")

    # -----------------------------------------------------
    # Main performance
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "1. MAIN MODEL PERFORMANCE"
    )

    lines.append(
        "-" * 78
    )

    for metric, source_column in (
        evaluation_metric_columns.items()
    ):

        values = pd.to_numeric(
            evaluation_df[
                source_column
            ],
            errors="coerce",
        )

        lines.append(
            f"{metric:>8}: "
            f"mean={values.mean():.6f}, "
            f"std={values.std(ddof=1) if len(values) > 1 else 0.0:.6f}"
        )

    lines.append("")

    # -----------------------------------------------------
    # Baseline comparison
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "2. SEVEN-METHOD BASELINE COMPARISON"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        f"Number of methods: "
        f"{len(baseline_df)}"
    )

    lines.append("")

    for _, row in baseline_df.iterrows():

        model = row[
            baseline_model_column
        ]

        lines.append(
            f"Method: {model}"
        )

        for metric in [
            "MAE",
            "RMSE",
            "PSNR",
            "SNR",
            "SSIM",
        ]:

            column = find_column(
                baseline_df,
                METRIC_ALIASES[metric],
            )

            if column is not None:

                lines.append(
                    f"    {metric}: "
                    f"{float(row[column]):.6f}"
                )

        lines.append("")

    # -----------------------------------------------------
    # Uncertainty
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "3. PREDICTIVE UNCERTAINTY"
    )

    lines.append(
        "-" * 78
    )

    uncertainty_df = (
        generic_dataframes[
            "Uncertainty Statistics"
        ]
    )

    lines.append(
        f"Rows: {len(uncertainty_df)}"
    )

    lines.append("")

    for column in uncertainty_df.columns:

        if pd.api.types.is_numeric_dtype(
            uncertainty_df[column]
        ):

            values = uncertainty_df[
                column
            ]

            lines.append(
                f"{column}: "
                f"{values.mean():.6f}"
            )

    lines.append("")

    # -----------------------------------------------------
    # Ablation
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "4. ABLATION STUDY"
    )

    lines.append(
        "-" * 78
    )

    ablation_df = generic_dataframes[
        "Ablation Summary"
    ]

    lines.append(
        f"Configurations: "
        f"{len(ablation_df)}"
    )

    lines.append("")

    # -----------------------------------------------------
    # Statistical significance
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "5. STATISTICAL SIGNIFICANCE"
    )

    lines.append(
        "-" * 78
    )

    significance_df = generic_dataframes[
        "Statistical Significance"
    ]

    lines.append(
        f"Comparisons: "
        f"{len(significance_df)}"
    )

    if "N_Pairs" in significance_df.columns:

        lines.append(
            "Paired observations per comparison: "
            f"{significance_df['N_Pairs'].tolist()}"
        )

    if "Holm_Adjusted_P_Value" in (
        significance_df.columns
    ):

        lines.append(
            "Holm-adjusted p-values:"
        )

        for value in (
            significance_df[
                "Holm_Adjusted_P_Value"
            ]
        ):

            lines.append(
                f"    {float(value):.6f}"
            )

    lines.append("")

    lines.append(
        "IMPORTANT STATISTICAL LIMITATION:"
    )

    lines.append(
        "The present ablation significance analysis "
        "contains only two paired validation samples. "
        "Therefore, the inferential results should be "
        "treated as exploratory and not as strong "
        "population-level evidence."
    )

    lines.append("")

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    lines.append(
        "-" * 78
    )

    lines.append(
        "6. EVALUATION PIPELINE STATUS"
    )

    lines.append(
        "-" * 78
    )

    lines.append(
        "Required evaluation result files were found "
        "and passed structural validation."
    )

    lines.append(
        "This report summarizes the currently completed "
        "evaluation outputs."
    )

    lines.append("")

    lines.append(
        "=" * 78
    )

    lines.append(
        "END OF FINAL EVALUATION REPORT"
    )

    lines.append(
        "=" * 78
    )

    # -----------------------------------------------------
    # Write report
    # -----------------------------------------------------

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "generated": datetime.now().isoformat(),
        "report_file": str(
            FINAL_REPORT_FILE
        ),
        "validation": {
            "evaluation_metrics": True,
            "baseline_comparison": True,
            "uncertainty_statistics": True,
            "statistical_significance": True,
            "ablation_study": True,
            "ablation_summary": True,
        },
        "statistical_limitation": (
            "Ablation significance testing uses "
            "two paired validation samples and "
            "should therefore be interpreted "
            "as exploratory."
        ),
    }

    with open(
        FINAL_REPORT_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL REPORT GENERATED SUCCESSFULLY"
    )

    print(
        f"Output: {FINAL_REPORT_FILE}"
    )

    print(
        "=" * 70
    )

    return True


# =========================================================
# PUBLIC ENTRY POINT
# =========================================================

def main():

    return generate_report()


if __name__ == "__main__":
    main()