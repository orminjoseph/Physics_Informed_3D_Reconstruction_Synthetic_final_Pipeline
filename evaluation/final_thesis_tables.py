"""
=========================================================
FINAL THESIS TABLE GENERATION
=========================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Generate thesis-ready tables from the canonical evaluation
outputs.

IMPORTANT
---------
This module is a REPORTING LAYER.

It must not modify:
    metrics/reconstruction_metrics.py

The canonical metrics pipeline may use lowercase metric keys:
    mae
    rmse
    psnr
    snr
    ssim

This module adapts those names for thesis reporting.

Author: Ormin Joseph
=========================================================
"""

from pathlib import Path
import json

import pandas as pd

from utils.config import (
    EXPERIMENT_NAME,
    REPORT_DIR,
)


# =========================================================
# DIRECTORIES
# =========================================================

REPORT_PATH = Path(REPORT_DIR)

THESIS_TABLE_DIR = REPORT_PATH / "thesis_tables"

THESIS_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SOURCE FILES
# =========================================================

EVALUATION_METRICS_FILE = (
    REPORT_PATH / "evaluation_metrics.csv"
)

ABLATION_SUMMARY_FILE = (
    REPORT_PATH / "ablation_summary.csv"
)

UNCERTAINTY_STATISTICS_FILE = (
    REPORT_PATH / "uncertainty_statistics.csv"
)

STATISTICAL_SIGNIFICANCE_FILE = (
    REPORT_PATH / "statistical_significance.csv"
)

BASELINE_COMPARISON_FILE = (
    REPORT_PATH / "baseline_comparison.csv"
)


# =========================================================
# OUTPUT FILES
# =========================================================

TABLE_4_1_FILE = (
    THESIS_TABLE_DIR /
    "Table_4_1_Main_Performance.csv"
)

TABLE_4_2_FILE = (
    THESIS_TABLE_DIR /
    "Table_4_2_Ablation_Study.csv"
)

TABLE_4_3_FILE = (
    THESIS_TABLE_DIR /
    "Table_4_3_Uncertainty_Statistics.csv"
)

TABLE_4_4_FILE = (
    THESIS_TABLE_DIR /
    "Table_4_4_Statistical_Significance.csv"
)

TABLE_4_5_FILE = (
    THESIS_TABLE_DIR /
    "Table_4_5_Baseline_Comparison.csv"
)

METADATA_FILE = (
    THESIS_TABLE_DIR /
    "thesis_tables_metadata.json"
)


# =========================================================
# METRIC NAME ADAPTER
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
    dataframe: pd.DataFrame,
    aliases,
):
    """
    Find the first matching column from a list of aliases.
    """

    for column in aliases:
        if column in dataframe.columns:
            return column

    return None


def adapt_metric_columns(
    dataframe: pd.DataFrame,
):
    """
    Convert canonical lowercase metric names to
    thesis-style uppercase names.

    The original dataframe is not modified.
    """

    output = dataframe.copy()

    missing = []

    for thesis_name, aliases in METRIC_ALIASES.items():

        source_column = find_column(
            output,
            aliases,
        )

        if source_column is None:
            missing.append(thesis_name)

        else:
            output[thesis_name] = output[source_column]

    return output, missing


def find_model_column(
    dataframe: pd.DataFrame,
):
    """
    Identify the method/model column used by the
    baseline comparison output.
    """

    candidates = [
        "Model",
        "Method",
        "model",
        "method",
    ]

    for column in candidates:

        if column in dataframe.columns:
            return column

    return None


# =========================================================
# TABLE 4.1
# =========================================================

def generate_table_4_1():
    """
    Generate the main model-performance table.
    """

    print("-" * 70)
    print("PROCESSING: Table 4.1 - Main Performance")
    print("-" * 70)

    if not EVALUATION_METRICS_FILE.exists():

        print(
            "[ERROR] Evaluation metrics file does not exist:"
        )

        print(
            EVALUATION_METRICS_FILE
        )

        return False

    dataframe = pd.read_csv(
        EVALUATION_METRICS_FILE
    )

    print(
        f"Source: {EVALUATION_METRICS_FILE}"
    )

    print(
        f"Rows   : {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    dataframe, missing = adapt_metric_columns(
        dataframe
    )

    if missing:

        print(
            "[ERROR] Table 4.1 is missing required "
            f"metric columns: {missing}"
        )

        return False

    required_columns = [
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    # Preserve useful identifying columns when present.
    preferred_columns = [
        "Experiment",
        "Sample_ID",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
    ]

    selected_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    # Ensure the five required metrics are always included.
    for column in required_columns:

        if column not in selected_columns:
            selected_columns.append(column)

    table = dataframe[selected_columns].copy()

    # Numerical validation.
    for column in required_columns:

        if not pd.api.types.is_numeric_dtype(
            table[column]
        ):

            print(
                f"[ERROR] {column} is not numeric."
            )

            return False

        if table[column].isna().any():

            print(
                f"[ERROR] {column} contains missing values."
            )

            return False

    table.to_csv(
        TABLE_4_1_FILE,
        index=False,
    )

    print(
        "[VALID] Table 4.1 - Main Performance"
    )

    print(
        "[CREATED] Table 4.1 - Main Performance"
    )

    print(
        f"Output: {TABLE_4_1_FILE}"
    )

    print(
        f"Rows: {len(table)}"
    )

    print(
        f"Columns: {len(table.columns)}"
    )

    return True


# =========================================================
# TABLE 4.2
# =========================================================

def generate_table_4_2():
    """
    Generate the ablation-study table.
    """

    print("-" * 70)
    print("PROCESSING: Table 4.2 - Ablation Study")
    print("-" * 70)

    if not ABLATION_SUMMARY_FILE.exists():

        print(
            "[ERROR] Ablation summary does not exist:"
        )

        print(
            ABLATION_SUMMARY_FILE
        )

        return False

    dataframe = pd.read_csv(
        ABLATION_SUMMARY_FILE
    )

    if dataframe.empty:

        print(
            "[ERROR] Ablation summary is empty."
        )

        return False

    dataframe.to_csv(
        TABLE_4_2_FILE,
        index=False,
    )

    print(
        "[VALID] Table 4.2 - Ablation Study"
    )

    print(
        "[CREATED] Table 4.2 - Ablation Study"
    )

    print(
        f"Output: {TABLE_4_2_FILE}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    return True


# =========================================================
# TABLE 4.3
# =========================================================

def generate_table_4_3():
    """
    Generate the predictive-uncertainty statistics table.
    """

    print("-" * 70)
    print("PROCESSING: Table 4.3 - Uncertainty Statistics")
    print("-" * 70)

    if not UNCERTAINTY_STATISTICS_FILE.exists():

        print(
            "[ERROR] Uncertainty statistics file does not exist:"
        )

        print(
            UNCERTAINTY_STATISTICS_FILE
        )

        return False

    dataframe = pd.read_csv(
        UNCERTAINTY_STATISTICS_FILE
    )

    if dataframe.empty:

        print(
            "[ERROR] Uncertainty statistics file is empty."
        )

        return False

    dataframe.to_csv(
        TABLE_4_3_FILE,
        index=False,
    )

    print(
        "[VALID] Table 4.3 - Uncertainty Statistics"
    )

    print(
        "[CREATED] Table 4.3 - Uncertainty Statistics"
    )

    print(
        f"Output: {TABLE_4_3_FILE}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    return True


# =========================================================
# TABLE 4.4
# =========================================================

def generate_table_4_4():
    """
    Generate the statistical-significance table.
    """

    print("-" * 70)
    print("PROCESSING: Table 4.4 - Statistical Significance")
    print("-" * 70)

    if not STATISTICAL_SIGNIFICANCE_FILE.exists():

        print(
            "[ERROR] Statistical significance file does not exist:"
        )

        print(
            STATISTICAL_SIGNIFICANCE_FILE
        )

        return False

    dataframe = pd.read_csv(
        STATISTICAL_SIGNIFICANCE_FILE
    )

    if dataframe.empty:

        print(
            "[ERROR] Statistical significance file is empty."
        )

        return False

    dataframe.to_csv(
        TABLE_4_4_FILE,
        index=False,
    )

    print(
        "[VALID] Table 4.4 - Statistical Significance"
    )

    print(
        "[CREATED] Table 4.4 - Statistical Significance"
    )

    print(
        f"Output: {TABLE_4_4_FILE}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    return True


# =========================================================
# TABLE 4.5
# =========================================================

def generate_table_4_5():
    """
    Generate the seven-method baseline comparison table.

    This table adapts Method -> Model when necessary.
    """

    print("-" * 70)
    print("PROCESSING: Table 4.5 - Baseline Comparison")
    print("-" * 70)

    if not BASELINE_COMPARISON_FILE.exists():

        print(
            "[ERROR] Baseline comparison file does not exist:"
        )

        print(
            BASELINE_COMPARISON_FILE
        )

        return False

    dataframe = pd.read_csv(
        BASELINE_COMPARISON_FILE
    )

    if dataframe.empty:

        print(
            "[ERROR] Baseline comparison file is empty."
        )

        return False

    model_column = find_model_column(
        dataframe
    )

    if model_column is None:

        print(
            "[ERROR] Could not identify the baseline "
            "method/model column."
        )

        print(
            "Available columns:"
        )

        print(
            list(dataframe.columns)
        )

        return False

    # Create thesis-standard Model column.
    if model_column != "Model":

        dataframe["Model"] = dataframe[
            model_column
        ]

    # Validate core comparison metrics.
    dataframe, missing = adapt_metric_columns(
        dataframe
    )

    if missing:

        print(
            "[ERROR] Baseline comparison is missing "
            f"required metrics: {missing}"
        )

        return False

    # Keep Model first, followed by metrics.
    preferred_columns = [
        "Model",
        "MAE",
        "RMSE",
        "PSNR",
        "SNR",
        "SSIM",
        "Missing_MAE",
        "Missing_RMSE",
        "Runtime_Seconds",
    ]

    selected_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    # If some implementation-specific columns are
    # present, preserve them after the core columns.
    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in selected_columns
        and column != model_column
    ]

    selected_columns.extend(
        remaining_columns
    )

    table = dataframe[
        selected_columns
    ].copy()

    table.to_csv(
        TABLE_4_5_FILE,
        index=False,
    )

    print(
        "[VALID] Table 4.5 - Baseline Comparison"
    )

    print(
        "[CREATED] Table 4.5 - Baseline Comparison"
    )

    print(
        f"Output: {TABLE_4_5_FILE}"
    )

    print(
        f"Rows: {len(table)}"
    )

    print(
        f"Columns: {len(table.columns)}"
    )

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("FINAL THESIS TABLE GENERATION")
    print("=" * 70)

    print()
    print(
        f"Experiment: {EXPERIMENT_NAME}"
    )

    print(
        f"Report directory: {REPORT_PATH}"
    )

    print(
        f"Thesis table directory: {THESIS_TABLE_DIR}"
    )

    print()

    results = {}

    # -----------------------------------------------------
    # Table 4.1
    # -----------------------------------------------------

    results[
        "Table_4_1_Main_Performance"
    ] = generate_table_4_1()

    print()

    # -----------------------------------------------------
    # Table 4.2
    # -----------------------------------------------------

    results[
        "Table_4_2_Ablation_Study"
    ] = generate_table_4_2()

    print()

    # -----------------------------------------------------
    # Table 4.3
    # -----------------------------------------------------

    results[
        "Table_4_3_Uncertainty_Statistics"
    ] = generate_table_4_3()

    print()

    # -----------------------------------------------------
    # Table 4.4
    # -----------------------------------------------------

    results[
        "Table_4_4_Statistical_Significance"
    ] = generate_table_4_4()

    print()

    # -----------------------------------------------------
    # Table 4.5
    # -----------------------------------------------------

    results[
        "Table_4_5_Baseline_Comparison"
    ] = generate_table_4_5()

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "table_generation_status": results,
        "source_files": {
            "evaluation_metrics": str(
                EVALUATION_METRICS_FILE
            ),
            "ablation_summary": str(
                ABLATION_SUMMARY_FILE
            ),
            "uncertainty_statistics": str(
                UNCERTAINTY_STATISTICS_FILE
            ),
            "statistical_significance": str(
                STATISTICAL_SIGNIFICANCE_FILE
            ),
            "baseline_comparison": str(
                BASELINE_COMPARISON_FILE
            ),
        },
        "metric_adapter": {
            "mae": "MAE",
            "rmse": "RMSE",
            "psnr": "PSNR",
            "snr": "SNR",
            "ssim": "SSIM",
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    created = [
        name
        for name, status in results.items()
        if status
    ]

    failed = [
        name
        for name, status in results.items()
        if not status
    ]

    print()
    print("=" * 70)
    print("THESIS TABLE GENERATION SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Tables created: {len(created)}"
    )

    for name in created:

        print(
            f"  [CREATED] {name}.csv"
        )

    print()
    print(
        f"Failed validation: {len(failed)}"
    )

    for name in failed:

        print(
            f"  [VALIDATION FAILED] {name}.csv"
        )

    print()
    print(
        "Thesis tables directory:"
    )

    print(
        THESIS_TABLE_DIR
    )

    print()

    if failed:

        print(
            "[INCOMPLETE]"
        )

        print(
            "The thesis-table generation run is "
            "not complete."
        )

    else:

        print(
            "[COMPLETE]"
        )

        print(
            "All thesis tables were generated "
            "successfully."
        )

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL THESIS TABLE GENERATION COMPLETE"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()