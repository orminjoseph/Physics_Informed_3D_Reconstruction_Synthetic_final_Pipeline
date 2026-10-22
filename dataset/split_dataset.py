"""
======================================================================
DATASET SPLITTER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Splits the complete training dataset into:

    1. Training subset
    2. Validation subset

The validation fraction is controlled by:

    VALIDATION_SPLIT

from:

    utils.config

Reproducibility
---------------
The split is deterministic through the global:

    SEED

configuration parameter.

Synthetic Dataset Coverage
--------------------------
For the synthetic seismic dataset, geological coverage is a
critical training requirement.

The synthetic dataset provides:

    get_sample_metadata(index)

which allows this module to determine the geological scenario
associated with each sample WITHOUT generating the complete
seismic cube.

When this metadata interface is available, the splitter uses
a coverage-aware strategy that guarantees:

    - every geological structure represented in the complete
      dataset remains represented in the training subset
    - training and validation samples remain disjoint
    - the requested validation size is preserved whenever
      mathematically possible
    - the split remains deterministic

This is especially important for the 10-sample smoke test.

Example:

    10 total samples
    8 training samples
    2 validation samples

The six geological structures remain represented in the
training subset.

For larger datasets, the same principle scales automatically.

General Dataset Compatibility
-----------------------------
Datasets that do not provide:

    get_sample_metadata(index)

fall back to a deterministic random split.

Important Evaluation Principle
------------------------------
This module creates ONLY the training and validation subsets.

An independent test/evaluation dataset must remain completely
separate from this procedure.

The final PhD evaluation, including:

    - controlled reconstruction experiments
    - baseline comparison
    - ablation studies
    - uncertainty evaluation
    - robustness experiments
    - statistical significance testing

must not use this training/validation split as the final test set.

Resume Compatibility
--------------------
The deterministic split is generated from the same configured
SEED every time.

Therefore, restarting or resuming training does not change which
samples belong to the training and validation subsets.

Author: Ormin Joseph
======================================================================
"""


# =====================================================================
# IMPORTS
# =====================================================================

import torch

from torch.utils.data import (
    Dataset,
    Subset,
)

from utils.config import (
    VALIDATION_SPLIT,
    SEED,
)


# =====================================================================
# HELPER: VALIDATE DATASET
# =====================================================================

def _validate_dataset(dataset):
    """
    Validate the supplied PyTorch dataset.
    """

    # -----------------------------------------------------------------
    # Dataset cannot be None.
    # -----------------------------------------------------------------

    if dataset is None:

        raise ValueError(
            "Dataset cannot be None."
        )

    # -----------------------------------------------------------------
    # Dataset must follow the PyTorch Dataset interface.
    # -----------------------------------------------------------------

    if not isinstance(
        dataset,
        Dataset,
    ):

        raise TypeError(
            "dataset must be an instance of "
            "torch.utils.data.Dataset."
        )

    # -----------------------------------------------------------------
    # Obtain dataset length.
    # -----------------------------------------------------------------

    total_size = len(dataset)

    # -----------------------------------------------------------------
    # A split requires at least two samples.
    # -----------------------------------------------------------------

    if total_size < 2:

        raise ValueError(
            "Dataset must contain at least 2 samples "
            "to create training and validation subsets."
        )

    return total_size


# =====================================================================
# HELPER: VALIDATE SPLIT CONFIGURATION
# =====================================================================

def _validate_split_configuration():
    """
    Validate VALIDATION_SPLIT and SEED.
    """

    # =================================================================
    # VALIDATE VALIDATION SPLIT TYPE
    # =================================================================

    if not isinstance(
        VALIDATION_SPLIT,
        (int, float),
    ):

        raise TypeError(
            "VALIDATION_SPLIT must be a numeric value."
        )

    # -----------------------------------------------------------------
    # Convert to float.
    # -----------------------------------------------------------------

    validation_fraction = float(
        VALIDATION_SPLIT
    )

    # -----------------------------------------------------------------
    # Reject NaN and infinity.
    # -----------------------------------------------------------------

    if not torch.isfinite(
        torch.tensor(
            validation_fraction,
            dtype=torch.float64,
        )
    ):

        raise ValueError(
            "VALIDATION_SPLIT must be finite."
        )

    # -----------------------------------------------------------------
    # Validation fraction must be strictly between zero and one.
    # -----------------------------------------------------------------

    if not (
        0.0
        <
        validation_fraction
        <
        1.0
    ):

        raise ValueError(
            "VALIDATION_SPLIT must be greater than 0 "
            "and less than 1."
        )

    # =================================================================
    # VALIDATE SEED
    # =================================================================

    if SEED is not None:

        if isinstance(
            SEED,
            bool,
        ):

            raise TypeError(
                "SEED must be an integer or None, "
                "not a boolean."
            )

        if not isinstance(
            SEED,
            int,
        ):

            raise TypeError(
                "SEED must be an integer or None."
            )

    return validation_fraction


# =====================================================================
# HELPER: CALCULATE SPLIT SIZES
# =====================================================================

def _calculate_split_sizes(total_size):
    """
    Calculate training and validation subset sizes.
    """

    # -----------------------------------------------------------------
    # Calculate requested validation size.
    # -----------------------------------------------------------------

    validation_size = int(
        round(
            total_size
            *
            float(VALIDATION_SPLIT)
        )
    )

    # -----------------------------------------------------------------
    # Always retain at least one validation sample.
    # -----------------------------------------------------------------

    validation_size = max(
        1,
        validation_size,
    )

    # -----------------------------------------------------------------
    # Never allow validation to consume the complete dataset.
    # -----------------------------------------------------------------

    validation_size = min(
        validation_size,
        total_size - 1,
    )

    # -----------------------------------------------------------------
    # Training size is whatever remains.
    # -----------------------------------------------------------------

    train_size = (
        total_size
        -
        validation_size
    )

    # -----------------------------------------------------------------
    # Safety validation.
    # -----------------------------------------------------------------

    if train_size < 1:

        raise RuntimeError(
            "Training subset contains no samples."
        )

    if validation_size < 1:

        raise RuntimeError(
            "Validation subset contains no samples."
        )

    if (
        train_size
        +
        validation_size
        !=
        total_size
    ):

        raise RuntimeError(
            "Training and validation subset sizes do not "
            "account for the complete dataset."
        )

    return (
        train_size,
        validation_size,
    )


# =====================================================================
# HELPER: CREATE LOCAL RANDOM GENERATOR
# =====================================================================

def _create_generator():
    """
    Create a local deterministic PyTorch random generator.

    The local generator does not modify PyTorch's global
    random-number generator.
    """

    generator = torch.Generator()

    if SEED is not None:

        generator.manual_seed(
            int(SEED)
        )

    else:

        generator.seed()

    return generator


# =====================================================================
# HELPER: GET GEOLOGICAL LABELS
# =====================================================================

def _get_geological_labels(
    dataset,
    total_size,
):
    """
    Obtain geological labels without generating seismic cubes.

    Returns
    -------
    list or None

        One geological label for each dataset index.

        Returns None when the dataset does not provide
        get_sample_metadata().
    """

    # -----------------------------------------------------------------
    # Check whether the dataset supports metadata-based scheduling.
    # -----------------------------------------------------------------

    metadata_method = getattr(
        dataset,
        "get_sample_metadata",
        None,
    )

    if metadata_method is None:

        return None

    # -----------------------------------------------------------------
    # Obtain metadata for every sample.
    #
    # This is lightweight because the synthetic dataset's metadata
    # method does not generate seismic volumes.
    # -----------------------------------------------------------------

    labels = []

    for index in range(
        total_size
    ):

        metadata = metadata_method(
            index
        )

        # -------------------------------------------------------------
        # Metadata must be dictionary-like.
        # -------------------------------------------------------------

        if not isinstance(
            metadata,
            dict,
        ):

            raise TypeError(
                "get_sample_metadata(index) must return "
                "a dictionary."
            )

        # -------------------------------------------------------------
        # Geological mode must be available.
        # -------------------------------------------------------------

        if "geological_mode" not in metadata:

            raise KeyError(
                "Dataset metadata must contain "
                "'geological_mode'."
            )

        labels.append(
            str(
                metadata["geological_mode"]
            )
        )

    return labels


# =====================================================================
# HELPER: COVERAGE-AWARE SPLIT
# =====================================================================

def _coverage_aware_split(
    dataset,
    total_size,
    train_size,
    validation_size,
    generator,
):
    """
    Create a deterministic coverage-aware split.

    The primary requirement is:

        Every geological structure represented in the
        complete dataset must remain represented in the
        training subset.

    This is achieved by constructing the validation subset
    only from samples belonging to geological structures that
    have at least one additional sample remaining in training.

    Therefore, no geological structure can disappear entirely
    from the training subset.

    This is particularly important for:

        num_samples = 10
        validation_split = 0.20

    where:

        training   = 8
        validation = 2

    and all six geological structures must remain in training.
    """

    # =================================================================
    # GET GEOLOGICAL LABELS
    # =================================================================

    labels = _get_geological_labels(
        dataset,
        total_size,
    )

    # -----------------------------------------------------------------
    # This function should only be called when metadata exist.
    # -----------------------------------------------------------------

    if labels is None:

        raise RuntimeError(
            "Coverage-aware splitting requires "
            "get_sample_metadata(index)."
        )

    # =================================================================
    # COUNT SAMPLES PER GEOLOGICAL STRUCTURE
    # =================================================================

    label_to_indices = {}

    for index, label in enumerate(labels):

        label_to_indices.setdefault(
            label,
            [],
        ).append(index)

    # =================================================================
    # DETERMINE WHETHER COVERAGE CAN BE GUARANTEED
    # =================================================================
    #
    # To retain every geological structure in training, at least one
    # sample from every structure must remain in training.
    #
    # Therefore:
    #
    #     maximum_validation_size
    #
    # is:
    #
    #     total samples - number of geological structures
    #
    # If the requested validation set is larger than this, the exact
    # requested split cannot preserve every geological structure.
    # =================================================================

    number_of_geological_structures = len(
        label_to_indices
    )

    maximum_validation_size = (
        total_size
        -
        number_of_geological_structures
    )

    if (
        validation_size
        >
        maximum_validation_size
    ):

        raise ValueError(
            "The requested validation split cannot preserve "
            "at least one training sample for every geological "
            "structure.\n"
            f"Total samples: {total_size}\n"
            f"Geological structures: "
            f"{number_of_geological_structures}\n"
            f"Requested validation samples: "
            f"{validation_size}\n"
            f"Maximum validation samples while preserving "
            f"geological coverage: "
            f"{maximum_validation_size}\n\n"
            "Reduce VALIDATION_SPLIT or increase the dataset "
            "size."
        )

    # =================================================================
    # RANDOMIZE SAMPLE INDICES
    # =================================================================
    #
    # A local generator is used, so the procedure remains
    # reproducible without modifying the global PyTorch RNG.
    # =================================================================

    random_order = torch.randperm(
        total_size,
        generator=generator,
    ).tolist()

    # =================================================================
    # BUILD RANDOMIZED GEOLOGICAL GROUPS
    # =================================================================
    #
    # The random permutation determines the order in which samples
    # are considered, while the coverage rule prevents removing the
    # final training sample of any geological structure.
    # =================================================================

    randomized_groups = {
        label: []
        for label in label_to_indices
    }

    for index in random_order:

        label = labels[index]

        randomized_groups[label].append(
            index
        )

    # =================================================================
    # SELECT VALIDATION INDICES
    # =================================================================
    #
    # A sample may enter validation only if another sample belonging
    # to the same geological structure remains available for training.
    #
    # Therefore:
    #
    #     current_group_size > 1
    #
    # is required before removing a sample.
    # =================================================================

    validation_indices = []

    # -----------------------------------------------------------------
    # First pass:
    #
    # Randomly consider all samples and remove samples from validation
    # only when their geological group still has another member.
    # -----------------------------------------------------------------

    for index in random_order:

        if len(
            validation_indices
        ) >= validation_size:

            break

        label = labels[index]

        # -------------------------------------------------------------
        # Count how many samples of this geological structure are
        # currently still available.
        # -------------------------------------------------------------

        available_group = (
            randomized_groups[label]
        )

        # -------------------------------------------------------------
        # Do not remove the last sample of a geological structure.
        # -------------------------------------------------------------

        if len(available_group) <= 1:

            continue

        # -------------------------------------------------------------
        # Remove this sample from the group's available training pool.
        # -------------------------------------------------------------

        available_group.remove(
            index
        )

        validation_indices.append(
            index
        )

    # =================================================================
    # VERIFY REQUESTED VALIDATION SIZE
    # =================================================================

    if len(
        validation_indices
    ) != validation_size:

        raise RuntimeError(
            "Coverage-aware splitting could not construct "
            "the requested validation subset while preserving "
            "geological coverage in training."
        )

    # =================================================================
    # CONSTRUCT TRAINING INDICES
    # =================================================================

    validation_index_set = set(
        validation_indices
    )

    train_indices = [
        index
        for index in range(total_size)
        if index not in validation_index_set
    ]

    # =================================================================
    # VERIFY TRAINING SIZE
    # =================================================================

    if len(
        train_indices
    ) != train_size:

        raise RuntimeError(
            "Coverage-aware split produced an incorrect "
            "training subset size."
        )

    # =================================================================
    # VERIFY GEOLOGICAL COVERAGE
    # =================================================================

    training_labels = [
        labels[index]
        for index in train_indices
    ]

    training_geological_structures = set(
        training_labels
    )

    complete_geological_structures = set(
        labels
    )

    if (
        training_geological_structures
        !=
        complete_geological_structures
    ):

        missing_structures = (
            complete_geological_structures
            -
            training_geological_structures
        )

        raise RuntimeError(
            "Training geological coverage failure.\n"
            f"Missing structures from training: "
            f"{sorted(missing_structures)}"
        )

    # =================================================================
    # SORT INDICES
    # =================================================================
    #
    # Sorting makes the resulting Subset indices easier to inspect and
    # compare across runs.
    #
    # The selection itself was already controlled by the deterministic
    # random generator.
    # =================================================================

    train_indices.sort()

    validation_indices.sort()

    return (
        train_indices,
        validation_indices,
        labels,
    )


# =====================================================================
# HELPER: STANDARD RANDOM SPLIT
# =====================================================================

def _random_split_indices(
    total_size,
    train_size,
    validation_size,
    generator,
):
    """
    Create a deterministic ordinary random split.

    Used for datasets that do not provide geological metadata.
    """

    # -----------------------------------------------------------------
    # Generate deterministic random permutation.
    # -----------------------------------------------------------------

    indices = torch.randperm(
        total_size,
        generator=generator,
    ).tolist()

    # -----------------------------------------------------------------
    # First portion becomes training.
    # -----------------------------------------------------------------

    train_indices = indices[
        :train_size
    ]

    # -----------------------------------------------------------------
    # Remaining portion becomes validation.
    # -----------------------------------------------------------------

    validation_indices = indices[
        train_size:
        train_size + validation_size
    ]

    return (
        train_indices,
        validation_indices,
    )


# =====================================================================
# MAIN DATASET SPLITTING FUNCTION
# =====================================================================

def split_dataset(dataset):
    """
    Split a complete dataset into training and validation subsets.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Complete dataset to be divided.

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        Training subset.

    validation_dataset : torch.utils.data.Subset
        Validation subset.

    Notes
    -----
    For the synthetic dataset, geological coverage is preserved
    in the training subset whenever the dataset provides the
    get_sample_metadata() interface.

    For other datasets, a deterministic random split is used.

    This function does not create an independent test dataset.
    """

    # =================================================================
    # VALIDATE DATASET
    # =================================================================

    total_size = _validate_dataset(
        dataset
    )

    # =================================================================
    # VALIDATE SPLIT CONFIGURATION
    # =================================================================

    validation_fraction = (
        _validate_split_configuration()
    )

    # =================================================================
    # CALCULATE SPLIT SIZES
    # =================================================================

    (
        train_size,
        validation_size,
    ) = _calculate_split_sizes(
        total_size
    )

    # =================================================================
    # CREATE LOCAL RANDOM GENERATOR
    # =================================================================

    generator = _create_generator()

    # =================================================================
    # CHECK FOR COVERAGE-AWARE METADATA
    # =================================================================

    metadata_method = getattr(
        dataset,
        "get_sample_metadata",
        None,
    )

    # =================================================================
    # CREATE SPLIT
    # =================================================================

    if metadata_method is not None:

        # -------------------------------------------------------------
        # Synthetic dataset path.
        #
        # Geological coverage is explicitly preserved in training.
        # -------------------------------------------------------------

        (
            train_indices,
            validation_indices,
            geological_labels,
        ) = _coverage_aware_split(
            dataset=dataset,
            total_size=total_size,
            train_size=train_size,
            validation_size=validation_size,
            generator=generator,
        )

        split_strategy = (
            "coverage-aware geological split"
        )

    else:

        # -------------------------------------------------------------
        # Generic dataset path.
        #
        # No geological metadata are available, so the normal
        # deterministic random split is used.
        # -------------------------------------------------------------

        (
            train_indices,
            validation_indices,
        ) = _random_split_indices(
            total_size=total_size,
            train_size=train_size,
            validation_size=validation_size,
            generator=generator,
        )

        geological_labels = None

        split_strategy = (
            "deterministic random split"
        )

    # =================================================================
    # CREATE PYTORCH SUBSETS
    # =================================================================

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    validation_dataset = Subset(
        dataset,
        validation_indices,
    )

    # =================================================================
    # VALIDATE SUBSET TYPES
    # =================================================================

    if not isinstance(
        train_dataset,
        Subset,
    ):

        raise RuntimeError(
            "Training split did not produce a "
            "torch.utils.data.Subset."
        )

    if not isinstance(
        validation_dataset,
        Subset,
    ):

        raise RuntimeError(
            "Validation split did not produce a "
            "torch.utils.data.Subset."
        )

    # =================================================================
    # VERIFY SPLIT SIZES
    # =================================================================

    if len(
        train_dataset
    ) != train_size:

        raise RuntimeError(
            "Training subset size does not match "
            "the requested split size."
        )

    if len(
        validation_dataset
    ) != validation_size:

        raise RuntimeError(
            "Validation subset size does not match "
            "the requested split size."
        )

    # =================================================================
    # VERIFY COMPLETE DATASET COVERAGE
    # =================================================================

    if (
        len(train_dataset)
        +
        len(validation_dataset)
        !=
        total_size
    ):

        raise RuntimeError(
            "Training and validation subsets do not "
            "account for the complete dataset."
        )

    # =================================================================
    # VERIFY NO INDEX OVERLAP
    # =================================================================

    train_index_set = set(
        train_dataset.indices
    )

    validation_index_set = set(
        validation_dataset.indices
    )

    overlapping_indices = (
        train_index_set
        &
        validation_index_set
    )

    if overlapping_indices:

        raise RuntimeError(
            "Data leakage detected: training and validation "
            "subsets contain overlapping sample indices."
        )

    # =================================================================
    # VERIFY COMPLETE INDEX COVERAGE
    # =================================================================

    combined_indices = (
        train_index_set
        |
        validation_index_set
    )

    expected_indices = set(
        range(total_size)
    )

    if combined_indices != expected_indices:

        raise RuntimeError(
            "Training and validation subsets do not "
            "cover every original dataset sample exactly once."
        )

    # =================================================================
    # VERIFY GEOLOGICAL COVERAGE AGAIN
    # =================================================================
    #
    # This is intentionally checked after Subset construction.
    #
    # The purpose is to make the experimental requirement explicit
    # and fail immediately if a future modification breaks it.
    # =================================================================

    if geological_labels is not None:

        complete_structures = set(
            geological_labels
        )

        training_structures = {
            geological_labels[index]
            for index in train_dataset.indices
        }

        missing_training_structures = (
            complete_structures
            -
            training_structures
        )

        if missing_training_structures:

            raise RuntimeError(
                "Training subset does not contain every "
                "geological structure.\n"
                f"Missing structures: "
                f"{sorted(missing_training_structures)}"
            )

    # =================================================================
    # CALCULATE ACTUAL VALIDATION FRACTION
    # =================================================================

    actual_validation_fraction = (
        len(validation_dataset)
        /
        total_size
    )

    # =================================================================
    # DISPLAY SPLIT INFORMATION
    # =================================================================

    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(
        f"Total samples            : "
        f"{total_size}"
    )

    print(
        f"Training samples         : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples       : "
        f"{len(validation_dataset)}"
    )

    print(
        f"Requested validation     : "
        f"{validation_fraction:.4f}"
    )

    print(
        f"Actual validation        : "
        f"{actual_validation_fraction:.4f}"
    )

    print(
        f"Random seed              : "
        f"{SEED}"
    )

    print(
        f"Split strategy           : "
        f"{split_strategy}"
    )

    print(
        "Training/validation "
        "overlap                 : 0"
    )

    # =================================================================
    # DISPLAY GEOLOGICAL COVERAGE
    # =================================================================

    if geological_labels is not None:

        print()
        print("Training Geological Coverage")
        print("-" * 70)

        training_structures = {}

        for index in train_dataset.indices:

            geological_mode = (
                geological_labels[index]
            )

            training_structures[
                geological_mode
            ] = (
                training_structures.get(
                    geological_mode,
                    0
                )
                + 1
            )

        for geological_mode in sorted(
            training_structures
        ):

            print(
                f"{geological_mode:<20}: "
                f"{training_structures[geological_mode]}"
            )

        print()
        print(
            "All geological structures "
            "represented in training : "
            f"{len(training_structures) == len(set(geological_labels))}"
        )

    print("=" * 70)
    print()

    # =================================================================
    # RETURN DATASETS
    # =================================================================

    return (
        train_dataset,
        validation_dataset,
    )


# =====================================================================
# END OF MODULE
# =====================================================================