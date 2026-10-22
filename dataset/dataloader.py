"""
======================================================================
PYTORCH DATALOADER
======================================================================

Physics-Informed 3D Encoder-Decoder Framework
with Predictive Uncertainty for Seismic Data Reconstruction

Purpose
-------
Creates PyTorch DataLoaders for:

    1. Training
    2. Validation
    3. Testing / Evaluation

The DataLoader converts individual dataset samples into mini-batches
for neural-network training and evaluation.

Tensor convention
-----------------

Individual dataset sample:

    [C, D, H, W]

DataLoader batch:

    [B, C, D, H, W]

where:

    B = batch size
    C = channel
    D = depth
    H = crossline
    W = inline

The actual dataset construction is handled by:

    dataset/build_dataset.py

The training/validation split is handled by:

    dataset/split_dataset.py

This module is responsible only for DataLoader construction.

Important
---------
This module does not:

    - construct datasets
    - split datasets
    - modify seismic tensors
    - generate masks
    - perform normalization
    - perform augmentation
    - perform training
    - perform evaluation

It only constructs PyTorch DataLoader objects.

Author: Ormin Joseph
======================================================================
"""

# =====================================================================
# IMPORTS
# =====================================================================

from torch.utils.data import (
    DataLoader,
    Dataset
)

from utils.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS
)


# =====================================================================
# DATALOADER FACTORY
# =====================================================================

def create_dataloader(
    dataset,
    batch_size=None,
    shuffle=True,
    num_workers=None
):
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset or Dataset subset to be loaded.

    batch_size : int, optional
        Number of samples per mini-batch.

        If None, BATCH_SIZE from utils.config is used.

    shuffle : bool, default=True
        Whether samples should be shuffled.

        Recommended:

            Training   -> True
            Validation -> False
            Testing    -> False

    num_workers : int, optional
        Number of worker processes used for loading data.

        If None, NUM_WORKERS from utils.config is used.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured PyTorch DataLoader.

    Notes
    -----
    The DataLoader does not explicitly modify tensor dimensions.

    If the dataset returns:

        [C, D, H, W]

    PyTorch's default collation produces:

        [B, C, D, H, W]

    for tensor fields.

    Metadata fields returned by the dataset are also collated
    according to PyTorch's default collation rules.
    """

    # =================================================================
    # 1. VALIDATE DATASET
    # =================================================================

    if dataset is None:

        raise ValueError(
            "dataset cannot be None."
        )

    # -----------------------------------------------------------------
    # The object must follow the PyTorch Dataset interface.
    # -----------------------------------------------------------------

    if not isinstance(
        dataset,
        Dataset
    ):

        raise TypeError(
            "dataset must be an instance of "
            "torch.utils.data.Dataset."
        )

    # -----------------------------------------------------------------
    # A DataLoader cannot meaningfully operate on an empty dataset.
    # -----------------------------------------------------------------

    dataset_size = len(
        dataset
    )

    if dataset_size < 1:

        raise ValueError(
            "dataset must contain at least one sample."
        )

    # =================================================================
    # 2. USE CONFIGURATION DEFAULTS
    # =================================================================

    if batch_size is None:

        batch_size = BATCH_SIZE

    if num_workers is None:

        num_workers = NUM_WORKERS

    # =================================================================
    # 3. VALIDATE BATCH SIZE
    # =================================================================

    # -----------------------------------------------------------------
    # bool is deliberately rejected because Python treats bool as an
    # integer subclass.
    # -----------------------------------------------------------------

    if isinstance(
        batch_size,
        bool
    ):

        raise TypeError(
            "batch_size must be an integer, "
            "not a boolean."
        )

    if not isinstance(
        batch_size,
        int
    ):

        raise TypeError(
            "batch_size must be an integer."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be at least 1."
        )

    # =================================================================
    # 4. VALIDATE NUMBER OF WORKERS
    # =================================================================

    if isinstance(
        num_workers,
        bool
    ):

        raise TypeError(
            "num_workers must be an integer, "
            "not a boolean."
        )

    if not isinstance(
        num_workers,
        int
    ):

        raise TypeError(
            "num_workers must be an integer."
        )

    if num_workers < 0:

        raise ValueError(
            "num_workers cannot be negative."
        )

    # =================================================================
    # 5. VALIDATE SHUFFLE OPTION
    # =================================================================

    if not isinstance(
        shuffle,
        bool
    ):

        raise TypeError(
            "shuffle must be a boolean."
        )

    # =================================================================
    # 6. VALIDATE CONFIGURATION FLAGS
    # =================================================================

    if not isinstance(
        PIN_MEMORY,
        bool
    ):

        raise TypeError(
            "PIN_MEMORY must be a boolean in utils/config.py."
        )

    if not isinstance(
        PERSISTENT_WORKERS,
        bool
    ):

        raise TypeError(
            "PERSISTENT_WORKERS must be a boolean "
            "in utils/config.py."
        )

    # =================================================================
    # 7. CONFIGURE PERSISTENT WORKERS
    # =================================================================
    #
    # PyTorch requires persistent_workers=True only when
    # num_workers > 0.
    #
    # Therefore:
    #
    #     num_workers = 0
    #         -> persistent_workers must be False
    #
    # For the current project configuration:
    #
    #     NUM_WORKERS = 0
    #     PERSISTENT_WORKERS = False
    #
    # the resulting setting is:
    #
    #     persistent_workers = False
    #
    # =================================================================

    if (
        PERSISTENT_WORKERS
        and
        num_workers == 0
    ):

        raise ValueError(
            "PERSISTENT_WORKERS=True requires "
            "num_workers > 0."
        )

    persistent_workers = (
        PERSISTENT_WORKERS
    )

    # =================================================================
    # 8. CREATE DATALOADER
    # =================================================================
    #
    # No custom collate function is required because the current
    # dataset returns tensors and metadata that can be handled by
    # PyTorch's default collation mechanism.
    #
    # The tensor dimension convention remains:
    #
    #     Individual:
    #         [C,D,H,W]
    #
    #     Batch:
    #         [B,C,D,H,W]
    #
    # =================================================================

    loader = DataLoader(

        dataset=dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=PIN_MEMORY,

        persistent_workers=persistent_workers
    )

    # =================================================================
    # 9. VALIDATE CREATED DATALOADER
    # =================================================================

    if loader is None:

        raise RuntimeError(
            "DataLoader creation failed."
        )

    # -----------------------------------------------------------------
    # For a non-empty dataset, at least one batch must be available.
    # -----------------------------------------------------------------

    if len(loader) < 1:

        raise RuntimeError(
            "Created DataLoader contains zero batches."
        )

    # =================================================================
    # 10. DISPLAY CONFIGURATION
    # =================================================================

    print()
    print("-" * 70)
    print("DATALOADER CREATED")
    print("-" * 70)

    print(
        f"Dataset size       : "
        f"{dataset_size}"
    )

    print(
        f"Batch size         : "
        f"{batch_size}"
    )

    print(
        f"Number of batches  : "
        f"{len(loader)}"
    )

    print(
        f"Shuffle            : "
        f"{shuffle}"
    )

    print(
        f"Workers            : "
        f"{num_workers}"
    )

    print(
        f"Pin memory         : "
        f"{PIN_MEMORY}"
    )

    print(
        f"Persistent workers : "
        f"{persistent_workers}"
    )

    print("-" * 70)
    print()

    # =================================================================
    # 11. RETURN DATALOADER
    # =================================================================

    return loader


# =====================================================================
# END OF MODULE
# =====================================================================