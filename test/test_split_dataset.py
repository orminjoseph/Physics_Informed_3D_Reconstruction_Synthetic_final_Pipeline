"""
======================================================================
SPLIT DIAGNOSTIC TEST
======================================================================

Purpose:
    Verify that the coverage-aware dataset splitter preserves all
    geological structures in the training subset.

This is a diagnostic test only.
It does not modify the dataset or splitter.
======================================================================
"""

from dataset.synthetic_dataset import SyntheticSeismicDataset
from dataset.split_dataset import split_dataset


# ----------------------------------------------------------------------
# 1. Create the same synthetic dataset used in the smoke test
# ----------------------------------------------------------------------

dataset = SyntheticSeismicDataset(
    num_samples=10,
    cube_size=(64, 128, 128),
    missing_probability=0.30,
    geological_mode="random",
    mask_mode="random",
    seed=42,
)


# ----------------------------------------------------------------------
# 2. Split the dataset using the coverage-aware splitter
# ----------------------------------------------------------------------

train_dataset, validation_dataset = split_dataset(dataset)


# ----------------------------------------------------------------------
# 3. Display the actual subset indices
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("SPLIT DIAGNOSTIC")
print("=" * 70)

print()
print("Training indices:")
print(train_dataset.indices)

print()
print("Validation indices:")
print(validation_dataset.indices)


# ----------------------------------------------------------------------
# 4. Inspect metadata for EVERY original dataset index
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("METADATA FOR ALL DATASET INDICES")
print("=" * 70)

all_geological_modes = set()

for index in range(len(dataset)):

    metadata = dataset.get_sample_metadata(index)

    geology = metadata["geological_mode"]
    mask_type = metadata["mask_type"]

    all_geological_modes.add(geology)

    print(
        f"Index {index:02d} | "
        f"Geology = {geology:<15} | "
        f"Mask = {mask_type}"
    )


# ----------------------------------------------------------------------
# 5. Display complete geological coverage of the dataset
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("COMPLETE DATASET GEOLOGICAL COVERAGE")
print("=" * 70)

print()

for geology in sorted(all_geological_modes):
    count = sum(
        1
        for index in range(len(dataset))
        if dataset.get_sample_metadata(index)["geological_mode"] == geology
    )

    print(f"{geology:<20}: {count}")


# ----------------------------------------------------------------------
# 6. Inspect metadata for EVERY training index
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("METADATA FOR TRAINING INDICES")
print("=" * 70)

training_geological_modes = set()

for index in train_dataset.indices:

    metadata = dataset.get_sample_metadata(index)

    geology = metadata["geological_mode"]
    mask_type = metadata["mask_type"]

    training_geological_modes.add(geology)

    print(
        f"Training index {index:02d} | "
        f"Geology = {geology:<15} | "
        f"Mask = {mask_type}"
    )


# ----------------------------------------------------------------------
# 7. Display the geological structures represented in training
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("TRAINING GEOLOGICAL SET")
print("=" * 70)

print()
print(sorted(training_geological_modes))

print()
print(
    "Number of training geological structures: "
    f"{len(training_geological_modes)}"
)


# ----------------------------------------------------------------------
# 8. Final verification
# ----------------------------------------------------------------------

expected_geologies = {
    "horizontal",
    "dipping",
    "faulted",
    "folded",
    "complex",
    "highly_complex",
}

missing_from_training = (
    expected_geologies - training_geological_modes
)

print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print()

if missing_from_training:
    print(
        "FAIL: The following geological structures are missing "
        "from training:"
    )

    for geology in sorted(missing_from_training):
        print(f"  - {geology}")

    raise AssertionError(
        "Training dataset does not contain all six geological structures."
    )


print("PASS: All six geological structures are represented in training.")

print()
print("=" * 70)
print("SPLIT DIAGNOSTIC PASSED")
print("=" * 70)