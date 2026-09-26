"""Central paths and tunable constants for the Bijection ER pipeline."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# DATA_DIR must contain train/ and test/ subdirectories of .tsv files.
# Override with BIJECTION_DATA_DIR when the dataset lives elsewhere.
_DEFAULT_DATA = (
    "/private/tmp/claude-502/-Users-aakarsh/e49f281a-9ae0-40b0-b28e-4116eda302d0"
    "/scratchpad/amazon_er/student_resource/dataset"
)
DATA_DIR = Path(os.environ.get("BIJECTION_DATA_DIR", _DEFAULT_DATA))

# .../Bijection/code/business_entity_resolution/src/bijection/config.py -> parents[4]
PROJECT_ROOT = Path(os.environ.get("BIJECTION_ROOT", Path(__file__).resolve().parents[4]))
WORK_DIR = Path(os.environ.get("BIJECTION_WORK_DIR", PROJECT_ROOT / "work"))
OUTPUT_DIR = Path(os.environ.get("BIJECTION_OUTPUT_DIR", PROJECT_ROOT / "output"))

TRAIN = {
    "s1": DATA_DIR / "train" / "train_source1.tsv",
    "s2": DATA_DIR / "train" / "train_source2.tsv",
    "s3": DATA_DIR / "train" / "train_source3.tsv",
    "gt": DATA_DIR / "train" / "train_ground_truth.tsv",
}
TEST = {
    "s1": DATA_DIR / "test" / "test_source1.tsv",
    "s2": DATA_DIR / "test" / "test_source2.tsv",
    "s3": DATA_DIR / "test" / "test_source3.tsv",
}

# ---------------------------------------------------------------- blocking
# Max candidates emitted per Source-1 entity. This is the number that decides
# both the recall ceiling and the reduction ratio reported in candidate_pairs.tsv.
TOP_K = 20

# A token appearing in more than this many records of a country-shard is treated
# as non-discriminative and never used as half of a conjunctive blocking key.
MAX_TOKEN_DF = 200_000

# Posting lists longer than this are skipped at probe time (runaway generic keys).
MAX_POSTING = 400

# ---------------------------------------------------------------- decision
# Observed per-entity caps in train ground truth (full scan): S2<=5, S3<=6.
# We allow one slot of headroom in case the test generator drew slightly higher.
MAX_PER_SOURCE = {"S2": 6, "S3": 7}

RANDOM_SEED = 17

__all__ = [
    "DATA_DIR", "PROJECT_ROOT", "WORK_DIR", "OUTPUT_DIR", "TRAIN", "TEST",
    "TOP_K", "MAX_TOKEN_DF", "MAX_POSTING", "MAX_PER_SOURCE", "RANDOM_SEED",
]
