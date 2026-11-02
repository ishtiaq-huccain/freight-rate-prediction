"""The provided assessment files must stay byte-for-byte unchanged.

Hashes were recorded when the files were first received (before any work).
The December inputs file is excluded: the assessment asks us to fill it.
"""

from __future__ import annotations

import hashlib

import pytest

from freight_rate.config import PROJECT_ROOT

ORIGINAL_SHA256 = {
    "score.py": "8da8b6bf3e39078f34c9998d14dcca4f7aa64ad5ddea1ce4449cd362168d2a28",
    "data/train_test.csv": "892980fd20bac7ac0ca4ec31d3981fca48d52dec88e4f02642ab955f1efb40fe",
    "data/validation.csv": "f6cdc7c367b397b90b319fc7ad30fd0e1893fd6a714816dda54c303b0fe71151",
    "data/validation_predictions_template.csv": (
        "79de15adabcf807b8636b9b47ebe964d451fbcb343056c44a2c4b54a219891c6"
    ),
}


@pytest.mark.parametrize("relative_path", sorted(ORIGINAL_SHA256))
def test_provided_file_unchanged(relative_path):
    digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
    assert digest == ORIGINAL_SHA256[relative_path], f"{relative_path} was modified"
