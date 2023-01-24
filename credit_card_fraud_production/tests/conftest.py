from pathlib import Path

import pytest

from fraud_analysis.config.core import TEST_DATA_DIR
from fraud_analysis.processing.data_manager import load_dataset


@pytest.fixture()
def get_test_data():

    X_test = load_dataset(Path(TEST_DATA_DIR, "X_test.csv"))
    y_test = load_dataset(Path(TEST_DATA_DIR, "y_test.csv"))

    return X_test, y_test
