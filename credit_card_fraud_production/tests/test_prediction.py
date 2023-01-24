import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from fraud_analysis.predict import make_prediction


def test_make_prediction(get_test_data):
    # Given
    expected_number_of_predictions = 56962
    expected_number_of_predicted_not_frauds = 56818
    expected_number_of_predicted_frauds = 144

    # Make prediction on the test data using the persisted trained pipeline
    res = make_prediction(input_data=get_test_data[0])
    y_test_pred = res["predictions"]
    number_of_predicted_not_frauds = np.unique(y_test_pred, return_counts=True)[1][0]
    number_of_predicted_frauds = np.unique(y_test_pred, return_counts=True)[1][1]

    # Then
    assert len(y_test_pred) == expected_number_of_predictions
    assert number_of_predicted_not_frauds == expected_number_of_predicted_not_frauds
    assert number_of_predicted_frauds == expected_number_of_predicted_frauds


def test_model_perf_on_test_set(get_test_data):

    # Given
    expected_auprc_fraud = 0.77
    expected_f1_score = 0.66

    # Make prediction on the test data using the persisted trained pipeline
    res = make_prediction(input_data=get_test_data[0])

    y_test = get_test_data[1]
    y_test_pred = res["predictions"]
    y_test_pred_scores = res["predicted_proba"]

    y_val_dummies = pd.get_dummies(y_test.iloc[:, 0], drop_first=False).values

    actual_auprcs_fraud = average_precision_score(
        y_val_dummies[:, 1], y_test_pred_scores[:, 1]
    )

    actual_f1_score = f1_score(y_test, y_test_pred)

    # Then
    assert math.isclose(expected_f1_score, actual_f1_score, abs_tol=0.01)
    assert math.isclose(expected_auprc_fraud, actual_auprcs_fraud, abs_tol=0.01)
