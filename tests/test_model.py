# from tests.test_data import data, cat_features
import numpy as np
import pandas as pd
import pytest
import sklearn

from starter.starter.ml.data import process_data
from starter.starter.ml.model import (compute_model_metrics, inference,
                                      train_model)


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "age": [23, 37],
            "workclass": ["Federal-gov", "Private"],
            "fnlgt": [314525, 172846],
            "education": ["Bachelors", "Some-college"],
            "education-num": [13, 10],
            "marital-status": ["Never-married", "Married-civ-spouse"],
            "occupation": ["Prof-specialty", "Sales"],
            "relationship": ["Not-in-family", "Husband"],
            "race": ["White", "White"],
            "sex": ["Male", "Male"],
            "capital-gain": [0, 0],
            "capital-loss": [0, 0],
            "hours-per-week": [40, 45],
            "native-country": ["United-States", "United-States"],
            "salary": ["<=50K", ">50K"],
        }
    )


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


def test_preprocess_data(data, cat_features):
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(X_test) == np.ndarray
    assert type(y_test) == np.ndarray
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer

def test_preprocess_data_None(data):
    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=None,
        label='salary',
        training=True
    )

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=None,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(X_test) == np.ndarray
    assert type(y_test) == np.ndarray
    
def test_training_inference():
    X_encoded = np.ones([2, 107])
    target = np.array([0, 1])
    model = train_model(X_train=X_encoded, y_train=target)
    preds = inference(model, X_encoded)

    assert type(model) == sklearn.linear_model._logistic.LogisticRegression
    assert type(preds) == np.ndarray


def test_compute_model_metrics():
    y_test = np.array([0, 1])
    preds = np.array([1, 1])
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64
