import json

import pytest
from fastapi.testclient import TestClient

from starter.main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!"}


@pytest.fixture
def salary_low():
    return {
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Cuba",
        "salary": "<=50K",
    }


@pytest.fixture
def salary_high():
    return {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 167309,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 1902,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": ">50K",
    }


def test_post(salary_low, salary_high):
    r_low = client.post("/inference/", data=json.dumps(salary_low))
    r_high = client.post("/inference/", data=json.dumps(salary_high))
    assert r_low.status_code == 200
    assert r_high.status_code == 200
    assert r_low.json() == ["<=50K"]
    assert r_high.json() == [">50K"]
