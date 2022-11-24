import pandas as pd
import numpy as np

data = pd.DataFrame({'age': [23, 37],
    'workclass': ['Federal-gov', 'Private'],
    'fnlgt': [314525, 172846],
    'education': ['Bachelors', 'Some-college'],
    'education-num': [13, 10],
    'marital-status': ['Never-married', 'Married-civ-spouse'],
    'occupation': ['Prof-specialty', 'Sales'],
    'relationship': ['Not-in-family', 'Husband'],
    'race': ['White', 'White'],
    'sex': ['Male', 'Male'],
    'capital-gain': [0, 0],
    'capital-loss': [0, 0],
    'hours-per-week': [40, 45],
    'native-country': ['United-States', 'United-States'],
    'salary': ['<=50K', '>50K']})

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_encoded = np.ones([2, 107])
target = [0, 1]