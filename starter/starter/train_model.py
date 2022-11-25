# Script to train machine learning model.

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.starter.ml.data import process_data
from starter.starter.ml.model import (compute_model_metrics, inference,
                                      train_model)
from starter.starter.slice_performance import slice_performance

# Add code to load in the data.
data = pd.read_csv("starter/data/clean_data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
test.iloc[:1, :].to_json("model/test_example.json", orient="records")


model_load = joblib.load("model/model.pkl")
preds = inference(model_load, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, recall: {recall}, fbeta: {fbeta}")

print(slice_performance(test, "sex", y_test, preds))

slice_path = "model/slice_output.txt"
if os.path.exists(slice_path):
    os.remove(slice_path)
tfile = open(slice_path, "w")
tfile.write(slice_performance(test, "sex", y_test, preds).to_string())
tfile.close()
