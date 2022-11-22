# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
# Add code to load in the data.
data = pd.read_csv('../data/clean_data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
            test, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)
joblib.dump(model, 'model.pkl')

model_load=joblib.load('model.pkl')
preds = inference(model_load, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'Precision: {precision}, recall: {recall}, fbeta: {fbeta}')
