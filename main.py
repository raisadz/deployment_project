from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd
from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()
@app.get('/')
async def say_hello():
    return {"greeting": "Welcome!"}


model = joblib.load('model/model.pkl')
encoder = joblib.load('model/encoder.pkl')
lb = joblib.load('model/lb.pkl')


class Data(BaseModel):
    age: int = Field(example = 30)
    workclass: str = Field(example = 'State-gov')
    fnlgt: int = Field(example = 77516)
    education: str = Field(example = 'Bachelors')
    education_num: int = Field(alias="education-num", example = 13)
    marital_status: str = Field(alias="marital-status", example = 'Never-married')
    occupation: str = Field(example = 'Adm-clerical')
    relationship: str = Field(example = 'Not-in-family')
    race: str = Field(example = 'Black')
    sex: str = Field(example = 'Female')
    capital_gain: int = Field(alias="capital-gain", example = 0)
    capital_loss: int = Field(alias="capital-loss", example = 0)
    hours_per_week: int = Field(alias="hours-per-week", example = 40)
    native_country: str = Field(alias="native-country", example = 'Cuba')
    salary: str = Field(example = '<=50K')

@app.post('/inference/')
async def predict(data: Data):
    test = pd.DataFrame([data.dict(by_alias=True)], index = [0])

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

    X_test, y_test, _, _ = process_data(
                test, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb)
    preds = inference(model, X_test)
    return preds