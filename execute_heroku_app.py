import requests
import json
data = {
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

r = requests.post("https://deployment-udacity.herokuapp.com/inference/", data=json.dumps(data))
print(f'Model prediction is {r.json()}, and the status code is {r.status_code}')

