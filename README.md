# Deploying a Machine Learning Model on Heroku with FastAPI

Project **Deploying a Machine Learning Model on Heroku with FastAPI** of Machine Learning DevOps Engineer Nanodegree Udacity.

## Project Description
The goal of the project is to deploy an ML classification model on Heroku using FastAPI. DVC on AWS S3 is used for data versioning. API tests and unit tests to monitor the model performance on various slices of the data were implemented and incorporated into a CI/CD framework using GitHub Actions.

## Files and data description
The root directory contains:

`starter/starter` contains the census data for the project, downloaded from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/census+income);

`starter/starter` contains scripts for data cleaning and model training;

`starter/main.py` main script for implementing a model deployment with FastAPI;

`model` contains model training output;

`tests` contains implemented unit and API tests.
 
## Installation
Clone the repo:

```bash
git clone git@github.com:raisadz/deployment_project.git
cd deployment_project
```

Install [mamba](https://pypi.org/project/mamba/).
Create a conda environment:

```bash
mamba create -n deployment_project python=3.8
```

Activate the environment:

```bash
mamba activate deployment_project 
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements:
```bash
pip install -r requirements.txt
```

## Reproducing the project
### Tracking data with DVC
The dataset has column names with white spaces. To remove them, run the following command. It will output the new data to `starter/data/clean_data/clean_data.csv`.
```bash
python -m starter.starter.clean_data
```
We want to keep track of `clean_data.csv` using DVC (data version control) and AWS S3 cloud storage. To do this, first create S3 bucket and IAM user which will have access to this bucket, see [instructions](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console). Give the user a name and select Programmatic access.
In the permissions selector, search for S3 and give it AmazonS3FullAccess or create a new policy with access to the created S3 bucket. Add the Access key ID and Secret Access key by running:
```bash
aws configure
```
Run the following commands to start tracking clean_data with dvc on aws s3: 
```bash
dvc init
dvc remote add --default s3store s3://created-s3-bucket
cd starter/data
dvc add clean_data
dvc push
```
Note, that you should add the data in the folder. Otherwise, you might have issues with pulling it. In case, you didn't initialize git, you need to run `git init` after initializing dvc. To pull the data run:
```bash
dvc pull
```
### Model training
To train the classification model:
```bash
python -m starter.starter.train_model
```
This will save the trained model to folder `model`. The script also outputs the performance metrics on the test data and on the slices of genders.

### Deploying model with FastAPI
To run the created model with FastAPI run:
```bash
uvicorn starter.main:app --host=0.0.0.0 --port=${PORT:-5000}
```
Go to `http://localhost:5000`, you should see the greeting message. To see the documentation of the implemented application navigate to `http://localhost:5000/docs`. CLick on `POST->Try it out-> Execute` to run the model prediction on the example provided in the docs. You should also see an example of a curl command to query the model. To run the model on the model/test_example.json:
```bash
curl -X 'POST' \
  'http://localhost:5000/inference/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @model/test_example.json
```

### Deploying the model with Heroku
It is possible to link GitHub repo with Heroku app. Note, that you need to set up Heroku to be able to pull data with DVC, see the [instructions](https://github.com/raisadz/deployment_project/blob/main/starter/dvc_on_heroku_instructions.md). You also need to set up access to AWS on Heroku, if using the CLI: 
```bash
heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy
```
Note, the implemented Heroku app will not be maintained here after the submission, as they are about to remove a free deployment access.

### Continious Integration and Continious Delivery (CI/CD) using GitHub Actions
This project implemented CI (located in `.github/workflows/flake8_pytest.yml`) that runs flake8 and pytest checks. The CI process also pulls data from dvc. Note, that AWS credentials were added as secrets to be able to do this. CD was enabled for the Heroku implementation. 

### Test coverage
To see test coverage:
```bash
pytest tests --cov=starter --cov-report=term-missing
```
