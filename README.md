# Deploying a Machine Learning Model on Heroku with FastAPI

Project **Deploying a Machine Learning Model on Heroku with FastAPI** of Machine Learning DevOps Engineer Nanodegree Udacity

## Project Description
The goal of the project is to deploy an ML classification model on Heroku using FastAPI. DVC on AWS S3 is used for data versioning. API tests and unit tests to monitor the model performance on various slices of the data were implemented nad incorporated into a CI/CD framework using GitHub Actions.

## Files and data description
The root directory contains:

 
## Running Files
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


