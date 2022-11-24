# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Raisa created this model card as part of the Udacity Machine Learning DevOps nanodegree "Deploying a Machine Learning Model on Heroku with FastAP". It is logistic regression using the default hyperparameters in scikit-learn==1.1.3 and random state=42.

## Intended Use
This model predicts whether a person makes over 50K a year.

## Training Data
The data was obtained from the UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/census+income. The original data contains 48842 observations and 14 attributes. The training data was obtained by randomly sampling 80% of the data. No stratification was used. OneHotEncoder was used to encode categorical variables, and LabelBinarizer was used to encode the target variable.

## Evaluation Data
The evaluation data set contains 20% randomly selectedly observations from the original data.

## Metrics
Three metrics were used for the model evaluation: precision, recall, and fbeta. The trained model obtains a precision of 0.73, recall of 0.27, and fbeta of 0.39 on the test data.  

## Ethical Considerations
The model performs much better on the 'male' dataset. The precision of the model with the fixed feature 'male' achieves a precision of 0.77, whereas for 'female' a precision is 0.53. This can be explained by the imbalanced training data: the ratio of males to females is around 2:1.

## Caveats and Recommendations
Apart from the dataset being gender imbalanced, it is also imbalanced in a race (85% of the observations have a 'White' race). In addition, the high earners are under-represented, this percentage is 24%. 