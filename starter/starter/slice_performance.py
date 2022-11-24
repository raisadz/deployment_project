import pandas as pd
from starter.starter.ml.model import compute_model_metrics

def slice_performance(df, col, y_test, preds):
    '''
    Function computes performance on a slice of a chosen categorical column

    Inputs:
    -------
    df: (pd.DataFrame) Data with features, col should be present
    col: (str) Name of the categorical column for slicing
    y_test: (np.ndarray) target values
    preds: (np.ndarray) predictions array

    Outputs:
    --------
    slice_mat (pd.DataFrame) df with index as unique feature values and columns- performace metrics
    '''
    assert df[col].dtype == 'O', (f'{col} is not categorical')

    cat_groups = sorted(list(df[col].unique()))
    precision_list = []
    recall_list = []
    fbeta_list = []
    for i, cat in enumerate(cat_groups):
        cat_slice = (df[col] == cat)
        precision, recall, fbeta = compute_model_metrics(y_test[cat_slice], preds[cat_slice])
        precision_list.append(precision)
        recall_list.append(recall)
        fbeta_list.append(fbeta)
    slice_mat = pd.DataFrame(df.groupby(col)[col].count()).rename({col: 'count'}, axis = 'columns')
    slice_mat['precision'] = precision_list
    slice_mat['recall'] = recall_list
    slice_mat['fbeta'] = fbeta_list
    return slice_mat