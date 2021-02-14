from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from azureml.core.model import Model
import json
import pandas as pd
import joblib
import numpy as np
import rpy2.robjects as ro
import os


def test_train_model():
    test_row = '''[{'checking_balance': '< 0 DM', 'months_loan_duration': 6, 'credit_history': 'critical', 
        'purpose': 'furniture/appliances', 'amount': 1169, 'savings_balance': 'unknown', 'employment_duration': '> 7 
        years', 'percent_of_income': 4, 'years_at_residence': 4, 'age': 67, 'other_credit': 'none', 'housing': 'own', 
        'existing_loans_count': 2, 'job': 'skilled', 'dependents': 1, 'phone': 'yes'}] '''

    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])
    model = joblib.load(model_path)

    predict = ro.r('predict')

    py_data = eval(json.loads(test_row))

    # convert data to Python Dataframe
    py_df = pd.DataFrame(data=py_data)

    schema = {'checking_balance': 'object', 'months_loan_duration': 'int32', 'credit_history': 'object',
              'purpose': 'object', 'amount': 'int32', 'savings_balance': 'object',
              'employment_duration': 'object', 'percent_of_income': 'int32', 'years_at_residence': 'int32',
              'age': 'int32', 'other_credit': 'object', 'housing': 'object',
              'existing_loans_count': 'int32', 'job': 'object', 'dependents': 'int32',
              'phone': 'object'}

    py_df = py_df[list(schema.keys())]

    for col, datatype in schema.items():
        py_df[col] = py_df[col].astype(datatype)

    with localconverter(ro.default_converter + pandas2ri.converter):
        py_df = ro.conversion.py2rpy(py_df)
    print(py_df)
    result = predict(model, py_df)
    result = list(result)
    assert np.any(np.isnan(result))
