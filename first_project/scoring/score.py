import numpy
import joblib
import os
import rpy2
import rpy2.robjects as ro
import json
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from azureml.core.model import Model

model, predict,  = rpy2.robjects.functions.SignatureTranslatedFunction


def init():
    # load the model from file into a global object
    global model, predict

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)s
    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])
    model = joblib.load(model_path)
    predict = ro.r('predict')


input_sample = numpy.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
output_sample = numpy.array([
    5021.509689995557,
    3693.645386402646])


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
# @input_schema('data', dict)
# @output_schema(NumpyParameterType(output_sample))
def run(data, request_headers):
    py_data = eval(json.loads(data)['data'])

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

    result: iter = predict(model, py_df)
    levels = list(result.levels)
    result = pd.DataFrame({'result': list(result)})
    result = result.replace({1: levels[0], 2: levels[1]})
    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'transparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
        request_headers.get("X-Ms-Request-Id", ""),
        request_headers.get("Traceparent", ""),
        len(result)
    ))

    return {"result": result.values}


if __name__ == "__main__":
    # Test scoring
    init()
    test_row = '''{"data": "[{'checking_balance': '< 0 DM', 'months_loan_duration': 6, 'credit_history': 'critical', 
    'purpose': 'furniture/appliances', 'amount': 1169, 'savings_balance': 'unknown', 'employment_duration': '> 7 
    years', 'percent_of_income': 4, 'years_at_residence': 4, 'age': 67, 'other_credit': 'none', 'housing': 'own', 
    'existing_loans_count': 2, 'job': 'skilled', 'dependents': 1, 'phone': 'yes'}]"} '''
    prediction = run(test_row, {})
    print("Test result: ", prediction)
