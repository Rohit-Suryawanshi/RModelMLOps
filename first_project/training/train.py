import pickle
import json
from azureml.core.model import Model
import rpy2
import rpy2.robjects as robject
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
import timeit
import logging
import pandas as pd
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

robject.r['options'](warn = -1)
utils = rpackages.importr('utils')
catools = rpackages.importr('caTools')


def split_data(x, y):
	subset = robject.r('subset')
	mask = catools.sample_split(y, 0.8)
	robject.globalenv['mask'] = mask
	inv_mask = robject.r('mask == FALSE')
	X_train = subset(x, mask)
	X_test = subset(x, inv_mask)
	
	y_train = subset(y, mask)
	y_test = subset(y, inv_mask)
	return {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
	
	
def read_data(file_path, label_col):
	if isinstance(file_path, str):
		robject.globalenv['file_path'] = file_path
		robject.r(f'''data <- read.csv(file_path)
		 		''')
	else:
		robject.globalenv['data'] = file_path

	X = robject.r(f'data[, !(names(data) %in% "{label_col}")]')
	y = robject.r(f'as.factor(data${label_col})')
	return X, y	

def train_model(data):
	try:
		C5_0 = rpackages.importr('C50')
	except:
		utils.install_packages('C50')
		C5_0 = rpackages.importr('C50')
	x = data['train']['X']
	y = data['train']['y']
	return C5_0.C5_0(x, y)


def get_model_metrics(model, data):
	predict = robject.r('predict')
	preds = predict(model, data['test']['X'])
	accuracy = accuracy_score(list(preds), list(data['test']['y']))
	return {'accuracy':accuracy}
	

def main():
    print("Running train.py")

    # Load the training data as dataframe
    data_dir = "/home/jagan-ds/Documents/azure-python-mlops"
    data_file = os.path.join(data_dir, 'first_project.csv')
    import pandas as pd
    data_file = pd.read_csv(data_file)
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_file = ro.conversion.py2rpy(data_file)
    x, y = read_data(data_file, 'default')

    data = split_data(x, y)

    # Train the model
    model = train_model(data)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()

