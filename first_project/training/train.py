import rpy2.robjects as rob
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rp
import pandas as pd
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from sklearn.metrics import accuracy_score

rob.r['options'](warn=-1)
utils = rp.importr('utils')
cat = rp.importr('caTools')


def split_data(x, y):
    subset = rob.r('subset')
    mask = cat.sample_split(y, 0.8)
    rob.globalenv['mask'] = mask
    inv_mask = rob.r('mask == FALSE')
    x_train = subset(x, mask)
    x_test = subset(x, inv_mask)

    y_train = subset(y, mask)
    y_test = subset(y, inv_mask)
    return {"train": {"X": x_train, "y": y_train},
            "test": {"X": x_test, "y": y_test}}


def read_data(file_path, label_col):
    if isinstance(file_path, str):
        rob.globalenv['file_path'] = file_path
        rob.r(f'''data <- read.csv(file_path)''')
    else:
        rob.globalenv['data'] = file_path

    x = rob.r(f'data[, !(names(data) %in% "{label_col}")]')
    y = rob.r(f'as.factor(data${label_col})')
    return x, y


def train_model(data):
    try:
        c5_0 = rp.importr('C50')
    except Exception as e:
        print(f"[ERROR] : {str(e)}")
        utils.install_packages('C50')
        c5_0 = rp.importr('C50')
    x = data['train']['X']
    y = data['train']['y']
    return c5_0.C5_0(x, y)


def get_model_metrics(model, data):
    predict = rob.r('predict')
    result = predict(model, data['test']['X'])
    accuracy = accuracy_score(list(result), list(data['test']['y']))
    return {'accuracy': accuracy}


def main():
    print("Running train.py")

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'first_project.csv')
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
