
import pandas as pd
import os


# Loads the diabetes sample data from sklearn and produces a csv file that can
# be used by the build/train pipeline script.
def create_sample_data_csv(file_name: str = "diabetes.csv",
                           for_scoring: bool = False):
    sample_data = os.path.join('data', file_name)
    df = pd.read_csv(sample_data)
    # Hard code to diabetes so we fail fast if the project has been
    # bootstrapped.
    df.to_csv(file_name, index=False)
