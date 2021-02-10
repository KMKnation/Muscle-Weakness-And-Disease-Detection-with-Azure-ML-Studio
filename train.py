import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.ensemble import RandomForestClassifier


# Data is located at:
path = "https://3366d6aa338e.ngrok.io/data.csv"

ds = TabularDatasetFactory.from_delimited_files(path,
                                                validate=True,
                                                include_path=False,
                                                infer_column_types=True,
                                                separator=',',
                                                header=True,
                                                support_multi_line=False,
                                                empty_as_string=False)


def clean_data(raw_df):
    # Clean and one hot encode data
    raw_df = raw_df.to_pandas_dataframe().dropna()
    raw_y = raw_df['Output']
    del raw_df['Output']
    del raw_df['Identification no']
    raw_x = raw_df

    labels_dict = {"Control": 0, "Disease": 1, "SPG4": 2}

    x = pd.get_dummies(raw_x)
    y = raw_y.replace(labels_dict)
    return x, y


x, y = clean_data(ds)

# TODO: Split data into train and test sets.
### YOUR CODE HERE ###a
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True)

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--d', type=int, default=2,
                        help="max depth for the random forest")
    parser.add_argument('--n', type=int, default=8, help="Maximum number of estimators")

    args = parser.parse_args()

    run.log("max_depth:", np.int(args.d))
    run.log("Maximum number of estimators:", np.int(args.n))

    model = RandomForestClassifier(n_estimators=args.n, max_depth=args.d).fit(x_train, y_train)

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
