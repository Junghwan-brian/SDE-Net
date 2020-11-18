import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


def load_MSD():
    # Load the raw data.
    num_attributes = 90
    names = ["Year"] + ["Attr_{}".format(i) for i in range(num_attributes)]
    df = pd.read_csv("data/YearPredictionMSD.txt", header=None, names=names)

    # Validate the data.
    num_examples = 515345
    assert len(df.columns) == num_attributes + 1
    assert len(df) == num_examples
    assert not df.isnull().values.any()

    # Train/test_sde split. See "Data Set Information".
    num_train = 463715
    df = df.values
    train = df[:num_train]
    test = df[num_train:]
    del df

    # Seperate inputs and outputs.
    X_train, y_train = train[:, 1:], train[:, 0]
    X_test, y_test = test[:, 1:], test[:, 0]
    del train
    del test

    standardize = StandardScaler().fit(X_train)
    X_train = standardize.transform(X_train)
    X_test = standardize.transform(X_test)

    y_train1 = np.expand_dims(y_train, axis=1)
    y_test1 = np.expand_dims(y_test, axis=1)

    standardize2 = StandardScaler().fit(y_train1)
    y_train = standardize2.transform(y_train1)
    y_train = np.squeeze(y_train)

    y_test = standardize2.transform(y_test1)
    y_test = np.squeeze(y_test)

    return (
        X_train.astype(np.float64),
        y_train.astype(np.float64),
        X_test.astype(np.float64),
        y_test.astype(np.float64),
    )


def load_boston():
    boston = datasets.load_boston()
    x = boston.data
    X = np.concatenate((x, x), axis=1)
    for i in range(5):
        X = np.concatenate((X, x), axis=1)
    X = X[:, 0:-1]
    standardize = StandardScaler().fit(X)
    X = standardize.transform(X)
    return X.astype(np.float64)


def load_dataset(dataset):
    if dataset == "MSD":  # training data
        X_train, y_train, X_test, y_test = load_MSD()
        return X_train, y_train, X_test, y_test
    if dataset == "boston":  # out of distribution data
        x = load_boston()
        return x
