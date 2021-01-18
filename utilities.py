from collections import Counter

import numpy as np
import openml
import scipy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# openml split, not the one used for my experiments.
"""
train_indices, test_indices = task.get_train_test_split_indices()
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
"""


def get_dataset_split(dataset, val_fraction=0.2, test_fraction=0.2, seed=11):

    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute,
    )
    # TODO move the imputer and scaler into its own method in the future.
    enc = OneHotEncoder(handle_unknown='ignore')
    imputer = SimpleImputer(strategy='most_frequent')
    label_encoder = LabelEncoder()
    X = imputer.fit_transform(X)
    X = enc.fit_transform(X)
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    # Center data on
    center_data = not scipy.sparse.issparse(X_train)
    scaler = StandardScaler(with_mean=center_data).fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    dataset_splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    if val_fraction != 0:
        new_val_fraction = val_fraction / (1 - test_fraction)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=new_val_fraction,
            random_state=seed,
            stratify=y_train,
        )
        dataset_splits['X_train'] = X_train
        dataset_splits['X_val'] = X_val
        dataset_splits['y_train'] = y_train
        dataset_splits['y_val'] = y_val

    return dataset_splits


def get_dataset_openml(task_id=11):

    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    return dataset

def check_leak_status(splits):

    X_train = splits['X_train']
    X_valid = splits['X_val']
    X_test = splits['X_test']

    for train_example in X_train:
        for valid_example in X_valid:
            if np.array_equal(train_example, valid_example):
                raise AssertionError('Leak between the training and validation set')
        for test_example in X_test:
            if np.array_equal(train_example, test_example):
                raise AssertionError('Leak between the training and test set')
    for valid_example in X_valid:
        for test_example in X_test:
            if np.array_equal(valid_example, test_example):
                raise AssertionError('Leak between the validation and test set')

    print('Leak check passed')

def check_split_stratification(splits):

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    y_train = splits['y_train']
    y_val = splits['y_val']
    y_test = splits['y_test']
    train_occurences = Counter(y_train)
    val_occurences = Counter(y_val)
    test_occurences = Counter(y_test)

    print(train_occurences)
    print(val_occurences)
    print(test_occurences)
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
