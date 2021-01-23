from collections import Counter
import json
import os
from typing import List

import numpy as np
import openml
import pandas as pd
import scipy
from scipy.stats import wilcoxon
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
    
def get_task_list(
    benchmark_task_file: str = 'path/to/tasks.txt',
) -> List[int]:
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


def status_exp_tasks(working_directory, seed=11, model_name='xgboost'):

    not_finished=0
    finished=0
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        print(task_result_directory)
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
                print(f'Task {task_id} finished.')
                finished += 1
                # TODO do something with the result
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            not_finished += 1
    print(f'Finished tasks: {finished} , not finished tasks: {not_finished}')


def read_xgboost_values(working_directory, seed=11, model_name='xgboost'):

    xgboost_result_dir = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
            xgboost_result_dir[task_id] = task_result['test_accuracy']
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            xgboost_result_dir[task_id] = None

    return xgboost_result_dir


def read_cocktail_values(cocktail_result_dir, benchmark_task_file_dir):

    cocktail_result_dict = {}
    result_path = os.path.join(
        cocktail_result_dir,
        'cocktail',
        '512',
    )
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(
        benchmark_task_file_dir,
        benchmark_task_file
    )
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:

            task_result_path = os.path.join(
                result_path,
                f'{task_id}',
                'refit_run',
                '11',
            )

            if os.path.exists(task_result_path):
                if not os.path.isdir(task_result_path):
                    task_result_path = os.path.join(
                        result_path,
                        f'{task_id}',
                    )
            else:
                task_result_path = os.path.join(
                    result_path,
                    f'{task_id}',
                )

            try:
                with open(os.path.join(task_result_path, 'run_results.txt')) as f:
                    test_results = json.load(f)
                cocktail_result_dict[task_id] = test_results['mean_test_bal_acc']
            except FileNotFoundError:
                cocktail_result_dict[task_id] = None

    return cocktail_result_dict

def compare_models(xgboost_dir, cocktail_dir):

    xgboost_results = read_xgboost_values(xgboost_dir)
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    table_dict = {
        'Task Id': [],
        'XGBoost': [],
        'Cocktail': [],
    }

    cocktail_wins = 0
    cocktail_losses = 0
    cocktail_ties = 0
    cocktail_performances = []
    xgboost_performances = []
    print(cocktail_results)
    print(xgboost_results)

    for task_id in xgboost_results:
        xgboost_task_result = xgboost_results[task_id]
        if xgboost_task_result is None:
            continue
        cocktail_task_result = cocktail_results[task_id]
        cocktail_performances.append(cocktail_task_result)
        xgboost_performances.append(xgboost_task_result)
        if cocktail_task_result > xgboost_task_result:
            cocktail_wins += 1
        elif cocktail_task_result < xgboost_task_result:
            cocktail_losses += 1
        else:
            cocktail_ties += 1
        table_dict['Task Id'].append(task_id)
        table_dict['XGBoost'].append(xgboost_task_result)
        table_dict['Cocktail'].append(cocktail_task_result)

        comparison_table = pd.DataFrame.from_dict(table_dict)
        comparison_table.to_csv(os.path.join(xgboost_dir, 'table_comparison.csv'), index=False)


    _, p_value = wilcoxon(cocktail_performances, xgboost_performances)
    print(f'Cocktail wins: {cocktail_wins}, ties: {cocktail_ties}, looses: {cocktail_losses}')
    print(f'P-value: {p_value}')

xgboost_dir = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'xgboost_results',
    )
)

cocktail_dir = os.path.expanduser(
    os.path.join(
        '~',
        'Desktop',
        'PhD',
        'Rezultate',
        'RegularizationCocktail',
        'NEMO',
    )
)
# compare_models(xgboost_dir, cocktail_dir)

