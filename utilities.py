from collections import Counter
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import openml
import pandas as pd
import scipy
from scipy.stats import wilcoxon, rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7, 8.27),"font.size": 31,"axes.titlesize": 31, "axes.labelsize": 31, "xtick.labelsize": 31, "ytick.labelsize": 31}, style="white")
# openml split, not the one used for my experiments.
"""
train_indices, test_indices = task.get_train_test_split_indices()
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
"""


def get_dataset_split(
        dataset: openml.datasets.OpenMLDataset,
        val_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 11,
) -> Tuple[Dict[str, Union[List, np.ndarray]], Dict[str, np.ndarray]]:
    """Split the dataset into training, test and possibly validation set.

    Based on the arguments given, splits the datasets into the corresponding
    sets.

    Parameters:
    -----------
    dataset: openml.datasets.OpenMLDataset
        The dataset that will be split into the corresponding sets.
    val_fraction: float
        The fraction for the size of the validation set from the whole dataset.
    test_fraction: float
        The fraction for the size of the test set from the whole dataset.
    seed: int
        The seed used for the splitting of the dataset.

    Returns:
    --------
    (categorical_information, dataset_splits): tuple(np.array, dict)
        Returns a tuple, where the first arguments provides categorical information
        about the features. While the second argument, is a dictionary with the splits
        for the different sets.
    """
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute,
    )
    # TODO move the imputer and scaler into its own method in the future.
    imputer = SimpleImputer(strategy='most_frequent')
    label_encoder = LabelEncoder()

    empty_features = []
    # detect features that are null
    for feature_index in range(0, X.shape[1]):
        nan_mask = np.isnan(X[:, feature_index])
        nan_verdict = np.all(nan_mask)
        if nan_verdict:
            empty_features.append(feature_index)
    # remove feature indicators from categorical indicator since
    # they will be deleted from simple imputer.
    for feature_index in sorted(empty_features, reverse=True):
        del categorical_indicator[feature_index]

    X = imputer.fit_transform(X)
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    # Center data only on not sparse matrices
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

    categorical_columns = []
    categorical_dimensions = []

    for index, categorical_column in enumerate(categorical_indicator):
        if categorical_column:
            column_unique_values = len(set(X[:, index]))
            column_max_index = int(max(X[:, index]))
            # categorical columns with only one unique value
            # do not need an embedding.
            if column_unique_values == 1:
                continue
            categorical_columns.append(index)
            categorical_dimensions.append(column_max_index + 1)

    categorical_information = {
        'categorical_ind': categorical_indicator,
        'categorical_columns': categorical_columns,
        'categorical_dimensions': categorical_dimensions,
    }

    return categorical_information, dataset_splits


def get_dataset_openml(
        task_id:int = 11,
) -> openml.datasets.OpenMLDataset:
    """Download a dataset from OpenML

    Based on a given task id, download the task and retrieve
    the dataset that belongs to the corresponding task.

    Parameters:
    -----------
    task_id: int
        The task id that represents the task for which the dataset will be downloaded.

    Returns:
    --------
    dataset: openml.datasets.OpenMLDataset
        The OpenML dataset that is requested..
    """
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    return dataset


def check_leak_status(splits):
    """Check the leak status.

    This function goes through the different splits of the dataset
    and checks if there is a leak between the different sets.

    Parameters:
    -----------
    splits: dict
        A dictionary that contains the different sets train, test (possibly validation)
        of the whole dataset.

    Returns:
    --------
    None - Does not return anything, only raises an error if there is a leak.
    """
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
    """Check the split stratification and the shape of the examples and labels
    for the different sets.

    This function goes through the different splits of the dataset
    and checks if there is stratification. In this example, if there
    is nearly the same number of examples for each class in the corresponding
    splits. The function also verifies that the shape of the examples and
    labels is the same for the different splits.

    Parameters:
    -----------
    splits: dict
        A dictionary that contains the different sets train, test (possibly validation)
        of the whole dataset.
    """
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
    """Get the task id list.

    Goes through the given file and collects all of the task
    ids.

    Parameters:
    -----------
    benchmark_task_file: str
        A string to the path of the benchmark task file. Including
        the task file name.

    Returns:
    --------
    benchmark_task_ids - list
        A list of all the task ids for the benchmark.
    """
    with open(os.path.join(benchmark_task_file), 'r') as f:
        benchmark_info_str = f.readline()
        benchmark_task_ids = [int(task_id) for task_id in benchmark_info_str.split(' ')]

    return benchmark_task_ids


def status_exp_tasks(
        working_directory: str,
        seed: int = 11,
        model_name: int = 'xgboost',
):
    """Analyze the different tasks of the experiment.

    Goes through the results in the directory given and
    it analyzes which one finished succesfully and which one
    did not.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.
    """
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


def read_baseline_values(
        working_directory: str,
        seed: int = 11,
        model_name: str = 'xgboost',
) -> Dict[int, float]:
    """Prepares the results of the experiment with the baselines.

    Goes through the results at the given directory and it generates a
    dictionary for the baseline with the performances on every task
    of the benchmark.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.

    Returns:
    --------
    baseline_results - dict
        A dictionary with the results of the baseline algorithm.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    baseline_results = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{task_id}', f'{seed}')
        try:
            with open(os.path.join(task_result_directory, 'refit_result.json'), 'r') as file:
                task_result = json.load(file)
            baseline_results[task_id] = task_result['test_accuracy']
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            baseline_results[task_id] = None

    return baseline_results


def read_autosklearn_values(
        working_directory,
        seed=11,
        model_name='autosklearn'
) -> Dict[int, float]:
    """Prepares the results of the experiment with auto-sklearn.

    Goes through the results at the given directory and it generates a
    dictionary for autosklearn with the performances on every task
    of the benchmark.

    Parameters:
    -----------
    working_directory: str
        The directory where the results are located.
    seed: int
        The seed that was used for the experiment.
    model_name: int
        The name of the model that was used.

    Returns:
    --------
    autosklearn_results - dict
        A dictionary with the results of the autosklearn algorithm.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    autosklearn_results = {}
    benchmark_task_file = 'benchmark_datasets.txt'
    benchmark_task_file_path = os.path.join(working_directory, benchmark_task_file)
    result_directory = os.path.join(working_directory, model_name)
    task_ids = get_task_list(benchmark_task_file_path)
    for task_id in task_ids:
        task_result_directory = os.path.join(result_directory, f'{seed}', f'{task_id}', 'results')
        try:
            with open(os.path.join(task_result_directory, 'performance.txt'), 'r') as baseline_file:
                baseline_test_acc = float(baseline_file.readline())
                autosklearn_results[task_id] = baseline_test_acc
        except FileNotFoundError:
            print(f'Task {task_id} not finished.')
            autosklearn_results[task_id] = None
            continue

    return autosklearn_results


def read_cocktail_values(
        cocktail_result_dir: str,
        benchmark_task_file_dir: str,
        seed: int = 11
) -> Dict[int, float]:
    """Prepares the results of the experiment with the regularization
    cocktail.

    Goes through the results at the given directory and it generates a
    dictionary for the regularization cocktails with the performances
    on every task of the benchmark.

    Parameters:
    -----------
    cocktail_result_dir: str
        The directory where the results are located for the regularization
        cocktails.
    benchmark_task_file_dir: str
        The directory where the benchmark task file is located.
        The file contains all the task ids. The file name is
        not needed to be given.
    seed: int
        The seed that was used for the experiment.

    Returns:
    --------
    cocktail_results - dict
        A dictionary with the results of the regularization cocktail method.
        Each key of the dictionary represents a task id, while,
        each value corresponds to the performance of the algorithm.
    """
    cocktail_results = {}

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
                f'{seed}',
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
                cocktail_results[task_id] = test_results['mean_test_bal_acc']
            except FileNotFoundError:
                cocktail_results[task_id] = None

    return cocktail_results


def compare_models(xgboost_dir, cocktail_dir):

    xgboost_results = read_baseline_values(xgboost_dir, model_name='xgboost')
    tabnet_results = read_baseline_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    autosklearn_results = read_autosklearn_values(cocktail_dir)

    table_dict = {
        'Task Id': [],
        'Tabnet': [],
        'XGBoost': [],
        'AutoSklearn': [],
        'Cocktail': [],
    }

    cocktail_wins = 0
    cocktail_losses = 0
    cocktail_ties = 0
    autosklearn_looses = 0
    autosklearn_ties = 0
    autosklearn_wins = 0
    cocktail_performances = []
    xgboost_performances = []
    autosklearn_performances = []
    print(cocktail_results)
    print(xgboost_results)

    for task_id in xgboost_results:
        xgboost_task_result = xgboost_results[task_id]
        if xgboost_task_result is None:
            continue
        tabnet_task_result = tabnet_results[task_id]
        cocktail_task_result = cocktail_results[task_id]
        autosklearn_task_result = autosklearn_results[task_id]
        cocktail_performances.append(cocktail_task_result)
        xgboost_performances.append(xgboost_task_result)
        autosklearn_performances.append(autosklearn_task_result)
        if cocktail_task_result > xgboost_task_result:
            cocktail_wins += 1
        elif cocktail_task_result < xgboost_task_result:
            cocktail_losses += 1
        else:
            cocktail_ties += 1
        if autosklearn_task_result > xgboost_task_result:
            autosklearn_wins += 1
        elif autosklearn_task_result < xgboost_task_result:
            autosklearn_looses += 1
        else:
            autosklearn_ties += 1
        table_dict['Task Id'].append(task_id)
        if tabnet_task_result is not None:
            table_dict['Tabnet'].append(tabnet_task_result)
        else:
            table_dict['Tabnet'].append(tabnet_task_result)
        table_dict['XGBoost'].append(xgboost_task_result)
        table_dict['Cocktail'].append(cocktail_task_result)
        table_dict['AutoSklearn'].append(autosklearn_task_result)

        comparison_table = pd.DataFrame.from_dict(table_dict)
    print(
        comparison_table.to_latex(
            index=False,
            caption='The performances of the Regularization Cocktail and the state-of-the-art competitors over the different datasets.',
            label='app:cocktail_vs_benchmarks_table',
        )
    )
    comparison_table.to_csv(os.path.join(xgboost_dir, 'table_comparison.csv'), index=False)



    _, p_value = wilcoxon(cocktail_performances, xgboost_performances)
    print(f'Cocktail wins: {cocktail_wins}, ties: {cocktail_ties}, looses: {cocktail_losses} against XGBoost')
    print(f'P-value: {p_value}')
    _, p_value = wilcoxon(xgboost_performances, autosklearn_performances)
    print(f'Xgboost vs AutoSklearn, P-value: {p_value}')
    print(f'AutoSklearn wins: {autosklearn_wins}, ties: {autosklearn_ties}, looses: {autosklearn_looses} against XGBoost')

    return comparison_table


def build_cd_diagram(
    xgboost_dir,
    cocktail_dir,
):
    xgboost_results = read_baseline_values(xgboost_dir, model_name='xgboost')
    tabnet_results = read_baseline_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    autosklearn_results = read_autosklearn_values(cocktail_dir)

    models = ['Regularization Cocktail', 'XGBoost', 'AutoSklearn-GB', 'TabNet']
    table_results = {
        'Network': [],
        'Task Id': [],
        'Balanced Accuracy': [],
    }
    for task_id in cocktail_results:
        for model_name in models:
            try:
                if model_name == 'Regularization Cocktail':
                    task_result = cocktail_results[task_id]
                elif model_name == 'XGBoost':
                    task_result = xgboost_results[task_id]
                elif model_name == 'TabNet':
                    task_result = tabnet_results[task_id]
                elif model_name == 'AutoSklearn-GB':
                    task_result = autosklearn_results[task_id]
                else:
                    raise ValueError("Illegal model value")
            except Exception:
                task_result = 0
                print(f'No results for task: {task_id} for model: {model_name}')

            table_results['Network'].append(model_name)
            table_results['Task Id'].append(task_id)
            table_results['Balanced Accuracy'].append(task_result)

    result_df = pd.DataFrame(data=table_results)
    result_df.to_csv(os.path.join(xgboost_dir, f'cd_data.csv'), index=False)


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
benchmark_table = compare_models(xgboost_dir, cocktail_dir)
#build_cd_diagram(xgboost_dir, cocktail_dir)

def plot_models(
    xgboost_dir,
    cocktail_dir,
):
    cocktail_wins = 0
    cocktail_draws = 0
    cocktail_looses = 0
    stat_reg_results = []
    stat_baseline_results = []
    comparison_train_accuracies = []
    comparison_test_accuracies = []
    task_nr_features = []
    task_nr_examples = []

    #xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    benchmark_results = read_baseline_values(xgboost_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, xgboost_dir)
    task_ids = benchmark_results.keys()

    with open(os.path.join(cocktail_dir, 'task_metadata.json'), 'r') as file:
        task_metadata = json.load(file)

    for task_id in task_ids:

        benchmark_task_result = benchmark_results[task_id]
        cocktail_task_result = cocktail_results[task_id]
        if benchmark_task_result is None:
            continue

        stat_reg_results.append(cocktail_task_result)
        stat_baseline_results.append(benchmark_task_result)
        if cocktail_task_result > benchmark_task_result:
            cocktail_wins +=1
        elif cocktail_task_result == benchmark_task_result:
            cocktail_draws += 1
        else:
            cocktail_looses +=1
        cocktail_task_result_error = 1 - cocktail_task_result
        benchmark_task_result_error = 1 - benchmark_task_result
        comparison_test_accuracies.append(benchmark_task_result_error / cocktail_task_result_error)
        task_nr_examples.append(task_metadata[f'{task_id}'][0])
        task_nr_features.append(task_metadata[f'{task_id}'][1])


    plt.scatter(task_nr_examples, comparison_test_accuracies, s=100, c='#273E47', label='Test accuracy')
    plt.axhline(y=1, color='r', linestyle='-', linewidth=3)
    plt.xscale('log')
    plt.xlabel("Number of data points")
    plt.ylabel("Gain")
    plt.ylim((0, 6))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    plt.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
    )

    _, p_value = wilcoxon(stat_reg_results, stat_baseline_results)
    print(f'P Value: {p_value:.5f}')
    print(f'Cocktail Win'
          f''
          f's: {cocktail_wins}, Draws:{cocktail_draws}, Loses: {cocktail_looses}')
    plt.title('TabNet')
    #plt.title(f'Wins: {cocktail_wins}, '
    #          f'Losses: {cocktail_looses}, '
    #          f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')
    plt.savefig(
        'cocktail_improvement_tabnet_examples.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )

# plot_models(xgboost_dir, cocktail_dir)

def generate_ranks_data(
    all_data: pd.DataFrame,
):
    """
    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists of a
        tasks values across networks with different
        regularization techniques.
    """
    all_ranked_data = []
    all_data.drop(columns=['Task Id'], inplace=True)
    column_names = all_data.columns



    for row in all_data.itertuples(index=False):
        task_regularization_data = list(row)
        task_ranked_data = rankdata(
            task_regularization_data,
            method='dense',
        )

        reversed_data = len(task_ranked_data) + 1 - task_ranked_data.astype(int)
        """for i, column_name in enumerate(column_names):
            all_ranked_data.append([column_name, task_ranked_data[i]])
        """
        all_ranked_data.append(reversed_data)
    ranks_df = pd.DataFrame(all_ranked_data, columns=column_names)

    return ranks_df


def patch_violinplot():
    """Patch seaborn's violinplot in current axis to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.3, 0.3, 0.3))

def generate_ranks_comparison(
    all_data: pd.DataFrame,
):
    """
    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists of a
        tasks values across networks with different
        regularization techniques.
    """
    all_data_ranked = generate_ranks_data(all_data)
    all_data = pd.melt(
        all_data_ranked,
        value_vars=all_data.columns,
        var_name='Method',
        value_name='Rank',
    )

    fig, _ = plt.subplots()
    print(all_data)
    sns.violinplot(x='Method', y='Rank', linewidth=3, data=all_data, cut=0, kind='violin')
    patch_violinplot()
    plt.title('Ranks of the baselines and the cocktail')
    plt.xlabel("")
    #plt.xticks(rotation=60)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        top=False,
        bottom=True,
        # ticks along the top edge are off
    )
    fig.autofmt_xdate()
    plt.savefig(
        'violin_ranks.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )

generate_ranks_comparison(benchmark_table)