import json
import os

import openml
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
sns.set(
    rc={
        'figure.figsize': (11.7, 8.27),
        'font.size': 31,
        'axes.titlesize': 31,
        'axes.labelsize': 31,
        'xtick.labelsize': 31,
        'ytick.labelsize': 31,
    },
    style="white"
)

from utilities import generate_ranks_data, read_baseline_values, read_cocktail_values


def plot_cocktail_models(
    cocktail_dir: str,
    benchmark_path: str,
):
    """Plot a comparison of the models and generate descriptive
    statistics based on the results of all the models.

    Generates plots which showcase the gain of the cocktail versus
    the baseline. (Plots the error rate of the baseline divided
    by the error rate of the cocktail.) Furthermore, it
    generates information regarding the wins, looses and draws
    of both methods, including a significance result.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.
    """
    fixed_cocktail_wins = 0
    fixed_cocktail_draws = 0
    fixed_cocktail_looses = 0
    fixed_cocktail_stats = []
    dynamic_cocktail_stats = []
    comparison_test_accuracies = []
    task_nr_features = []
    task_nr_examples = []

    fixed_cocktail_results = read_cocktail_values(
        cocktail_dir,
        benchmark_path,
        cocktail_version='cocktail',
    )
    dynamic_cocktail_results = read_cocktail_values(
        cocktail_dir,
        benchmark_path,
        cocktail_version='cocktail_lr',
    )

    task_ids = fixed_cocktail_results.keys()

    with open(os.path.join(cocktail_dir, 'task_metadata.json'), 'r') as file:
        task_metadata = json.load(file)

    for task_id in task_ids:

        fixed_task_result = fixed_cocktail_results[task_id]
        dynamic_task_result = dynamic_cocktail_results[task_id]

        fixed_cocktail_stats.append(fixed_task_result)
        dynamic_cocktail_stats.append(dynamic_task_result)
        if fixed_task_result > dynamic_task_result:
            fixed_cocktail_wins += 1
        elif fixed_task_result < dynamic_task_result:
            fixed_cocktail_looses += 1
        else:
            fixed_cocktail_draws += 1

        fixed_task_result_error = 1 - fixed_task_result
        dynamic_task_result_error = 1 - dynamic_task_result
        comparison_test_accuracies.append(fixed_task_result_error / dynamic_task_result_error)
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

    _, p_value = wilcoxon(dynamic_cocktail_stats, fixed_cocktail_stats)
    print(f'P Value: {p_value:.5f}')
    print(f'Cocktail Win'
          f''
          f's: {fixed_cocktail_wins}, Draws:{fixed_cocktail_draws}, Loses: {fixed_cocktail_looses}')
    plt.title('Dynamic LR Cocktail')
    # plt.title(f'Wins: {cocktail_wins}, '
    #          f'Losses: {cocktail_looses}, '
    #          f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')
    plt.savefig(
        'lr_cocktail_improvement_examples.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )

def plot_models(
    baseline_dir: str,
    cocktail_dir: str,
):
    """Plot a comparison of the models and generate descriptive
    statistics based on the results of all the models.

    Generates plots which showcase the gain of the cocktail versus
    the baseline. (Plots the error rate of the baseline divided
    by the error rate of the cocktail.) Furthermore, it
    generates information regarding the wins, looses and draws
    of both methods, including a significance result.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.
    """
    cocktail_wins = 0
    cocktail_draws = 0
    cocktail_looses = 0
    stat_reg_results = []
    stat_baseline_results = []
    comparison_test_accuracies = []
    task_nr_features = []
    task_nr_examples = []

    # xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    benchmark_results = read_baseline_values(baseline_dir, model_name='tabnet')
    cocktail_results = read_cocktail_values(cocktail_dir, baseline_dir)
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
            cocktail_wins += 1
        elif cocktail_task_result == benchmark_task_result:
            cocktail_draws += 1
        else:
            cocktail_looses += 1
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
    # plt.title(f'Wins: {cocktail_wins}, '
    #          f'Losses: {cocktail_looses}, '
    #          f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')
    plt.savefig(
        'cocktail_improvement_tabnet_examples.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )

def patch_violinplot():
    """Patch seaborn's violinplot in current axis
    to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.3, 0.3, 0.3))


def generate_ranks_comparison(
    all_data: pd.DataFrame,
):
    """Generate a ranks comparison between all methods.

    Creates a violin plot that showcases the ranks that
    the different methods achieve over all the tasks/datasets.

    Parameters
    ----------
    all_data: pd.DataFrame
        A dataframe where each row consists method ranks
        over a certain task.
    """
    all_data_ranked = generate_ranks_data(all_data)
    all_data = pd.melt(
        all_data_ranked,
        value_vars=all_data.columns,
        var_name='Method',
        value_name='Rank',
    )

    fig, _ = plt.subplots()
    sns.violinplot(x='Method', y='Rank', linewidth=3, data=all_data, cut=0, kind='violin')
    patch_violinplot()
    plt.title('Ranks of the baselines and the cocktail')
    plt.xlabel("")
    # plt.xticks(rotation=60)
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


def plot_models_frank(
    baseline_dir: str,
    cocktail_dir: str,
):
    """Plot a comparison of the models and generate descriptive
    statistics based on the results of all the models.

    Generates plots which showcase the gain of the cocktail versus
    the baseline. (Plots the error rate of the baseline divided
    by the error rate of the cocktail.) Furthermore, it
    generates information regarding the wins, looses and draws
    of both methods, including a significance result.

    Parameters:
    -----------
    baseline_dir: str
        The directory where the results are located for the baseline
        methods.
    cocktail_dir: str
        The directory where the results are located for the regularization
        cocktails.
    """
    cocktail_wins = 0
    cocktail_draws = 0
    cocktail_looses = 0
    stat_reg_results = []
    stat_baseline_results = []
    cocktail_error_rates = []
    baseline_error_rates = []

    # xgboost_results = read_xgboost_values(xgboost_dir, model_name='xgboost')
    benchmark_results = read_baseline_values(baseline_dir, model_name='xgboost')
    cocktail_results = read_cocktail_values(cocktail_dir, baseline_dir)
    task_ids = benchmark_results.keys()
    suite = openml.study.get_suite(218)
    benchmark_tasks = suite.tasks
    benchmark_datasets = []
    filter_task_ids = []

    for task_id in benchmark_tasks:
        task = openml.tasks.get_task(task_id, download_data=False)
        benchmark_datasets.append(task.dataset_id)

    benchmark_datasets = set(benchmark_datasets)
    for task_id in task_ids:
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset_id = task.dataset_id
        if dataset_id in benchmark_datasets:
            filter_task_ids.append(task_id)

    for task_id in filter_task_ids:

        benchmark_task_result = benchmark_results[task_id]
        cocktail_task_result = cocktail_results[task_id]
        if benchmark_task_result is None:
            continue

        stat_reg_results.append(cocktail_task_result)
        stat_baseline_results.append(benchmark_task_result)
        if cocktail_task_result > benchmark_task_result:
            cocktail_wins += 1
        elif cocktail_task_result == benchmark_task_result:
            cocktail_draws += 1
        else:
            cocktail_looses += 1
        cocktail_task_result_error = 1 - cocktail_task_result
        benchmark_task_result_error = 1 - benchmark_task_result
        cocktail_error_rates.append(cocktail_task_result_error)
        baseline_error_rates.append(benchmark_task_result_error)

    fig, ax = plt.subplots()
    plt.scatter(baseline_error_rates, cocktail_error_rates, s=100, c='#273E47', label='Test Error Rate')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color='r')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel("XGBoost Error Rate")
    plt.ylabel("Cocktail Error Rate")

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
    """print(f'P Value: {p_value:.5f}')
    print(f'Cocktail Win'
          f''
          f's: {cocktail_wins}, Draws:{cocktail_draws}, Loses: {cocktail_looses}')
    plt.title(f'Wins: {cocktail_wins}, '
              f'Losses: {cocktail_looses}, '
              f'Draws: {cocktail_draws} \n p-value: {p_value:.4f}')"""
    plt.title("Comparison with XGBoost")
    plt.savefig(
        'cocktail_vs_xgboost.pdf',
        bbox_inches='tight',
        pad_inches=0.15,
        margins=0.1,
    )


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

"""plot_cocktail_models(
    cocktail_dir,
    xgboost_dir,
)"""