import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import os
import random

import hpbandster.core.result as hpres
import numpy as np
import openml

from data.loader import Loader
from worker import XGBoostWorker, TabNetWorker


parser = argparse.ArgumentParser(
    description='XGBoost experiment.'
)
parser.add_argument(
    '--run_id',
    type=str,
    help='The run id of the optimization run.',
    default='XGBoost',
)
parser.add_argument(
    '--working_directory',
    type=str,
    help='The working directory where results will be stored.',
    default='.',
)
parser.add_argument(
    '--model',
    type=str,
    help='Which model to use for the experiment.',
    default='tabnet',
)
parser.add_argument(
    '--task_id',
    type=int,
    help='Minimum budget used during the optimization.',
    default=233109,
)
parser.add_argument(
    '--seed',
    type=int,
    help='Seed used for the experiment.',
    default=11,
)

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

loader = Loader(task_id=args.task_id, val_fraction=0)
nr_classes = int(openml.datasets.get_dataset(loader.get_dataset_id()).qualities['NumberOfClasses'])

# TODO refactor dictionary to be different for xgboost and tabnet
if nr_classes != 2:
    param = {
        'objective': 'multi:softmax',
        'num_class': nr_classes + 1,
        'disable_default_eval_metric': 1,
        'seed': args.seed,
        'nthread': 2,
    }
else:
    param = {
        'objective': 'binary:logistic',
        'disable_default_eval_metric': 1,
        'seed': args.seed,
        'nthread': 2,
    }

worker_choices = {
    'tabnet': TabNetWorker,
    'xgboost': XGBoostWorker,
}

model_worker = worker_choices[args.model]

print(f'Experiment started with task id: {args.task_id}')
run_directory = os.path.join(
    args.working_directory,
    f'{args.task_id}',
    f'{args.seed}',
)
os.makedirs(run_directory, exist_ok=True)

worker = model_worker(
    args.run_id,
    param=param,
    splits=loader.get_splits(),
    categorical_ind=loader.categorical_information,
    nameserver='127.0.0.1',
)

result = hpres.logged_results_to_HBS_result(run_directory)
all_runs = result.get_all_runs()
id2conf = result.get_id2config_mapping()



inc_id = result.get_incumbent_id()
inc_runs = result.get_runs_by_id(inc_id)
inc_config = id2conf[inc_id]['config']
print(f"Best Configuration So far {inc_config}")
refit_result = worker.refit(inc_config)
with open(os.path.join(run_directory, 'refit_result.json'), 'w') as file:
    json.dump(refit_result, file)
