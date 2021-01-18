from copy import deepcopy

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb


class XGBoostWorker(Worker):

    def __init__(self, *args, param=None, splits=None, **kwargs):
        print(kwargs)
        self.param=param
        self.splits = splits
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)
        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_val = self.splits['X_val']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_val = self.splits['y_val']
        y_test = self.splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_val = xgb.DMatrix(X_val, label=y_val)
        d_test = xgb.DMatrix(X_test, label=y_test)

        gb_model = xgb.train(xgboost_config, d_train, num_rounds)

        # make prediction
        y_train_preds = gb_model.predict(d_train)
        y_val_preds = gb_model.predict(d_val)
        y_test_preds = gb_model.predict(d_test)

        train_performance = balanced_accuracy_score(y_train, y_train_preds)
        val_performance = balanced_accuracy_score(y_val, y_val_preds)
        test_performance = balanced_accuracy_score(y_test, y_test_preds)

        if val_performance is None or val_performance is np.inf:
            val_error_rate = 1
        else:
            val_error_rate = 1 - val_performance

        res = {
            'train_accuracy': float(train_performance),
            'val_accuracy': float(val_performance),
            'test_accuracy': float(test_performance),
        }

        return ({
            'loss': float(val_error_rate),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

    def refit(self, config):

        xgboost_config = deepcopy(self.param)
        xgboost_config.update(config)
        num_rounds = xgboost_config['num_round']
        del xgboost_config['num_round']
        X_train = self.splits['X_train']
        X_test = self.splits['X_test']
        y_train = self.splits['y_train']
        y_test = self.splits['y_test']

        d_train = xgb.DMatrix(X_train, label=y_train)
        d_test = xgb.DMatrix(X_test, label=y_test)

        gb_model = xgb.train(xgboost_config, d_train, num_rounds)

        # make prediction
        train_preds = gb_model.predict(d_train)
        test_preds = gb_model.predict(d_test)
        train_performance = balanced_accuracy_score(y_train, train_preds)
        test_performance = balanced_accuracy_score(y_test, test_preds)

        if test_performance is None or test_performance is np.inf:
            test_performance = 0

        res = {
            'train_accuracy': float(train_performance),
            'test_accuracy': float(test_performance),
        }

        return res

    @staticmethod
    def get_configspace():

        config_space = CS.ConfigurationSpace()
        # learning rate
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'eta',
                lower=0.01,
                upper=1,
            )
        )
        # l2 regularization
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'lambda',
                lower=1E-5,
                upper=100,
            )
        )
        # l1 regularization
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                'alpha',
                lower=1E-5,
                upper=100,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'num_round',
                lower=1,
                upper=1000,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'gamma',
                lower=0,
                upper=100,
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                'max_depth',
                lower=1,
                upper=10,
            )
        )

        return config_space
