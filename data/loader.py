from typing import Dict

import numpy as np

from utilities import get_dataset_openml, get_dataset_split


# Loader class which provides the data splits
class Loader():

    def __init__(
            self,
            task_id: int,
            val_fraction: int = 0.2,
            test_fraction: int = 0.2,
            seed: int = 11,
    ):

        # download the dataset
        dataset = get_dataset_openml(task_id)
        # get the splits according to the given fractions and seed,
        # together with the categorical indicator
        self.categorical_information, self.splits = get_dataset_split(
            dataset,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        self.dataset_id = dataset.dataset_id


    def get_splits(self) -> Dict[str, np.array]:

        return self.splits

    def get_dataset_id(self) -> int:

        return self.dataset_id
