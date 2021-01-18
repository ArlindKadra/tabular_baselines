from utilities import get_dataset_openml, get_dataset_split

class Loader():

    def __init__(self, task_id):

        dataset = get_dataset_openml(task_id)
        self.splits = get_dataset_split(
            dataset,
        )
        self.dataset_id = dataset.dataset_id

    def get_splits(self):

        return self.splits

    def get_dataset_id(self):

        return self.dataset_id

