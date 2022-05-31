#  Copyright (c) 2021-2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from avalanche.benchmarks import nc_benchmark

from avalanche.benchmarks.datasets.mini_imagenet.mini_imagenet import \
    MiniImageNetDataset

_default_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

_default_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitMiniImageNet(root_path, n_experiences=20, return_task_id=False, seed=0,
                      fixed_class_order=None,
                      train_transform=_default_train_transform,
                      test_transform=_default_test_transform,
                      preprocessed=True):
    """
    Creates a CL scenario using the Mini ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    :param preprocessed: Use preprocessed images for Mini-Imagenet if True, otherwise use original Imagenet.
    :param root_path: Root path of the downloaded dataset.
    :param n_experiences: The number of experiences in the current scenario.
    :param return_task_id: if True, for every experience the task id is returned
        and the Scenario is Multi Task. This means that the scenario returned
        will be of type ``NCMultiTaskScenario``. If false the task index is
        not returned (default to 0 for every batch) and the returned scenario
        is of type ``NCSingleTaskScenario``.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param test_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT scenario if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT scenario otherwise.
        """

    if preprocessed:
        train_set, test_set = _get_preprocessed_split_mini_imagenet(root_path)
    else:
        train_set, test_set = _get_mini_imagenet_dataset(root_path)

    if return_task_id:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes=None,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=test_transform)
    else:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            per_exp_classes=None,
            train_transform=train_transform,
            eval_transform=test_transform)


def _get_mini_imagenet_dataset(path):
    """ Create from ImageNet. """
    train_set = MiniImageNetDataset(path, split='train')
    test_set = MiniImageNetDataset(path, split='test')

    return train_set, test_set


def _get_preprocessed_split_mini_imagenet(root_path):
    """
    Use preprocessed train (500 imgs) /test (100 imgs) split in Numpy Array. For 100 classes.
    Download: https://github.com/yaoyao-liu/mini-imagenet-tools
    """
    import pickle
    import numpy as np

    with open(f"{root_path}/miniImageNet.pkl", "rb") as f:
        dataset = pickle.load(f)

    train_x, test_x = [], []
    train_y, test_y = [], []

    for i in range(0, len(dataset["labels"]), 600):
        train_x.extend(dataset["data"][i:i + 500])
        test_x.extend(dataset["data"][i + 500:i + 600])
        train_y.extend(dataset["labels"][i:i + 500])
        test_y.extend(dataset["labels"][i + 500:i + 600])

    train_x, test_x = np.array(train_x), np.array(test_x)
    train_y, test_y = np.array(train_y), np.array(test_y)

    return XYDataset(train_x, train_y), XYDataset(test_x, test_y)


class XYDataset(Dataset):
    """ Template Dataset with Labels """

    def __init__(self, x, y, **kwargs):
        self.x, self.targets = x, y
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.targets[idx]
        return x, y


__all__ = [
    'SplitMiniImageNet'
]
