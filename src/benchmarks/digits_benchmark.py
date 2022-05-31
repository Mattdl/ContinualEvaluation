#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import torchvision

from avalanche.benchmarks.datasets import default_dataset_location
from pathlib import Path
from typing import Union, Optional, Any

from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks import dataset_benchmark
from torchvision import transforms
from src.benchmarks.utils import wrap_with_task_labels

grayscale_normalize = transforms.Normalize((0.1307,) * 3, (0.3081,) * 3)  # From MNIST
rgb_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))  # From ImageNet

RESIZED_W = 32
_default_digit_train_transform = transforms.Compose([
    transforms.Resize(RESIZED_W),  # All to SVHN resolution
    transforms.ToTensor(),  # PIL [0,255] range to [0,1]
    transforms.Lambda(lambda x: x.view(-1, RESIZED_W, RESIZED_W).expand(3, -1, -1)),  # Batch size
])

_default_digit_eval_transform = _default_digit_train_transform


def DigitsBenchmark(
        *,
        train_transform: Optional[Any] = _default_digit_train_transform,
        eval_transform: Optional[Any] = _default_digit_eval_transform,
        dataset_root: Union[str, Path] = None):
    """
    Creates a CL benchmark using a sequence of the MNIST, SVHN and USPS datasets.
    This is domain incremental for the digits.

    Input-sizes
    MNIST: 1x28x28
    SVHN: 3x32x32
    USPS: 1x16x16 (Pytorch pixel values in [0,255])

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'cifar10' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    # Transforms with different normalizations (e.g. same for 3 channel copies in grayscale)
    train_grayscale_transform = transforms.Compose([train_transform, grayscale_normalize])
    eval_grayscale_transform = transforms.Compose([eval_transform, grayscale_normalize])

    train_rgb_transform = transforms.Compose([train_transform, rgb_normalize])
    eval_rgb_transform = transforms.Compose([eval_transform, rgb_normalize])

    # Datasets
    """ All datasets return in range [0,255]"""
    mnist_train, mnist_test = _get_MNIST_dataset(dataset_root, train_grayscale_transform, eval_grayscale_transform)
    svhn_train, svhn_test = _get_SVHN_dataset(dataset_root, train_rgb_transform, eval_rgb_transform)
    usps_train, usps_test = _get_USPS_dataset(dataset_root, train_grayscale_transform, eval_grayscale_transform)

    train_sets = [mnist_train, svhn_train, usps_train]
    test_sets = [mnist_test, svhn_test, usps_test]

    target_to_int = transforms.Lambda(lambda x: int(x))
    return dataset_benchmark(
        train_datasets=wrap_with_task_labels(train_sets, target_transform=target_to_int),
        test_datasets=wrap_with_task_labels(test_sets, target_transform=target_to_int),
        complete_test_set_only=False,  # Return test set per task (not a single test set)
        train_transform=None,  # For TRAIN Add in dataset itself! (Otherwise trouble in restoring Replay state)
        eval_transform=None,
    )


def _get_MNIST_dataset(dataset_root, train_transform, test_transform):
    if dataset_root is None:
        dataset_root = default_dataset_location('MNIST')

    train_set = torchvision.datasets.MNIST(dataset_root, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.MNIST(dataset_root, train=False, download=True, transform=test_transform)

    return train_set, test_set


def _get_SVHN_dataset(dataset_root, train_transform, test_transform):
    if dataset_root is None:
        dataset_root = default_dataset_location('SVHN')

    train_set = torchvision.datasets.SVHN(dataset_root, split="train", download=True, transform=train_transform)
    test_set = torchvision.datasets.SVHN(dataset_root, split="test", download=True, transform=test_transform)

    return train_set, test_set


def _get_USPS_dataset(dataset_root, train_transform, test_transform):
    if dataset_root is None:
        dataset_root = default_dataset_location('USPS')

    train_set = torchvision.datasets.USPS(dataset_root, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.USPS(dataset_root, train=False, download=True, transform=test_transform)

    return train_set, test_set


if __name__ == "__main__":
    import sys

    benchmark_instance = DigitsBenchmark()
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)

__all__ = [
    'DigitsBenchmark'
]
