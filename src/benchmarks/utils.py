#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from avalanche.benchmarks.utils.avalanche_dataset import AvalancheDataset


def wrap_with_task_labels(datasets, target_transform=None):
    return [AvalancheDataset(ds, task_labels=idx, target_transform=target_transform) for idx, ds in enumerate(datasets)]
