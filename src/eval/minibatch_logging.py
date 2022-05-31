"""
Metrics that track additional attributes from the current minibatch in training.
Examples are sublosses, norms of the classifier weights etc.
"""

#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from avalanche.evaluation.metrics.loss import MinibatchLoss
from typing import TYPE_CHECKING, List, TypeVar

import torch
from torch import Tensor
from src.utils import get_prototypes_from_classifier

from avalanche.training.plugins.strategy_plugin import StrategyPlugin

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')


class StrategyAttributeAdderPlugin(StrategyPlugin):
    """
     Assign new attributes to the Strategy.
     These attributes can then be tracked with the StrategyAttributeTrackerPlugin.
    """

    def __init__(self, classes: list):
        super().__init__()
        self.classes = classes

        # State
        self.protos_weight, self.protos_bias = None, None

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        """ Gather prototypes."""
        self.protos_weight, self.protos_bias = get_prototypes_from_classifier(strategy.model.classifier, get_clone=True)

    @torch.no_grad()
    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        """ Calculate different stats of the loss etc of current minibatch."""
        self._track_loss_components(strategy)

    @torch.no_grad()
    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        # Deltas
        new_protos_weight, new_protos_bias = get_prototypes_from_classifier(strategy.model.classifier, get_clone=True)
        for y, new_param in new_protos_weight.items():
            p_delta = (new_param - self.protos_weight[y]) ** 2
            setattr(strategy, f'protodelta_weight_c{y}', p_delta)
            setattr(strategy, f'protonorm_weight_c{y}', torch.linalg.norm(new_param))
        for y, new_param in new_protos_bias.items():
            p_delta_b = (new_param - self.protos_bias[y]) ** 2
            setattr(strategy, f'protodelta_bias_c{y}', p_delta_b)
            setattr(strategy, f'protonorm_bias_c{y}', torch.linalg.norm(new_param))

    def _track_loss_components(self, strategy):
        unique_y: Tensor = torch.unique(strategy.mb_y)
        for y in unique_y:  # Per class
            y_mb_idxs = torch.nonzero(y == strategy.mb_y, as_tuple=True)
            y_outputs = strategy.mb_output[y_mb_idxs]

            # Per class
            Lce_numerator_y = - torch.mean(y_outputs[:, y.item()])  # Logits of final layer
            Lce_denominator_y = torch.mean(torch.log(y_outputs.exp().sum(dim=1)))
            Lce_y = Lce_numerator_y + Lce_denominator_y

            setattr(strategy, f'Lce_numerator_c{y.item()}', Lce_numerator_y)
            setattr(strategy, f'Lce_denominator_c{y.item()}', Lce_denominator_y)
            setattr(strategy, f'Lce_c{y.item()}', Lce_y)

        # Reset others
        for y_other in self.classes:
            if y_other in unique_y:
                continue
            try:
                delattr(strategy, f'Lce_numerator_c{y_other}')
                delattr(strategy, f'Lce_denominator_c{y_other}')
                delattr(strategy, f'Lce_c{y_other}')
            except:
                pass


class StrategyAttributeTrackerPlugin(MinibatchLoss):
    def __init__(self, *, strategy_attr: List[str], metric_label=None, ):
        """
        Metric that tracks the values of an attribute of the Strategy each mini-batch in training.
        """
        super().__init__()
        if isinstance(strategy_attr, str):
            strategy_attr = [strategy_attr]
        self.strategy_attr = strategy_attr  # First retrieve the object from the strategy to retrieve the loss from
        self.metric_label = self.strategy_attr[-1] if metric_label is None else metric_label
        self.except_cnt = 0

    def update(self, strategy):
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            # fall back to single task case
            task_label = 0
        else:
            task_label = task_labels[0]

        try:
            ref_obj = strategy
            # import pdb;pdb.set_trace()
            for attr in self.strategy_attr:
                ref_obj = getattr(ref_obj, attr)
        except Exception as e:
            if self.except_cnt == 0:
                self.except_cnt += 1
                print(e)
                print(f"strategy_attr not valid for strategy: '{self.strategy_attr}'")
            return

        if ref_obj is None:
            return

        if isinstance(ref_obj, torch.Tensor):
            ret_obj = ref_obj.detach().clone()
        else:
            ret_obj = torch.tensor(ref_obj)
        self._loss.update(ret_obj,
                          patterns=len(strategy.mb_y), task_label=task_label)

    def __str__(self):
        return f"CURRENT_MB_{self.metric_label}"
