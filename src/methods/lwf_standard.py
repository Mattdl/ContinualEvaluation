#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.

#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452


import copy

from avalanche.models import MultiTaskModule
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

import torch
from typing import Optional, Sequence, Union, List

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import default_logger
from typing import TYPE_CHECKING

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from src.utils import get_grad_normL2
from src.eval.continual_eval import ContinualEvaluationPhasePlugin

if TYPE_CHECKING:
    from avalanche.training.plugins import StrategyPlugin


class LwFStandardPlugin(StrategyPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        self.alpha = alpha
        self.temperature = temperature
        self.prev_model = None

        self.prev_classes = {'0': set()}
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """

    def _distillation_loss(self, out, prev_out, active_units):
        """
        Compute distillation loss between output of the current model
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
        au = list(active_units)
        log_p = torch.log_softmax(out / self.temperature, dim=1)[:, au]
        q = torch.softmax(prev_out / self.temperature, dim=1)[:, au]
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def penalty(self, out, x, alpha, curr_model):
        """
        Compute weighted distillation loss.
        """

        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                if isinstance(self.prev_model, MultiTaskModule):
                    # output from previous output heads.
                    y_prev = avalanche_forward(self.prev_model, x, None)
                    # in a multitask scenario we need to compute the output
                    # from all the heads, so we need to call forward again.
                    y_curr = avalanche_forward(curr_model, x, None)
                else:  # no task labels
                    y_prev = {'0': self.prev_model(x)}
                    y_curr = {'0': out}

            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if task_id in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    au = self.prev_classes[task_id]
                    dist_loss += self._distillation_loss(yc, yp, au)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        if strategy.clock.train_exp_iterations == 0:
            # First iteration = same model = zero loss (Otherwise noise in log/exp calculations)
            penalty = 0
        else:
            alpha = self.alpha[strategy.clock.train_exp_counter] \
                if isinstance(self.alpha, (list, tuple)) else self.alpha
            penalty = self.penalty(strategy.mb_output, strategy.mb_x, alpha,
                                   strategy.model)

        # track gradient of penalty
        self.get_stability_gradnorm(strategy, penalty)

        # Tracking
        strategy.loss_new = strategy.loss
        strategy.loss_reg = float(penalty)
        strategy.loss += penalty

    def get_stability_gradnorm(self, strategy, loss_stab):
        if not torch.is_tensor(loss_stab):
            strategy.gradnorm_stab = torch.tensor(0, dtype=torch.float)
            return
        _prev_state, _prev_training_modes = ContinualEvaluationPhasePlugin.get_strategy_state(strategy)

        # Set eval mode
        strategy.model.eval()

        loss_stab.backward(retain_graph=True)  # (Might have to reuse intermediate results, use retain_graph)
        strategy.gradnorm_stab = get_grad_normL2(strategy.model)  # Tracking

        # Restore training mode(s)
        ContinualEvaluationPhasePlugin.restore_strategy_(strategy, _prev_state, _prev_training_modes)
        strategy.optimizer.zero_grad()  # Zero grad to ensure no interference

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.prev_model = copy.deepcopy(strategy.model)
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[task_id] \
                    .union(pc)


class LwFStandard(BaseStrategy):
    """ Learning without Forgetting (LwF) strategy.

    See LwF plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 alpha: Union[float, Sequence[float]], temperature: float,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        lwf = LwFStandardPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
