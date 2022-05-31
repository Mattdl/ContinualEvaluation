#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from collections import defaultdict

import warnings
import torch
from torch.utils.data import DataLoader

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict

from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy

class EWCStandardPlugin(StrategyPlugin):
    """
    Differs from avalanche implementation that propagates online EWC with decay factor.
    Here we keep nb of IW matrixes and always make sure it's an average over all.
    """
    def __init__(self, iw_strength: float, mode='online', keep_importance_data=False):
        """
        :param iw_strength: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience.
               `online` to keep a single penalty summed with a decay factor
               over all previous tasks.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        """

        super().__init__()
        assert mode == 'separate' or mode == 'online', \
            'Mode must be separate or online.'

        self.iw_strength = iw_strength
        self.mode = mode

        if self.mode == 'separate':
            self.keep_importance_data = True
        else:
            self.keep_importance_data = keep_importance_data

        # Running
        self.saved_params = defaultdict(list)
        self.importances = defaultdict(list)
        self.iw_cnt = 0

    def before_backward(self, strategy, **kwargs):
        """
        Compute Importance-weight penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0:
            return

        penalty = torch.tensor(0).float().to(strategy.device)

        if self.mode == 'separate':
            for experience in range(exp_counter):
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                        strategy.model.named_parameters(),
                        self.saved_params[experience],
                        self.importances[experience]):
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == 'online':
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[prev_exp],
                    self.importances[prev_exp]):
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError('Wrong EWC mode.')

        strategy.loss_reg = self.iw_strength * penalty  # Already weigh
        strategy.loss += strategy.loss_reg

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        exp_counter = strategy.clock.train_exp_counter
        importances = self.compute_importances(strategy.model,
                                               strategy._criterion,
                                               strategy.optimizer,
                                               strategy.experience.dataset,
                                               strategy.device,
                                               strategy.train_mb_size)
        self.update_importances(importances, exp_counter, self.iw_cnt, 1)
        self.iw_cnt += 1

        # Save model params
        self.saved_params[exp_counter] = \
            copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and \
                (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def compute_importances(self, model, criterion, optimizer,
                            dataset, device, batch_size):
        """
        Compute EWC importance matrix for each parameter
        """

        model.eval()

        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == 'cuda':
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        'RNN-like modules do not support '
                        'backward calls while in `eval` mode on CUDA '
                        'devices. Setting all `RNNBase` modules to '
                        '`train` mode. May produce inconsistent '
                        'output if such modules have `dropout` > 0.'
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            x, y, task_labels = batch[0], batch[1], batch[-1]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(out, y)
            loss.backward()

            for (k1, param), (k2, imp) in zip(model.named_parameters(),
                                              importances):
                assert (k1 == k2)
                if param.grad is not None:
                    imp += param.grad.data.clone().pow(2)

        # average over mini batches
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances

    @torch.no_grad()
    def update_importances(self, importances, t, w_old, w_new):
        """
        Update importance for each parameter based on the currently computed
        importances.
        """

        if self.mode == 'separate' or t == 0:
            self.importances[t] = importances
        elif self.mode == 'online':
            for (k1, old_imp), (k2, curr_imp) in \
                    zip(self.importances[t - 1], importances):
                assert k1 == k2, 'Error in importance computation.'
                # Unnormalize prev one and add new weighted
                self.importances[t].append(
                    (k1,
                     (w_old * old_imp + curr_imp * w_new) / (w_old + w_new)
                     ))

            # clear previous parameter importances
            if t > 0 and (not self.keep_importance_data):
                del self.importances[t - 1]

        else:
            raise ValueError("Wrong IW update mode.")


class EWCStandard(BaseStrategy):
    """ Elastic Weight Consolidation (EWC) strategy.

    See EWC plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 ewc_lambda: float, mode: str = 'separate',
                 keep_importance_data: bool = False,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1):
        """ Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
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
        ewc = EWCStandardPlugin(ewc_lambda, mode, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)