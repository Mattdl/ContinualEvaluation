#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from avalanche.training.storage_policy import ClassBalancedBuffer
from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import random
import copy
from torch.utils.data import DataLoader
from pprint import pprint
from typing import TYPE_CHECKING, Optional, List
import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies import BaseStrategy
from src.utils import get_grad_normL2

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy


class ERPlugin(StrategyPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """
    store_criteria = ['rnd']

    def __init__(self, n_total_memories, num_tasks):
        """
        Standard samples the same batch-size of new samples.

        :param n_total_memories: The maximal number of input samples to store in total.
        :param num_tasks:        The number of tasks being seen in the scenario.
        :param mode:             'ER'=regular replay, 'ERaverse'=Replay with Ridge Aversion.
        :param init_epochs:      Number of epochs for the first experience/task.
        """
        super().__init__()

        # Memory
        self.n_total_memories = n_total_memories  # Used dynamically
        self.num_tasks = num_tasks

        # a Dict<task_id, Dataset>
        self.storage_policy = ClassBalancedBuffer(  # Samples to store in memory
            max_size=self.n_total_memories,
            adaptive_size=True,
        )

        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")
        print(f"[METHOD CONFIG] SUMMARY: ", end='')
        pprint(self.__dict__, indent=2)

    def before_forward(self, strategy, **kwargs):
        """Add samples from rehearsal memory to current batch and
        calculate perturbation to be applied in next forward."""
        # Sample memory batch
        x_s, y_s, t_s = None, None, None
        if self.n_total_memories > 0 and len(self.storage_policy.buffer) > 0:  # Only sample if there are stored
            x_s, y_s, t_s = self.load_buffer_batch(self.storage_policy, strategy, nb=strategy.train_mb_size)

        # Append to current new-data batch
        if x_s is not None:  # Add
            assert y_s is not None
            assert t_s is not None
            # Assemble minibatch
            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], x_s])
            strategy.mbatch[1] = torch.cat([strategy.mbatch[1], y_s])
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], t_s])

    def after_training_exp(self, strategy, **kwargs):
        """ Update memories."""
        self.storage_policy.update(strategy, **kwargs)  # Storage policy: Store the new exemplars in this experience

    @staticmethod
    def load_buffer_batch(storage_policy, strategy, nb=None):
        """
        Wrapper to retrieve a batch of exemplars from the rehearsal memory
        :param nb: Number of memories to return
        :return: input-space tensor, label tensor
        """
        ret_x, ret_y, ret_t = None, None, None
        # Equal amount as batch: Last batch can contain fewer!
        n_exemplars = strategy.train_mb_size if nb is None else nb
        new_dset = ERPlugin.retrieve_random_buffer_batch(storage_policy, n_exemplars)  # Dataset object

        # Load the actual data
        for sample in DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False):
            x_s, y_s = sample[0].to(strategy.device), sample[1].to(strategy.device)
            t_s = sample[-1].to(strategy.device)  # Task label (for multi-head)
            ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
            ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
            ret_t = y_s if ret_t is None else torch.cat([ret_t, t_s])
        return ret_x, ret_y, ret_t

    @staticmethod
    def retrieve_random_buffer_batch(storage_policy, n_samples):
        """
        Retrieve a batch of exemplars from the rehearsal memory.
        First sample indices for the available tasks at random, then actually extract from rehearsal memory.
        There is no resampling of exemplars.

        :param n_samples: Number of memories to return
        :return: input-space tensor, label tensor
        """
        assert n_samples > 0, "Need positive nb of samples to retrieve!"

        # Determine how many mem-samples available
        q_total_cnt = 0  # Total samples
        free_q = {}  # idxs of which ones are free in mem queue
        tasks = []
        for t, ex_buffer in storage_policy.buffer_groups.items():
            mem_cnt = len(ex_buffer.buffer)  # Mem cnt
            free_q[t] = list(range(0, mem_cnt))  # Free samples
            q_total_cnt += len(free_q[t])  # Total free samples
            tasks.append(t)

        # Randomly sample how many samples to idx per class
        free_tasks = copy.deepcopy(tasks)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in tasks}  # How many sampled already
        max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < max_samples:
            t_idx = random.randrange(len(free_tasks))
            t = free_tasks[t_idx]  # Sample a task

            if sample_cnt[t] >= len(storage_policy.buffer_group(t)):  # No more memories to sample
                free_tasks.remove(t)
                continue
            sample_cnt[t] += 1
            tot_sample_cnt += 1

        # Actually sample
        s_cnt = 0
        subsets = []
        for t, t_cnt in sample_cnt.items():
            if t_cnt > 0:
                # Set of idxs
                cnt_idxs = torch.randperm(len(storage_policy.buffer_group(t)))[:t_cnt]
                sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
                sample_idxs = sample_idxs.view(-1)

                # Actual subset
                s = Subset(storage_policy.buffer_group(t), sample_idxs.tolist())
                subsets.append(s)
                s_cnt += t_cnt
        assert s_cnt == tot_sample_cnt == max_samples
        new_dset = ConcatDataset(subsets)

        return new_dset


class ERStrategy(BaseStrategy):
    """ Overwrite original BaseStrategy to enable avoiding loss reduction, getting the loss per sample."""

    def __init__(self,
                 n_total_memories,
                 num_tasks,
                 model, optimizer, criterion=torch.nn.CrossEntropyLoss(reduction='none'),
                 record_stability_gradnorm: bool = False,
                 Lw_new=0.5,  # Weighing of the new loss w.r.t. old loss
                 new_data_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1,
                 ):
        criterion.reduction = 'none'  # Overwrite

        # Checks
        assert criterion.reduction == 'none', "Must have per-sample losses available for ER."
        assert 0 <= Lw_new <= 1

        self.Lw_new = Lw_new
        self.record_stability_gradnorm = record_stability_gradnorm

        # Store/retrieve samples
        plug = ERPlugin(
            n_total_memories=n_total_memories,
            num_tasks=num_tasks,
        )
        if plugins is None:
            plugins = [plug]
        else:
            plugins = [plug] + plugins

        if isinstance(criterion, StrategyPlugin):
            plugins += [criterion]

        super().__init__(
            model, optimizer, criterion=criterion,
            train_mb_size=new_data_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

        # State vars
        self.loss_reg = None
        self.loss_new = None
        self.gradnorm_stab = None

    def training_epoch(self, **kwargs):

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            nb_new_samples = self.mb_x.shape[0]
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)  # Loads memory samples
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss_batch = self.criterion()  # HERE UPDATED: NO PLUS, BUT NO-REDUCTION OPTION

            # Disentangle losses
            nb_samples = self.loss_batch.shape[0]

            # New loss
            mb_with_replay = False
            self.loss_new = self.Lw_new * self.loss_batch[:nb_new_samples].mean()
            self.loss = self.loss_new

            # Mem loss
            if nb_samples > nb_new_samples:
                mb_with_replay = True
                self.loss_reg = (1 - self.Lw_new) * self.loss_batch[nb_new_samples:].mean()
                if self.record_stability_gradnorm:
                    self.get_stability_gradnorm(self.loss_reg)
                self.loss = self.loss_new + self.loss_reg

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def get_stability_gradnorm(self, loss_stab):
        """ Given the partial stability loss, return the gradient norm for this loss only. """
        _prev_state, _prev_training_modes = ContinualEvaluationPhasePlugin.get_strategy_state(self)

        # Set eval mode
        self.model.eval()

        loss_stab.backward(retain_graph=True)  # (Might have to reuse intermediate results, use retain_graph)
        self.gradnorm_stab = get_grad_normL2(self.model)  # Tracking

        # Restore training mode(s)
        ContinualEvaluationPhasePlugin.restore_strategy_(self, _prev_state, _prev_training_modes)
        self.optimizer.zero_grad()  # Zero grad to ensure no interference
