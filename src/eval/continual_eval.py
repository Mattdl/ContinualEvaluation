#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, List

import torch
from torch.utils.data import DataLoader
from avalanche.evaluation.metric_results import MetricResult
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin

from src.eval.continual_eval_metrics import TrackerPluginMetric, TrackingCollector

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy


class ContinualEvaluationPhasePlugin(StrategyPlugin):
    """ Collect stats on external stream each K-iterations. """

    def __init__(self,
                 tracking_plugins: List[TrackerPluginMetric],
                 max_task_subset_size=None,  # Max nb of iterations on task-datasets
                 eval_stream=None, eval_stream_task_labels=None,
                 mb_update_freq=100,
                 num_workers=4, pin_memory=True, skip_unseen_tasks=True,
                 ):
        """
        Introduces Continual Evaluation tracking flow after training iterations.
        As this plugin collects additional information in the training phases, depending on the plugins,
        the plugins can add and set 'tracking_dict' attributes, that will enable/disable the collecting of this
        information in this Plugin. e.g. the FeatureDriftPlugin could set 'collect_features' to True.

        New flow:
        ...
        - before_training_iteration
        - after_training_iteration
            - before_tracking_step
            - before_tracking_batch
            - after_tracking_batch
            - after_tracking_step
        - after_training_epoch
        ...

        It assumes this Plugin is called first, hence updating all connected TrackerPluginMetrics.
        The TrackerPluginMetric is only called after this ContinualEvaluationPhasePlugin, hence the
        after_training_iteration will enable emitting the results obtained during collection with this
        ContinualEvaluationPhasePlugin.
        """
        super().__init__()
        self.tracking_collector = TrackingCollector()

        # Checks
        for p in tracking_plugins:
            assert isinstance(p, TrackerPluginMetric)
        self.plugins = tracking_plugins

        # Hyperparams
        self.max_task_subset_size = max_task_subset_size if max_task_subset_size > 0 else None
        self.eval_stream = eval_stream
        self.eval_stream_task_labels = eval_stream_task_labels
        if self.eval_stream and self.eval_stream_task_labels is None:
            self.eval_stream_task_labels = list(range(len(self.eval_stream)))

        self.mb_update_freq = mb_update_freq
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.skip_unseen_tasks = skip_unseen_tasks

        # State vars
        self.seen_tasks = []
        self.subset_idxs = [None] * len(self.eval_stream)
        self.initial_step = False
        print(f"TRACKING: subsetsize={self.max_task_subset_size},freq={self.mb_update_freq},stream={self.eval_stream}")

    def set_subset_idxs(self):
        """ If only using subset of tracking stream, fix idxs beforehand. """
        if self.max_task_subset_size is not None:
            self.subset_idxs = []
            for exp, task_label in zip(self.eval_stream, self.eval_stream_task_labels):
                dataset = exp.dataset
                task_subset_idxs = torch.randperm(len(dataset))[:self.max_task_subset_size]
                self.subset_idxs.append(task_subset_idxs)

    #############################################
    # Phases in framework
    #############################################
    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        """ Initialize configs."""
        strategy.tracking_collector = self.tracking_collector  # For params to set in metrics

        for p in self.plugins:
            p.init_tracking(strategy)

    def before_training_exp(self, strategy: 'BaseStrategy') -> 'MetricResult':
        """ Update seen experiences."""
        exp_counter = strategy.clock.train_exp_counter
        self.seen_tasks.append(exp_counter)

    def before_training_iteration(self, strategy: 'BaseStrategy') -> 'MetricResult':
        """ Phase called from main flow.
        Retrieve feats for delta metric. Can't reuse feats because otherwise delta depends on the logging freq.
        We only want a one-step delta."""
        # Subsetting of tracking stream
        self.set_subset_idxs()

        # Track only based on frequency or last in experience.
        strategy.tracking_collector.is_tracking_iteration = (
                strategy.clock.train_epoch_iterations % self.mb_update_freq == 0
                or strategy.clock.train_epoch_iterations == len(strategy.dataloader))

        # Pass the very first step where no updates were performed yet
        if self.tracking_collector.is_first_preupdate_step:
            self.continual_eval_phase(strategy)

        # Collect features before update
        if self.tracking_collector.collect_features and strategy.tracking_collector.is_tracking_iteration:
            self._collect_exps_feats(strategy, self.eval_stream)  # Collect pre-update feats

    def after_training_iteration(self, strategy: 'BaseStrategy') -> 'MetricResult':
        """ Add ContinualEval phase after iteration in main Avalanche flow."""

        # After first update is never the preupdate step anymore
        self.tracking_collector.is_first_preupdate_step = False

        # Get stats for all the evalstream experiences.
        if strategy.tracking_collector.is_tracking_iteration:
            self.continual_eval_phase(strategy)

    def continual_eval_phase(self, strategy):
        self.before_tracking(strategy)
        self.track(strategy, self.eval_stream)
        self.after_tracking(strategy)

    #############################################
    # Additional Continual Evaluation Tracking phases (After training_iteration)
    #############################################
    def before_tracking(self, strategy: 'BaseStrategy'):
        for p in self.plugins:
            p.before_tracking(strategy)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        for p in self.plugins:
            p.before_tracking_step(strategy)

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        for p in self.plugins:
            p.before_tracking_batch(strategy)

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        for p in self.plugins:
            p.after_tracking_batch(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        for p in self.plugins:
            p.after_tracking_step(strategy)

    def after_tracking(self, strategy: 'BaseStrategy'):
        for p in self.plugins:
            p.after_tracking(strategy)

    #############################################
    # Helper methods
    #############################################
    def forward_data(self, dataset, strategy, tracking_task, collect_feats=False, subset_idxs: list = None):
        """
        Both task-incremental and class-incremental are supported.
        Task-incremental will automatically return outputs and metrics from the specific Task-Head.
        Class-incremental metrics are always calculated on the entire output space.

        Batch-wise iterating data to calculate grad-norm allows to remove the inputs/activations that are irrelevant,
        but the computatinoal graph for the parameters with requries_grad = True remains. Therefore, memory is saved,
        but for large tasks the computational graph can become very large leading to memory-exceptions.

        :param collect_feats: Return features of entire experience.
        :param track_acc_task: From which task to track accuracy for in the acc_plugin.
        :return:
        """
        # Subset if possible
        if subset_idxs is not None:
            dataset = AvalancheSubset(dataset, indices=subset_idxs)

        bs = strategy.eval_mb_size
        dataloader = DataLoader(dataset,
                                num_workers=self.num_workers,
                                batch_size=bs,
                                pin_memory=self.pin_memory)

        col = self.tracking_collector
        col.current_tracking_task = tracking_task

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # Sum and keep cnt

        # Collected over batches
        loss_batch = 0
        feats_all = None
        sample_cnt = 0
        for it, mbatch in enumerate(dataloader):
            self.before_tracking_batch(strategy)

            # Unpack
            for i in range(len(mbatch)):
                mbatch[i] = mbatch[i].to(strategy.device)  # unpack
            col.x, col.y, col.task_id = mbatch
            samples_in_batch = col.x.shape[0]

            feats_batch = strategy.model.forward_feats(col.x)  # Forward
            col.preds_batch = strategy.model.forward_classifier(feats_batch, task_labels=col.task_id)
            loss_batch += criterion(col.preds_batch, col.y)  # Criterion avgs over batch dim

            # Collect features
            if collect_feats:
                if feats_all is None:
                    feats_all = torch.zeros((len(dataset),) + feats_batch.shape[1:])
                feats_all[sample_cnt:sample_cnt + samples_in_batch] = feats_batch.detach().clone()

            sample_cnt += samples_in_batch  # Update cnt
            self.after_tracking_batch(strategy)  # Set collector

        loss_batch_avg = loss_batch / sample_cnt  # Avg over task
        return loss_batch_avg, feats_all  # Feats is None if not tracking

    def track(self, strategy, eval_stream):
        """ During training, eval on arbitrary stream of experiences on the current model.
        We collect stats such as avg gradnorm/loss per experience.

        In task-incremental setting, only forward seen tasks as they have no head yet available.
        """
        col = self.tracking_collector
        _prev_state, _prev_training_modes = self.get_strategy_state(strategy)
        for exp, task_label, subset_idxs in zip(eval_stream, self.eval_stream_task_labels, self.subset_idxs):
            if self.skip_unseen_tasks and task_label not in self.seen_tasks:
                continue

            strategy.optimizer.zero_grad()  # Zero grad to ensure no interference
            strategy.is_training = True
            strategy.model.eval()  # Set to eval mode for BN/Dropout
            dataset = exp.dataset.eval()  # Set transforms

            # Forward and get grads
            self.before_tracking_step(strategy)

            # With or without grads forward
            if col.forward_with_grad:
                col.loss, col.post_update_features = self.forward_data(
                    dataset, strategy, task_label, collect_feats=col.collect_features, subset_idxs=subset_idxs)
                col.loss.backward()
            else:
                with torch.no_grad():
                    col.loss, col.post_update_features = self.forward_data(
                        dataset, strategy, task_label, collect_feats=col.collect_features, subset_idxs=subset_idxs)

            self.after_tracking_step(strategy)

        # Reset grads for safety
        self.restore_strategy_(strategy, _prev_state, _prev_training_modes)
        strategy.optimizer.zero_grad()  # Zero grad to ensure no interference

    @torch.no_grad()
    def _collect_exps_feats(self, strategy, eval_streams):
        """Collect features only for all in eval_streams."""
        _prev_state, _prev_training_modes = self.get_strategy_state(strategy)
        col = strategy.tracking_collector
        for exp, task_label, subset_idxs in zip(eval_streams, self.eval_stream_task_labels, self.subset_idxs):
            if self.skip_unseen_tasks and task_label not in self.seen_tasks:
                continue

            # Forward (no grads)
            strategy.is_training = True
            strategy.model.eval()  # Set to eval mode for BN/Dropout
            dataset = exp.dataset.eval()  # Set transforms
            with torch.no_grad():
                _, feats = self.forward_data(dataset, strategy, task_label, collect_feats=True, subset_idxs=subset_idxs)
            col.pre_update_features[task_label] = feats

        self.restore_strategy_(strategy, _prev_state, _prev_training_modes)

    @staticmethod
    def get_strategy_state(strategy):
        # Since we are switching from train to eval model inside the training
        # loop, we need to save the training state, and restore it after the
        # eval is done.
        _prev_state = (
            strategy.experience,
            strategy.adapted_dataset,
            strategy.dataloader,
            strategy.is_training)

        # save each layer's training mode, to restore it later
        _prev_training_modes = {}
        for name, layer in strategy.model.named_modules():
            _prev_training_modes[name] = layer.training
        return _prev_state, _prev_training_modes

    @staticmethod
    def restore_strategy_(strategy, prev_state, prev_training_modes):
        # restore train-state variables and training mode.
        strategy.experience, strategy.adapted_dataset = prev_state[:2]
        strategy.dataloader = prev_state[2]
        strategy.is_training = prev_state[3]

        # restore each layer's training mode to original
        for name, layer in strategy.model.named_modules():
            prev_mode = prev_training_modes[name]
            layer.train(mode=prev_mode)
