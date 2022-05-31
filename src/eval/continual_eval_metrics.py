#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
from collections import deque, defaultdict
import torch

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics.loss import Loss
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name

from src.utils import get_grad_normL2

if TYPE_CHECKING:
    from avalanche.training import BaseStrategy

TResult = TypeVar('TResult')


class TrackingCollector:
    """Holder object used in continual evaluation. Defines behavior and temporary storage for the metrics."""

    def __init__(self):
        # CONFIG
        self.collect_features = False
        self.forward_with_grad = False

        # COLLECTED
        self.pre_update_features: dict = {}  # Featuredrift detection per task
        self.post_update_features = None  # Features current task only
        self.loss = None
        self.gradnorm = None
        self.current_tracking_task = None

        # Running
        self.is_first_preupdate_step = True  # Before doing any updates, all other tracking is after iterations
        self.is_tracking_iteration = False
        self.x, self.y, self.task_id = None, None, None
        self.preds_batch = None


class TrackerPluginMetric(PluginMetric[TResult]):
    """ General Tracker Plugin for Continual Evaluation.
     Implements (optional) resetting after training iteration.
     """

    def __init__(self, name, metric, reset_at='iteration'):
        """Emits and updates metrics at each iteration"""
        super().__init__()
        self._metric = metric
        self.name = name

        # Mode is train
        assert reset_at in {'iteration', 'never'}  # Not at stream
        self._reset_at = reset_at

    # Basic methods
    def reset(self, strategy=None) -> None:
        """Default behavior metric."""
        self._metric.reset()

    def result(self, strategy=None):
        """Default behavior metric."""
        return self._metric.result()

    def update(self, strategy=None):
        """(Optional) Template method to overwrite by subclass.
        Subclass can define own update methods instead.
        """
        pass

    # PHASES
    def before_training_iteration(self, strategy: 'BaseStrategy') -> 'MetricResult':
        """Enable passing the first pre-update step."""
        col: TrackingCollector = strategy.tracking_collector
        if col.is_first_preupdate_step:
            return self._package_result(strategy, x_pos=-1)

    def init_tracking(self, strategy: 'BaseStrategy'):
        """ Init config params. """
        pass

    def before_tracking(self, strategy: 'BaseStrategy'):
        """ Reset metrics just before a new tracking on after_training_iteration. """
        if self._reset_at == 'iteration':
            self.reset(strategy)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking(self, strategy: 'BaseStrategy'):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy') -> None:
        """ Pass to evaluator plugin."""
        col: TrackingCollector = strategy.tracking_collector

        if col.is_tracking_iteration:
            return self._package_result(strategy)

    def _package_result(self, strategy: 'BaseStrategy', x_pos=None) -> 'MetricResult':
        metric_value = self.result(strategy)
        add_exp = False
        plot_x_position = strategy.clock.train_iterations if x_pos is None else x_pos  # Allows pre-update step at -1

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        """ Task label is determined by subclass, not current name. (e.g. Accuracy returns dict of per-task results.)"""
        reset_map = {'iteration': 'MB', 'never': 'STREAM'}
        assert self._reset_at in reset_map
        return f"TRACK_{reset_map[self._reset_at]}_{self.name}"


class TaskTrackingLossPluginMetric(TrackerPluginMetric[float]):
    """Get loss of evaluation tasks at current iteration."""

    def __init__(self):
        self._loss = Loss()
        super().__init__(name="loss", metric=self._loss, reset_at='iteration')

    def update(self, strategy):
        """ Loss is updated externally from common stat collector."""
        col: TrackingCollector = strategy.tracking_collector
        self._loss.update(col.loss, patterns=1, task_label=col.current_tracking_task)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        pass

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        self.update(strategy)


class TaskTrackingAccuracyPluginMetric(TrackerPluginMetric[float]):
    """ Get accuracy of evaluation tasks at current iteration."""

    def __init__(self):
        self._acc = Accuracy()
        super().__init__(name="acc", metric=self._acc, reset_at='iteration')

    def update(self, strategy):
        """ Loss is updated externally from common stat collector."""
        col: TrackingCollector = strategy.tracking_collector
        self._acc.update(col.preds_batch, col.y, task_labels=col.current_tracking_task)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        self.update(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        pass


class WindowedForgettingPluginMetric(TrackerPluginMetric[float]):
    """ For metric definition, see original paper: https://arxiv.org/abs/2205.13452"""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self._current_acc = Accuracy()  # Per-task acc
        super().__init__(name=f"F{self.window_size}", metric=self._current_acc, reset_at='iteration')

        self.acc_window: Dict[int, deque] = defaultdict(deque)
        self.max_forgetting: Dict[int, float] = defaultdict(float)

    def reset(self, strategy) -> None:
        """Only current acc is reset (each iteration), not the window"""
        self._current_acc.reset()

    def result(self, strategy=None) -> Dict[int, float]:
        return self.max_forgetting  # Always return all task results

    def update_current_task_acc(self, strategy):
        col: TrackingCollector = strategy.tracking_collector
        self._current_acc.update(col.preds_batch, col.y, task_labels=col.current_tracking_task)

    def update_task_window(self, strategy):
        col: TrackingCollector = strategy.tracking_collector
        new_acc_dict: Dict[int, float] = self._current_acc.result(task_label=col.current_tracking_task)
        new_acc = new_acc_dict[col.current_tracking_task]

        # Add to window
        task_acc_window = self.acc_window[col.current_tracking_task]
        task_acc_window.append(new_acc)
        if len(task_acc_window) > self.window_size:
            task_acc_window.popleft()

        # Update forgetting
        self.max_forgetting[col.current_tracking_task] = max(self.max_forgetting[col.current_tracking_task],
                                                             self.max_consec_delta_from_window(task_acc_window))
        assert len(task_acc_window) <= self.window_size

    def max_consec_delta_from_window(self, window) -> float:
        """Return max A_i - A_j for i<j in the window."""
        if len(window) <= 1:
            return 0
        max_delta = float('-inf')
        max_found_acc = float('-inf')
        for idx, val in enumerate(window):
            if val < max_found_acc:  # Delta can only increase if higher
                continue
            max_found_acc = val
            for other_idx in range(idx + 1, len(window)):  # Deltas with next ones
                other_val = window[other_idx]
                delta = self.delta(val, other_val)

                if delta > max_delta:
                    max_delta = delta
        return max_delta

    @staticmethod
    def delta(first_val, next_val):
        """ May overwrite to define increase/decrease.
        For forgetting we look for the largest decrease."""
        return first_val - next_val

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        """ Update over batches."""
        self.update_current_task_acc(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        """ Use the final accuracy (over batches), add to the window and calculate the forgetting."""
        self.update_task_window(strategy)


class WindowedPlasticityPluginMetric(WindowedForgettingPluginMetric):
    """ For metric definition, see original paper: https://arxiv.org/abs/2205.13452"""

    def __init__(self, window_size):
        super().__init__(window_size)
        self.name = f"P{self.window_size}"  # overwrite name

    @staticmethod
    def delta(first_val, next_val):
        """ Largest increase. """
        return next_val - first_val


class TaskTrackingMINAccuracyPluginMetric(TrackerPluginMetric[float]):
    """ The accuracy measured per iteration. The minimum accuracy is updated (or created) for tasks that are not
    currently learning. Returns a dictionary of available Acc Minima of all tasks.

    Average over dictionary values to obtain the Average Minimum Accuracy.
    For metric definition, see original paper: https://arxiv.org/abs/2205.13452
    """

    def __init__(self):
        self._current_acc = Accuracy()
        self.min_acc_tasks: dict = defaultdict(lambda: float('inf'))
        super().__init__(name="acc_MIN", metric=self._current_acc, reset_at='iteration')

    def result(self, strategy=None) -> Dict[int, float]:
        return {task: min_acc for task, min_acc in self.min_acc_tasks.items()}

    def update(self, strategy):
        """ Loss is updated externally from common stat collector."""
        col: TrackingCollector = strategy.tracking_collector
        self._current_acc.update(col.preds_batch, col.y, task_labels=col.current_tracking_task)

    def update_acc_minimum(self, strategy):
        """Update minimum."""
        current_learning_task = strategy.clock.train_exp_counter
        current_acc_results: Dict[int, float] = self._current_acc.result()
        for task, task_result in current_acc_results.items():
            if task != current_learning_task:  # Not for current learning task
                self.min_acc_tasks[task] = min(self.min_acc_tasks[task], task_result)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        self.update(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        self.update_acc_minimum(strategy)


class WCACCPluginMetric(TrackerPluginMetric[float]):
    """ Avg over minimum accuracies previous tasks and current accuracy at this training step."""

    def __init__(self, min_acc_plugin: TaskTrackingMINAccuracyPluginMetric):
        self._current_acc = Accuracy()
        self.min_acc_plugin = min_acc_plugin
        self.WCACC = None
        super().__init__(name="WCACC", metric=self._current_acc, reset_at='iteration')  # Reset current_acc at iteration

    def result(self, strategy=None) -> dict:
        return {0: self.WCACC}

    def update(self, strategy):
        """ Update current acc"""
        col: TrackingCollector = strategy.tracking_collector
        current_learning_task = strategy.clock.train_exp_counter
        if current_learning_task == col.current_tracking_task:
            self._current_acc.update(col.preds_batch, col.y, task_labels=col.current_tracking_task)

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        """ Update current task acc"""
        self.update(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        """ Update final metric. """
        self.update_WCACC(strategy)

    def update_WCACC(self, strategy: 'BaseStrategy'):
        avg_list = []
        col: TrackingCollector = strategy.tracking_collector
        current_learning_task = strategy.clock.train_exp_counter

        if current_learning_task != col.current_tracking_task:  # Only update once on current task step
            return

        # Current task
        current_learning_task = strategy.clock.train_exp_counter
        current_learning_task_acc: float = self._current_acc.result()[current_learning_task]
        avg_list.append(current_learning_task_acc)

        # Min-ACC results of OTHER tasks
        min_acc_results: Dict[int, float] = self.min_acc_plugin.result()
        if len(min_acc_results) > 0:
            avg_list.extend([min_acc for task_id, min_acc in min_acc_results.items()
                             if task_id != current_learning_task])

        self.WCACC = torch.mean(torch.tensor(avg_list)).item()


class TrackingLCAPluginMetric(TrackerPluginMetric[float]):
    """ Learning curve area from A-GEM.
    We keep per task the average over the first 'beta' b-shot values.
    On returning the results, it is averaged over tasks.
    This differs from the original paper A-GEM, which first sums over tasks for the bth-shot, then over b-values.
    The result however is identical.
    """

    def __init__(self, beta=10):
        """beta is window size of the 'beta' first accuracies."""
        assert beta > 0
        self.beta = beta
        self._current_acc = Accuracy()

        self.acc_window_tasks: Dict[int, list] = defaultdict(list)
        self.lca_task_counts: Dict[int, int] = defaultdict(int)
        self.lca_tasks: Dict[int, float] = {}  # Avged over beta subsequent minibatch values (range of b-shot values)
        super().__init__(name="LCA", metric=self._current_acc, reset_at='iteration')

    def result(self, strategy=None) -> Dict[int, float]:
        """Should always return average, avg over b-shot values and tasks."""
        if len(self.lca_tasks) > 0:
            return {0: torch.mean(torch.tensor([lca_task for t, lca_task in self.lca_tasks.items()])).item()}

    def update(self, strategy):
        """ Update running accuracy. """
        col: TrackingCollector = strategy.tracking_collector
        current_learning_task = strategy.clock.train_exp_counter
        if col.current_tracking_task == current_learning_task:  # Only for current learning task
            self._current_acc.update(col.preds_batch, col.y, task_labels=col.current_tracking_task)

    def update_LCA(self, strategy):
        """Add accuracy to fixed window if window not full, and calculate LCA if window is filled completely.
        Only proceed if not current learning task."""
        current_learning_task = strategy.clock.train_exp_counter
        col: TrackingCollector = strategy.tracking_collector
        if col.current_tracking_task != current_learning_task:  # Only for current learning task
            return

        if self.lca_task_counts[current_learning_task] >= self.beta:  # Window is full
            return

        current_acc_results: Dict[int, float] = self._current_acc.result()
        current_acc_task = current_acc_results[current_learning_task]

        # Update windows/counts
        self.acc_window_tasks[current_learning_task].append(current_acc_task)
        self.lca_task_counts[current_learning_task] += 1

        if self.lca_task_counts[current_learning_task] == self.beta:  # Can calculate LCA
            self.lca_tasks[current_learning_task] = torch.mean(
                torch.tensor(self.acc_window_tasks[current_learning_task])).item()

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        self.update(strategy)

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        self.update_LCA(strategy)


class TaskTrackingGradnormPluginMetric(TrackerPluginMetric[float]):
    def __init__(self):
        """
        General instantiation of the GradNorm Metric.

        :param classes: List of classes for which the deltas are averaged over.
        Only 1 class will simply give the deltas for this one class' prototypes.
        """
        self._gradnorm_mean = Loss()  # Also returns for multiple tasks!
        super().__init__(name="gradnorm", metric=self._gradnorm_mean, reset_at='iteration')

    def update(self, strategy):
        """ Grad is already avg, so nb patterns should be 1, not len(strategy.mb_y)."""
        col = strategy.tracking_collector
        gradnorm = get_grad_normL2(strategy.model)
        self._gradnorm_mean.update(torch.tensor(gradnorm), patterns=1, task_label=col.current_tracking_task)

    def init_tracking(self, strategy: 'BaseStrategy'):
        col = strategy.tracking_collector
        col.forward_with_grad = True

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        pass

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        self.update(strategy)


class TaskTrackingFeatureDriftPluginMetric(TrackerPluginMetric[float]):
    """ Track the feature drift comparing feature just before/after update."""

    def __init__(self):
        self._featdrift = Loss()
        super().__init__(name="featdrift", metric=self._featdrift, reset_at='iteration')

        # State vars
        self._pre_update_feats = {}  # Per task

    def update(self, strategy):
        """ Loss is updated externally from common stat collector."""
        col: TrackingCollector = strategy.tracking_collector
        if col.is_first_preupdate_step:  # Pre-update step = 0 feature drift
            return
        assert col.pre_update_features is not None
        assert col.post_update_features is not None
        featdrift = self.get_feat_delta(col.pre_update_features[col.current_tracking_task], col.post_update_features)
        self._featdrift.update(featdrift, patterns=1, task_label=col.current_tracking_task)

    def init_tracking(self, strategy: 'BaseStrategy') -> 'MetricResult':
        col: TrackingCollector = strategy.tracking_collector
        col.collect_features = True

    def before_tracking_step(self, strategy: 'BaseStrategy'):
        pass

    def before_tracking_batch(self, strategy: 'BaseStrategy'):
        pass

    def after_tracking_batch(self, strategy: 'BaseStrategy') -> None:
        pass

    def after_tracking_step(self, strategy: 'BaseStrategy'):
        self.update(strategy)
        self._pre_update_feats = {}  # Reset

    @staticmethod
    def get_feat_delta(f1_batch, f2_batch):
        """ Sum MSE values over featdims, avg over samples"""
        return torch.mean(torch.sum((f1_batch - f2_batch) ** 2, dim=-1))


class TaskAveragingPluginMetric(TrackerPluginMetric[float]):
    """ Instead of returning task-specific dict, return averaged over tasks."""

    def __init__(self, task_metric_plugin):
        self.task_metric_plugin = task_metric_plugin
        super().__init__(name=self.task_metric_plugin.name, metric=self.task_metric_plugin, reset_at='never')

    def reset(self, strategy) -> None:
        """ Never reset the original metric plugin."""
        pass

    def update(self, strategy):
        """ Never update the original metric plugin."""
        pass

    def result(self, strategy=None) -> dict:
        task_dict = self.task_metric_plugin.result()  # Always return all task results
        if len(task_dict) > 0:
            return {0: torch.mean(torch.tensor([t_acc for t, t_acc in task_dict.items()])).item()}

    def __str__(self):
        return f"{str(self.task_metric_plugin)}_AVG"
