#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from pathlib import Path
from typing import List
import uuid
import random
import numpy
import torch
import datetime
import argparse
from distutils.util import strtobool
import time

from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

# Avalanche
from avalanche.logging import TextLogger, TensorboardLogger, WandBLogger
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10, RotatedMNIST, PermutedMNIST
from avalanche.evaluation.metrics import ExperienceForgetting, StreamForgetting, accuracy_metrics, loss_metrics, \
    StreamConfusionMatrix, timing_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive
from avalanche.models.dynamic_modules import MultiHeadClassifier

# CUSTOM
from src.utils import MetricOverSeed
from src.model import FeatClassifierModel, MLPfeat, ResNet18feat, L2NormalizeLayer
from src.eval.continual_eval import ContinualEvaluationPhasePlugin
from src.eval.continual_eval_metrics import TaskTrackingLossPluginMetric, \
    TaskTrackingGradnormPluginMetric, TaskTrackingFeatureDriftPluginMetric, TaskTrackingAccuracyPluginMetric, \
    TaskAveragingPluginMetric, WindowedForgettingPluginMetric, \
    TaskTrackingMINAccuracyPluginMetric, TrackingLCAPluginMetric, WCACCPluginMetric, WindowedPlasticityPluginMetric
from src.eval.minibatch_logging import StrategyAttributeAdderPlugin, StrategyAttributeTrackerPlugin
from src.utils import ExpLRSchedulerPlugin, IterationsInsteadOfEpochs
from src.benchmarks.domainnet_benchmark import MiniDomainNetBenchmark
from src.benchmarks.digits_benchmark import DigitsBenchmark
from src.methods.lwf_standard import LwFStandard
from src.methods.ewc_standard import EWCStandard
from src.methods.replay import ERStrategy
from src.methods.gem_standard import GEMStandard
from src.benchmarks.miniimagenet_benchmark import SplitMiniImageNet

parser = argparse.ArgumentParser()

# Meta hyperparams
parser.add_argument('exp_name', default=None, type=str, help='Name for the experiment.')
parser.add_argument('--config_path', type=str, default=None,
                    help='Yaml file with config for the args.')

parser.add_argument('--exp_postfix', type=str, default='#now,#uid',
                    help='Extension of the experiment name. A static name enables continuing if checkpointing is define'
                         'Needed for fixes/gridsearches without needing to construct a whole different directory.'
                         'To use argument values: use # before the term, and for multiple separate with commas.'
                         'e.g. #cuda,#featsize,#now,#uid')
parser.add_argument('--save_path', type=str, default='./results/', help='save eval results.')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for the dataloaders.')
parser.add_argument('--cuda', default=True, type=lambda x: bool(strtobool(x)), help='Enable cuda?')
parser.add_argument('--disable_pbar', default=True, type=lambda x: bool(strtobool(x)), help='Disable progress bar')
parser.add_argument('--debug', default=False, type=lambda x: bool(strtobool(x)), help='Eval on few samples.')
parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
parser.add_argument('--deterministic', default=False, type=lambda x: bool(strtobool(x)),
                    help='Set deterministic option for CUDNN backend.')
parser.add_argument('--wandb', default=False, type=lambda x: bool(strtobool(x)), help="Use wandb for exp tracking.")

# Dataset
parser.add_argument('--scenario', type=str, default='smnist',
                    choices=['smnist', 'CIFAR10', 'miniimgnet', 'minidomainnet', 'pmnist', 'rotmnist', 'digits'])
parser.add_argument('--dset_rootpath', default='./data', type=str,
                    help='Root path of the downloaded dataset for e.g. Mini-Imagenet')  # Mini Imagenet
parser.add_argument('--partial_num_tasks', type=int, default=None,
                    help='Up to which task to include, e.g. to consider only first 2 tasks of 5-task Split-MNIST')

# Feature extractor
parser.add_argument('--featsize', type=int, default=400,
                    help='The feature size output of the feature extractor.'
                         'The classifier uses this embedding as input.')
parser.add_argument('--backbone', type=str, choices=['input', 'mlp', 'resnet18', 'cifar_mlp'], default='mlp')
parser.add_argument('--use_GAP', default=True, type=lambda x: bool(strtobool(x)),
                    help="Use Global Avg Pooling after feature extractor (for Resnet18).")

# Classifier
parser.add_argument('--classifier', type=str, choices=['linear', 'norm_embed', 'identity'], default='linear',
                    help='linear classifier (prototype=weight vector for a class)'
                         'For feature-space classifiers, we output the embedding (identity) '
                         'or normalized embedding (norm_embed)')
parser.add_argument('--lin_bias', default=True, type=lambda x: bool(strtobool(x)),
                    help="Use bias in Linear classifier")

# Optimization
parser.add_argument('--optim', type=str, choices=['sgd'], default='sgd')
parser.add_argument('--bs', type=int, default=256, help='Minibatch size.')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs/step.')
parser.add_argument('--iterations_per_task', type=int, default=None,
                    help='When this is defined, it overwrites the epochs per task.'
                         'This enables equal compute per task for imbalanced scenarios.')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
parser.add_argument('--lr_milestones', type=str, default=None, help='Learning rate epoch decay milestones.')
parser.add_argument('--lr_decay', type=float, default=None, help='Multiply factor on milestones.')

# Continual Evaluation
parser.add_argument('--eval_with_test_data', default=True, type=lambda x: bool(strtobool(x)),
                    help="Continual eval with the test or train stream, default True for test data of datasets.")
parser.add_argument('--enable_continual_eval', default=True, type=lambda x: bool(strtobool(x)),
                    help='Enable evaluation each eval_periodicity iterations.')
parser.add_argument('--eval_periodicity', type=int, default=1,
                    help='Periodicity in number of iterations for continual evaluation. (None for no continual eval)')
parser.add_argument('--eval_task_subset_size', type=int, default=1000,
                    help='Max nb of samples per evaluation task. (-1 if not applicable)')

# Expensive additional continual logging
parser.add_argument('--track_class_stats', default=False, type=lambda x: bool(strtobool(x)),
                    help="To track per-class prototype statistics, if too many classes might be better to turn off.")
parser.add_argument('--track_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                    help="Track the gradnorm of the evaluation tasks."
                         "This accumulates computational graphs from the entire task and is very expensive memory wise."
                         "Can be made more feasible with reducing 'eval_task_subset_size'.")
parser.add_argument('--track_features', default=False, type=lambda x: bool(strtobool(x)),
                    help="Track the features before and after a single update. This is very expensive as "
                         "entire evaluation task dataset features are forwarded and stored twice in memory."
                         "Can be made more feasible with reducing 'eval_task_subset_size'.")

# Strategy
parser.add_argument('--strategy', type=str, default='ER',
                    choices=['ER', 'EWC', 'LWF', 'GEM', 'finetune'])
parser.add_argument('--task_incr', default=False, type=lambda x: bool(strtobool(x)),
                    help="Give task ids during training to single out the head to the current task.")
# ER
parser.add_argument('--Lw_new', type=float, default=0.5,
                    help='Weight for the CE loss on the new data, in range [0,1]')
parser.add_argument('--record_stability_gradnorm', default=False, type=lambda x: bool(strtobool(x)),
                    help="Record the gradnorm of the memory samples in current batch?")
parser.add_argument('--mem_size', default=1000, type=int, help='Total nb of samples in rehearsal memory.')

# GEM
parser.add_argument('--gem_gamma', default=0.5, type=float, help='Gem param to favor BWT')

# LWF
parser.add_argument('--lwf_alpha', type=float, default=1, help='Distillation loss weight')
parser.add_argument('--lwf_softmax_t', type=float, default=2, help='Softmax temperature (division).')

# EWC
parser.add_argument('--iw_strength', type=float, default=1, help="IW regularization strength.")


def get_scenario(args, seed):
    print(f"[SCENARIO] {args.scenario}, Task Incr = {args.task_incr}")

    if args.scenario == 'smnist':  #
        args.input_size = (1, 28, 28)
        n_classes = 10
        n_experiences = 5
        scenario = SplitMNIST(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                              fixed_class_order=[i for i in range(n_classes)])
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif args.scenario == 'pmnist':  #
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (1, 28, 28)
        n_classes = 10
        scenario = PermutedMNIST(n_experiences=5, seed=seed)
        scenario.n_classes = n_classes

    elif args.scenario == 'rotmnist':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (1, 28, 28)
        n_classes = 10
        n_experiences = 20
        scenario = RotatedMNIST(n_experiences=n_experiences,
                                rotations_list=[t * (180 / n_experiences) for t in range(n_experiences)])
        scenario.n_classes = n_classes

    elif args.scenario == 'digits':  # Domain-incremental
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (3, 32, 32)
        n_classes = 10
        scenario = DigitsBenchmark()
        scenario.n_classes = n_classes

    elif args.scenario == 'minidomainnet':
        assert not args.task_incr, "Domain incremental can't be multi-head."
        args.input_size = (3, 96, 96)
        n_classes = 126
        scenario = MiniDomainNetBenchmark(dataset_root=args.dset_rootpath)
        scenario.n_classes = n_classes

    elif args.scenario == 'CIFAR10':
        args.input_size = (3, 32, 32)
        n_classes = 10
        n_experiences = 5

        # Use minimal transforms
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        test_transform = train_transform
        scenario = SplitCIFAR10(n_experiences=n_experiences, return_task_id=args.task_incr, seed=seed,
                                fixed_class_order=[i for i in range(n_classes)],
                                train_transform=train_transform,
                                eval_transform=test_transform)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    elif args.scenario == 'miniimgnet':
        args.input_size = (3, 84, 84)
        n_classes = 100
        n_experiences = 20
        scenario = SplitMiniImageNet(args.dset_rootpath, n_experiences=n_experiences, return_task_id=args.task_incr,
                                     seed=seed,
                                     fixed_class_order=[i for i in range(n_classes)], preprocessed=True)
        scenario.n_classes = n_classes
        args.initial_out_features = n_classes // n_experiences  # For Multi-Head

    else:
        raise ValueError("Wrong scenario name.")

    # Cutoff if applicable
    scenario.train_stream = scenario.train_stream[: args.partial_num_tasks]
    scenario.test_stream = scenario.test_stream[: args.partial_num_tasks]

    print(f"Scenario = {args.scenario}")
    return scenario


def get_continual_evaluation_plugins(args, scenario):
    """Plugins for per-iteration evaluation in Avalanche."""
    assert args.eval_periodicity >= 1, "Need positive "

    if args.eval_with_test_data:
        args.evalstream_during_training = scenario.test_stream  # Entire test stream
    else:
        args.evalstream_during_training = scenario.train_stream  # Entire train stream
    print(f"Evaluating on stream (eval={args.eval_with_test_data}): {args.evalstream_during_training}")

    # Metrics
    loss_tracking = TaskTrackingLossPluginMetric()
    gradnorm_tracking = TaskTrackingGradnormPluginMetric() if args.track_gradnorm else None  # Memory+compute expensive
    featdrift_tracking = TaskTrackingFeatureDriftPluginMetric() if args.track_features else None  # Memory expensive

    # Acc derived plugins
    acc_tracking = TaskTrackingAccuracyPluginMetric()
    lca = TrackingLCAPluginMetric()

    acc_min = TaskTrackingMINAccuracyPluginMetric()
    acc_min_avg = TaskAveragingPluginMetric(acc_min)
    wc_acc_avg = WCACCPluginMetric(acc_min)

    wforg_10 = WindowedForgettingPluginMetric(window_size=10)
    wforg_10_avg = TaskAveragingPluginMetric(wforg_10)
    wforg_100 = WindowedForgettingPluginMetric(window_size=100)
    wforg_100_avg = TaskAveragingPluginMetric(wforg_100)

    wplast_10 = WindowedPlasticityPluginMetric(window_size=10)
    wplast_10_avg = TaskAveragingPluginMetric(wplast_10)
    wplast_100 = WindowedPlasticityPluginMetric(window_size=100)
    wplast_100_avg = TaskAveragingPluginMetric(wplast_100)

    tracking_plugins = [
        loss_tracking, gradnorm_tracking, featdrift_tracking, acc_tracking,
        lca,  # LCA from A-GEM (is always avged)
        acc_min, acc_min_avg, wc_acc_avg,  # min-acc/worst-case accuracy
        wforg_10, wforg_10_avg,  # Per-task metric, than avging metric
        wforg_100, wforg_100_avg,  # Per-task metric, than avging metric
        wplast_10, wplast_10_avg,  # Per-task metric, than avging metric
        wplast_100, wplast_100_avg,  # Per-task metric, than avging metric
    ]
    tracking_plugins = [p for p in tracking_plugins if p is not None]

    trackerphase_plugin = ContinualEvaluationPhasePlugin(tracking_plugins=tracking_plugins,
                                                         max_task_subset_size=args.eval_task_subset_size,
                                                         eval_stream=args.evalstream_during_training,
                                                         mb_update_freq=args.eval_periodicity,
                                                         num_workers=args.num_workers,
                                                         )
    return [trackerphase_plugin, *tracking_plugins]


def get_metrics(scenario, args):
    """Metrics are calculated efficiently as running avgs."""

    # Pass plugins, but stat_collector must be called first
    minibatch_tracker_plugins = []

    # Stats on external tracking stream
    if args.enable_continual_eval:
        tracking_plugins = get_continual_evaluation_plugins(args, scenario)
        minibatch_tracker_plugins.extend(tracking_plugins)

    # Current minibatch stats per class
    if args.track_class_stats:
        for y in range(scenario.n_classes):
            minibatch_tracker_plugins.extend([
                # Loss components
                StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_numerator_c{y}"]),
                StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_denominator_c{y}"]),
                StrategyAttributeTrackerPlugin(strategy_attr=[f"Lce_c{y}"]),

                # Prototypes
                StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_weight_c{y}']),
                StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_weight_c{y}']),
                StrategyAttributeTrackerPlugin(strategy_attr=[f'protodelta_bias_c{y}']),
                StrategyAttributeTrackerPlugin(strategy_attr=[f'protonorm_bias_c{y}']),
            ])

    # METRICS FOR STRATEGIES (Will only track if available for method)
    minibatch_tracker_plugins.extend([
        StrategyAttributeTrackerPlugin(strategy_attr=["loss_new"]),
        StrategyAttributeTrackerPlugin(strategy_attr=["loss_reg"]),
        StrategyAttributeTrackerPlugin(strategy_attr=["gradnorm_stab"]),
        StrategyAttributeTrackerPlugin(strategy_attr=["avg_gradnorm_G"]),
    ])

    metrics = [
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(minibatch=True, experience=True, stream=True),
        ExperienceForgetting(),  # Test only
        StreamForgetting(),  # Test only
        StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=True),

        # CONTINUAL EVAL
        *minibatch_tracker_plugins,

        # LOG OTHER STATS
        timing_metrics(epoch=True, experience=False),
        # cpu_usage_metrics(experience=True),
        # DiskUsageMonitor(),
        # MinibatchMaxRAM(),
        # GpuUsageMonitor(0),
    ]
    return metrics


def get_model(args, n_classes):
    """ Build model from feature extractor and classifier."""
    feat_extr = _get_feat_extr(args)  # Feature extractor
    classifier = _get_classifier(args, n_classes, feat_extr.feature_size)  # Classifier
    model = FeatClassifierModel(feat_extr, classifier)  # Combined model
    return model


def _get_feat_extr(args):
    """ Get embedding network. """
    nonlin_embedding = args.classifier in ['linear']  # Layer before linear should have nonlinearities
    input_size = math.prod(args.input_size)

    if args.backbone == "mlp":  # MNIST mlp
        feat_extr = MLPfeat(hidden_sizes=(400, args.featsize), nb_layers=2,
                            nonlinear_embedding=nonlin_embedding, input_size=input_size)

    elif args.backbone == "resnet18":
        feat_extr = ResNet18feat(nf=20, global_pooling=args.use_GAP, input_size=args.input_size)

    else:
        raise ValueError()
    assert hasattr(feat_extr, 'feature_size'), "Feature extractor requires attribute 'feature_size'"
    return feat_extr


def _get_classifier(args, n_classes, feat_size):
    """ Get classifier head. For embedding networks this is normalization or identity layer."""
    # No prototypes, final linear layer for classification
    if args.classifier == 'linear':  # Lin layer
        if args.task_incr:
            classifier = MultiHeadClassifier(in_features=feat_size,
                                             initial_out_features=args.initial_out_features,
                                             use_bias=args.lin_bias)
        else:
            classifier = torch.nn.Linear(in_features=feat_size, out_features=n_classes, bias=args.lin_bias)
    # Prototypes held in strategy
    elif args.classifier == 'norm_embed':  # Get feature normalization
        classifier = L2NormalizeLayer()
    elif args.classifier == 'identity':  # Just extract embedding output
        classifier = torch.nn.Flatten()
    else:
        raise NotImplementedError()
    return classifier


def get_strategy(args, model, eval_plugin, scenario, plugins=None):
    plugins = [] if plugins is None else plugins

    # CRIT/OPTIM
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError()

    # lr-schedule over experiences
    if args.lr_milestones is not None:
        assert args.lr_decay is not None, "Should specify lr_decay when specifying lr_milestones"
        milestones = [int(m) for m in args.lr_milestones.split(',')]
        sched = ExpLRSchedulerPlugin(MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay))
        plugins.append(sched)
        print(f"MultiStepLR schedule over experiences, decaying '{args.lr_decay}' at exps '{milestones}'")

    # Use Iterations if defined
    if args.iterations_per_task is not None:
        args.epochs = int(1e9)
        it_stopper = IterationsInsteadOfEpochs(max_iterations=args.iterations_per_task)
        plugins.append(it_stopper)

    # STRATEGY
    if args.strategy == 'finetune':
        strategy = Naive(model, optimizer, criterion,
                         train_epochs=args.epochs, device=args.device,
                         train_mb_size=args.bs, evaluator=eval_plugin,
                         plugins=plugins
                         )

    elif args.strategy == 'ER':
        strategy = ERStrategy(
            record_stability_gradnorm=args.record_stability_gradnorm,
            Lw_new=args.Lw_new,
            n_total_memories=args.mem_size,
            num_tasks=scenario.n_experiences,
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=args.epochs, device=args.device,
            new_data_mb_size=args.bs,  # Max batch size, disregarding of including replay samples or not
            evaluator=eval_plugin,
            plugins=plugins,
        )

    elif args.strategy == 'GEM':
        strategy = GEMStandard(
            record_stability_gradnorm=args.record_stability_gradnorm,
            memory_strength=args.gem_gamma,
            patterns_per_exp=args.mem_size // scenario.n_experiences,
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=args.epochs, device=args.device,
            train_mb_size=args.bs, evaluator=eval_plugin,
            plugins=plugins,
        )

    elif args.strategy == 'LWF':
        strategy = LwFStandard(
            alpha=args.lwf_alpha,
            temperature=args.lwf_softmax_t,
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=args.epochs, device=args.device,
            train_mb_size=args.bs, evaluator=eval_plugin,
            plugins=plugins,
        )

    elif args.strategy == 'EWC':
        strategy = EWCStandard(
            ewc_lambda=args.iw_strength,
            mode='online', keep_importance_data=False,
            model=model, optimizer=optimizer, criterion=criterion,
            train_epochs=args.epochs, device=args.device,
            train_mb_size=args.bs, evaluator=eval_plugin,
            plugins=plugins,
        )
    else:
        raise NotImplementedError(f"Non existing strategy arg: {args.strategy}")

    print(f"Running strategy:{strategy}")
    if hasattr(strategy, 'plugins'):
        print(f"with Plugins: {strategy.plugins}")
    return strategy


def overwrite_args_with_config(args):
    """
    Directly overwrite the input args with values defined in config yaml file.
    Only if args.config_path is defined.
    """
    if args.config_path is None:
        return
    assert os.path.isfile(args.config_path), f"Config file does not exist: {args.config_path}"

    import yaml
    with open(args.config_path, 'r') as stream:
        arg_configs = yaml.safe_load(stream)

    for arg_name, arg_val in arg_configs.items():  # Overwrite
        assert hasattr(args, arg_name), \
            f"'{arg_name}' defined in config is not specified in args, config: {args.config_path}"
        if isinstance(arg_val, (list, tuple)):
            arg_val = arg_val[0]  # unpack first
        setattr(args, arg_name, arg_val)
    print(f"Overriden args with config: {args.config_path}")


def main():
    args = parser.parse_args()
    overwrite_args_with_config(args)

    args.now = str(datetime.datetime.now().date()) + "_" + '-'.join(str(datetime.datetime.now().time()).split(':')[:-1])
    args.uid = uuid.uuid4().hex
    args.exp_name = '_'.join([args.exp_name, f"now={args.now}", f"uid={args.uid}"])
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"STARTING TIME: {args.now}\nEXP NAME:{args.exp_name}\nargs: {vars(args)}")

    # Paths
    args.setupname = '_'.join([args.strategy, args.scenario, f"e={args.epochs}", args.exp_name])
    args.results_path = Path(os.path.join(args.save_path, args.setupname)).resolve()
    args.eval_results_dir = args.results_path / 'results_summary'  # Eval results
    for path in [args.results_path, args.eval_results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Metrics to track and average in runtime
    metrics_over_seeds = [
        # Standard metrics
        MetricOverSeed('avg_acc_final[test]',
                       extract_name='Top1_Acc_Stream/eval_phase/test_stream/Task000',
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_forg_final[test]',
                       extract_name='StreamForgetting/eval_phase/test_stream',
                       extract_idx=-1, mul_factor=100),

        # Continual Eval metrics
        MetricOverSeed('WCACC_final[test]',
                       extract_name='TRACK_MB_WCACC/train_phase/train_stream/Task000',
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_acc_MIN_final[test]',
                       extract_name="TRACK_MB_acc_MIN_AVG/train_phase/train_stream/Task000",
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_F10_final[test]',
                       extract_name='TRACK_MB_F10_AVG/train_phase/train_stream/Task000',
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_F100_final[test]',
                       extract_name='TRACK_MB_F100_AVG/train_phase/train_stream/Task000',
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_P10_final[test]',
                       extract_name='TRACK_MB_P10_AVG/train_phase/train_stream/Task000',
                       extract_idx=-1, mul_factor=100),
        MetricOverSeed('avg_P100_final[test]',
                       extract_name='TRACK_MB_P100_AVG/train_phase/train_stream/Task000',
                       extract_idx=-1, mul_factor=100),
    ]

    # Iterate seeds
    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
    for seed in seeds:
        print("STARTING SEED {}/{}".format(seed, len(seeds) - 1))
        process_seed(args, seed, metrics_over_seeds)

    # Avg results over all seeds
    final_results_file = args.eval_results_dir / f'seed_summary.pt'
    seed_avg_metrics(metrics_over_seeds, final_results_file)
    print(f"[FILE:FINAL-RESULTS]: {final_results_file}")
    print("FINISHED SCRIPT")


def process_seed(args, seed, metrics_over_seeds):
    """ Single seed processing of entire data stream, both train and eval."""
    # initialize seeds
    args.seed = seed
    set_seed(seed, deterministic=args.deterministic)

    # create scenario
    scenario = get_scenario(args, seed)

    # LOGGING
    loggers = []
    print_logger = TextLogger() if args.disable_pbar else InteractiveLogger()  # print to stdout
    loggers.append(print_logger)

    # tensorboard logging
    tb_log_dir = os.path.join(args.results_path, 'tb_run', f'seed={seed}')  # Group all runs
    loggers.append(TensorboardLogger(tb_log_dir=tb_log_dir))  # log to Tensorboard
    print(f"[Tensorboard] tb_log_dir={tb_log_dir}")

    # wandb logging
    if args.wandb:
        wandb_logger = WandBLogger(project_name="ContinualEval", group_name=args.setupname,
                                   run_name=f"seed={seed}_{args.setupname}", config=vars(args))
        loggers.append(wandb_logger)

    # MODEL
    model = get_model(args, scenario.n_classes)

    # METRICS
    metrics = get_metrics(scenario, args)
    eval_plugin = EvaluationPlugin(*metrics, loggers=loggers, benchmark=scenario)

    # STRATEGY
    strategy_plugins = [StrategyAttributeAdderPlugin(list(range(scenario.n_classes)))]
    strategy = get_strategy(args, model, eval_plugin, scenario, plugins=strategy_plugins)

    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    for experience in scenario.train_stream:
        # TRAIN
        print(f"\n{'-' * 40} TRAIN {'-' * 40}")
        print(f"Start training on experience {experience.current_experience}")
        strategy.train(experience, num_workers=args.num_workers, eval_streams=None)
        print(f"End training on experience {experience.current_experience}")

        # EVAL ALL TASKS (ON TASK TRANSITION)
        print(f"\n{'=' * 40} EVAL {'=' * 40}")
        print(f'Standard Continual Learning eval on entire test set on task transition.')
        task_results_file = args.eval_results_dir / f'seed={seed}' / f'task{experience.current_experience}_results.pt'
        task_results_file.parent.mkdir(parents=True, exist_ok=True)
        res = strategy.eval(scenario.test_stream)  # Gathered by EvalLogger

        # Store eval task results
        task_metrics = dict(strategy.evaluator.all_metric_results)
        torch.save(task_metrics, task_results_file)
        print(f"[FILE:TASK-RESULTS]: {task_results_file}")

    # Save results for entire seed
    all_results_file = args.eval_results_dir / f'seed={seed}_finalresults.pt'  # Backup all results
    save_seed_results(strategy, all_results_file, metrics_over_seeds, seed=seed)
    print(f"[FILE:TB-SEED-RESULTS]: {tb_log_dir}")
    if args.wandb:
        wandb_logger.finish()  # Finish run


def save_seed_results(strategy, all_results_file: str, metrics_over_seeds: List[MetricOverSeed], seed: int):
    """ Save the results from metrics in the current seed-run.
    Append to metrics_over_seeds."""
    # save seed results
    all_metrics = dict(strategy.evaluator.all_metric_results)
    if not os.path.exists(all_results_file):
        torch.save(all_metrics, all_results_file)
    else:
        print(f"Not overwriting, seed results already exists at {all_results_file}")
    print(f"[FILE:SEED-RESULTS]: {all_results_file}")

    # Collect over seeds
    for metric in metrics_over_seeds:
        metric.add_result(all_metrics, seed=seed)


def seed_avg_metrics(metric_over_seeds: List[MetricOverSeed], save_file=None):
    """Save and process average and std over accuracy and forgetting metrics."""

    avg_results = []
    for metric in metric_over_seeds:
        mean, std = metric.get_mean_std_results()
        avg_results.append((metric.name, mean, std))

    # Log results
    print("{}|{}|{}".format(
        MetricOverSeed.logging_token,
        f'{MetricOverSeed.loggin_result_separator}'.join([entry[0] for entry in avg_results]),
        f'{MetricOverSeed.loggin_result_separator}'.join([
            MetricOverSeed.logging_result_format.format(entry[1], entry[2]) for entry in avg_results]),
    ))

    if save_file is not None:
        try:
            torch.save(avg_results, save_file)
        except Exception as e:
            print(f"NOT SAVING SUMMARY: {e}")


def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    t_start = time.time()  # Seconds
    main()
    t_end = time.time()
    t_delta = t_end - t_start
    print(f"TIMING: PYTHON EXECUTION FINISHED IN {datetime.timedelta(seconds=t_delta)} (h:mm:ss)")
