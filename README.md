# Continual Evaluation for Lifelong Learning
[//]: # (**What is this repo?**)
This is a Pytorch and Avalanche based repository to enable finegrained continual evaluation in continual learning, opposed to the standard task transition based evaluation.
It is the main codebase for the *Spotlight ICLR 2023 paper*:
["Continual evaluation for lifelong learning: Identifying the stability gap"](https://openreview.net/forum?id=Zy350cRstc6).

**Why continual evaluation?**
Using continual evaluation, [our work](https://arxiv.org/abs/2205.13452) finds a *stability gap*, where representative continual learning
methods falter in maintaining robust performance during the learning process.
Measuring continual worst-case performance is important to enable continual
learners in the real world, especially for safety-critical applications and real-world actuators. 


**Main features of this repo:**
- **Continual Eval**: Per-iteration evaluation, including test-set subsampling and adjustable evaluation periodicity
- **Continual evaluation metrics**: 
  - **New metrics**: Worst-case Accuracy (WC-ACC), Average Minimum Accuracy (Min-ACC), Windowed-Forgetting (WF), Windowed-Plasticity (WP). 
  - **Existing metrics**: Learning Curve Area (LCA), Average Forgetting (FORG), Average Accuracy (ACC).
- **Extensive tracking**: Track all stats of your continual learning model, e.g. per-iteration feature drift and gradient norms.
- **7 Continual Learning benchmarks** based on: MNIST, CIFAR10, Mini-Imagenet, Mini-DomainNet, PermutedMNIST, RotatedMNIST, Digits

*Project Status*: Codebase delivered as is, no support available.

## Setup
This code uses
- Python 3.8
- [Avalanche](https://github.com/ContinualAI/avalanche) 0.1.0 (beta)
- Pytorch 1.8.1

To setup your environment, you can use the [install script](./install_script.sh), which automatically creates an Anaconda environment for you. 
The script defines the default versions used for the paper. 
    
    ./install_script.sh

You can also define your own conda environment with the [environment.yml](environment.yml) file.

    conda env create -n CLEVAL_ENV -f environment.yml python=3.8
    conda activate CLEVAL_ENV


## Reproducing results
All configs for the experiments can be found in [reproduce/configs](reproduce/configs).
The yaml config files enable a structured way to pass arguments to the python script.

To reproduce an experiment, simply run *./reproduce/run.sh* and pass the yaml filename. For example:
    
    ./reproduce/run.sh splitmnist_ER.yaml

Note that we didn't run with deterministic CUDNN backbone for computational efficiency, which might result in small deviations in results.
We average all results over 5 initialization seeds, these can be run at once (with n_seeds=5),
or define the specific seed per run (e.g. seed=0).

## Continual Evaluation Implementation
The continual evaluation is integrated in the Avalanche flow of continual learning.
For documentation, see [here](https://avalanche.continualai.org/).


- [src/eval/continual_eval.py](src/eval/continual_eval.py):  Introduces Continual Evaluation tracking flow after training iterations in Avalanche `after_training_iteration`.
  The additional phases are defined as:

        # Standard Avalanche CL flow
        ...                          
        - before_training_iteration
        - after_training_iteration
            - before_tracking        # BEGIN Integrated Continual Evaluation
            - before_tracking_step
            - before_tracking_batch
            - after_tracking_batch
            - after_tracking_step
            - after_tracking         # END
        - after_training_epoch
        ...
- [src/eval/continual_eval_metrics.py](src/eval/continual_eval_metrics.py): Contains all the continual evaluation metrics, which all inherit from `TrackerPluginMetric`.
  This Plugin defines all the Continual Evaluation tracking phases, which each metric can overwrite as appropriate.
- [main.py](main.py) first passes a list of plugins to Avalanche's `EvaluationPlugin` with `ContinualEvaluationPhasePlugin` first to update the Continual Evaluation metric states on `after_training_iteration`. 
 Next in the list, the metric plugins are passed to `EvaluationPlugin` so `after_training_iteration` logs the metrics on each iteration.

## Visualize results
We support both tensorboard and WandB.
To view results for Tensorboard, run:

    tensorboard --logdir=OUTPUT_DIR


## Citing and license
Please consider citing us upon using this repo:

    @inproceedings{
      delange2023continual,
      title={Continual evaluation for lifelong learning: Identifying the stability gap},
      author={Matthias De Lange and Gido M van de Ven and Tinne Tuytelaars},
      booktitle={The Eleventh International Conference on Learning Representations },
      year={2023},
      url={https://openreview.net/forum?id=Zy350cRstc6}
    }
  
Code is available under MIT license: A short and simple permissive license with conditions only requiring preservation of copyright and license notices.
See [LICENSE](LICENSE) for the full license.
