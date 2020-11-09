
#### ICLR 2020 submission notes

Code used to produce the results presented in https://openreview.net/forum?id=J40FkbdldTX

##### Training of the super-networks
- example script: */experiments/examples/example_run_bench_s1.py*
- remove/uncomment the example changes that you don't want (e.g. test run, few epochs, batch size, masked cells)
- you can change lines 3 and 4 to run either strict fairness or uniform random sampling

##### To reproduce their evaluation
- example script: */experiments/examples/example_run_bench_hpo.py*
    - run the super-net training first
    - the script will automatically load settings and checkpoint from the super-net training
    ("{cls_task}.s1_path"), e.g. masked operations and which parts of the data set are for evaluation
- remove/uncomment the example changes that you don't want (e.g. test run)
- requires the mini-bench save file located in {path_data}.
    - The mini-bench save file is provided in the supplementary material of the ICLR submission.
    - {path_data} has to be set in global_config.json.

##### Notes
- the code has been improved since the submission, so the produced log/tensorboard files may differ slightly from the
ones in the supplementary material
- you can find the full task descriptions in /experiments/configs/tasks_bench/
- since the code is designed for more than the submitted evaluation, it is probably quite a lot to get into.
The most relevant pieces are:
    - (all the used classes are presented in the respective config files and logged when running the task)
    - The tasks: */uninas/tasks/single.py* and */uninas/tasks/hpo_self.py*
    - The network: */uninas/model/networks/stackedcells.py*
    - The trainer: */uninas/training/trainer/simple.py*



# UniNAS

A highly modular PyTorch framework with a focus on Neural Architecture Search (NAS). 


#### under active development
- APIs may change
- argparse arguments may be moved to more fitting classes
- ...

## Features
- modular and therefore reusable
    - data set loading,
    - network building code and topologies,
    - methods to train architecture weights,
    - sets of operations (primitives),
    - weight initializers,
    - metrics,
    - ... and more
- everything is configurable from the command line and/or config files
    - improved reproducibility, since detailed run configurations are saved and logged
    - powerful search network descriptions enable e.g. highly customizable weight sharing settings
    - the underlying argparse mechanism enables using a GUI for configurations
- compare results of different methods in the same environment
- import and export detailed network descriptions
- integrate new methods and more with fairly little effort
- NAS-Benchmark integration
    - NAS-Bench 201
- ... and more


## Where is this code from?

Except for a few pieces, the code is entirely self-written.
However, sometimes the (official) code is useful to learn from or clear up some details,
and other frameworks can be used for their nice features.

- Repositories that implement algorithms or validation code:
    - [DARTS](https://github.com/quark0/darts)
    - [PR-DARTS](https://github.com/cogsys-tuebingen/prdarts)
    - [ProxylessNAS](https://github.com/mit-han-lab/ProxylessNAS)
    - [FairNAS](https://github.com/xiaomi-automl/FairNAS)
    - [Scarlet-NAS](https://github.com/xiaomi-automl/SCARLET-NAS)
    - [Single-Path NAS](https://github.com/dstamoulis/single-path-nas)
    - [Single Path One-Shot](https://github.com/megvii-model/SinglePathOneShot)
    - [MDENAS](https://github.com/tanglang96/MDENAS)
    - [D-X-Y NAS-Projects](https://github.com/D-X-Y/NAS-Projects)
    - [DNA](https://github.com/changlin31/DNA)
    - [Cream of the Crop](https://github.com/microsoft/cream)
    - [autoaugment](https://github.com/DeepVoltaire/AutoAugment)
- Some algorithms without a repository:
    - [ASAP](https://arxiv.org/abs/1904.04123)
    - [HURRICANE](https://arxiv.org/abs/1910.11609)
    - [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
- External repositories/frameworks that we use:
    - [PyTorch](https://github.com/pytorch/pytorch)
    - [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
    - [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
    - [Torchprofile](https://github.com/mit-han-lab/torchprofile)
    - [NAS-Bench 201](https://github.com/D-X-Y/NAS-Bench-201)
    - [pymoo](https://github.com/msu-coinlab/pymoo)


## Other meta-NAS frameworks
- [Deep Architect](https://github.com/negrinho/deep_architect)
    - highly customizable search spaces, hyperparameters, ...
    - the searchers (SMBO, MCTS, ...) focus on fully training (many) models and are not differentiable 
 - [D-X-Y NAS-Projects](https://github.com/D-X-Y/NAS-Projects)
 - [Auto-PyTorch](https://github.com/automl/Auto-PyTorch)
    - stronger focus on model selection than optimizing one architecture
 - [Vega](https://github.com/huawei-noah/vega)
 - [NNI](https://github.com/microsoft/nni)





## Repository notes


### Dynamic argparse tree

Everything is an argument. Learning rate? Argument. Scheduler? Argument.
The exact topology of a Network, including how many of each cell and
whether they share their architecture weights? Also arguments.

This is enabled by the idea that each used class (method, network, cells, regularizers, ...) can add arguments to argparse,
including which further classes are required (e.g. a method needs a network, which needs a stem).

It starts with the Main class adding a **Task** (cls_task), which itself adds all required components (cls_*).

To see all available (meta) arguments, run *Main.list_all_arguments()* in *uninas/main.py*


#### Graphical user interface
Since putting together the arguments correctly is not trivial
(and requires some familiarity with the code base),
an easier approach is using a GUI.

Hava a look at *uninas/gui/tk_gui/main.py*, a tkinter GUI frontend.

The GUI can automatically filter usable classes, display available arguments, and display tooltips;
based only on the implemented argparse (meta) arguments in the respective classes.



#### Some meta arguments take a single class name:
e.g: cls_task, cls_trainer, cls_data, cls_criterion, cls_method

The chosen classes define their own arguments,
e.g.:
- cls_trainer="SimpleTrainer"
- SimpleTrainer.max_epochs=100
- SimpleTrainer.test_last=10

Their names are also available as wildcards, automatically using their respectively set class name:
- cls_trainer="SimpleTrainer"
- {cls_trainer}.max_epochs --> SimpleTrainer.max_epochs
- {cls_trainer}.test_last --> SimpleTrainer.test_last


#### Some meta arguments take a comma-separated list of class names:
e.g. cls_metrics, cls_initializers, cls_regularizers, cls_optimizers, cls_schedulers

The chosen classes also define their own arguments, but always include an index,
e.g.:
- cls_regularizers="DropOutRegularizer, DropPathRegularizer"
- DropOutRegularizer#0.prob=0.5
- DropPathRegularizer#1.max_prob=0.3
- DropPathRegularizer#1.drop_id_paths=false

And they are also available as indexed wildcards:
- cls_regularizers="DropOutRegularizer, DropPathRegularizer"
- {cls_regularizers#0}.prob  --> DropOutRegularizer#0.prob
- {cls_regularizers#1}.max_prob --> DropPathRegularizer#1.max_prob
- {cls_regularizers#1}.drop_id_paths --> DropPathRegularizer#1.drop_id_paths


### Register

UniNAS makes heavy use of a registering mechanism (via decorators in *uninas/register.py*).
Classes of the same type (e.g. optimizers, networks, ...) will register in one *RegisterDict*.

Registered classes can be accessed via their name in the Register, no matter of their actual
location in the code. This enables e.g. saving network topologies as nested dictionaries,
no matter how complicated they are,
since the class names are enough to find the classes in the code.
(It also grants a certain amount of refactoring-freedom.)


### Exporting networks
(Trained) Networks can easily be used by other PyTorch frameworks/scripts,
see verify.py for an easy example.

             
## Citation
We are planning a publication for this framework,
but there is currently no related paper (just link the repository).
```
@article{TODO str,
        title={TODO title},
        author={TODO authors},
        journal={TODO arxiv},
        year={TODO year}
}
```
