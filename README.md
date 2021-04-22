
# UniNAS

A highly modular PyTorch framework with a focus on Neural Architecture Search (NAS). 


#### under development
(which happens mostly on our internal GitLab, we push only every once in a while to Github)
- APIs may change
- argparse arguments may be moved to more fitting classes
- there may be incomplete or not-yet-working pieces of code
- ...

---

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


---

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
- Some algorithms without a repository:
    - [ASAP](https://arxiv.org/abs/1904.04123)
    - [HURRICANE](https://arxiv.org/abs/1910.11609)
    - [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
- NAS Benchmarks:
    - [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)
    - [HW-NAS-Bench](https://github.com/RICE-EIC/HW-NAS-Bench)
    - [NAS-Bench-301](https://github.com/automl/nasbench301)
- External repositories/frameworks that we use:
    - [PyTorch](https://github.com/pytorch/pytorch)
    - [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
    - [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
    - [Torchprofile](https://github.com/mit-han-lab/torchprofile)
    - [pymoo](https://github.com/msu-coinlab/pymoo)
    - [autoaugment](https://github.com/DeepVoltaire/AutoAugment)


## Other meta-NAS frameworks
- [Deep Architect](https://github.com/negrinho/deep_architect)
    - highly customizable search spaces, hyperparameters, ...
    - the searchers (SMBO, MCTS, ...) focus on fully training (many) models and are not differentiable 
 - [D-X-Y NAS-Projects](https://github.com/D-X-Y/NAS-Projects)
 - [Auto-PyTorch](https://github.com/automl/Auto-PyTorch)
    - stronger focus on model selection than optimizing one architecture
 - [Vega](https://github.com/huawei-noah/vega)
 - [NNI](https://github.com/microsoft/nni)


---




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

Have a look at *uninas/gui/tk_gui/main.py*, a tkinter GUI frontend.

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

#### The framework

we will possibly create a whitepaper at some point

```
@misc{kl2020uninas,
  author = {Kevin Alexander Laube},
  title = {UniNAS},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cogsys-tuebingen/uninas}}
}
```

#### Inter-choice dependent super-network weights

1) Train super-networks, e.g. via *experiments/demo/inter_choice_weights/icw1_train_supernet_nats.py*
    - you will need Cifar10, but can also easily use fake data or download it
    - to generate SubImageNet see *uninas/utils/generate/data/subImageNet*
2) Evaluate the super-network, e.g. via *experiments/demo/inter_choice_weights/icw2_eval_supernet.py*
    - this step requires you to have the bench data, see https://cs-cloud.cs.uni-tuebingen.de/index.php/s/tBwgjBNcYqsst55
    - set the path to the bench in the script
3) View the evaluation results in the save dir, in TensorBoard or plotted directly


```
@article{TODO str,
  title={TODO title},
  author={TODO authors},
  journal={TODO arxiv},
  year={TODO year}
}
```
