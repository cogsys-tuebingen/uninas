
## configs

### Tasks

Tasks mainly fall into these categories:

- DARTS-based:
    - *d1_\*.run_config* to search an architecture on CIFAR-10
    - *d2_\*.run_config* to retrain it
- Super-Net
    - *s1_\*.run_config* to train a super-net on ImageNet
    - *s2_\*.run_config* to search for the best architecture subset of a trained super-net
    - *s3_\*.run_config* to retrain an extracted architecture subset from scratch
    
The *d1* and *super1* tasks require a search space, see *example_run_config.py* in the examples folder.

### Networks - originals

A number of networks matching their respective original implementation.

They are automatically generated via the *uninas.utils.generate* scripts,
matching number of parameters and FLOPs can be assured via the *tests.test_model.py* unit tests.

### Networks - search

Search spaces for over-complete networks.

### Demos

Run scripts that highlight how publication-related code can be run.

### Examples

Run scripts for various implemented features or experiments, generally not stable.
