"""
verify the top1/top5 test accuracy of a network
"""

import argparse
from uninas.model.networks.abstract import AbstractNetworkBody
from uninas.data.abstract import AbstractDataSet
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.paths import replace_standard_paths, find_net_config_path
from uninas.utils.shape import Shape
from uninas.utils.misc import split
from uninas.builder import Builder
from uninas.register import Register


def get_network(config_path: str, input_shape: Shape, output_shape: Shape, weights_path: str = None) -> AbstractNetworkBody:
    """
    create a network (model) from a config file, optionally load weights
    """
    builder = Builder()

    # get a new network
    network = builder.load_from_config(find_net_config_path(config_path))
    network.build(s_in=input_shape, s_out=output_shape)

    # load network weights; they are saved from a method, so the keys have to be mapped accordingly
    if isinstance(weights_path, str):
        CheckpointCallback.load_network(weights_path, network)

    return network


def get_imagenet(data_dir: str, num_workers=8, batch_size=8, aug_dict: dict = None) -> AbstractDataSet:
    Builder()

    data_kwargs = {
        "cls_data": "Imagenet1000Data",
        "Imagenet1000Data.fake": False,
        "Imagenet1000Data.dir": replace_standard_paths(data_dir),
        "Imagenet1000Data.num_workers": num_workers,
        "Imagenet1000Data.batch_size_train": batch_size,
        "Imagenet1000Data.batch_size_test": batch_size,

    }
    if aug_dict is None:
        aug_dict = {
            "cls_augmentations": "DartsImagenetAug",
            "DartsImagenetAug#0.crop_size": 224,
        }
    data_kwargs.update(aug_dict)

    # get the data set
    parser = argparse.ArgumentParser()
    cls_data = Register.get(data_kwargs.get('cls_data'))
    cls_data.add_arguments(parser, index=None)
    for i, cls_aug in enumerate([Register.get(cls) for cls in split(data_kwargs.get('cls_augmentations'))]):
        cls_aug.add_arguments(parser, index=i)
    data_args = parser.parse_args(args=[])
    for k, v in data_kwargs.items():
        data_args.__setattr__(k, v)
    return cls_data(data_args)
