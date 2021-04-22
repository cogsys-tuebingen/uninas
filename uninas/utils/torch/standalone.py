"""
some functions that make exporting easier
"""

import argparse
from uninas.models.networks.uninas.abstract import AbstractUninasNetwork
from uninas.data.abstract import AbstractDataSet
from uninas.training.callbacks.checkpoint import CheckpointCallback
from uninas.utils.args import ArgumentParser, ArgsTreeNode, arg_list_from_json
from uninas.utils.paths import replace_standard_paths
from uninas.utils.shape import Shape
from uninas.utils.misc import split
from uninas.builder import Builder
from uninas.register import Register
from uninas.main import Main


def get_network(config_path: str, input_shape: Shape, output_shape: Shape, weights_path: str = None) -> AbstractUninasNetwork:
    """
    create a network (model) from a config file, optionally load weights
    """
    builder = Builder()

    # get a new network
    network = builder.load_from_config(Builder.find_net_config_path(config_path))
    network = AbstractUninasNetwork(model_name="standalone", net=network, checkpoint_path="", assert_output_match=True)
    network.build(s_in=input_shape, s_out=output_shape)

    # load network weights; they are saved from a method, so the keys have to be mapped accordingly
    if isinstance(weights_path, str):
        CheckpointCallback.load_network(weights_path, network, num_replacements=1)

    return network


def get_dataset_from_json(path: str, fake=False) -> AbstractDataSet:
    """ parse a task config to re-create the used data set and augmentations """
    Builder()
    args_list = arg_list_from_json(path)
    args_list.append('--{cls_task}.save_dir=""')
    if fake:
        args_list.append('--{cls_data}.fake=True')
    parser = ArgumentParser("tmp")

    node = ArgsTreeNode(Main)
    node.build_from_args(args_list, parser)
    args, wildcards, failed_args, descriptions = node.parse(args_list, parser, raise_unparsed=True)

    return Register.data_sets.get(args.cls_data).from_args(args, index=None)


def get_dataset(data_kwargs: dict) -> AbstractDataSet:
    Builder()
    # get the data set
    parser = argparse.ArgumentParser()
    cls_data = Register.data_sets.get(data_kwargs.get('cls_data'))
    cls_data.add_arguments(parser, index=None)
    classes = [cls_data]
    for i, cls_aug in enumerate([Register.augmentation_sets.get(cls) for cls in split(data_kwargs.get('cls_augmentations'))]):
        cls_aug.add_arguments(parser, index=i)
        classes.append(cls_aug)
    data_args = parser.parse_args(args=[])
    for k, v in data_kwargs.items():
        data_args.__setattr__(k, v)
    for c in classes:
        data_args = c.sanitize_args(data_args)
    return cls_data.from_args(data_args, index=None)


def get_imagenet(data_dir: str, num_workers=8, batch_size=8, aug_dict: dict = None) -> AbstractDataSet:
    data_kwargs = {
        "cls_data": "Imagenet1000Data",
        "Imagenet1000Data.fake": False,
        "Imagenet1000Data.dir": replace_standard_paths(data_dir),
        "Imagenet1000Data.num_workers": num_workers,
        "Imagenet1000Data.batch_size_train": batch_size,
        "Imagenet1000Data.batch_size_test": batch_size,
        "Imagenet1000Data.valid_as_test": True,

    }
    if aug_dict is None:
        aug_dict = {
            "cls_augmentations": "DartsImagenetAug",
            "DartsImagenetAug#0.crop_size": 224,
        }
    data_kwargs.update(aug_dict)
    return get_dataset(data_kwargs)


def get_imagenet16(data_dir: str, num_workers=8, batch_size=8, aug_dict: dict = None) -> AbstractDataSet:
    data_kwargs = {
        "cls_data": "ImageNet16Data",
        "ImageNet16Data.fake": False,
        "ImageNet16Data.dir": replace_standard_paths(data_dir),
        "ImageNet16Data.num_workers": num_workers,
        "ImageNet16Data.batch_size_train": batch_size,
        "ImageNet16Data.batch_size_test": batch_size,

    }
    if aug_dict is None:
        aug_dict = {
            "cls_augmentations": "",
            # "cls_augmentations": "DartsImagenetAug",
            # "DartsImagenetAug#0.crop_size": 224,
        }
    data_kwargs.update(aug_dict)
    return get_dataset(data_kwargs)
