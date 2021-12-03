import json
import torch
from uninas.methods.abstract_method import AbstractMethod
from uninas.training.trainer.simple import SimpleTrainer
from uninas.utils.loggers.python import Logger, log_headline
from uninas.utils.paths import get_task_config_path
from uninas.utils.misc import split
from uninas.utils.args import Argument, find_in_args_list, all_meta_args
from uninas.utils.torch.misc import reset_bn as reset_bn_fun


def common_s2_net_args_to_add() -> [Argument]:
    """ list arguments to add to argparse when this class (or a child class) is chosen """
    return [
        # loading network / weights from a previous training session
        Argument('s1_path', default='{path_tmp}', type=str, help='save dir of s1 training', is_path=True),
        Argument('reset_bn', default='False', type=str, help='reset batch norm stats', is_bool=True),
    ]


def common_s2_extend_args(cls, args_list: [str]):
    """
    allow modifying the arguments list before other classes' arguments are dynamically added
    this should be used sparsely, as it is hard to keep track of
    """
    print('\treading arguments of the supernet training to figure out some things:')

    # use the arguments of the supernet training task to figure out the network design, criterion, metrics, ...
    s1_path = find_in_args_list(args_list, ['{cls_task}.s1_path', '%s.s1_path' % cls.__name__])
    s1_path = split(s1_path)
    if len(s1_path) > 1:
        print('\t\thave multiple s1_paths, will use the arguments of the first path')
    arguments_path = get_task_config_path(s1_path[0])
    with open(arguments_path) as args_file:
        args_in_file = json.load(args_file)
        all_meta_args_ = all_meta_args(args_list=None, args_in_file=args_in_file)
        to_ignore = ['cls_task', 'cls_trainer', 'cls_initializers', 'cls_schedulers']
        meta_names = [a for a in all_meta_args_ if a not in to_ignore]
        cls._add_meta_from_argsfile_to_args(all_args=args_list, meta_keys=meta_names,
                                            args_in_file=args_in_file, overwrite=False)

    """
    # ensure that there is validation data, set split to 0.5 if not already in .json and disable shuffling
    cls_data = find_in_args_list(args_list, ['cls_data'])
    valid_split = find_in_args_list(args_list, ['{cls_data}.valid_split', '%s.valid_split' % cls_data])
    if valid_split is None or float(valid_split) <= 0:
        args_list.append('--{cls_data}.valid_split=0.5')
    args_list.append('--{cls_data}.valid_shuffle=False')
    """

    # set all augmentations to use only the test cases
    augmentations = find_in_args_list(args_list, ['cls_augmentations'])
    for i, s in enumerate(split(augmentations)):
        args_list.append('--%s#%d.force_type=test' % (s, i))

    # set fixed method and trainer, disable EMA models
    print('\tfixed:')
    args_list.append('--cls_trainer=%s' % SimpleTrainer.__name__)
    print('\t\tusing %s as cls_trainer, disabling EMA weights' % SimpleTrainer.__name__)
    args_list.append('--cls_initializers=""')
    args_list.append('--cls_scheduler=""')
    print('\t\tusing no cls_scheduler or cls_initializers')


def common_s2_prepare_run(logger: Logger, trainer: SimpleTrainer,
                          s1_path: str, tmp_load_path: str, reset_bn: bool, methods: [AbstractMethod]):
    """
    common things done when starting s2
    """
    # reset batch norm, create temporary new weights
    log_headline(logger, "Recovering trained super-net weights")
    trainer[0].load(s1_path)

    if reset_bn:
        logger.info('Resetting batchnorm statistics')
        for method in methods:
            reset_bn_fun(method.get_network())
    trainer[0].save(tmp_load_path)
    methods[0].get_network().set_forward_strategy(False)
