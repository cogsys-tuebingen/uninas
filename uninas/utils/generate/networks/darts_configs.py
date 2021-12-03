"""
generate network configs from genotypes
currently ignoring normal_concat, reduce_concat
"""

import argparse
from collections import namedtuple

from uninas.builder import Builder
from uninas.methods.strategy_manager import StrategyManager
from uninas.main import Main
from uninas.utils.paths import replace_standard_paths, get_net_config_dir


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat primitives source')

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DartsPrimitives',
    source='original/darts'
)

DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DartsPrimitives',
    source='original/darts'
)

PDARTS = Genotype(
    normal=[
        ('skip_connect', 0),
        ('dil_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 4)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('dil_conv_3x3', 1),
        ('dil_conv_3x3', 1),
        ('dil_conv_5x5', 3)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DartsPrimitives',
    source='original/darts'
)

PR_DARTS_DL1 = Genotype(
    normal=[
        ('sep_conv_5x5_2', 0),
        ('sep_conv_3x3_2', 1),
        ('sep_conv_3x3_2', 1),
        ('skip_connect', 2),
        ('sep_conv_3x3_2', 1),
        ('skip_connect', 2),
        ('skip_connect', 0),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3_2', 0),
        ('skip_connect', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_5x5_2', 2),
        ('sep_conv_3x3_2', 0),
        ('sep_conv_3x3_2', 1),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3_2', 4)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DNU_PRDartsPrimitives',
    source='original/darts'
)

PR_DARTS_DL2 = Genotype(
    normal=[
        ('sep_conv_3x3_2', 0),
        ('sep_conv_3x3_1', 1),
        ('sep_conv_3x3_1', 1),
        ('sep_conv_7x7_2', 2),
        ('skip_connect', 0),
        ('sep_conv_5x5_2', 1),
        ('sep_conv_3x3_2', 1),
        ('sep_conv_7x7_2', 4)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('sep_conv_3x3_1', 0),
        ('sep_conv_3x3_2', 1),
        ('sep_conv_3x3_2', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3_1', 3),
        ('sep_conv_3x3_1', 0),
        ('skip_connect', 3)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DNU_PRDartsPrimitives',
    source='original/darts'
)

ASAP = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('skip_connect', 2),
        ('skip_connect', 1),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('skip_connect', 2)
    ],
    reduce_concat=[2, 3, 4, 5],
    primitives='DartsPrimitives',
    source='original/darts'
)

MdeNAS = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 3),
        ('sep_conv_3x3', 1),
        ('dil_conv_5x5', 3),
        ('max_pool_3x3', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('skip_connect', 0),
        ('skip_connect', 1),
        ('sep_conv_3x3', 3),
        ('skip_connect', 2),
        ('dil_conv_3x3', 3),
        ('sep_conv_5x5', 0)
    ],
    reduce_concat=range(2, 6),
    primitives='DartsBnPrimitives',
    source='original/darts'
)


def compact_from_name(name: str, verbose=True) -> (Genotype, dict):
    genotype = eval(name)
    if verbose:
        print('Genotype:\t%s' % str(genotype))

    primitive_names = {
        # primitive names in Genotypes and their corresponding number in the list in primitives/*.py
        'DartsPrimitives': {
            'sep_conv_3x3': 0,
            'sep_conv_5x5': 1,
            'dil_conv_3x3': 2,
            'dil_conv_5x5': 3,
            'max_pool_3x3': 4,
            'avg_pool_3x3': 5,
            'skip_connect': 6,
            'none': 7,
        },
        'DNU_PRDartsPrimitives': {
            'sep_conv_3x3_2': 0,
            'sep_conv_5x5_2': 1,
            'dil_conv_3x3_2': 2,
            'dil_conv_5x5_2': 3,
            'max_pool_3x3': 4,
            'avg_pool_3x3': 5,
            'skip_connect': 6,
            'sep_conv_3x3_1': 7,
            'sep_conv_5x5_1': 8,
            'sep_conv_7x7_1': 9,
            'sep_conv_7x7_2': 10,
        },
    }
    primitive_names['DartsBnPrimitives'] = primitive_names['DartsPrimitives']
    op_to_id = primitive_names.get(genotype.primitives)

    def to_list(names_inputs: list):
        lst = []
        for i, (name_, input_) in enumerate(names_inputs):
            if i % 2 == 0:
                lst.append([])
            lst[-1].append((op_to_id[name_], input_))
        return lst

    compact = {
        'n': to_list(genotype.normal),
        'n_concat': genotype.normal_concat,
        'r': to_list(genotype.reduce),
        'r_concat': genotype.reduce_concat,
        'primitives': genotype.primitives,
    }
    return genotype, compact


def generate_from_name(name: str, save=True, verbose=True):
    genotype, compact = compact_from_name(name, verbose=verbose)
    run_configs = '{path_conf_tasks}/d1_dartsv1.run_config, {path_conf_net_search}darts.run_config'
    # create weight sharing cell model
    changes = {
        'cls_data': 'Cifar10Data',
        '{cls_data}.fake': True,

        '{cls_task}.save_del_old': False,

        '{cls_network_body}.cell_order': 'n, r',
        '{cls_network_body}.features_first_cell': 36*4,
        '{cls_network_stem}.features': 36*3,

        'cls_network_cells_primitives': "%s, %s" % (compact.get('primitives'), compact.get('primitives')),
    }
    task = Main.new_task(run_configs, args_changes=changes)
    net = task.get_method().get_network()
    args = task.args

    wss = StrategyManager().get_strategies()
    assert len(wss) == 1
    ws = wss[list(wss.keys())[0]]

    # fix arc, all block inputs use different weights
    # go through all weights in the search cell
    for n, w in ws.named_parameters_single():
        # figure out cell type ("normal", "reduce"), block index, and if it's the first, second, ... op of that block
        c_type, block_idx, num_inputs, num_idx = n.split('/')[-4:]
        block_idx = int(block_idx.split('-')[-1])
        num_idx = int(num_idx.split('-')[-1])
        # set all paths weights to zero
        w.data.zero_()
        # go through the cell description of the genotype, if input and op number match, set the weight to be higher
        for op_idx, from_idx in compact.get(c_type)[block_idx]:
            if num_idx == from_idx:
                w[op_idx] = 1
    ws.forward()

    # saving config now will only use the highest weighted connections, since we have a search network
    cfg = net.config(finalize=True, num_block_ops=2)
    if save:
        path = Builder.save_config(cfg, get_net_config_dir(genotype.source), name)
        print('Saved config: %s' % path)
    return net, cfg, args


def main():
    parser = argparse.ArgumentParser(description='uninas generate a network config from simple genotype description')
    parser.add_argument('--cells', type=str, default=None, help='which config to generate, all available if None')
    args = parser.parse_args()
    args.save_dir = replace_standard_paths('{path_conf_net_originals}/')

    if args.cells is not None:
        all_cell_names = [args.cells]
    else:
        all_cell_names = []
        for key, value in list(globals().items()):
            if isinstance(value, Genotype):
                all_cell_names.append(key)

    for cell_name in all_cell_names:
        print('Name:\t\t%s' % cell_name)
        generate_from_name(cell_name)


if __name__ == '__main__':
    main()
