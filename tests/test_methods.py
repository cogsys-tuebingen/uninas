import unittest
import torch
from uninas.main import Main
from uninas.models.networks.uninas.search import SearchUninasNetwork
from uninas.training.result import LogResult
from uninas.methods.darts import DartsSearchMethod
from uninas.utils.args import Argument
from uninas.register import Register


search_darts_config = '{path_conf_tasks}/d1_dartsv1.run_config, {path_conf_net_search}darts.run_config'
super1_fairnas = '{path_conf_tasks}/s1_fairnas.run_config, {path_conf_net_search}fairnas.run_config'
super3 = '{path_conf_tasks}/s3.run_config'
retrain_darts_cifar_config = '{path_conf_tasks}/d2_cifar.run_config'
dna1_config = '{path_conf_tasks}/dna1.run_config, {path_conf_net_search}fairnas.run_config'


@Register.method(search=True)
class TestMaskGradientsSearchMethod(DartsSearchMethod):

    def training_step(self, batch: (torch.Tensor, torch.Tensor), batch_idx: int, **net_kwargs) -> LogResult:
        assert self.training, "The network must be in training mode here"
        result = super().training_step(batch, batch_idx, **net_kwargs)
        print('\n', 'test mask idx', self.hparams.test_mask_idx, self.opt_idx, result.minimize.item())
        if self.opt_idx == self.hparams.test_mask_idx:
            result.minimize *= 0
            print('skipping, setting to', result.minimize.item())
        return result

    @classmethod
    def args_to_add(cls, index=None) -> [Argument]:
        return super().args_to_add(index) + [
            Argument('test_mask_idx', type=int, default=0, help='', is_fixed=True),
        ]


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_search_param_updates(self):
        """
        set gradients of network/architecture weights to zero and ensure that they are not updated, while the others are

        expecting two arc weights:
            28x8, with gradients, the actual weights
            28x8, not requiring gradients, only masks
        """
        for mask_idx in range(2):
            exp1 = Main.new_task(search_darts_config, args_changes={
                "test_mask_idx": mask_idx,
                "cls_method": "TestMaskGradientsSearchMethod",

                "{cls_task}.is_test_run": True,

                "{cls_trainer}.max_epochs": 1,

                "cls_data": "Cifar10Data",
                "{cls_data}.fake": True,
                "{cls_data}.download": False,
                "{cls_data}.batch_size_train": 2,
                "{cls_data}.batch_size_test": -1,

                "{StackedCellsNetworkBody}.cell_order": "n, r, n, r, n",

                "cls_optimizers": "DebugOptimizer, DebugOptimizer",
                # "cls_optimizers": "SGDOptimizer, SGDOptimizer",
                "{cls_optimizers#0}.weight_decay": 0.0,
                "{cls_optimizers#1}.weight_decay": 0.0,
            }, raise_unparsed=False)

            # get initial weights, make copies
            optimizers, _ = exp1.get_method().configure_optimizers()
            assert len(optimizers) == 1, "expecting only one multi optimizer"
            weights = [opt.param_groups[0]['params'][0] for opt in optimizers[0].optimizers]
            weight_copies = [w.clone().detach().cpu() for w in weights]
            s = ['Network', 'Architecture']

            for i, w in enumerate(weights):
                print('%s sample-weight shape:' % s[i], w.shape)

            # run, thus change weights
            exp1.run()

            for i, (w, w_copy) in enumerate(zip(weights, weight_copies)):
                diff = (w.cpu() - w_copy).abs().sum().item()
                if i == mask_idx:
                    if diff > 0.00001:
                        print(w[0:3, 0:3])
                        print(w_copy[0:3, 0:3])
                        assert False, '%s gradients were masked but weights changed anyway; diff: %s' % (s[i], diff)
                else:
                    if diff < 0.00001:
                        print(w[0:3, 0:3])
                        print(w_copy[0:3, 0:3])
                        assert False, '%s gradients were not masked but weights did not change; diff: %s' % (s[i], diff)
            del exp1

    @staticmethod
    def _assert_same_tensors(name: str, t0: torch.Tensor, t1: torch.Tensor):
        if isinstance(t0, (tuple, list)) and isinstance(t1, (tuple, list)):
            for i, (t00, t10) in enumerate(zip(t0, t1)):
                TestMethods._assert_same_tensors('%s-%d' % (name, i), t00, t10)
            return
        diff = (t0 - t1).abs().sum().cpu().item()
        assert diff < 0.001, 'Output difference for "%s", by %s' % (name, diff)

    def test_model_save_load(self):
        """
        make sure that saving+loading for methods/models works correctly, so that we get the same outputs after loading
        """

        for i, (config, trainer, ds, change_net, fix_topology) in enumerate([
            (dna1_config,                   'SimpleTrainer', 'Imagenet1000Data', False, True),
            (super1_fairnas,                'SimpleTrainer', 'Imagenet1000Data', False, True),
            (search_darts_config,           'SimpleTrainer', 'Cifar10Data', True, False),
            (retrain_darts_cifar_config,    'SimpleTrainer', 'Cifar10Data', True, False),

            # (search_darts_config,           'LightningTrainer'),
            # (retrain_darts_cifar_config,    'LightningTrainer'),
        ]):
            save_dir = "{path_tmp}/tests/%d/"
            arg_changes = {
                "cls_data": ds,
                "{cls_data}.fake": True,
                "{cls_data}.batch_size_train": 4,
                "{cls_data}.batch_size_test": 4,

                "cls_trainer": trainer,
                "{cls_trainer}.max_epochs": 2,

                "{cls_task}.seed": 0,
                "{cls_task}.is_test_run": True,
                "{cls_task}.save_dir": save_dir % 1,
                "{cls_task}.save_del_old": True,

                "{cls_schedulers#0}.warmup_epochs": 0,
            }
            if change_net:
                arg_changes.update({
                    "{cls_network_body}.features_first_cell": 8,
                    "{cls_network_body}.cell_order": "n, r, n, r, n",
                    "{cls_network_stem}.features": 4,
                })

            print(config)
            exp1 = Main.new_task(config, args_changes=arg_changes).run()
            data = exp1.get_method().get_data_set().sample_random_data(batch_size=4).cuda()
            net = exp1.get_method().get_network()
            if fix_topology and isinstance(net, SearchUninasNetwork):
                net.set_forward_strategy(False)
                net.get_strategy_manager().forward_const(0)
            with torch.no_grad():
                outputs1 = exp1.get_method()(data)

            arg_changes["{cls_task}.save_dir"] = save_dir % 2
            arg_changes["{cls_task}.seed"] += 1
            exp2 = Main.new_task(config, args_changes=arg_changes).run().load(save_dir % 1)
            net = exp2.get_method().get_network()
            if fix_topology and isinstance(net, SearchUninasNetwork):
                net.set_forward_strategy(False)
                net.get_strategy_manager().forward_const(0)
            with torch.no_grad():
                outputs2 = exp2.get_method()(data)

            for o1, o2 in zip(outputs1, outputs2):
                self._assert_same_tensors('i=%d method=%s' % (i, exp1.get_method().__class__.__name__), o1, o2)

    def test_deterministic_method(self):
        """
        make sure that using the same seed results in the same outcome when:
            - using deterministic cudnn settings
            - using a single worker for data loading
        note that this does not test:
            - the lightning trainer
            - multi gpu setups, ddp distribution
            - advanced augmentation strategies, e.g. mixup
        """
        for i, config in enumerate([
            super1_fairnas,
            super3,
            search_darts_config,
            retrain_darts_cifar_config,
        ]):
            for seed in range(2):
                args_changes = {
                    "{cls_task}.seed": seed,
                    "{cls_task}.is_deterministic": True,
                    "{cls_task}.is_test_run": True,
                    "{cls_task}.save_del_old": True,

                    "{cls_device}.num_devices": 1,
                    "{cls_device}.use_cudnn_benchmark": False,

                    "cls_trainer": "SimpleTrainer",
                    "{cls_trainer}.max_epochs": 1,

                    "cls_data": "Cifar10Data",
                    "{cls_data}.fake": False,
                    "{cls_data}.batch_size_train": 2,
                    "{cls_data}.batch_size_test": -1,
                    "{cls_data}.dir": "{path_data}/cifar_data/",
                    "{cls_data}.download": True,
                    "{cls_data}.num_workers": 0,
                    "{cls_data}.valid_as_test": False,

                    "{cls_schedulers#0}.warmup_epochs": 0,
                }
                exp1 = Main.new_task(config, args_changes=args_changes.copy()).run()
                name = exp1.get_method().__class__.__name__
                data = exp1.get_method().data_set.sample_random_data(batch_size=4).cuda()
                outputs1 = [o.clone().detach() for o in exp1.get_method()(data)]
                del exp1

                exp2 = Main.new_task(config, args_changes=args_changes).run()
                outputs2 = [o.clone().detach() for o in exp2.get_method()(data)]
                del exp2

                for o1, o2 in zip(outputs1, outputs2):
                    self._assert_same_tensors('i=%d seed=%d method=%s' % (i, seed, name), o1, o2)


if __name__ == '__main__':
    unittest.main()
