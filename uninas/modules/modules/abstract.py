import types
from collections.abc import Iterator
import torch
import torch.nn as nn
from uninas.register import Register
from uninas.utils.shape import Shape, ShapeList, ShapeOrList
from uninas.utils.args import ArgsInterface
from uninas.utils.paths import make_base_dirs
from uninas.utils.torch.misc import randomize_parameters
from typing import Union, List

tensor_type = Union[torch.Tensor, List[torch.Tensor]]


class AbstractModule(nn.Module):
    """
    the basis for all .config() saving + restoring
    """

    def __init__(self, *_, **__):
        nn.Module.__init__(self)

        if len(_) > 0:
            print('unknown args (%s):' % self.__class__.__name__, __)
        if len(__) > 0:
            print('unknown kwargs (%s):' % self.__class__.__name__, __)

        # dicts that contain the keys of everything that goes into a config and can be restored
        self._kwargs = []           # saved, printed
        self._np_kwargs = []        # saved, not printed
        self._p_kwargs = []         # not saved, printed
        self._submodules = []
        self._submodule_lists = []
        self._submodule_dicts = []

        self._add_to_print_kwargs(**__)
        self.dropout_rate = None

        self.cached = dict(built=False)  # some info about shapes in/out

    def set(self, **kwargs):
        """ set new value to a parameter and kwargs / misc_kwargs """
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def get_cached(self, k: str, default=None):
        return self.cached.get(k, default)

    def is_built(self) -> bool:
        return self.cached.get("built", False)

    def get_shape_in(self, may_be_none=False) -> ShapeOrList:
        s_in = self.get_cached('shape_in')
        if not may_be_none:
            assert isinstance(s_in, ShapeOrList.__args__)
        return s_in

    def get_shape_out(self, may_be_none=False) -> ShapeOrList:
        s_out = self.get_cached('shape_out')
        if not may_be_none:
            assert isinstance(s_out, ShapeOrList.__args__)
        return s_out

    # listing modules/kwargs to save+restore via configs ---------------------------------------------------------------

    def _add(self, lst: list, are_modules=False, **kwargs):
        for k, v in kwargs.items():
            lst.append(k)
            if are_modules:
                self.add_module(k, v)
            else:
                self.__dict__[k] = v

    def _add_to_kwargs(self, **kwargs):
        """ store named values (not Modules, which need to have config stored and be rebuilt) """
        self._add(self._kwargs, are_modules=False, **kwargs)

    def _add_to_kwargs_np(self, **kwargs):
        """ store named values (not Modules, which need to have config stored and be rebuilt) """
        self._add(self._np_kwargs, are_modules=False, **kwargs)

    def _add_to_print_kwargs(self, **kwargs):
        """ store named values for printing only """
        self._add(self._p_kwargs, are_modules=False, **kwargs)

    def _add_to_submodules(self, **kwargs):
        """ store named modules """
        self._add(self._submodules, are_modules=True, **kwargs)

    def _add_to_submodule_lists(self, **kwargs):
        """ store named lists of modules (nn.ModuleList) """
        self._add(self._submodule_lists, are_modules=True, **kwargs)

    def _add_to_submodule_dict(self, **kwargs):
        """ store named dicts of modules (nn.ModuleDict) """
        self._add(self._submodule_dicts, are_modules=True, **kwargs)

    def kwargs(self):
        return {k: self.__dict__[k] for k in self._kwargs+self._np_kwargs}

    def config(self, **_) -> dict:
        """
        get a dictionary describing this module, so that a builder can assemble it correctly again
        subclasses may receive specific instructions via kwargs, e.g. whether to finalize a search architecture
        """
        cfg_keys = ['kwargs', 'submodules', 'submodule_lists', 'submodule_dicts']
        cfg = dict(name=self.__class__.__name__)
        cfg.update({k: {} for k in cfg_keys})

        for k in self._kwargs+self._np_kwargs:
            cfg['kwargs'][k] = self.__dict__[k]
        for k in self._submodules:
            cfg['submodules'][k] = self._modules[k].config(**_)
        for k in self._submodule_lists:
            lst = self._modules[k]
            cfg['submodule_lists'][k] = [v.config(**_) if v is not None else None for v in iter(lst)]
        for k in self._submodule_dicts:
            dct = self._modules[k]
            cfg['submodule_dicts'][k] = {dk: dv.config(**_) if dv is not None else None for dk, dv in dct.items()}

        # remove empty dicts
        for k in list(cfg_keys):
            if len(cfg[k]) == 0:
                cfg.pop(k)
        return cfg

    @classmethod
    def from_config(cls, **kwargs):
        """ upon receiving a dictionary as created in self.config(), reassemble this module properly """
        kwargs_ = kwargs.pop('kwargs', {})
        submodules_ = {k: Register.builder.from_config(v) if v is not None else None
                       for k, v in kwargs.pop('submodules', {}).items()}
        submodule_lists_ = {k: nn.ModuleList([Register.builder.from_config(v) if v is not None else None for v in lst])
                            for k, lst in kwargs.pop('submodule_lists', {}).items()}
        submodule_dicts_ = {k: {dk: Register.builder.from_config(dv) if dv is not None else None
                                for dk, dv in dct.items()}
                            for k, dct in kwargs.pop('submodule_dicts', {}).items()}
        return cls(**kwargs_, **submodules_, **submodule_lists_, **submodule_dicts_, **kwargs)

    # presenting as string ---------------------------------------------------------------------------------------------

    def _str_kwargs(self) -> str:
        lst = []
        for k in self._kwargs+self._p_kwargs:
            lst.append('%s=%s' % (k, str(self.__dict__[k])))
        return ', '.join(lst)

    @staticmethod
    def _str_tuple_submodule(obj, depth: int, max_depth: int, name='') -> [(int, str)]:
        """ describe this module via indentation instructions and strings """
        ss = []
        if obj is not None and len(obj) > 0:
            if depth < max_depth:
                if isinstance(obj, (dict, nn.ModuleDict)):
                    for n, m in obj.items():
                        if isinstance(m, AbstractModule):
                            ss += m.str_tuples(depth=depth+1, max_depth=max_depth, name=n)
                        else:
                            ss += AbstractModule._str_tuple_submodule(m, depth + 1, max_depth, name=n)
                elif isinstance(obj, (list, nn.ModuleList)):
                    for i, m in enumerate(obj):
                        n = '(%d)' % i
                        if isinstance(m, AbstractModule):
                            ss += m.str_tuples(depth=depth+1, max_depth=max_depth, name=n)
                        else:
                            ss += AbstractModule._str_tuple_submodule(m, depth + 1, max_depth, name=n)
            else:
                ss.append((depth, '<%d entries>' % (len(obj))))
        if len(ss) == 0:
            return []
        s0, s1 = '%s = [' % name, ']'
        if len(ss) == 1:
            return [(depth, s0 + ss[0][1] + s1)]
        return [(depth, s0)] + ss + [(depth, s1)]

    def str_tuples(self, depth=0, max_depth=5, name='', add_s=None, add_sl=None, add_sd=None) -> [(int, str)]:
        """ describe this module via indentation instructions and strings """
        add_s = {} if add_s is None else add_s.copy()
        add_sl = {} if add_sl is None else add_sl.copy()
        add_sd = {} if add_sd is None else add_sd.copy()
        add_s['Modules'] = {k: self._modules[k] for k in self._submodules}
        add_sl['Module Lists'] = {k: self._modules[k] for k in self._submodule_lists}
        add_sd['Module Dicts'] = {k: self._modules[k] for k in self._submodule_dicts}
        s0 = '{n}{cls}({k}) ['.format(**{
            'n': ('%s = ' % name) if len(name) > 0 else '',
            'cls': self.__class__.__name__,
            'k': self._str_kwargs(),
        })
        s1 = ']'
        if depth >= max_depth:
            ss = [(depth, '<%d modules, %d module lists, %d module dicts>' % (len(add_s), len(add_sl), len(add_sd)))]
        else:
            ss = []
            for k, v in add_s.copy().items():
                ss.extend(self._str_tuple_submodule(v, depth+1, max_depth, name=k))
            for k, v in add_sl.copy().items():
                ss.extend(self._str_tuple_submodule(v, depth+1, max_depth, name=k))
            for k, v in add_sd.copy().items():
                ss.extend(self._str_tuple_submodule(v, depth+1, max_depth, name=k))
        ss = [s for s in ss if s is not None]
        if len(ss) == 0:
            return [(depth, s0 + s1)]
        if len(ss) == 1:
            return [(depth, s0 + ss[0][1] + s1)]
        return [(depth, s0)] + ss + [(depth, s1)]

    def str(self, depth=0, max_depth=5, name='', add_s=None, add_sl=None, add_sd=None) -> str:
        strings = self.str_tuples(depth, max_depth, name, add_s, add_sl, add_sd)
        return ''.join('\n%s%s' % ('. '*d, s) for d, s in strings)

    # (recursive) utility ----------------------------------------------------------------------------------------------

    @classmethod
    def _get_base_modules(cls, m) -> list:
        base_modules = []
        if isinstance(m, AbstractModule):
            base_modules.append(m)
        elif isinstance(m, nn.ModuleList):
            for m2 in iter(m):
                base_modules.extend(cls._get_base_modules(m2))
        elif isinstance(m, nn.ModuleDict):
            for m2 in m.values():
                base_modules.extend(cls._get_base_modules(m2))
        return base_modules

    def base_modules(self, recursive=True) -> Iterator:
        """ yield all base modules, therefore all layers/modules of this project """
        fun = self.modules if recursive else self.children
        for m in fun():
            for m2 in self._get_base_modules(m):
                yield m2

    def base_modules_by_condition(self, condition, recursive=True) -> Iterator:
        """ get list of all base modules that pass a condition, condition is a function that returns a boolean """
        for m in self.base_modules(recursive=recursive):
            if condition(m):
                yield m

    def hierarchical_base_modules(self) -> (type, ShapeOrList, ShapeOrList, list):
        """ get a hierarchical/recursive representation of (class, shapes_in, shapes_out, submodules) """
        submodules = list(self.base_modules(recursive=False))
        r0 = self.get_shape_in(may_be_none=True)
        r1 = self.get_shape_out(may_be_none=True)
        r2 = [m.hierarchical_base_modules() for m in submodules]
        return self, r0, r1, r2

    def set_dropout_rate(self, p=None) -> int:
        """ set the dropout rate of every dropout layer to p, no change for p=None. return num of affected modules """
        n = 0
        for m in self.base_modules(recursive=False):
            n += m.set_dropout_rate(p)
        return n

    def get_device(self) -> torch.device:
        """ get the device of one of the weights """
        for w in self.parameters():
            return w.device

    def is_layer(self, cls) -> bool:
        return isinstance(self, cls)

    # building and running ---------------------------------------------------------------------------------------------

    def probe_outputs(self, s_in: ShapeOrList, module: nn.Module = None, multiple_outputs=False) -> ShapeOrList:
        """ returning the output shape of one forward pass using zero tensors """
        with torch.no_grad():
            if module is None:
                module = self
            x = s_in.random_tensor(batch_size=2)
            s = module(x)
            if multiple_outputs:
                return ShapeList([Shape(list(sx.shape)[1:]) for sx in s])
            return Shape(list(s.shape)[1:])

    def build(self, *args, **kwargs) -> ShapeOrList:
        """ build/compile this module, save input/output shape(s), return output shape """
        assert not self.is_built(), "The module is already built"
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, (Shape, ShapeList)):
                self.cached['shape_in'] = arg.copy(copy_id=True)
                break
        s_out = self._build(*args, **kwargs)
        self.cached['shape_out'] = s_out.copy(copy_id=True)
        self.cached['built'] = True
        return s_out

    def _build(self, *args, **kwargs) -> ShapeOrList:
        """ build/compile this module, return output shape """
        raise NotImplementedError

    def forward(self, x: tensor_type) -> tensor_type:
        raise NotImplementedError

    def export_onnx(self, save_path: str, **kwargs):
        save_path = make_base_dirs(save_path)
        x = self.get_shape_in(may_be_none=False).random_tensor(batch_size=2).to(self.get_device())
        torch.onnx.export(model=self, args=x, f=save_path, **kwargs)

    # can disable state dict

    def disable_state_dict(self):
        """
        makes the state_dict irreversibly disfunctional for this module and all children
        this is used to prevent specific modules to save/load
        """

        def state_dict(self_, *args, **kwargs):
            return None

        def _load_from_state_dict(self_, *args, **kwargs):
            pass

        def _disable_state_dict(module: nn.Module):
            for name, child in module._modules.items():
                if child is not None:
                    _disable_state_dict(child)
            module.state_dict = types.MethodType(state_dict, self)
            module._load_from_state_dict = types.MethodType(_load_from_state_dict, self)

        _disable_state_dict(self)
        _disable_state_dict = None

    # misc -------------------------------------------------------------------------------------------------------------

    def randomize_parameters(self):
        """ set all parameters to normally distributed values """
        randomize_parameters(self)


class AbstractArgsModule(AbstractModule, ArgsInterface):
    """
    an AbstractModule that can easily store+reuse the parsed argparse arguments of previous times
    """

    def __init__(self, *_, **kwargs_to_store):
        AbstractModule.__init__(self, *_)
        ArgsInterface.__init__(self)
        self._add_to_kwargs(**kwargs_to_store)

    def _build(self, *args) -> ShapeOrList:
        raise NotImplementedError

    def forward(self, x: tensor_type) -> tensor_type:
        raise NotImplementedError
