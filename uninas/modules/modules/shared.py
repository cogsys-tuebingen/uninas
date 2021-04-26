import torch
from uninas.modules.modules.abstract import AbstractModule
from uninas.utils.shape import Shape


class _AbstractSharedPathsOp(AbstractModule):

    def __init__(self, name: str, strategy_name: str):
        super().__init__()
        self._add_to_kwargs(name=name, strategy_name=strategy_name)
        self._all_paths = []

    def _add_shared_path(self, p: tuple):
        self._all_paths.append(p)

    def build(self, s_in: Shape, c_out: int) -> Shape:
        return super().build(s_in, c_out)

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # only called for building, path-specific forward passes will use the forward_path method
        return self.forward_path(x, self._all_paths[0])

    def forward_path(self, x: torch.Tensor, path: tuple) -> torch.Tensor:
        raise NotImplementedError

    def config(self, path: tuple, **__) -> dict:
        cfg = super().config()
        return self.config_path(cfg=cfg, path=path, **__)

    def config_path(self, cfg: dict, path: tuple, **__) -> dict:
        raise NotImplementedError


class SpecificPath(AbstractModule):
    """
    wrapper, execute a specific path of the AbstractSharedPathsOp
    """

    def __init__(self, module: _AbstractSharedPathsOp, path: tuple, is_first=True):
        super().__init__()
        self.module = module
        self.path = path
        self._is_first = is_first

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        if self._is_first:
            self.module.build(s_in, c_out)
        return self.module.get_shape_out()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.forward_path(x, self.path)

    def config(self, **__) -> dict:
        return self.module.config(path=self.path, **__)


class AbstractSharedPathsOp(_AbstractSharedPathsOp):
    """
    abstract class that makes using shared weights in different paths a lot easier
    """

    def get_paths_as_modules(self) -> [AbstractModule]:
        return [SpecificPath(self, path, is_first=i == 0) for i, path in enumerate(self._all_paths)]

    def _build(self, s_in: Shape, c_out: int) -> Shape:
        raise NotImplementedError

    def forward_path(self, x: torch.Tensor, path: tuple) -> torch.Tensor:
        raise NotImplementedError

    def config_path(self, cfg: dict, path: tuple, **__) -> dict:
        raise NotImplementedError
