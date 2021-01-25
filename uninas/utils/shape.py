from copy import deepcopy
from typing import Union
import torch


class Shape:
    """
    shapes to represent data and module input/output sizes
    by default they do not contain batch size info
    """
    _next_id = 0

    def __init__(self, shape: [int], id_: int = None):
        self.id = self.next_id(id_)
        self.shape = shape

    @classmethod
    def next_id(cls, id_: int = None) -> int:
        if id_ is None:
            cls._next_id = cls._next_id + 1
            id_ = cls._next_id
        return id_

    def copy(self, copy_id=False):
        return self.__class__(self.shape.copy(), id_=self.id if copy_id else None)

    def deepcopy(self):
        return deepcopy(self)

    def random_tensor(self, batch_size=2, device: str = 'cpu') -> torch.Tensor:
        shape = [batch_size] + self.shape
        return torch.randn(size=shape, device=device)

    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def num_dims(self) -> int:
        return len(self.shape)

    def num_features(self):
        return self.shape[0]

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join([str(s) for s in self.shape]))

    def str(self) -> str:
        return str(self)

    def __eq__(self, other):
        if len(self.shape) != len(other.shape):
            return False
        for a, b in zip(self.shape, other.shape):
            if a != b:
                return False
        return True

    def __getitem__(self, item: int) -> int:
        return self.shape[item-1]

    def __setitem__(self, key: int, value: int):
        self.shape[key-1] = value

    @classmethod
    def same_spatial_sizes(cls, shape1, shape2) -> bool:
        return shape1.shape[1] == shape2.shape[1] and shape1.shape[2] == shape2.shape[2]

    @classmethod
    def from_tensor(cls, x: torch.Tensor):
        shape = list(x.shape)
        return cls(shape[1:])


class ShapeList:
    _next_id = 0

    def __init__(self, shapes: [Shape], id_: int = None):
        self.id = Shape.next_id(id_)
        self.shapes = shapes

    def copy(self, copy_id=False):
        return self.__class__(self.shapes.copy(), id_=self.id if copy_id else None)

    def random_tensor(self, batch_size=2, device: str = 'cpu') -> [torch.Tensor]:
        return [s.random_tensor(batch_size=batch_size, device=device) for s in self.shapes]

    def __str__(self):
        return '[%s]' % (', '.join([s.str() for s in self.shapes]))

    def str(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.shapes)

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            return self.__class__(self.shapes[idx])
        return self.shapes[idx]

    def append(self, shape: Shape):
        return self.shapes.append(shape)

    def extend(self, shapes):
        if isinstance(shapes, (list, tuple)):
            return self.shapes.extend(shapes)
        return self.shapes.extend(shapes.shapes)

    def flatten(self, b: True) -> 'ShapeList':
        if b:
            return ShapeList([s[0] if (isinstance(s, ShapeList) and len(s) == 1) else s for s in self.shapes])
        return self

    @property
    def num_features(self) -> int:
        assert len(self.shapes) == 1
        return self.shapes[0].num_features()

    @classmethod
    def from_tensors(cls, x: [torch.Tensor]):
        return cls([Shape.from_tensor(xs) for xs in x])


ShapeOrList = Union[Shape, ShapeList]
