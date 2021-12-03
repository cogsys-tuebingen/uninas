"""
specific kinds of data loaders
"""

from typing import Iterator
from torch.utils.data.distributed import DistributedSampler


class CustomIterator(Iterator):
    def _set_attrs(self, loader):
        for k in ['dataset', 'batch_size', 'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                  'timeout', 'worker_init_fn', 'sampler']:
            self.__setattr__(k, loader.__getattribute__(k))

    def get_dist_sampler(self) -> DistributedSampler:
        sampler = self.__getattribute__("sampler")
        assert isinstance(sampler, DistributedSampler)
        return sampler

    def __next__(self):
        raise NotImplementedError

    def next(self):
        return self.__next__()

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class InfIterator(CustomIterator):
    """
    prevents exceptions from exhausted iterators
    """

    def __init__(self, loader):
        super().__init__()
        self.loader = loader
        self.iterator = None
        self._set_attrs(self.loader)

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next()

    def __next__(self):
        while True:
            try:
                if self.iterator is None:
                    self.iterator = iter(self.loader)
                return self.iterator.__next__()
            except Exception as e:
                del self.iterator
                self.iterator = None

    def __len__(self):
        return len(self.loader)


class MultiLoader(CustomIterator):
    """
    iterates multiple loaders at the same time
    will likely not work with PyTorch Lightning ddp/ddp2 distribution
    """

    def __init__(self, loaders: list):
        super().__init__()
        self.iterators = [InfIterator(loader) for loader in loaders]
        self._set_attrs(self.iterators[0])

    def __next__(self):
        return [it.next() for it in self.iterators]

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next()

    def __len__(self):
        return min([len(it) for it in self.iterators])


class InterleavedLoader(CustomIterator):
    """
    iterates multiple loaders at the same time, interleaving batches
    will likely not work with PyTorch Lightning ddp/ddp2 distribution
    """

    def __init__(self, loaders: list, multiples: list):
        """
        picks multiples[i] batches from loader[i], then switches to loader[i+1] and continues

        :param loaders: list of pytorch data loaders / iterators of data
        :param multiples: list of multiples, same length as loaders
        """
        super().__init__()
        self.iterators = [InfIterator(loader) for loader in loaders]
        self.multiples = multiples
        self._loader_idx, self._mult = 0, 0
        self._set_attrs(self.iterators[0])

    def set_multiples(self, multiples: list):
        self.multiples = multiples

    def __next__(self):
        while self._mult >= self.multiples[self._loader_idx]:
            self._loader_idx = (self._loader_idx+1) % len(self.iterators)
            self._mult = 0
        batch = (self._loader_idx, self.iterators[self._loader_idx].next())
        self._mult += 1
        return batch

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next()

    def __len__(self):
        assert len(self.multiples) == len(self.iterators)
        assert sum(self.multiples) > 0
        cycle_size = sum(self.multiples)
        it_lengths = [len(it) for it in self.iterators]
        it_steps = [999999999999 if m == 0 else it_lengths[i] // m for i, m in enumerate(self.multiples)]
        return min(it_steps) * cycle_size
