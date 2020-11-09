import os
from typing import Union


class PbtSave:
    """
    class to keep track of checkpoints and log_dicts
    """

    def __init__(self, epoch: int, client_id: int, log_dict: dict, path: str):
        self.key = "%d-%d" % (epoch, client_id)  # unique, to save in a dict
        self.epoch = epoch
        self.client_id = client_id
        self.log_dict = log_dict
        self._path = path
        self._current_usages = 0

    def reset(self):
        self._current_usages = 0

    def add_usage(self):
        self._current_usages += 1

    def is_used(self) -> bool:
        return self._current_usages > 0

    def get_path(self) -> Union[str, None]:
        # simplifies selector code
        return self._path if self.is_used() else None

    def remove_file(self):
        if isinstance(self._path, str) and os.path.isfile(self._path):
            os.remove(self._path)
