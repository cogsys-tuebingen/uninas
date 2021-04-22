from enum import Enum
from typing import Union


class CheckpointType(Enum):
    MODEL = "model"
    STATE = "state"

    def __eq__(self, other: Union['CheckpointType', str]) -> bool:
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, CheckpointType):
            return self.name == other.name
        raise NotImplementedError

    @classmethod
    def all_types(cls):
        return [v.value for v in CheckpointType]


if __name__ == '__main__':
    a = CheckpointType.MODEL
    print(a == "a")
    print(a == CheckpointType.MODEL)
    print(a == CheckpointType.STATE)
    print(a == "model")
    print(CheckpointType.all_types())
