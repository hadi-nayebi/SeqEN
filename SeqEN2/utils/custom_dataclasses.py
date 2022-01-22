"""Collection of data structures and validations"""

# DefaultVal and NoneRefersDefault -> https://stackoverflow.com/users/2128545/mikeschneeberger
from dataclasses import dataclass, fields
from typing import Any

from dataclasses_json import Undefined, dataclass_json


@dataclass
class DefaultVal:
    val: Any


@dataclass
class NoneRefersDefault:
    def __post_init__(self):
        for field in fields(self):
            # if a field of this data class defines a default value of type
            # `DefaultVal`, then use its value in case the field after
            # initialization has either not changed or is None.
            if isinstance(field.default, DefaultVal):
                field_val = getattr(self, field.name)
                if isinstance(field_val, DefaultVal) or field_val is None:
                    setattr(self, field.name, field.default.val)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class TrainingParams(NoneRefersDefault):
    lr: float = DefaultVal(0.01)
    factor: float = DefaultVal(0.9)
    patience: float = DefaultVal(10000)
    min_lr: float = DefaultVal(0.00001)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class AETrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams())


# @dataclass
# class AAETrainingSettings(NoneRefersDefault):
#     reconstructor: TrainingParams = DefaultVal(TrainingParams())
#     reconstructor: TrainingParams = DefaultVal(TrainingParams())
#     reconstructor: TrainingParams = DefaultVal(TrainingParams())
