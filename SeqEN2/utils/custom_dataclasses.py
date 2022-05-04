"""Collection of data structures and validations"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from dataclasses_json import Undefined, dataclass_json


@dataclass
class DefaultVal:  # -> https://stackoverflow.com/users/2128545/mikeschneeberger
    val: Any


@dataclass
class NoneRefersDefault:  # -> https://stackoverflow.com/users/2128545/mikeschneeberger
    def __post_init__(self):
        for field in fields(self):
            # if a field of this data class defines a default value of type
            # `DefaultVal`, then use its value in case the field after
            # initialization has either not changed or is None.
            if isinstance(field.default, DefaultVal):
                field_val = getattr(self, field.name)
                if isinstance(field_val, DefaultVal) or field_val is None:
                    setattr(self, field.name, field.default.val)


# decorator to wrap original __init__ -> https://www.geeksforgeeks.org/creating-nested-dataclass-objects-in-python/
def nested_deco(*args, **kwargs):
    """decorator for assigning nested dict to nested dataclass"""

    def wrapper(check_class):
        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):
            # getting class fields to filter extra keys
            class_fields = {f.name for f in fields(check_class)}
            for key in list(kwargs.keys()):
                if key not in class_fields:
                    del kwargs[key]
            for name, value in kwargs.items():
                # getting field type
                ft = check_class.__annotations__.get(name, None)
                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name] = obj
                o_init(self, *args, **kwargs)

        check_class.__init__ = __init__
        return check_class

    return wrapper(args[0]) if args else wrapper


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class TrainingParams(NoneRefersDefault):
    lr: float = DefaultVal(0.01)
    factor: float = DefaultVal(0.99)
    patience: float = DefaultVal(1000)
    min_lr: float = DefaultVal(0.01)


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AETrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AAETrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    generator: TrainingParams = DefaultVal(TrainingParams()).val
    discriminator: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AAECTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    generator: TrainingParams = DefaultVal(TrainingParams()).val
    discriminator: TrainingParams = DefaultVal(TrainingParams()).val
    classifier: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AAECSSTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    generator: TrainingParams = DefaultVal(TrainingParams()).val
    discriminator: TrainingParams = DefaultVal(TrainingParams()).val
    classifier: TrainingParams = DefaultVal(TrainingParams()).val
    ss_decoder: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AAESSTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    generator: TrainingParams = DefaultVal(TrainingParams()).val
    discriminator: TrainingParams = DefaultVal(TrainingParams()).val
    ss_decoder: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AECTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    classifier: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AESSTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    ss_decoder: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class AECSSTrainingSettings(NoneRefersDefault):
    reconstructor: TrainingParams = DefaultVal(TrainingParams()).val
    continuity: TrainingParams = DefaultVal(TrainingParams()).val
    classifier: TrainingParams = DefaultVal(TrainingParams()).val
    ss_decoder: TrainingParams = DefaultVal(TrainingParams()).val


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModularTrainingParams(NoneRefersDefault):
    lr: float = DefaultVal(0.0001)
    factor: float = DefaultVal(0.99)
    patience: float = DefaultVal(1000)
    min_lr: float = DefaultVal(0.0001)
    max_lr: float = DefaultVal(0.001)
    max_loss_change: float = DefaultVal(0.1)
    min_loss_change: float = DefaultVal(0.01)


@dataclass_json(undefined=Undefined.RAISE)
@nested_deco
class ModularTrainingSettings(NoneRefersDefault):
    focused: ModularTrainingParams = DefaultVal(ModularTrainingParams()).val
