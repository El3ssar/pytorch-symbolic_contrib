#  Copyright (c) 2022 Szymon Mikler

from pytorch_symbolic import CustomInput
from pytorch_symbolic.symbolic_data import SymbolicFactory


def make_class(value):
    class Payload:
        def __init__(self):
            self.value = value

    return Payload


def test_same_name_different_types_get_distinct_classes():
    cls_a = make_class(1)
    cls_b = make_class(2)
    assert cls_a.__name__ == cls_b.__name__
    assert cls_a is not cls_b

    assert SymbolicFactory(cls_a) is not SymbolicFactory(cls_b)


def test_same_type_reuses_cached_class():
    cls = make_class(3)
    assert SymbolicFactory(cls) is SymbolicFactory(cls)


def test_custom_input_with_same_name_types():
    a = CustomInput(make_class(1)())
    b = CustomInput(make_class(2)())

    assert type(a) is not type(b)
    assert a.v.value == 1
    assert b.v.value == 2
