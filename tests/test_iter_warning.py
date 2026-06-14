#  Copyright (c) 2022 Szymon Mikler

import warnings

from pytorch_symbolic import Input


def test_warns_when_iterating_unknown_batch():
    x = Input(shape=(2, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nodes = list(x)

    assert len(nodes) == 1  # the placeholder batch dimension is 1
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_no_warning_with_known_batch():
    x = Input(batch_shape=(2, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a, b = x

    assert len(caught) == 0
