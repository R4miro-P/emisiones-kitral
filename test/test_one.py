import numpy as np


def breakit():
    # fmt: off
    from IPython.terminal.embed import InteractiveShellEmbed
    from qgis.PyQt.QtCore import pyqtRemoveInputHook  # type: ignore
    pyqtRemoveInputHook()
    # fmt: on
    return InteractiveShellEmbed()


def test_esto():
    assert True, "no es verdad"


def test_esto_otro():
    from fuego_superficial import surface_fuel_consumed_vectorized

    a = np.random.randint(0, 100, (3, 4))
    b = np.random.randint(0, 100, (3, 4))
    c = surface_fuel_consumed_vectorized(a, b)
    assert c.shape == a.shape, "no es verdad"
    assert np.all(c == a * b), "no es verdad"
