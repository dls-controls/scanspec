import pytest

from scanspec.specs import Line


def test_line_view() -> None:
    x = "x"
    inst = Line(x, 0, 1, 5)
    path = inst.path()
    assert len(path) == 5
    dim = path.consume()
    assert dim.midpoints == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1.0])}
    assert len(path) == 0


def test_line_midpoints() -> None:
    x = "x"
    inst = Line(x, 0, 1, 5)
    it = inst.midpoints()
    assert it.axes == [x]
    assert len(it) == 5
    midpoints = [d[x] for d in it]
    assert midpoints == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
