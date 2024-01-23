from typing import Any, Tuple

import pytest

from scanspec.core import Path, SnakedFrames
from scanspec.regions import Circle, Ellipse, Polygon, Rectangle
from scanspec.specs import (
    DURATION,
    Concat,
    Line,
    Mask,
    Repeat,
    Spec,
    Spiral,
    Squash,
    Static,
    Zip,
    fly,
    step,
)

x, y, z = "x", "y", "z"


def ints(s):
    return pytest.approx([int(t) for t in s])


def test_one_point_duration() -> None:
    duration = Static.duration(1.0)
    (dim,) = duration.calculate()
    assert dim.midpoints == {DURATION: pytest.approx([1.0])}
    assert dim.lower == {DURATION: pytest.approx([1.0])}
    assert dim.upper == {DURATION: pytest.approx([1.0])}
    assert not isinstance(dim, SnakedFrames)
    assert dim.gap == ints("0")


def test_one_point_line() -> None:
    inst = Line(x, 0, 1, 1)
    (dim,) = inst.calculate()
    assert dim.midpoints == {x: pytest.approx([0])}
    assert dim.lower == {x: pytest.approx([-0.5])}
    assert dim.upper == {x: pytest.approx([0.5])}
    assert not isinstance(dim, SnakedFrames)
    assert dim.gap == ints("1")


def test_two_point_line() -> None:
    inst = Line(x, 0, 1, 2)
    (dim,) = inst.calculate()
    assert dim.midpoints == {x: pytest.approx([0, 1])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5])}
    assert dim.gap == ints("10")


def test_two_point_stepped_line() -> None:
    inst = step(Line(x, 0, 1, 2), 0.1)
    dimx, dimt = inst.calculate()
    assert dimx.midpoints == dimx.lower == dimx.upper == {x: pytest.approx([0, 1])}
    assert (
            dimt.midpoints == dimt.lower == dimt.upper == {DURATION: pytest.approx([0.1])}
    )
    assert inst.frames().gap == ints("11")


def test_two_point_fly_line() -> None:
    inst = fly(Line(x, 0, 1, 2), 0.1)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1]),
        DURATION: pytest.approx([0.1, 0.1]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5]),
        DURATION: pytest.approx([0.1, 0.1]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5]),
        DURATION: pytest.approx([0.1, 0.1]),
    }
    assert dim.gap == ints("10")


def test_many_point_line() -> None:
    inst = Line(x, 0, 1, 5)
    (dim,) = inst.calculate()
    assert dim.midpoints == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: pytest.approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: pytest.approx([0.125, 0.375, 0.625, 0.875, 1.125])}
    assert dim.gap == ints("10000")


def test_one_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 1)
    assert inst == Line(x, 0.5, 1.5, 1)


def test_many_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 4)
    assert inst == Line(x, 0.125, 0.875, 4)


def test_spiral() -> None:
    inst = Spiral(x, y, 0, 10, 5, 50, 10)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        y: pytest.approx(
            [5.4, 6.4, 19.7, 23.8, 15.4, 1.7, -8.6, -10.7, -4.1, 8.3], abs=0.1
        ),
        x: pytest.approx(
            [0.3, -0.9, -0.7, 0.5, 1.5, 1.6, 0.7, -0.6, -1.8, -2.4], abs=0.1
        ),
    }
    assert dim.lower == {
        y: pytest.approx(
            [10.0, 2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6], abs=0.1
        ),
        x: pytest.approx(
            [0.0, -0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2], abs=0.1
        ),
    }
    assert dim.upper == {
        y: pytest.approx(
            [2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6, 15.3], abs=0.1
        ),
        x: pytest.approx(
            [-0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2, -2.4], abs=0.1
        ),
    }
    assert not isinstance(dim, SnakedFrames)
    assert dim.gap == ints("1000000000")


def test_spaced_spiral() -> None:
    inst = Spiral.spaced(x, y, 0, 10, 5, 1)
    assert inst == Spiral(x, y, 0, 10, 10, 10, 78)


def test_zipped_lines() -> None:
    inst = Line(x, 0, 1, 5).zip(Line(y, 1, 2, 5))
    assert inst.dimension_info().axes == ((x, y),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        y: pytest.approx([1, 1.25, 1.5, 1.75, 2]),
    }
    assert dim.gap == ints("10000")


def test_zipped_snaked_lines() -> None:
    inst = Line(x, 0, 1, 5).zip(~Line(y, 1, 2, 5))
    with pytest.raises(AssertionError) as ae:
        inst.calculate()
    assert ae.match("Mismatching types")


def test_zipped_both_snaked_lines() -> None:
    inst = (~Line(x, 0, 1, 5)).zip(~Line(y, 1, 2, 5))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((x, y),)
    assert dimension_info.snaked == (True,)
    assert dimension_info.shape == (5,)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        y: pytest.approx([1, 1.25, 1.5, 1.75, 2]),
    }
    assert dim.gap == ints("00000")  # Why is this 00000 not 10000?


def test_concat_snaked_lines() -> None:
    inst = Line(y, 0, 1, 5).concat(~Line(y, 1, 2, 5))
    with pytest.raises(AssertionError) as ae:
        inst.calculate()
    assert ae.match("Mismatching types")


def test_concat_both_snaked_lines() -> None:
    inst = (~Line(y, 0, 1, 5)).concat(~Line(y, 1, 2, 5))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y,),)
    assert dimension_info.snaked == (True,)
    assert dimension_info.shape == (10,)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        y: pytest.approx([0, 0.25, 0.5, 0.75, 1, 1, 1.25, 1.5, 1.75, 2]),
    }
    assert dim.gap == ints("0000010000")


def test_product_lines() -> None:
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 2)
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y,), (x,))
    assert dimension_info.shape == (3, 2)
    dims = inst.calculate()
    assert len(dims) == 2 == len(dimension_info.shape)
    dim = Path(dims).consume()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_zipped_product_lines() -> None:
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 5).zip(Line(z, 2, 3, 5))
    assert inst.dimension_info().axes == ((y,), (x, z))
    dimy, dimxz = inst.calculate()
    assert dimxz.midpoints == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        z: pytest.approx([2, 2.25, 2.5, 2.75, 3]),
    }
    assert dimy.midpoints == {
        y: pytest.approx([1, 1.5, 2]),
    }
    assert inst.frames().gap == ints("100001000010000")


def test_zipping_multiple_axes() -> None:
    spiral = Spiral(x, y, 0, 10, 5, 50, 10)
    spiral_midpoints = {
        y: pytest.approx(
            [5.4, 6.4, 19.7, 23.8, 15.4, 1.7, -8.6, -10.7, -4.1, 8.3], abs=0.1
        ),
        x: pytest.approx(
            [0.3, -0.9, -0.7, 0.5, 1.5, 1.6, 0.7, -0.6, -1.8, -2.4], abs=0.1
        ),
    }
    inst = spiral.zip(Line(z, 0, 9, 10))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y, x, z),)  # Spiral reverses y, x axes
    assert dimension_info.shape == (10,)
    (dimyxz,) = inst.calculate()
    assert dimyxz.midpoints == {
        **spiral_midpoints,
        z: pytest.approx([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    }
    assert inst.frames().gap == ints("1000000000")

    inst = spiral.zip(Line(z, 0, 0, 1))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y, x, z),)
    assert dimension_info.shape == (10,)
    (dimyxz,) = inst.calculate()
    assert dimyxz.midpoints == {**spiral_midpoints, z: pytest.approx([0] * 10)}
    assert inst.frames().gap == ints("1000000000")


def test_zipping_higher_dimensionality() -> None:
    # If dimensions(left) > dimensions(right), zip from innermost to outermost
    grid = Line(y, 1, 2, 3) * Line(x, 0, 1, 2)
    dimy_midpoints = {
        y: pytest.approx([1, 1.5, 2]),
    }
    dimx_midpoints = {
        x: pytest.approx([0, 1]),
    }

    inst = grid.zip(Line(z, 0, 5, 2))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y,), (x, z))
    assert dimension_info.shape == (3, 2)
    dimy, dimxz = inst.calculate()
    assert dimy.midpoints == dimy_midpoints
    assert dimxz.midpoints == {**dimx_midpoints, z: pytest.approx([0, 5])}

    threed_grid = Line(z, 0, 5, 2) * grid
    inst = threed_grid.zip(Line("p", 0, 5, 3) * Line("q", 0, 5, 2))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((z,), (y, "p"), (x, "q"))
    assert dimension_info.shape == (2, 3, 2)
    dimz, dimyp, dimxq = inst.calculate()
    assert dimz.midpoints == {z: pytest.approx([0, 5])}
    assert dimyp.midpoints == {**dimy_midpoints, "p": pytest.approx([0, 2.5, 5])}
    assert dimxq.midpoints == {**dimx_midpoints, "q": pytest.approx([0, 5])}

    # If dimensions(right) == 1 and len(dimensions(right)[0]) == 1,
    # dimensions(right)[0] *= len(dimensions(left)[-1]
    inst = grid.zip(Line(z, 0, 0, 1))
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y,), (x, z))
    assert dimension_info.shape == (3, 2)
    dimy, dimxz = inst.calculate()
    assert dimy.midpoints == dimy_midpoints
    assert dimxz.midpoints == {**dimx_midpoints, z: pytest.approx([0] * 2)}


def test_squashed_product() -> None:
    inst = Squash(Line(y, 1, 2, 3) * Line(x, 0, 1, 2))
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_squashed_multiplied_snake_scan() -> None:
    inst = Line(z, 1, 2, 2) * Squash(
        Line(y, 1, 2, 2) * ~Line.bounded(x, 3, 7, 2) * Static.duration(9, 2)
    )
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((z,), (y, x, DURATION))
    assert dimension_info.snaked == (False, False)
    dimz, dimxyt = inst.calculate()
    for d in dimxyt.midpoints, dimxyt.lower, dimxyt.upper:
        assert d == {
            x: pytest.approx([4, 4, 6, 6, 6, 6, 4, 4]),
            y: pytest.approx([1, 1, 1, 1, 2, 2, 2, 2]),
            DURATION: pytest.approx([9, 9, 9, 9, 9, 9, 9, 9]),
        }
    assert dimz.midpoints == dimz.lower == dimz.upper == {z: pytest.approx([1, 2])}
    assert inst.frames().gap == ints("1010101010101010")


def test_product_snaking_lines() -> None:
    inst = Line(y, 1, 2, 3) * ~Line(x, 0, 1, 2)
    dimension_info = inst.dimension_info()
    assert dimension_info.axes == ((y,), (x,))
    assert dimension_info.snaked == (False, True)
    dims = inst.calculate()
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 1, 0, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, 1.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, -0.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_concat_lines() -> None:
    inst = Concat(Line(x, 0, 1, 2), Line(x, 1, 2, 3))
    assert inst.dimension_info().axes == ((x,),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {x: pytest.approx([0, 1, 1, 1.5, 2])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5, 1.25, 1.75, 2.25])}
    assert dim.gap == ints("10100")


def test_rect_region() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1, 1.5, 2.2)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_rect_region_3D() -> None:
    inst = Static(z, 3.2, 2) * Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    )
    assert inst.dimension_info().axes == ((z,), (y, x))
    zdim, xydim = inst.calculate()
    assert zdim.midpoints == {z: pytest.approx([3.2, 3.2])}
    assert zdim.midpoints is zdim.upper
    assert zdim.midpoints is zdim.lower
    assert xydim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert inst.frames().gap == ints("101010101010")


def test_rect_region_union() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) | Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 2, 0, 1, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2.5, 2.5]),
    }
    assert dim.gap == ints("1010010010")


def test_rect_region_intersection() -> None:
    inst = (
            Line(y, 1, 3, 5) * Line(x, 0, 2, 3)
            & Rectangle(x, y, 0, 1, 1.5, 2.2)
            & Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    )
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([1, 1]),
        y: pytest.approx([1.5, 2]),
    }
    assert dim.gap == ints("11")


def test_rect_region_difference() -> None:
    # Bracket to force testing Mask.__sub__ rather than Region.__sub__
    inst = (
                   Line(y, 1, 3, 5) * Line(x, 0, 2, 3).zip(Static(DURATION, 0.1))
                   & Rectangle(x, y, 0, 1, 1.5, 2.2)
           ) - Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.dimension_info().axes == ((y, x, DURATION),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 0]),
        y: pytest.approx([1, 1, 1.5, 2]),
        DURATION: pytest.approx([0.1, 0.1, 0.1, 0.1]),
    }
    assert dim.gap == ints("1011")


def test_rect_region_symmetricdifference() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) ^ Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 2, 0, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5]),
    }
    assert dim.gap == ints("10111110")


def test_circle_region() -> None:
    inst = Line(y, 1, 3, 3) * Line(x, 0, 2, 3) & Circle(x, y, 1, 2, 1)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([1, 0, 1, 2, 1]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, -0.5, 0.5, 1.5, 0.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 0.5, 1.5, 2.5, 1.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.gap == ints("11001")


def test_circle_snaked_region() -> None:
    inst = Mask(
        Line(y, 1, 3, 3) * ~Line(x, 0, 2, 3),
        Circle(x, y, 1, 2, 1),
        check_path_changes=False,
    )
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([1, 2, 1, 0, 1]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, 2.5, 1.5, 0.5, 0.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 1.5, 0.5, -0.5, 1.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.gap == ints("11001")


def test_ellipse_region() -> None:
    inst = Line("y", 1, 3, 3) * Line("x", 0, 2, 3) & Ellipse(x, y, 1, 2, 2, 1, 45)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 2, 1, 2]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 2.5, 1.5, 2.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.gap == ints("1010010")


def test_polygon_region() -> None:
    x_verts = [0, 0.5, 4.0, 2.5]
    y_verts = [0, 3.5, 3.5, 0.5]
    inst = Line("y", 1, 3, 3) * Line("x", 0, 4, 5) & Polygon(x, y, x_verts, y_verts)
    assert inst.dimension_info().axes == ((y, x),)
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: pytest.approx([1, 2, 1, 2, 3, 1, 2, 3]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 2.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.gap == ints("10100100")


def test_xyz_stack() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    spec = Line(z, 0, 1, 2) * ~Line(y, 0, 2, 3) * ~Line(x, 0, 3, 4)
    info = spec.dimension_info()
    assert info.axes == (
        (z,),
        (y,),
        (x,),
    )
    assert info.shape == (2, 3, 4)
    dim = spec.frames()
    assert len(dim) == 24
    assert dim.lower == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: pytest.approx([-0.5, 0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5] * 3),
    }
    assert dim.upper == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: pytest.approx([0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5, -0.5] * 3),
    }
    assert dim.midpoints == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: ints("012332100123321001233210"),
    }
    assert dim.gap == ints("100010001000100010001000")
    # Check that it still works if you consume then start on a point that should
    # be False
    p = Path(spec.calculate())
    assert p.consume(4).gap == ints("1000")
    assert p.consume(4).gap == ints("1000")
    assert p.consume(5).gap == ints("10001")
    assert p.consume(2).gap == ints("00")
    assert p.consume().gap == ints("010001000")
    assert p.consume().gap == ints("")


def test_beam_selector() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    spec = 10 * ~Line.bounded(x, 11, 19, 1)
    info = spec.dimension_info()
    assert info.axes == ((DURATION,), (x,))
    assert info.shape == (10, 1)
    dim = spec.frames()
    assert len(dim) == 10
    assert dim.lower == {x: pytest.approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: pytest.approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: pytest.approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("1111111111")


def test_gap_repeat() -> None:
    # Check that no gap propogates to dim.gap for snaked axis
    spec: Spec[Any] = Repeat(10, gap=False) * ~Line.bounded(x, 11, 19, 1)
    dim = spec.frames()
    assert len(dim) == 10
    assert dim.lower == {x: pytest.approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: pytest.approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: pytest.approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("0000000000")


def test_gap_repeat_non_snake() -> None:
    # Check that no gap doesn't propogate to dim.gap for non-snaked axis
    spec: Spec[Any] = Repeat(3, gap=False) * Line.bounded(x, 11, 19, 1)
    dim = spec.frames()
    assert len(dim) == 3
    assert dim.lower == {x: pytest.approx([11, 11, 11])}
    assert dim.upper == {x: pytest.approx([19, 19, 19])}
    assert dim.midpoints == {x: pytest.approx([15, 15, 15])}
    assert dim.gap == ints("111")


def test_multiple_statics():
    part_1 = Static("y", 2) * Static("z", 3) * Line("x", 0, 10, 2)
    part_2 = Static("y", 4) * Static("z", 5) * Line("x", 0, 10, 2)
    spec = part_1.concat(part_2)
    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 2, "z": 3},
        {"x": 10.0, "y": 2, "z": 3},
        {"x": 0.0, "y": 4, "z": 5},
        {"x": 10.0, "y": 4, "z": 5},
    ]
    assert spec.frames().gap == ints("1010")


def test_multiple_statics_with_grid():
    grid = Line("y", 0, 10, 2) * Line("x", 0, 10, 2)
    part_1 = grid.zip(Static("a", 2)).zip(Static("b", 3))
    part_2 = grid.zip(Static("a", 4)).zip(Static("b", 5))
    spec = part_1.concat(part_2)
    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 0.0, "a": 2, "b": 3},
        {"x": 10.0, "y": 0.0, "a": 2, "b": 3},
        {"x": 0.0, "y": 10.0, "a": 2, "b": 3},
        {"x": 10.0, "y": 10.0, "a": 2, "b": 3},
        {"x": 0.0, "y": 0.0, "a": 4, "b": 5},
        {"x": 10.0, "y": 0.0, "a": 4, "b": 5},
        {"x": 0.0, "y": 10.0, "a": 4, "b": 5},
        {"x": 10.0, "y": 10.0, "a": 4, "b": 5},
    ]
    assert spec.frames().gap == ints("10101010")


@pytest.mark.parametrize(
    "spec,expected_shape,expected_axes",
    [
        (Line("x", 0.0, 1.0, 1), (1,), (("x",),)),
        (Line("x", 0.0, 1.0, 5), (5,), (("x",),)),
        (Spiral("x", "y", 0.0, 0.0, 1.0, 1.0, 5, 0.0), (5,), (("y", "x"),)),
        (Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2), (2, 2), (("x",), ("y",))),
        (Squash(Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2)), (4,), (("x", "y"),)),
        (Zip(Line("x", 0.0, 1.0, 2), Line("y", 0.0, 1.0, 2)), (2,), (("x", "y"),)),
        (Concat(Line("x", 0.0, 1.0, 2), Line("x", 0.0, 1.0, 2)), (4,), (("x",),)),
        (
            Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2) * Line("z", 0.0, 2.0, 2),
            (2, 2, 2),
            (("x",), ("y",), ("z",)),
        ),
        (
            Zip(Line("x", 0.0, 1.0, 2), Line("y", 0.0, 1.0, 2))
            * Line("z", 0.0, 2.0, 2),
            (2, 2),
            (("x", "y"), ("z",)),
        ),
        (
            Concat(Line("x", 0.0, 1.0, 2), Line("x", 0.0, 1.0, 2))
            * Line("z", 0.0, 2.0, 2),
            (4, 2),
            (("x",), ("z",)),
        ),
    ],
)
def test_dimension_info(
    spec: Spec, expected_shape: Tuple[int, ...], expected_axes: Tuple[Tuple[str, ...]]
):
    dimension_info = spec.dimension_info()
    assert expected_shape == dimension_info.shape
    assert expected_axes == dimension_info.axes
