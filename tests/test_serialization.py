import pytest
from apischema.validation.errors import ValidationError

from scanspec.core import to_gql_input
from scanspec.regions import Circle, Rectangle, UnionOf
from scanspec.specs import Line, Mask, Spec, Spiral


def test_line_serializes() -> None:
    ob = Line("x", 0, 1, 4)
    serialized = {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4}
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_bad_sgql_serialization() -> None:
    with pytest.raises(ValueError) as ctx:
        to_gql_input(Line("x", 0, 1, 4))
    assert str(ctx.value) == "Cannot format Line(axis='x', start=0, stop=1, num=4)"


def test_masked_circle_serializes() -> None:
    ob = Mask(Line("x", 0, 1, 4), Circle("x", "y", x_middle=0, y_middle=1, radius=4))
    serialized = {
        "type": "Mask",
        "spec": {"type": "Line", "axis": "x", "start": 0, "stop": 1, "num": 4},
        "region": {
            "Circle": {
                "x_axis": "x",
                "y_axis": "y",
                "x_middle": 0,
                "y_middle": 1,
                "radius": 4,
            }
        },
        "check_path_changes": True,
    }
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_product_lines_serializes() -> None:
    ob = Line("z", 4, 5, 6) * Line("y", 2, 3, 5) * Line("x", 0, 1, 4)
    serialized = {
        "type": "Product",
        "outer": {
            "type": "Product",
            "outer": {
                "type": "Line",
                "axis": "z",
                "start": 4.0,
                "stop": 5.0,
                "num": 6,
            },
            "inner": {"type": "Line", "axis": "y", "start": 2.0, "stop": 3.0, "num": 5},
        },
        "inner": {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
    }

    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_complex_nested_serializes() -> None:
    ob = Mask(
        Spiral.spaced("x", "y", 0, 0, 10, 3),
        UnionOf(
            Circle("x", "y", x_middle=0, y_middle=1, radius=4),
            Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30),
        ),
    )
    serialized = {
        "Mask": {
            "spec": {
                "Spiral": {
                    "x_axis": "x",
                    "y_axis": "y",
                    "x_start": 0,
                    "y_start": 0,
                    "x_range": 20,
                    "y_range": 20,
                    "num": 34,
                    "rotate": 0.0,
                }
            },
            "region": {
                "UnionOf": {
                    "left": {
                        "Circle": {
                            "x_axis": "x",
                            "y_axis": "y",
                            "x_middle": 0,
                            "y_middle": 1,
                            "radius": 4,
                        }
                    },
                    "right": {
                        "Rectangle": {
                            "x_axis": "x",
                            "y_axis": "y",
                            "x_min": 0,
                            "y_min": 1.1,
                            "x_max": 1.5,
                            "y_max": 2.1,
                            "angle": 30,
                        }
                    },
                }
            },
            "check_path_changes": True,
        }
    }
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_extra_arg_fails() -> None:
    with pytest.raises(ValidationError):
        serialized = {
            "Line": {"axis": "x", "start": 0.0, "stop": 1.0, "num": 4, "foo": "bar"}
        }
        Spec.deserialize(serialized)
