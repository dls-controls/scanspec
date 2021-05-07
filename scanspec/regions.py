from dataclasses import dataclass, field, is_dataclass
from typing import Iterator, List, Set

import numpy as np
from apischema import schema

from .core import AxesPoints, Serializable, if_instance_do

__all__ = [
    "Region",
    "get_mask",
    "CombinationOf",
    "UnionOf",
    "IntersectionOf",
    "DifferenceOf",
    "SymmetricDifferenceOf",
    "Range",
    "Rectangle",
    "Circle",
    "find_regions",
]


@dataclass
class Region(Serializable):
    """Abstract baseclass for a Region that can `Mask` a `Spec`. Supports operators:

    - ``|``: `UnionOf` two Regions, midpoints present in either
    - ``&``: `IntersectionOf` two Regions, points present in both
    - ``-``: `DifferenceOf` two Regions, points present in first not second
    - ``^``: `SymmetricDifferenceOf` two Regions, points present in one not both
    """

    def axis_sets(self) -> List[Set[str]]:
        """Implemented by subclasses to produce the non-overlapping sets of axes
        this region spans"""
        raise NotImplementedError(self)

    def mask(self, points: AxesPoints) -> np.ndarray:
        """Implemented by subclasses to produce a mask of which points are in
        the region"""
        raise NotImplementedError(self)

    def __or__(self, other) -> "UnionOf":
        return if_instance_do(other, Region, lambda o: UnionOf(self, o))

    def __and__(self, other) -> "IntersectionOf":
        return if_instance_do(other, Region, lambda o: IntersectionOf(self, o))

    def __sub__(self, other) -> "DifferenceOf":
        return if_instance_do(other, Region, lambda o: DifferenceOf(self, o))

    def __xor__(self, other) -> "SymmetricDifferenceOf":
        return if_instance_do(other, Region, lambda o: SymmetricDifferenceOf(self, o))


def get_mask(region: Region, points: AxesPoints) -> np.ndarray:
    """If there is an overlap of axes of region and frames return a
    mask of the frames in the region, otherwise return all ones
    """
    axes = set(points)
    needs_mask = any(ks & axes for ks in region.axis_sets())
    if needs_mask:
        return region.mask(points)
    else:
        return np.ones(len(list(points.values())[0]))


def _merge_axis_sets(axis_sets: List[Set[str]]) -> Iterator[Set[str]]:
    # Take overlapping axis sets and merge any that overlap into each
    # other
    for ks in axis_sets:  # ks = key_sets - left over from a previous naming standard
        axis_set = ks.copy()
        # Empty matching sets into this axis_set
        for ks in axis_sets:
            if ks & axis_set:
                while ks:
                    axis_set.add(ks.pop())
        # It might be emptied already, only yield if it isn't
        if axis_set:
            yield axis_set


@dataclass
class CombinationOf(Region):
    """Abstract baseclass for a combination of two regions, left and right"""

    left: Region = field(metadata=schema(description="The left-hand Region to combine"))
    right: Region = field(
        metadata=schema(description="The right-hand Region to combine")
    )

    def axis_sets(self) -> List[Set[str]]:
        axis_sets = list(
            _merge_axis_sets(self.left.axis_sets() + self.right.axis_sets())
        )
        return axis_sets


# Naming so we don't clash with typing.Union
@dataclass
class UnionOf(CombinationOf):
    """A point is in UnionOf(a, b) if it is in either a or b. Typically
    created with the ``|`` operator

    >>> r = Range("x", 0.5, 2.5) | Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True,  True, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) | get_mask(self.right, points)
        return mask


@dataclass
class IntersectionOf(CombinationOf):
    """A point is in IntersectionOf(a, b) if it is in both a and b. Typically
    created with the ``&`` operator

    >>> r = Range("x", 0.5, 2.5) & Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False, False,  True, False, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) & get_mask(self.right, points)
        return mask


@dataclass
class DifferenceOf(CombinationOf):
    """A point is in DifferenceOf(a, b) if it is in a and not in b. Typically
    created with the ``-`` operator

    >>> r = Range("x", 0.5, 2.5) - Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False, False, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        left_mask = get_mask(self.left, points)
        # Return the xor restricted to the left region
        mask = left_mask ^ get_mask(self.right, points) & left_mask
        return mask


@dataclass
class SymmetricDifferenceOf(CombinationOf):
    """A point is in SymmetricDifferenceOf(a, b) if it is in either a or b,
    but not both. Typically created with the ``^`` operator

    >>> r = Range("x", 0.5, 2.5) ^ Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False,  True, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) ^ get_mask(self.right, points)
        return mask


@dataclass
class Range(Region):
    """Mask contains points of key >= min and <= max

    >>> r = Range("x", 1, 2)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True, False, False])
    """

    axis: str = field(
        metadata=schema(description="The name matching the axis to mask in spec")
    )
    min: float = field(
        metadata=schema(description="The minimum inclusive value in the region")
    )
    max: float = field(
        metadata=schema(description="The minimum inclusive value in the region")
    )

    def axis_sets(self) -> List[Set[str]]:
        return [{self.axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        v = points[self.axis]
        mask = np.bitwise_and(v >= self.min, v <= self.max)
        return mask


@dataclass
class Rectangle(Region):
    """Mask contains points of axis within a rotated xy rectangle

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Rectangle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30)
    """

    x_axis_name: str = field(
        metadata=schema(description="The name matching the x axis of the spec")
    )
    y_axis_name: str = field(
        metadata=schema(description="The name matching the y axis of the spec")
    )
    x_min: float = field(
        metadata=schema(description="Minimum inclusive x value in the region")
    )
    y_min: float = field(
        metadata=schema(description="Minimum inclusive y value in the region")
    )
    x_max: float = field(
        metadata=schema(description="Maximum inclusive x value in the region")
    )
    y_max: float = field(
        metadata=schema(description="Maximum inclusive y value in the region")
    )
    angle: float = field(
        default=0.0,
        metadata=schema(description="Clockwise rotation angle of the rectangle"),
    )

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis_name, self.y_axis_name}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        x = points[self.x_axis_name] - self.x_min
        y = points[self.y_axis_name] - self.y_min
        if self.angle != 0:
            # Rotate src points by -angle
            phi = np.radians(-self.angle)
            rx = x * np.cos(phi) - y * np.sin(phi)
            ry = x * np.sin(phi) + y * np.cos(phi)
            x = rx
            y = ry
        mask_x = np.bitwise_and(x >= 0, x <= (self.x_max - self.x_min))
        mask_y = np.bitwise_and(y >= 0, y <= (self.y_max - self.y_min))
        return mask_x & mask_y


@dataclass
class Circle(Region):
    """Mask contains points of axis within an xy circle of given radius

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Circle("x", "y", 1, 2, 0.9)
    """

    x_axis_name: str = field(
        metadata=schema(description="The name matching the x axis of the spec")
    )
    y_axis_name: str = field(
        metadata=schema(description="The name matching the x axis of the spec")
    )
    x_centre: float = field(
        metadata=schema(description="Minimum inclusive x value in the region")
    )
    y_centre: float = field(
        metadata=schema(description="Minimum inclusive y value in the region")
    )
    radius: float = field(metadata=schema(description="Radius of the circle"))

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis_name, self.y_axis_name}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        x = points[self.x_axis_name] - self.x_centre
        y = points[self.y_axis_name] - self.y_centre
        mask = x * x + y * y <= (self.radius * self.radius)
        return mask


def find_regions(obj) -> Iterator[Region]:
    """Recursively iterate over obj and its children, yielding any Region
    instances found"""
    if is_dataclass(obj):
        if isinstance(obj, Region):
            yield obj
        for name in obj.__dict__.keys():
            yield from find_regions(getattr(obj, name))
