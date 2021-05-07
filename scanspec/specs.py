from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypeVar

import numpy as np
from apischema import schema
from typing_extensions import Annotated

from .core import (
    Dimension,
    Midpoints,
    Path,
    Serializable,
    alternative_constructor,
    if_instance_do,
    squash_dimensions,
)
from .regions import Region, get_mask

T = TypeVar("T")

__all__ = [
    "Spec",
    "Product",
    "Zip",
    "Mask",
    "Snake",
    "Concat",
    "Squash",
    "Line",
    "Static",
    "Spiral",
    "TIME",
    "REPEAT",
    "fly",
    "step",
    "repeat",
]


@dataclass
class Spec(Serializable):
    """Definition: A spec is a serializable representation of the type, parameters
    and axis names required to produce one or more dimensions.

    Description: Abstract baseclass for the specification of a scan. Supports operators:

    - ``*``: Outer `Product` of two Specs, nesting the second within the first
    - ``+``: `Zip` two Specs together, iterating in tandem
    - ``&``: `Mask` the Spec with a `Region`, excluding midpoints outside of it
    - ``~``: `Snake` the Spec, reversing every other iteration of it
    """

    def axes(self) -> List:
        """Return the list of axes that are present in the scan, from
        slowest moving to fastest moving"""
        raise NotImplementedError(self)

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        """Implemented by subclasses to produce the `Dimension` list that
        contribute to midpoints, from slowest moving to fastest moving"""
        raise NotImplementedError(self)

    def path(self) -> Path:
        """Return a `Path` through the scan that can be consumed in chunks
        to give midpoints and bounds"""
        path = Path(self.create_dimensions())
        return path

    def midpoints(self) -> Midpoints:
        """Return `Midpoints` that can be iterated point by point"""
        mp = Midpoints(self.create_dimensions(bounds=False))
        return mp

    def __mul__(self, other) -> "Product":
        return if_instance_do(other, Spec, lambda o: Product(self, o))

    def __add__(self, other) -> "Zip":
        return if_instance_do(other, Spec, lambda o: Zip(self, o))

    def __and__(self, other) -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self, o))

    def __invert__(self) -> "Snake":
        return Snake(self)


@dataclass
class Product(Spec):
    """Outer product of two Specs, nesting inner within outer. This means that
    inner will run in its entirety at each point in outer.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("y", 1, 2, 3) * Line("x", 3, 4, 12)
    """

    outer: Spec = field(metadata=schema(description="Will be executed once"))
    inner: Spec = field(
        metadata=schema(description="Will be executed len(outer) times")
    )

    def axes(self) -> List:
        return self.outer.axes() + self.inner.axes()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_outer = self.outer.create_dimensions(bounds=False, nested=nested)
        dims_inner = self.inner.create_dimensions(bounds, nested=True)
        return dims_outer + dims_inner


@dataclass
class Zip(Spec):
    """Run two Specs in parallel, merging their midpoints together. Typically
    formed using the ``+`` operator.

    Dimensions are merged by:

    - If right creates a single Dimension of size 1, expand it to the size of
      the fastest Dimension created by left
    - Merge individual dimensions together from fastest to slowest

    This means that Zipping a Spec producing Dimensions [l2, l1] with a
    Spec producing Dimension [r1] will assert len(l1)==len(r1), and produce
    Dimensions [l2, l1+r1].

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("z", 1, 2, 3) * Line("y", 3, 4, 5) + Line("x", 4, 5, 5)
    """

    left: Spec = field(
        metadata=schema(
            description="The left-hand Spec to Zip, will appear earlier in axes"
        )
    )
    right: Spec = field(
        metadata=schema(
            description="The right-hand Spec to Zip, will appear later in axes"
        )
    )

    def axes(self) -> List:
        return self.left.axes() + self.right.axes()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds, nested)
        dims_right = self.right.create_dimensions(bounds, nested)
        assert len(dims_left) >= len(
            dims_right
        ), f"Zip requires len({self.left}) >= len({self.right})"

        # Pad and expand the right to be the same size as left
        # Special case, if only one dim with size 1, expand to the right size
        if len(dims_right) == 1 and len(dims_right[0]) == 1:
            repeated = dims_right[0].repeat(len(dims_left[-1]))
            repeated.snake = dims_left[-1].snake
            dims_right = [repeated]

        # Left pad dims_right with Nones so they are the same size
        npad = len(dims_left) - len(dims_right)
        padded_right: List[Optional[Dimension]] = [None] * npad
        padded_right += dims_right

        # Work through, zipping them together one by one
        dimensions = []
        for dim_left, dim_right in zip(dims_left, padded_right):
            if dim_right is None:
                dim = dim_left
            else:
                dim = dim_left.zip(dim_right)
            assert isinstance(
                dim, Dimension
            ), f"Padding went wrong {dims_left} {padded_right}"
            dimensions.append(dim)
        return dimensions


@dataclass
class Mask(Spec):
    """Restrict the given Spec to only the midpoints that fall inside of the given
    Region.

    Typically created with the ``&`` operator. It also pushes down the
    ``& | ^ -`` operators to its `Region` to avoid the need for brackets on
    combinations of Regions.

    If a Region spans multiple Dimensions, these Dimensions will be squashed
    together.

    See Also:
        `why-squash-can-change-path`

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        spec = Line("y", 1, 3, 3) * Line("x", 3, 5, 5) & Circle("x", "y", 4, 2, 1.2)
    """

    spec: Spec = field(
        metadata=schema(description="The Spec containing the source midpoints")
    )
    region: Region = field(
        metadata=schema(description="The Region that midpoints will be inside")
    )
    check_path_changes: bool = field(
        default=True,
        metadata=schema(
            description="If True path through scan will not be modified by squash"
        ),
    )

    def axes(self) -> List:
        return self.spec.axes()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds, nested)
        for axis_set in self.region.axis_sets():
            # Find the start and end index of any dimensions containing these axes
            matches = [i for i, d in enumerate(dims) if set(d.axes()) & axis_set]
            assert matches, f"No Specs match axes {list(axis_set)}"
            si, ei = matches[0], matches[-1]
            if si != ei:
                # The axis_set spans multiple Dimensions, squash them together
                # If the spec to be squashed is nested (inside the Mask or outside)
                # then check the path changes if requested
                check_path_changes = (nested or si) and self.check_path_changes
                squashed = squash_dimensions(dims[si : ei + 1], check_path_changes)
                dims = dims[:si] + [squashed] + dims[ei + 1 :]
        # Generate masks from the midpoints showing what's inside
        masked_dims = [dim.mask(get_mask(self.region, dim.midpoints)) for dim in dims]
        return masked_dims

    # *+ bind more tightly than &|^ so without these overrides we
    # would need to add brackets to all combinations of Regions
    def __or__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region | o))

    def __and__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region & o))

    def __xor__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region ^ o))

    # This is here for completeness, tends not to be called as - binds
    # tighter than &
    def __sub__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region - o))


@dataclass
class Snake(Spec):
    """Run the Spec in reverse on every other iteration when nested inside
    another Spec. Typically created with the ``~`` operator.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("y", 1, 3, 3) * ~Line("x", 3, 5, 5)
    """

    spec: Spec = field(
        metadata=schema(description="The Spec to run in reverse every other iteration")
    )

    def axes(self) -> List:
        return self.spec.axes()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds, nested)
        for dim in dims:
            dim.snake = True
        return dims


@dataclass
class Concat(Spec):
    """Concatenate two Specs together, running one after the other. Each Dimension
    of left and right must contain the same axes.

    .. example_spec::

        from scanspec.specs import Line, Concat

        spec = Concat(Line("x", 1, 3, 3), Line("x", 4, 5, 5))
    """

    left: Spec = field(
        metadata=schema(
            description="The left-hand Spec to Zip, midpoints will appear earlier"
        )
    )
    right: Spec = field(
        metadata=schema(
            description="The right-hand Spec to Zip, midpoints will appear later"
        )
    )

    def axes(self) -> List:
        left_axes, right_axes = self.left.axes(), self.right.axes()
        assert left_axes == right_axes, f"axes {left_axes} != {right_axes}"
        return left_axes

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds, nested)
        dims_right = self.right.create_dimensions(bounds, nested)
        assert len(dims_right) == len(
            dims_left
        ), f"Specs {self.left} and {self.right} don't have same number of dimensions"
        dimensions = [dl.concat(dr) for dl, dr in zip(dims_left, dims_right)]
        return dimensions


@dataclass
class Squash(Spec):
    """Squash the Dimensions together of the scan (but not the midpoints) into one
    linear stack.

    See Also:
        `why-squash-can-change-path`

    .. example_spec::

        from scanspec.specs import Line, Squash

        spec = Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4))
    """

    spec: Spec = field(
        metadata=schema(description="The Spec to squash the dimensions of")
    )
    check_path_changes: bool = field(
        default=True,
        metadata=schema(
            description="If True path through scan will not be modified by squash"
        ),
    )

    def axes(self) -> List:
        return self.spec.axes()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        # TODO: if we squash we explode the size, can we avoid this?
        dims = self.spec.create_dimensions(bounds, nested)
        dim = squash_dimensions(dims, nested and self.check_path_changes)
        return [dim]


def _dimensions_from_indexes(
    func: Callable[[np.ndarray], Dict[str, np.ndarray]],
    axes: List,
    num: int,
    bounds: bool,
) -> List[Dimension]:
    # Calc num midpoints (fences) from 0.5 .. num - 0.5
    midpoints_calc = func(np.linspace(0.5, num - 0.5, num))
    midpoints = {a: midpoints_calc[a] for a in axes}
    if bounds:
        # Calc num + 1 bounds (posts) from 0 .. num
        bounds_calc = func(np.linspace(0, num, num + 1))
        lower = {a: bounds_calc[a][:-1] for a in axes}
        upper = {a: bounds_calc[a][1:] for a in axes}
        dimension = Dimension(midpoints, lower, upper)
    else:
        dimension = Dimension(midpoints)
    return [dimension]


@dataclass
class Line(Spec):
    """Linearly spaced points in the given axis, with first and last points
    centred on start and stop.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("x", 1, 2, 5)
    """

    axis: str = field(metadata=schema(description="An identifier for what to move"))
    start: float = field(
        metadata=schema(description="Centre point of the first point of the line")
    )
    stop: float = field(
        metadata=schema(description="Centre point of the last point of the line")
    )
    num: int = field(metadata=schema(min=1, description="Number of points to produce"))

    def axes(self) -> List:
        return [self.axis]

    def _line_from_indexes(self, indexes: np.ndarray) -> Dict[str, np.ndarray]:
        if self.num == 1:
            # Only one point, stop-start gives length of one point
            step = self.stop - self.start
        else:
            # Multiple points, stop-start gives length of num-1 points
            step = (self.stop - self.start) / (self.num - 1)
        # self.start is the first centre point, but we need the lower bound
        # of the first point as this is where the index array starts
        first = self.start - step / 2
        return {self.axis: indexes * step + first}

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._line_from_indexes, self.axes(), self.num, bounds
        )

    @alternative_constructor
    def bounded(
        axis: Annotated[str, schema(description="An identifier for what to move")],
        lower: Annotated[
            float, schema(description="Lower bound of the first point of the line")
        ],
        upper: Annotated[
            float, schema(description="Upper bound of the last point of the line")
        ],
        num: Annotated[int, schema(min=1, description="Number of points to produce")],
    ) -> "Line":
        """Specify a Line by extreme bounds instead of centre points.

        .. example_spec::

            from scanspec.specs import Line

            spec = Line.bounded("x", 1, 2, 5)
        """
        half_step = (upper - lower) / num / 2
        start = lower + half_step
        if num == 1:
            # One point, stop will only be used for step size
            stop = upper + half_step
        else:
            # Many points, stop will be produced
            stop = upper - half_step
        return Line(axis, start, stop, num)


@dataclass
class Static(Spec):
    """A static point, repeated "num" times, with "axis" at "value". Can
    be used to set axis=value at every point in a scan.

    .. example_spec::

        from scanspec.specs import Line, Static

        spec = Line("y", 1, 2, 3) + Static("x", 3)
    """

    axis: str = field(metadata=schema(description="An identifier for what to move"))
    value: float = field(metadata=schema(description="The value at each point"))
    num: int = field(
        default=1,
        metadata=schema(min=1, description="How many times to repeat this point"),
    )

    def axes(self) -> List:
        return [self.axis]

    def _repeats_from_indexes(self, indexes: np.ndarray) -> Dict[str, np.ndarray]:
        return {self.axis: np.full(len(indexes), self.value)}

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._repeats_from_indexes, self.axes(), self.num, bounds
        )


@dataclass
class Spiral(Spec):
    """Archimedean spiral of "x_axis_name" and "y_axis_name", starting at centre point
    ("x_start", "y_start") with angle "rotate". Produces "num" points
    in a spiral spanning width of "x_range" and height of "y_range"

    .. example_spec::

        from scanspec.specs import Spiral

        spec = Spiral("x", "y", 1, 5, 10, 50, 30)
    """

    x_axis_name: str = field(
        metadata=schema(description="An identifier for what to move for x")
    )
    y_axis_name: str = field(
        metadata=schema(description="An identifier for what to move for y")
    )
    # TODO: do we like these names?
    x_start: float = field(metadata=schema(description="x centre of the spiral"))
    y_start: float = field(metadata=schema(description="y centre of the spiral"))
    x_range: float = field(metadata=schema(description="x width of the spiral"))
    y_range: float = field(metadata=schema(description="y width of the spiral"))
    num: int = field(metadata=schema(description="Number of points in the spiral"))
    rotate: float = field(
        default=0.0,
        metadata=schema(description="How much to rotate the angle of the spiral"),
    )

    def axes(self) -> List:
        # TODO: reversed from __init__ args, a good idea?
        return [self.y_axis_name, self.x_axis_name]

    def _spiral_from_indexes(self, indexes: np.ndarray) -> Dict[str, np.ndarray]:
        # simplest spiral equation: r = phi
        # we want point spacing across area to be the same as between rings
        # so: sqrt(area / num) = ring_spacing
        # so: sqrt(pi * phi^2 / num) = 2 * pi
        # so: phi = sqrt(4 * pi * num)
        phi = np.sqrt(4 * np.pi * indexes)
        # indexes are 0..num inclusive, and diameter is 2x biggest phi
        diameter = 2 * np.sqrt(4 * np.pi * self.num)
        # scale so that the spiral is strictly smaller than the range
        x_scale = self.x_range / diameter
        y_scale = self.y_range / diameter
        return {
            self.y_axis_name: self.y_start + y_scale * phi * np.cos(phi + self.rotate),
            self.x_axis_name: self.x_start + x_scale * phi * np.sin(phi + self.rotate),
        }

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._spiral_from_indexes, self.axes(), self.num, bounds
        )

    @alternative_constructor
    def spaced(
        x_axis_name: Annotated[
            str, schema(description="An identifier for what to move for x")
        ],
        y_axis_name: Annotated[
            str, schema(description="An identifier for what to move for y")
        ],
        x_start: Annotated[float, schema(description="x centre of the spiral")],
        y_start: Annotated[float, schema(description="y centre of the spiral")],
        radius: Annotated[float, schema(description="radius of the spiral")],
        dr: Annotated[float, schema(description="difference between each ring")],
        rotate: Annotated[
            float,
            schema(
                default=0.0, description="How much to rotate the angle of the spiral"
            ),
        ] = 0.0,
    ) -> "Spiral":
        """Specify a Spiral equally spaced in "x_axis_name" and "y_axis_name" by specifying
        the "radius" and difference between each ring of the spiral "dr"

        .. example_spec::

            from scanspec.specs import Spiral

            spec = Spiral.spaced("x", "y", 0, 0, 10, 3)
        """
        # phi = sqrt(4 * pi * num)
        # and: n_rings = phi / (2 * pi)
        # so: n_rings * 2 * pi = sqrt(4 * pi * num)
        # so: num = n_rings^2 * pi
        n_rings = radius / dr
        num = int(n_rings ** 2 * np.pi)
        return Spiral(
            x_axis_name,
            y_axis_name,
            x_start,
            y_start,
            radius * 2,
            radius * 2,
            num,
            rotate,
        )


#: Can be used as a special key to indicate how long each point should be
TIME = "TIME"


#: Can be used as a special key to indicate repeats of a whole spec
REPEAT = "REPEAT"


def fly(spec: Spec, duration: float) -> Spec:
    """Flyscan, zipping TIME=duration for every frame

    Args:
        spec: The source `Spec` to continuously move
        duration: How long to spend at each point in the spec

    .. example_spec::

        from scanspec.specs import Line, fly

        spec = fly(Line("x", 1, 2, 3), 0.1)
    """
    return spec + Static(TIME, duration)


def step(spec: Spec, duration: float, num: int = 1):
    """Step scan, adding num x TIME=duration as an inner dimension for
    every midpoint

    Args:
        spec: The source `Spec` with midpoints to move to and stop
        duration: The duration of each scan point
        num: Number of points to produce with given duration at each of point
            in the spec

    .. example_spec::

        from scanspec.specs import Line, step

        spec = step(Line("x", 1, 2, 3), 0.1)
    """
    return spec * Static(TIME, duration, num)


def repeat(spec: Spec, num: int, blend=False):
    """Repeat spec num times

    Args:
        spec: The source `Spec` that will be iterated
        num: The number of times to repeat it
        blend: If True and the slowest dimension of spec is snaked then
            the end and start of consecutive iterations of Spec will be
            blended together, leaving no gap
    """
    if blend:
        return Static(REPEAT, num, num) * spec
    else:
        return Line(REPEAT, 1, num, num) * spec
