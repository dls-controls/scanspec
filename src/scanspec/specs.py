from __future__ import annotations

from dataclasses import asdict
from functools import reduce
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Tuple, Type
from warnings import warn

import numpy as np
from pydantic import Field, parse_obj_as
from pydantic.dataclasses import dataclass

from .core import (
    Axis,
    DimensionInfo,
    Frames,
    Midpoints,
    Path,
    SnakedFrames,
    StrictConfig,
    discriminated_union_of_subclasses,
    gap_between_frames,
    if_instance_do,
    squash_frames,
)
from .regions import Region, get_mask

__all__ = [
    "DURATION",
    "Spec",
    "Product",
    "Repeat",
    "Zip",
    "Mask",
    "Snake",
    "Concat",
    "Squash",
    "Line",
    "Static",
    "Spiral",
    "fly",
    "step",
]


#: Can be used as a special key to indicate how long each point should be
DURATION = "DURATION"


@discriminated_union_of_subclasses(config=StrictConfig)
class Spec(Generic[Axis]):
    """A serializable representation of the type and parameters of a scan.

    Abstract baseclass for the specification of a scan. Supports operators:

    - ``*``: Outer `Product` of two Specs, nesting the second within the first.
      If the first operand is an integer, wrap it in a `Repeat`
    - ``&``: `Mask` the Spec with a `Region`, excluding midpoints outside of it
    - ``~``: `Snake` the Spec, reversing every other iteration of it
    """

    def axes(self) -> List[Axis]:
        """Return the list of axes that are present in the scan.

        Ordered from slowest moving to fastest moving.
        """
        warn(
            "axes() is deprecated, call dimension_info()",
            DeprecationWarning,
            stacklevel=2,
        )
        return reduce(
            lambda a, b: a + list(b), self.dimension_info().axes, initial=list()
        )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        """Produce a stack of nested `Frames` that form the scan.

        Ordered from slowest moving to fastest moving.
        """
        raise NotImplementedError(self)

    def dimension_info(self) -> DimensionInfo:
        """Returns the list of axes in each dimension of the scan,
        paired with the information on how large each dimension of the scan is,
        and whether the dimension is snaked in the dimension outside it.

        Deprecates shape() as does not need to call calculate()
        Deprecates axes() as has the per-dimension information
        """
        raise NotImplementedError(self)

    def frames(self) -> Frames[Axis]:
        """Expand all the scan `Frames` and return them."""
        return Path(self.calculate()).consume()

    def midpoints(self) -> Midpoints[Axis]:
        """Return `Midpoints` that can be iterated point by point."""
        return Midpoints(self.calculate(bounds=False))

    def shape(self) -> Tuple[int, ...]:
        """Return the final, simplified shape of the scan."""
        warn(
            "shape() is deprecated, call dimension_info()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dimension_info().shape

    def __rmul__(self, other) -> Product[Axis]:
        return if_instance_do(other, int, lambda o: Product(Repeat(o), self))

    def __mul__(self, other) -> Product[Axis]:
        return if_instance_do(other, Spec, lambda o: Product(self, o))

    def __and__(self, other) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self, o))

    def __invert__(self) -> Snake[Axis]:
        return Snake(self)

    def zip(self, other: Spec) -> Zip[Axis]:
        """`Zip` the Spec with another, iterating in tandem."""
        return Zip(self, other)

    def concat(self, other: Spec) -> Concat[Axis]:
        """`Concat` the Spec with another, iterating one after the other."""
        return Concat(self, other)

    def serialize(self) -> Mapping[str, Any]:
        """Serialize the spec to a dictionary."""
        return asdict(self)

    @classmethod
    def deserialize(cls, obj):
        """Deserialize the spec from a dictionary."""
        return parse_obj_as(cls, obj)


@dataclass(config=StrictConfig)
class Product(Spec[Axis]):
    """Outer product of two Specs, nesting inner within outer.

    This means that inner will run in its entirety at each point in outer.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("y", 1, 2, 3) * Line("x", 3, 4, 12)
    """

    outer: Spec[Axis] = Field(description="Will be executed once")
    inner: Spec[Axis] = Field(description="Will be executed len(outer) times")

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        frames_outer = self.outer.calculate(bounds=False, nested=nested)
        frames_inner = self.inner.calculate(bounds, nested=True)
        return frames_outer + frames_inner

    def dimension_info(self) -> DimensionInfo:
        outer_info = self.outer.dimension_info()
        inner_info = self.inner.dimension_info()
        return DimensionInfo(
            axes=outer_info.axes + inner_info.axes,
            shape=outer_info.shape + inner_info.shape,
            snaked=outer_info.snaked + inner_info.snaked,
        )


@dataclass(config=StrictConfig)
class Repeat(Spec[Axis]):
    """Repeat an empty frame num times.

    Can be used on the outside of a scan to repeat the same scan many times.

    .. example_spec::

        from scanspec.specs import Line

        spec = 2 * ~Line.bounded("x", 3, 4, 1)

    If you want snaked axes to have no gap between iterations you can do:

    .. example_spec::

        from scanspec.specs import Line, Repeat

        spec = Repeat(2, gap=False) * ~Line.bounded("x", 3, 4, 1)

    .. note:: There is no turnaround arrow at x=4
    """

    num: int = Field(min=1, description="Number of frames to produce")
    gap: bool = Field(
        description="If False and the slowest of the stack of Frames is snaked "
        "then the end and start of consecutive iterations of Spec will have no gap",
        default=True,
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        return [Frames({}, gap=np.full(self.num, self.gap))]

    def dimension_info(self) -> DimensionInfo:
        return DimensionInfo(axes=((DURATION,),), shape=(self.num,))


@dataclass(config=StrictConfig)
class Zip(Spec[Axis]):
    """Run two Specs in parallel, merging their midpoints together.

    Typically formed using `Spec.zip`.

    Stacks of Frames are merged by:

    - If right creates a stack of a single Frames object of size 1, expand it to
      the size of the fastest Frames object created by left
    - Merge individual Frames objects together from fastest to slowest

    This means that Zipping a Spec producing stack [l2, l1] with a Spec
    producing stack [r1] will assert len(l1)==len(r1), and produce
    stack [l2, l1.zip(r1)].

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("z", 1, 2, 3) * Line("y", 3, 4, 5).zip(Line("x", 4, 5, 5))
    """

    left: Spec[Axis] = Field(
        description="The left-hand Spec to Zip, will appear earlier in axes"
    )
    right: Spec[Axis] = Field(
        description="The right-hand Spec to Zip, will appear later in axes"
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        frames_left = self.left.calculate(bounds, nested)
        frames_right = self.right.calculate(bounds, nested)
        assert len(frames_left) >= len(
            frames_right
        ), f"Zip requires len({self.left}) >= len({self.right})"

        # Pad and expand the right to be the same size as left. Special case, if
        # only one Frames object with size 1, expand to the right size
        if len(frames_right) == 1 and len(frames_right[0]) == 1:
            # Take the 0th element N times to make a repeated Frames object
            indices = np.zeros(len(frames_left[-1]), dtype=np.int8)
            repeated = frames_right[0].extract(indices)
            if isinstance(frames_left[-1], SnakedFrames):
                repeated = SnakedFrames.from_frames(repeated)
            frames_right = [repeated]

        # Left pad frames_right with Nones so they are the same size
        npad = len(frames_left) - len(frames_right)
        padded_right: List[Optional[Frames[Axis]]] = [None] * npad
        # Mypy doesn't like this because lists are invariant:
        # https://github.com/python/mypy/issues/4244
        padded_right += frames_right  # type: ignore

        # Work through, zipping them together one by one
        frames = []
        for left, right in zip(frames_left, padded_right):
            if right is None:
                combined = left
            else:
                combined = left.zip(right)
            assert isinstance(
                combined, Frames
            ), f"Padding went wrong {frames_left} {padded_right}"
            frames.append(combined)
        return frames

    def dimension_info(self) -> DimensionInfo:
        left_info = self.left.dimension_info()
        right_info = self.right.dimension_info()
        left = left_info.axes
        padded_right = ((None,),) * (
            len(left_info.axes) - len(right_info.axes)
        ) + right_info.axes
        axes = tuple(
            left[i] if padded_right[i] == (None,) else left[i] + padded_right[i]
            for i in range(len(left_info.axes))
        )

        return DimensionInfo(
            axes=axes,
            shape=left_info.shape,
            snaked=left_info.snaked,  # Non-matching Snake axes cannot be Zipped
        )


@dataclass(config=StrictConfig)
class Mask(Spec[Axis]):
    """Restrict Spec to only midpoints that fall inside the given Region.

    Typically created with the ``&`` operator. It also pushes down the
    ``& | ^ -`` operators to its `Region` to avoid the need for brackets on
    combinations of Regions.

    If a Region spans multiple Frames objects, they will be squashed together.

    .. example_spec::

        from scanspec.regions import Circle
        from scanspec.specs import Line

        spec = Line("y", 1, 3, 3) * Line("x", 3, 5, 5) & Circle("x", "y", 4, 2, 1.2)

    See Also: `why-squash-can-change-path`
    """

    spec: Spec[Axis] = Field(description="The Spec containing the source midpoints")
    region: Region[Axis] = Field(description="The Region that midpoints will be inside")
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        frames = self.spec.calculate(bounds, nested)
        for axis_set in self.region.axis_sets():
            # Find the start and end index of any dimensions containing these axes
            matches = [i for i, d in enumerate(frames) if set(d.axes()) & axis_set]
            assert matches, f"No Specs match axes {list(axis_set)}"
            si, ei = matches[0], matches[-1]
            if si != ei:
                # The axis_set spans multiple Dimensions, squash them together
                # If the spec to be squashed is nested (inside the Mask or outside)
                # then check the path changes if requested
                check_path_changes = (nested or si) and self.check_path_changes
                squashed = squash_frames(frames[si : ei + 1], check_path_changes)
                frames = frames[:si] + [squashed] + frames[ei + 1 :]
        # Generate masks from the midpoints showing what's inside
        masked_frames = []
        for f in frames:
            indices = get_mask(self.region, f.midpoints).nonzero()[0]
            masked_frames.append(f.extract(indices))
        return masked_frames

    def dimension_info(self) -> DimensionInfo:
        """
        As Mask applies a Region to a Spec, which may alter the Spec, but this is not
        knowable without calculating the entire Spec, we have to calculate the Spec
        here.
        Currently we throw away the results of this calculation, but in future we may
        want to cache the result, or else modify the behaviour of this method generally
        to match.
        """
        frames = self.calculate(bounds=False, nested=False)
        shape = tuple(len(x.midpoints) for x in frames)
        axes = tuple(tuple(x.axes()) for x in frames)
        snaked = tuple(isinstance(x, SnakedFrames) for x in frames)
        return DimensionInfo(axes=axes, shape=shape, snaked=snaked)

    # *+ bind more tightly than &|^ so without these overrides we
    # would need to add brackets to all combinations of Regions
    def __or__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region | o))

    def __and__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region & o))

    def __xor__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region ^ o))

    # This is here for completeness, tends not to be called as - binds
    # tighter than &
    def __sub__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region - o))


@dataclass(config=StrictConfig)
class Snake(Spec[Axis]):
    """Run the Spec in reverse on every other iteration when nested.

    Typically created with the ``~`` operator.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("y", 1, 3, 3) * ~Line("x", 3, 5, 5)
    """

    spec: Spec[Axis] = Field(
        description="The Spec to run in reverse every other iteration"
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        return [
            SnakedFrames.from_frames(segment)
            for segment in self.spec.calculate(bounds, nested)
        ]

    def dimension_info(self) -> DimensionInfo:
        spec_info = self.spec.dimension_info()
        return DimensionInfo(
            axes=spec_info.axes,
            shape=spec_info.shape,
            snaked=(True,) * len(spec_info.shape),
        )
        return self.spec.dimension_info()


@dataclass(config=StrictConfig)
class Concat(Spec[Axis]):
    """Concatenate two Specs together, running one after the other.

    Each Dimension of left and right must contain the same axes. Typically
    formed using `Spec.concat`.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("x", 1, 3, 3).concat(Line("x", 4, 5, 5))
    """

    left: Spec[Axis] = Field(
        description="The left-hand Spec to Concat, midpoints will appear earlier"
    )
    right: Spec[Axis] = Field(
        description="The right-hand Spec to Concat, midpoints will appear later"
    )

    gap: bool = Field(
        description="If True, force a gap in the output at the join", default=False
    )
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        dim_left = squash_frames(
            self.left.calculate(bounds, nested), nested and self.check_path_changes
        )
        dim_right = squash_frames(
            self.right.calculate(bounds, nested), nested and self.check_path_changes
        )
        dim = dim_left.concat(dim_right, self.gap)
        return [dim]

    def dimension_info(self) -> DimensionInfo:
        left_info = self.left.dimension_info()
        right_info = self.right.dimension_info()
        assert left_info.axes == right_info.axes
        # We will squash each spec into 1 dimension
        left_size = reduce(lambda a, b: a * b, left_info.shape)
        right_size = reduce(lambda a, b: a * b, right_info.shape)
        return DimensionInfo(
            axes=left_info.axes,
            shape=(left_size + right_size,),
            snaked=left_info.snaked,  # Non-matching Snake axes cannot be Concat
        )


@dataclass(config=StrictConfig)
class Squash(Spec[Axis]):
    """Squash a stack of Frames together into a single expanded Frames object.

    See Also:
        `why-squash-can-change-path`

    .. example_spec::

        from scanspec.specs import Line, Squash

        spec = Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4))
    """

    spec: Spec[Axis] = Field(description="The Spec to squash the dimensions of")
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        dims = self.spec.calculate(bounds, nested)
        dim = squash_frames(dims, nested and self.check_path_changes)
        return [dim]

    def dimension_info(self) -> DimensionInfo:
        spec_info = self.spec.dimension_info()
        return DimensionInfo(
            axes=(reduce(lambda a, b: a + b, spec_info.axes),),
            shape=(reduce(lambda a, b: a * b, spec_info.shape),),
        )


def _dimensions_from_indexes(
    func: Callable[[np.ndarray], Dict[Axis, np.ndarray]],
    axes: List,
    num: int,
    bounds: bool,
) -> List[Frames[Axis]]:
    # Calc num midpoints (fences) from 0.5 .. num - 0.5
    midpoints_calc = func(np.linspace(0.5, num - 0.5, num))
    midpoints = {a: midpoints_calc[a] for a in axes}
    if bounds:
        # Calc num + 1 bounds (posts) from 0 .. num
        bounds_calc = func(np.linspace(0, num, num + 1))
        lower = {a: bounds_calc[a][:-1] for a in axes}
        upper = {a: bounds_calc[a][1:] for a in axes}
        # Points must have no gap as upper[a][i] == lower[a][i+1]
        # because we initialized it to be that way
        gap = np.zeros(num, dtype=np.bool_)
        dimension = Frames(midpoints, lower, upper, gap)
        # But calc the first point as difference between first
        # and last
        gap[0] = gap_between_frames(dimension, dimension)
    else:
        # Gap can be calculated in Dimension
        dimension = Frames(midpoints)
    return [dimension]


@dataclass(config=StrictConfig)
class Line(Spec[Axis]):
    """Linearly spaced frames with start and stop as first and last midpoints.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("x", 1, 2, 5)
    """

    axis: Axis = Field(description="An identifier for what to move")
    start: float = Field(description="Midpoint of the first point of the line")
    stop: float = Field(description="Midpoint of the last point of the line")
    num: int = Field(min=1, description="Number of frames to produce")

    def dimension_info(self) -> DimensionInfo:
        return DimensionInfo(axes=((self.axis,),), shape=(self.num,))

    def _line_from_indexes(self, indexes: np.ndarray) -> Dict[Axis, np.ndarray]:
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

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        return _dimensions_from_indexes(
            self._line_from_indexes, [self.axis], self.num, bounds
        )

    @classmethod
    def bounded(
        cls,
        axis: Axis = Field(description="An identifier for what to move"),
        lower: float = Field(description="Lower bound of the first point of the line"),
        upper: float = Field(description="Upper bound of the last point of the line"),
        num: int = Field(min=1, description="Number of frames to produce"),
    ) -> Line[Axis]:
        """Specify a Line by extreme bounds instead of midpoints.

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
        return cls(axis, start, stop, num)


@dataclass(config=StrictConfig)
class Static(Spec[Axis]):
    """A static frame, repeated num times, with axis at value.

    Can be used to set axis=value at every point in a scan.

    .. example_spec::

        from scanspec.specs import Line, Static

        spec = Line("y", 1, 2, 3).zip(Static("x", 3))
    """

    axis: Axis = Field(description="An identifier for what to move")
    value: float = Field(description="The value at each point")
    num: int = Field(min=1, description="Number of frames to produce", default=1)

    @classmethod
    def duration(
        cls: Type[Static],
        duration: float = Field(description="The duration of each static point"),
        num: int = Field(min=1, description="Number of frames to produce", default=1),
    ) -> Static[str]:
        """A static spec with no motion, only a duration repeated "num" times.

        .. example_spec::

            from scanspec.specs import Line, Static

            spec = Line("y", 1, 2, 3).zip(Static.duration(0.1))
        """
        return cls(DURATION, duration, num)

    def _repeats_from_indexes(self, indexes: np.ndarray) -> Dict[Axis, np.ndarray]:
        return {self.axis: np.full(len(indexes), self.value)}

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        return _dimensions_from_indexes(
            self._repeats_from_indexes, [self.axis], self.num, bounds
        )

    def dimension_info(self) -> DimensionInfo:
        return DimensionInfo(axes=((self.axis,),), shape=(self.num,))


@dataclass(config=StrictConfig)
class Spiral(Spec[Axis]):
    """Archimedean spiral of "x_axis" and "y_axis".

    Starts at centre point ("x_start", "y_start") with angle "rotate". Produces
    "num" points in a spiral spanning width of "x_range" and height of "y_range"

    .. example_spec::

        from scanspec.specs import Spiral

        spec = Spiral("x", "y", 1, 5, 10, 50, 30)
    """

    # TODO: Make use of typing.Annotated upon fix of
    # https://github.com/pydantic/pydantic/issues/3496
    x_axis: Axis = Field(description="An identifier for what to move for x")
    y_axis: Axis = Field(description="An identifier for what to move for y")
    x_start: float = Field(description="x centre of the spiral")
    y_start: float = Field(description="y centre of the spiral")
    x_range: float = Field(description="x width of the spiral")
    y_range: float = Field(description="y width of the spiral")
    num: int = Field(min=1, description="Number of frames to produce")
    rotate: float = Field(
        description="How much to rotate the angle of the spiral", default=0.0
    )

    def dimension_info(self) -> DimensionInfo:
        return DimensionInfo(
            # TODO: reversed from __init__ args, a good idea?
            axes=((self.y_axis, self.x_axis),),
            shape=(self.num,),
        )

    def _spiral_from_indexes(self, indexes: np.ndarray) -> Dict[Axis, np.ndarray]:
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
            self.y_axis: self.y_start + y_scale * phi * np.cos(phi + self.rotate),
            self.x_axis: self.x_start + x_scale * phi * np.sin(phi + self.rotate),
        }

    def calculate(self, bounds=True, nested=False) -> List[Frames[Axis]]:
        return _dimensions_from_indexes(
            self._spiral_from_indexes, [self.y_axis, self.x_axis], self.num, bounds
        )

    @classmethod
    def spaced(
        cls,
        x_axis: Axis = Field(description="An identifier for what to move for x"),
        y_axis: Axis = Field(description="An identifier for what to move for y"),
        x_start: float = Field(description="x centre of the spiral"),
        y_start: float = Field(description="y centre of the spiral"),
        radius: float = Field(description="radius of the spiral"),
        dr: float = Field(description="difference between each ring"),
        rotate: float = Field(
            description="How much to rotate the angle of the spiral", default=0.0
        ),
    ) -> Spiral[Axis]:
        """Specify a Spiral equally spaced in "x_axis" and "y_axis".

        .. example_spec::

            from scanspec.specs import Spiral

            spec = Spiral.spaced("x", "y", 0, 0, 10, 3)
        """
        # phi = sqrt(4 * pi * num)
        # and: n_rings = phi / (2 * pi)
        # so: n_rings * 2 * pi = sqrt(4 * pi * num)
        # so: num = n_rings^2 * pi
        n_rings = radius / dr
        num = int(n_rings**2 * np.pi)
        return cls(
            x_axis, y_axis, x_start, y_start, radius * 2, radius * 2, num, rotate
        )


def fly(spec: Spec[Axis], duration: float) -> Spec[Axis]:
    """Flyscan, zipping with fixed duration for every frame.

    Args:
        spec: The source `Spec` to continuously move
        duration: How long to spend at each frame in the spec

    .. example_spec::

        from scanspec.specs import Line, fly

        spec = fly(Line("x", 1, 2, 3), 0.1)
    """
    return spec.zip(Static.duration(duration))


def step(spec: Spec[Axis], duration: float, num: int = 1) -> Spec[Axis]:
    """Step scan, with num frames of given duration at each frame in the spec.

    Args:
        spec: The source `Spec` with midpoints to move to and stop
        duration: The duration of each scan frame
        num: Number of frames to produce with given duration at each of frame
            in the spec

    .. example_spec::

        from scanspec.specs import Line, step

        spec = step(Line("x", 1, 2, 3), 0.1)
    """
    return spec * Static.duration(duration, num)
