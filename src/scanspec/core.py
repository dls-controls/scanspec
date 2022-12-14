from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import new_class
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from typing_extensions import Literal, get_args

__all__ = [
    "if_instance_do",
    "Axis",
    "AxesPoints",
    "Frames",
    "SnakedFrames",
    "gap_between_frames",
    "squash_frames",
    "Path",
    "Midpoints",
    "type_adding_dataclass",
    "union_of_subclasses",
]


def _rec_subclasses(cls: type) -> Iterator[type]:
    for sub_cls in cls.__subclasses__():
        yield sub_cls
        yield from _rec_subclasses(sub_cls)


def to_gql_input(ob) -> str:
    """Convert plain Python objects to their GraphQL representation.

    >>> to_gql_input({"a": {"b": 1, "c": True, "d": [4.2, 3.4], "e": "e"}})
    '{a: {b: 1, c: true, d: [4.2, 3.4], e: "e"}}'
    """
    if isinstance(ob, dict):
        inner = ", ".join(f"{k}: {to_gql_input(v)}" for k, v in ob.items())
        return "{%s}" % inner
    elif isinstance(ob, list):
        inner = ", ".join(to_gql_input(v) for v in ob)
        return "[%s]" % inner
    elif isinstance(ob, (str, int, float, bool)):
        return json.dumps(ob)
    else:
        raise ValueError("Cannot format %r" % ob)


def type_adding_dataclass(cls):
    if not hasattr(cls, "__annotations__"):
        setattr(cls, "__annotations__", {})
    cls.__annotations__["type"] = Literal[cls.__name__]
    setattr(cls, "type", cls.__name__)
    return dataclass(cls, config=ConfigDict(arbitrary_types_allowed=True))


def union_of_subclasses(cls, ann=None):
    # Union of subclasses without base
    if ann is not None:
        args = get_args(ann)
        subclasses = tuple(c[args] for c in _rec_subclasses(cls))
    else:
        subclasses = tuple(_rec_subclasses(cls))
    if len(subclasses) > 1:
        return Union[subclasses]


def if_instance_do(x: Any, cls: Type, func: Callable):
    """If x is of type cls then return func(x), otherwise return NotImplemented.

    Used as a helper when implementing operator overloading.
    """
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


#: A type variable for an `axis_` that can be specified for a scan
Axis = TypeVar("Axis")

#: Map of axes to float ndarray of points
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
AxesPoints = Dict[Axis, np.ndarray]


class Frames(Generic[Axis]):
    """Represents a series of scan frames along a number of axes.

    During a scan each axis will traverse lower-midpoint-upper for each frame.

    Args:
        midpoints: The midpoints of scan frames for each axis
        lower: Lower bounds of scan frames if different from midpoints
        upper: Upper bounds of scan frames if different from midpoints
        gap: If supplied, define if there is a gap between frame and previous
            otherwise it is calculated by looking at lower and upper bounds

    Typically used in two ways:

    - A list of Frames objects returned from `Spec.calculate` represents a scan
      as a linear stack of frames. Interpreted as nested from slowest moving to
      fastest moving, so each faster Frames object will iterate once per
      position of the slower Frames object. It is passed to a `Path` for
      calculation of the actual scan path.
    - A single Frames object returned from `Path.consume` represents a chunk of
      frames forming part of a scan path, for interpretation by the code
      that will actually perform the scan.

    See Also:
        `technical-terms`
    """

    def __init__(
        self,
        midpoints: AxesPoints[Axis],
        lower: AxesPoints[Axis] = None,
        upper: AxesPoints[Axis] = None,
        gap: np.ndarray = None,
    ):
        #: The midpoints of scan frames for each axis
        self.midpoints = midpoints
        #: The lower bounds of each scan frame in each axis for fly-scanning
        self.lower = lower or midpoints
        #: The upper bounds of each scan frame in each axis for fly-scanning
        self.upper = upper or midpoints
        if gap is not None:
            #: Whether there is a gap between this frame and the previous. First
            #: element is whether there is a gap between the last frame and the first
            self.gap = gap
        else:
            # Need to calculate gap as not passed one
            # We have a gap if upper[i] != lower[i+1] for any axes
            axes_gap = [
                np.roll(u, 1) != l
                for u, l in zip(self.upper.values(), self.lower.values())
            ]
            self.gap = np.logical_or.reduce(axes_gap)
        # Check all axes and ordering are the same
        assert list(self.midpoints) == list(self.lower) == list(self.upper), (
            f"Mismatching axes "
            f"{list(self.midpoints)} != {list(self.lower)} != {list(self.upper)}"
        )
        # Check all lengths are the same
        lengths = set(
            len(arr)
            for d in (self.midpoints, self.lower, self.upper)
            for arr in d.values()
        )
        lengths.add(len(self.gap))
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    def axes(self) -> List[Axis]:
        """The axes which will move during the scan.

        These will be present in `midpoints`, `lower` and `upper`.
        """
        return list(self.midpoints.keys())

    def __len__(self) -> int:
        """The number of frames in this section of the scan."""
        # All axespoints arrays are same length, pick the first one
        return len(self.gap)

    def extract(self, indices: np.ndarray, calculate_gap=True) -> Frames[Axis]:
        """Return a new Frames object restricted to the indices provided.

        Args:
            indices: The indices of the frames to extract, modulo scan length
            calculate_gap: If True then recalculate the gap from upper and lower

        >>> frames = Frames({"x": np.array([1, 2, 3])})
        >>> frames.extract(np.array([1, 0, 1])).midpoints
        {'x': array([2, 1, 2])}
        """
        dim_indices = indices % len(self)

        def extract_dict(ds: Iterable[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            for d in ds:
                return {k: v[dim_indices] for k, v in d.items()}
            return {}

        def extract_gap(gaps: Iterable[np.ndarray]) -> Optional[np.ndarray]:
            for gap in gaps:
                if not calculate_gap:
                    return gap[dim_indices]
            return None

        return _merge_frames(self, dict_merge=extract_dict, gap_merge=extract_gap)

    def concat(self, other: Frames[Axis], gap: bool = False) -> Frames[Axis]:
        """Return a new Frames object concatenating self and other.

        Requires both Frames objects to have the same axes, but not necessarily in
        the same order. The order is inherited from self, so other may be reordered.

        Args:
            other: The Frames to concatenate to self
            gap: Whether to force a gap between the two Frames objects

        >>> frames = Frames({"x": np.array([1, 2, 3]), "y": np.array([6, 5, 4])})
        >>> frames2 = Frames({"y": np.array([3, 2, 1]), "x": np.array([4, 5, 6])})
        >>> frames.concat(frames2).midpoints
        {'x': array([1, 2, 3, 4, 5, 6]), 'y': array([6, 5, 4, 3, 2, 1])}
        """
        assert set(self.axes()) == set(
            other.axes()
        ), f"axes {self.axes()} != {other.axes()}"

        def concat_dict(ds: Sequence[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            # Concat each array in midpoints, lower, upper. E.g.
            # lower[ax] = np.concatenate(self.lower[ax], other.lower[ax])
            return {a: np.concatenate([d[a] for d in ds]) for a in self.axes()}

        def concat_gap(gaps: Sequence[np.ndarray]) -> np.ndarray:
            g = np.concatenate(gaps)
            # Calc the first frame
            g[0] = gap_between_frames(other, self)
            # And the join frame
            g[len(self)] = gap or gap_between_frames(self, other)
            return g

        return _merge_frames(self, other, dict_merge=concat_dict, gap_merge=concat_gap)

    def zip(self, other: Frames[Axis]) -> Frames[Axis]:
        """Return a new Frames object merging self and other.

        Require both Frames objects to not share axes.

        >>> fx = Frames({"x": np.array([1, 2, 3])})
        >>> fy = Frames({"y": np.array([5, 6, 7])})
        >>> fx.zip(fy).midpoints
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        overlapping = list(set(self.axes()).intersection(other.axes()))
        assert not overlapping, f"Zipping would overwrite axes {overlapping}"

        def zip_dict(ds: Sequence[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            # Merge dicts for midpoints, lower, upper. E.g.
            # lower[ax] = {**self.lower[ax], **other.lower[ax]}
            return dict(kv for d in ds for kv in d.items())

        def zip_gap(gaps: Sequence[np.ndarray]) -> np.ndarray:
            # Gap if either frames has a gap. E.g.
            # gap[i] = self.gap[i] | other.gap[i]
            return np.logical_or.reduce(gaps)

        return _merge_frames(self, other, dict_merge=zip_dict, gap_merge=zip_gap)


def _merge_frames(
    *stack: Frames[Axis],
    dict_merge=Callable[[Sequence[AxesPoints[Axis]]], AxesPoints[Axis]],
    gap_merge=Callable[[Sequence[np.ndarray]], Optional[np.ndarray]],
) -> Frames[Axis]:
    types = set(type(fs) for fs in stack)
    assert len(types) == 1, f"Mismatching types for {stack}"
    cls = types.pop()

    # If any lower or upper are different, apply to those
    kwargs = {}
    for a in ("lower", "upper"):
        if any(fs.midpoints is not getattr(fs, a) for fs in stack):
            kwargs[a] = dict_merge([getattr(fs, a) for fs in stack])

    # Apply to midpoints, force calculation of gap
    return cls(
        midpoints=dict_merge([fs.midpoints for fs in stack]),
        gap=gap_merge([fs.gap for fs in stack]),
        **kwargs,
    )


class SnakedFrames(Frames[Axis]):
    """Like a `Frames` object, but each alternate repetition will run in reverse."""

    def __init__(
        self,
        midpoints: AxesPoints[Axis],
        lower: AxesPoints[Axis] = None,
        upper: AxesPoints[Axis] = None,
        gap: np.ndarray = None,
    ):
        super().__init__(midpoints, lower=lower, upper=upper, gap=gap)
        # Override first element of gap to be True, as subsequent runs
        # of snake scans are always joined end -> start
        self.gap[0] = False

    @classmethod
    def from_frames(cls, frames: Frames[Axis]) -> SnakedFrames[Axis]:
        """Create a snaked version of a `Frames` object."""
        return cls(frames.midpoints, frames.lower, frames.upper, frames.gap)

    def extract(self, indices: np.ndarray, calculate_gap=True) -> Frames[Axis]:
        """Return a new Frames object restricted to the indices provided.

        Args:
            indices: The indices of the frames to extract, can extend past len(self)
            calculate_gap: If True then recalculate the gap from upper and lower

        >>> frames = SnakedFrames({"x": np.array([1, 2, 3])})
        >>> frames.extract(np.array([0, 1, 2, 3, 4, 5])).midpoints
        {'x': array([1, 2, 3, 3, 2, 1])}
        """
        # Calculate the indices
        # E.g for len = 4
        # indices:       0123456789
        # backwards:     0000111100
        # snake_indices: 0123321001
        # gap_indices:   0123032101
        length = len(self)
        backwards = (indices // length) % 2
        snake_indices = np.where(backwards, (length - 1) - indices, indices) % length
        if not calculate_gap:
            cls = Frames
            gap = self.gap[np.where(backwards, length - indices, indices) % length]
        else:
            cls = type(self)
            gap = None

        # If lower or upper are different, apply to those
        kwargs = {}
        if self.midpoints is not self.lower:
            # If going backwards select from the opposite bound
            kwargs["lower"] = {
                k: np.where(backwards, self.upper[k][snake_indices], v[snake_indices])
                for k, v in self.lower.items()
            }
        if self.midpoints is not self.upper:
            kwargs["upper"] = {
                k: np.where(backwards, self.lower[k][snake_indices], v[snake_indices])
                for k, v in self.upper.items()
            }

        # Apply to midpoints
        return cls(
            {k: v[snake_indices] for k, v in self.midpoints.items()}, gap=gap, **kwargs
        )


def gap_between_frames(frames1: Frames[Axis], frames2: Frames[Axis]) -> bool:
    """Is there a gap between end of frames1 and start of frames2."""
    return any(frames1.upper[a][-1] != frames2.lower[a][0] for a in frames1.axes())


def squash_frames(stack: List[Frames[Axis]], check_path_changes=True) -> Frames[Axis]:
    """Squash a stack of nested Frames into a single one.

    Args:
        stack: The Frames stack to squash, from slowest to fastest moving
        check_path_changes: If True then check that nesting the output
            Frames object within others will provide the same path
            as nesting the input Frames stack within others

    See Also:
        `why-squash-can-change-path`

    >>> fx = SnakedFrames({"x": np.array([1, 2])})
    >>> fy = Frames({"y": np.array([3, 4])})
    >>> squash_frames([fy, fx]).midpoints
    {'y': array([3, 3, 4, 4]), 'x': array([1, 2, 2, 1])}
    """
    path = Path(stack)
    # Consuming a Path through these Frames performs the squash
    squashed = path.consume()
    # Check that the squash is the same as the original
    if stack and isinstance(stack[0], SnakedFrames):
        squashed = SnakedFrames.from_frames(squashed)
        # The top level is snaking, so this Frames object will run backwards
        # This means any non-snaking axes will run backwards, which is
        # surprising, so don't allow it
        if check_path_changes:
            non_snaking = [
                k for d in stack for k in d.axes() if not isinstance(d, SnakedFrames)
            ]
            if non_snaking:
                raise ValueError(
                    f"Cannot squash non-snaking Frames inside a SnakingFrames "
                    f"otherwise {non_snaking} would run backwards"
                )
    elif check_path_changes:
        # The top level is not snaking, so make sure there is an even
        # number of iterations of any snaking axis within it so it
        # doesn't jump when this frames object is iterated a second time
        for i, frames in enumerate(stack):
            # A SnakedFrames within a non-snaking top level must repeat
            # an even number of times
            if isinstance(frames, SnakedFrames) and np.product(path.lengths[:i]) % 2:
                raise ValueError(
                    f"Cannot squash SnakingFrames inside a non-snaking Frames "
                    f"when they do not repeat an even number of times "
                    f"otherwise {frames.axes()} would jump in position"
                )
    return squashed


class Path(Generic[Axis]):
    """A consumable route through a stack of Frames, representing a scan path.

    Args:
        stack: The Frames stack describing the scan, from slowest to fastest
            moving
        start: The index of where in the Path to start
        num: The number of scan frames to produce after start. None means up to
            the end

    See Also:
        `iterate-a-spec`
    """

    def __init__(self, stack: List[Frames[Axis]], start: int = 0, num: int = None):
        #: The Frames stack describing the scan, from slowest to fastest moving
        self.stack = stack
        #: Index that is next to be consumed
        self.index = start
        #: The lengths of all the stack
        self.lengths = np.array([len(f) for f in stack])
        #: Index of the end frame, one more than the last index that will be
        #: produced
        self.end_index = np.product(self.lengths)
        if num is not None and start + num < self.end_index:
            self.end_index = start + num

    def consume(self, num: int = None) -> Frames[Axis]:
        """Consume at most num frames from the Path and return as a Frames object.

        >>> fx = SnakedFrames({"x": np.array([1, 2])})
        >>> fy = Frames({"y": np.array([3, 4])})
        >>> path = Path([fy, fx])
        >>> path.consume(3).midpoints
        {'y': array([3, 3, 4]), 'x': array([1, 2, 2])}
        >>> path.consume(3).midpoints
        {'y': array([4]), 'x': array([1])}
        >>> path.consume(3).midpoints
        {'y': array([], dtype=int64), 'x': array([], dtype=int64)}
        """
        if num is None:
            end_index = self.end_index
        else:
            end_index = min(self.index + num, self.end_index)
        indices = np.arange(self.index, end_index)
        self.index = end_index
        stack: Frames[Axis] = Frames(
            {}, {}, {}, np.zeros(indices.shape, dtype=np.bool_)
        )
        # Example numbers below from a 2x3x4 ZxYxX scan
        for i, frames in enumerate(self.stack):
            # Number of times each frame will repeat: Z:12, Y:4, X:1
            repeats = np.product(self.lengths[i + 1 :])
            # Scan indices mapped to indices within Frames object:
            # Z:000000000000111111111111
            # Y:000011112222000011112222
            # X:012301230123012301230123
            if repeats > 1:
                dim_indices = indices // repeats
            else:
                dim_indices = indices
            # Create the sliced frames
            sliced = frames.extract(dim_indices, calculate_gap=False)
            if repeats > 1:
                # Whether this frames contributes to the gap bit
                # Z:000000000000100000000000
                # Y:000010001000100010001000
                # X:111111111111111111111111
                in_gap = (indices % repeats) == 0
                # If in_gap, then keep the relevant gap bit
                sliced.gap &= in_gap
            # Zip it with the output Frames object
            stack = stack.zip(sliced)
        return stack

    def __len__(self) -> int:
        """Number of frames left in a scan, reduces when `consume` is called."""
        return self.end_index - self.index


class Midpoints(Generic[Axis]):
    """Convenience iterable that produces the scan midpoints for each axis.

    For better performance, consume from a `Path` instead.

    Args:
        stack: The stack of Frames describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> fx = SnakedFrames({"x": np.array([1, 2])})
    >>> fy = Frames({"y": np.array([3, 4])})
    >>> mp = Midpoints([fy, fx])
    >>> for p in mp: print(p)
    {'y': 3, 'x': 1}
    {'y': 3, 'x': 2}
    {'y': 4, 'x': 2}
    {'y': 4, 'x': 1}
    """

    def __init__(self, stack: List[Frames[Axis]]):
        #: The stack of Frames describing the scan, from slowest to fastest moving
        self.stack = stack

    @property
    def axes(self) -> List[Axis]:
        """The axes that will be present in each points dictionary."""
        axes = []
        for frames in self.stack:
            axes += frames.axes()
        return axes

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over."""
        return np.product([len(frames) for frames in self.stack])

    def __iter__(self) -> Iterator[Dict[Axis, float]]:
        """Yield {axis: midpoint} for each frame in the scan."""
        path = Path(self.stack)
        while len(path):
            frames = path.consume(1)
            yield {a: frames.midpoints[a][0] for a in frames.axes()}
