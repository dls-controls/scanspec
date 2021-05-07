from types import new_class
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Type,
    TypeVar,
)

import numpy as np
from apischema import deserialize, schema_ref, serialize
from apischema.conversions import (
    Conversion,
    dataclass_input_wrapper,
    deserializer,
    identity,
    reset_deserializers,
    serializer,
)
from apischema.conversions.converters import serializer
from apischema.metadata import conversion
from apischema.tagged_unions import Tagged, TaggedUnion, get_tagged
from typing_extensions import Annotated

#: The type of class the function will return
T = TypeVar("T")

__all__ = [
    "Serializable",
    "T",
    "AxesPoints",
    "if_instance_do",
    "Dimension",
    "squash_dimensions",
    "Path",
    "Midpoints",
]


# Recursive implementation of type.__subclasses__
def rec_subclasses(cls: Type[T]) -> Iterator[Type[T]]:
    for sub_cls in cls.__subclasses__():
        yield sub_cls
        yield from rec_subclasses(sub_cls)


# {cls_name: [functions]}
_alternative_constructors: Dict[str, List[Callable]] = {}

if TYPE_CHECKING:
    # Close enough for mypy
    alternative_constructor = staticmethod
else:

    def alternative_constructor(f):
        """Register an alternative constructor for this class. This will be returned
        as a staticmethod so the signature should not include self/cls.

        >>> import dataclasses
        >>> @dataclasses.dataclass
        ... class Foo:
        ...     a: int
        ...     @alternative_constructor
        ...     def doubled(b: int) -> "Foo":
        ...         return Foo(b * 2)
        ...
        >>> Foo.doubled(2)
        Foo(a=4)
        """
        cls_name = f.__qualname__.split(".")[0]
        _alternative_constructors.setdefault(cls_name, []).append(f)
        return staticmethod(f)


def _make_tagged_union(base: Type, is_serialization: bool) -> Type[TaggedUnion]:
    # base is a direct subclass of Serializable, like Spec or Region
    namespace: Dict[str, Any] = dict(__annotations__={})
    for cls in rec_subclasses(base):
        # Add tagged field for the Serializable subclass
        namespace["__annotations__"][cls.__name__] = Tagged[cls]  # type: ignore
        if is_serialization:
            # Specify that we should use the identity serialization rather
            # than our registered to_tagged_union() serializer when inside the
            # tagged union
            serialization = Conversion(
                identity,
                source=cls,
                # Tagged field default serialization (to tagged union) must be
                # bypassed. However, dynamic conversion discards schema_ref, so
                # you must put it back manually, and do the bypass in a sub
                # conversion.
                target=Annotated[cls, schema_ref(cls.__name__)],
                sub_conversions=identity,
            )
            namespace[cls.__name__] = Tagged(conversion(serialization=serialization))
        else:
            # Build deserialization aliases for each alternative constructor alias
            for constructor in _alternative_constructors.get(cls.__name__, []):
                alias = (
                    "".join(map(str.capitalize, constructor.__name__.split("_")))
                    + cls.__name__
                )
                # dataclass_input_wrapper uses get_type_hints, but the constructor
                # return type is stringified and the class not defined yet,
                # so it must be assigned manually
                constructor.__annotations__["return"] = cls
                # Wraps the constructor and rename its input class
                wrapper, wrapper_cls = dataclass_input_wrapper(constructor)
                wrapper_cls.__name__ = alias
                # Add constructor tagged field with its conversion
                namespace["__annotations__"][alias] = Tagged[cls]  # type: ignore
                namespace[alias] = Tagged(conversion(deserialization=wrapper))
    # Create the tagged union class
    union = new_class(
        f"Tagged{base.__name__}Union",
        (TaggedUnion,),
        exec_body=lambda ns: ns.update(namespace),
    )
    return union


class Serializable:
    """Base class for registering apischema (de)serialization conversions.
    The conversion class variable of child classes holds the necessary information to
    (de)serialize grandchild classes. Each time a grandchild class is added
    conversion is updated to create a full TaggedUnion for the child class."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Retrieved the base class inheriting Serializable
        bases = [c for c in cls.__mro__ if Serializable in c.__bases__]
        assert (
            len(bases) == 1
        ), f"Cannot have multiple base classes inheriting Serializable {bases}"
        base = bases[0]
        assert issubclass(base, Serializable)
        # Create the serialization tagged union class
        serialization_union = _make_tagged_union(base, is_serialization=True)
        # Register the serializer
        serializer(
            Conversion(
                lambda obj: serialization_union(**{obj.__class__.__name__: obj}),
                source=base,
                target=serialization_union,
            )
        )
        # Create the deserialization tagged union class
        deserialization_union = _make_tagged_union(base, is_serialization=False)
        # Because deserializers stack, they must be reset before being reassigned
        reset_deserializers(base)
        # Register the deserializer using get_tagged
        deserializer(
            Conversion(
                lambda obj: get_tagged(obj)[1],
                source=deserialization_union,
                target=base,
            )
        )

    def serialize(self) -> Mapping[str, Any]:
        """Serialize to a dictionary representation"""
        return serialize(self)

    @classmethod
    def deserialize(cls: Type[T], serialization: Mapping[str, Any]) -> T:
        """Deserialize from a dictionary representation"""
        return deserialize(cls, serialization)


#: Map of axes to points_ndarray
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
AxesPoints = Dict[str, np.ndarray]


def if_instance_do(x, cls: Type[T], func: Callable[[T], Any]):
    """If x is of type cls then return func(x), otherwise return NotImplemented.
    Used as a helper when implementing operator overloading"""
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


class Dimension:
    """A dimension is a repeatable, possibly snaking structure of frames along a
    number of axes.

    Represents a linear stack of frames. A list of Dimensions
    is interpreted as nested from slowest moving to fastest moving, so each
    faster Dimension will iterate once per position of the slower Dimension.
    When fly-scanning the axis will traverse lower-midpoints-upper on the fastest
    Dimension for each point in the scan.

    Args:
        midpoints: The centre points of the scan for each axis
        lower: Lower bounds if different from midpoints
        upper: Upper bounds if different from midpoints
        snake: If True then every other iteration of this Dimension within a
            slower moving Dimension will be reversed

    See Also:
        `what-are-dimensions`
    """

    def __init__(
        self,
        midpoints: AxesPoints,
        lower: AxesPoints = None,
        upper: AxesPoints = None,
        snake: bool = False,
    ):
        #: The centre points of the scan for each axis
        self.midpoints = midpoints
        #: The lower bounds of each scan point in each axis for fly-scanning
        self.lower = lower or midpoints
        #: The upper bounds of each scan point in each axis for fly-scanning
        self.upper = upper or midpoints
        #: Whether every other iteration of this Dimension within a slower
        #: moving Dimension will be reversed
        self.snake = snake
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
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    def axes(self) -> List:
        """The axes that are present in `midpoints`, `lower` and `upper`
        which will move during the scan"""
        return list(self.midpoints.keys())

    def __len__(self) -> int:
        """The number of `frames` in the scan"""
        # All axespoints arrays are same length, pick the first one
        return len(list(self.midpoints.values())[0])

    def _dim_with(self, func: Callable[[str, Any], np.ndarray]) -> "Dimension":
        def apply_func(a: str):
            return {k: func(a, k) for k in getattr(self, a)}

        # Apply to every array in axes
        kwargs = dict(midpoints=apply_func("midpoints"), snake=self.snake)
        # If lower and upper are different, apply to those too
        if self.lower is not self.midpoints:
            kwargs["lower"] = apply_func("lower")
        if self.upper is not self.midpoints:
            kwargs["upper"] = apply_func("upper")
        return Dimension(**kwargs)

    def tile(self, reps: int) -> "Dimension":
        """Return a new Dimension that iterates self reps times

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.tile(reps=2).midpoints
        {'x': array([1, 2, 3, 1, 2, 3])}
        """
        return self._dim_with(lambda a, k: np.tile(getattr(self, a)[k], reps))

    def repeat(self, reps: int) -> "Dimension":
        """Return a new Dimension that repeats each point in self reps times

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.repeat(reps=2).midpoints
        {'x': array([1, 1, 2, 2, 3, 3])}
        """
        return self._dim_with(lambda a, k: np.repeat(getattr(self, a)[k], reps))

    def mask(self, mask: np.ndarray) -> "Dimension":
        """Return a new Dimension that produces only points from self in the
        mask

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.mask(np.array([1, 0, 1])).midpoints
        {'x': array([1, 3])}
        """
        indices = mask.nonzero()[0]
        return self._dim_with(lambda a, k: getattr(self, a)[k][indices])

    def copy(self) -> "Dimension":
        """Return a shallow copy of the current Dimension (dicts copied,
        arrays within them are not)"""
        return self._dim_with(lambda a, k: getattr(self, a)[k])

    def _check_dim(self, other: "Dimension"):
        assert isinstance(other, Dimension), f"Expected Dimension, got {other}"
        assert self.snake == other.snake, "Snake settings don't match"

    def concat(self, other: "Dimension") -> "Dimension":
        """Return a new Dimension with arrays from self and other concatenated
        together. Require both Dimensions to have the same axes and snake
        settings

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim2 = Dimension({"x": np.array([5, 6, 7])})
        >>> dim.concat(dim2).midpoints
        {'x': array([1, 2, 3, 5, 6, 7])}
        """
        self._check_dim(other)
        assert self.axes() == other.axes(), f"axes {self.axes()} != {other.axes()}"
        return self._dim_with(
            lambda a, k: np.concatenate((getattr(self, a)[k], getattr(other, a)[k]))
        )

    def zip(self, other: "Dimension") -> "Dimension":
        """Return a new Dimension with arrays from axes of self and other
        merged together. Require both Dimensions to not share axes, and
        to have the snake settings

        >>> dimx = Dimension({"x": np.array([1, 2, 3])})
        >>> dimy = Dimension({"y": np.array([5, 6, 7])})
        >>> dimx.zip(dimy).midpoints
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        self._check_dim(other)
        overlapping = list(set(self.axes()).intersection(other.axes()))
        assert not overlapping, f"Zipping would overwrite axes {overlapping}"
        # rely on the constructor to check lengths
        dim = Dimension(
            midpoints={**self.midpoints, **other.midpoints},
            lower={**self.lower, **other.lower},
            upper={**self.upper, **other.upper},
            snake=self.snake,
        )
        return dim


def squash_dimensions(
    dimensions: List[Dimension], check_path_changes=True
) -> Dimension:
    """Squash a list of nested Dimensions into a single one.

    Args:
        dimensions: The Dimensions to squash, from slowest to fastest moving
        check_path_changes: If True then check that nesting the output
            Dimension within other Dimensions will provide the same path
            as nesting the input Dimension within other Dimensions

    See Also:
        `why-squash-can-change-path`

    >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
    >>> dimy = Dimension({"y": np.array([3, 4])})
    >>> squash_dimensions([dimy, dimx]).midpoints
    {'y': array([3, 3, 4, 4]), 'x': array([1, 2, 2, 1])}
    """
    path = Path(dimensions)
    # Comsuming a Path of these dimensions performs the squash
    # TODO: dim.tile might give better performance but is much longer
    squashed = path.consume()
    # Check that the squash is the same as the original
    if dimensions and dimensions[0].snake:
        squashed.snake = True
        # The top level is snaking, so this dimension will run backwards
        # This means any non-snaking axes will run backwards, which is
        # surprising, so don't allow it
        if check_path_changes:
            non_snaking = [k for d in dimensions for k in d.axes() if not d.snake]
            if non_snaking:
                raise ValueError(
                    f"Cannot squash non-snaking Specs in a snaking Dimension "
                    f"otherwise {non_snaking} would run backwards"
                )
    elif check_path_changes:
        # The top level is not snaking, so make sure there is an even
        # number of iterations of any snaking axis within it so it
        # doesn't jump when this dimension is iterated a second time
        for i, dim in enumerate(dimensions):
            # A snaking dimension within a non-snaking top level must repeat
            # an even number of times
            if dim.snake and np.product(path._lengths[:i]) % 2:
                raise ValueError(
                    f"Cannot squash snaking Specs in a non-snaking Dimension "
                    f"when they do not repeat an even number of times "
                    f"otherwise {dim.axes()} would jump in position"
                )
    return squashed


class Path:
    """A path is a consumable route through one or more dimensions,
    representing a scan path.

    Args:
        dimensions: The Dimensions describing the scan, from slowest to fastest
            moving
        start: The index of where in the Path to start
        num: The number of scan points to produce after start. None means up to
            the end

    See Also:
        `iterate-a-spec`
    """

    def __init__(
        self, dimensions: List[Dimension], start: int = 0, num: int = None,
    ):
        #: The Dimensions describing the scan, from slowest to fastest moving
        self.dimensions = dimensions
        #: Index that is next to be consumed
        self.index = start
        self._lengths = np.array([len(dim) for dim in dimensions])
        #: Index of the end point, one more than the last index that will be
        #: produced
        self.end_index = np.product(self._lengths)
        if num is not None and start + num < self.end_index:
            self.end_index = start + num

    def consume(self, num: int = None) -> Dimension:
        """Consume at most num points from the Path and return as a Dimension

        >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
        >>> dimy = Dimension({"y": np.array([3, 4])})
        >>> path = Path([dimy, dimx])
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
        midpoints, lower, upper = {}, {}, {}
        if len(indices) > 0:
            self.index = indices[-1] + 1
        # Example numbers below from a 2x3x4 ZxYxX scan
        for i, dim in enumerate(self.dimensions):
            # Number of times each point will repeat: Z:12, Y:4, X:1
            repeats = np.product(self._lengths[i + 1 :])
            # How big is this dim: Z:2, Y:3, X:4
            dim_len = self._lengths[i]
            # Scan indices mapped to indices within dimension:
            # Z:000000000000111111111111
            # Y:000011112222000011112222
            # X:012301230123012301230123
            dim_indices = (indices // repeats) % dim_len
            if dim.snake:
                # Whether this point is running backwards:
                # Z:000000000000000000000000
                # Y:000000000000111111111111
                # X:000011110000111100001111
                backwards = (indices // (repeats * dim_len)) % 2
                # The scan indices mapped to snaking ones:
                # Z:000000000000111111111111
                # Y:000011112222222211110000
                # X:012332100123321001233210
                snake_indices = np.where(
                    backwards, dim_len - 1 - dim_indices, dim_indices
                )
                for axis in dim.axes():
                    midpoints[axis] = dim.midpoints[axis][snake_indices]
                    # If going backwards, select from the opposite bound
                    lower[axis] = np.where(
                        backwards,
                        dim.upper[axis][snake_indices],
                        dim.lower[axis][snake_indices],
                    )
                    upper[axis] = np.where(
                        backwards,
                        dim.lower[axis][snake_indices],
                        dim.upper[axis][snake_indices],
                    )
            else:
                for axis in dim.axes():
                    midpoints[axis] = dim.midpoints[axis][dim_indices]
                    lower[axis] = dim.lower[axis][dim_indices]
                    upper[axis] = dim.upper[axis][dim_indices]
        return Dimension(midpoints, lower, upper)

    def __len__(self) -> int:
        """Number of points left in a scan, reduces when `consume` is called"""
        return self.end_index - self.index


class Midpoints:
    """Convenience iterable that produces the scan points for each axis. For
    better performance, consume from a `Path` instead.

    Args:
        dimensions: The Dimensions describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
    >>> dimy = Dimension({"y": np.array([3, 4])})
    >>> mp = Midpoints([dimy, dimx])
    >>> for p in mp: print(p)
    {'y': 3, 'x': 1}
    {'y': 3, 'x': 2}
    {'y': 4, 'x': 2}
    {'y': 4, 'x': 1}
    """

    def __init__(self, dimensions: List[Dimension]):
        #: The Dimensions describing the scan, from slowest to fastest moving
        self.dimensions = dimensions

    @property
    def axes(self) -> List:
        """The axes that will be present in each points dictionary"""
        axes = []
        for dim in self.dimensions:
            axes += dim.axes()
        return axes

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over"""
        return np.product([len(dim) for dim in self.dimensions])

    def __iter__(self) -> Iterator[AxesPoints]:
        path = Path(self.dimensions)
        while len(path):
            dim = path.consume(1)
            yield {a: dim.midpoints[a][0] for a in dim.axes()}
