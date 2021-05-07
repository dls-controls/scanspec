.. _iterate-a-spec:

How to Iterate a Spec
=====================

A Spec is only the specification for a scan. To execute the scan we need the
frames. We can do this in a few ways.

If you only need the midpoints
------------------------------

If you are conducting a step scan, you only need the midpoints of each
scan frame. You can get these by using the `Spec.midpoints()` method to produce a
`Midpoints` iterator of scan points:

>>> from scanspec.specs import Line
>>> spec = Line("x", 1, 2, 3)
>>> for d in spec.midpoints():
...     print(d)
...
{'x': 1.0}
{'x': 1.5}
{'x': 2.0}

This is simple, but not particularly performant, as the numpy arrays of
points are unpacked point by point into point dictionaries

If you need to do a fly scan
----------------------------

If you are conducting a fly scan then you need the Path that the motor moves
through. You can get that from the lower and upper bounds of each point. It is
more efficient to consume this Path in numpy array chunks that can be processed
into a trajectory:

>>> path = spec.path()
>>> len(path)
3
>>> chunk = path.consume()
>>> chunk.lower
{'x': array([0.75, 1.25, 1.75])}
>>> chunk.midpoints
{'x': array([1. , 1.5, 2. ])}
>>> chunk.upper
{'x': array([1.25, 1.75, 2.25])}
>>> len(path)
0

If ``upper[i] == lower[i+1]`` then we say that two scan points are joined, and
will be blended together in a motion trajectory, otherwise the scanning system
should insert a gap where the motors move between segments


If you need to do multiple runs of the same scan
------------------------------------------------

A `Path` instance is a one-shot consumable view of the list of `Dimension`
instances created by `Spec.create_dimensions()`. Once you have consumed it you
should create a new instance of it. For performance reasons, you can keep the
intermediate Dimensions and create as many `Path` wrappers to them as you need.
You can also give a maximum size to `Path.consume()`

>>> from scanspec.core import Path
>>> dims = spec.create_dimensions()
>>> path = Path(dims)
>>> len(path)
3
>>> chunk = path.consume(2)
>>> len(chunk)
2
>>> len(path)
1
>>> chunk = path.consume(2)
>>> len(chunk)
1
>>> len(path)
0

You can also use this method to only run a subset of the scan:

>>> path = Path(dims, start=1, num=1)
>>> len(path)
1
>>> chunk = path.consume()
>>> len(chunk)
1
>>> len(path)
0

