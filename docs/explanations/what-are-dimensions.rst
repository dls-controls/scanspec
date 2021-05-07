.. _what-are-dimensions:

What are Dimensions?
====================

If a Spec tells you the parameters of a scan, Dimensions gives you the points
that will let you actually exectute the scan. A list of Dimensions is
interpreted as nested from slowest moving to fastest moving, so each faster
Dimension will iterate once per position of the slower Dimension. When
fly-scanning the axis will traverse lower-midpoints-upper on the fastest
Dimension for each point in the scan.

An Example
----------

>>> from scanspec.specs import Line
>>> spec = Line("y", 4, 5, 6) * Line("x", 1, 2, 3)
>>> dims = spec.create_dimensions()
>>> len(dims)
2
>>> len(dims[0])
6
>>> len(dims[1])
3
>>> dims[0].midpoints
{'y': array([4. , 4.2, 4.4, 4.6, 4.8, 5. ])}
>>> dims[1].midpoints
{'x': array([1. , 1.5, 2. ])}

So the `Product` of two `Lines <Line>` creates a list of 2 Dimensions, the first
having the same size as the outer line, and the second having the same size as
the inner. Executing the scan will iterate the inner dimension 6 times, once for
each point in the outer dimension, 18 points in all.

Why a list of Dimensions?
-------------------------

A list of Dimensions are created to give the most compact representation of the
scan. Imagine a 100x2000x2000 point scan, which creates a list of 3 Dimensions.
Considering just the midpoint arrays, they would take 100 + 2000 + 2000 float64
= 32.8kB RAM in our list form. If we squashed the list into a single Dimension
100 * 2000 * 2000 float64 = 3.2GB of RAM. The scan itself is likely to be
executed over a long period of time, so it makes sense to save on the memory and
calculate the squashed points as they are needed.

What about Regions?
-------------------

Regions will stop the regularity of the nested Dimensions, so will cause them to
be squashed into a single Dimension. Taking our example above, if we `Mask` the
grid with a `Circle`, then the `Line` in ``x`` won't have 3 points in each
iteration, the number of points is dependent on ``y``. This means that a
`Mask` will squash any Specs together referred to by its Regions.

How do ScanSpec Dimensions relate to HDF5 Dimensions?
-----------------------------------------------------

A Spec by itself does not specify how the scan points should be written to disk.
The simplest strategy is to simply write a stack of images, along with the
setpoint positions specified in the Dimensions, and let the reader of the data
pick out the required frames from the positions. However, the list of Dimensions
contain all the information required to reshape a stack of HDF5 frames into the
dimensionality of the scan using a VDS. This holds until we turn snake on, at
which point it destroys the performance of the VDS. For this reason, it is
advisable to `Squash` any snaking Specs with the first non-snaking axis above it
so that the Dimension will not be snaking. See `./why-squash-can-change-path` for
some details on this.

