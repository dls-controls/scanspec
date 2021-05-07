Creating a Scan Spec
====================

This tutorial shows how to create Scan Specs of increasing complexity, plotting
the results.

Line
----

We'll start with a simple one, a `Line`. If you enter the following
code into an interactive Python terminal, it should plot a graph of a 1D line:

.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("x", 1, 2, 5)
    plot_spec(spec)

This will create a Spec with 5 points, the first centred on 1, the last centred
on 2. The black dots mark the midpoints on the Path. The coloured
lines mark the motion from the lower to upper bound of each point on the Path.
The grey arrowhead marks the start of the scan, and the cross marks the end.


Plotting from the commandline
-----------------------------

To quickly plot Scan Specs you can use the commandline client. The input is
evaluated with variables ``a`` to ``z`` defined and the output plotted.

For example, for the Line example above you would type::

    $ scanspec plot 'Line(x, 1, 2, 5)'


Line with 2 axes
----------------

If we want to plot a Line in two axes, we can do this with `Zip`, or the ``+`` operator:

.. example_spec::

    from scanspec.specs import Line

    spec = Line("y", 3, 4, 5) + Line("x", 1, 2, 5)


Grid
----

We can make a grid by creating the `Product` of 2 Lines with the ``*`` operator:

.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * Line("x", 1, 2, 5)
    plot_spec(spec)

The plot shows grey arrowed lines marking the turnarounds. These are added by
the plotting function as an indication of what a scanning program might do between
two disjoint points, it is not a guarantee of the path that will be taken.

.. _snaked-grid:

Snaked Grid
-----------

We can `Snake` a Spec with the ``~`` operator. If we apply this to the inner Spec of
our grid we get:

.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5)
    plot_spec(spec)


Masking with Regions
--------------------

We can apply a `Mask` to only include points within a given `Region` using
the ``&`` operator:

.. plot::

    from scanspec.specs import Line
    from scanspec.regions import Circle
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6)
    plot_spec(spec)


Masking with Multiple Regions
-----------------------------

We can apply set-like operators to Masked Specs:

- ``|``: `UnionOf` two Regions
- ``&``: `IntersectionOf` two Regions
- ``-``: `DifferenceOf` two Regions
- ``^``: `SymmetricDifferenceOf` two Regions

For example:

.. plot::

    from scanspec.specs import Line
    from scanspec.regions import Circle
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6) - Circle("x", "y", 1.4, 3.5, 0.2)
    plot_spec(spec)


Conclusion
----------

This tutorial has demonstrated some Specs and combinations of them. From here
you may like to read `iterate-a-spec` to see how a scanning system could use
these Specs and `serialize-a-spec` to see how you might send one to such a
scanning system.
