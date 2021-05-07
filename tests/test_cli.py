from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

from scanspec import cli
from scanspec.service import schema_text


def assert_min_max_2d(line, xmin, xmax, ymin, ymax, length=None):
    if length is not None:
        assert len(line.get_data()[0]) == length
    mins = np.min(line.get_data(), axis=1)
    maxs = np.max(line.get_data(), axis=1)
    assert list(zip(mins, maxs)) == [
        pytest.approx([xmin, xmax]),
        pytest.approx([ymin, ymax]),
    ]


def assert_min_max_3d(line, xmin, xmax, ymin, ymax, zmin, zmax, length=None):
    if length is not None:
        assert len(line.get_data_3d()[0]) == length
    mins = np.min(line.get_data_3d(), axis=1)
    maxs = np.max(line.get_data_3d(), axis=1)
    assert list(zip(mins, maxs)) == [
        pytest.approx([xmin, xmax]),
        pytest.approx([ymin, ymax]),
        pytest.approx([zmin, zmax]),
    ]


def assert_3d_arrow(artist, x, y, z):
    assert artist._verts3d[0][1] == pytest.approx(x)
    assert artist._verts3d[1][1] == pytest.approx(y)
    assert artist._verts3d[2][1] == pytest.approx(z)


def test_plot_1D_line() -> None:
    runner = CliRunner()
    spec = 'Line("x", 1, 2, 2)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 4
    # Splines
    assert_min_max_2d(lines[0], 0.5, 1.5, 0, 0)
    assert_min_max_2d(lines[1], 1.5, 2.5, 0, 0)
    # Capture points
    assert_min_max_2d(lines[2], 1, 2, 0, 0, length=2)
    # End
    assert_min_max_2d(lines[3], 2.5, 2.5, 0, 0)
    # Arrows
    texts = axes.texts
    assert len(texts) == 1
    assert texts[0].xy == [0.5, 0]


def test_plot_1D_step() -> None:
    runner = CliRunner()
    spec = 'step(Line("x", 1, 2, 2), 0.1)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 4
    # Start marker
    assert_min_max_2d(lines[0], 1, 1, 0, 0)
    # Step
    assert_min_max_2d(lines[1], 1, 2, 0, 0)
    # Capture points
    assert_min_max_2d(lines[2], 1, 2, 0, 0, length=2)
    # End
    assert_min_max_2d(lines[3], 2, 2, 0, 0)
    # Arrows
    texts = axes.texts
    assert len(texts) == 1
    assert texts[0].xy == [2, 0]


def test_plot_2D_line() -> None:
    runner = CliRunner()
    spec = 'Line("y", 2, 3, 2) * Snake(Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 7
    # First row
    assert_min_max_2d(lines[0], 0.5, 1.5, 2, 2)
    assert_min_max_2d(lines[1], 1.5, 2.5, 2, 2)
    # Turnaround
    assert_min_max_2d(lines[2], 2.5, 2.7537562, 1.9999880, 3.0000119)
    # Second row
    assert_min_max_2d(lines[3], 1.5, 2.5, 3, 3)
    assert_min_max_2d(lines[4], 0.5, 1.5, 3, 3)
    # Capture points
    assert_min_max_2d(lines[5], 1, 2, 2, 3, length=4)
    # End
    assert_min_max_2d(lines[6], 0.5, 0.5, 3, 3)
    # Arrows
    texts = axes.texts
    assert len(texts) == 2
    assert texts[0].xy == [0.5, 2]
    assert texts[1].xy == [2.5, 3]


def test_plot_2D_line_rect_region() -> None:
    runner = CliRunner()
    spec = "Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1.1, 1.5, 2.1, 30)"
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 6
    # First row
    assert_min_max_2d(lines[0], -0.5, 0.5, 1.5, 1.5)
    # Turnaround
    assert_min_max_2d(lines[1], -0.6071045, 0.60710456, 1.4999969, 2.000003)
    # Second row
    assert_min_max_2d(lines[2], -0.5, 0.5, 2, 2)
    assert_min_max_2d(lines[3], 0.5, 1.5, 2, 2)
    # Capture points
    assert_min_max_2d(lines[4], 0, 1, 1.5, 2, length=3)
    # End
    assert_min_max_2d(lines[5], 1.5, 1.5, 2, 2)
    # Arrows
    texts = axes.texts
    assert len(texts) == 2
    assert texts[0].xy == [-0.5, 1.5]
    assert texts[1].xy == [-0.5, 2]
    # Regions
    patches = axes.patches
    assert len(patches) == 1
    assert type(patches[0]).__name__ == "Rectangle"
    assert patches[0].xy == (0, 1.1)
    assert patches[0].get_height() == 1.0
    assert patches[0].get_width() == 1.5
    assert patches[0].angle == 30


def test_plot_3D_line() -> None:
    runner = CliRunner()
    spec = 'Snake(Line("z", 5, 6, 2) * Line("y", 2, 3, 2) * Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec],)
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 13
    # First grid
    # First row
    assert_min_max_3d(lines[0], 0.5, 1.5, 2, 2, 5, 5)
    assert_min_max_3d(lines[1], 1.5, 2.5, 2, 2, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[2], 2.5, 2.7537562, 1.9999880, 3.0000119, 5, 5)
    # Second row
    assert_min_max_3d(lines[3], 1.5, 2.5, 3, 3, 5, 5)
    assert_min_max_3d(lines[4], 0.5, 1.5, 3, 3, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[5], 0.2462437, 0.5, 3, 3, 4.9999880, 6.0000178)
    # Second grid
    # First row
    assert_min_max_3d(lines[6], 0.5, 1.5, 3, 3, 6, 6)
    assert_min_max_3d(lines[7], 1.5, 2.5, 3, 3, 6, 6)
    # Turnaround
    assert_min_max_3d(lines[8], 2.5, 2.7537562, 1.9999880, 3.0000119, 6, 6)
    # Second row
    assert_min_max_3d(lines[9], 1.5, 2.5, 2, 2, 6, 6)
    assert_min_max_3d(lines[10], 0.5, 1.5, 2, 2, 6, 6)
    # Capture points
    assert_min_max_3d(lines[11], 1, 2, 2, 3, 5, 6, length=8)
    # End
    assert_min_max_3d(lines[12], 0.5, 0.5, 2, 2, 6, 6)
    # Arrows
    artists = axes.artists
    assert len(artists) == 4
    assert_3d_arrow(artists[0], 0.5, 2, 5)
    assert_3d_arrow(artists[1], 2.5, 3, 5)
    assert_3d_arrow(artists[2], 0.5, 3, 6)
    assert_3d_arrow(artists[3], 2.5, 2, 6)


def test_schema() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.schema)
    assert result.exit_code == 0
    # Note: Click.echo adds a newline when formatting the string
    assert result.output == schema_text() + "\n"
