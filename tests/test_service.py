from unittest import mock

import graphql
import pytest
from graphql.type.schema import assert_schema
from numpy import array

from scanspec.service import Points, schema, schema_text


# Returns a dummy 'points' dataclass for resolver testing
@pytest.fixture
def points() -> Points:
    return Points(array([1.5, 0.0, 0.25, 1.0, 0.0]))


# Returns an empty 'points' dataclass for resolver testing
@pytest.fixture
def points_none() -> Points:
    return Points(None)


# GET_POINTS RESOLVER TEST(S) #
def test_float_list(points) -> None:
    assert points.float_list() == [1.5, 0.0, 0.25, 1.0, 0.0]


def test_float_list_none(points_none) -> None:
    assert points_none.float_list() is None


def test_string(points) -> None:
    assert points.string() == "[1.5  0.   0.25 1.   0.  ]"


def test_b64(points) -> None:
    assert points.b64() == "AAAAAAAA+D8AAAAAAAAAAAAAAAAAANA/AAAAAAAA8D8AAAAAAAAAAA=="


def test_b64_none(points_none) -> None:
    assert points_none.b64() is None


def test_decodeb64(points) -> None:
    assert points.b64Decode() == "[1.5  0.   0.25 1.   0.  ]"


def test_decodeb64_none(points_none) -> None:
    assert points_none.b64Decode() is None


# VALIDATE SPEC QUERY TEST(S) #
def test_validate_spec() -> None:
    query_str = """
{
    validateSpec(spec: {BoundedLine: {axis: "x", lower: 0, upper: 1, num: 5}})
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "validateSpec": {"Line": {"axis": "x", "start": 0.1, "stop": 0.9, "num": 5}}
    }


# GET POINTS QUERY TEST(S) #
def test_get_points_axis() -> None:
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {axis: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {axis: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      axis
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"axes": [{"axis": "x"}, {"axis": "y"}]}
    }


def test_get_points_lower() -> None:
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {axis: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {axis: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      lower{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"lower": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"lower": {"floatList": [-0.25, 0.25, 0.75, -0.25, 0.25, 0.75]}},
            ]
        }
    }


def test_get_points_midpoints() -> None:
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {axis: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {axis: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      midpoints{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"midpoints": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"midpoints": {"floatList": [0, 0.5, 1, 0, 0.5, 1]}},
            ]
        }
    }


def test_get_points_upper() -> None:
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {axis: "x", start: 0, stop: 1, num: 2}},
  inner: {Line: {axis: "y", start: 0, stop: 1, num: 3}}}}) {
    axes {
      upper{
        floatList
      }
    }
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {
            "axes": [
                {"upper": {"floatList": [0, 0, 0, 1, 1, 1]}},
                {"upper": {"floatList": [0.25, 0.75, 1.25, 0.25, 0.75, 1.25]}},
            ]
        }
    }


def test_get_points_numPoints() -> None:
    query_str = """
{
  getPoints(spec: {Product: {outer: {Line: {axis: "x", start: 0, stop: 1, num: 2}}
  inner: {Line: {axis: "y", start: 0, stop: 1, num: 3}}}}) {
    numPoints
  }
}
    """
    assert graphql.graphql_sync(schema, query_str).data == {
        "getPoints": {"numPoints": 6}
    }


# SCHEMA TEST(S)
def test_schema() -> None:
    assert_schema(schema)


def test_schema_text() -> None:
    with mock.patch("graphql.utilities.print_schema") as mock_print_schema:
        schema_text()
        mock_print_schema.assert_called()
