.. _graphql-service:

Running a GraphQL service for generating points
===============================================

The `creating-a-spec` tutorial shows how you would use the commandline client to
plot a `Spec`. This maps nicely to using scanspec in a commanline utility, but
not if you want to expose those points to a web GUI. To do this we will bring up
a GraphQL_ service that allows a web GUI to request the points it would like to
plot.

Running the server
------------------

In a terminal, run::

    $ scanspec service --cors
    ======== Running on http://0.0.0.0:8080 ========
    (Press CTRL+C to quit)

You can now open a browser to http://0.0.0.0:8080/graphql and see a GraphiQL_ editor
which will allow you to send GraphQL queries to the server

.. note::

    All the examples below will embed a GraphiQL_ editor in your browser talking
    to the scanspec service, they are live and you can modify the queries and press
    the play button to get an updated response from the server. If the server is
    not running (or you didn't pass ``--cors`` to it) then the play button will not
    be visible.


Validating a Spec
-----------------

The first use case for our service is producing the canonical serialization of a
Spec. Some of the Specs like `Line.bounded` will return an instance of Line that
has different parameters to those that are passed.

For example, on the commandline we might do::

    >>> from scanspec.specs import Line
    >>> Line.bounded("x", 0, 1, 5).serialize()
    {'Line': {'axis': 'x', 'start': 0.1, 'stop': 0.9, 'num': 5}}

.. seealso:: `serialize-a-spec`

The equivalent in our service is:

.. graphiql:: http://localhost:8080/graphql
    :query:
      {
        validateSpec(
          spec: {
            BoundedLine: {
              axis: "x"
              lower: 0
              upper: 1
              num: 5
            }
          }
        )
      }
    :response:
      {
        "data": {
          "validateSpec": {
            "Line": {
              "axis": "x",
              "start": 0.1,
              "stop": 0.9,
              "num": 5
            }
          }
        }
      }

Getting Points from a Spec
--------------------------

Most importantly, is the ability to obtain a list of scan points from a Spec. 
GraphQL gives the user the ability to request one or more fields from an object, 
allowing them to obtain data that is relevant only to their application.

The 'getPoints' query makes use of this, giving users the ability to select from
one or more of the following fields:

- numFrames: the total number of frames produced by the Spec
- returnedFrames (WIP): the number of frames returned, limited by the maxPoint arguement
- axes: a list of axes present in the Spec and its associated scan points

Within axes:
- axis: a list of the axes present in the Spec
- lower: a list of lower bounds that are each present in a frame
- midpoints: a list of midpoints that are each present in a frame
- upper: a list of upper bounds that are each present in a frame

Within lower, middle and upper:
- string: returns the requested points as a string
- floatList: returns the requested points as a list of floatList
- b64: returns the requested points encoded into base64

Using the example above, we can request to return points from it:

.. graphiql:: http://localhost:8080/graphql
    :query:
      {
        getPoints(
          spec: {
            BoundedLine: {
              axis: "x"
              lower: 0
              upper: 1
              num: 5
            }
          }
        )
        {
          numPoints
          axes {
            axis
            upper {
              string
            }
            midpoints{
              floatList
            }
            lower{
              b64
            }
          }
        }
      }
    :response:
      {
        "data": {
          "getPoints": {
            "numPoints": 5,
            "axes": [
              {
                "axis": "x",
                "upper": {
                  "string": "[0.2 0.4 0.6 0.8 1. ]"
                },
                "midpoints": {
                  "floatList": [
                    0.1,
                    0.30000000000000004,
                    0.5,
                    0.7000000000000001,
                    0.9
                  ]
                },
                "lower": {
                  "b64": "AAAAAAAAAACamZmZmZnJP5qZmZmZmdk/NDMzMzMz4z+amZmZmZnpPw=="
                }
              }
            ]
          }
        }
      }


Content to move
---------------

When we move this sphinx extension into its own repo we will use the following to demo it:

.. graphiql:: https://countries.trevorblades.com/
    :query:
      {
        country(code: "BR") {
          name
          native
          capital
          emoji
          currency
          languages {
            code
            name
          }
        }
      }
    :response:
      {
        "data": {
          "country": {
            "name": "Brazil",
            "native": "Brasil",
            "capital": "Brasília",
            "emoji": "🇧🇷",
            "currency": "BRL",
            "languages": [
              {
                "code": "pt",
                "name": "Portuguese"
              }
            ]
          }
        }
      }


.. graphiql::
    :query:
      {
        country(code: "BR") {
          name
          native
          capital
          emoji
          currency
          languages {
            code
            name
          }
        }
      }
    :response:
      {
        "data": {
          "country": {
            "name": "Brazil",
            "native": "Brasil",
            "capital": "Brasília",
            "emoji": "🇧🇷",
            "currency": "BRL",
            "languages": [
              {
                "code": "pt",
                "name": "Portuguese"
              }
            ]
          }
        }
      }

.. _GraphQL: https://www.graphql.com/
.. _GraphiQL: https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme