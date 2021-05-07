import logging
import string

import click

# Need this so we can eval() below
from .regions import *  # noqa
from .specs import *  # noqa


@click.group(invoke_without_command=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False
    ),
)
@click.version_option()
@click.pass_context
def cli(ctx, log_level: str):
    """ScanSpec library command line interface."""

    level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(format="%(levelname)s:%(message)s", level=level)

    # if no command is supplied, print the help message
    if ctx.invoked_subcommand is None:
        click.echo(cli.get_help(ctx))


@cli.command()
@click.argument("spec")
def plot(spec: str):
    """Plot a ScanSpec"""
    from scanspec.plot import plot_spec

    for letter in string.ascii_lowercase:
        locals()[letter] = letter
    eval_spec = eval(spec)
    plot_spec(eval_spec)


@cli.command()
@click.option("--cors", is_flag=True)
def service(cors):
    """Run up a GraphQL service"""
    from scanspec.service import run_app

    run_app(cors)


@cli.command()
def schema():
    """Print schema"""
    from scanspec.service import schema_text

    click.echo(schema_text())
