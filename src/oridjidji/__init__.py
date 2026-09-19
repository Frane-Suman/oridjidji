"""Oriđiđi — estimates the probability that a Croatian text was machine-generated."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oridjidji")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0.dev0"
