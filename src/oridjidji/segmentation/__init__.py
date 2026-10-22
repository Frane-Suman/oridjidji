"""Sentence-window construction and span aggregation for the token-trace explanation."""

from oridjidji.segmentation.aggregate import SpanScore, aggregate_windows
from oridjidji.segmentation.windows import Window, build_windows

__all__ = ["SpanScore", "Window", "aggregate_windows", "build_windows"]
