"""Load this trusted plugin before opening setup_project.json."""
from opencv_dataflow.simple_gdrnpp.dataflow import register

__all__ = ["register"]
