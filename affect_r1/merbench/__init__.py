"""
Utility modules for running MER-UniBench style inference and evaluation
within the HumanOmniV2/affect_r1 workflow.
"""

from .result_writer import MerBenchResultWriter, parse_think_answer  # noqa: F401
from .json_loader import iter_prediction_rows  # noqa: F401

