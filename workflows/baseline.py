"""Deprecated compatibility alias for the former V1 one-pass baseline.

V5 registers ``workflows.one_pass`` as Variant 1 and reserves
``agentic_baseline`` for the parser/retriever/creator design in Variant 2.
This module remains importable for historical scripts but is not registered.
"""

from workflows.one_pass import run

NAME = "baseline"
DISPLAY_NAME = "Legacy V1 baseline alias"
STATUS = "legacy"
