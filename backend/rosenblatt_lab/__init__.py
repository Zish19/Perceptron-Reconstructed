"""Rosenblatt Perceptron Lab.

A research-oriented implementation inspired by Rosenblatt's 1958 model. The
package deliberately labels modeling choices so it is not mistaken for a
complete reproduction of every equation in the original paper.
"""

from .model import ActivationTrace, PerceptronConfig, RosenblattPerceptron
from .experiments import ExperimentResult, run_experiment

__all__ = [
    "ActivationTrace",
    "ExperimentResult",
    "PerceptronConfig",
    "RosenblattPerceptron",
    "run_experiment",
]
