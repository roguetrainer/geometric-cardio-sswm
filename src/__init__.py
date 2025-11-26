"""
Geometric Cardio SSWM: Self-Supervised World Models for Cardiac Mechanics

A framework for learning predictive models of cardiac dynamics using geometric priors.
"""

__version__ = "0.1.0"
__author__ = "Geometric SSWM Contributors"

from .encoder import SSWMEncoder
from .dynamics_model import SSWMDynamicsModel
from .decoder import SSWMDecoder
from .sswm_model import SynthCardioSSWM
from .data_generator import SynthCardioDataGenerator
from .geometric_utils import (
    compute_strain_invariants,
    fiber_strain,
    rotation_matrix_from_axis_angle,
    geodesic_distance
)

__all__ = [
    "SSWMEncoder",
    "SSWMDynamicsModel",
    "SSWMDecoder",
    "SynthCardioSSWM",
    "SynthCardioDataGenerator",
    "compute_strain_invariants",
    "fiber_strain",
    "rotation_matrix_from_axis_angle",
    "geodesic_distance",
]
