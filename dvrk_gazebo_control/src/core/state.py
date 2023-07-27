from isaacgym import gymtorch # Needed before torch import

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

from std_msgs.msg import ColorRGBA

from ll4ma_isaacgym.core import ObjectConfig


@dataclass
class EnvironmentState:
    dt: float = None
    timestep: int = 0
    joint_position: torch.Tensor = None
    joint_velocity: torch.Tensor = None
    joint_torque: torch.Tensor = None
    joint_names: List[str] = field(default_factory=list)
    n_arm_joints: int = 0
    n_ee_joints: int = 0
    ee_state: torch.Tensor = None
    objects: Dict[str, ObjectConfig] = field(default_factory=dict)
    object_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    object_colors: Dict[str, ColorRGBA] = field(default_factory=dict)
    prev_action: Dict = field(default_factory=dict)
    rgb: np.ndarray = None
    depth: np.ndarray = None
    goal: torch.Tensor = None
