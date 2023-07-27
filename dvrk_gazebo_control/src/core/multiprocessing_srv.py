from isaacgym import gymapi

import torch
from dataclasses import dataclass, field
from typing import List, Tuple

from ll4ma_isaacgym.core import EnvironmentState


@dataclass
class OfflineSimulatorRequest:
    fk_joint_pos: torch.Tensor = None
    state: EnvironmentState = None
    act_joint_pos: torch.Tensor = None
    contact_exclude_pairs: List[Tuple[str, str]] = field(default_factory=list)
    get_forward_kinematics: bool = False
    get_contacts: bool = False


@dataclass
class OfflineSimulatorResponse:
    """
    fk_poses: (n_steps, n_samples, 7)
    contacts: List of n_samples, each element a list of n_steps, each element a list of contacts.
    """
    fk_poses: torch.Tensor = None
    contacts: List[List[List[gymapi.RigidContact]]] = field(default_factory=list)
