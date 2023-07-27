import numpy as np
import torch

class Action:
    """
    Generic action interface. It's unlikely you'll use this directly, see
    extending classes below.
    """

    def __init__(self, joint_pos=None):
        """
        Args:
            joint_pos (ndarray): Joint position to set initially
        """
        self.set_joint_position(joint_pos)

    def set_joint_position(self, joint_pos):
        """
        Set joint position action.

        Args:
            joint_pos (ndarray): Joint position action to set
        """
        self.joint_pos = joint_pos

    def get_joint_position(self):
        """
        Returns current set joint position
        """
        return self.joint_pos.copy() if self.joint_pos is not None else None


class ArmAction(Action):
    """
    Arm action interface.

    TODO currently nothing arm-specific that needs done, but can add e.g.
    checking joint limits.
    """
    ...


class EndEffectorAction(Action):
    """
    End-effector action interface. This handles different action modes,
    currently supported:
        - Joint position: directly set raw joint position of fingers
        - Discrete: set discrete actions. Currently supported:
            - open: Open fingers to joint angle specified in EE config
            - close: Close fingers to joint angle specified in EE config
        - Same-angle: Command all fingers to same joint position

    When action modes besides joint position is used, the function
    self.update_action_joint_pos is used to compute the joint position
    in terms of the other action mode being used and the EE config.

    TODO: can support other low-level command interfaces to Isaac Gym such
          as velocity or torque, but we're currently only utilizing position.
    TODO: can add other discrete actions and other action modes entirely
          as needed.
    """
    def __init__(self, joint_pos=None, discrete=None, same_angle=None):
        """
        Args:
            joint_pos (ndarray): Joint position to initialize command
            discrete (str): String name of discrete action to initialize with
            same_angle (float): Joint angle to initialize same-angle command to
        """
        super().__init__(joint_pos)
        self.set_discrete(discrete)
        self.set_same_angle(same_angle)

    def set_discrete(self, discrete):
        self.discrete = discrete

    def set_same_angle(self, angle):
        self.same_angle = angle

    def get_discrete(self):
        return self.discrete

    def get_same_angle(self):
        return self.same_angle

    def has_discrete(self):
        return self.discrete is not None

    def has_same_angle(self):
        return self.same_angle is not None

    def update_action_joint_pos(self, config):
        """
        Computes low-level joint position command to send to Isaac Gym in terms of
        other action modes that are set. Currently supports discrete actions
        (e.g. open/close fingers) and same-angle (i.e. set all fingers to the same
        angle).

        The action modes are mutually exclusive, so you can't for example use both
        same-angle and discrete modes. If the other action modes are not being used,
        this will be a pass-through function that does not modify the joint position
        command already set.

        Note the joint position updates are performed in-place.

        Args:
            config (EndEffectorConfig): Config object of EE so ee-specific functionality
                                        can be provided
        """
        if self.has_discrete() and self.has_same_angle():
            raise RuntimeError("Cannot set both discrete and same-angle actions")

        idxs = config.grip_finger_indices
        if self.has_discrete():
            if self.discrete == 'close':
                # print(config.close_finger_joint_pos)
                self.joint_pos[idxs] = config.close_finger_joint_pos
            elif self.discrete == 'open':
                self.joint_pos[idxs] = config.open_finger_joint_pos
            else:
                raise ValueError(f"Unknown discrete action for EE: {self.discrete}")
        elif self.has_same_angle():
            self.joint_pos[idxs] = [self.same_angle] * len(idxs)


class RobotAction:
    """
    Robot action interface. This provides a single interface to engage
    with both the arm and EE action interfaces.
    """

    def __init__(self, arm_joint_pos=None, ee_joint_pos=None):
        self.arm = ArmAction()
        self.end_effector = EndEffectorAction()
        self.set_joint_position(arm_joint_pos, ee_joint_pos)

    def set_joint_position(self, arm_joint_pos=None, ee_joint_pos=None):
        self.set_arm_joint_position(arm_joint_pos)
        self.set_ee_joint_position(ee_joint_pos)

    def set_arm_joint_position(self, joint_pos):
        self.arm.set_joint_position(joint_pos)

    def set_ee_joint_position(self, joint_pos):
        self.end_effector.set_joint_position(joint_pos)

    def set_ee_discrete(self, discrete):
        self.end_effector.set_discrete(discrete)

    def set_ee_same_angle(self, angle):
        self.end_effector.set_same_angle(angle)

    def get_joint_position(self):
        """
        Returns the combined arm and joint positions, useful for computing the
        final joint position commands that will be sent to Isaac Gym.

        If there is no end-effector action then only the arm joints will be returned.
        """
        arm_joints = self.get_arm_joint_position()
        ee_joints = self.get_ee_joint_position()
        joint_pos = arm_joints if ee_joints is None else np.concatenate([arm_joints, ee_joints])
        return joint_pos

    def get_arm_joint_position(self):
        return self.arm.get_joint_position()

    def get_ee_joint_position(self):
        return self.end_effector.get_joint_position()

    def get_ee_discrete(self):
        return self.end_effector.get_discrete()

    def get_ee_same_angle(self):
        return self.end_effector.get_same_angle()

    def has_ee_discrete(self):
        return self.end_effector.has_discrete()

    def has_ee_same_angle(self):
        return self.end_effector.has_same_angle()

    def get_state_tensor(self):
        """
        Returns tensor of arm joint angles combined with discrete EE action.
        """
        joint = self.get_arm_joint_position()
        tensor = torch.zeros((joint.shape[0] + 1))
        tensor[:-1] += joint
        if self.has_ee_discrete() and self.end_effector.discrete == 'close':
            tensor[-1] = 1
        return tensor

    def clear_ee_discrete(self):
        self.set_ee_discrete(None)

    def clear_ee_same_angle(self):
        self.set_ee_same_angle(None)
