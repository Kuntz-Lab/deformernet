import os
import sys
import numpy as np
from isaacgym import gymapi
import torch
import rospy
import random
from copy import deepcopy

import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from behaviors import Behavior
from utils import ros_util, math_util

from core import RobotAction

# TODO: sys, float32, attach end-effector to target pos, fix set_policy, utils


class MoveToPose(Behavior):
    """
    Behavior to move robot end-effector to a desired pose.

    A motion-planned trajectory (joint space or Cartesian) is generated that will
    move the end-effector from its current pose to the desired pose.

    This supports setting a single target pose as well as multiple candidate target
    poses so that if a plan cannot be generated for a particular candidate, it will
    continue to randomly sample candidates until a plan for one of the is found or
    no plan can be found for any of the candidates.
    """

    def __init__(self, target_pose, robot, dt, traj_duration, open_gripper=True):
        super().__init__()

        self.name = "move_to_pose"
        self.robot = robot
        self.dt = dt
        self.traj_duration = traj_duration
        self.action = RobotAction()
        self.open_gripper = open_gripper

        self._plan = None
        self._trajectory = None

        self.set_target_pose(target_pose)

        # rospy.loginfo(f"Running behavior: {self.name}")
        self.set_policy()


    def get_action(self):
        """
        Returns the next action from the motion-planned trajectory.

        Args:
            state (EnvironmentState): Current state from simulator
        Returns:
            action (Action): Action to be applied next in the simulator

        TODO populate joint velocity and set on action. The action interface currently
        only supports position commands so that's all we're commanding here.
        """

        # if self.is_not_started() and self._plan is None:
        #     rospy.loginfo(f"Running behavior: {self.name}")
        #     self.set_policy()

        if self.is_complete():
            return None

        # action = state.prev_action
        if self._plan is not None:
            if len(self._trajectory.points) > 0:
                if self.open_gripper:
                    target_pos = list(self._trajectory.points.pop(0).positions) + [1.5,0.8]
                    self.action.set_arm_joint_position(np.array(target_pos, dtype=np.float32))
                else:
                    target_pos = list(self._trajectory.points.pop(0).positions) + [0.35,-0.35]
                    self.action.set_arm_joint_position(np.array(target_pos, dtype=np.float32))      
                                  
            if len(self._trajectory.points) == 0:
                self.set_success()

                ee_state = self.robot.get_ee_cartesian_position()
                target_pos = torch.from_numpy(self.target_pose[:3]).unsqueeze(0)
                actual_pos = torch.from_numpy(ee_state[:3]).unsqueeze(0)
                target_rot = torch.from_numpy(self.target_pose[3:]).unsqueeze(0)
                actual_rot = torch.from_numpy(ee_state[3:7]).unsqueeze(0)
                pos_error = math_util.position_error(target_pos, actual_pos, True, flatten=True)
                rot_error = math_util.quaternion_error(target_rot, actual_rot, True, flatten=True)
                if torch.any(pos_error > 1e-3) or torch.any(rot_error > 1e-3):
                    self.set_failure()
        return self.action

    def set_target_pose(self, pose):
        """
        Sets target end-effector pose that motion planner will generate plan for.

        Args:
            pose (list-like): Target pose as 7D vector (3D position and quaternion)
        
        Input pose can be a list, numpy array, or torch tensor.
        """
        if isinstance(pose, list):
            self.target_pose = np.array(pose)
        elif isinstance(pose, np.ndarray):
            self.target_pose = pose
        else:
            raise ValueError(f"Unknown data type for setting target pose: {type(pose)}")


        
 

    def set_policy(self):
        """
        Sets policy (in this case a motion-planned trajectory) given the current state.

        Used both as a helper function on this class as well as allowing parent 
        behaviors to force setting the plan before an action is requested (e.g. 
        generating open loop sequences of behaviors).
        """        


        target_pose = ros_util.convert_list_to_Pose(self.target_pose)
        plan, success = self.robot.arm_moveit_planner_client(go_home=False, cartesian_goal=target_pose, current_position=self.robot.get_full_joint_positions())
        self._plan = plan

        if self._plan is not None and success:
            
            # Post-process to interpolate for the simulation timestep
            nominal_traj = self._plan
            self._trajectory = ros_util.interpolate_joint_trajectory(nominal_traj, self.dt, self.traj_duration)
            self.set_in_progress()

        else:
            rospy.logwarn("Plan not found for that target pose")
            self.target_pose = None
            self.set_failure()

    def override_state(self, state):
        """
        Overrides the state by setting the joint position and corresponding end-effector
        pose (as computed from FK) as the last point in the computed motion plan.
        """
        if self._trajectory is None:
            return
        joint_pos = torch.from_numpy(self._trajectory.points[-1].positions).unsqueeze(-1)
        state.joint_position[:7] = joint_pos
        pose = self.sim.forward_kinematics(state.joint_position.squeeze().unsqueeze(0)).squeeze()
        state.ee_state[:7] = pose
        



