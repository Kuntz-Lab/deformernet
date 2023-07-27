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

from dvrk_gazebo_control.srv import *
import rospy

two_robot_offset = 1.0

class TaskVelocityControl(Behavior):
    '''
    Task velocity control.
    '''

    def __init__(self, delta_xyz, robot, dt, traj_duration, vel_limits=None, init_pose=None, \
                        error_threshold = 1e-3, open_gripper=True, second_robot=True):
        super().__init__()

        self.name = "task velocity control"
        self.robot = robot
        self.dt = dt
        self.traj_duration = traj_duration
        self.action = RobotAction()
        self.open_gripper = open_gripper
        self.err_thres = error_threshold
        self.dq = 10**-5 * np.ones(self.robot.n_arm_dof)
        self.init_pose = init_pose
        self.vel_limits = vel_limits
        self.second_robot = second_robot


        # self._plan = None
        # self._trajectory = None

        self.set_target_pose(delta_xyz)


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

        if self.is_not_started():
            # rospy.loginfo(f"Running behavior: {self.name}")
            self.set_in_progress()
            # self.set_policy()

        ee_cartesian_pos = self.robot.get_ee_cartesian_position()
        if self.second_robot:
            ee_cartesian_pos[:2] = -ee_cartesian_pos[:2]
        else:
            ee_cartesian_pos[1] += two_robot_offset
        ee_cartesian_pos[2] -= 0.25

        delta_ee = self.target_pose[:6] - ee_cartesian_pos[:6]
        delta_ee[3:6] = 0
        # rospy.loginfo("current delta x,y,z: " + str(delta_ee[:3]))
        # rospy.loginfo("target: " + str(self.target_pose[:3]))
        # rospy.loginfo("current: " + str(ee_cartesian_pos[:3]))
        # rospy.loginfo("============================")
        
        if np.any(abs(delta_ee) > self.err_thres):
            q_cur = self.robot.get_arm_joint_positions()
            J = self.get_pykdl_client(q_cur)
            J_pinv = self.damped_pinv(J)
            q_vel = np.matmul(J_pinv, delta_ee)
            # q_vel = np.array(q_vel)[0]
            # q_vel = self.null_space_projection(q_cur, q_vel, J, J_pinv)

            # delta_q = q_vel * self.dt
            # desired_q_pos = np.copy(q_cur) + delta_q
            desired_q_vel = q_vel * 4#2#6
            # print("desired_q_vel:", desired_q_vel.shape)
            if self.vel_limits is not None:
                exceeding_ratios = abs(np.divide(desired_q_vel, self.vel_limits[:8]))
                if np.any(exceeding_ratios > 1.0):
                    scale_factor = max(exceeding_ratios)
                    desired_q_vel /= scale_factor
            self.action.set_arm_joint_position(np.array(desired_q_vel, dtype=np.float32))
            return self.action

        else:
            self.set_success()
            return None

    def set_target_pose(self, delta_xyz):
        """
        Sets target end-effector pose that motion planner will generate plan for.

        Args:
            pose (list-like): Target pose as 7D vector (3D position and quaternion)
        
        Input pose can be a list, numpy array, or torch tensor.
        """
        if self.init_pose is not None:
            pose = deepcopy(self.init_pose)
        else:
            pose = self.robot.get_ee_cartesian_position()
        if self.second_robot:
            pose[:2] = -pose[:2]
        else:
            pose[1] += two_robot_offset
        pose[2] -= 0.25
        pose[:3] += np.array(delta_xyz) 
        self.target_pose = pose

    def damped_pinv(self, A, rho=0.017):
        AA_T = np.dot(A, A.T)
        damping = np.eye(A.shape[0]) * rho**2
        inv = np.linalg.inv(AA_T + damping)
        d_pinv = np.dot(A.T, inv)
        return d_pinv

    def null_space_projection(self, q_cur, q_vel, J, J_pinv):
        identity = np.identity(self.robot.n_arm_dof)
        q_vel_null = \
            self.compute_redundancy_manipulability_resolution(q_cur, q_vel, J)
        q_vel_constraint = np.array(np.matmul((
            identity - np.matmul(J_pinv, J)), q_vel_null))[0]
        q_vel_proj = q_vel + q_vel_constraint
        return q_vel_proj    

    def compute_redundancy_manipulability_resolution(self, q_cur, q_vel, J):
        m_score = self.compute_manipulability_score(J)
        J_prime = self.get_pykdl_client(q_cur + self.dq)
        m_score_prime = self.compute_manipulability_score(J_prime)
        q_vel_null = (m_score_prime - m_score) / self.dq
        return q_vel_null

    def compute_manipulability_score(self, J):
        return np.sqrt(np.linalg.det(np.matmul(J, J.transpose())))    

    def get_pykdl_client(self, q_cur):
        '''
        get Jacobian matrix
        '''
        # rospy.loginfo('Waiting for service get_pykdl.')
        # rospy.wait_for_service('get_pykdl')
        # rospy.loginfo('Calling service get_pykdl.')
        try:
            pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
            pykdl_request = PyKDLRequest()
            pykdl_request.q_cur = q_cur
            pykdl_response = pykdl_proxy(pykdl_request) 
        
        except(rospy.ServiceException, e):
            rospy.loginfo('Service get_pykdl call failed: %s'%e)
        # rospy.loginfo('Service get_pykdl is executed.')    
        
        return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape))    

