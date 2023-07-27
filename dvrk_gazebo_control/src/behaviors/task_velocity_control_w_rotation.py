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
import transformations



class TaskVelocityControl2(Behavior):
    '''
    Task velocity control.
    '''

    def __init__(self, delta_xyz, robot, dt, traj_duration, vel_limits=None, init_pose=None, 
                pos_threshold = 3e-3, ori_threshold=1e-1, use_euler_target=True, open_gripper=True):
        super().__init__()

        self.name = "task velocity control"
        self.robot = robot
        self.dt = dt
        self.traj_duration = traj_duration
        self.action = RobotAction()
        self.open_gripper = open_gripper
        self.err_thres = [pos_threshold]*3 + [ori_threshold]*3
        # self.err_thres = 1e-3
        self.dq = 10**-5 * np.ones(self.robot.n_arm_dof)
        self.init_pose = init_pose
        self.vel_limits = vel_limits


        self.use_euler_target = use_euler_target
        self.set_target_pose(delta_xyz)


    def get_action(self, debug=False):
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

        # curr_ee_pose = self.robot.get_ee_cartesian_position()
        # delta_ee_quat = 100#self.target_pose - curr_ee_pose
        # delta_ee = np.hstack((delta_ee_quat[:3], self.compute_orientation_error(curr_ee_pose).T[0]))


        # curr_trans = self.convert_to_matrix(curr_ee_pose[:3], curr_ee_pose[3:])

        q_cur = self.robot.get_arm_joint_positions()
        J, curr_trans = self.get_pykdl_client(q_cur)
        # print("np.linalg.det(J):", np.linalg.det(J))
        
        # desired_trans = self.convert_to_matrix(self.target_pose[:3], self.target_pose[3:])
        desired_trans = self.target_pose

        delta_ee = self.computeError(curr_trans, desired_trans).flatten()

        # print("==============")
        # rospy.loginfo("target: " + str(transformations.euler_from_quaternion(self.target_pose[3:])))
        # rospy.loginfo("current: " + str(transformations.euler_from_quaternion(curr_ee_pose[3:])))
        # print("target:", self.target_pose[:3,3], transformations.euler_from_matrix(self.target_pose))
        # print("current: ", curr_trans[:3,3], transformations.euler_from_matrix(curr_trans))
        # # print("///")
        # # # print("err1:", delta_ee_quat)
        # print("err2:", delta_ee)
        # rospy.loginfo("target: " + str(self.target_pose[:3]))
        # rospy.loginfo("current: " + str(curr_ee_pose[:3]))

        # rospy.loginfo("target: " + str(self.target_pose[3:]))
        # rospy.loginfo("current: " + str(ee_cartesian_pos[3:]))
        # rospy.loginfo("current delta: " + str(delta_ee_quat))
        # rospy.loginfo("============================")
        
        # if all(abs(delta_ee[:3]) <= self.err_thres[:3]):
        #     rospy.logerr("DONE POSITION")
        # if all(abs(delta_ee[3:]) <= self.err_thres[3:]):
        #     rospy.logerr("DONE ORI")
                
        if np.any(abs(delta_ee) > self.err_thres):
            
            # q_cur = self.robot.get_arm_joint_positions()
            # print("q_cur:", q_cur)
            # J = self.get_pykdl_client(q_cur)
            J_pinv = self.damped_pinv(J)

            q_vel = np.matmul(J_pinv, delta_ee)
            # q_vel = np.array(q_vel)
            # q_vel = self.null_space_projection(q_cur, q_vel, J, J_pinv)

            # delta_q = q_vel * self.dt
            # desired_q_pos = np.copy(q_cur) + delta_q
            desired_q_vel = q_vel * 4


            # print("desired_q_vel:", desired_q_vel)
            if self.vel_limits is not None:
                exceeding_ratios = abs(np.divide(desired_q_vel, self.vel_limits[:self.robot.n_arm_dof]))
                if np.any(exceeding_ratios > 1.0):
                    scale_factor = max(exceeding_ratios)
                    desired_q_vel /= scale_factor
            self.action.set_arm_joint_position(np.array(desired_q_vel, dtype=np.float32))
            if debug:
                data = []
                data.extend(list(curr_trans[:3,3]))                
                data.extend(list(self.target_pose[:3,3]))
                data.extend(list(transformations.euler_from_matrix(curr_trans)))
                data.extend(list(transformations.euler_from_matrix(self.target_pose)))       
                data.extend(list(delta_ee)) 
                data.extend(list(desired_q_vel))           
                return self.action, data
            return self.action

        else:
            # print("===========!!!!!!!", delta_ee)
            self.set_success()
            return None

    def set_target_pose(self, delta):
        """
        Sets target end-effector pose that motion planner will generate plan for.

        Target pose is a 6D vector (3D position and euler angles for resolved-rate control)
        
        Input pose can be a list, numpy array, or torch tensor.
        """
        if self.init_pose is not None:
            pose = deepcopy(self.init_pose)
        else:
            # pose = self.robot.get_ee_cartesian_position(euler_format=True)
            # q_cur = self.robot.get_arm_joint_positions()
            # _, pose = self.get_pykdl_client(q_cur)
            # pose[:3, 3] += delta[:3]
            # # eulers = np.array(transformations.euler_from_matrix(pose))
            # eulers += delta[3:]

            if self.use_euler_target:
                pose = transformations.euler_matrix(*delta[3:])
                pose[:3, 3] = delta[:3]
            else:
                pose = np.eye(4)
                pose[:3,:3] = delta[3]
                pose[:3, 3] = delta[:3]                
            
        
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
        J_prime,_ = self.get_pykdl_client(q_cur + self.dq)
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
        # print("np.reshape(pykdl_response.ee_pose_flattened, (4,4)", np.reshape(pykdl_response.ee_pose_flattened, (4,4)).shape)
        return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape)), np.reshape(pykdl_response.ee_pose_flattened, (4,4))   

    def matrixLog(self, Ae):
        # if(np.abs(np.trace(Ae)-1)>2.0):
        #     Ae = np.copy(Ae)
        #     Ae[:,0] /= np.linalg.norm(Ae[:,0]) #unit vectorize because numerical issues
        #     Ae[:,1] /= np.linalg.norm(Ae[:,1])
        #     Ae[:,2] /= np.linalg.norm(Ae[:,2])
        # if np.trace(Ae)-1 == 0:
        # print("Ae:",Ae)
        
        # if abs(np.trace(Ae)+1) <= 0.0001:    
        #     phi = np.pi
        #     w = 1/np.sqrt(2*(1+Ae[2,2])) * np.array([Ae[0,2], Ae[1,2], 1+Ae[2,2]])
        #     skew = np.array([[0, -w[2], w[1]], 
        #                     [w[2], 0, -w[0]],
        #                     [-w[1], w[0], 0]])
        #     print("case 2 mat log")                
        # else:
        phi = np.arccos(0.5*(np.trace(Ae)-1))
        # print("phi:", phi)
        skew = (1/(2*np.sin(phi)))*(Ae-np.transpose(Ae))
        # print("trace:", np.trace(Ae))
        # print("phi:", phi)
        return skew,phi

    def computeError(self,currentTransform,desiredTransform):
        errorTransform = np.dot(desiredTransform, np.linalg.inv(currentTransform))
        
        linearError = errorTransform[:3,3:]
        linearError = desiredTransform[:3,3:] - currentTransform[:3,3:]
        # print(desiredTransform[:3,3:] - currentTransform[:3,3:])
        skew,theta = self.matrixLog(errorTransform[:3,:3])
        if(theta == 0.0):
            rotationError = np.zeros((3,1))
        else:
            w_hat = self.skewToVector(skew)
            rotationError = w_hat * theta
        
        G = 1/theta*np.eye(3) - 1/2*skew + (1/theta - 1/2*(1/np.tan(theta/2))) *(skew @ skew) 
        
        # return np.concatenate((rotationError, theta * G @ linearError))
        return np.concatenate((theta * G @ linearError, rotationError))


    def skewToVector(self, skew):
        w = np.zeros((3,1))
        w[0,0] = skew[2,1]
        w[1,0] = skew[0,2]
        w[2,0] = skew[1,0]
        return w

    def convert_to_matrix(self, p, quat):
        rot_mat = transformations.quaternion_matrix(quat)
        rot_mat[:3,3] = p
        return rot_mat


    def get_transform(self, curr_trans, desired_trans, get_twist=False):
        '''
        Returns:
        p: xyz displacement vector from initial to current pose
        R: 3x3 rotation matrix from initial to current pose
        '''

        # _, curr_trans = self.get_pykdl_client(q_cur) 

        p = desired_trans[:3,3:] - curr_trans[:3,3:]
        R = np.dot(desired_trans, np.linalg.inv(curr_trans))[:3,:3]
        
        if get_twist:
            twist = self.computeError(curr_trans, desired_trans)
            return p, R, twist

        return p, R



