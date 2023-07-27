#!/usr/bin/env python



import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np

from copy import copy, deepcopy
import rospy
from dvrk_gazebo_control.srv import *
import transformations
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics


rospy.init_node('Jacobian')

def get_pykdl_client(q_cur):
    '''
    get Jacobian matrix
    '''
    try:
        pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
        pykdl_request = PyKDLRequest()
        pykdl_request.q_cur = q_cur
        pykdl_response = pykdl_proxy(pykdl_request) 
    
    except(rospy.ServiceException, e):
        rospy.loginfo('Service get_pykdl call failed: %s'%e)
    # rospy.loginfo('Service get_pykdl is executed.')    
    
    return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape))    


# def construct_homo_mat(p_base, p_end_effector, R_base, R_end_effector):

def construct_homo_mat(base_pose, eef_pose):

    p_base = np.array(list(base_pose[0]))
    R_base = transformations.quaternion_matrix(list(base_pose[1]))[:3,:3]
    
    p_end_effector = np.array(list(eef_pose[0]))
    R_end_effector = transformations.quaternion_matrix(list(eef_pose[1]))[:3,:3]

    # Compute the translation vector
    t = p_end_effector - p_base

    # Compute the rotation matrix
    # R = R_end_effector @ R_base.T
    R = np.dot(R_end_effector, R_base.T)

    # Assemble the homogeneous transformation matrix
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1

    return T


def compute_jacobian(Ts):
    """ 
    Compute Jacobian matrix using the equation at 4:31 of this Youtube video: https://www.youtube.com/watch?v=C6Zho88S8vY
    """
    num_joints = 8
    t_eef = Ts[-1][:3,3]
    J = np.zeros((6, num_joints))

    for i in range(num_joints-1):
        t_i = Ts[i][:3,3]
        d_i_n = t_eef - t_i
        z_i = Ts[i][:3, 2]
        J[:3, i] = np.cross(z_i, d_i_n)
        J[3:, i] = z_i

    return J


def compute_jacobian(T):
    num_joints = 8

    # Extract the rotation matrix from the homogeneous transformation matrix
    R = T[:3, :3]

    # Initialize the Jacobian matrix
    J = np.zeros((6, num_joints))

    # Compute the linear and angular components of the Jacobian matrix
    for i in range(num_joints):
        # Extract the position vector of the current joint
        p = T[:3, 3]

        # Compute the linear component of the Jacobian
        # J[:3, i] = np.cross(R @ p, np.array([0, 0, 1]))
        J[:3, i] = np.cross(np.dot(R,p), np.array([0, 0, 1]))

        # Compute the angular component of the Jacobian
        J[3:, i] = R[:, 2]

    return J

# # Example usage
# p_base = np.array([1, 2, 3])  # Base frame position
# p_end_effector = np.array([4, 5, 6])  # End-effector frame position
# R_base = np.eye(3)  # Base frame orientation (identity matrix)
# R_end_effector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # End-effector frame orientation

# homo_mat = construct_homo_mat(p_base, p_end_effector, R_base, R_end_effector)

base_pose = ((0., -0.99999994, 0.25), (0., 0., 0., 1.))
eef_pose = ((-0.02470973, -0.5895776, 0.03500031), (-4.5901083e-06, 0.70713687, 0.70707667, -8.158427e-05))
q_cur = [6.7146473e-02,  2.2263610e-01,  3.6187898e-02, -2.7152264e-01,
  2.3741621e-01,  2.2828581e-05, -1.2791561e-02, -6.7035034e-02]

homo_mat = construct_homo_mat(base_pose, eef_pose)

print("Homogeneous transformation matrix:")
print(np.round(homo_mat, decimals=2))

jacobian = compute_jacobian(homo_mat)

print("Jacobian matrix:")
print(np.round(jacobian, decimals=2))

print("============xxxxxxxxxxxxx")
# jac_ground_truth = get_pykdl_client(q_cur)
# print("Jacobian matrix ground_truth:")
# print(np.round(jac_ground_truth, decimals=2))

robot = URDF.from_parameter_server()
base_link = 'psm_base_link'
end_link = 'psm_tool_yaw_link'
kdl_kin = KDLKinematics(robot, base_link, end_link)

homo_mat_gt = np.array(kdl_kin.forward(q_cur)) 
jacobian_gt = np.array(kdl_kin.jacobian(q_cur)) 
jacobian_test = compute_jacobian(homo_mat_gt)

print("Homogeneous transformation matrix:")
print(np.round(homo_mat_gt, decimals=2))


print("Jacobian matrix:")
print(np.round(jacobian_gt, decimals=2))

print("Jacobian matrix test:")
print(np.round(jacobian_test, decimals=2))