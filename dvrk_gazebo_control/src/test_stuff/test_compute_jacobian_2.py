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
# np.random.seed(10000)


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
    num_joints = len(Ts) - 1
    t_eef = Ts[-1][:3,3]
    J = np.zeros((6, num_joints))
    z_unit = np.array([[0, 0, 1]]).T

    for i in range(1, num_joints+1):
        t_i = Ts[i][:3,3]
        d_i_n = t_eef - t_i
        z_i = Ts[i][:3, 2]
        # print("t_i, z_i:", i, t_i, z_i)

        # J[:3, i-1] = np.cross(z_i, d_i_n)
        J[:3, i-1] = np.cross(z_i, d_i_n)
        J[3:, i-1] = z_i
        if i == 1:
            J[:, i-1] *= -1

    # for i in range(0, num_joints):
    #     t_i = Ts[i][:3,3]
    #     d_i_n = t_eef - t_i
    #     z_i = Ts[i][:3, 2]
    #     print(np.dot(Ts[i][:3,:3], z_unit).shape,  d_i_n.shape)

    #     J[:3, i] = np.cross(np.dot(Ts[i][:3,:3], z_unit)[:,0], d_i_n)
    #     # np.cross(Ts[i][:3,:3], np.cross(z_unit.T, d_i_n))

    #     # Compute the angular component of the Jacobian
    #     J[3:, i] = np.dot(Ts[i][:3,:3], z_unit)[:,0]

    return J



# base_pose = ((0., -0.99999994, 0.25), (0., 0., 0., 1.))
# eef_pose = ((-0.02470973, -0.5895776, 0.03500031), (-4.5901083e-06, 0.70713687, 0.70707667, -8.158427e-05))
# q_cur = [6.7146473e-02,  2.2263610e-01,  3.6187898e-02, -2.7152264e-01,
#   2.3741621e-01,  2.2828581e-05, -1.2791561e-02, -6.7035034e-02]
# q_cur = np.array([0,0,0,0,0.1,0,0,0])
# q_cur = np.zeros(4)
q_cur = np.random.uniform(low=-0.5,high=0.5,size=8)

# homo_mat = construct_homo_mat(base_pose, eef_pose)

# print("Homogeneous transformation matrix:")
# print(np.round(homo_mat, decimals=2))

# jacobian = compute_jacobian(homo_mat)

# print("Jacobian matrix:")
# print(np.round(jacobian, decimals=2))

# print("============xxxxxxxxxxxxx")


robot = URDF.from_parameter_server()

base_link = 'psm_base_link'
end_link = "psm_tool_yaw_link" #'psm_tool_yaw_link'
kdl_kin = KDLKinematics(robot, base_link, end_link)

homo_mat_gt = np.array(kdl_kin.forward(q_cur)) 
jacobian_gt = np.array(kdl_kin.jacobian(q_cur)) 




homo_mats = []
link_names = kdl_kin.get_link_names()
print("link_names:", link_names)
print("joint names:", kdl_kin.get_joint_names(), len(kdl_kin.get_joint_names()))

for i in range(0, len(link_names)):
    link = link_names[i]
    # print(link)
    homo_mat = np.array(kdl_kin.forward(q_cur, end_link=link))  #
    homo_mats.append(homo_mat)
    print(np.round(homo_mat, decimals=2))
    print("===========")

# print("len(homo_mats):", len(homo_mats))

jacobian_test = compute_jacobian(homo_mats)

# print("Homogeneous transformation matrix:")
# print(np.round(homo_mat_gt, decimals=2))


print("Jacobian matrix:")
print(np.round(jacobian_gt, decimals=2))

print("Jacobian matrix test:")
print(np.round(jacobian_test, decimals=2))

print("Correct Jacobian computation:", all((np.round(jacobian_gt, decimals=5) == np.round(jacobian_test, decimals=5)).flatten()))
