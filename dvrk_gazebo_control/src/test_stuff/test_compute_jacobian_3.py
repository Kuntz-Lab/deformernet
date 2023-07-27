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
np.random.seed(10000)
sys.float_repr_style = 'plain'
np.set_printoptions(suppress=True, precision=5)

rospy.init_node('Jacobian')



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

        if i != 5:  # revolute joints
            J[:3, i-1] = np.cross(z_i, d_i_n)
            J[3:, i-1] = z_i
        else:  # prismatic joint (main insertion joint)
            J[:3, i-1] = z_i
            J[3:, i-1] = 0           
            
            
        if i in [1,7]:  # these joints have to multiple by -1 for some weird reasons
            J[:, i-1] *= -1

    return J



base_pose = ((0., -0.99999994, 0.25), (0., 0., 0., 1.))
eef_pose = ((-6.869061e-06, -0.61335856, 0.14226243), (0.00010981, 0.70698094, 0.70723253, 0.0001132))
q_cur = [1.6437267e-06, -0.00017067791, -0.00018930438, -4.768372e-06, 0.23050272, 2.9206276e-06, -7.712044e-06, -7.2717676e-06]
# q_cur = np.array([0,0,0,0,0.1,0,0,0])
# q_cur = np.zeros(4)
# q_cur = np.random.uniform(low=-3,high=3,size=8)


robot = URDF.from_parameter_server()
base_link = "world" #'psm_base_link'
end_link = "psm_tool_yaw_link" #'psm_tool_yaw_link'
kdl_kin = KDLKinematics(robot, base_link, end_link)

# homo_mat = construct_homo_mat(base_pose, eef_pose)

# print("Homogeneous transformation matrix:")
# print(np.round(homo_mat, decimals=2))

# homo_mat_gt = np.array(kdl_kin.forward(q_cur)) 








jacobian_gt = np.array(kdl_kin.jacobian(q_cur)) 


homo_mats = []
link_names = kdl_kin.get_link_names()
print("link_names:", link_names)
print("joint names:", kdl_kin.get_joint_names(), len(kdl_kin.get_joint_names()))

for i in range(0, len(link_names)):
    
    if i == 1:  # if use world as the base link, not psm_base_link
        continue
    
    link = link_names[i]
    # print(link)
    homo_mat = np.array(kdl_kin.forward(q_cur, end_link=link))  # 'psm_base_link'
    homo_mats.append(homo_mat)
    print(np.round(homo_mat, decimals=2))
    print("===========")



jacobian_test = compute_jacobian(homo_mats)




print("Jacobian matrix:")
print(np.round(jacobian_gt, decimals=5))

print("Jacobian matrix test:")
print(np.round(jacobian_test, decimals=5))



print("Correct Jacobian computation:", all((np.round(jacobian_gt, decimals=5) == np.round(jacobian_test, decimals=5)).flatten()))
print("(Approx) Correct?:", np.allclose(jacobian_gt, jacobian_test, atol=1e-4))
print("Correct Jacobian computation:")
print(np.round(jacobian_gt, decimals=5) == np.round(jacobian_test, decimals=5))