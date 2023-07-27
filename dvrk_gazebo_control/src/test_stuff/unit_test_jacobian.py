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
np.random.seed(0)

rospy.init_node('Jacobian')


def compute_jacobian(Ts):
    """ 
    Compute Jacobian matrix using the equation at 4:31 of this Youtube video: https://www.youtube.com/watch?v=C6Zho88S8vY
    
    1. Input:
    Ts: a list of 9 4x4 homogenous transformation matrices that transform the base frame to all other frames along the kinematic chain.
    
    2. Output:
    J: Jacobian matrix. Shape (6,8).
    
    """
    num_joints = len(Ts) 
    t_eef = Ts[-1][:3,3]
    J = np.zeros((6, num_joints))

    for i in range(0, num_joints):
        t_i = Ts[i][:3,3]
        d_i_n = t_eef - t_i
        z_i = Ts[i][:3, 2]

        if i != 4:  # revolute joints
            J[:3, i] = np.cross(z_i, d_i_n)
            J[3:, i] = z_i
        else:  # prismatic joint (main insertion joint)
            J[:3, i] = z_i
            J[3:, i] = 0           
            
            
        if i in [0,6]:  # these joints have to multiple by -1 for some weird reasons
            J[:, i] *= -1

    return J

def get_transformation_matrices(q_cur):
    homo_mats = []
    for i in range(1, len(link_names)):

        if i == 1:  # if use world as the base link, not psm_base_link
            continue
            
        link = link_names[i]
        homo_mat = np.array(kdl_kin.forward(q_cur, end_link=link))  #
        homo_mats.append(homo_mat)
   
    return homo_mats



robot = URDF.from_parameter_server()
base_link = 'world'
end_link = 'psm_tool_yaw_link'
kdl_kin = KDLKinematics(robot, base_link, end_link)
link_names = kdl_kin.get_link_names()

num_tests = 3
q_curs_test = np.random.uniform(low = kdl_kin.joint_limits_lower, high = kdl_kin.joint_limits_upper, size=(num_tests,8))
# print(q_curs_test)

correct = 0
for i, q_cur in enumerate(q_curs_test):
    print("\n\nTest {}: =======================================".format(i+1))
    
    jacobian_gt = np.array(kdl_kin.jacobian(q_cur))
    print("Ground-truth Jacobian:")
    print(np.round(jacobian_gt, decimals=2))
    
    homo_mats = get_transformation_matrices(q_cur)
    jacobian_computed = compute_jacobian(homo_mats)
    print("Your answer:")
    print(np.round(jacobian_computed, decimals=2))

    if np.allclose(jacobian_gt, jacobian_computed, atol=1e-4):
        print("\nCongrats! Your answer for test {} is correct.".format(i+1))
    else:
        print("\nSorry. Your answer for test {} is wrong.".format(i+1))
        
    correct += np.allclose(jacobian_gt, jacobian_computed, atol=1e-4)
    
print("\n\n** Final remarks: You got {}/{} answers correct.".format(correct,num_tests))




