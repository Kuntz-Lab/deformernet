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


def compute_jacobian(Ts):
    """ 
    Compute Jacobian matrix using the equation at 4:31 of this Youtube video: https://www.youtube.com/watch?v=C6Zho88S8vY
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


def construct_homo_mat(base_pose, eef_pose, fix_frame_type=None, rotate_robot=True):
    """ 
    Construct 4x4 homo matrices from base and end-effector poses. 
    fix_frame_type: Whether to align Isaacgym's coordinate frame with PyKDL frame. 
            Perform some simple swap of quaternion elements. Only applicable to a few joints, not all.
    """


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

    if fix_frame_type == 1:
        quat = transformations.quaternion_from_matrix(T)
        quat[1], quat[3] = -quat[3], -quat[1]
        T[:3,:3] = transformations.quaternion_matrix(quat)[:3,:3]

    if not rotate_robot:
        if fix_frame_type == 2:
            quat = transformations.quaternion_from_matrix(T)
            quat[1], quat[3] = quat[3], quat[1]
            T[:3,:3] = transformations.quaternion_matrix(quat)[:3,:3]

        if fix_frame_type == 3:
            quat = transformations.quaternion_from_matrix(T)
            quat[1], quat[3] = quat[3], quat[1]
            T[:3,:3] = transformations.quaternion_matrix(quat)[:3,:3]

    else:
        if fix_frame_type == 2:
            quat = transformations.quaternion_from_matrix(T)
            quat[1], quat[3] = -quat[3], -quat[1]
            T[:3,:3] = transformations.quaternion_matrix(quat)[:3,:3]

        elif fix_frame_type == 3:
            quat = transformations.quaternion_from_matrix(T)
            quat[1], quat[3] = quat[3], quat[1]
            quat[2] *= -1
            T[:3,:3] = transformations.quaternion_matrix(quat)[:3,:3]

        T[:2,3] *= -1

    return T


base_pose = ((0., 0., 0.25), (0., 0., 1., 0.))

q_cur = [-0.4663934, -0.23121195, 0.4126832, -0.42409793, 0.21888301, -0.00038343677, -0.24193324, 0.46662676]
eef_poses = ((9.313226e-09, 1.1175871e-08, 0.40240002), (0.60201114, 0.37092116, 0.37091887, -0.60201263)), ((-9.49949e-08, 0.02959999, 0.4024), (0.82254636, 0.21966963, 0.50679964, 0.13533972)), ((0.05846981, 0.10438356, 0.5185408), (0.5449979, -0.6540745, 0.33578643, -0.40300265)), ((0.02439427, -0.30872175, 0.4508445), (0.6704493, -0.524722, 0.41308326, -0.3233059)), ((0.04298548, -0.33877966, 0.48776874), (0.66322905, 0.70259804, -0.24547218, -0.07890809)), ((-0.13868162, -0.4387792, 0.1269092), (-0.70272434, 0.6630953, 0.07895496, -0.24545705)), ((-0.13868146, -0.4387792, 0.12690915), (-0.00032505, 0.8513774, -0.00014352, 0.5245535)), ((-0.14277354, -0.4387856, 0.11878112), (-0.7069026, -6.40962e-05, -0.00021807, -0.7073108))
fix_frame_1 = [0,1,2,3,4,5]
fix_frame_2 = [6]
fix_frame_3 = [7]

link_names = ['psm_base_link', 'psm_yaw_link', 'psm_pitch_back_link', 'psm_pitch_bottom_link', 'psm_pitch_end_link', 'psm_main_insertion_link', 'psm_tool_roll_link', 'psm_tool_pitch_link', 'psm_tool_yaw_link']
robot = URDF.from_parameter_server()
base_link = 'world'
end_link = 'psm_tool_yaw_link'
kdl_kin = KDLKinematics(robot, base_link, end_link)


homo_mats = []
for i, eef_pose in enumerate(eef_poses):
    fix_frame_type = None
    if i in fix_frame_1:
        fix_frame_type = 1
    if i in fix_frame_2:
        fix_frame_type = 2    
    if i in fix_frame_3:
        fix_frame_type = 3  
    homo_mat = construct_homo_mat(base_pose, eef_pose, fix_frame_type)
    homo_mats.append(homo_mat)
    # print(np.round(homo_mat, decimals=2))


link_names = kdl_kin.get_link_names()
homo_mat_gts = []
for i in range(1, len(link_names)):

    if i == 1:  # if use world as the base link, not psm_base_link
        continue
        
    link = link_names[i]
    homo_mat_gt = np.array(kdl_kin.forward(q_cur, end_link=link))  #
    homo_mat_gts.append(homo_mat_gt)
    # print(np.round(homo_mat_gt, decimals=2))



jacobian_gt = np.array(kdl_kin.jacobian(q_cur)) 
print(len(homo_mats))
jacobian_sim = compute_jacobian(homo_mats)

print("==========xxxxxxxxxxxxxxxxxxxxxxxxxxx")

print("Sim")
print(jacobian_sim)
print("Ground Truth")
print(jacobian_gt)

print("Correct Jacobian computation:", all((np.round(jacobian_gt, decimals=5) == np.round(jacobian_sim, decimals=5)).flatten()))
print("(Approx) Correct?:", np.allclose(jacobian_gt, jacobian_sim, atol=1e-3))





