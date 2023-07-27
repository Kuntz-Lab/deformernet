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
    num_joints = len(Ts) - 1
    t_eef = Ts[-1][:3,3]
    J = np.zeros((6, num_joints))

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

def SVD(A,C):
    """
    Find rotation matrix B such that A*B=C
    """

    # Calculate the SVD of matrix A
    U, _, V = np.linalg.svd(A)

    # Construct the rotation matrix B
    B = np.matmul(U, V)

    # Solve for matrix B
    solution = np.matmul(np.linalg.inv(B), C)

    # Reconstruct B as a rotation matrix
    _, singular_values, VT = np.linalg.svd(solution)
    B = np.matmul(U, np.matmul(np.diag(singular_values), VT))

    return B


def get_sim_to_real_adjustment_matrices(Ts_sim, Ts_ground_truth):
    """
    fix 4x4 homogeneous transformation matrices obatined form Isaacgym
    """
    adjustment_mats = []
    for i in range(len(Ts_sim)):
        adjustment_mat = np.zeros((4,4))
        adjustment_mat[3,3] = 1
        # adjustment_mat[:3,:3] = np.matmul(np.linalg.inv(Ts_sim[i][:3,:3]), Ts_ground_truth[i][:3,:3])
        adjustment_mat[:3,:3] = SVD(Ts_sim[i][:3,:3], Ts_ground_truth[i][:3,:3])
        adjustment_mats.append(np.round(adjustment_mat, decimals=0))
        
    return adjustment_mats
        
def adjust_sim_homo_mats(Ts_sim, adjustment_mats):
    adjusted_homo_mats = []
    for i in range(len(Ts_sim)):
        adjusted_homo_mats.append(np.dot(Ts_sim[i],adjustment_mats[i]))
        
    return adjusted_homo_mats

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

# eef_pose = ((-5.7900766e-06, -0.61336434, 0.14226009), (-6.2935855e-05, 0.70698524, 0.7072283, -7.1378825e-05))
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

# homo_mat = construct_homo_mat(base_pose, eef_pose)

# print("From Isaacgym:")
# print(np.round(homo_mat, decimals=2))

# homo_mat_gt = np.array(kdl_kin.forward(q_cur)) 
# print("Ground truth")
# print(np.round(homo_mat_gt, decimals=2))
# print(transformations.quaternion_from_matrix(homo_mat_gt))

print("From Isaacgym:")
homo_mats = []
for i, eef_pose in enumerate(eef_poses):
    print("==========", i+1)
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

print("==========xxxxxxxxxxxxxxxxxxxxxxxxxxx")

print("Ground truth:")
link_names = kdl_kin.get_link_names()
homo_mat_gts = []
for i in range(1, len(link_names)):

    if i == 1:  # if use world as the base link, not psm_base_link
        continue
    if i == 0:
        print("==========", i)
    else:
        print("==========", i-1)  
        
    link = link_names[i]
    homo_mat_gt = np.array(kdl_kin.forward(q_cur, end_link=link))  #
    homo_mat_gts.append(homo_mat_gt)
    # print(np.round(homo_mat_gt, decimals=2))

print("adjustment_mats hello: ")

adjustment_mats = get_sim_to_real_adjustment_matrices(homo_mats, homo_mat_gts)
print(adjustment_mats)
# adjustment_mats = [np.array([[ 1., -0., -0.,  0.],
#                             [-0., -1.,  0.,  0.],
#                             [-0., -0., -1.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), 
#                     np.array([[ 1.,  0.,  0.,  0.],
#                             [-0.,  1., -0.,  0.],
#                             [-0.,  0.,  1.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), 
#                     np.array([[ 1.,  0.,  0.,  0.],
#                             [-0.,  1.,  0.,  0.],
#                             [-0., -0.,  1.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), np.array([[ 1., -0., -0.,  0.],
#                             [ 0.,  1.,  0.,  0.],
#                             [ 0., -0.,  1.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), np.array([[-0.,  1.,  0.,  0.],
#                             [-0.,  0., -1.,  0.],
#                             [-1., -0.,  0.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), np.array([[-0., -1., -0.,  0.],
#                             [-0., -0.,  1.,  0.],
#                             [-1.,  0., -0.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), np.array([[ 1.,  0.,  0.,  0.],
#                             [-0.,  1., -0.,  0.],
#                             [-0.,  0.,  1.,  0.],
#                             [ 0.,  0.,  0.,  1.]]), np.array([[-0.,  0., -1.,  0.],
#                             [ 1.,  0., -0.,  0.],
#                             [ 0., -1., -0.,  0.],
#                             [ 0.,  0.,  0.,  1.]])]


adjusted_homo_mats = adjust_sim_homo_mats(homo_mats, adjustment_mats) 


for i in range(len(homo_mat_gts)):
    print("==========", i)
    print("Correct?:", all((np.round(homo_mats[i], decimals=5) == np.round(homo_mat_gts[i], decimals=5)).flatten()))
    print("(Approx) Correct?:", np.allclose(homo_mats[i], homo_mat_gts[i], atol=1e-3))
    
    if not np.allclose(homo_mats[i], homo_mat_gts[i], atol=1e-3):
        print("index wrong:", i)
        print("Original:")
        print(homo_mats[i])
        # print("Adjustment mat:")
        # print(adjustment_mats[i][:3,:3])
        # print("Adjusted:")
        # print(adjusted_homo_mats[i][:3,:3])
        print("Ground truth:")
        print(homo_mat_gts[i])    

        print("index wrong:", i)
        print("Original:")
        print(transformations.quaternion_from_matrix(homo_mats[i]))
        print("Ground truth:")
        print(transformations.quaternion_from_matrix(homo_mat_gts[i]))


jacobian_gt = np.array(kdl_kin.jacobian(q_cur)) 
homo_mats.insert(0,np.eye(4))
print(len(homo_mats))
jacobian_sim = compute_jacobian(homo_mats)

print("==========xxxxxxxxxxxxxxxxxxxxxxxxxxx")

print("Sim")
print(jacobian_sim)
print("Ground Truth")
print(jacobian_gt)

print("Correct Jacobian computation:", all((np.round(jacobian_gt, decimals=5) == np.round(jacobian_sim, decimals=5)).flatten()))
print("(Approx) Correct?:", np.allclose(jacobian_gt, jacobian_sim, atol=1e-3))





