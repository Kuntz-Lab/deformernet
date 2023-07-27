#!/usr/bin/env python3



import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np

from copy import copy, deepcopy
import rospy
import transformations
np.random.seed(10000)
sys.float_repr_style = 'plain'
np.set_printoptions(suppress=True, precision=5)

rospy.init_node('Jacobian')


def convert_3x3_to_4x4(rot_mat):
    homo_mat = np.eye(4)
    homo_mat[3,3] = 1
    homo_mat[:3,:3] = rot_mat

    return homo_mat

def SVD(A,C):
    """
    Find rotation matrix B such that A*B=C
    """


    # B = np.dot(C, np.linalg.inv(A))
    # Compute the SVD of matrix A
    U, _, Vt = np.linalg.svd(A)

    # Calculate the rotation matrix R
    R = np.matmul(U, Vt)

    # Compute the rotation matrix B
    B = np.matmul(C, np.linalg.inv(R))

    return B



A = np.array([[ 0.,       1.,       0.,     ],
 [-0.54611, -0.,       0.83771],
 [ 0.83771, -0.,       0.54611]])

C = np.array([[ 0.54611, -0.83771,  0.,     ],
 [ 0.,      -0.,      -1.,     ],
 [ 0.83771,  0.54611,  0.     ]])

A2 = np.array([[ 0.54611,  0.00001, -0.83771],
 [ 0.29113,  0.93767,  0.1898 ],
 [ 0.7855,  -0.34754,  0.51207]])

C2 = np.array([[ 0.51207, -0.1898,  -0.83771],
 [ 0.34754,  0.93767, -0.00001],
 [ 0.7855,  -0.29113,  0.54611]])

A3 = np.array([[-0.80276, -0.28585, -0.52333],
 [ 0.54605,  0.00027, -0.83775],
 [ 0.23961, -0.95827,  0.15587]])
C3 = np.array([[ 0.1561,   0.83771, -0.52333],
 [ 0.95827,  0.00001,  0.28585],
 [ 0.23946, -0.54612, -0.80275]]
)

A6 = np.array([[ 0.54605,  0.00026, -0.83775],
 [ 0.00018, -1.,      -0.00019],
 [-0.83775, -0.00005, -0.54605]]
)
C6 = np.array([[-0.54612, -0.00003, -0.83771],
 [ 0.00005, -1.,       0.00001],
 [-0.83771, -0.00003,  0.54612]]
)

A7 = np.array([[ 0.00017, -1.,      -0.0002 ],
 [-1.,      -0.00017,  0.00007],
 [-0.00007,  0.0002,  -1.     ]])
C7 = np.array([[-1.,       0.00001,  0.00003],
 [ 0.00003, -0.00003,  1.     ],
 [ 0.00001,  1.,       0.00003]]
)

# print(np.linalg.det(C))


print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(A)))
print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(C)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(A2)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(C2)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(A3)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(C3)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(A6)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(C6)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(A7)))
# print(transformations.quaternion_from_matrix(convert_3x3_to_4x4(C7)))
# B = SVD(A,C)
# print(B)
# print("results:")
# print(B@A)

# print("==============")

# B_gt = np.array([[ 0.,   0.,  -1. ],
#  [ 1.,   0.,   0. ],
#  [ 0.,   1.,   0. ]])
# print(B_gt)
# print("results:")
# print(B_gt@A)


a = [1,2,3,4]
a[1],a[3] = -a[3],-a[1]
print(a)

a= ((0., -0.99999994, 0.25), (0., 0., 0., 1.))
print(list(a))
