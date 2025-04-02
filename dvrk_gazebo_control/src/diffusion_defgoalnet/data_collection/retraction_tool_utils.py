import open3d
import pyransac3d as pyrsc
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


def print_color(text, color="red"):

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

    if color == "red":
        print(RED + text + RESET)
    elif color == "green":
        print(GREEN + text + RESET)
    elif color == "yellow":
        print(YELLOW + text + RESET)
    elif color == "blue":
        print(BLUE + text + RESET)
    else:
        print(text)


def pcd_ize(pc, color=None, vis=False):
    """ 
    Convert point cloud numpy array to an open3d object (usually for visualization purpose).
    """
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd


def vecalign(a, b):
    '''
    Returns the rotation matrix that can rotate the 3 dimensional b vector to
    be aligned with the a vector.

    @param a (3-dim array-like): destination vector
    @param b (3-dim array-like): vector to rotate align with a in direction

    the vectors a and b do not need to be normalized.  They can be column
    vectors or row vectors
    '''
    with np.errstate(divide='raise', under='raise', over='raise', invalid='raise'):
        a = np.asarray(a)
        b = np.asarray(b)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        cos_theta = a.dot(b) # since a and be are unit vecs, a.dot(b) = cos(theta)

        # if dot is close to -1, vectors are nearly equal
        if np.isclose(cos_theta, 1):
            # TODO: solve better than just assuming they are exactly identical
            return np.eye(3)

        # if dot is close to -1, vectors are nearly opposites
        if np.isclose(cos_theta, -1):
            # TODO: solve better than just assuming they are exactly opposite
            return -np.eye(3)

        axis = np.cross(b, a)
        sin_theta = np.linalg.norm(axis)
        axis = axis / sin_theta
        c = cos_theta
        s = sin_theta
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]

        # angle-axis formula to create a rotation matrix
        return np.array([
            [t*x*x + c  , t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c  , t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c  ],
            ])


def check_plane(constrain_plane, current_pc, vis=False, x_range=[-1,1], y_range=[-1,1], z_range=0.2):

    success_points = np.array([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])                   
    total_num_pts = len(current_pc)
    total_passed = len(success_points)
    
    print_color(f"\n*** Percentage passed: {total_passed/total_num_pts*100:.2f}\n\n", color="green")

    if vis:
        pcd = pcd_ize(current_pc, color=[1,0,0])
        pcd_plane = pcd_ize(visualize_plane(constrain_plane, 
                                            x_range=x_range, y_range=y_range, z_range=z_range), 
                                            color=[0,0,0])
        open3d.visualization.draw_geometries([pcd, pcd_plane])


def visualize_plane(plane_eq, x_range=[-1,0], y_range=[-1,0], z_range=0.2,num_pts = 10000):
    plane = []
    for i in range(num_pts):
        x = np.random.uniform(x_range[0], x_range[1])
        z = np.random.uniform(-0.05, z_range)
        y = -(plane_eq[0]*x + plane_eq[2]*z + plane_eq[3])/plane_eq[1]
        if y_range[0] < y < y_range[1]:
            plane.append([x, y, z])     
    return plane  
