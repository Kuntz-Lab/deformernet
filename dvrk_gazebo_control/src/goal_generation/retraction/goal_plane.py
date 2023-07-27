import open3d
import pyransac3d as pyrsc
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy


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


def check_success(constrain_plane, current_pc):
    failed_points = np.array([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])                   
    if len(failed_points) == 0:
        return True
    else:
        return False

def get_goal_plane(constrain_plane, initial_pc, delta = None):
    
    if delta is not None:
        constrain_plane = constrain_plane.copy()
        constrain_plane[3] += delta


    constrain_plane = constrain_plane.copy()
    constrain_plane[3] += 0.03


    failed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]])
    passed_points = np.array([p for p in initial_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] <= -constrain_plane[3]])

    if len(passed_points) == 0:
        return None

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(failed_points)
    pcd2.paint_uniform_color([1,0,0])
    # open3d.visualization.draw_geometries([pcd2])

    # Find rotation point:
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
    center = pcd.get_center().reshape(3)
    dist = (np.dot(center, constrain_plane[:3]) + constrain_plane[3])/np.linalg.norm(constrain_plane[:3])
    unit_normal = np.array(constrain_plane[:3])/np.linalg.norm(constrain_plane[:3])
    rot_pt = center - dist*unit_normal

    pcd.points = open3d.utility.Vector3dVector(passed_points)
    pcd.paint_uniform_color([0,0,0])


    # Fit a plane to the failed_points
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(failed_points, thresh=0.001, maxIteration=1000)
    if best_eq[2] > 0:
        best_eq = -np.array(best_eq)


    r = vecalign(np.array(best_eq)[:3], constrain_plane[:3])
    pcd2.rotate(R = r.T, center = rot_pt.reshape((3,1)))

    goal_pcd = pcd + pcd2
    # open3d.visualization.draw_geometries([goal_pcd])
    return np.asarray(goal_pcd.points)

def visualize_plane(plane_eq, x_range=1, y_range=[-1,0], z_range=0.2,num_pts = 10000):
    plane = []
    for i in range(num_pts):
        x = np.random.uniform(-x_range, x_range)
        z = np.random.uniform(-0.05, z_range)
        y = -(plane_eq[0]*x + plane_eq[2]*z + plane_eq[3])/plane_eq[1]
        if y_range[0] < y < y_range[1]:
            plane.append([x, y, z])     
    return plane  


def get_action(tissue_angle, y_mag, z_mag):
    return -y_mag*np.sin(tissue_angle), y_mag*np.cos(tissue_angle), z_mag

def visualize_plane(plane_eq, x_range=1, y_range=[-1,0], z_range=[-0.05,0.1],num_pts = 10000):
    plane = []
    for i in range(num_pts):
        x = np.random.uniform(-x_range, x_range)
        z = np.random.uniform(z_range[0], z_range[1])
        y = -(plane_eq[0]*x + plane_eq[2]*z + plane_eq[3])/plane_eq[1]
        if y_range[0] < y < y_range[1]:
            plane.append([x, y, z])     
    return plane  

def pcd_ize(pc, color=None, vis=False):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)   
    if color is not None:
        pcd.paint_uniform_color(color) 
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd

def get_kidney_and_tissue_angle():
    kidney_angle = np.random.uniform(low=-np.pi/4.5, high=np.pi/4.5)
    tissue_angle = np.random.uniform(low=kidney_angle-np.pi/4.5, high=kidney_angle+np.pi/4.5) 
    tissue_angle = np.clip(tissue_angle, a_min=-np.pi/4.5, a_max=np.pi/4.5)
    return kidney_angle, tissue_angle

def generate_new_target_plane():
      
    # return np.array([-1, 1, 0, 0.35])   #min
    # return np.array([0, 1, 0, 0.45])   #max
    # return np.array([-1, 1, 0, 0.44])

    direction = np.random.uniform(low=-1, high=1)
    shift = np.random.uniform(low=0.35, high=0.45) 
    
    return np.array([direction, 1, 0, shift])

    
    # choice = np.random.randint(0,3)
    # if choice == 0: # horizontal planes
    #     pos = np.random.uniform(low=0.45, high=0.50)
    #     return np.array([0, 1, 0, pos])
    # elif choice == 1:   # tilted left
    #     pos = np.random.uniform(low=0.38, high=0.45) 
    #     return np.array([1, 1, 0, pos])
    # elif choice == 2:   # tilted right
    #     pos = np.random.uniform(low=0.45, high=0.50)  
    #     return np.array([-1, 1, 0, pos]) 