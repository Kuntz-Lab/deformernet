import numpy as np
import trimesh
import os
import pickle
from copy import deepcopy
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from utils.mesh_utils import create_tet_mesh, apply_euler_rotation_trimesh, simplify_mesh_pymeshlab
from utils.miscellaneous_utils import read_pickle_data, write_pickle_data
from scipy.spatial.transform import Rotation as R
import transformations


def sample_cylindrical_points(num_points, r_min, r_max, theta_min, theta_max, z_min, z_max):
    points = []
    for _ in range(num_points):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(theta_min, theta_max)
        z = np.random.uniform(z_min, z_max)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)


def get_random_tool_pose(translation_vector: np.ndarray, 
                         min_rotation: np.ndarray = [0, -np.pi/8, -np.pi/8], max_rotation: np.ndarray = [0, np.pi/8, np.pi/8], 
                         return_format: str ="quaternion"):
    T = np.eye(4)
    angles = [np.random.uniform(min_rotation[i], max_rotation[i]) for i in range(3)]
    # angles = [0, np.pi/8, np.pi + np.pi/8]  #  [x, -np.pi/3, 0]   # rool, yaw, pitch
    T[:3, :3] = transformations.euler_matrix(*angles)[:3, :3]
    T[:3, 3] = translation_vector
    if return_format == "matrix":
        return T    # shape (4, 4)
    elif return_format == "quaternion":
        quat = transformations.quaternion_from_matrix(T)
        return np.concatenate([translation_vector, quat])  # shape (7,) [x, y, z, w, x, y, z]


tissue_height = 0.05   #0.05 - 0.12  
ratio = 5
radius = tissue_height / ratio  


slicing_angles = [0, 0, 0]   #[30, 30, 30] [30, 0, 0] 
plane_normal=[0,0,1]
plane_origin=[0,0.0,-tissue_height * 0.2]

tissue_mesh = trimesh.creation.cylinder(radius=radius, height=tissue_height)
apply_euler_rotation_trimesh(tissue_mesh, *slicing_angles, degrees=True)
tissue_mesh = trimesh.intersections.slice_mesh_plane(mesh=tissue_mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
lowest_point_z = tissue_mesh.bounds[0, 2]
tissue_mesh.apply_translation([0, 0, -lowest_point_z])

print("tissue_mesh.extents: ", tissue_mesh.extents)

coordinate_frame = trimesh.creation.axis()  
coordinate_frame.apply_scale(0.2)


height = 0.2   #np.random.uniform(0.05, 0.15)
ratio = 40 #np.random.uniform(10./4, 10./2)
radius = height / ratio  

# Create the cylinder
cylinder = trimesh.creation.cylinder(radius=radius, height=height)

# # Translate the mesh to center it at the origin
# cylinder.apply_translation(-cylinder.centroid)
cylinder.apply_translation([0, 0, height / 2])


# Parameters
num_points = 1000
r_min, r_max = radius + 0.01, radius + 0.03    #0.03, 0.05
# theta_min, theta_max = 0, np.pi
# theta_min, theta_max = 0, np.pi/3
theta_min, theta_max = 2*np.pi/3, 3*np.pi/3
z_min, z_max = 0, tissue_height * 0.7

# Sample points
sampled_points = sample_cylindrical_point(r_min, r_max, theta_min, theta_max, z_min, z_max)

tool_meshes = []
for i in range(100):
    translation_vector = sampled_points[i]
    # T = get_random_tool_pose(translation_vector, return_format="matrix",
    #                          min_rotation = [-np.pi/4, -np.pi/8, 0], max_rotation = [0, np.pi/8, 0])
    # T = get_random_tool_pose(translation_vector, return_format="matrix",
    #                          min_rotation = [-np.pi/4, np.pi/2-np.pi/6, 0], max_rotation = [0, np.pi/2-np.pi/3, 0])
    # T = get_random_tool_pose(translation_vector, return_format="matrix",
    #                          min_rotation = [-np.pi/4, np.pi/2-np.pi/3-np.pi/12, 0], max_rotation = [0, -np.pi/12, 0])
    T = get_random_tool_pose(translation_vector, return_format="matrix",
                             min_rotation = [-np.pi/4, -np.pi/2 + np.pi/6, 0], max_rotation = [0, -np.pi/2 + np.pi/3, 0])
    tool_mesh = cylinder.copy()
    tool_mesh.apply_transform(T)
    tool_meshes.append(tool_mesh)


trimesh.Scene(tool_meshes + [tissue_mesh, coordinate_frame]).show()