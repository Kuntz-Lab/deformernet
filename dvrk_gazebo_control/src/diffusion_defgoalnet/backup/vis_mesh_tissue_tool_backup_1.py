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


# def randomize_pose(mesh, 
#                    min_translation=[-0.5, -0.5, -0.5], 
#                    max_translation=[0.5, 0.5, 0.5],
#                    min_rotation=[-np.pi, -np.pi, -np.pi], 
#                    max_rotation=[np.pi, np.pi, np.pi]):
#     T = np.eye(4)
#     angles = [np.random.uniform(min_rotation[i], max_rotation[i]) for i in range(3)]
#     T[:3, :3] = R.from_euler('xyz', angles).as_matrix()
#     T[:3, 3] = [np.random.uniform(min_translation[i], max_translation[i]) for i in range(3)]
#     mesh = mesh.copy()
#     mesh.apply_transform(T)
#     return mesh
def randomize_pose(mesh, min_rotation, max_rotation, translation):
    T = np.eye(4)
    angles = [np.random.uniform(min_rotation[i], max_rotation[i]) for i in range(3)]
    T[:3, :3] = trimesh.transformations.euler_matrix(*angles)[:3, :3]
    T[:3, 3] = translation
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return mesh
def create_quadrant_planes(A, alpha=0.5):
    planes = []
    size = 4 * A  # Size to ensure planes are large enough to visualize quadrants

    # Horizontal Plane
    plane_xy = trimesh.creation.box(extents=[size, size, 0.001])
    plane_xy.visual.face_colors = [0, 0, 255, int(255 * alpha)]  # Blue color with transparency
    planes.append(plane_xy)

    # Vertical Planes (YZ and XZ)
    plane_yz = trimesh.creation.box(extents=[0.001, size, size])
    plane_yz.visual.face_colors = [255, 0, 0, int(255 * alpha)]  # Red color with transparency
    planes.append(plane_yz)

    plane_xz = trimesh.creation.box(extents=[size, 0.001, size])
    plane_xz.visual.face_colors = [0, 255, 0, int(255 * alpha)]  # Green color with transparency
    planes.append(plane_xz)

    return planes

height = 0.1  
ratio = 5
radius = height / ratio  


slicing_angles = [10, 10, 10]   #[30, 30, 30] [30, 0, 0] 
plane_normal=[0,0,1]
plane_origin=[0,0.0,-height * 0.2]

tissue_mesh = trimesh.creation.cylinder(radius=radius, height=height)
apply_euler_rotation_trimesh(tissue_mesh, *slicing_angles, degrees=True)
tissue_mesh = trimesh.intersections.slice_mesh_plane(mesh=tissue_mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)
lowest_point_z = tissue_mesh.bounds[0, 2]
tissue_mesh.apply_translation([0, 0, -lowest_point_z])

coordinate_frame = trimesh.creation.axis()  
coordinate_frame.apply_scale(0.2)


height = 0.2 
ratio = 40
radius = height / ratio  

# Create the cylinder
cylinder = trimesh.creation.cylinder(radius=radius, height=height)

# Translate the mesh to center it at the origin
cylinder.apply_translation(-cylinder.centroid)

tool_meshes = []
# for j in range(4):
#     tool_mesh = randomize_pose(cylinder, 
#                         min_translation=[-0.5, -0.5, -0.2], 
#                         max_translation=[0.5, 0.5, 0.2],
#                         min_rotation=[-np.pi/4, -np.pi/6, -np.pi/3], 
#                         max_rotation=[np.pi/4, np.pi/6, np.pi/3])
#     tool_meshes.append(tool_mesh)

A = 0.05  # Defined translation value

# Each tool_mesh is positioned in a different quadrant (positive z region only)
translations = [
    [ A,  A,  A],
    [ A, -A,  A],
    [-A,  A,  A],
    [-A, -A,  A]
]

for j in range(4):
    tool_mesh = randomize_pose(cylinder,
                               min_rotation=[-np.pi/4, -np.pi/4, -np.pi/4],
                               max_rotation=[ np.pi/4,  np.pi/4,  np.pi/4],
                               translation=translations[j])
    tool_meshes.append(tool_mesh)

# trimesh.Scene(tool_meshes + [tissue_mesh, coordinate_frame]).show()

quadrant_planes = create_quadrant_planes(A, alpha=0.2)
trimesh.Scene(tool_meshes + [tissue_mesh, coordinate_frame] + quadrant_planes).show()