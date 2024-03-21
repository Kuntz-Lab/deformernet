import numpy as np
import trimesh
import os
import pickle
from copy import deepcopy
import roslib.packages as rp
import sys
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
from utils.mesh_utils import create_tet_mesh, apply_euler_rotation_trimesh


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_cutting"
os.makedirs(os.path.join(save_dir, "mesh"), exist_ok=True)


object_name = "cylinder_1"
radius = 0.04
height = 0.2
mesh = trimesh.creation.cylinder(radius=radius, height=height)
apply_euler_rotation_trimesh(mesh, 10, 0, 0, degrees=True)
mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0.00,-height/3], cap=True)
# plane = trimesh.creation.box(extents=[0.5, 0.5, 0.001])
# plane.apply_translation([0, 0, -height/2])

# mesh.show()

meshes = [mesh]
# Find the intersection lines
lines, face_index = trimesh.intersections.mesh_plane(mesh, plane_normal=[0,0,1], plane_origin=[0,0.00,-height/3], return_faces=True)
# print(lines.shape)

points = lines.reshape(-1, 3)  # Reshape to a list of points
print(points.shape)

# Perform a batch query to find the nearest vertex in the mesh for each point
# This function returns indices of the nearest vertices to each point
nearest_vertices_indices = mesh.nearest.vertex(points)
# print(nearest_vertices_indices[1].shape)
nearest_vertices_indices = np.array(nearest_vertices_indices[1], dtype=int)
nearest_vertices_positions = mesh.vertices[nearest_vertices_indices].reshape(-1, 3)
# print(nearest_vertices_positions.shape)

# To visualize these as large, noticeable points, we'll create a set of spheres at each vertex position
# Adjust the sphere radius as needed to make them clearly visible
for position in nearest_vertices_positions:
    # print(position)
    sphere = trimesh.creation.icosphere(radius=0.002)
    sphere.apply_translation(position)
    meshes.append(sphere)
    # print("aaaaa")

trimesh.Scene(meshes).show()

# # Check if any lines were found
# if lines.shape[0] > 0:
#     # Create a LineCollection from the line segments for visualization
#     line_collection = trimesh.load_path(lines.reshape(-1, 3)[16:16+64])
    
#     # Create a scene and add the original mesh and the line collection
#     scene = trimesh.Scene([mesh, line_collection])

#     # Show the scene
#     scene.show()

# coordinate_frame = trimesh.creation.axis()  
# coordinate_frame.apply_scale(0.2)

# meshes = [mesh, coordinate_frame, plane]
# trimesh.Scene(meshes).show()

# mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj"))
# create_tet_mesh(save_dir, "cylinder")

# info = {"height": height, "radius": radius}
# with open(os.path.join(os.path.join(save_dir, "info"), f"{object_name}.pickle"), 'wb') as handle:
#     pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)