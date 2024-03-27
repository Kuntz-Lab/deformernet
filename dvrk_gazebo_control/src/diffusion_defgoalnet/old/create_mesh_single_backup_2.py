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


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_cutting"
os.makedirs(os.path.join(save_dir, "mesh"), exist_ok=True)

export_tri_mesh = True#False
export_tet_mesh = False

object_name = "cylinder_0"

if export_tri_mesh:
    # radius = 0.02
    # height = 0.1
    # # radius = 0.03
    # # height = 0.1
    
    
    # slicing_angles = [30, 0, 0] 
    # plane_normal=[0,0,1]
    # plane_origin=[0,0.0,-height * 0.2]

    # mesh = trimesh.creation.cylinder(radius=radius, height=height)
    
    

    height = 0.1
    width = 0.04
    depth = 0.04
    
    
    slicing_angles = [30, 0, 0] 
    plane_normal=[0,0,1]
    plane_origin=[0,0.0,-height * 0.2]

    mesh = trimesh.creation.box((depth, width, height)) 
    mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.1, iterations=100)

    
    apply_euler_rotation_trimesh(mesh, *slicing_angles, degrees=True)
    # mesh.show()
    mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)

    # Shift the object such that the bottom of the mesh is aligned with the z=0 plane
    lowest_point_z = mesh.bounds[0, 2]
    mesh.apply_translation([0, 0, -lowest_point_z])

    coordinate_frame = trimesh.creation.axis()  
    coordinate_frame.apply_scale(0.2)
    meshes = [mesh, coordinate_frame]

    # Find the intersection lines
    lines, face_index = trimesh.intersections.mesh_plane(mesh, plane_normal=plane_normal, plane_origin=plane_origin-lowest_point_z, return_faces=True)
    # points = lines[:lines.shape[0]//2,0,:]  # Just get the start points of each line segments
    # print("lines.shape, points.shape:", lines.shape, points.shape)
    # # points = points[0:32]
    # points = points[0:lines.shape[0]//2:1] # Only take every other point to reduce the number of points
    points = lines.reshape(-1, 3)

    # Perform a batch query to find the nearest vertex in the mesh for each point
    # This function returns indices of the nearest vertices to each point
    nearest_vertices_indices = mesh.nearest.vertex(points)
    nearest_vertices_indices = np.array(nearest_vertices_indices[1], dtype=int)
    # print(nearest_vertices_indices.shape)


    # # # Save the mesh and other information
    # # mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj")) 
    # # info = {"radius": radius, "height": height, "nearest_vertices_indices": nearest_vertices_indices, "slicing_angles": slicing_angles}
    # # write_pickle_data(info, os.path.join(save_dir, "mesh", f"{object_name}_info.pickle"))


    # To visualize these as large, noticeable points, we'll create a set of spheres at each vertex position
    # Adjust the sphere radius as needed to make them clearly visible
    nearest_vertices_positions = mesh.vertices[nearest_vertices_indices].reshape(-1, 3)
    for position in nearest_vertices_positions:
        sphere = trimesh.creation.icosphere(radius=0.002)
        sphere.apply_translation(position)
        sphere.visual.face_colors = [250, 0, 0, 128]
        meshes.append(sphere)


    # spheres = trimesh.util.concatenate(meshes[2:])
    # spheres.export(os.path.join(save_dir, "mesh", f"{object_name}_base.obj")) 
    # # spheres.show()

    trimesh.Scene(meshes).show()

    # # Check if any lines were found
    # if lines.shape[0] > 0:
    #     # Create a LineCollection from the line segments for visualization
    #     line_collection = trimesh.load_path(lines.reshape(-1, 3))
        
    #     # Create a scene and add the original mesh and the line collection
    #     scene = trimesh.Scene([mesh, line_collection])

    #     # Show the scene
    #     scene.show()





if export_tet_mesh:
    print("Generating tetrahedral mesh ...")
    create_tet_mesh(os.path.join(save_dir, "mesh"), 
                    object_name, mesh_extension='.obj',
                    coarsen=False, verbose=False)

