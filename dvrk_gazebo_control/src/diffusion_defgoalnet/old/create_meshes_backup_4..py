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
from utils.miscellaneous_utils import read_pickle_data, write_pickle_data


save_dir = "/home/baothach/shape_servo_data/diffusion_defgoalnet/object_data/retraction_cutting"
os.makedirs(os.path.join(save_dir, "mesh"), exist_ok=True)

export_tri_mesh = True#False
export_tet_mesh = True
vis = False
num_object_per_category = 50

categoies = ["cylinder", "ellipsoid"]

for category in categoies:
    for object_idx in range(num_object_per_category):
        object_name = f"{category}_{object_idx}"
        print(f"\n========= Object: {object_name}")
        if export_tri_mesh:
            if category == "cylinder":
                height = np.random.uniform(0.05, 0.15)
                ratio = np.random.uniform(10./4, 10./2)
                radius = height / ratio  
                
                plane_origin=[0,0.0,-height * 0.2]

                mesh = trimesh.creation.cylinder(radius=radius, height=height)

            elif category == "ellipsoid":
                radius = np.random.uniform(0.025, 0.075)
                ratio = np.random.uniform(2/5, 4/5)

                plane_origin=[0,0,-radius * 0.4]

                mesh = trimesh.creation.icosphere(radius = radius) 
                vertices_transformed = mesh.vertices * np.array([ratio,ratio,1])
                mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh.faces)

            slicing_angles = np.random.uniform([-30, -30, -180], [30, 30, 180], 3)
            plane_normal=[0,0,1]
            apply_euler_rotation_trimesh(mesh, *slicing_angles, degrees=True)
            mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=True)

            # Shift the object such that the bottom of the mesh is aligned with the z=0 plane
            lowest_point_z = mesh.bounds[0, 2]
            mesh.apply_translation([0, 0, -lowest_point_z])

            coordinate_frame = trimesh.creation.axis()  
            coordinate_frame.apply_scale(0.2)
            meshes = [mesh, coordinate_frame]

            # Find the intersection lines
            lines, face_index = trimesh.intersections.mesh_plane(mesh, plane_normal=plane_normal, plane_origin=plane_origin-lowest_point_z, return_faces=True)
            points = lines[:lines.shape[0]//2,0,:]  # Just get the start points of each line segments
            # print("lines.shape, points.shape:", lines.shape, points.shape)



            # Perform a batch query to find the nearest vertex in the mesh for each point
            nearest_vertices_indices = mesh.nearest.vertex(points)
            nearest_vertices_indices = np.array(nearest_vertices_indices[1], dtype=int)

            attachment_positions = mesh.vertices[nearest_vertices_indices]
            mean_y = np.mean(attachment_positions[:, 1])
            filtered_attachment_positions = attachment_positions[np.where(attachment_positions[:, 1] < mean_y)]
            # print("final attachment_positions.shape:", filtered_attachment_positions.shape)

            # Save the mesh and other information
            if category == "cylinder":
                mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj")) 
                info = {"radius": radius, "height": height, "attachment_positions": filtered_attachment_positions, "slicing_angles": slicing_angles}
                write_pickle_data(info, os.path.join(save_dir, "mesh", f"{object_name}_info.pickle"))

            elif category == "ellipsoid":
                # Save the mesh and other information
                mesh.export(os.path.join(save_dir, "mesh", f"{object_name}.obj")) 
                info = {"radius": radius, "ratio": ratio, "attachment_positions": filtered_attachment_positions, "slicing_angles": slicing_angles}
                write_pickle_data(info, os.path.join(save_dir, "mesh", f"{object_name}_info.pickle"))


            # Visualize  and save the attachment positions
            for position in attachment_positions[0: attachment_positions.shape[0]: 2]:
            # for position in filtered_attachment_positions[0: filtered_attachment_positions.shape[0]: 1]:
                sphere = trimesh.creation.icosphere(radius=0.002)
                sphere.apply_translation(position)
                sphere.visual.face_colors = [250, 0, 0, 128]
                meshes.append(sphere)

            spheres = trimesh.util.concatenate(meshes[2:])
            spheres.export(os.path.join(save_dir, "mesh", f"{object_name}_base.obj")) 
            # spheres.show()

            if vis:
                trimesh.Scene(meshes).show()



        if export_tet_mesh:
            print("Generating tetrahedral mesh ...")
            create_tet_mesh(os.path.join(save_dir, "mesh"), 
                            object_name, mesh_extension='.obj',
                            coarsen=False, verbose=False)

