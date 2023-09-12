import os
import numpy as np
import trimesh
import open3d
import pickle
import transformations
from copy import deepcopy


def get_rays(cylinder_shift, num_rays=1024):

    # Create target cylinder
    quat = [0.5, 0.5,0.5, 0.5]
    trans_mat = transformations.quaternion_matrix(quat)
    trans_mat[:3,3] = np.array([0, -0.5, 0.04]) + cylinder_shift
    # cylinder_mesh = trimesh.creation.annulus(r_min=0.0149*0.6, r_max=0.015*0.6, height=0.1, transform = trans_mat)
    cylinder_mesh = trimesh.creation.annulus(r_min=0.01498*0.6, r_max=0.015*0.6, height=0.1*0.7, transform = trans_mat)


    # Sample points (and corresponding normals) on mesh
    pc, faces = trimesh.sample.sample_surface_even(cylinder_mesh, count=num_rays)   # might not sample enough num_rays points
    normals = cylinder_mesh.face_normals[faces]

    ray_origins, ray_directions = pc, normals/25    # /25: shorten normals for visualization

    # stack rays into line segments for visualization as Path3D
    ray_visualize = trimesh.load_path(np.hstack((
        ray_origins,
        ray_origins + ray_directions)).reshape(-1, 2, 3))

    return ray_origins, ray_directions, ray_visualize, cylinder_mesh


# def compute_intersection_percent(ray_origins, ray_directions, ray_visualize, cylinder_mesh, backbone, cylinder_shift, vis = False):

def compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, vis = False, num_rays=1024):
    return None
    
    ray_origins, ray_directions, ray_visualize, cylinder_mesh = get_rays(cylinder_shift, num_rays)

    tissue_mesh = trimesh.Trimesh(vertices=final_full_pc,
                            faces=np.array(tri_indices).reshape(-1,3).astype(np.int32))

    ### run the mesh- ray test
    locations, index_ray, index_tri = tissue_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)
    intersection = trimesh.points.PointCloud(locations)

    index_ray = set(index_ray)
    # print("number of intersections:", len(index_ray))
    # print("Percent coverage:", len(index_ray)/ray_origins.shape[0])    


    if vis:
        tissue_mesh.visual.face_colors = [255, 0, 0,200]
        scene = trimesh.Scene([
            cylinder_mesh, tissue_mesh,
            intersection,
            ray_visualize])
        scene.show()


    return len(index_ray)/ray_origins.shape[0]



def record_data(init_full_pc, final_full_pc, tri_indices, cylinder_shift, save_path):
    data = {"init_full_pc": init_full_pc, "final_full_pc": final_full_pc, "tri_indices": tri_indices,
            "cylinder_shift": cylinder_shift}    
    
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
