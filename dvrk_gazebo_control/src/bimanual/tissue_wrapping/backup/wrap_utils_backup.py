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
    cylinder_mesh = trimesh.creation.annulus(r_min=0.01498*0.6, r_max=0.015*0.6, height=0.1*0.8, transform = trans_mat)


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

def compute_intersection_percent(backbone, cylinder_shift, vis = False, num_rays=1024):
    
    ray_origins, ray_directions, ray_visualize, cylinder_mesh = get_rays(cylinder_shift, num_rays)

    xs = backbone[:,0]#*0
    ys = backbone[:,1]
    zs = backbone[:,2]

    # endpoints = np.vstack((xs,ys,zs)).T
    # print(endpoints.shape)
    # pcd2 = open3d.geometry.PointCloud()
    # pcd2.points = open3d.utility.Vector3dVector(endpoints)
    # pcd2.paint_uniform_color([1,0,0])
    # open3d.visualization.draw_geometries([pcd2]) 


    meshes = []

    ### Create tissue mesh:
    for x_value in np.linspace(start=-0.15*0.7/2+cylinder_shift[0], stop=0.15*0.7/2+cylinder_shift[0], num = 100):
        for i in range(xs.shape[0]-1):            
            endpoints = np.array([[x_value,ys[i],zs[i]],[x_value,ys[i+1],zs[i+1]]])
            mesh = trimesh.creation.cylinder(radius=0.003, segment=endpoints)
            meshes.append(mesh)    

    tissue_mesh = trimesh.util.concatenate(meshes)
    
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



def record_data(edges_idxs_1, edges_idxs_2, backbone_idxs, init_full_pc, final_full_pc, cylinder_shift, save_path):
    data = {"init_full_pc": init_full_pc, "final_full_pc": final_full_pc, "backbone_idxs": backbone_idxs, \
            "edges_idxs_1": edges_idxs_1,  "edges_idxs_2": edges_idxs_2, "cylinder_shift": cylinder_shift}    
    
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
