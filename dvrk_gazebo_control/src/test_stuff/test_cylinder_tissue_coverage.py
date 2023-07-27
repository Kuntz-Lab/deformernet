from re import X
import numpy as np
import trimesh
from trimesh.geometry import align_vectors
from trimesh.creation import _segment_to_cylinder as seg_to_object
import open3d
import pickle
import transformations

import transformations
from copy import deepcopy

quat = [0.5, 0.5,0.5, 0.5]

trans_mat = transformations.quaternion_matrix(quat)

cylinder_shift = np.array([0.05,-0.05,0.05])
trans_mat[:3,3] = np.array([0.00+0.00, -0.5+0.00, 0.04]) + cylinder_shift


# cylinder_mesh = trimesh.creation.cylinder(radius=0.015, height=0.1)
cylinder_mesh = trimesh.creation.annulus(r_min=0.0149*0.6, r_max=0.015*0.6, height=0.1, transform = trans_mat)
pc, faces = trimesh.sample.sample_surface_even(cylinder_mesh, count=1024)  
cloud = trimesh.PointCloud(pc)
normals = cylinder_mesh.face_normals[faces]



# normals = get_normals_cylinder(mesh)
ray_origins, ray_directions = pc, normals/25 #pc, normals/200


# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(np.hstack((
    ray_origins,
    ray_origins + ray_directions)).reshape(-1, 2, 3))



with open("/home/baothach/Downloads/wrap_tissue_backbone.pickle", 'rb') as handle:
    data = pickle.load(handle)    

xs = data[:,0]#*0
ys = data[:,1]
zs = data[:,2]

edge_lengths = np.array([0.01, 0.01, 0.01])

endpoints = np.vstack((xs,ys,zs)).T
print(endpoints.shape)
# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(endpoints)
# pcd2.paint_uniform_color([1,0,0])
# open3d.visualization.draw_geometries([pcd2]) 


meshes = []
endpoints_vis = []

for x_value in np.linspace(start=-0.15*0.7/2+cylinder_shift[0], stop=0.15*0.7/2+cylinder_shift[0], num = 100):
# for x_value in [0]:
    for i in range(xs.shape[0]-1):

        
        if True: #i != 0 and abs(ys[i] - ys[i-1]) > 0.005:
            endpoints_vis.append([xs[i],ys[i],zs[i]])
        
        endpoints = np.array([[x_value,ys[i],zs[i]],[x_value,ys[i+1],zs[i+1]]])
        mesh = trimesh.creation.cylinder(radius=0.003, segment=endpoints)
        # transform, height = seg_to_object(endpoints)
        # mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=height, transform=transform)


        meshes.append(mesh)


# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(np.array(endpoints_vis))
# pcd2.paint_uniform_color([1,0,0])


tissue_mesh = trimesh.util.concatenate(meshes)
tissue_mesh.visual.face_colors = [255, 0, 0,200]


# run the mesh- ray test
locations, index_ray, index_tri = tissue_mesh.ray.intersects_location(
    ray_origins=ray_origins,
    ray_directions=ray_directions)
intersection = trimesh.points.PointCloud(locations)

index_ray = set(index_ray)
print("locations length:", len(index_ray))
print("Percent coverage:", len(index_ray)/pc.shape[0])


scene = trimesh.Scene([
    cylinder_mesh, tissue_mesh,
    intersection,
    ray_visualize])


scene.show()