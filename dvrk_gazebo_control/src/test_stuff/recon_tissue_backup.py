from re import X
import numpy as np
import trimesh
from trimesh.geometry import align_vectors
from trimesh.creation import _segment_to_cylinder as seg_to_object
import open3d
import pickle

with open("/home/baothach/Downloads/wrap_tissue_backbone.pickle", 'rb') as handle:
    data = pickle.load(handle)    

xs = data[:,0]*0
ys = data[:,1]
zs = data[:,2]
# print(xs.shape, ys.shape, zs.shape)
# sorted_idx = np.argsort(ys)
# xs = xs[sorted_idx]
# ys = ys[sorted_idx]
# zs = zs[sorted_idx]

# xs = np.arange(20) *0.01
# ys = np.arange(20)*0
# zs = np.concatenate((np.arange(10) *0.01, (np.array(range(10))[::-1])*0.01))
# print(xs.shape, ys.shape, zs.shape)

# edge_lengths = np.array([0.015*0.7, 0.15*0.7, 0.1])
edge_lengths = np.array([0.01, 0.01, 0.01])

endpoints = np.vstack((xs,ys,zs)).T
print(endpoints.shape)
# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(endpoints)
# pcd2.paint_uniform_color([1,0,0])
# open3d.visualization.draw_geometries([pcd2]) 


meshes = []
endpoints_vis = []

for x_value in np.linspace(start=-0.15*0.7/2, stop=0.15*0.7/2, num = 10):
    for i in range(xs.shape[0]-1):
        # print(abs(ys[i+1] - ys[i]))
        
        if True: #i != 0 and abs(ys[i] - ys[i-1]) > 0.005:
            endpoints_vis.append([xs[i],ys[i],zs[i]])
        
        # if i != 0 and abs(ys[i] - ys[i-1]) <= 0.005:
        #     print("-----")
        #     continue

        # endpoints = np.array([[xs[i],ys[i],zs[i]],[xs[i+1],ys[i+1],zs[i+1]]])
        # transform, height = seg_to_object(endpoints)
        # edge_lengths_new = edge_lengths
        # edge_lengths_new[2] = height
        # # print(transform, height)
        # mesh = trimesh.creation.box(edge_lengths_new, transform)
        # # mesh = trimesh.creation.box(edge_lengths)
        
        endpoints = np.array([[x_value,ys[i],zs[i]],[x_value,ys[i+1],zs[i+1]]])
        mesh = trimesh.creation.cylinder(radius=0.008, segment=endpoints)
        # transform, height = seg_to_object(endpoints)
        # mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=height, transform=transform)


        meshes.append(mesh)


pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(np.array(endpoints_vis))
pcd2.paint_uniform_color([1,0,0])


goal_mesh = trimesh.util.concatenate(meshes)

goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(goal_pc)
coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=.2)

# open3d.visualization.draw_geometries([coor, pcd.paint_uniform_color([0,0,0]), pcd2]) 


trimesh.Scene(meshes).show() 
# trimesh.creation.box