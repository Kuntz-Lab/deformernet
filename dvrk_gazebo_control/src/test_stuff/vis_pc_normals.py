from re import X
import numpy as np
import trimesh
from trimesh.geometry import align_vectors
from trimesh.creation import _segment_to_cylinder as seg_to_object
import open3d
import pickle
import transformations
from copy import deepcopy

quat = [0.5, 0.5,0.5, 0.5]

trans_mat = transformations.quaternion_matrix(quat)
trans_mat[:3,3] = np.array([0.00, -0.5, 0.04])


# cylinder_mesh = trimesh.creation.cylinder(radius=0.015, height=0.1)
cylinder_mesh = trimesh.creation.annulus(r_min=0.0149*0.6, r_max=0.015*0.6, height=0.1, transform = trans_mat)
pc, faces = trimesh.sample.sample_surface_even(cylinder_mesh, count=1024)  
cloud = trimesh.PointCloud(pc)
normals = cylinder_mesh.face_normals[faces]

tissue_mesh = trimesh.creation.annulus(r_min=0.013*2.5, r_max=0.015*2.5, height=0.1, transform = trans_mat)

# normals = get_normals_cylinder(mesh)
ray_origins, ray_directions = pc, normals/25 #pc, normals/200


# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(np.hstack((
    ray_origins,
    ray_origins + ray_directions)).reshape(-1, 2, 3))

# scene = trimesh.Scene([
#     cloud,
#     ray_visualize])

# scene.show()

# make mesh transparent- ish
# print(tissue_mesh.visual.face_colors)
tissue_mesh.visual.face_colors = [255, 0, 0,200]


# scene = trimesh.Scene([
#     cloud, tissue_mesh,
#     ray_visualize])


T = trimesh.transformations.translation_matrix([0, 0.1, 0.00])
cylinder_mesh.apply_transform(T)
tissue_mesh_2 = deepcopy(tissue_mesh)
tissue_mesh_2.apply_transform(T)

scene = trimesh.Scene([
    cloud, tissue_mesh,
    tissue_mesh_2, cylinder_mesh,
    ray_visualize])

scene.show()