import numpy as np
import trimesh
import open3d
import transformations
from copy import deepcopy

# # def get_cylinder_goal_pc(radius, height, translation=None, num_points=1024, vis=False):
    
# #     rot_mat = transformations.euler_matrix(0, np.pi/2, 0)
# #     if translation is not None:
# #         # T = trimesh.transformations.translation_matrix(translation)
# #         rot_mat[:3,3] = np.array(translation)
    
# #     mesh = trimesh.creation.cylinder(radius=radius, height=height, transform=rot_mat)
    
# #     # mesh.apply_transform(T)
    
# #     sampled_pts = trimesh.sample.sample_surface(mesh, count=1024)[0]

# #     if vis:
# #         pcd = open3d.geometry.PointCloud()
# #         pcd.points = open3d.utility.Vector3dVector(sampled_pts)
# #         open3d.visualization.draw_geometries([pcd]) 

# #     return sampled_pts

# mesh0 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5)

# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# mesh1 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)

# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# mesh2.apply_transform(rot_mat)


# second_mesh2 = deepcopy(mesh2)
# T = trimesh.transformations.translation_matrix([0, 0, -0.5])
# second_mesh2.apply_transform(T)
# rot_mat = transformations.euler_matrix(np.pi/2, 0, 0)
# second_mesh2.apply_transform(rot_mat)

# # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# # rot_mat[:3,3] = np.array([0,0.4,0])
# # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)
# # rot_mat = transformations.euler_matrix(-np.pi/4, 0, 0)
# # second_mesh2.apply_transform(rot_mat)
# # 
# # transformed_mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transformations=rot)
# # mesh = trimesh.creation.cylinder(radius=radius, height=height, transformations=rot_mat)

# # meshes = [mesh0, mesh1, mesh2]
# meshes = [mesh0, mesh1, mesh2, second_mesh2]

# # trimesh.Scene(meshes).show()    

# T = trimesh.transformations.translation_matrix([0, -0.714+0.25*np.cos(np.pi/4), -0.05+0.25*np.sin(np.pi/4)])
# mesh2.apply_transform(T)
# T = trimesh.transformations.translation_matrix([0, -0.714+-0.25*np.cos(np.pi/4), -0.05+0.25*np.sin(np.pi/4)])
# second_mesh2.apply_transform(T)

# goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])
# rot_mat = transformations.euler_matrix(0, 0, -np.pi/4)
# rot_mat[:3,3] = np.array([0.26,0,0])
# goal_mesh.apply_transform(rot_mat)

# goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(goal_pc)
# coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# open3d.visualization.draw_geometries([coor, pcd]) 





#############################################
# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.35*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# mesh2.apply_transform(rot_mat)


# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# second_mesh2.apply_transform(rot_mat)


# T = trimesh.transformations.translation_matrix([0, 0, -0.5])
# second_mesh2.apply_transform(T)
# rot_mat = transformations.euler_matrix(np.pi/2, 0, 0)
# second_mesh2.apply_transform(rot_mat)

# # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# # rot_mat[:3,3] = np.array([0,0.4,0])
# # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)
# # rot_mat = transformations.euler_matrix(-np.pi/4, 0, 0)
# # second_mesh2.apply_transform(rot_mat)
# # 
# # transformed_mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transformations=rot)
# # mesh = trimesh.creation.cylinder(radius=radius, height=height, transformations=rot_mat)

# # meshes = [mesh0, mesh1, mesh2]
# # meshes = [mesh0, mesh1, mesh2, second_mesh2]

# # trimesh.Scene(meshes).show()    

# T = trimesh.transformations.translation_matrix([0, -0.714+0.25*np.cos(np.pi/4), -0.1+0.25*np.sin(np.pi/4)])
# mesh2.apply_transform(T)
# T = trimesh.transformations.translation_matrix([0, -0.714-0.3*np.cos(np.pi/4), -0.1+0.3*np.sin(np.pi/4)])
# second_mesh2.apply_transform(T)

# goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])
# # rot_mat = transformations.euler_matrix(0, 0, -np.pi/4)
# # rot_mat[:3,3] = np.array([0.26,0,0])
# # goal_mesh.apply_transform(rot_mat)

# goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(goal_pc)
# coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# open3d.visualization.draw_geometries([coor, pcd]) 

###################################################
# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.25*0.5, transform=rot_mat)
# # rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# # mesh2.apply_transform(rot_mat)


# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# second_mesh2.apply_transform(rot_mat)


# T = trimesh.transformations.translation_matrix([0, 0.08, 0.03])
# second_mesh2.apply_transform(T)
# # # rot_mat = transformations.euler_matrix(np.pi/2, 0, 0)
# # # second_mesh2.apply_transform(rot_mat)

# # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# # rot_mat[:3,3] = np.array([0,0.4,0])
# # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)
# # rot_mat = transformations.euler_matrix(-np.pi/4, 0, 0)
# # second_mesh2.apply_transform(rot_mat)
# # 
# # transformed_mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transformations=rot)
# # mesh = trimesh.creation.cylinder(radius=radius, height=height, transformations=rot_mat)

# # meshes = [mesh0, mesh1, mesh2]
# # meshes = [mesh0, mesh1, mesh2, second_mesh2]

# # trimesh.Scene(meshes).show()    

# # T = trimesh.transformations.translation_matrix([0, -0.714+0.25*np.cos(np.pi/4), -0.1+0.25*np.sin(np.pi/4)])
# # mesh2.apply_transform(T)
# # T = trimesh.transformations.translation_matrix([0, -0.714-0.3*np.cos(np.pi/4), -0.1+0.3*np.sin(np.pi/4)])
# # second_mesh2.apply_transform(T)

# goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])
# T = trimesh.transformations.translation_matrix([0, 0.4-0.86, 0])
# goal_mesh.apply_transform(T)
# # rot_mat = transformations.euler_matrix(0, 0, -np.pi/4)
# # rot_mat[:3,3] = np.array([0.26,0,0])
# # goal_mesh.apply_transform(rot_mat)

# goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(goal_pc)
# coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# open3d.visualization.draw_geometries([coor, pcd]) 


###################################################
# # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# # # rot_mat[:3,3] = np.array([0, -0.5, 0.02])
# # # mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)

# two_robot_offset = 1.0
# # rot_mat[:3,3] = np.array([0, -1.35*two_robot_offset/3.5, 0.02])
# # mesh1 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)
# # rot_mat[:3,3] = np.array([0, -2.15*two_robot_offset/3.5, 0.02])
# # mesh2  = trimesh.creation.cylinder(radius=0.02*0.5, height=0.4*0.5, transform=rot_mat)


# # # goal_mesh = mesh
# # goal_mesh = trimesh.util.concatenate([mesh1, mesh2])

# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.3*0.5, transform=rot_mat)



# rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
# second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# second_mesh2.apply_transform(rot_mat)


# T = trimesh.transformations.translation_matrix([0, 0.1, 0.02])
# second_mesh2.apply_transform(T)

# rot_mat = transformations.euler_matrix(np.pi/3, 0 , 0)
# rot_mat[:3,3] = np.array([0, 0.17, 0.00])
# mesh3 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.3*0.5, transform=rot_mat)


# goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2, mesh3])
# T = trimesh.transformations.translation_matrix([0, 0.4-1.0, 0])
# goal_mesh.apply_transform(T)




# goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(goal_pc)
# coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# open3d.visualization.draw_geometries([coor, pcd]) 


###################################################

rot_mat = transformations.euler_matrix(-np.pi/4, 0 , -np.pi/8)
mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.3*0.5, transform=rot_mat)


rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# mesh2.apply_transform(rot_mat)
T = trimesh.transformations.translation_matrix([0.02, 0.08, 0.05])
mesh2.apply_transform(T)
goal_mesh_1 = trimesh.util.concatenate([mesh, mesh2])

rot_mat = transformations.euler_matrix(np.pi/4, 0 , np.pi/8)
mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=0.3*0.5, transform=rot_mat)


rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
# rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
# mesh2.apply_transform(rot_mat)
T = trimesh.transformations.translation_matrix([0.02, -0.08, 0.05])
mesh2.apply_transform(T)
goal_mesh_2 = trimesh.util.concatenate([mesh, mesh2])
T = trimesh.transformations.translation_matrix([0,0.25,0])
goal_mesh_2.apply_transform(T)
goal_mesh = trimesh.util.concatenate([goal_mesh_1, goal_mesh_2])


# goal_mesh = mesh2




goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(goal_pc)
coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

open3d.visualization.draw_geometries([coor, pcd]) 