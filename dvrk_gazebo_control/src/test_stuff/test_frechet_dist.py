import numpy as np
import trimesh
import open3d
import transformations
from copy import deepcopy
import csv
from matplotlib import pyplot as plt


object_id = 2

# csv file name
filename = "/home/baothach/Downloads/" + f"curve_{object_id}.csv"

# initializing the titles and rows list

xs = []
ys = [] 

# reading csv file
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    
    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        # rows.append(row)
        # print(row)
        xs.append(float(str(row[0])))
        ys.append(float(str(row[1])))
        # print(float(str(row[0])))
        # break

    # get total number of rows
    # print("Total no. of rows: %d"%(csvreader.line_num))

# # print(data)
xs = np.array(xs)
ys = np.array(ys)
sorted_idx = np.argsort(xs)
xs = xs[sorted_idx]
ys = ys[sorted_idx]

# # degree = 10
# # eqn = np.polyfit(xs, ys, deg=degree)
# # y_prime_s = np.zeros(xs.shape)
# # for i, coe in enumerate(eqn):
# #     y_prime_s += coe*(xs**(degree-i))



meshes = []
# ys = y_prime_s
delta_x = 0.1/xs.shape[0]
for i in range(xs.shape[0]-1):
    endpoints = np.array([[xs[i],ys[i],0],[xs[i+1],ys[i+1],0]])
    # endpoints = np.array([[xs[i],ys[i],delta_x*i],[xs[i+1],ys[i+1],delta_x*i]])
    # endpoints = np.array([[0,xs[i],ys[i]],[0,xs[i+1],ys[i+1]]])
    # endpoints = np.array([[delta_x*i,xs[i],ys[i]],[delta_x*i,xs[i+1],ys[i+1]]])
    mesh = trimesh.creation.cylinder(radius=0.02*0.5, segment=endpoints)
    meshes.append(mesh)

goal_mesh = trimesh.util.concatenate(meshes)
# # T = trimesh.transformations.translation_matrix([-0.01,soft_pose_2.p.y-0.1,0.01])
# # if object_id in [1,3]:
# T = trimesh.transformations.translation_matrix([-0.00,-0.46-0.10,0.01])
# # elif object_id == 2:
# #     T = trimesh.transformations.translation_matrix([-0.01,soft_pose_2.p.y-0.11,0.01])

# # T = trimesh.transformations.translation_matrix([0.00,soft_pose_2.p.y-0.1,0.0])
# goal_mesh.apply_transform(T)


goal_pc = trimesh.sample.sample_surface_even(goal_mesh, count=512)[0]    


goal_backbone = np.column_stack((np.array(xs), np.array(ys)))


mesh = trimesh.creation.cylinder(radius=0.02*0.5, height=1.5*0.5)
# current_pc = trimesh.sample.sample_ssurface_even(mesh, count=512)[0]
current_pc = deepcopy(goal_pc)



current_pc_proj = current_pc[:,:2]
# current_backbone = np.mean(current_pc[:,:2], axis=0)
# print(current_backbone.shape)
plt.plot(current_pc_proj[:,0],current_pc_proj[:,1], 'go')

degree = 10
eqn = np.polyfit(current_pc_proj[:,0],current_pc_proj[:,1], deg=degree)
y_prime_s = np.zeros(current_pc_proj[:,0].shape)
x_prime_s = current_pc_proj[:,0]
for i, coe in enumerate(eqn):
    y_prime_s += coe*(x_prime_s**(degree-i))
sorted_idx = np.argsort(x_prime_s)
xs = x_prime_s[sorted_idx]
ys = y_prime_s[sorted_idx]
plt.plot(xs, ys)



plt.plot(goal_backbone[:,0],goal_backbone[:,1])
plt.show()
