#!/usr/bin/env python3
import cv2
import pickle
import numpy as np
import os
import trimesh
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *


def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def get_tube_connect_goal_pc(object_id):

    import csv
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

            xs.append(float(str(row[0])))
            ys.append(float(str(row[1])))


    # # print(data)
    xs = np.array(xs)
    ys = np.array(ys)
    # sorted_idx = np.argsort(xs)
    
    if object_id == 1:
        sorted_idx = np.argsort(xs)[6:]
    elif object_id == 2:
        sorted_idx = np.argsort(xs)[:-3]    
    
    xs = xs[sorted_idx]
    ys = ys[sorted_idx]
    zs = np.zeros(ys.shape)
    
    meshes = []
    # ys = y_prime_s
    delta_x = 0.05/xs.shape[0]
    for i in range(xs.shape[0]-1):
        endpoints = np.array([[xs[i],ys[i],0],[xs[i+1],ys[i+1],0]])

        if object_id == 5:
            endpoints = np.array([[delta_x*i,xs[i],ys[i]],[delta_x*i,xs[i+1],ys[i+1]]])
            # endpoints = np.array([[0,xs[i],ys[i]],[0,xs[i+1],ys[i+1]]])
        elif object_id == 6:
            endpoints = np.array([[-delta_x*(xs.shape[0]-i),xs[i],ys[i]],[-delta_x*(xs.shape[0]-i),xs[i+1],ys[i+1]]])
        mesh = trimesh.creation.cylinder(radius=0.02*0.5, segment=endpoints)
        meshes.append(mesh)

    goal_mesh = trimesh.util.concatenate(meshes)
    T = trimesh.transformations.translation_matrix([-0.00,-0.46-0.10,0.01])

    
    if object_id == 5:
        T = trimesh.transformations.translation_matrix([0.05,-0.46-0.10,0.01])
    elif object_id==6:
        T = trimesh.transformations.translation_matrix([0.05,-0.46-0.10,0.01])

    goal_mesh.apply_transform(T)


    goal_pc = trimesh.sample.sample_surface_even(goal_mesh, count=3072)[0]    
    

    return goal_pc

def get_goal_projected_on_image(goal_pc):
    
    vis_cam_width = 1000
    vis_cam_height = 1000
    cam_view_matrix = np.array([[-0.7071068,  -0.5144958,   0.48507124,  0.],
                        [ 0.7071068,  -0.5144958,   0.48507124,  0.        ],
                        [ 0.,          0.6859944,   0.7276069 ,  0.        ],
                        [ 0.30405593, -0.2212332,  -0.20372993,  1.        ]])
    

    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(cam_view_matrix)
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    

    return x_s, y_s

input_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_unprocessed" 
output_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_3" 
os.makedirs(output_path, exist_ok=True)



num_pts = 128



points1 = down_sampling(get_tube_connect_goal_pc(object_id=1), num_pts=num_pts)
points2 = down_sampling(get_tube_connect_goal_pc(object_id=2), num_pts=num_pts)

pc = np.concatenate((points1, points2), axis=0)
sorted_idx = np.argsort(pc[:,0])
points1 = pc[sorted_idx[:pc.shape[0]//2]]
points2 = pc[sorted_idx[pc.shape[0]//2:]]


goal_xs_1, goal_ys_1 = get_goal_projected_on_image(points1)

goal_xs_2, goal_ys_2 = get_goal_projected_on_image(points2)

pc_1 = np.column_stack((np.array(goal_ys_1), np.array(goal_xs_1)))
pc_2 = np.column_stack((np.array(goal_ys_2), np.array(goal_xs_2)))
pcs = [pc_1, pc_2]



radius = 1 #1        
# Red color in BGR
color = (0, 0, 255)
thickness = 2 

for i in range(0, len(os.listdir(input_path))-1):
# for i in range(74, 75):
    image = cv2.imread(os.path.join(input_path, f'img{i:03}.png'))

    overlay = image.copy()


    radius = 2 #1        
    # Red color in BGR
    color = (0, 0, 255)
    thickness = 5
    
    # for point in points: 
    #     overlay = cv2.circle(overlay, tuple(point), radius, color, thickness)        


    # alpha = 0.7  # Transparency factor.

    # # Following line overlays transparent rectangle over the image
    # image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


    # # for point in points:
    # #     image = cv2.circle(image, tuple(point), radius, color, thickness)   
        
    # cv2.imwrite(os.path.join(output_path, f'img{i:03}.png'), image)
    
    for j, pc in enumerate(pcs):
        for point in pc: 
            
            if j == 0:
                color = (0, 0, 255) # color in BGR
            elif j == 1:
                color = (0, 255, 0) # color in BGR
            
            overlay = cv2.circle(overlay, tuple(point), radius, color, thickness)        


    alpha = 0.7  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


    # for point in points:
    #     image = cv2.circle(image, tuple(point), radius, color, thickness)   

    # cv2.imshow('asxasx',image)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
        
    cv2.imwrite(os.path.join(output_path, f'img{i:03}.png'), image)