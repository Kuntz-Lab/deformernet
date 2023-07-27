#!/usr/bin/env python3
import cv2
import pickle
import numpy as np
import os
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *



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
    sorted_idx = np.argsort(xs)
    xs = xs[sorted_idx]
    ys = ys[sorted_idx]
    zs = np.zeros(ys.shape)
    
    ys_shifted = ys + (-0.46) -0.10
    goal_backbone = np.column_stack((xs, ys_shifted, zs))
    return goal_backbone

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

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

def complete_line(pc):
    # selected_idx = np.argsort(pc[:,1])[-pc.shape[0]//3:]
    break_pt = pc.shape[0]//2
    new_pc = pc[np.argsort(pc[:,1])[-break_pt:-1]]
    target_points = new_pc#pc[selected_idx]
    modified_pc = []
    for i in list(range(target_points.shape[0]-1)):
        x = [target_points[i][0], target_points[i+1][0]]
        y = [target_points[i][1], target_points[i+1][1]]
        vals = np.linspace(target_points[i][0], target_points[i+1][0], 10)
        
        interp = np.interp(vals, x, y)
        modified_pc.extend(list(np.column_stack((vals,interp,np.zeros(vals.shape)))))
        
        # print(start, end)
        # # print(interp)
        # print(vals)
        # break
        
    # print(np.array(modified_pc).shape)
    
    return np.concatenate((pc[np.argsort(pc[:,1])[:-break_pt]], np.array(modified_pc)), axis=0)
        

input_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_unprocessed" 
output_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_test" 
os.makedirs(output_path, exist_ok=True)

# with open(os.path.join(input_path, "goal_pc.pickle"), 'rb') as handle:
#     points = pickle.load(handle)

# print(sorted_idxs)
points1 = complete_line(get_tube_connect_goal_pc(object_id=1))
# points1 = get_tube_connect_goal_pc(object_id=1)
goal_xs_1, goal_ys_1 = get_goal_projected_on_image(points1)
points2 = get_tube_connect_goal_pc(object_id=2)
goal_xs_2, goal_ys_2 = get_goal_projected_on_image(points2)

pc_1 = np.column_stack((np.array(goal_ys_1), np.array(goal_xs_1)))
pc_2 = np.column_stack((np.array(goal_ys_2), np.array(goal_xs_2)))



pcs = [pc_1, pc_2]

radius = 1 #1        
# Red color in BGR
color = (0, 0, 255)
thickness = 2 

for i in range(0, 1):
    image = cv2.imread(os.path.join(input_path, f'img{i:03}.png'))

    overlay = image.copy()


    radius = 2 #1        
    # Red color in BGR
    color = (0, 0, 255)
    thickness = 5
    

    
    for j, pc in enumerate(pcs):
        for point in pc: 
            
            if j == 0:
                color = (0, 0, 255) # color in BGR
            elif j == 1:
                color = (0, 0, 255) # color in BGR
                # color = (0, 255, 0) # color in BGR
            
            overlay = cv2.circle(overlay, tuple(point), radius, color, thickness)        


    alpha = 0.7  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


    # for point in points:
    #     image = cv2.circle(image, tuple(point), radius, color, thickness)   
        
    cv2.imwrite(os.path.join(output_path, f'img{i:03}.png'), image)