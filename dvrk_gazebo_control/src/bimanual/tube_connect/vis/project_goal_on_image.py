#!/usr/bin/env python3
import cv2
import pickle
import numpy as np
import os
import sys
sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *


def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

input_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_unprocessed" 
output_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/4_3" 
os.makedirs(output_path, exist_ok=True)

with open(os.path.join(input_path, "goal_pc.pickle"), 'rb') as handle:
    points = pickle.load(handle)

num_pts = 128
pc = down_sampling(points, num_pts=num_pts) 
sorted_idxs = np.argsort(pc[:,1])
# print(sorted_idxs)
pc_1 = pc[sorted_idxs[:round(num_pts/3)+4]] 
pc_2 = pc[sorted_idxs[round(num_pts/3)+4:]] 
pcs = [pc_1, pc_2]

radius = 1 #1        
# Red color in BGR
color = (0, 0, 255)
thickness = 2 

# for i in range(0, len(os.listdir(input_path))-1):
for i in range(0, 1):
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

    cv2.imshow('asxasx',image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
        
    # cv2.imwrite(os.path.join(output_path, f'img{i:03}.png'), image)