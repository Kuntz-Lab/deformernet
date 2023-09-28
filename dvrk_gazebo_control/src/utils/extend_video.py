import numpy as np
import cv2
import os


image_path = "/home/baothach/shape_servo_data/tissue_wrap/visualization/final_TRO_vis"

num_images = len(os.listdir(image_path))

final_image = cv2.imread(os.path.join(image_path, f'img{(num_images-1):03}.png'))

for i in range(20):
    cv2.imwrite(os.path.join(image_path, f'img{num_images+i:03}.png'), final_image)