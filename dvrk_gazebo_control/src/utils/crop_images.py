#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import cv2


key_frame_path = "/home/baothach/shape_servo_data/real_robot/TRO/videos/key_frames"
category = ["single", "video_cylinder_goal_2_init_3"]

cropped_image_path = os.path.join(key_frame_path, category[0], f"cropped_{category[1]}")
original_image_path = os.path.join(key_frame_path, category[0], category[1])
os.makedirs(cropped_image_path, exist_ok=True)

images = [f"img{i:04}.png" for i in [0]]


if category[0] == "bimanual":
  y_1, y_2 = 0, 620   # 0, 720
  x_1 = 1280//5 - 50
  x_2 = x_1 + 720
  print(x_1, x_2)
if category[0] == "single":
  y_1, y_2 = 100, 720   # 0, 720
  x_1 = 1280//5 #- 50
  x_2 = x_1 + 720
  print(x_1, x_2)

for image_name in os.listdir(original_image_path):
# for image_name in images:

  image_path = os.path.join(original_image_path, image_name)

  img = cv2.imread(image_path)
  crop_img = img[y_1:y_2, x_1:x_2]
  
  # cv2.imshow("cropped", crop_img)
  # cv2.waitKey(0)
  
  cv2.imwrite(os.path.join(cropped_image_path, image_name), crop_img)