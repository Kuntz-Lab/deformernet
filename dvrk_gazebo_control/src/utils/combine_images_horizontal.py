#!/usr/bin/env python3

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import product

from PIL import Image

# key_frame_path = "/home/baothach/shape_servo_data/tube_connect/cylinder_rot_attached/visualization/"
# category = "4"

# processed_image_path = os.path.join(key_frame_path, "processed_key_frames")
# key_frame_path = os.path.join(key_frame_path, "key_frames", category)

# os.makedirs(processed_image_path, exist_ok=True)

# images = sorted(os.listdir(key_frame_path))
# # images = [f"img{i:04}.png" for i in [0,30,35,62,65,71]]
# print(images)

# images = [Image.open(os.path.join(key_frame_path, image)) for image in images] # if image != f"img{72:04}.png"]
# widths, heights = zip(*(i.size for i in images))

# total_width = sum(widths)
# max_height = max(heights)

# new_im = Image.new('RGB', (total_width, max_height))

# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]

# new_im.save(os.path.join(processed_image_path, f"{category}.png"))


key_frame_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/TRO_sim_results_bimanual/quartiles/bimanual_box_node"

processed_image_path = "/home/baothach/shape_servo_data/rotation_extension/visualization/TRO_sim_results_bimanual/processed_images"


os.makedirs(processed_image_path, exist_ok=True)

images = ["min.png", "25.png", "median.png", "75.png", "max.png"]
# print(images)

images = [Image.open(os.path.join(key_frame_path, image)) for image in images] # if image != f"img{72:04}.png"]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save(os.path.join(processed_image_path, f"bimanual_quartiles_node.png"))