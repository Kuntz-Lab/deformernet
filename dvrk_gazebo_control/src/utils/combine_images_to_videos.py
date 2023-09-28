import os
from camera_utils import add_frames

# img_dir = "/home/baothach/shape_servo_data/new_task/plane_vis/6_4"
# img_dir = "/home/baothach/shape_servo_data/tissue_wrap/visualization/3" 
# img_dir = "/home/baothach/shape_servo_data/tissue_wrap/visualization/final_TRO_vis"
# img_dir = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/evaluate/visualization/pointconv_1000/run2"
img_dir = "/home/baothach/shape_servo_data/tanner/evaluate/visualization/run1"

os.chdir(img_dir)


# # os.system("ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4")
# os.system("ffmpeg -framerate 10 -i img%03d.png -pix_fmt yuv420p tissue_wrap.mp4")
# os.system("ffmpeg -start_number 0 -framerate 10 -i img%03d.png -pix_fmt yuv420p tube_connect_sim.mp4")

# # add_frames(img_dir, source_frame=32, num_new_frames=20)
# os.system("ffmpeg -framerate 10 -i img%03d.png -pix_fmt yuv420p tissue_wrap_sim_goal_gen.mp4")

# add_frames(img_dir, source_frame=291, num_new_frames=40)
os.system("ffmpeg -framerate 20 -i img%03d.png -pix_fmt yuv420p retraction_kidney_sim_goal_gen.mp4")