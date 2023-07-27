#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('dvrk_gazebo_control')
sys.path.append(pkg_path + '/src')
import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import copy, deepcopy
import rospy
# from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle5 as pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import transformations
import random

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.25
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)






if __name__ == "__main__":

    main_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinderes_1000Pa/evaluate"
    objects_path = "/home/baothach/shape_servo_data/evaluation"

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_name", "type": str, "default": 'cylinder_0', "help": "select variations of a primitive shape"},
            {"name": "--obj_type", "type": str, "default": 'cylinder_1k', "help": "cylinder1k, cylinder5k, etc."},
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"



    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        # print("=================sim_params.dt:", sim_params.dt)
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 10#10
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

    # Get primitive shape dictionary to know the dimension of the object   
    if args.inside:
        # object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside_sample_ratio")
        if args.obj_name[0] == "c":
            object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    if args.obj_name[0] == "b":
        h = data["height"]
        w = data["width"]
        thickness = data["thickness"]
    elif args.obj_name[0] == "c":
        r = data["radius"]
        h = data["height"]
    elif args.obj_name[0] == "h":
        r = data["radius"]
        o = data["origin"]

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up ground
    gym.add_ground(sim, plane_params)

    # create viewer
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()

    # load robot assets
    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    # pose_2.p = gymapi.Vec3(0.0, 0.85, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.001#0.0001


    asset_root = "./src/dvrk_env"
    kuka_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"


    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40000.

    print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
    kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)

    if args.inside:
        # asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside_smaller")
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside_sample_ratio")#sample_ratio")
        if args.obj_name[0] == "c":
            asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "outside") 

    
    soft_thickness = 0.0005    # important to add some thickness to the soft body to avoid interpenetrations
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True


    soft_assets = []
    soft_poses = []
    origin_x, origin_y = -0.3, -2
    num_objs = 10
    num_rows = 3
    for k in range(num_objs):

        soft_asset_file = f"cylinder_{k}.urdf"    
        # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/inside"
        # soft_asset_file = "long_cylinder.urdf"
        # asset_root = "/home/baothach/Downloads"
        # soft_asset_file = "test_cylinder.urdf"


        soft_pose = gymapi.Transform()
        # soft_pose.p = gymapi.Vec3(origin_x + k // 3 * 0.3, origin_y + k % 3 * 0.3, 0.02)
        # soft_pose.p = gymapi.Vec3(origin_x + k // 3 * 0.3, origin_y + k % 3 * 0.3, thickness/2*0.7)
        x = origin_x + k // num_rows * 0.3
        y = origin_y + k % num_rows * 0.3
        if args.obj_name[0] == "b": 
            soft_pose.p = gymapi.Vec3(x, y, thickness/2*0.5)
            soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
        elif args.obj_name[0] == "c": 
            soft_pose.p = gymapi.Vec3(x, y, r/2.0)
            soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
        elif args.obj_name[0] == "h":
            soft_pose.p = gymapi.Vec3(x, y, -o/2.+0.01)
            # soft_pose.p = gymapi.Vec3(x, y, (r-o)/2.)
            # soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
        # rot_angle = np.pi/2 + np.random.uniform(low = -np.pi/4.5, high = np.pi/4.5) 
        # # rot_angle = np.pi/2-np.pi/4.5
        # print(" ******* Current orientation angle: " + str(rot_angle))
        # quat = transformations.quaternion_from_euler(*[rot_angle,0,0])
        # soft_pose.r = gymapi.Quat(*quat)


        soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
        
        soft_poses.append(deepcopy(soft_pose))
        soft_assets.append(soft_asset)

        
    
 
    
    # set up the env grid
    # spacing = 0.75
    spacing = 0.0
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
  

    # cache some common handles for later use
    envs = []
    envs_obj = []
    kuka_handles = []
    kuka_handles_2 = []
    object_handles = []
    


    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)        
        

        # add soft obj        
        for k in range(num_objs):  
      
            
            soft_actor = gym.create_actor(env, soft_assets[k], soft_poses[k], "soft", i, 0)
            object_handles.append(soft_actor)


        kuka_handles_2.append(kuka_2_handle)

    import random
    l_color = gymapi.Vec3(0.3,0.3,0.3)
    l_ambient = gymapi.Vec3(1,1,1)
    l_direction = gymapi.Vec3(1,1,1)
    gym.set_light_parameters(sim, 0, l_color, l_ambient, l_direction)


    # Camera setup
    if not args.headless:
        # cam_pos = gymapi.Vec3(1, 0.5, 1)
        # # cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # # cam_pos = gymapi.Vec3(0.3, -0.1, 0.5)  # final setup for thin layer tissue
        # # cam_pos = gymapi.Vec3(0.5, -0.36, 0.3)
        # # cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        # cam_pos = gymapi.Vec3(0.5, -0.8, 0.5)
        cam_pos = gymapi.Vec3(0.0, -1.360002, 1)
        cam_target = gymapi.Vec3(0.0, -1.36, 0.1)

        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    close_viewer = False
    while (not close_viewer): 
        # print("camera matrix: ", np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])).shape)
        # print("projection matrix: ", gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i]))
        # print("color_image", gym.get_camera_image(sim, envs_obj[i], cam_handles[0], gymapi.IMAGE_COLOR))


        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 


  
   
                
                
   

        
        # if sample_count == max_sample_count:             
        #     all_done = True    

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  




    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)