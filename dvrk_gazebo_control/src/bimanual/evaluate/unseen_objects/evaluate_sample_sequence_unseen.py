#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys
# from turtle import width

from numpy.lib.polynomial import polyint
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
# #import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle5 as pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import cv2
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2
from scipy import interpolate


from utils.miscellaneous_utils import down_sampling, get_extents_object

import torch



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 1.0



def init():
    for i in range(num_envs):
        # Kuka 1
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.24
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles[i], davinci_dof_states, gymapi.STATE_POS)

        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.24
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)


def get_point_cloud():
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  
    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))
    # open3d.visualization.draw_geometries([pcd])     
    # return list(point_cloud)
    return point_cloud.astype('float32')


def get_partial_point_cloud(i):

    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    print("Converting Depth images to point clouds. Have patience...")
    # for c in range(len(cam_handles)):
    
    # print("Deprojecting from camera %d, %d" % i))
    # Retrieve depth and segmentation buffer
    depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0])))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 11] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] < -3:
                continue

            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            # print("p2:", p2)
            if p2[0, 2] > 0.01:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(points))
    # open3d.visualization.draw_geometries([pcd]) 

    # return points
    return np.array(points).astype('float32')

def get_goal_projected_on_image(goal_pc, i, thickness = 0):
    # proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    # fu = 2/proj[0, 0]
    # fv = 2/proj[1, 1]
    

    u_s =[]
    v_s = []
    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], vis_cam_handles[0]))
        # print("point_cam_frame:", point_cam_frame)
        # image_coordinates = (gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[0]) * point_cam_frame)
        # print("image_coordinates:",image_coordinates)
        # u_s.append(image_coordinates[1, 0]/image_coordinates[2, 0]*2)
        # v_s.append(image_coordinates[0, 0]/image_coordinates[2, 0]*2)
        # print("fu fv:", fu, fv)
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    # print(centerU - np.array(u_s)*cam_width)
    # y_s = (np.array(u_s)*cam_width).astype(int)
    # x_s = (np.array(v_s)*cam_height).astype(int)
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    

    if thickness != 0:
        new_y_s = deepcopy(list(y_s))
        new_x_s = deepcopy(list(x_s))
        for y, x in zip(y_s, x_s):
            for t in range(1, thickness+1):
                new_y_s.append(max(y-t,0))
                new_x_s.append(max(x-t,0))
                new_y_s.append(max(y-t,0))
                new_x_s.append(min(x+t, vis_cam_height-1))                
                new_y_s.append(min(y+t, vis_cam_width-1))
                new_x_s.append(max(x-t,0))                    
                new_y_s.append(min(y+t, vis_cam_width-1))                
                new_x_s.append(min(x+t, vis_cam_height-1))
        y_s = new_y_s
        x_s = new_x_s
    # print(x_s)
    return x_s, y_s

if __name__ == "__main__":



    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_name", "type": str, "default": None, "help": "chicken_breast, meat, kidney, etc."},
            {"name": "--obj_idx", "type": int, "default": None, "help": "Index of the object. From 0 to 99."},
            {"name": "--model_category", "type": str, "default": "combined", "help": "which trained DeformerNet to use? Options: combined, box_1k, cylinder_5k, hemis_10k, etc. "},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
        
    args.headless = args.headless == "True"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/unseen_objects/evaluate"
    objects_path = "/home/baothach/sim_data/Custom/Custom_objects/random_stuff"
    object_meshes_path = os.path.join(objects_path, "mesh")

    model_category = args.model_category


    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -10)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        # print("=================sim_params.dt:", sim_params.dt)
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 10#4
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

    


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
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -two_robot_offset, ROBOT_Z_OFFSET) 
    
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


    asset_root = os.path.join(objects_path, "urdf", args.obj_name)
    soft_asset_file = f"{args.obj_name}_{8}.urdf" 
    tet_file = os.path.join(object_meshes_path, f"{args.obj_name}.tet")
    extents = get_extents_object(tet_file)



    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, -extents[0][2])
    soft_thickness = 0.001#0.0005    # important to add some thickness to the soft body to avoid interpenetrations





    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

        
    
 
    
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
    

    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add kuka
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=11)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 2, segmentationId=11)       
        

        # add soft obj        
        env_obj = env
        # env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)


        kuka_handles.append(kuka_handle)
        kuka_handles_2.append(kuka_2_handle)



    dof_props_2 = gym.get_asset_dof_properties(kuka_asset)
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    vel_limits = dof_props_2['velocity']    
    print("======vel_limits:", vel_limits)  

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        # cam_pos = gymapi.Vec3(0.5, -0.8, 0.5)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 256
    cam_height = 256
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(0.17, -0.62-(two_robot_offset/2 - 0.42), 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-0.86-(two_robot_offset/2 - 0.42), 0.01)) 
  

    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props_2)

    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 1000
    vis_cam_height = 1000
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height

    # # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.2, 0.2))  # sample 4
    # # vis_cam_positions.append(gymapi.Vec3(0.1, -0.3, 0.1)) # worst
    # # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.3, 0.1)) # 2nd worst, used for all cases
    # # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.35, 0.1)) # 75 percentile, index 59, IMSR
    # # vis_cam_positions.append(gymapi.Vec3(0.1, -0.6, 0.15))
    # # vis_cam_positions.append(gymapi.Vec3(0.1, -0.4, 0.15))
    # # vis_cam_positions.append(gymapi.Vec3(0.1, -0.4, 0.1))
    # vis_cam_positions.append(gymapi.Vec3(0.2, -0.65, 0.15))
    # # vis_cam_positions.append(gymapi.Vec3(-0.05, -0.65, 0.1))

    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))

    ### Bimanual Box
    #Max
    # vis_cam_positions.append(gymapi.Vec3(-0.2, -0.5, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    # vis_cam_positions.append(gymapi.Vec3(-0.2, -0.5, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
            
    # #75th
    # vis_cam_positions.append(gymapi.Vec3(0.15, -0.5, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    # vis_cam_positions.append(gymapi.Vec3(-0.15, -0.5, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    
    # #Median
    # vis_cam_positions.append(gymapi.Vec3(0.15, -0.4, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    # vis_cam_positions.append(gymapi.Vec3(-0.25, -0.5, 0.2))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))

    # #25th
    # vis_cam_positions.append(gymapi.Vec3(0.15, -0.4, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    # vis_cam_positions.append(gymapi.Vec3(0.25, -0.5, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    
    # #Min
    # vis_cam_positions.append(gymapi.Vec3(0.1, -0.5, 0.1))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    # vis_cam_positions.append(gymapi.Vec3(0.1, -0.5, 0.1))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))
    
    # ### Bimanual cylinder
    # vis_cam_positions.append(gymapi.Vec3(-0.15, -0.4, 0.1))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))    

    ### Bimanual hemis
    # vis_cam_positions.append(gymapi.Vec3(0.15, -0.55, 0.05))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01)) 


    ### Bimanual chicken breast Node
    # # Max
    # vis_cam_positions.append(gymapi.Vec3(0.05, -0.56, 0.1))
    # # # vis_cam_positions.append(gymapi.Vec3(0.05, -0.56, 0.08))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.52, 0.1))

    # # 75th
    # vis_cam_positions.append(gymapi.Vec3(-0.15, -0.58, 0.15))    # -0.60
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01)) 

    # # 50th
    # # # vis_cam_positions.append(gymapi.Vec3(0.1, -0.55, 0.1))
    # # # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))    
    # vis_cam_positions.append(gymapi.Vec3(0.2, -0.55, 0.1))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01)) 
    
    # # 25th     
    # vis_cam_positions.append(gymapi.Vec3(-0.05, -0.51, 0.12))
    # vis_cam_targets.append(gymapi.Vec3(0.05, -0.5, 0.05))  
    
    # # Min
    # vis_cam_positions.append(gymapi.Vec3(0.1, -0.43, 0.12))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))  
    # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.53, 0.12))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))      
    
    ### Bimanual chicken breast Chamfer
    # # Max
    # vis_cam_positions.append(gymapi.Vec3(0.02, -0.42, 0.1)) # -0.45
    # vis_cam_targets.append(gymapi.Vec3(-0.1, -0.5, 0.01))   
    
    # # 75th     
    # vis_cam_positions.append(gymapi.Vec3(0.05, -0.5, 0.08))
    # vis_cam_targets.append(gymapi.Vec3(-0.1, -0.5, 0.01))  
    
    # # 50th
    # vis_cam_positions.append(gymapi.Vec3(0.15, -0.5, 0.05))
    # vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01))  
    
    # # 25th  
    # vis_cam_positions.append(gymapi.Vec3(-0.05, -0.5, 0.18))
    # vis_cam_targets.append(gymapi.Vec3(0.05, -0.5, 0.05))     
    
    # # Min   
    # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.45, 0.15))
    # vis_cam_targets.append(gymapi.Vec3(0.05, -0.5, 0.05))     
    
    
    ### Chicken breast sample sequence
    vis_cam_positions.append(gymapi.Vec3(-0.15, -0.55, 0.12))   # 0.15
    vis_cam_targets.append(gymapi.Vec3(0.0, -0.5, 0.01)) 
    

    # Visualization stuff
    prepare_vis_cam = True
    start_vis_cam = True #False 
    vis_frame_count = 0
    num_image = 0
    image_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/unseen_objects/visualization/recordings"
    save_path = os.path.join(image_path, args.obj_name, "test_2")  
    os.makedirs(save_path, exist_ok=True)



    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
            gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])
                    

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
    rospy.logerr(f"Object name ... {args.obj_name}; Model type ... {model_category}") 

 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True    


    # Set up DNN:
    device = torch.device("cuda")

    
    # Model trained on all objects
    if model_category == "combined":
        deformernet_epoch_num = 160 #200#200  
        mp_dense_epoch_num = 160  
    
    ## Box
    elif model_category == "box_1k":
        deformernet_epoch_num = 160 #200#200  
        mp_dense_epoch_num = 160  
          
    elif model_category == "box_5k":
        deformernet_epoch_num = 160 #200  
        mp_dense_epoch_num = 160  
          
    elif model_category == "box_10k":
        deformernet_epoch_num = 160 #200  
        mp_dense_epoch_num = 160  
          
    
    ## Cylinder
    elif model_category == "cylinder_1k":
        deformernet_epoch_num = 160 #200  
        mp_dense_epoch_num = 160  
            
    elif model_category == "cylinder_5k":
        deformernet_epoch_num = 160 #200  
        mp_dense_epoch_num = 160  
          
    elif model_category == "cylinder_10k":
        deformernet_epoch_num = 160 #200  
        mp_dense_epoch_num = 160  
          
    
    ## Hemisphere
    elif model_category == "hemis_1k":
        deformernet_epoch_num = 160  
        mp_dense_epoch_num = 160  
           
    elif model_category == "hemis_5k":
        deformernet_epoch_num = 160  
        mp_dense_epoch_num = 160  
          
    elif model_category == "hemis_10k":
        deformernet_epoch_num = 160  
        mp_dense_epoch_num = 160  
          


    ### Set up DeformerNet
    deformernet_model_main_path = "/home/baothach/shape_servo_DNN"
    sys.path.append(f"{deformernet_model_main_path}/bimanual")

    from bimanual_architecture import DeformerNetBimanualRot
    model = DeformerNetBimanualRot().to(device)
        
  
    if model_category == "combined":
        weight_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/all_objects/weights/run1"
    else:
        weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{model_category}Pa/weights/run2/"
                  
    model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {deformernet_epoch_num}")))  
    
    # weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_box_1kPa/weights/run2/"     #_backup
    # model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {160}")))
    model.eval()

    """
    Notes: 
    -box1k model use for evaluation all box 1 5 and 10k
    -cylinder1k model for 1k, cylinder5k for both 5 and 10k
    -each evaluation hemis uses its own correspinding model. 
    """

    # mp_model_main_path = "/home/baothach/shape_servo_DNN/learn_mp"
    # sys.path.append(mp_model_main_path)



    # ### Set up manipulation point model
    # from test_pointconv import ManiPointSegment
    
    # mp_seg_model = ManiPointSegment(num_classes=2).to(device)
    # weight_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{model_category}Pa/weights/seg/run1"
    
    # mp_seg_model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {mp_dense_epoch_num}")))       
    # mp_seg_model.eval()




    goal_recording_path = os.path.join(main_path, "goal_data_sample_sequence", args.obj_name)
    # chamfer_recording_path = os.path.join(main_path, "chamfer_results_2", f"model_{model_category}", args.obj_name)
    
    # os.makedirs(chamfer_recording_path, exist_ok=True)


    goal_count = 0 #0
    frame_count = 0
    max_goal_count =  1#0  #10

    max_shapesrv_time = 2.0*60    # 2 mins
    min_chamfer_dist = 0.1 #0.2
    fail_mtp = False
    saved_nodes = []
    saved_chamfers = []
    final_node_distances = []  
    final_chamfer_distances = []    
    random_bool = True

    plan_count = 0

    dc_client = GraspDataCollectionClient()   

   
    # Get 10 goal pc data for 1 object:
    with open(os.path.join(goal_recording_path, f"{args.obj_name}_{args.obj_idx}" + ".pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = down_sampling(goal_datas["partial pcs"][1])   # first goal pc
    goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)  
    pcd_goal.paint_uniform_color([1,0,0]) 

    pcd_goal_full = open3d.geometry.PointCloud()
    pcd_goal_full.points = open3d.utility.Vector3dVector(goal_datas["full pcs"][1])  
    pcd_goal_full.paint_uniform_color([1,0,0]) 

    full_pc_goal = goal_datas["full pcs"][1]
    rospy.logwarn(f"number of nodes on mesh: {full_pc_goal.shape}")

    init_pc_numpy = down_sampling(goal_datas["partial pcs"][0])  # first goal pc
    init_pc_tensor = torch.from_numpy(init_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)  
    gt_mps = np.array(goal_datas["mani_point"])
    gt_mp_1 = [gt_mps[0][0,3], gt_mps[0][1,3]-two_robot_offset, gt_mps[0][2,3] + ROBOT_Z_OFFSET]
    gt_mp_2 = [-gt_mps[1][0,3], -gt_mps[1][1,3], gt_mps[1][2,3] + ROBOT_Z_OFFSET] 

    full_pc_numpy = goal_datas["full pcs"][0]
      
    goal_pos_1 = goal_datas["pos"][0]
    goal_rot_1 = goal_datas["rot"][0]         
    goal_pos_2 = goal_datas["pos"][1]
    goal_rot_2 = goal_datas["rot"][1]    

    
    start_time = timeit.default_timer()    
    close_viewer = False
    robot_2 = Robot(gym, sim, envs[0], kuka_handles_2[0])
    robot_1 = Robot(gym, sim, envs[0], kuka_handles[0])


    while (not close_viewer) and (not all_done): 



        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if prepare_vis_cam:
            radius = 4  #5  #2 #1        
            # Red color in BGR
            color = (0, 0, 255)
            thickness = -1  #-1  #2 
            goal_xs, goal_ys = get_goal_projected_on_image(down_sampling(goal_datas["full pcs"][1], num_pts=512), i, thickness = 0)
            points = np.column_stack((np.array(goal_ys), np.array(goal_xs)))
            prepare_vis_cam = False

 
        if start_vis_cam:   
            if vis_frame_count % 1 == 0:
                gym.render_all_camera_sensors(sim)
                im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))[:,:,:3]
                # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)
                im[goal_xs, goal_ys, :] = [255,0,0]
                image = im.astype(np.uint8)


                im = Image.fromarray(im)
                
                for point in points:
                    image = cv2.circle(image, tuple(point), radius, color, thickness)        

                path =  os.path.join(save_path, f'img{num_image:03}.png')                  
                cv2.imwrite(path, image)

                num_image += 1        

            vis_frame_count += 1 
            

        if state == "home" :   
            frame_count += 1
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
            
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.24)    
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)        
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state)
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    init_robot_state_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_ALL))
                    
                    first_time = False
                


                frame_count = 0
                state = "generate preshape"


        if state == "generate preshape":                   

            rospy.loginfo("**Current state: " + state)
            # preshape_response = boxpcopy(preshape_response.palm_goal_pose_world[0].pose)        
            with torch.no_grad():
                
                ### Ground truth:
                # if args.mp_method == "ground_truth":
                best_mp_1 = gt_mp_1
                best_mp_2 = gt_mp_2

                # ### Seg (Dense predictor):                
                # if args.mp_method == "dense_predictor":
                #     best_mp = get_mp_seg(seg_model=mp_seg_model, pc_init_numpy=init_pc_numpy, pc_initial_tensor=init_pc_tensor, pc_goal_tensor=goal_pc_tensor)


            # target_pose = [-best_mp[0], -best_mp[1], best_mp[2] - ROBOT_Z_OFFSET, 0, 0.707107, 0.707107, 0]
            target_pose_1 = [best_mp_1[0], best_mp_1[1] + two_robot_offset, best_mp_1[2] - ROBOT_Z_OFFSET-0.01, 0, 0.707107, 0.707107, 0]  
            target_pose_2 = [-best_mp_2[0], -best_mp_2[1], best_mp_2[2] - ROBOT_Z_OFFSET-0.01, 0, 0.707107, 0.707107, 0]  

            mtp_behavior_1 = MoveToPose(target_pose_1, robot_1, sim_params.dt, 1) 
            mtp_behavior_2 = MoveToPose(target_pose_2, robot_2, sim_params.dt, 1) 
            
            if mtp_behavior_1.is_complete_failure() or mtp_behavior_2.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset" 
                fail_mtp = True               
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))



        if state == "move to preshape":         
            action_1 = mtp_behavior_1.get_action()
            action_2 = mtp_behavior_2.get_action()

            if action_1 is not None:
                gym.set_actor_dof_position_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())      
                prev_action_1 = action_1
            else:
                gym.set_actor_dof_position_targets(robot_1.env_handle, robot_1.robot_handle, prev_action_1.get_joint_position())

            if action_2 is not None:
                gym.set_actor_dof_position_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())      
                prev_action_2 = action_2
            else:
                gym.set_actor_dof_position_targets(robot_2.env_handle, robot_2.robot_handle, prev_action_2.get_joint_position())


            if mtp_behavior_1.is_complete() and mtp_behavior_2.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!")

        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -3.0)         

            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper2_joint"), -3.0)  

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states_1 = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
            dof_states_2 = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)
            if dof_states_1['pos'][8] < 0.35 and dof_states_2['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"

                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper2_joint"), g_2_pos)                     
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), g_2_pos)         
        
                _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                init_eulers_1 = transformations.euler_from_matrix(init_pose_1)

                _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                init_eulers_2 = transformations.euler_from_matrix(init_pose_2)               
                
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(robot_1.env_handle, robot_1.robot_handle, dof_props_2)
                gym.set_actor_dof_properties(robot_2.env_handle, robot_2.robot_handle, dof_props_2)
                
                shapesrv_start_time = timeit.default_timer()

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)
    

            # print("xxxxxxxxxxxxxxxxxxxxx:", goal_pos_1.squeeze().shape, init_pose_1[:3,3].shape)
            desired_pos_1 = (goal_pos_1.squeeze() + init_pose_1[:3,3]).flatten()
            desired_rot_1 = goal_rot_1 @ init_pose_1[:3,:3]
            desired_pos_2 = (goal_pos_2.squeeze() + init_pose_2[:3,3]).flatten()
            desired_rot_2 = goal_rot_2 @ init_pose_2[:3,:3]

            tvc_behavior_1 = TaskVelocityControl2([*desired_pos_1, desired_rot_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                pos_threshold = 2e-3, ori_threshold=5e-2)
            tvc_behavior_2 = TaskVelocityControl2([*desired_pos_2, desired_rot_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                pos_threshold = 2e-3, ori_threshold=5e-2)
            
            # temp1 = np.eye(4)
            # temp1[:3,:3] = rot_mat_1
            # temp2 = np.eye(4)
            # temp2[:3,:3] = goal_rot_1    
            # print("========ROBOT 1=========")        
            # print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
            # print("goal_pos, goal_rot:", goal_pos_1, transformations.euler_from_matrix(temp2)) 
            # print("\n")

            # # temp1 = np.eye(4)
            # temp1[:3,:3] = rot_mat_2
            # # temp2 = np.eye(4)
            # temp2[:3,:3] = goal_rot_2    
            # print("========ROBOT 2=========")        
            # print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
            # print("goal_pos, goal_rot:", goal_pos_2, transformations.euler_from_matrix(temp2)) 
            # print("\n")

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))

            state = "move to goal"


        if state == "move to goal":           

            main_ins_pos_1 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"))
            main_ins_pos_2 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))

            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if main_ins_pos_1 <= 0.042 or main_ins_pos_2 <= 0.042 or (not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts)):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                # final_node_distances.append(999) 
                
                chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                
                saved_nodes.append(chamfer_dist)
                final_node_distances.append(999+min(saved_nodes)) 

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_sampling(get_partial_point_cloud(i)))   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(999+min(saved_chamfers))         

                print("***final node distance: ", min(saved_nodes)/full_pc_goal.shape[0]*1000)
                print("***final chamfer distance: ", min(saved_chamfers))

                goal_count += 1
            
            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time \
                    or plan_count > 6: # get new plan k times
                    
                    rospy.logerr("Timeout")
                    state = "reset" 
                    
                    chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                    
                    saved_nodes.append(chamfer_dist)
                    final_node_distances.append(1999+min(saved_nodes)) 

                    current_pc = down_sampling(get_partial_point_cloud(i))
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)   
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                    
                    saved_chamfers.append(chamfer_dist)
                    final_chamfer_distances.append(1999+min(saved_chamfers)) 

                    print("***final node distance: ", min(saved_nodes)/full_pc_goal.shape[0]*1000)
                    print("***final chamfer distance: ", min(saved_chamfers))
                    
                    goal_count += 1

                else:
                    action_1 = tvc_behavior_1.get_action()  
                    action_2 = tvc_behavior_2.get_action()  
                    if action_1 is None or action_2 is None or gym.get_sim_time(sim) - closed_loop_start_time >= 3:   
                        node_dist = np.linalg.norm(full_pc_goal - get_point_cloud())             
                        print("***final node distance: ", node_dist/full_pc_goal.shape[0]*1000)
                        final_node_distances.append(node_dist) 


                        current_pc = down_sampling(get_partial_point_cloud(i))
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(current_pc)  
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        final_chamfer_distances.append(chamfer_dist) 
                        print("***final chamfer distance: ", chamfer_dist)

                        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        pcd.paint_uniform_color([0,0,0])
                        open3d.visualization.draw_geometries([pcd, pcd_goal, coor.translate((0,-0.5,0))])

                        all_done = True

                    else:
                        gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position()/2)
                        gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position()/2)




        if state == "reset":   

            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pcd.paint_uniform_color([0,0,0])
            open3d.visualization.draw_geometries([pcd, pcd_goal, coor.translate((0,-0.5,0))])

            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            saved_chamfers = []
            saved_nodes = []
            
            rospy.logwarn(("=== JUST ENDED goal_count " + str(goal_count)))




            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            

            gym.set_actor_rigid_body_states(envs[i], kuka_handles[i], init_robot_state_1, gymapi.STATE_ALL) 
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state_2, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            
            gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, [0]*8)
            gym.set_actor_dof_position_targets(robot_1.env_handle, robot_1.robot_handle, [0,0,0,0,0.22,0,0,0,1.5,0.8])
             
            gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, [0]*8)
            gym.set_actor_dof_position_targets(robot_2.env_handle, robot_2.robot_handle, [0,0,0,0,0.22,0,0,0,1.5,0.8]) 
            
            print("Sucessfully reset robot and object")
            pc_on_trajectory = []
            full_pc_on_trajectory = []
            curr_trans_on_trajectory = []
                

            gym.set_actor_dof_properties(robot_1.env_handle, robot_1.robot_handle, dof_props_2) 
            gym.set_actor_dof_properties(robot_2.env_handle, robot_2.robot_handle, dof_props_2)  



            shapesrv_start_time = timeit.default_timer()
            
            state = "home"
            first_time = True

            if fail_mtp:
                state = "home"  
                fail_mtp = False


        if  goal_count >= max_goal_count:                    
            all_done = True 
           



        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


  
   



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    # print("total data pt count: ", data_point_count)
