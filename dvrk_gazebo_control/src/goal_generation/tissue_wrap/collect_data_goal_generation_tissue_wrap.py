#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys

from numpy import random
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
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2


# sys.path.append('/home/baothach/shape_servo_DNN/bimanual')
# # from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
# from bimanual_architecture import DeformerNetBimanual
import torch
import trimesh
import transformations
# from wrap_utils import *

from utils.miscellaneous_utils import pcd_ize, read_pickle_data, write_pickle_data, print_color
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views
from tissue_wrap_utils.util_functions import compute_intersection_percent

ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 1.0#0.93#0.86

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

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


def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc


def get_tissue_partial_pc_multi_views(vis=False):
    """ 
    Compute tissue partial point cloud from more than 1 camera view. Currently, using 2 cameras.
    """
    goal_1 = get_partial_pointcloud_vectorized(gym, sim, envs[0], tissue_cam_handles[0], cam_props, 
                                            segmentationId_dict, object_name="deformable", color=None, min_z=0.01, 
                                            visualization=False, device="cpu")  
    goal_2 = get_partial_pointcloud_vectorized(gym, sim, envs[0], tissue_cam_handles[1], cam_props, 
                                            segmentationId_dict, object_name="deformable", color=None, min_z=0.01, 
                                            visualization=False, device="cpu")                              
    combined_goal_pc = np.concatenate((goal_1, goal_2), axis=0)
    
    if vis:
        pcd_ize(combined_goal_pc, color=[0,0,0], vis=True)
    
    return combined_goal_pc                        

if __name__ == "__main__":

    objects_path = "/home/baothach/shape_servo_data/evaluation"

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_name", "type": str, "default": 'box_0', "help": "select variations of a primitive shape"},
            {"name": "--obj_type", "type": str, "default": 'box_1k', "help": "box1k, box5k, etc."},
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"},
            {"name": "--object_idx", "type": int, "default": 10000, "help": "which index of cylinder and tissue mesh to use. From 0 to 199"}])

    num_envs = args.num_envs
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"



    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        # print("=================sim_params.dt:", sim_params.dt)
        sim_params.dt = 1./60.
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 10   #4
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
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "outside") 

    # with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
    #     mesh_data = pickle.load(handle)    
    # h = mesh_data["height"]
    # w = mesh_data["width"]
    # thickness = mesh_data["thickness"]

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
    asset_options.thickness = 0.0005
    # asset_options.thickness = 0.002


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


    main_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/objects_dataset"    # FIX
    cylinder_path = os.path.join(main_path, "cylinder/urdf")
    tissue_path = os.path.join(main_path, "tissue/urdf")
    cylinder_dict_path = os.path.join(main_path, "cylinder/specification")
    tissue_dict_path = os.path.join(main_path, "tissue/specification")
    
    cylinder_dict = read_pickle_data(os.path.join(cylinder_dict_path, f"cylinder_{args.object_idx}.pickle"))
    tissue_dict = read_pickle_data(os.path.join(tissue_dict_path, f"tissue_{args.object_idx}.pickle"))
    cylinder_radius, cylinder_length = cylinder_dict["radius"], cylinder_dict["length"]
    tissue_width, tissue_length, tissue_thickness = tissue_dict["width"], tissue_dict["length"], tissue_dict["thickness"]
    print_color(f"\n******** Cylinder length: {cylinder_length}\n")

    asset_root = tissue_path
    soft_asset_file = f"tissue_{args.object_idx}.urdf"



    soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
    soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, tissue_thickness)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.001#0.001   # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
    
    cylinder_asset_root = cylinder_path
    cylinder_asset_file = f"cylinder_{args.object_idx}.urdf"   #trimesh.creation.cylinder(radius=0.015, height=0.1)      
    cylinder_pose = gymapi.Transform()
    # cylinder_pose.p = gymapi.Vec3(0.01, -two_robot_offset/2+0.01, 0.04)
    # cylinder_pose.p = gymapi.Vec3(0.00, -two_robot_offset/2+0.00, 0.04)


    # cylinder_shift = np.array([-0.05,-0.05,0])  #np.array([0.06,-0.02,0.03])
    shift_x = np.random.uniform(low = -0.05, high = 0.05, size=1) 
    shift_y = np.random.uniform(low = -0.05, high = 0.05, size=1) 
    shift_z = np.random.uniform(low = 0, high = 0.05, size=1) 
    cylinder_shift = np.concatenate((shift_x, shift_y, shift_z))
    print("****cylinder_shift:", cylinder_shift)
    
    
    cylinder_pose.p = gymapi.Vec3(cylinder_shift[0], -two_robot_offset/2+cylinder_shift[1], cylinder_shift[2]+0.04) 
    # cylinder_pose.p = gymapi.Vec3(cylinder_shift[0], -two_robot_offset/2+cylinder_shift[1], cylinder_shift[2])
    cylinder_pose.r = gymapi.Quat(0.5,0.5,0.5,0.5)
    # euler = [np.pi/2, 0, 0]
    # cylinder_pose.r = gymapi.Quat(*euler)
    
    
    # asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = False
    # asset_options.thickness = 0.01
    # asset_options.disable_gravity = False
    cylinder_asset = gym.load_asset(sim, cylinder_asset_root, cylinder_asset_file, asset_options)
    
 
    
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
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=10)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)        
        

        # add soft obj    
        
              
        # env_obj = env
        
        
        # env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        # envs_obj.append(env_obj)    
        # soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)         
        
        # object_handles.append(soft_actor)

        # temp_env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        # cylinder_actor = gym.create_actor(temp_env, cylinder_asset, cylinder_pose, "cylinder", i+2, 1, segmentationId=11)
        # color = gymapi.Vec3(1,0,0)
        # gym.set_rigid_body_color(temp_env, cylinder_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        env_obj = env
        envs_obj.append(env_obj)    
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)         
        
        object_handles.append(soft_actor)

        temp_env = env
        cylinder_actor = gym.create_actor(temp_env, cylinder_asset, cylinder_pose, "cylinder", i, 1, segmentationId=12)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(temp_env, cylinder_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


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
        # cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # cam_pos = gymapi.Vec3(0.3, -0.1, 0.5)  # final setup for thin layer tissue
        # cam_pos = gymapi.Vec3(0.5, -0.36, 0.3)
        # cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
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


    tissue_cam_positions = []
    tissue_cam_targets = []
    tissue_cam_handles = []
    tissue_cam_width = 256
    tissue_cam_height = 256
    tissue_cam_props = gymapi.CameraProperties()
    tissue_cam_props.width = tissue_cam_width
    tissue_cam_props.height = tissue_cam_height
   
    tissue_cam_positions.append(gymapi.Vec3(0.3, -two_robot_offset/2 - 0.3, 0.1))
    tissue_cam_targets.append(gymapi.Vec3(0, -two_robot_offset/2, 0.01))    

    tissue_cam_positions.append(gymapi.Vec3(-0.3, -two_robot_offset/2 + 0.3, 0.1))
    tissue_cam_targets.append(gymapi.Vec3(0, -two_robot_offset/2, 0.01))    

   
    for j in range(len(tissue_cam_positions)):
            tissue_cam_handles.append(gym.create_camera_sensor(env_obj, tissue_cam_props))
            gym.set_camera_location(tissue_cam_handles[j], env_obj, tissue_cam_positions[j], tissue_cam_targets[j])


    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('wrap_tissue')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    

    # goal_recording_path = "/home/baothach/shape_servo_data/comparison/RRT/goal_data"
    # goal_point_count = 0

    data_recording_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/data"    # FIX
    os.makedirs(data_recording_path, exist_ok=True)
    data_point_count = len(os.listdir(data_recording_path))
    max_data_point_count = 10000    # 10000    # FIX
    sample_count = 0
    frame_count = 0
    group_count = 0
    iter_count = 0
    percent = 0.0




    pc_on_trajectory = []
    full_pc_on_trajectory = []
    poses_on_trajectory_1 = []  
    poses_on_trajectory_2 = [] 
    first_time = True
    save_intial_pc = True
    switch = True
    total_computation_time = 0
    move_to_centroid = True
    visualization = False
    segmentationId_dict = {"robot_1": 10, "robot_2": 11, "cylinder": 12}

    max_shapesrv_time = 2*60    # 2 mins
    min_chamfer_dist = 0.2
    fail_mtp = False
    saved_chamfers = []

    dc_client = GraspDataCollectionClient()
    
    print_color(f"\n******** object_idx: {args.object_idx}\n")

    
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
 

        if state == "home" :   

            if visualization:
                output_file = "/home/baothach/Downloads/test_cam_views.png"     # cam_handles         
                visualize_camera_views(gym, sim, envs[0], tissue_cam_handles, \
                                    resolution=[cam_props.height, cam_props.width], output_file=output_file)
                # all_done = True
            

            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.24)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)            
            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    init_robot_state_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_ALL))
        

                    cylinder_pc = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_props,
                                segmentationId_dict, object_name="cylinder", color=None, min_z=0.01, 
                                visualization=False, device="cpu")  

                    first_time = False

                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg) 

                init_full_pc = deepcopy(current_pc)


                state = "generate preshape"                
                frame_count = 0              




        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
                
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]
            mtp_behavior_2 = MoveToPose(target_pose, robot_2, sim_params.dt, 1)
            
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg, non_random = True)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [cartesian_goal.position.x, cartesian_goal.position.y + two_robot_offset, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]
            mtp_behavior_1 = MoveToPose(target_pose, robot_1, sim_params.dt, 1)            
            
            if mtp_behavior_1.is_complete_failure() or mtp_behavior_2.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')   
                all_done = True              
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
                _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                mp_pose_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)[-3])
                mp_pose_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])  

                # switch to velocity mode
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                gym.set_actor_dof_properties(env, kuka_handles[i], dof_props_2)
                

                shapesrv_start_time = timeit.default_timer()
                # open3d.visualization.draw_geometries([pcd_goal])

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 


            if move_to_centroid:     
                robot_1_desired_pos = np.array([cylinder_shift[0], cylinder_shift[1], cylinder_shift[2]+0.04]) 
                robot_2_desired_pos = np.array([-cylinder_shift[0], -cylinder_shift[1], cylinder_shift[2]+0.04])   
                print("======  move to centroid")
            else:

                
                rospy.logerr("Use handcoded actions")
                if iter_count == 0:
                    # desired_position = np.array([0,0.06,0.03,0,0.06,0.03])
                    desired_position = np.array([0,tissue_length*0.4,0.05,0,tissue_length*0.4,0.05])  # 0.06
                    iter_count += 1
                else:
                    desired_position = np.array([0,0.00,0.00,0,0.00,0.00])   
                    
                print(f"====== desired_position: {list(desired_position)}")
                   
                robot_1_desired_pos = desired_position[:3]
                robot_2_desired_pos = desired_position[3:]
             

                # print("from model:", robot_1_desired_pos, robot_2_desired_pos)    


            tvc_behavior_1 = TaskVelocityControl(robot_1_desired_pos, robot_1, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 5e-3, second_robot=False)
            tvc_behavior_2 = TaskVelocityControl(robot_2_desired_pos, robot_2, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 5e-3)


            # closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            
            state = "move to goal"

        if state == "move to goal":
            
            # print("Moving ...")

            main_ins_pos_1 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"))
            main_ins_pos_2 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))

            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if main_ins_pos_1 <= 0.042 or main_ins_pos_2 <= 0.042 or \
                (not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts)):  # lose contact or exeed joint limit. 
                
                rospy.logerr("Lost contact with robot")
                all_done = True


            if not move_to_centroid:

                if frame_count == 0: 
                    initial_full_pc = get_point_cloud()
                    initial_partial_pc = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_props, 
                                        segmentationId_dict, object_name="deformable", color=None, min_z=0.01, 
                                        visualization=False, device="cpu")  
                    
                frame_count += 1  

            if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                rospy.logerr("Timeout")
                
                all_done = True
            
            else:
                action_1 = tvc_behavior_1.get_action()  
                action_2 = tvc_behavior_2.get_action() 
                
                if action_1 is None or action_2 is None:   
                    state = "get shape servo plan"    
                    _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                    _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                    mp_pose_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)[-3])
                    mp_pose_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])  

                    
                    final_full_pc = get_point_cloud()
                    (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)
                    percent = compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, 
                                                           cylinder_radius, cylinder_length, vis = False)
                    print_color(f"*** Final percent intersect: {percent*100} %")


                    if not move_to_centroid:
                    # Success conditions               
                        if percent >= 0.96:           
                            
                            print_color(f"Valid demonstration!")              

                            if visualization:
                                output_file = "/home/baothach/Downloads/test_cam_views_2.png"     # cam_handles         
                                visualize_camera_views(gym, sim, envs[0], tissue_cam_handles, \
                                                    resolution=[cam_props.height, cam_props.width], output_file=output_file)

                                    
                            goal_full_pc = get_point_cloud()
                            goal_partial_pc = get_tissue_partial_pc_multi_views(vis=False)
                            # print(f"*** Final goal pc shape: {goal_partial_pc.shape}")
                                              
                            partial_pcs = (initial_partial_pc, goal_partial_pc)
                            full_pcs = (initial_full_pc, goal_full_pc)

                            data = {"full pcs": full_pcs, "partial pcs": partial_pcs,\
                                    "cylinder_pc": cylinder_pc, "cylinder pose": cylinder_shift,
                                    "object_idx": args.object_idx}
                            
                            with open(os.path.join(data_recording_path, f"sample {data_point_count}.pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
                                    
                                                            
                            data_point_count += 1 
                            print_color("============= Current data_point_count:", data_point_count)
                                

                        state = "reset"
                        reset_frame_count = 0

                    else:
                        move_to_centroid = False

                else:
                    gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                    gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())



        if state == "reset": 
            all_done = True  
            
            # reset_frame_count += 1
            # rospy.logerr(f"reset_frame_count: {reset_frame_count}")
            # gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, np.zeros(10).astype('float32'))
            # gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, np.zeros(10).astype('float32'))
            # if reset_frame_count >= 1:
            #     all_done = True
           
            #     # percent = compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, 
            #     #                                     cylinder_radius, cylinder_length, vis = True)
            #     # print(f"Final percent intersect: {percent*100} %") 
        
        
        if data_point_count >= max_data_point_count:           
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