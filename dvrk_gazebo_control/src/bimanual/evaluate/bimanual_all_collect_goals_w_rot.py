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
# from geometry_msgs.msg import PoseStamped, Pose
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
# from PIL import Image

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2
import transformations
from scipy import interpolate

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

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

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

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
    # depth_buffer[seg_buffer == 11] = -10001

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




if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--obj_type", "type": str, "default": None, "help": "hemis1k, hemis5k, etc."},
            {"name": "--prim_name", "type": str, "default": "box", "help": "Select primitive shape. Options: box, cylinder, hemis"},
            {"name": "--stiffness", "type": str, "default": "1k", "help": "Select object stiffness. Options: 1k, 5k, 10k"},
            {"name": "--obj_name", "type": int, "default": 0, "help": "select variations of a primitive shape"},            
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    mesh_name = f"{args.prim_name}_{args.obj_name%10}"
    
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"
    args.obj_type = f"{args.prim_name}_{args.stiffness}"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"

    main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_{args.obj_type}Pa/evaluate"
    objects_path = "/home/baothach/shape_servo_data/evaluation"
    


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
        sim_params.flex.num_outer_iterations = 10
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
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(objects_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, mesh_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    if args.prim_name == "box":
        h = data["height"]
        w = data["width"]
        thickness = data["thickness"]
    elif args.prim_name == "cylinder":
        r = data["radius"]
        h = data["height"]
    elif args.prim_name == "hemis":
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
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, -two_robot_offset, ROBOT_Z_OFFSET) 
 
    pose_2 = gymapi.Transform()
    pose_2.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    pose_2.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0001


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
        asset_root = os.path.join(objects_path, "bimanual_urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(objects_path, "bimanual_urdf", args.obj_type, "outside") 


    soft_asset_file = args.obj_name + ".urdf" 

    soft_pose = gymapi.Transform()

    if args.prim_name == "box": 
        soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, thickness/2*0.5)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    elif args.prim_name == "cylinder": 
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, r/2.0)
        soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    elif args.prim_name == "hemis":
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, -o/2.)

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
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
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



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    
    
    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 

    os.makedirs(goal_recording_path, exist_ok=True)


    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    data_point_count = 0

    # max_group_count = 150000
    max_sample_count = 1
    max_data_point_count = 1#10
    # if args.obj_name == 'box_64':
    #     max_data_point_per_variation = 9600
    # else:

    # max_data_point_per_variation = data_point_count + 150


    pc_on_trajectory = []
    full_pc_on_trajectory = []
    curr_trans_on_trajectory_1 = []
    curr_trans_on_trajectory_2 = []
    first_time = True
    save_intial_pc = True
    switch = True
    total_computation_time = 0
    data = []

    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 

    os.makedirs(goal_recording_path, exist_ok=True)

    dc_client = GraspDataCollectionClient()
    


    
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
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.24)    
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)             
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    init_robot_state_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_ALL))
                    first_time = False

                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)
                
                state = "generate preshape"                
                frame_count = 0              


        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)

            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal_2 = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            # target_pose = [-cartesian_goal_2.position.x, -cartesian_goal_2.position.y, cartesian_goal_2.position.z-ROBOT_Z_OFFSET,
            #                 0, 0.707107, 0.707107, 0]               
                      
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg, non_random = True)               
            cartesian_goal_1 = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        

            
            if args.prim_name in ["box", "cylinder"]:
                target_pose_1 = [cartesian_goal_1.position.x, cartesian_goal_1.position.y + two_robot_offset, cartesian_goal_1.position.z-ROBOT_Z_OFFSET,
                                0, 0.707107, 0.707107, 0]
                target_pose_2 = [-cartesian_goal_2.position.x, -cartesian_goal_2.position.y, cartesian_goal_2.position.z-ROBOT_Z_OFFSET,
                                0, 0.707107, 0.707107, 0]
            elif args.prim_name == "hemis":  
                target_pose_1 = [cartesian_goal_1.position.x, cartesian_goal_1.position.y + two_robot_offset, cartesian_goal_1.position.z-ROBOT_Z_OFFSET-0.01,
                                0, 0.707107, 0.707107, 0]
                target_pose_2 = [-cartesian_goal_2.position.x, -cartesian_goal_2.position.y, cartesian_goal_2.position.z-ROBOT_Z_OFFSET-0.01,
                                0, 0.707107, 0.707107, 0]      

            mtp_behavior_1 = MoveToPose(target_pose_1, robot_1, sim_params.dt, 1)   
            mtp_behavior_2 = MoveToPose(target_pose_2, robot_2, sim_params.dt, 1)         
            
            if mtp_behavior_1.is_complete_failure() or mtp_behavior_2.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"                
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"


            pc_init = get_partial_point_cloud(i)
            full_pc_init = get_point_cloud()

            if args.prim_name == "hemis":
                ys = get_point_cloud()[:,1]
                object_length = abs(max(ys)-min(ys))
                # print("object_length:", object_length)
                ee_loc_on_obj_1 = abs(cartesian_goal_1.position.y-soft_pose.p.y) + object_length/2 
                ee_loc_on_obj_2 = abs(cartesian_goal_2.position.y-soft_pose.p.y) + object_length/2
                # print("where ee:", ee_loc_on_obj_1, ee_loc_on_obj_2)  # 0.85 and 0.5
                ee_ratio_1 = ee_loc_on_obj_1/object_length
                ee_ratio_2 = ee_loc_on_obj_2/object_length
                # print("ee_ratio:", ee_ratio_1, ee_ratio_2)



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

                gym.refresh_particle_state_tensor(sim)
                saved_object_contact_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                saved_robot_state_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                saved_robot_state_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_ALL))
        
                _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                init_eulers_1 = transformations.euler_from_matrix(init_pose_1)

                _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                init_eulers_2 = transformations.euler_from_matrix(init_pose_2)               
                
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(robot_1.env_handle, robot_1.robot_handle, dof_props_2)
                gym.set_actor_dof_properties(robot_2.env_handle, robot_2.robot_handle, dof_props_2)

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 
            
            if args.prim_name in ["box", "cylinder"]:
                max_x = max_y = max_z = h * 0.5 * 0.8
                max_x *= 3/4 
                max_y *= 3/4 * 1/2 * 2/3
                max_z *= 3/4

            elif args.prim_name == "hemis":    
                ee_ratio = (ee_ratio_1 + ee_ratio_2) / 2 
                # print("ee_ratio:", ee_ratio)
                ee_ratio_val = [0.5, 0.85]
                max_val = np.array([object_length/2 * 3/4, r/0.2 * 0.08])
                f = interpolate.interp1d(ee_ratio_val, max_val, fill_value='extrapolate') 
                max_x = f(ee_ratio)
                max_y, max_z = deepcopy(max_x), deepcopy(max_x)
                print("computed max:", max_x, max_y, max_z)
                max_z *= 3/4 
                max_y *= 1/2 * 2/3            

            rospy.logerr(f"max xyx: {max_x}, {max_y}, {max_z}")



            delta_x_1 = np.random.uniform(low = -max_x, high = max_x)
            delta_y_1 = np.random.uniform(low = 0.0, high = max_y)
            delta_z_1 = np.random.uniform(low = max_z/2, high = max_z)
            # delta_z_1 = np.random.uniform(low = 0.00, high = max_z)     
            # delta_alpha_1 = np.random.uniform(low = -np.pi/3, high = np.pi/3)
            # delta_beta_1 = np.random.uniform(low = -np.pi/3, high = np.pi/3) 
            # delta_gamma_1 = np.random.uniform(low = -np.pi/2, high = np.pi/2)
            delta_alpha_1 = np.random.uniform(low = -np.pi/6, high = np.pi/6)
            delta_beta_1 = np.random.uniform(low = -np.pi/6, high = np.pi/6) 
            delta_gamma_1 = np.random.uniform(low = -np.pi/6, high = np.pi/6)

            if delta_x_1 >= 0:
                delta_x_2 = np.random.uniform(low = -max_x, high = 0)
            else:
                delta_x_2 = np.random.uniform(low = 0, high = max_x)
                
            delta_y_2 = np.random.uniform(low = 0.0, high = max_y)
            delta_z_2 = np.random.uniform(low = max_z/2, high = max_z)
            # delta_z_2 = np.random.uniform(low = 0.00, high = max_z)  
            # delta_alpha_2 = np.random.uniform(low = -np.pi/3, high = np.pi/3)
            # delta_beta_2 = np.random.uniform(low = -np.pi/3, high = np.pi/3) 
            # delta_gamma_2 = np.random.uniform(low = -np.pi/2, high = np.pi/2)
            delta_alpha_2 = np.random.uniform(low = -np.pi/6, high = np.pi/6)
            delta_beta_2 = np.random.uniform(low = -np.pi/6, high = np.pi/6) 
            delta_gamma_2 = np.random.uniform(low = -np.pi/6, high = np.pi/6)
            
            # delta_alpha_1, delta_beta_1, delta_gamma_1 = np.pi/300000, np.pi/300000, np.pi/300000
            # delta_alpha_2, delta_beta_2, delta_gamma_2 = np.pi/300000, np.pi/300000, np.pi/300000
            # delta_x_1, delta_y_1, delta_z_1 = max_x, max_y, max_z/2
            # delta_x_2, delta_y_2, delta_z_2 = -max_x, max_y, max_z/2

            print("Max xyz:", max_x, max_y, max_z)
            print("Robot 1 selects x, y, z, a, b, g:", delta_x_1, delta_y_1, delta_z_1, " | ", delta_alpha_1, delta_beta_1, delta_gamma_1) 
            print("Robot 2 selects x, y, z, a, b, g:", delta_x_2, delta_y_2, delta_z_2, " | ", delta_alpha_2, delta_beta_2, delta_gamma_2) 

            x_1 = delta_x_1 + init_pose_1[0,3]
            y_1 = delta_y_1 + init_pose_1[1,3]
            z_1 = delta_z_1 + init_pose_1[2,3]
            alpha_1 = delta_alpha_1 + init_eulers_1[0]
            beta_1 = delta_beta_1 + init_eulers_1[1]
            gamma_1 = delta_gamma_1 + init_eulers_1[2]

            x_2 = delta_x_2 + init_pose_2[0,3]
            y_2 = delta_y_2 + init_pose_2[1,3]
            z_2 = delta_z_2 + init_pose_2[2,3]
            alpha_2 = delta_alpha_2 + init_eulers_2[0]
            beta_2 = delta_beta_2 + init_eulers_2[1]
            gamma_2 = delta_gamma_2 + init_eulers_2[2]


            tvc_behavior_1 = TaskVelocityControl2([x_1,y_1,z_1,alpha_1,beta_1,gamma_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits)
            tvc_behavior_2 = TaskVelocityControl2([x_2,y_2,z_2,alpha_2,beta_2,gamma_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits)
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            
            state = "move to goal"


        if state == "move to goal":
            main_ins_pos_1 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"))
            main_ins_pos_2 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            if main_ins_pos_1 <= 0.042 or main_ins_pos_2 <= 0.042:
                rospy.logerr("Exceed joint constraint")
                group_count += 1
                state = "reset"

            rigid_contacts = gym.get_env_rigid_contacts(envs[0])
            if len(list(rigid_contacts)) != 0:
                for k in range(len(list(rigid_contacts))):
                    if rigid_contacts[k]['body0'] > -1 and rigid_contacts[k]['body1'] > -1 : # ignore collision with the ground which has a value of -1
                        state = "reset"
                        rospy.logerr("Two robots collided !!")
                        break


            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if (not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts)):  # lose contact w either robot 2 or robot 1    
                rospy.logerr("Lost contact with robot")
                # all_done = True
                
                group_count += 1
                state = "reset"

            else:
                if frame_count == 0:
                    full_pc_on_trajectory.append(get_point_cloud())
                    pc_on_trajectory.append(get_partial_point_cloud(i))
                    curr_trans_1 = get_pykdl_client(robot_1.get_arm_joint_positions())[1]
                    curr_trans_2 = get_pykdl_client(robot_2.get_arm_joint_positions())[1]
                              
                frame_count += 1           
                
                action_1 = tvc_behavior_1.get_action()  
                action_2 = tvc_behavior_2.get_action() 
                # print("action_1, action_2:", action_1, action_2)
                if (action_1 is not None) and (action_2 is not None) and gym.get_sim_time(sim) - closed_loop_start_time <= 8: 
                    gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                    gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())

                else:   
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  
                    
                    # gt_mp_1 = [curr_trans_1[0,3], curr_trans_1[1,3]-two_robot_offset, curr_trans_1[2,3] + ROBOT_Z_OFFSET]
                    # gt_mp_2 = [curr_trans_2[0,3], -curr_trans_2[1,3], curr_trans_2[2,3] + ROBOT_Z_OFFSET]
                    # # current_pc_numpy = down_sampling(get_partial_point_cloud(i))                  
                            
                    # pcd = open3d.geometry.PointCloud()
                    # pcd.points = open3d.utility.Vector3dVector(full_pc_init)  
                    # pcd.paint_uniform_color([0,0,0])
                    # mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    # mani_point_1_sphere.paint_uniform_color([0,0,1])
                    # # mani_point_1_sphere.translate(tuple(mani_point_1))
                    # mani_point_1_sphere.translate(tuple(gt_mp_1))
                    # mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    # mani_point_2_sphere.paint_uniform_color([1,0,0])
                    # # mani_point_1_sphere.translate(tuple(mani_point_2))
                    # mani_point_2_sphere.translate(tuple(gt_mp_2))
                    # open3d.visualization.draw_geometries([pcd, mani_point_1_sphere, mani_point_2_sphere])             

                    _, final_trans_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                    _, final_trans_2 = get_pykdl_client(robot_2.get_arm_joint_positions())

                                      
                    p_1, R_1, twist_1 = tvc_behavior_1.get_transform(curr_trans_1, final_trans_1, get_twist=True)
                    mani_point_1 = curr_trans_1

                    p_2, R_2, twist_2 = tvc_behavior_2.get_transform(curr_trans_2, final_trans_2, get_twist=True)
                    mani_point_2 = curr_trans_2

                    partial_goal_pcs = (pc_init, get_partial_point_cloud(i))
                    full_goal_pcs = (full_pc_init, get_point_cloud())     

                    data = {"full pcs": full_goal_pcs, "partial pcs": partial_goal_pcs, "pos": (p_1, p_2), "rot": (R_1, R_2), "twist": (twist_1, twist_2), \
                            "mani_point": (mani_point_1, mani_point_2), \
                            "obj_name": args.obj_name}

                    with open(os.path.join(goal_recording_path, f"{args.obj_name}.pickle"), 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                               
                    
                    print("data_point_count:", data_point_count)
                    data_point_count += 1       


    
                    frame_count = 0
                    terminate_count = 0
                    sample_count += 1
                    print("group ", group_count, ", sample ", sample_count)
                    pc_on_trajectory = []
                    full_pc_on_trajectory = []
                    curr_trans_on_trajectory_1 = []  
                    curr_trans_on_trajectory_2 = []
                    state = "get shape servo plan"
             
        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            sample_count = 0
            terminate_count = 0

            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)            
            
            gym.set_actor_rigid_body_states(envs[i], kuka_handles[i], init_robot_state_1, gymapi.STATE_ALL) 
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state_2, gymapi.STATE_ALL) 

            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, [0]*8)
            gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, [0]*8)

            
            print("Sucessfully reset robot and object")
            pc_on_trajectory = []
            full_pc_on_trajectory = []
            curr_trans_on_trajectory_1 = []
            curr_trans_on_trajectory_2 = []
                
            gym.set_actor_dof_properties(robot_1.env_handle, robot_1.robot_handle, dof_props_2) 
            gym.set_actor_dof_properties(robot_2.env_handle, robot_2.robot_handle, dof_props_2)  
            gym.set_actor_dof_position_targets(robot_1.env_handle, robot_1.robot_handle, [0,0,0,0,0.24,0,0,0,1.5,0.8]) 
            gym.set_actor_dof_position_targets(robot_2.env_handle, robot_2.robot_handle, [0,0,0,0,0.24,0,0,0,1.5,0.8]) 

            state = "home"
 
 
        if  data_point_count >= max_data_point_count:                    
            all_done = True 

        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)


  
   



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("total data pt count: ", data_point_count)
