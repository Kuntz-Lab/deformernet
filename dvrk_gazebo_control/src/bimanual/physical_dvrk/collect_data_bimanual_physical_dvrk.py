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
from copy import deepcopy
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
from utils.miscellaneous_utils import get_object_particle_state, write_pickle_data, print_lists_with_formatting, print_color, read_pickle_data
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views
import pickle
import timeit
from copy import deepcopy
from scipy import interpolate

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2
import transformations


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


if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--prim_name", "type": str, "default": "box", "help": "Select primitive shape. Options: box, cylinder, hemis"},
            {"name": "--stiffness", "type": str, "default": "1k", "help": "Select object stiffness. Options: 1k, 5k, 10k"},
            {"name": "--obj_name", "type": int, "default": 0, "help": "select variations of a primitive shape"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"
    object_category = f"{args.prim_name}_{args.stiffness}"

    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0)
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
    object_meshes_path = f"/home/baothach/sim_data/Custom/Custom_mesh/physical_dvrk/multi_{object_category}Pa"    
    with open(os.path.join(object_meshes_path, f"primitive_dict_{args.prim_name}.pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    if args.prim_name == "box":
        h = data[args.obj_name]["height"]
        w = data[args.obj_name]["width"]
        thickness = data[args.obj_name]["thickness"]
    elif args.prim_name == "cylinder":
        r = data[args.obj_name]["radius"]
        h = data[args.obj_name]["height"]
    elif args.prim_name == "hemis":
        r = data[args.obj_name]["radius"]
        o = data[args.obj_name]["origin"]


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

    asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/physical_dvrk/bimanual/multi_{object_category}Pa"
    soft_asset_file = args.obj_name + ".urdf"    


    soft_pose = gymapi.Transform()
    
    # soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, thickness/2*0.7)
    if args.prim_name == "box": 
        soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, thickness/2 + 0.001)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    elif args.prim_name == "cylinder": 
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, r)
        soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    elif args.prim_name == "hemis":
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, -o)


    soft_thickness = 0.001 #0.0005#0.0005    # important to add some thickness to the soft body to avoid interpenetrations






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
    cam_width = 400 #256
    cam_height = 400 #256
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height

    # cam_positions.append(gymapi.Vec3(0.17, -0.62-(two_robot_offset/2 - 0.42), 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, 0.40-0.86-(two_robot_offset/2 - 0.42), 0.01))  

    # cam_positions.append(gymapi.Vec3(-0.3, soft_pose.p.y, 0.25))   # put camera on the side of object
    # cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.01))

    cam_positions.append(gymapi.Vec3(-0.0, soft_pose.p.y + 0.001, 0.5))   # put camera on top of object
    cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.01))

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
    print_color(f"Object: {args.obj_name}, Stiffness: {args.stiffness}", "green") 
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    
    
    data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{object_category}Pa/data"

    mp_data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/multi_{object_category}Pa/mp_data"
 
    os.makedirs(data_recording_path, exist_ok=True)
    os.makedirs(mp_data_recording_path, exist_ok=True)

    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    data_point_count = len(os.listdir(data_recording_path))
    mp_data_point_count = len(os.listdir(mp_data_recording_path))
    max_group_count = 150000
    max_sample_count = 3    #2
    
    
    # max_data_point_count = 5000
    # max_data_point_per_variation = data_point_count + 50  

    max_mp_data_point_count = 2000
    max_mp_data_point_per_variation = mp_data_point_count + 20  

    pc_on_trajectory = []
    full_pc_on_trajectory = []
    curr_trans_on_trajectory_1 = []
    curr_trans_on_trajectory_2 = []
    first_time = True
    save_intial_pc = True
    switch = True
    total_computation_time = 0
    data = []
    

    dc_client = GraspDataCollectionClient()
    segmentationId_dict = {"robot_1": 10, "robot_2": 11, "cylinder": 12}
    camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                    segmentationId_dict, "deformable", None, 0.002, False, "cpu"]    
    visualization = False


    
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

            if visualization:
                if frame_count == 5:                
                    output_file = "/home/baothach/Downloads/test_cam_views_init.png"           
                    visualize_camera_views(gym, sim, envs_obj[0], cam_handles, \
                                        resolution=[cam_props.height, cam_props.width], output_file=output_file)

            if frame_count == 10:
                rospy.loginfo("**Current state: " + state + ", current sample count: " + str(sample_count))
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    init_robot_state_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_ALL))
                    first_time = False

                    current_pc = get_object_particle_state(gym, sim)
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)
                
                state = "generate preshape"                
                frame_count = 0    

                shift = np.array([0.0, -soft_pose.p.y, camera_args[-3]])    # shift object to centroid 


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


            pc_init = get_partial_pointcloud_vectorized(*camera_args) + shift
            full_pc_init = get_object_particle_state(gym, sim) + shift

            # ys = get_object_particle_state(gym, sim)[:,1]
            # object_length = abs(max(ys)-min(ys))
            # print("object_length:", object_length)
            # ee_loc_on_obj = abs(cartesian_goal_2.position.y-soft_pose.p.y) + object_length/2
            # print("where ee:", ee_loc_on_obj)  # 0.85 and 0.5
            # ee_ratio = ee_loc_on_obj/object_length
            # print("ee_ratio:", ee_ratio)


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

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 
            
            if args.prim_name in ["box", "cylinder"]:
                max_x, max_y, max_z = 0.2, 0.15, 0.2

                # print("computed max:", max_x, max_y, max_z)
            # elif args.prim_name == "hemis":    
            #     # print("ee_ratio:", ee_ratio)
            #     ee_ratio_val = [0.5, 0.85]
            #     max_val = np.array([object_length/2 * 3/4, r/0.2 * 0.08])
            #     f = interpolate.interp1d(ee_ratio_val, max_val, fill_value='extrapolate') 
            #     max_x = f(ee_ratio)
            #     max_y, max_z = deepcopy(max_x), deepcopy(max_x)
            #     # print("computed max:", max_x, max_y, max_z)
            #     max_z *= 3/4                
            # else:
            #     raise Exception("Wrong object category")

            max_delta_x1_x2 = h * 3/4
            delta_x_1 = np.random.uniform(low = -max_x, high = max_x)
            lower_bound_x = max(-max_x, delta_x_1 - max_delta_x1_x2)
            upper_bound_x = min(max_x, delta_x_1 + max_delta_x1_x2)
            delta_x_2 = -np.random.uniform(low = lower_bound_x, high = upper_bound_x)
            
            max_delta_z1_z2 = h * 2/3
            delta_z_1 = np.random.uniform(low = 0.0, high = max_z)  
            lower_bound_z = max(-0.0, delta_z_1 - max_delta_z1_z2)
            upper_bound_z = min(max_z, delta_z_1 + max_delta_z1_z2)
            delta_z_2 = np.random.uniform(low = lower_bound_z, high = upper_bound_z)

            max_delta_y1_y2, min_delta_y1_y2 = h * 3/4, 0
            delta_y_1 = np.random.uniform(low = -max_y, high = max_delta_y1_y2 + max_y)
            lower_bound_y = min_delta_y1_y2 - delta_y_1
            upper_bound_y = max_delta_y1_y2 - delta_y_1
            delta_y_2 = np.random.uniform(low = lower_bound_y, high = upper_bound_y)


            # delta_y_1 = -max_y#np.random.uniform(low = -max_y/2, high = max_y)
            # delta_y_2 = h * 3/4 - delta_y_1
            # delta_y_1 = h * 3/4 + max_y
            # delta_y_2 = -max_y

            # print(max_delta_x1_x2)
            # print("delta_x_1, delta_z_1:", delta_x_1, delta_z_1)
            # print("lower_bound_x, upper_bound_x:", lower_bound_x, upper_bound_x)
            # print("lower_bound_z, upper_bound_z:", lower_bound_z, upper_bound_z)
                  
            delta_alpha_1 = np.random.uniform(low = -np.pi/8, high = np.pi/8)
            delta_beta_1 = np.random.uniform(low = -np.pi/8, high = np.pi/8) 
            delta_gamma_1 = np.random.uniform(low = -np.pi/8, high = np.pi/8)

            delta_alpha_2 = np.random.uniform(low = -np.pi/8, high = np.pi/8)
            delta_beta_2 = np.random.uniform(low = -np.pi/8, high = np.pi/8) 
            delta_gamma_2 = np.random.uniform(low = -np.pi/8, high = np.pi/8)

            print(f"Robot 1 xyz: {delta_x_1:.2f}, {delta_y_1:.2f}, {delta_z_1:.2f}")
            print(f"Robot 2 xyz: {delta_x_2:.2f}, {delta_y_2:.2f}, {delta_z_2:.2f}")



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
                if frame_count % 15 == 0:
                    full_pc_on_trajectory.append(get_object_particle_state(gym, sim) + shift)
                    pc_on_trajectory.append(get_partial_pointcloud_vectorized(*camera_args) + shift)
                    curr_trans_on_trajectory_1.append(get_pykdl_client(robot_1.get_arm_joint_positions())[1])
                    curr_trans_on_trajectory_2.append(get_pykdl_client(robot_2.get_arm_joint_positions())[1])
                              
                    if frame_count == 0:
                        mp_mani_point_1 = deepcopy(gym.get_actor_rigid_body_states(robot_1.env_handle, robot_1.robot_handle, gymapi.STATE_POS)[-3])
                        mp_mani_point_2 = deepcopy(gym.get_actor_rigid_body_states(robot_2.env_handle, robot_2.robot_handle, gymapi.STATE_POS)[-3])

                    terminate_count += 1
                    if terminate_count >= 10:
                        print("+++ Taking too long")
                        state = "reset"
                        terminate_count = 0
                frame_count += 1           
                
                action_1 = tvc_behavior_1.get_action()  
                action_2 = tvc_behavior_2.get_action() 
                # print("action_1, action_2:", action_1, action_2)
                if (action_1 is not None) and (action_2 is not None) and gym.get_sim_time(sim) - closed_loop_start_time <= 1.5: 
                    gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                    gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())

                else:   
                    if visualization:
                        output_file = f"/home/baothach/Downloads/test_cam_views_{data_point_count}.png"           
                        visualize_camera_views(gym, sim, envs_obj[0], cam_handles, \
                                            resolution=[cam_props.height, cam_props.width], output_file=output_file)   
                        
                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  
                    
                    # if sample_count == 0:
                    
                    pc_goal = get_partial_pointcloud_vectorized(*camera_args) + shift 
                    full_pc_goal = get_object_particle_state(gym, sim) + shift
                    _, final_trans_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                    _, final_trans_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                    # print("***Final x, y, z: ", final_pose["pose"]["p"]["x"], final_pose["pose"]["p"]["y"], final_pose["pose"]["p"]["z"] ) 
                    
                    for j, (curr_trans_1, curr_trans_2) in enumerate(zip(curr_trans_on_trajectory_1, curr_trans_on_trajectory_2)):    
                        # print(j)                    
                        p_1, R_1, twist_1 = tvc_behavior_1.get_transform(curr_trans_1, final_trans_1, get_twist=True)
                        mani_point_1 = curr_trans_1

                        p_2, R_2, twist_2 = tvc_behavior_2.get_transform(curr_trans_2, final_trans_2, get_twist=True)
                        mani_point_2 = curr_trans_2

                        partial_pcs = (pc_on_trajectory[j], pc_goal)
                        full_pcs = (full_pc_on_trajectory[j], full_pc_goal)

                        mani_point = np.array([mani_point_1[0,3], mani_point_1[1,3]- two_robot_offset, mani_point_1[2,3] + ROBOT_Z_OFFSET,
                                               -mani_point_2[0,3], -mani_point_2[1,3], mani_point_2[2,3] + ROBOT_Z_OFFSET]) \
                                    + np.concatenate((shift, shift))

                        data = {"full pcs": full_pcs, "partial pcs": partial_pcs, 
                                "pos": (p_1, p_2), "rot": (R_1, R_2), "twist": (twist_1, twist_2), \
                                "mani_point": mani_point, "obj_name": args.obj_name}

                        # with open(os.path.join(data_recording_path, "sample " + str(data_point_count) + ".pickle"), 'wb') as handle:
                        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                               
                        print("New data_point_count:", data_point_count)
                        data_point_count += 1       


                    if sample_count == 0:
                        for j in range(1,len(pc_on_trajectory)):    
                            partial_pcs = (pc_init, pc_on_trajectory[j])
                            full_pcs = (full_pc_init, full_pc_on_trajectory[j])

                            mani_point = np.array(list(mp_mani_point_1["pose"][0]) + list(mp_mani_point_2["pose"][0])) + np.concatenate((shift, shift))
                            data = {"full pcs": full_pcs, "partial pcs": partial_pcs, 
                                    "mani_point": mani_point, "obj_name": args.obj_name}
                            with open(os.path.join(mp_data_recording_path, "sample " + str(mp_data_point_count) + ".pickle"), 'wb') as handle:
                                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)                          
                            

                            print("New mp_data_point_count:", mp_data_point_count)
                            mp_data_point_count += 1      

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
 
        
        if sample_count == max_sample_count:  
            sample_count = 0            
            group_count += 1
            print("group count: ", group_count)
            state = "reset" 



        # if  data_point_count >= max_data_point_count or data_point_count >= max_data_point_per_variation:  
        if  mp_data_point_count >= max_mp_data_point_count or mp_data_point_count >= max_mp_data_point_per_variation:
            print_color(f"Done collecting data for {args.obj_name}")                  
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
