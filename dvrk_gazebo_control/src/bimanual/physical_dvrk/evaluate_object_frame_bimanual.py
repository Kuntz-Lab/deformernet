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
# from PIL import Image
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2
from scipy import interpolate


import torch

from utils.miscellaneous_utils import get_object_particle_state, write_pickle_data, print_lists_with_formatting, print_color, read_pickle_data, pcd_ize
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views
from utils.point_cloud_utils import down_sampling, object_to_world_frame, transform_point_cloud, compute_world_to_eef, compose_4x4_homo_mat, compute_object_to_eef, rotate_around_z
from utils.object_frame_utils import world_to_object_frame_PCA


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
    
    mesh_name = f"{args.prim_name}_{args.obj_name%10}"
    
    args.headless = args.headless == "True"
    # args.inside = args.inside == "True"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"
    

    object_category = f"{args.prim_name}_{args.stiffness}"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{object_category}Pa/evaluate"
    objects_path = "/home/baothach/sim_data/Custom"        



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
        sim_params.flex.num_outer_iterations = 10#4
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

    # Get primitive shape dictionary to know the dimension of the object   
    object_meshes_path = os.path.join(objects_path, f"Custom_mesh/physical_dvrk/multi_{object_category}Pa")

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


    asset_root = os.path.join(objects_path, f"Custom_urdf/physical_dvrk/bimanual/multi_{object_category}Pa")    # _eval
    # print_color(f"\n\n***asset_root: {asset_root}\n\n", "green")
    # print("object name: ", args.obj_name)



    soft_asset_file = args.obj_name + ".urdf"    
    # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/test"
    # soft_asset_file = "long_box.urdf"
    # asset_root = "/home/baothach/Downloads"
    # soft_asset_file = "test_box.urdf"



    soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.5)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)

    if args.prim_name == "box": 
        soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, thickness/2 + 0.001)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    elif args.prim_name == "cylinder": 
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, r)
        soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    elif args.prim_name == "hemis":
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, -o)

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
        cam_target = gymapi.Vec3(0.0, -0.36, 0.1)

        # cam_pos = gymapi.Vec3(0.5, -0.8, 0.5)
        # cam_target = gymapi.Vec3(0.0, -0.36, 0.1)
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
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
    rospy.logerr(f"Object type ... {object_category}.") 

 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True
    object_frame = True    
    print_color(f"Using object frame: {object_frame}", "green")


    # Set up DNN:
    device = torch.device("cuda")        


    ### Set up DeformerNet
    deformernet_model_main_path = "/home/baothach/shape_servo_DNN"
    sys.path.append(f"{deformernet_model_main_path}/bimanual")

    from bimanual_architecture import DeformerNetBimanualRot
    model = DeformerNetBimanualRot(use_mp_input=True).to(device)
        
    if object_frame:
        # weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects_object_frame/weights/run1_w_rot_w_MP" 
        weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects_object_frame_5_fix_direction/weights/run1_w_rot_w_MP" 
    else:
        weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects/weights/run3_w_rot_w_MP"    
    model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {200}")))    # 200
    model.eval()






    goal_recording_path = os.path.join(main_path, "goal_data")
    
    chamfer_recording_path = os.path.join(main_path, "chamfer_results")
    if object_frame:
        chamfer_recording_path = os.path.join(chamfer_recording_path, "object_frame")
    else:
        chamfer_recording_path = os.path.join(chamfer_recording_path, "world_frame")
    
    os.makedirs(chamfer_recording_path, exist_ok=True)


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
    with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'rb') as handle:
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


    init_pc_numpy = down_sampling(goal_datas["partial pcs"][0])  # first goal pc
    init_pc_tensor = torch.from_numpy(init_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)  
    gt_mps = np.array(goal_datas["mani_point"])
    gt_mp_1 = gt_mps[:3]
    gt_mp_2 = gt_mps[3:]

    full_pc_numpy = goal_datas["full pcs"][0]
      
    goal_pos_1 = goal_datas["pos"][0]
    goal_rot_1 = goal_datas["rot"][0]         
    goal_pos_2 = goal_datas["pos"][1]
    goal_rot_2 = goal_datas["rot"][1]    

    
    start_time = timeit.default_timer()    
    close_viewer = False
    robot_2 = Robot(gym, sim, envs[0], kuka_handles_2[0])
    robot_1 = Robot(gym, sim, envs[0], kuka_handles[0])

    segmentationId_dict = {"robot_1": 10, "robot_2": 11, "cylinder": 12}
    camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                    segmentationId_dict, "deformable", None, 0.002, False, "cpu"]    
    shift = np.array([0.0, -soft_pose.p.y, camera_args[-3]])

    while (not close_viewer) and (not all_done): 



        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

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
                best_mp_1 = gt_mp_1 - shift
                best_mp_2 = gt_mp_2 - shift



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

            plan_count += 1

            current_pc_numpy = down_sampling(get_partial_pointcloud_vectorized(*camera_args))# + shift                   

            mani_point_1 = init_pose_1[:3,3] + np.array([0,-two_robot_offset, ROBOT_Z_OFFSET])# + shift
            mani_point_2 = init_pose_2[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET])# + shift

            # Transform points and manipulation positions
            if object_frame:
                # world_to_object_H = world_to_object_frame_PCA(current_pc_numpy)
                world_to_object_H = world_to_object_frame_PCA(current_pc_numpy, mani_point_1)
                current_pc_numpy = transform_point_cloud(current_pc_numpy, world_to_object_H)
                
                transformed_goal_pc_numpy = transform_point_cloud(goal_pc_numpy - shift, world_to_object_H)
                # transformed_goal_pc_numpy = transform_point_cloud(goal_pc_numpy, world_to_object_H)
                goal_pc_tensor = torch.from_numpy(transformed_goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 
                pcd_goal = pcd_ize(transformed_goal_pc_numpy, color=[1,0,0])

                mani_point_1 = transform_point_cloud(mani_point_1.reshape(1, -1), world_to_object_H).flatten()
                mani_point_2 = transform_point_cloud(mani_point_2.reshape(1, -1), world_to_object_H).flatten()
            else:
                current_pc_numpy += shift
                mani_point_1 += shift
                mani_point_2 += shift

            pcd = pcd_ize(current_pc_numpy, color=[0,0,0])

            node_dist = np.linalg.norm(full_pc_goal - (get_object_particle_state(gym, sim) + shift))
            saved_nodes.append(node_dist)
            rospy.logwarn(f"Node distance: {node_dist}")

            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            rospy.logwarn(f"chamfer distance: {chamfer_dist}")

            neigh = NearestNeighbors(n_neighbors=50)
            neigh.fit(current_pc_numpy)
            
            _, nearest_idxs_1 = neigh.kneighbors(mani_point_1.reshape(1, -1))
            mp_channel_1 = np.zeros(current_pc_numpy.shape[0])
            mp_channel_1[nearest_idxs_1.flatten()] = 1

            _, nearest_idxs_2 = neigh.kneighbors(mani_point_2.reshape(1, -1))
            mp_channel_2 = np.zeros(current_pc_numpy.shape[0])
            mp_channel_2[nearest_idxs_2.flatten()] = 1
            
            modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel_1, mp_channel_2])
            current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)                
            
            assert modified_pc.shape == (5,1024) 

            colors = np.zeros((1024,3))
            colors[nearest_idxs_1.flatten()] = [1,0,0]
            colors[nearest_idxs_2.flatten()] = [0,1,0]
            pcd.colors =  open3d.utility.Vector3dVector(colors)
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_1_sphere.paint_uniform_color([0,0,1])
            mani_point_1_sphere.translate(tuple(mani_point_1))
            mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_2_sphere.paint_uniform_color([1,0,0])
            mani_point_2_sphere.translate(tuple(mani_point_2))
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

            if object_frame: 
                coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
                coor_object.transform(world_to_object_H)   
                open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.00,0,0)), \
                                                    mani_point_1_sphere, mani_point_2_sphere, coor, coor_object])
            else:
                open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.00,0,0)), \
                                                    mani_point_1_sphere, mani_point_2_sphere, coor]) 

            print("\n===================\n")
            with torch.no_grad():
                print(current_pc_tensor.shape, goal_pc_tensor.shape)
                pos, rot_mat_1, rot_mat_2 = model(current_pc_tensor, goal_pc_tensor)    # [:,:3,:]
                pos *= 0.001
                pos, rot_mat_1, rot_mat_2 = pos.detach().cpu().numpy(), rot_mat_1.detach().cpu().numpy(), rot_mat_2.detach().cpu().numpy()
                
            
            if object_frame:
                eef_pose_1_object_frame = compose_4x4_homo_mat(rot_mat_1, pos[0][:3])
                eef_pose_2_object_frame = compose_4x4_homo_mat(rot_mat_2, pos[0][3:])   
                
                modified_world_to_object_H = deepcopy(world_to_object_H)
                modified_world_to_object_H[:3,3] = 0

                eef_pose_1_world_frame = compute_world_to_eef(modified_world_to_object_H, eef_pose_1_object_frame)
                eef_pose_2_world_frame = compute_world_to_eef(modified_world_to_object_H, eef_pose_2_object_frame) 
                # print("BEFORE eef_pose_2_world_frame:", eef_pose_2_world_frame)
                eef_pose_2_world_frame = rotate_around_z(eef_pose_2_world_frame, np.pi) 
                # print("AFTER eef_pose_2_world_frame:", eef_pose_2_world_frame)
                # print("\n")
                    
                desired_pos_1 = (eef_pose_1_world_frame[:3,3] + init_pose_1[:3,3]).flatten()
                desired_rot_1 = eef_pose_1_world_frame[:3,:3] @ init_pose_1[:3,:3]
                desired_pos_2 = (eef_pose_2_world_frame[:3,3] + init_pose_2[:3,3]).flatten()
                desired_rot_2 = eef_pose_2_world_frame[:3,:3] @ init_pose_2[:3,:3]
                
                goal_1_world_frame = compose_4x4_homo_mat(np.eye(3), goal_pos_1.flatten())
                goal_1_object_frame = compute_object_to_eef(world_to_object_H, goal_1_world_frame)
                goal_2_world_frame = compose_4x4_homo_mat(np.eye(3), goal_pos_2.flatten())
                goal_2_world_frame = rotate_around_z(goal_2_world_frame, np.pi)
                goal_2_object_frame = compute_object_to_eef(world_to_object_H, goal_2_world_frame) 

                print("model prediction (obj frame):", pos)
                print("goal 1 obj frame:", goal_1_object_frame)
                print("goal 2 obj frame:", goal_2_object_frame)
                print("\n")
                
                print("predicted_pos_1_world:", eef_pose_1_world_frame[:3,3])
                print("goal_pos_1:", goal_pos_1.flatten())
                print("\n")
                
                print("predicted_pos_2_world:", eef_pose_2_world_frame[:3,3])
                print("goal_pos_2:", goal_pos_2.flatten())
            else:            
                desired_pos_1 = (pos[0][:3] + init_pose_1[:3,3]).flatten()
                desired_rot_1 = rot_mat_1 @ init_pose_1[:3,:3]
                desired_pos_2 = (pos[0][3:] + init_pose_2[:3,3]).flatten()
                desired_rot_2 = rot_mat_2 @ init_pose_2[:3,:3]

                temp1 = np.eye(4)
                temp1[:3,:3] = rot_mat_1
                temp2 = np.eye(4)
                temp2[:3,:3] = goal_rot_1    
                print("========ROBOT 1=========")        
                print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                print("goal_pos, goal_rot:", goal_pos_1, transformations.euler_from_matrix(temp2)) 
                print("\n")

                # temp1 = np.eye(4)
                temp1[:3,:3] = rot_mat_2
                # temp2 = np.eye(4)
                temp2[:3,:3] = goal_rot_2    
                print("========ROBOT 2=========")        
                print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                print("goal_pos, goal_rot:", goal_pos_2, transformations.euler_from_matrix(temp2)) 
                print("\n")

            tvc_behavior_1 = TaskVelocityControl2([*desired_pos_1, desired_rot_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                pos_threshold = 2e-3, ori_threshold=5e-2)
            tvc_behavior_2 = TaskVelocityControl2([*desired_pos_2, desired_rot_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                pos_threshold = 2e-3, ori_threshold=5e-2)
            


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
                
                chamfer_dist = np.linalg.norm(full_pc_goal - (get_object_particle_state(gym, sim) + shift))                
                saved_nodes.append(chamfer_dist)
                final_node_distances.append(999+min(saved_nodes)) 

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_sampling(get_partial_pointcloud_vectorized(*camera_args)) + shift)   

                current_pc = down_sampling(get_partial_pointcloud_vectorized(*camera_args))
                if object_frame:
                    current_pc = transform_point_cloud(current_pc, world_to_object_H)
                else:
                    current_pc += shift
                pcd = pcd_ize(current_pc)

                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(999+min(saved_chamfers))         

                print("***final node distance: ", min(saved_nodes)/full_pc_goal.shape[0]*1000)
                print("***final chamfer distance: ", min(saved_chamfers))

                goal_count += 1
            
            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time \
                    or plan_count > 6:  #6: # get new plan k times
                    
                    rospy.logerr("Timeout")
                    state = "reset" 
                    
                    chamfer_dist = np.linalg.norm(full_pc_goal - (get_object_particle_state(gym, sim) + shift))                    
                    saved_nodes.append(chamfer_dist)
                    final_node_distances.append(1999+min(saved_nodes)) 

                    current_pc = down_sampling(get_partial_pointcloud_vectorized(*camera_args))
                    if object_frame:
                        current_pc = transform_point_cloud(current_pc, world_to_object_H)
                    else:
                        current_pc += shift
                    pcd = pcd_ize(current_pc) 
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
                        _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                        init_eulers_1 = transformations.euler_from_matrix(init_pose_1)

                        _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                        init_eulers_2 = transformations.euler_from_matrix(init_pose_2)    
                        state = "get shape servo plan"    
                    else:
                        gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())

                    # Terminal conditions
                                     
                    converge = all(abs(pos.squeeze()) <= 0.005) 

                    if converge or chamfer_dist < min_chamfer_dist:
                        print_color("Converged or chamfer distance is small enough. Let's reset")
                        # print(converge, chamfer_dist)
                        
                        node_dist = np.linalg.norm(full_pc_goal - (get_object_particle_state(gym, sim) + shift))             
                        print("***final node distance: ", node_dist/full_pc_goal.shape[0]*1000)
                        final_node_distances.append(node_dist) 


                        current_pc = down_sampling(get_partial_pointcloud_vectorized(*camera_args))
                        if object_frame:
                            current_pc = transform_point_cloud(current_pc, world_to_object_H)
                        else:
                            current_pc += shift
                        pcd = pcd_ize(current_pc)
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        final_chamfer_distances.append(chamfer_dist) 
                        print("***final chamfer distance: ", chamfer_dist)


                        goal_count += 1

                        state = "reset" 



        if state == "reset":   

            # pcd.paint_uniform_color([0,0,0])
            # open3d.visualization.draw_geometries([pcd, pcd_goal])

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

            # final_data = {"node": final_node_distances, "chamfer": final_chamfer_distances, "num_nodes": full_pc_goal.shape[0]}
            # with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
            #     pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        if  goal_count >= max_goal_count:                    
            all_done = True 
           

            # final_data = {"node": final_node_distances, "chamfer": final_chamfer_distances, "num_nodes": full_pc_goal.shape[0]}
            # with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
            #     pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


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
