#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys

import open3d.visualization
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
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
import pickle
import timeit
from copy import deepcopy
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2
import transformations
import trimesh
sys.path.append("../")
sys.path.append(pkg_path + '/src/diffusion_defgoalnet')
from util.retraction_tool_utils import get_random_tool_pose, random_boolean_mask, sample_cylindrical_point
from diffusion_defgoalnet.util.retraction_cutting_utils import get_eef_position
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views, compute_pointcloud_pixel_coordinates_on_image, project_pointcloud_on_image
from utils.miscellaneous_utils import get_object_particle_state, write_pickle_data, print_lists_with_formatting, print_color, read_pickle_data, seed_all
from utils.point_cloud_utils import pcd_ize, down_sampling, transform_point_cloud, compute_world_to_eef, compose_4x4_homo_mat, compute_object_to_eef, rotate_around_z, invert_4x4_transformation_matrix
from utils.object_frame_utils import world_to_object_frame_camera_align
from retraction_tool_utils import check_plane, visualize_plane


ROBOT_Z_OFFSET = 0.22    #0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        davinci_dof_states['pos'][4] = 0.24

        if obj_height <= 0.12:
            davinci_dof_states['pos'][4] = 0.24
        else:
            davinci_dof_states['pos'][4] = 0.20
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)



if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--obj_name", "type": str, "default": 'cylinder_0', "help": "select variations of a primitive shape"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"},
            {"name": "--goal_model", "type": str, "default": "diffdef", "help": "Select goal model. Options: diffdef, defgoalnet"},
            {"name": "--datasize", "type": int, "help": "dataset size"},
            {"name": "--model_seed", "type": int, "help": "model seed"},
            {"name": "--sim_seed", "type": int, "help": "simulation seed", "default": 0},
            {"name": "--data_category", "type": str, "default": "deformernet", "help": "deformernet or MP"}])
    
    # Good combos for retraction tool sim camera recoding:
    # 1) --datasize 1000 --model_seed 0 --sim_seed 0
    
    
    seed_all(args.sim_seed)
    
    num_envs = args.num_envs
    args.headless = args.headless == "True"
    
    main_path = "/home/baothach/shape_servo_data/diffusion_defgoalnet"

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
        
        # if args.data_category == "deformernet":
        #     sim_params.flex.num_outer_iterations = 10
        # elif args.data_category == "MP":
        #     sim_params.flex.num_outer_iterations = 4

        sim_params.flex.num_outer_iterations = 6 #4 10 6
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-4
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-4
        sim_params.flex.deterministic_mode = True

        # # Set stress visualization parameters
        # sim_params.stress_visualization = True
        # sim_params.stress_visualization_min = 0.0   #1.0e2
        # sim_params.stress_visualization_max = 5e4   #1e5

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, sim_type, sim_params)

    # Get primitive shape dictionary to know the dimension of the object   
    with open(os.path.join(main_path, "object_data/retraction_tool/mesh", f"{args.obj_name}_info.pickle"), 'rb') as handle:
        data = pickle.load(handle)   
        # obj_height = data["height"] #data["height"]  data["radius"]*2
        if "height" in data:
            obj_height = data["height"]
            radius = data["radius"]
        # elif "radius" in data:
        #     obj_height = data["radius"]*2
        SCALE_RATIO = obj_height / 0.1
            


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

    asset_root = os.path.join(main_path, "object_data/retraction_tool/urdf")
    soft_asset_file = args.obj_name + ".urdf"    


    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.001)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    # soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    soft_thickness = 0.001#0.0005    # important to add some thickness to the soft body to avoid interpenetrations






    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

    # random_tool_pose = get_random_tool_pose(quadrant=1, translation=0.05, return_format="quaternion")

    rigid_asset_root = os.path.join(main_path, "object_data/retraction_tool/urdf_tool")
    rigid_asset_file = f"tool_1.urdf"   #f"tool_0.urdf"
    # rigid_pose = gymapi.Transform()
    # rigid_pose.p = gymapi.Vec3(random_tool_pose[0], soft_pose.p.y + random_tool_pose[1], random_tool_pose[2])
    # rigid_pose.r = gymapi.Quat(*list(random_tool_pose[3:]))    
    # rigid_pose.p = gymapi.Vec3(0.05, soft_pose.p.y, 0.1)
    # rigid_pose.r = gymapi.Quat(0.0, 0.0, 0, 1)
    asset_options.thickness = 0.003 # 0.002
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, asset_options)

 
    
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
    sphere_handles = []
    

    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    # create sphere asset
    asset_options.disable_gravity = False   


    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)             
        
        # add soft obj        
        env_obj = env
        # env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)

        # add rigid obj
        color = gymapi.Vec3(0,0,1)

        # # Create random boolean mask whether a tool will show up or not
        # tool_mask = np.array([1,0,0,0])
        # empty_quandrant = np.where(tool_mask == 0)[0]
        # print("\n\n===== PIKA PIKA =====")
        # print(f"Number of tools: {np.sum(tool_mask)}")
        # for j in range(4):
        #     if j in empty_quandrant:
        #         continue
        #     random_tool_pose = get_random_tool_pose(quadrant=j+1, translation=max(0.05, 0.05 * SCALE_RATIO), return_format="quaternion")
        #     # print("\n\n===== PIKA PIKA =====")
        #     # print(f"random_tool_pose: {random_tool_pose}")
        #     rigid_pose = gymapi.Transform()
        #     # rigid_pose.p = gymapi.Vec3(random_tool_pose[0], soft_pose.p.y + random_tool_pose[1], random_tool_pose[2])
        #     # rigid_pose.r = gymapi.Quat(*list(random_tool_pose[3:]))

        #     rigid_pose.p = gymapi.Vec3(0, soft_pose.p.y + 0.03, 0.03)
        #     rigid_pose.r = gymapi.Quat(*list(random_tool_pose[3:]))   
        #     # print("random_tool_pose:", random_tool_pose)
        #     # rigid_pose.r = gymapi.Quat(0.0, 0.0, 0, 1)        
             
        #     rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, f'tool_{j}', i, 1, segmentationId=12+j)
        #     gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        area_idx = 2    #np.random.randint(0, 3)   
        r_min, r_max = radius + 0.01, radius + 0.03    #0.03, 0.05
        if area_idx == 0:
            theta_min, theta_max = np.pi/6, np.pi/3
        elif area_idx == 1:
            theta_min, theta_max = np.pi/2 - np.pi/12, np.pi/2 + np.pi/12
        elif area_idx == 2:
            theta_min, theta_max = np.pi - np.pi/3, np.pi - np.pi/6
        
        # theta_min, theta_max = np.pi/1, np.pi/1
             
        z_min, z_max = 0, obj_height * 0.7

        # Sample points
        translation_vector = sample_cylindrical_point(r_min, r_max, theta_min, theta_max, z_min, z_max) 
        # translation_vector[1] += soft_pose.p.y    


        random_tool_pose = get_random_tool_pose(area_idx, translation_vector, return_format="quaternion")
        print("\n\n===== PIKA PIKA =====")
        print(f"random_tool_pose: {random_tool_pose}")
        print_color(f"area_idx: {area_idx}", color="green")
        rigid_pose = gymapi.Transform()
        rigid_pose.p = gymapi.Vec3(random_tool_pose[0], random_tool_pose[1] + soft_pose.p.y, random_tool_pose[2])
        rigid_pose.r = gymapi.Quat(*list(random_tool_pose[3:]))

        # rigid_pose.p = gymapi.Vec3(0, soft_pose.p.y + 0.03, 0.03)
        # rigid_pose.r = gymapi.Quat(*list(random_tool_pose[3:]))   
        # print("random_tool_pose:", random_tool_pose)
        # rigid_pose.r = gymapi.Quat(0.0, 0.0, 0, 1)        
            
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, f'tool', i, 1, segmentationId=12)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # object_handles.append(soft_actor)


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
        cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # cam_pos = gymapi.Vec3(0.3, -0.2, 0.3)

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
    cam_positions.append(gymapi.Vec3(0.0, soft_pose.p.y - 0.1, 0.1))   # put camera in front of the object
    cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.01))
  
    # cam_positions.append(gymapi.Vec3(0.15, soft_pose.p.y, 0.1))   # put camera on the side of object
    # cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.01))
    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])



    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    
    # args.datasize = 10  #1000
    # args.model_seed = 0
    data_recording_path = os.path.join(main_path, f"retraction_tool/evaluation/{args.goal_model}/datasize_{args.datasize}_seed_{args.model_seed}")
    os.makedirs(data_recording_path, exist_ok=True)
    
        
    print_color(f"\n\n*** Object name: {args.obj_name}\n\n")

    action_count = 0
    max_action_count = 6
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    data_point_count = len(os.listdir(data_recording_path))
    rospy.logwarn(f"data_point_count: {data_point_count}")
    # max_sample_count = 1    #2


    max_data_point_count = 1000 #100



    first_time = True
    save_intial_pc = True
    angle_idx = 0
    recorded_goal_pcs = []
    segmentationId_dict = {"robot": 11, "tool": 12}
    visualization = False

    dc_client = GraspDataCollectionClient()
    
    camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                    segmentationId_dict, "deformable", None, 0.004, False, "cpu"]
    camera_tool_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props,
                        segmentationId_dict, "tool", None, 0.004, False, "cpu"]

    
    start_time = timeit.default_timer()    

    close_viewer = False

    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])

    sys.path.append("/home/baothach/diffusion-point-cloud")
    sys.path.append("/home/baothach/diffusion-point-cloud/utils")
    sys.path.append("/home/baothach/diffusion-point-cloud/models")
    from models.vae_flow_goal_idx import BaoFlowVAE
    from models.defgoalnet import DefGoalNet
    import torch    
    
    # DiffDef Model
    goal_model = args.goal_model
    print_color(f"\n***** USING {goal_model} *****\n", "green")
    args.device = "cuda"

    
    if goal_model == "diffdef":
        # ckpt = torch.load("/home/baothach/diffusion-point-cloud/weights/retraction_cutting/weights_03.31.2025-22:30/ckpt_0.000000_22000.pt")
        if args.datasize == 10:
            ckpt = torch.load(f"/home/baothach/diffusion-point-cloud/weights/retraction_cutting/datasize_{args.datasize}_seed_{args.model_seed}/ckpt_0.000000_20000.pt")
        else:
            ckpt = torch.load(f"/home/baothach/diffusion-point-cloud/weights/retraction_cutting/datasize_{args.datasize}/ckpt_0.000000_20000.pt")
        model = BaoFlowVAE(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    elif goal_model == "defgoalnet":
        model = DefGoalNet(num_points = 512, embedding_size=256).to(args.device)
        model.load_state_dict(torch.load("/home/baothach/shape_servo_data/diffusion_defgoalnet/weights/retraction_tool/run1_object_frame_camera_align/epoch 200"))
        model.eval()    
    else:
        raise ValueError("Invalid goal model name")    

    ### Set up DeformerNet
    deformernet_model_main_path = "/home/baothach/shape_servo_DNN"
    sys.path.append(f"{deformernet_model_main_path}/rotation")
    from architecture_2 import DeformerNetMP as DeformerNet                                                                                                                      
    deformernet = DeformerNet(use_mp_input=True).to(args.device)
        
    # weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects/weights/run2_w_rot_no_MP" 
    weight_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/all_objects_retraction_tool/weights/run1_w_rot_no_MP"  
    deformernet.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {200}")))
    deformernet.eval()

    camera_view_matrix = np.array([
        [ 1.0,        0.0,       -0.0,        0.0       ],
        [-0.0,        0.6689648, -0.7432941,  0.0       ],
        [ 0.0,        0.7432941,  0.6689648,  0.0       ],
        [-0.0,        0.27353227, -0.45340943, 1.0     ]
    ])

    def transform_pc_world_to_object_frame(pc, camera_view_matrix, T_camera_to_object=None):
        pc = transform_point_cloud(pc, camera_view_matrix.T)    # Transform world to camera frame
        if T_camera_to_object is None:
            T_camera_to_object = world_to_object_frame_camera_align(pc)
        pc = transform_point_cloud(pc, T_camera_to_object)  # Transform camera to object frame
        return pc, T_camera_to_object
    
    max_percentage_passed = 0.0
    collide_tool = False
    percentage_passed = 0.0

    # Image stuff
    prepare_vis_cam = True
    if prepare_vis_cam:
        start_vis_cam = False 
        # cam_record_frequency = 5    # record image every N frames

        vis_frame_count = 0
        img_count = 0
        img_save_dir = os.path.join(main_path, f"retraction_tool/recordings", "test")
        os.makedirs(img_save_dir, exist_ok=True)
   
        vis_cam_width = 1000
        vis_cam_height = 1000
        vis_cam_props = gymapi.CameraProperties()
        vis_cam_props.width = vis_cam_width
        vis_cam_props.height = vis_cam_height

        # vis_cam_position = gymapi.Vec3(0.0, soft_pose.p.y - 0.1, 0.1)
        # vis_cam_target = gymapi.Vec3(0.0, soft_pose.p.y, 0.01)   

        # vis_cam_position = gymapi.Vec3(-0.05, soft_pose.p.y - 0.05, 0.15)
        # vis_cam_target = gymapi.Vec3(0.0, soft_pose.p.y, 0.01)   

        vis_cam_position = gymapi.Vec3(-0.0, soft_pose.p.y - 0.05, 0.15)    # [IMPORTANT] worked for visualizing predicted goals from DiffDef
        vis_cam_target = gymapi.Vec3(0.0, soft_pose.p.y, 0.01)   

        # vis_cam_position = gymapi.Vec3(0.00, soft_pose.p.y - 0.01, 0.15)    # [IMPORTANT] worked for visualizing target plane
        # vis_cam_target = gymapi.Vec3(0.0, soft_pose.p.y, 0.01)           


        vis_cam_handle = gym.create_camera_sensor(env_obj, vis_cam_props)
        gym.set_camera_location(vis_cam_handle, env_obj, vis_cam_position, vis_cam_target)  


    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.15)    # 0.24           

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
                    saved_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))

                    tool_pc = get_partial_pointcloud_vectorized(*camera_tool_args)
                    init_pc = get_partial_pointcloud_vectorized(*camera_args) 

                    num_pts = 512       
                    num_points_deformernet = 1024                        

                    init_pc = down_sampling(init_pc, num_pts)
                    context_pc = down_sampling(tool_pc, num_pts)

                    init_pc, T_camera_to_object = transform_pc_world_to_object_frame(init_pc, camera_view_matrix, T_camera_to_object=None)
                    context_pc, _ = transform_pc_world_to_object_frame(context_pc, camera_view_matrix, T_camera_to_object)

                    if goal_model == "diffdef":
                        diffdef_shift = init_pc.mean(axis=0)
                        scale = (init_pc - diffdef_shift).flatten().std()                 
                        init_pc_tensor = torch.from_numpy((init_pc - diffdef_shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)
                        context_pc_tensor = torch.from_numpy((context_pc - diffdef_shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)                         
                        
                        z = torch.randn([1, ckpt['args'].latent_dim]).to(args.device) #* 0.1
                        # x = model.sample(z, context_pc_tensor, init_pc_tensor, num_points_deformernet, flexibility=ckpt['args'].flexibility)
                        x = model.sample(z, context_pc_tensor, init_pc_tensor, 2048, flexibility=ckpt['args'].flexibility)
                        
                        first_goal_pc_object_frame = x[0].detach().cpu().numpy()*scale + diffdef_shift
                        
                    elif goal_model == "defgoalnet":
                        diffdef_shift = init_pc.mean(axis=0)
                        scale = (init_pc - diffdef_shift).flatten().std()                 
                        init_pc_tensor = torch.from_numpy((init_pc - diffdef_shift) / scale).unsqueeze(0).permute(0,2,1).float().to(args.device)  # shape (1, 3, num_pts)
                        context_pc_tensor = torch.from_numpy((context_pc - diffdef_shift) / scale).unsqueeze(0).permute(0,2,1).float().to(args.device)  # shape (1, 3, num_pts)
                        first_goal_pc_object_frame = model(init_pc_tensor, context_pc_tensor).permute(0,2,1).squeeze().detach().cpu().numpy()*scale + diffdef_shift      

                    goal_pc_camera_frame = transform_point_cloud(first_goal_pc_object_frame, invert_4x4_transformation_matrix(T_camera_to_object))  # Transform object frame to camera frame
                                            
                    if visualization:                       
                        pcd = pcd_ize(init_pc, color=[0,0,0])
                        pcd_goal = pcd_ize(first_goal_pc_object_frame, color=[1,0,0])
                        pcd_context = pcd_ize(context_pc, color=[0,1,0])
                        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        open3d.visualization.draw_geometries([pcd, pcd_goal, pcd_context, coor])

                    first_time = False

                    # Evaluation with plane
                    constrain_plane = np.array([0, 1, 0, -soft_pose.p.y - 0.0])
                    x_range=[-0.1,0.1]  # [-0.1,0.1]
                    y_range=[soft_pose.p.y-0.05, soft_pose.p.y+0.05]
                    z_range=0.13
                    num_pts=1000000     # 1000000 for plane projection, 
                    vis_plane = visualization

                    max_percentage_passed = check_plane(constrain_plane, get_object_particle_state(gym, sim), 
                                                    vis=vis_plane, 
                                                    x_range=x_range, y_range=y_range, z_range=z_range)
                    
                    if prepare_vis_cam:
                        radius = 5 #5  #5  #2 #1        
                        # color in BGR format
                        color = (0, 0, 255)
                        thickness = -1  #-1  #2 
                        
                        ### Predicted goal point cloud from DiffDef/DefGoalNet
                        # goal_pc_world_frame =  transform_point_cloud(goal_pc_camera_frame, invert_4x4_transformation_matrix(camera_view_matrix.T))
                        # goal_pc_world_frame = down_sampling(goal_pc_world_frame, 200)

                        ### Target plane point cloud
                        goal_pc_world_frame = np.array(visualize_plane(constrain_plane, x_range, y_range, z_range, num_pts=num_pts))

                        # Project point cloud on image
                        projected_pixels = compute_pointcloud_pixel_coordinates_on_image(gym, sim, env_obj, 
                                                                                         goal_pc_world_frame, 
                                                                                         vis_cam_handle, vis_cam_props)
                        prepare_vis_cam = False  
                        start_vis_cam = True                  


                state = "generate preshape"
                
                frame_count = 0

                current_pc = get_object_particle_state(gym, sim)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                # pc_ros_msg = fix_object_frame(pc_ros_msg)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 

        if start_vis_cam:   
            if vis_frame_count % 50 == 0:    # 2
                # output_file = os.path.join(img_save_dir, f"cam_view_{img_count}.png")
                img_path = os.path.join(img_save_dir, f'img{img_count:03}.png')
                project_pointcloud_on_image(gym, sim, env_obj, 
                                            vis_cam_handle, vis_cam_props, projected_pixels, 
                                            img_save_dir=img_path, color=color, radius=radius, thickness=thickness, overlay_alpha=0.6)
                img_count += 1
                
            vis_frame_count += 1 


        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]


            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                # state = "reset"  
                all_done = True              
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


             
            # full_pc_init = get_object_particle_state(gym, sim)




        if state == "move to preshape":         
            action = mtp_behavior.get_action()

            if action is not None:
                gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())      
                        
            if mtp_behavior.is_complete():
                state = "grasp object"   
                rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!") 

        
        if state == "grasp object":             
            rospy.loginfo("**Current state: " + state)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), -2.5)
            gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -3.0)         

            g_1_pos = 0.35
            g_2_pos = -0.35
            dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)
            if dof_states['pos'][8] < 0.35:
                                       
                state = "get shape servo plan"
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), g_1_pos)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), g_2_pos) 

                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                saved_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))

                # eef_positions = get_eef_position(attachment_positions[start_idx,:2], 
                #                                 attachment_positions[end_idx,:2], 
                #                                 45, obj_height*1, vis=False)

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 

            delta_alpha, delta_beta, delta_gamma = 1e-6, 1e-6, 1e-6   

            delta = -0.07 * SCALE_RATIO    #-0.07 * SCALE_RATIO -0.085 * SCALE_RATIO # 0.07 for predicted goal projection, 0.085 for plane projection
            magnitude = np.sqrt(delta**2 * 2)
            if area_idx == 0:
                options = [np.pi/6, np.pi/3]
                retract_angle = options[angle_idx]
                delta_x, delta_y = magnitude * np.cos(retract_angle), -magnitude * np.sin(retract_angle)
            elif area_idx == 1:
                options = [np.pi/6, np.pi/6 + 2*np.pi/3]
                retract_angle = options[angle_idx]
                delta_x, delta_y = magnitude * np.cos(retract_angle), -magnitude * np.sin(retract_angle)
            elif area_idx == 2:
                # options = [np.pi/6, np.pi/3]
                retract_angle = np.pi * 0.4
                delta_x, delta_y = -magnitude * np.cos(retract_angle), -magnitude * np.sin(retract_angle)
            delta_z = 0

            x = delta_x + init_pose[0,3]
            y = delta_y + init_pose[1,3]
            z = init_pose[2,3]  #delta_z + init_pose[2,3]
            alpha = delta_alpha + init_eulers[0]
            beta = delta_beta + init_eulers[1]
            gamma = delta_gamma + init_eulers[2]

            tvc_behavior = TaskVelocityControl2([x,y,z,alpha,beta,gamma], robot, sim_params.dt, 3, vel_limits=vel_limits)
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            
            action_count += 1
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            state = "move to goal"


        if state == "move to goal":
            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            if main_ins_pos <= 0.042:
                rospy.logerr("Exceed joint constraint")
                group_count += 1
                # state = "reset"
                all_done = True


            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]

            collide_tool = 12 in contacts
            if collide_tool:
                rospy.logerr("Tissue collided with surgical tool")


            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                
                group_count += 1
                # state = "reset"
                all_done = True

            else:      
                action = tvc_behavior.get_action() 
                # print(f"{gym.get_sim_time(sim) - closed_loop_start_time:.2f} s") 
                if action is not None and gym.get_sim_time(sim) - closed_loop_start_time <= 1.5: 
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())
                    # print("\n\n\n=================PIKA PIKA\n\n\n=================")
                    # print("type(action.get_joint_position()):", type(action.get_joint_position()))
                    # print("shape action.get_joint_position():", action.get_joint_position().shape)
                else:   

                    rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  

                    if visualization:
                        output_file = f"/home/baothach/Downloads/test_cam_views_{angle_idx}.png"           
                        visualize_camera_views(gym, sim, envs_obj[0], cam_handles, \
                                            resolution=[cam_props.height, cam_props.width], output_file=output_file)                    

                    frame_count = 0
                    angle_idx += 1
                    state = "get shape servo plan"
                    # state = "reset"

                    _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                    init_eulers = transformations.euler_from_matrix(init_pose)

                    percentage_passed = check_plane(constrain_plane, get_object_particle_state(gym, sim), 
                                                    vis=vis_plane, 
                                                    x_range=x_range, y_range=y_range, z_range=z_range)
                    
                    max_percentage_passed = max(max_percentage_passed, percentage_passed)

                    zero_velocity_action = np.zeros(8, dtype=np.float32)
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, zero_velocity_action)
                    state = "meoww"

        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0         

            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            # gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            
            print("Sucessfully reset robot and object!")

            state = "get shape servo plan"
 
        

        if  data_point_count >= max_data_point_count:                    
            all_done = True 

        # if max_percentage_passed >= 99.0 or action_count > max_action_count or collide_tool:
        #     print_color(f"\n\nFINAL RESULT:\ncollide_tool: {collide_tool}.\nmax_percentage_passed: {max_percentage_passed}.\n\n", color="green")

        #     zero_velocity_action = np.zeros(8, dtype=np.float32)
        #     gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, zero_velocity_action)
        #     # all_done = True
        #     rospy.logerr("Simulation succesfully completed. Data saved!")


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
    print("total data pt count: ", data_point_count)
