#!/usr/bin/env python3
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
from utils.point_cloud_utils import pcd_ize, down_sampling, transform_point_cloud, compute_world_to_eef, compose_4x4_homo_mat, compute_object_to_eef, rotate_around_z, invert_4x4_transformation_matrix
from utils.object_frame_utils import world_to_object_frame_camera_algin
import pickle5 as pickle
import timeit
from copy import deepcopy
from scipy import interpolate

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl2
import transformations
from sklearn.neighbors import NearestNeighbors


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
            {"name": "--stiffness", "type": str, "default": "5k", "help": "Select object stiffness. Options: 1k, 5k, 10k"},
            {"name": "--obj_name", "type": int, "default": 0, "help": "select variations of a primitive shape"},
            {"name": "--goal_model", "type": str, "default": "diffdef", "help": "Select goal model. Options: diffdef, defgoalnet"},
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    mesh_name = f"{args.prim_name}_{args.obj_name%10}"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"
    object_category = f"{args.prim_name}_{args.stiffness}"

    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.)
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
    if args.prim_name == "box":
        object_meshes_path = f"/home/baothach/sim_data/Custom/Custom_mesh/physical_dvrk/multi_{object_category}Pa"    
        with open(os.path.join(object_meshes_path, f"primitive_dict_{args.prim_name}.pickle"), 'rb') as handle:
            data = pickle.load(handle)       
        h = data[args.obj_name]["height"]
        w = data[args.obj_name]["width"]
        thickness = data[args.obj_name]["thickness"]
    elif args.prim_name == "cylinder":
        object_meshes_path = f"/home/baothach/sim_data/Custom/Custom_mesh/multi_{object_category}Pa"    
        with open(os.path.join(object_meshes_path, f"primitive_dict_{args.prim_name}.pickle"), 'rb') as handle:
            data = pickle.load(handle)       
        r = data[args.obj_name]["radius"] / 2
        h = data[args.obj_name]["height"] / 2
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

    # asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/physical_dvrk/bimanual/multi_{object_category}Pa"
    # soft_asset_file = args.obj_name + ".urdf"    
    # asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/object_packaging"
    # soft_asset_file = f"dog_1.urdf"

    if args.prim_name == "box":
        asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/physical_dvrk/bimanual/multi_{object_category}Pa"
        soft_asset_file = args.obj_name + ".urdf"   
    elif args.prim_name == "cylinder":
        asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/bimanual_multi_{object_category}Pa" 
        soft_asset_file = args.obj_name + ".urdf"

    # asset_root = f"/home/baothach/sim_data/Custom/Custom_urdf/physical_dvrk/bimanual/multi_{object_category}Pa"
    # soft_asset_file = args.obj_name + ".urdf" 

    soft_pose = gymapi.Transform()
    
    if args.prim_name == "box": 
        soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, thickness/2 + 0.001)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    elif args.prim_name == "cylinder": 
        soft_pose.p = gymapi.Vec3(0, -two_robot_offset/2, r + 0.001)
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

        
    # Load rigid kidney asset
    rigid_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/object_packaging"
    rigid_asset_file = f"amazon_box.urdf"
    amazon_box_scale = 1.5
    
    rigid_pose = gymapi.Transform()
    # # rigid_pose.p = gymapi.Vec3(0.2, -two_robot_offset/2 - 0.1, 0.00) 
    # # rigid_pose.p = gymapi.Vec3(w / 2 + 0.1*amazon_box_scale/2 + 0.05, -two_robot_offset/2, 0.00) 
    # rigid_pose.p = gymapi.Vec3(0.25, -two_robot_offset/2, 0.00) 
    
    object_span = w/2 if args.prim_name == "box" else r
    # # box_x = np.random.uniform(0.2, 0.3) * np.random.choice([-1, 1])
    box_x = np.random.uniform(object_span + 0.1*amazon_box_scale/2 + 0.05, 0.3) * np.random.choice([-1, 1])
    box_y = np.random.uniform(-0.1, 0.1)
    rigid_pose.p = gymapi.Vec3(box_x, box_y-two_robot_offset/2, 0.00)
    print_color(f"rigid_pose.p: {rigid_pose.p}", "green")
        
    # kidney_angle, tissue_angle = get_kidney_and_tissue_angle()  # get random kidney and tissue orientations
    kidney_angle = 0#np.pi/6
    eulers = [kidney_angle, 0, np.pi/2]
    quat = transformations.quaternion_from_euler(*eulers)
    rigid_pose.r = gymapi.Quat(*list(quat))

    rigid_asset_options = gymapi.AssetOptions()
    rigid_asset_options.fix_base_link = True
    rigid_asset_options.thickness = 0.003 # 0.002
    rigid_asset_options.disable_gravity = True    
    
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, rigid_asset_options)    
 
    
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

        # add rigid kidney obj
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, 'rigid', i, 0, segmentationId=10)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        

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

    # cam_positions.append(gymapi.Vec3(-0.0, soft_pose.p.y + 0.001, 0.5))   # put camera on top of object
    # cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.01))
    cam_positions.append(gymapi.Vec3(-0.0, -two_robot_offset/2 + 0.001, 0.5))   # put camera on top of object
    cam_targets.append(gymapi.Vec3(0.0, -two_robot_offset/2, 0.01))

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
    
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0


    pc_on_trajectory = []
    full_pc_on_trajectory = []
    curr_trans_on_trajectory_1 = []
    curr_trans_on_trajectory_2 = []
    first_time = True
    save_intial_pc = True
    switch = True
    total_computation_time = 0
    data = []
    action_count = 0
    goal_count = 0
    max_goal_count = 5
    action_length = 4
    reset_count = 0
    max_reset_count = 10
    max_action_count = 10
    enclosed_percent = 0.0
    

    dc_client = GraspDataCollectionClient()
    segmentationId_dict = {"robot_11": 10, "robot_2": 11, "rigid": 10}
    camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                    segmentationId_dict, "deformable", None, 0.002, False, "cpu"]    
    rigid_camera_args = [gym, sim, envs_obj[0], cam_handles[0], cam_props, 
                        segmentationId_dict, "rigid", None, 0.002, False, "cpu"] 
    visualization = False
    output_file = f"/home/baothach/Downloads/test_cam_views.png" 
    
    sys.path.append("/home/baothach/diffusion-point-cloud")
    sys.path.append("/home/baothach/diffusion-point-cloud/utils")
    sys.path.append("/home/baothach/diffusion-point-cloud/models")
    from models.vae_flow_3 import BaoFlowVAE
    from models.defgoalnet import DefGoalNet
    import torch    
    
    # DiffDef Model
    goal_model = args.goal_model
    print_color(f"\n***** USING {goal_model} *****\n", "green")
    args.device = "cuda"
    
    if goal_model == "diffdef":
        ckpt = torch.load("/home/baothach/diffusion-point-cloud/weights/object_packaging/weights_04.07.2025-17:44/ckpt_0.000000_2000.pt")
        model = BaoFlowVAE(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
    elif goal_model == "defgoalnet":
        model = DefGoalNet(num_points = 512, embedding_size=256).to(args.device)
        model.load_state_dict(torch.load("/home/baothach/shape_servo_data/diffusion_defgoalnet/object_packaging_multimodal/weights/defgoalnet/run2_object_frame_camera_align/epoch 1000"))
        model.eval()    
    else:
        raise ValueError("Invalid goal model name")    

    ### Set up DeformerNet
    deformernet_model_main_path = "/home/baothach/shape_servo_DNN"
    sys.path.append(f"{deformernet_model_main_path}/bimanual")

    from bimanual_architecture import DeformerNetBimanualRot
    # deformernet = DeformerNetBimanualRot(use_mp_input=False).to(args.device)
    deformernet = DeformerNetBimanualRot(use_mp_input=True).to(args.device)
        
    # weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects/weights/run2_w_rot_no_MP" 
    weight_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/all_objects_object_frame_camera_align/weights/run1_w_rot_w_MP"  
    deformernet.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {200}")))
    deformernet.eval()
 
    results_path = f"/home/baothach/shape_servo_data/diffusion_defgoalnet/object_packaging_multimodal/evaluation/{object_category}/{goal_model}" 
    os.makedirs(results_path, exist_ok=True)
    data_point_count = len(os.listdir(results_path))            
    max_data_point_count = 100
            
    start_time = timeit.default_timer()    

    close_viewer = False

    robot_2 = Robot(gym, sim, envs[0], kuka_handles_2[0])
    robot_1 = Robot(gym, sim, envs[0], kuka_handles[0])

    def check_points_enclosed_percentage(point_cloud, square_size=0.1 * amazon_box_scale):
        square_center = [rigid_pose.p.x, rigid_pose.p.y, 0]
        transformed_points = point_cloud[:, :3] - square_center
        
        # Define bounds for enclosed region
        half_size = square_size / 2
        x_min, x_max = -half_size, half_size
        y_min, y_max = -half_size, half_size
        
        # Check if points are within bounds
        enclosed_points = (transformed_points[:, 0] >= x_min) & (transformed_points[:, 0] <= x_max) & \
                        (transformed_points[:, 1] >= y_min) & (transformed_points[:, 1] <= y_max)
        
        # Calculate the percentage of points within the bounds
        enclosed_percentage = np.sum(enclosed_points) / len(point_cloud) * 100
        return enclosed_percentage

    camera_view_matrix = np.array([
        [-1.0000000e+00,  0.0000000e+00, -0.0000000e+00,  0.0000000e+00],
        [-0.0000000e+00, -9.9999791e-01,  2.0408120e-03,  0.0000000e+00],
        [ 0.0000000e+00,  2.0408120e-03,  9.9999791e-01,  0.0000000e+00],
        [-0.0000000e+00, -2.0408072e-05, -5.0000101e-01,  1.0000000e+00]
    ])

    def transform_pc_world_to_object_frame(pc, camera_view_matrix, T_camera_to_object=None):
        pc = transform_point_cloud(pc, camera_view_matrix.T)    # Transform world to camera frame
        if T_camera_to_object is None:
            T_camera_to_object = world_to_object_frame_camera_algin(pc)
        pc = transform_point_cloud(pc, T_camera_to_object)  # Transform camera to object frame
        return pc, T_camera_to_object

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
                # visualize_enclosure(get_object_particle_state(gym, sim))
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

                    # init_pc = down_sampling(get_partial_pointcloud_vectorized(*camera_args), 1024)
                    # # T_camera_to_object = world_to_object_frame_camera_algin(init_pc)
                    # init_pc, _ = transform_pc_world_to_object_frame(init_pc, camera_view_matrix)
                    # pcd_init = pcd_ize(init_pc, color=[0,0,0])
                    # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # open3d.visualization.draw_geometries([pcd_init, coor])       

                state = "generate preshape"                
                frame_count = 0    
                
                shift = np.array([0.0, -soft_pose.p.y, camera_args[-3]])    # shift object to centroid 
                pc_rigid = get_partial_pointcloud_vectorized(*rigid_camera_args) + shift


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
                # dof_props_2["damping"][:8].fill(1000.0)
                gym.set_actor_dof_properties(robot_1.env_handle, robot_1.robot_handle, dof_props_2)
                gym.set_actor_dof_properties(robot_2.env_handle, robot_2.robot_handle, dof_props_2)

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 
            print("\n==============================\n")
            print("action_count:", action_count)
            if action_count == 0:
                obj_angle = 0    #np.random.uniform(-np.pi/2, np.pi/2)
            print_color(f"Object angle: {obj_angle}", "green")
            
            # side_angle = 0*-np.pi/2
            if action_count == 0:
                # delta_x_1, delta_y_1, delta_z_1 = (0, 0.00, 0.15)
                # delta_x_2, delta_y_2, delta_z_2 = (0, 0.00, 0.15)
                delta_x_1, delta_y_1, delta_z_1 = (0, 0.00, 0.17)
                delta_x_2, delta_y_2, delta_z_2 = (0, 0.00, 0.17)
            elif action_count == 1:             
                y_displacement = rigid_pose.p.y + two_robot_offset/2
                delta_x_1, delta_y_1, delta_z_1 = (rigid_pose.p.x, y_displacement, 0)
                delta_x_2, delta_y_2, delta_z_2 = (-rigid_pose.p.x, -y_displacement, 0)                
            
            elif action_count == 2:    
                eef1_to_eef2_y = abs(robot_1.get_ee_cartesian_position()[1] - robot_2.get_ee_cartesian_position()[1])
                
                delta_x_1 = 0   #eef1_to_eef2_y/2 * np.sin(obj_angle)
                delta_y_1 = 0   #(eef1_to_eef2_y/2 - eef1_to_eef2_y/2 * np.cos(obj_angle))
                delta_z_1 = 0
                delta_x_2 = 0   #eef1_to_eef2_y/2 * np.sin(obj_angle)
                delta_y_2 = 0   #(eef1_to_eef2_y/2 - eef1_to_eef2_y/2 * np.cos(obj_angle))
                delta_z_2 = 0   

            elif action_count >= 3: 
                current_pc = get_object_particle_state(gym, sim)
                enclosed_percent = check_points_enclosed_percentage(current_pc)
                print_color(f"Enclosed percentage: {enclosed_percent:.2f}", "red")
                # visualize_enclosure(current_pc)

                num_pts = 512  
                init_pc = down_sampling(get_partial_pointcloud_vectorized(*camera_args) + shift, num_pts)
                context_pc = down_sampling(pc_rigid, num_pts)

                init_pc, T_camera_to_object = transform_pc_world_to_object_frame(init_pc, camera_view_matrix, T_camera_to_object=None)
                context_pc, _ = transform_pc_world_to_object_frame(context_pc, camera_view_matrix, T_camera_to_object)

                with torch.no_grad():   
                    if action_count == 3:                                 
                        if goal_model == "diffdef":
                            diffdef_shift = context_pc.mean(axis=0)
                            scale = (context_pc - diffdef_shift).flatten().std()                 
                            init_pc_tensor = torch.from_numpy((init_pc - diffdef_shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3)
                            context_pc_tensor = torch.from_numpy((context_pc - diffdef_shift) / scale).unsqueeze(0).float().to(args.device)  # shape (1, num_pts, 3) 
                            
                            sample_num_points = 1024
                            z = torch.randn([1, ckpt['args'].latent_dim]).to(args.device) * 0.1
                            x = model.sample(z, context_pc_tensor, init_pc_tensor, sample_num_points, flexibility=ckpt['args'].flexibility)
                            
                            first_goal_pc_object_frame = x[0].detach().cpu().numpy()*scale + diffdef_shift
                            
                        elif goal_model == "defgoalnet":
                            diffdef_shift = init_pc.mean(axis=0)
                            scale = (init_pc - diffdef_shift).flatten().std()                 
                            init_pc_tensor = torch.from_numpy((init_pc - diffdef_shift) / scale).unsqueeze(0).permute(0,2,1).float().to(args.device)  # shape (1, 3, num_pts)
                            context_pc_tensor = torch.from_numpy((context_pc - diffdef_shift) / scale).unsqueeze(0).permute(0,2,1).float().to(args.device)  # shape (1, 3, num_pts)
                            first_goal_pc_object_frame = model(init_pc_tensor, context_pc_tensor).permute(0,2,1).squeeze().detach().cpu().numpy()*scale + diffdef_shift      

                        goal_pc_camera_frame = transform_point_cloud(first_goal_pc_object_frame, invert_4x4_transformation_matrix(T_camera_to_object))  # Transform object frame to camera frame                                  

                    # goal_pc_object_frame = first_goal_pc_object_frame 
                    goal_pc_object_frame = transform_point_cloud(goal_pc_camera_frame, T_camera_to_object)  # Transform camera frame to object frame
                    goal_pc_tensor = torch.from_numpy(goal_pc_object_frame).unsqueeze(0).float().permute(0,2,1).to(args.device)  # shape (1, num_pts, 3)

                    pcd_goal = pcd_ize(goal_pc_object_frame, color=[1,0,0])
                    coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # pcd_init = pcd_ize(init_pc, color=[0,0,0])
                    pcd_context = pcd_ize(context_pc, color=[0,1,0])

                    # pcd = pcd_ize(current_pc_numpy, color=[0,0,0])
                    # pcd_goal = pcd_ize(goal_pc_numpy, color=[1,0,0])
                    # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    # open3d.visualization.draw_geometries([pcd, pcd_goal, coor])

                    mani_point_1 = init_pose_1[:3,3] + np.array([0,-two_robot_offset, ROBOT_Z_OFFSET]) + shift
                    mani_point_2 = init_pose_2[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET]) + shift

                    mani_point_1 = transform_pc_world_to_object_frame(mani_point_1.reshape(1, -1), camera_view_matrix, T_camera_to_object)[0].flatten()
                    mani_point_2 = transform_pc_world_to_object_frame(mani_point_2.reshape(1, -1), camera_view_matrix, T_camera_to_object)[0].flatten()


                    current_pc_numpy = init_pc
                    neigh = NearestNeighbors(n_neighbors=50)
                    neigh.fit(current_pc_numpy)
                    
                    _, nearest_idxs_1 = neigh.kneighbors(mani_point_1.reshape(1, -1))
                    mp_channel_1 = np.zeros(current_pc_numpy.shape[0])
                    mp_channel_1[nearest_idxs_1.flatten()] = 1

                    _, nearest_idxs_2 = neigh.kneighbors(mani_point_2.reshape(1, -1))
                    mp_channel_2 = np.zeros(current_pc_numpy.shape[0])
                    mp_channel_2[nearest_idxs_2.flatten()] = 1
                    
                    modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel_1, mp_channel_2])
                    current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(args.device)    


                    pos, rot_mat_1, rot_mat_2 = deformernet(current_pc_tensor, goal_pc_tensor) 
                    # pos, rot_mat_1, rot_mat_2 = deformernet(goal_pc_tensor, current_pc_tensor) 
                    pos *= 0.001
                    pos, rot_mat_1, rot_mat_2 = pos.detach().cpu().numpy(), rot_mat_1.detach().cpu().numpy(), rot_mat_2.detach().cpu().numpy()
                
                pcd = pcd_ize(init_pc, color=[0,0,0])
                colors = np.zeros(init_pc.shape)
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

                coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
                coor_object.transform(T_camera_to_object) 
                open3d.visualization.draw_geometries([pcd, pcd_goal, \
                                                    mani_point_1_sphere, mani_point_2_sphere, coor, coor_object, pcd_context])

                temp1 = np.eye(4)
                temp1[:3,:3] = rot_mat_1 
                print("\n========MODEL OUTPUT=========\n")        
                print("pos:", pos)
                print("\n")

                eef_pose_1_object_frame = compose_4x4_homo_mat(rot_mat_1, pos[0][:3])
                eef_pose_2_object_frame = compose_4x4_homo_mat(rot_mat_2, pos[0][3:])   
                
                modified_world_to_object_H = deepcopy(T_camera_to_object)
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



                tvc_behavior_1 = TaskVelocityControl2([*desired_pos_1, desired_rot_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                    pos_threshold = 2e-3, ori_threshold=5e-2)
                tvc_behavior_2 = TaskVelocityControl2([*desired_pos_2, desired_rot_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                    pos_threshold = 2e-3, ori_threshold=5e-2)
                  
            # else:
            #     rospy.logerr("Exceed max action count")
            #     all_done = True
                 

            if False:   #action_count == 2: 
                delta_alpha_1, delta_beta_1, delta_gamma_1 = 1e-6, 1e-6, obj_angle
                delta_alpha_2, delta_beta_2, delta_gamma_2 = 1e-6, 1e-6, obj_angle
            else:                 
                delta_alpha_1, delta_beta_1, delta_gamma_1 = 1e-6, 1e-6, 1e-6
                delta_alpha_2, delta_beta_2, delta_gamma_2 = 1e-6, 1e-6, 1e-6
                                
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

            if action_count < 3:
                tvc_behavior_1 = TaskVelocityControl2([x_1,y_1,z_1,alpha_1,beta_1,gamma_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits)
                tvc_behavior_2 = TaskVelocityControl2([x_2,y_2,z_2,alpha_2,beta_2,gamma_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits)
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            action_count += 1
            
            
            state = "move to goal"


        if state == "move to goal":
            main_ins_pos_1 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"))
            main_ins_pos_2 = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if main_ins_pos_1 <= 0.042 or main_ins_pos_2 <= 0.042 or (not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts)):  # lose contact w 1 robot
                rospy.logerr("Exeeded joint limits")
                # state = "reset"
                all_done = True

            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if (not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts)):  # lose contact w either robot 2 or robot 1    
                rospy.logerr("Lost contact with robot")
                # state = "reset"
                all_done = True

            if True:

                frame_count += 1           
                
                action_1 = tvc_behavior_1.get_action()  
                action_2 = tvc_behavior_2.get_action() 
                # print("action_1, action_2:", action_1, action_2)
                if (action_1 is not None) and (action_2 is not None) and gym.get_sim_time(sim) - closed_loop_start_time <= 1.5: 
                    # if action_count == 2:
                    #     scaled_action_1 = action_1.get_joint_position()
                    #     scaled_action_2 = action_2.get_joint_position()
                    # else:
                    scaled_action_1 = action_1.get_joint_position() * 4
                    scaled_action_2 = action_2.get_joint_position() * 4
                    gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, scaled_action_1)
                    gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, scaled_action_2)
                    # gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position() * 4)
                    # gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position() * 4)

                else:   
                    rospy.logerr(f"Complete action: {action_count-1}")
                    # enclosed = check_points_enclosed_by_square(get_object_particle_state(gym, sim))                                                             
                    # print("enclosed?:", enclosed)
                    # # visualize_enclosure(get_object_particle_state(gym, sim))
                    
                    if visualization:
                        if action_count == 3:
                            pcd_ize(get_partial_pointcloud_vectorized(*camera_args), vis=True)                                        
                            visualize_camera_views(gym, sim, envs_obj[0], cam_handles, \
                                                resolution=[cam_props.height, cam_props.width], output_file=output_file)   
                        
                    # rospy.loginfo("Succesfully executed moveit arm plan. Let's record point cloud!!")  
                    
                    # if sample_count == 0:
                    
             

                    frame_count = 0
                    state = "get shape servo plan"
                    _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                    init_eulers_1 = transformations.euler_from_matrix(init_pose_1)

                    _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                    init_eulers_2 = transformations.euler_from_matrix(init_pose_2)
                            
                    if action_count >= max_action_count or enclosed_percent >= 99:
                        print_color(f"Final enclosed percentage: {enclosed_percent:.2f}", "green")
                        # data = {"enclosed_percent": enclosed_percent, 
                        #         "obj_name": args.obj_name, 
                        #         "rigid_pose": np.array([rigid_pose.p.x, rigid_pose.p.y, rigid_pose.p.z, rigid_pose.r.w, rigid_pose.r.x, rigid_pose.r.y, rigid_pose.r.z])}
                        # data_path = f"{results_path}/sample_{data_point_count}.pickle"
                        # write_pickle_data(data, data_path)       
                        all_done = True 
                        print_color(f"\n*** Total data point count: {len(os.listdir(results_path))}\n")


        if state == "reset":   
            rospy.logerr("**Current state: " + state)
            frame_count = 0
            sample_count = 0
            terminate_count = 0
            action_count = 0
            reset_count += 1
            
            if len(all_recorded_pcs) == action_length:
                # len(all_recorded_pcs) != action_count means failure due to losing contact or exceeding joint limits.
                assert len(all_recorded_pcs) == len(all_recorded_full_pcs)
                all_recorded_pcs = []
                all_recorded_full_pcs = []   
                                                
            gym.set_actor_rigid_body_states(envs[i], kuka_handles[i], saved_robot_state_1, gymapi.STATE_ALL) 
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_state_2, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_contact_state))

            _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
            init_eulers_1 = transformations.euler_from_matrix(init_pose_1)

            _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
            init_eulers_2 = transformations.euler_from_matrix(init_pose_2)
            
            state = "get shape servo plan"
            

        # if data_point_count >= max_data_point_count:
        #     all_done = True



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
