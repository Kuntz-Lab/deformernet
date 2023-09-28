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
import cv2

import torch
import trimesh
import transformations

from utils.miscellaneous_utils import pcd_ize, read_pickle_data, write_pickle_data, print_color, down_sampling
from utils.camera_utils import get_partial_pointcloud_vectorized, visualize_camera_views
from tissue_wrap_utils.util_functions import compute_intersection_percent, record_eval_data



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 1.0#0.93#0.86



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
    
    return point_cloud.astype('float32')


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
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = vis_cam_width/2
    centerV = vis_cam_height/2    
    # print(centerU - np.array(u_s)*cam_width)
    # y_s = (np.array(u_s)*cam_width).astype(int)
    # x_s = (np.array(v_s)*cam_height).astype(int)
    y_s = (centerU - np.array(u_s)*vis_cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*vis_cam_height).astype(int)    


    return x_s, y_s

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
            {"name": "--eval_sample_idx", "type": int, "default": 10000, "help": "which evaluation sample to use. From 0 to 100"},
            {"name": "--model_name", "type": str, "help": "which model to use. Options: pointconv_100, pointconv_1000, pointconv_10_random_0, etc."}])

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

    eval_data_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/eval_data" 
    eval_data = read_pickle_data(os.path.join(eval_data_path, f"sample {args.eval_sample_idx}.pickle"))
    object_idx = eval_data["object_idx"]
    cylinder_shift = eval_data["cylinder pose"]
    gt_goal_pc = eval_data["partial pcs"][1]
    pcd_gt_goal = pcd_ize(gt_goal_pc, color=[0,0,1])
    print_color(f"cylinder_shift: {cylinder_shift}")


    main_path = "/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/objects_dataset_eval"    # FIX
    cylinder_path = os.path.join(main_path, "cylinder/urdf")
    tissue_path = os.path.join(main_path, "tissue/urdf")
    cylinder_dict_path = os.path.join(main_path, "cylinder/specification")
    tissue_dict_path = os.path.join(main_path, "tissue/specification")
    
    cylinder_dict = read_pickle_data(os.path.join(cylinder_dict_path, f"cylinder_{object_idx}.pickle"))
    tissue_dict = read_pickle_data(os.path.join(tissue_dict_path, f"tissue_{object_idx}.pickle"))
    cylinder_radius, cylinder_length = cylinder_dict["radius"], cylinder_dict["length"]
    tissue_width, tissue_length, tissue_thickness = tissue_dict["width"], tissue_dict["length"], tissue_dict["thickness"]
    print_color(f"\n******** Cylinder length: {cylinder_length}\n")

    asset_root = tissue_path
    soft_asset_file = f"tissue_{object_idx}.urdf"



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
    cylinder_asset_file = f"cylinder_{object_idx}.urdf"   #trimesh.creation.cylinder(radius=0.015, height=0.1)      
    cylinder_pose = gymapi.Transform()
   
    cylinder_pose.p = gymapi.Vec3(cylinder_shift[0], -two_robot_offset/2+cylinder_shift[1], cylinder_shift[2]+0.04) 
    cylinder_pose.r = gymapi.Quat(0.5,0.5,0.5,0.5)
    
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

    # cam_positions.append(gymapi.Vec3(0.17, -0.62-(two_robot_offset/2 - 0.42), 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, 0.40-0.86-(two_robot_offset/2 - 0.42), 0.01)) 

    cam_positions.append(gymapi.Vec3(-0.17, -0.62-(two_robot_offset/2 - 0.42), 0.2))
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


    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 1000
    vis_cam_height = 1000
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height

    # vis_cam_positions.append(gymapi.Vec3(-0.15, -0.4, 0.15)) 
    vis_cam_positions.append(gymapi.Vec3(-0.2, soft_pose.p.y - 0.08, 0.2))
    

    vis_cam_targets.append(gymapi.Vec3(0.0, soft_pose.p.y, 0.00))

    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
            gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])


    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props_2)

        

    '''
    Main stuff is here
    '''
    rospy.init_node('wrap_tissue')
    rospy.logerr("======Loading object ... " + str(args.eval_sample_idx))  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    

    # data_recording_path = f"/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/evaluate/{args.model_name}"
    # os.makedirs(data_recording_path, exist_ok=True)
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    # save_path = os.path.join(data_recording_path, f"sample {args.eval_sample_idx}.pickle")
    max_group_count = 150000
    max_sample_count = 1
    max_data_point_count = 10000
    iter_count = 0
    random_count_2 = 0
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

    max_shapesrv_time = 2*60    # 2 mins
    min_chamfer_dist = 0.2
    fail_mtp = False
    saved_chamfers = []
    segmentationId_dict = {"robot_1": 10, "robot_2": 11, "cylinder": 12}

    dc_client = GraspDataCollectionClient()
    

    # Set up DNN:
    device = torch.device("cuda")
    use_rot = False
    
    if not use_rot:
        sys.path.append('/home/baothach/shape_servo_DNN/bimanual')
        from bimanual_architecture import DeformerNetBimanual   
        model = DeformerNetBimanual(normal_channel=False).to(device)
        weight_path = "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/weights/run1"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 300")))  
        model.eval()
    else:
        sys.path.append(f"/home/baothach/shape_servo_DNN/bimanual")
        from bimanual_architecture import DeformerNetBimanualRot
        model = DeformerNetBimanualRot(normal_channel=False).to(device)
        weight_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_box_5kPa/weights/run2/"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 160")))  
        # weight_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_box_1kPa/weights/run1/"
        # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 200")))  
        model.eval()

    sys.path.append(f"/home/baothach/shape_servo_data/tanner/goal_generation_net")
    from pointconv_model_bao import GoalGenNet
    goal_model = GoalGenNet(num_points=512, embedding_size=256).to(device)
    weight_path = f"/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/weights/{args.model_name}"
    goal_model.load_state_dict(torch.load(f"{weight_path}/epoch 2000"))

    goal_model.eval()


    use_record_goal = True
    visualization = False   #True
    goal_pc_numpy = None

    # Visualization stuff
    prepare_vis_cam = False
    start_vis_cam = False #False
    goal_pc_numpy = None #gt_goal_pc
    record_images = True
    
    vis_frame_count = 0
    num_image = 0
    image_save_path = f"/home/baothach/shape_servo_data/goal_generation/tissue_wrap_multi_objects/evaluate/visualization/{args.model_name}/run4"  
    os.makedirs(image_save_path, exist_ok=True)

    print_color(f"\n******** eval_sample_idx: {args.eval_sample_idx}\n")

    
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
 
        if record_images:
            if prepare_vis_cam:
                radius = 2 #1        
                # Red color in BGR
                color = (0, 0, 255)
                thickness = 2 
                goal_xs, goal_ys = get_goal_projected_on_image(down_sampling(goal_pc_numpy, num_pts=512), i, thickness = 0)
                points = np.column_stack((np.array(goal_ys), np.array(goal_xs)))
                prepare_vis_cam = False

    
            if start_vis_cam:   
                if vis_frame_count % 1 == 0:
                    gym.render_all_camera_sensors(sim)
                    im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))[:,:,:3]
                    # # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)
                    # im[goal_xs, goal_ys, :] = [255,0,0]
                    image = im.astype(np.uint8)


                    # im = Image.fromarray(im)
                    
                    for point in points:
                        image = cv2.circle(image, tuple(point), radius, color, thickness)        

                    path =  os.path.join(image_save_path, f'img{num_image:03}.png')                  
                    cv2.imwrite(path, image)

                    num_image += 1        

                vis_frame_count += 1


        if state == "home" :   
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
                    # first_time = False

                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                    open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                    pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                    pc_ros_msg = fix_object_frame(pc_ros_msg)

                # open3d.visualization.draw_geometries([pcd.paint_uniform_color([0,0,0]), \
                #                                     pcd_goal.paint_uniform_color([1,0,0])])       

                init_full_pc = deepcopy(current_pc)

               
                xs = current_pc[:,0]
                ys = current_pc[:,1]
                zs = current_pc[:,2]
                print("***** dim h, w, thickness:", 
                      (max(xs)-min(xs)), (max(ys)-min(ys)), (max(zs)-min(zs)))  #0.15 0.20



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
                state = "reset"  
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
                
                # start_vis_cam = True

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 


            if move_to_centroid:     
                # desired_position = np.ones(6)
                # desired_pos_1 = np.array([cylinder_shift[0], cylinder_shift[1], cylinder_shift[2]+0.04]) 
                # desired_pos_2 = np.array([-cylinder_shift[0], -cylinder_shift[1], cylinder_shift[2]+0.04])   
                # rospy.logerr("move to centroid")

                # desired_pos_1 = (desired_pos_1 + init_pose_1[:3,3]).flatten()                    
                # desired_pos_2 = (desired_pos_2 + init_pose_2[:3,3]).flatten()
                # eulers = [np.pi/1000000, np.pi/1000000, np.pi/1000000]
                # rot_mat_1 = transformations.euler_matrix(*eulers)[:3,:3]
                # rot_mat_2 = rot_mat_1
                desired_position = np.ones(6)
                robot_1_desired_pos = np.array([cylinder_shift[0], cylinder_shift[1], cylinder_shift[2]+0.04]) 
                robot_2_desired_pos = np.array([-cylinder_shift[0], -cylinder_shift[1], cylinder_shift[2]+0.04])   
                print("======  move to centroid")

                tvc_behavior_1 = TaskVelocityControl(robot_1_desired_pos, robot_1, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 2e-3, second_robot=False)
                tvc_behavior_2 = TaskVelocityControl(robot_2_desired_pos, robot_2, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 2e-3)

       
                
            else:

                
                
                
                rospy.logerr("Use DeformerNet")

                current_pc = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_props, 
                                    segmentationId_dict, object_name="deformable", color=None, min_z=0.01, 
                                    visualization=False, device="cpu")     
                current_pc = down_sampling(current_pc, num_pts=512)

                        
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc)
                
                _, nearest_idxs_1 = neigh.kneighbors(np.array(list(mp_pose_1["pose"][0])).reshape(1, -1))
                mp_channel_1 = np.zeros(current_pc.shape[0])
                mp_channel_1[nearest_idxs_1.flatten()] = 1
                
                _, nearest_idxs_2 = neigh.kneighbors(np.array(list(mp_pose_2["pose"][0])).reshape(1, -1))
                mp_channel_2 = np.zeros(current_pc.shape[0])
                mp_channel_2[nearest_idxs_2.flatten()] = 1 

                modified_pc = np.vstack([current_pc.transpose(1,0), mp_channel_1, mp_channel_2])
                
                
                shift_pc = cylinder_shift #+ np.array([0,0,0.04])
                modified_pc = modified_pc.transpose(1,0)
                modified_pc[:,:3] -= shift_pc
                modified_pc = modified_pc.transpose(1,0)
                goal_pc_tensor = torch.from_numpy(predicted_goal - shift_pc).permute(1,0).unsqueeze(0).float().to(device)   
                # pcd_test = pcd_ize(modified_pc.transpose(1,0)[:,:3], color=[0,0,0])  
                # pcd_test_goal = pcd_ize(goal_pc_tensor.squeeze().permute(1,0).detach().cpu().numpy(), color=[1,0,0])     
                # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)    
                # coor.translate((0,-two_robot_offset/2,0))   
                # open3d.visualization.draw_geometries([pcd_test, pcd_test_goal, coor])
                
                
                
                current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)

            
                pcd.points = open3d.utility.Vector3dVector(current_pc)

                # colors = np.zeros((512,3))
                # colors[nearest_idxs_1.flatten()] = [1,0,0]
                # colors[nearest_idxs_2.flatten()] = [0,1,0]
                # pcd.colors =  open3d.utility.Vector3dVector(colors)
                # mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # mani_point_1_sphere.paint_uniform_color([0,0,1])
                # mani_point_1_sphere.translate(tuple(mp_pose_1["pose"][0]))
                # mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # mani_point_2_sphere.paint_uniform_color([0,1,0])
                # mani_point_2_sphere.translate(tuple(mp_pose_2["pose"][0]))
                # open3d.visualization.draw_geometries([pcd, pcd_goal.paint_uniform_color([1,0,0]), \
                #                                     mani_point_1_sphere, mani_point_2_sphere])  
                # # open3d.visualization.draw_geometries([pcd.paint_uniform_color([0,0,0]), pcd_goal.paint_uniform_color([1,0,0])]) 

                if use_rot:
                    with torch.no_grad():
                        pos, rot_mat_1, rot_mat_2 = model(current_pc_tensor, goal_pc_tensor) 
                        pos *= 0.001
                        pos, rot_mat_1, rot_mat_2 = pos.detach().cpu().numpy(), rot_mat_1.detach().cpu().numpy(), rot_mat_2.detach().cpu().numpy()
                        # desired_position = pos[0]
                        
                        # if iter_count == 0:
                        #     pos[0][2] = max(0.01, pos[0][2])
                        #     pos[0][5] = max(0.01, pos[0][5])
                        # #     iter_count += 1

                        if args.model_name == "pointconv_1000":
                            pos[0][2] = max(0.00, pos[0][2])
                            pos[0][5] = max(0.00, pos[0][5])

                        # pos[0] = np.array([0,tissue_length*0.4,0.05,0,tissue_length*0.4,0.05])  # FIX

                        
                        desired_pos_1 = (pos[0][:3] + init_pose_1[:3,3]).flatten()                    
                        desired_pos_2 = (pos[0][3:] + init_pose_2[:3,3]).flatten()
                        
                        temp1 = np.eye(4)
                        temp1[:3,:3] = rot_mat_1 
                        temp2 = np.eye(4)
                        temp2[:3,:3] = rot_mat_2 
                        print("from model:", pos, transformations.euler_from_matrix(temp1), transformations.euler_from_matrix(temp2))

                        desired_rot_1 = rot_mat_1 @ init_pose_1[:3,:3]
                        desired_rot_2 = rot_mat_2 @ init_pose_2[:3,:3]

                        tvc_behavior_1 = TaskVelocityControl2([*desired_pos_1, desired_rot_1], robot_1, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                            pos_threshold = 2e-3, ori_threshold=5e-2)
                        tvc_behavior_2 = TaskVelocityControl2([*desired_pos_2, desired_rot_2], robot_2, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                                pos_threshold = 2e-3, ori_threshold=5e-2)
                else:
                    if use_record_goal:
                        ### No rot
                        with torch.no_grad():
                            desired_position = model(current_pc_tensor, goal_pc_tensor)[0].cpu().detach().numpy()*(0.001)   
                            # print("=====desired_position:", desired_position) 
                            # iter_count += 1

                        N = 0.02
                        desired_position[0] = min(max(desired_position[0], -N), N)
                        desired_position[3] = min(max(desired_position[3], -N), N)

                    # else:
                    #     rospy.logerr("Use handcoded actions")
                    #     if iter_count == 0:
                    #         desired_position = np.array([0,0.07,0.05,0,0.07,0.05])
                    #         # iter_count += 1
                    #     else:
                    #         desired_position = np.array([0,0.00,0.00,0,0.00,0.00])   
                   
                    robot_1_desired_pos = desired_position[:3]
                    robot_2_desired_pos = desired_position[3:]
             

                    print("from model:", robot_1_desired_pos, robot_2_desired_pos)    

                    tvc_behavior_1 = TaskVelocityControl(robot_1_desired_pos, robot_1, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 2e-3, second_robot=False)
                    tvc_behavior_2 = TaskVelocityControl(robot_2_desired_pos, robot_2, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 2e-3)
                
                iter_count += 1

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            
            state = "move to goal"

        if state == "move to goal":
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(20 in contacts or 21 in contacts) or not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                all_done = True


            else:    
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    
                    state = "reset" 
                    all_done = True
                
                else:
                    action_1 = tvc_behavior_1.get_action()  
                    action_2 = tvc_behavior_2.get_action() 
                    
                    if action_1 is None or action_2 is None or gym.get_sim_time(sim) - closed_loop_start_time >= 1.5:   
                        state = "get shape servo plan"    
                        _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                        _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                        mp_pose_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)[-3])
                        mp_pose_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])  

                        
                        final_full_pc = get_point_cloud()
                        (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)
                        if not move_to_centroid:
                            percent = compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, 
                                                                cylinder_radius, cylinder_length, vis = False)
                            print_color(f"Percent intersect: {percent*100} %")


                        if move_to_centroid:
                            move_to_centroid = False
                            cylinder_pc = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_props,
                                        segmentationId_dict, object_name="cylinder", color=None, min_z=0.01, 
                                        visualization=False, device="cpu")  
                            
                            tissue_pc = get_partial_pointcloud_vectorized(gym, sim, envs[0], cam_handles[0], cam_props, 
                                                segmentationId_dict, object_name="deformable", color=None, min_z=0.01, 
                                                visualization=False, device="cpu")                  
                            
                            pc_tensor = torch.from_numpy(tissue_pc).permute(1,0).unsqueeze(0).float().to(device)
                            cylinder_pc_tensor = torch.from_numpy(cylinder_pc).permute(1,0).unsqueeze(0).float().to(device)
                            goal_pc_tensor = goal_model(pc_tensor, cylinder_pc_tensor)
                            
                            
                            predicted_goal = goal_pc_tensor.squeeze().cpu().detach().numpy().transpose(1,0)
                            pcd_goal = pcd_ize(predicted_goal, color=[1,0,0])
                            pcd_cylinder = pcd_ize(cylinder_pc, color=[0,1,0])   
                            pcd_tissue = pcd_ize(tissue_pc, color=[0,0,0])   
                            
                            if visualization:
                                open3d.visualization.draw_geometries([pcd_goal, pcd_tissue, pcd_cylinder, pcd_gt_goal])     
                                
                            start_vis_cam = True
                            prepare_vis_cam = True
                            goal_pc_numpy = predicted_goal                          


                    else:
                        gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())

                    # Terminal conditions

                    
                    condition_1 = (not move_to_centroid) and all(abs(desired_position) <= 0.006)
                    condition_2 = percent >= 0.95 and iter_count > 3
                    condition_3 = percent >= 0.97
                    condition_4 = iter_count > 7

                    
                    if condition_1 or condition_2 or condition_3 or condition_4:                         
                        final_full_pc = get_point_cloud()
                        (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)
                        
                        final_percent = compute_intersection_percent(final_full_pc, tri_indices, cylinder_shift, 
                                                            cylinder_radius, cylinder_length, vis = visualization)
                        print_color(f"***Final percent intersect: {final_percent*100} %")

                        final_partial_pc = get_tissue_partial_pc_multi_views(vis=False)
                        # record_eval_data(final_partial_pc, final_full_pc, tri_indices, cylinder_shift, final_percent, save_path)
                        all_done = True


                        pcd_partial = pcd_ize(final_partial_pc, color=[0,0,0])
                        
                        if visualization:
                            open3d.visualization.draw_geometries([pcd_partial, pcd_gt_goal])
                        

        # if state == "reset":   
        #     rospy.loginfo("**Current state: " + state)
        #     frame_count = 0
        #     sample_count = 0
        #     terminate_count = 0

            
            
        #     gym.set_actor_rigid_body_states(envs[i], kuka_handles[i], init_robot_state_1, gymapi.STATE_ALL) 
        #     gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state_2, gymapi.STATE_ALL) 
        #     gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))

            
        #     print("Sucessfully reset robot and object")
        #     pc_on_trajectory = []
        #     full_pc_on_trajectory = []
        #     poses_on_trajectory_1 = []  
        #     poses_on_trajectory_2 = [] 
                


        #     state = "home"
 
        
        # if sample_count == max_sample_count:  
        #     sample_count = 0            
        #     group_count += 1
        #     print("group count: ", group_count)
        #     state = "reset" 



        # # if group_count == max_group_count or data_point_count >= max_data_point_count: 
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
