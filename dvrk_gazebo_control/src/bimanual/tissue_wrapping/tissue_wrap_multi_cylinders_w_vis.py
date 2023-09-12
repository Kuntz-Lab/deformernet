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
from behaviors import MoveToPose, TaskVelocityControl2


# sys.path.append('/home/baothach/shape_servo_DNN/bimanual')
# # from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
# from bimanual_architecture import DeformerNetBimanual
import torch
import trimesh
import transformations
from wrap_utils import *
import cv2


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
        davinci_dof_states['pos'][4] = 0.15
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        gym.set_actor_dof_states(envs[i], kuka_handles[i], davinci_dof_states, gymapi.STATE_POS)

        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][4] = 0.15
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

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def get_cylinder_goal_pc(radius, height, translation=None, num_points=1024, vis=False):
    
    rot_mat = transformations.euler_matrix(0, np.pi/2, 0)
    
    # # rot_mat = transformations.euler_matrix(np.pi/6, np.pi/2, 0)
    translation = np.array(translation) + cylinder_shift  #np.array([0.03,-0.02,0.04])

    if translation is not None:
        # T = trimesh.transformations.translation_matrix(translation)
        rot_mat[:3,3] = np.array(translation)
    
    # mesh = trimesh.creation.cylinder(radius=radius, height=height, transform=rot_mat)
    mesh = trimesh.creation.annulus(r_min=radius-0.001, r_max=radius, height=height, transform=rot_mat)

    
    sampled_pts = trimesh.sample.sample_surface_even(mesh, count=1024)[0]

    if vis:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(sampled_pts)
        open3d.visualization.draw_geometries([pcd]) 

    return sampled_pts

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
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"


    # chamfer_recording_path =  "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/chamfer_results"
    # data_point_count = len(os.listdir(chamfer_recording_path))
    # rospy.logerr(f"goal count: {data_point_count}")
    # goal_recording_path =  "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/evaluate/goals"
    # with open(os.path.join(goal_recording_path, "sample " + str(11) + ".pickle"), 'rb') as handle:
    # # with open(os.path.join(goal_recording_path, "sample " + str(data_point_count) + ".pickle"), 'rb') as handle:    
    #     data = pickle.load(handle)
    #     args.obj_name = data["obj_name"]

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
        sim_params.flex.num_outer_iterations = 4
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

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        mesh_data = pickle.load(handle)    
    h = mesh_data["height"]
    w = mesh_data["width"]
    thickness = mesh_data["thickness"]

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

    # if args.inside:
    #     asset_root = os.path.join(objects_path, "bimanual_urdf", args.obj_type, "inside")
    # else:
    #     asset_root = os.path.join(objects_path, "bimanual_urdf", args.obj_type, "outside") 


    # soft_asset_file = args.obj_name + ".urdf"

    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf"
    soft_asset_file = 'thin_tissue_layer.urdf'



    soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, 0.01818)
    soft_pose.p = gymapi.Vec3(0.0, -two_robot_offset/2, 0.01818)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.001#0.001   # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True

    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
    
    cylinder_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/multi_cylinders"
    cylinder_asset_file = "cylinder_wrap_tissue.urdf"        
    cylinder_pose = gymapi.Transform()
    # cylinder_pose.p = gymapi.Vec3(0.01, -two_robot_offset/2+0.01, 0.04)
    # cylinder_pose.p = gymapi.Vec3(0.00, -two_robot_offset/2+0.00, 0.04)


    # cylinder_shift = np.array([0.03,-0.05,0.05])  #np.array([0.06,-0.02,0.03])
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
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=11)

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i+3, 1, segmentationId=11)        
        

        # add soft obj    
        
        
            
        # env_obj = env
        env_obj = env # gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)    
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)         
        
        object_handles.append(soft_actor)

        temp_env = env # gym.create_env(sim, env_lower, env_upper, num_per_row)
        cylinder_actor = gym.create_actor(temp_env, cylinder_asset, cylinder_pose, "cylinder", i+2, 1, segmentationId=11)
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
    # cam_positions.append(gymapi.Vec3(0.12, -0.55, 0.15))
    # cam_positions.append(gymapi.Vec3(0.1, -0.5-(two_robot_offset/2 - 0.42), 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, -0.45-(two_robot_offset/2 - 0.42), 0.00))

    # cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
    cam_positions.append(gymapi.Vec3(0.17, -0.62-(two_robot_offset/2 - 0.42), 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-0.86-(two_robot_offset/2 - 0.42), 0.01)) 
  

    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])
    print("============cam_handle:", cam_handles[0])


    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 1000
    vis_cam_height = 1000
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height

    # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.2, 0.2))  # sample 4
    # vis_cam_positions.append(gymapi.Vec3(0.1, -0.3, 0.1)) # worst
    # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.3, 0.1)) # 2nd worst, used for all cases
    vis_cam_positions.append(gymapi.Vec3(-0.15, -0.4, 0.15)) # 75 percentile, index 59, IMSR

    vis_cam_targets.append(gymapi.Vec3(0.0, -two_robot_offset/2, 0.00))

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
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    

    # goal_recording_path = "/home/baothach/shape_servo_data/comparison/RRT/goal_data"
    # goal_point_count = 0

    data_recording_path = "/home/baothach/shape_servo_data/tissue_wrap/evaluate/run1"
    os.makedirs(data_recording_path, exist_ok=True)
    terminate_count = 0
    sample_count = 0
    frame_count = 0
    group_count = 0
    data_point_count = len(os.listdir(data_recording_path))
    save_path = os.path.join(data_recording_path, f"sample {data_point_count}.pickle")
    max_group_count = 150000
    max_sample_count = 1
    max_data_point_count = 10000
    iter_count = 0
    random_count_2 = 0
    percent = 0.0
    # min_chamfer_dist = 0.2



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
    
    # Visualization stuff
    prepare_vis_cam = True
    start_vis_cam = False #False
    vis_frame_count = 0
    num_image = 0
    image_save_path = "/home/baothach/shape_servo_data/tissue_wrap/visualization/3"  
    os.makedirs(image_save_path, exist_ok=True)

    dc_client = GraspDataCollectionClient()
    
    # Set up DNN:
    device = torch.device("cuda")
    # model = DeformerNetBimanual(normal_channel=False).to(device)
    # weight_path = "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/weights/run1"
    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 300")))  

    sys.path.append(f"/home/baothach/shape_servo_DNN/bimanual")
    from bimanual_architecture import DeformerNetBimanualRot
    model = DeformerNetBimanualRot(normal_channel=False).to(device)
    weight_path = "/home/baothach/shape_servo_data/rotation_extension/bimanual/multi_box_1kPa/weights/run1/"
    model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 200")))  

    model.eval()


    goal_pc_numpy = get_cylinder_goal_pc(radius=(0.2*0.7)/(2*np.pi), height=0.15*0.7, translation=[0,-two_robot_offset/2, 0.9*0.04]) # circumfrence = tissue length


    

    goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
    goal_position = [0,0,0]
 

    

    
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
            radius = 1 #1        
            # Red color in BGR
            color = (0, 0, 255)
            thickness = 2 
            goal_xs, goal_ys = get_goal_projected_on_image(down_sampling(goal_pc_numpy, num_pts=512), i, thickness = 0)
            points = np.column_stack((np.array(goal_ys), np.array(goal_xs)))
            prepare_vis_cam = False

 
        if start_vis_cam:   
            if vis_frame_count % 2 == 0:
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
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.203)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
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

                zs = current_pc[:,2]
                # z_idxs = np.where(abs(zs) <= 0.001)[0]
                z_idxs = np.where(abs(zs) >= 0.01)[0]
                xs = current_pc[:,0]
                x_idxs = np.where(abs(xs) <= 0.002)[0]
                backbone_idxs = np.array(list(set(z_idxs) & set(x_idxs)))
                # sampled_idx = np.random.randint(low = 0, high=backbone_idxs.shape, size= goal_backbones[1].shape[0])               
                backbone = current_pc[backbone_idxs]

                sorted_idx = np.argsort(backbone[:,1])
                backbone_idxs = deepcopy(backbone_idxs[sorted_idx])


                # backbone_pcd = open3d.geometry.PointCloud()
                # backbone_pcd.points = open3d.utility.Vector3dVector(backbone)
                # backbone_pcd.paint_uniform_color([1,0,0])
                # open3d.visualization.draw_geometries([pcd.paint_uniform_color([0,0,0]), backbone_pcd])  


                zs = current_pc[:,2]
                z_idxs = np.where(abs(zs) <= 0.001)[0]
                ys = current_pc[:,1]
                y_idxs = np.where(abs(ys-min(ys)) <= 0.002)[0]
                edges_idxs_2 = np.array(list(set(z_idxs) & set(y_idxs)))
                y_idxs = np.where(abs(ys-max(ys)) <= 0.002)[0]
                edges_idxs_1 = np.array(list(set(z_idxs) & set(y_idxs)))

                # print(edges_idxs.shape)

                # edge_pcd = open3d.geometry.PointCloud()
                # edge_pcd.points = open3d.utility.Vector3dVector(current_pc[edges_idxs])
                # edge_pcd.paint_uniform_color([1,0,0])
                # open3d.visualization.draw_geometries([pcd.paint_uniform_color([0,0,0]), edge_pcd])  

                state = "generate preshape"                
                frame_count = 0              
                
                # pcd.points = open3d.utility.Vector3dVector(get_partial_point_cloud(i))
                # open3d.visualization.draw_geometries([pcd])

        # random_count_2 += 1
        # if random_count_2 % 10 == 0:
        #     full_pc = get_point_cloud()     
        #     print("distance 2 edges:", np.mean(full_pc[edges_idxs_1], axis=0)-np.mean(full_pc[edges_idxs_2], axis=0))       


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
                
                start_vis_cam = True

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state) 


            if move_to_centroid:     
                desired_position = np.ones(6)
                desired_pos_1 = np.array([cylinder_shift[0], cylinder_shift[1], cylinder_shift[2]+0.04]) 
                desired_pos_2 = np.array([-cylinder_shift[0], -cylinder_shift[1], cylinder_shift[2]+0.04])   
                rospy.logerr("move to centroid")

                desired_pos_1 = (desired_pos_1 + init_pose_1[:3,3]).flatten()                    
                desired_pos_2 = (desired_pos_2 + init_pose_2[:3,3]).flatten()
                eulers = [np.pi/1000000, np.pi/1000000, np.pi/1000000]
                rot_mat_1 = transformations.euler_matrix(*eulers)[:3,:3]
                rot_mat_2 = rot_mat_1

            else:

                rospy.logerr("Use DeformerNet")
                # current_pc = down_sampling(get_point_cloud()) #down_sampling(get_partial_point_cloud(i))
                current_pc = down_sampling(get_partial_point_cloud(i))
          
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc)
                
                _, nearest_idxs_1 = neigh.kneighbors(np.array(list(mp_pose_1["pose"][0])).reshape(1, -1))
                mp_channel_1 = np.zeros(current_pc.shape[0])
                mp_channel_1[nearest_idxs_1.flatten()] = 1
                
                _, nearest_idxs_2 = neigh.kneighbors(np.array(list(mp_pose_2["pose"][0])).reshape(1, -1))
                mp_channel_2 = np.zeros(current_pc.shape[0])
                mp_channel_2[nearest_idxs_2.flatten()] = 1 

                modified_pc = np.vstack([current_pc.transpose(1,0), mp_channel_1, mp_channel_2])
                current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)
                # assert modified_pc.shape == (5,1024)
            
                # pcd.points = open3d.utility.Vector3dVector(current_pc)
 
                # colors = np.zeros((1024,3))
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

                with torch.no_grad():
                    pos, rot_mat_1, rot_mat_2 = model(current_pc_tensor, goal_pc_tensor) 
                    pos *= 0.001
                    pos, rot_mat_1, rot_mat_2 = pos.detach().cpu().numpy(), rot_mat_1.detach().cpu().numpy(), rot_mat_2.detach().cpu().numpy()
                    
                    if iter_count == 0:
                    #    pos[0][np.array([0,3])] += 0.02
                        # pos[0][np.array([2,5])] += 0.04
                        # pos[0][np.array([1,4])] += 0.02
                        pos[0][np.array([2,5])] = 0.04
                        pos[0][1] = 0.05
                        pos[0][4] = 0.05
                    else:
                        pos[0][np.array([1,4])] = 0.02
                        pos[0][np.array([2,5])] = 0
                    iter_count += 1
                    pos[0][np.array([0,3])] = 0
                    # pos[0][np.array([2,5])] = 0.04
                    
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
            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            
            state = "move to goal"

        if state == "move to goal":
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                all_done = True


            else:    
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    
                    final_full_pc = get_point_cloud()
                    # record_data(edges_idxs_1, edges_idxs_2, backbone_idxs, init_full_pc, final_full_pc, cylinder_shift, save_path)
                    # all_done = True

                
                else:
                    action_1 = tvc_behavior_1.get_action()  
                    action_2 = tvc_behavior_2.get_action() 
                    
                    if action_1 is None or action_2 is None or gym.get_sim_time(sim) - closed_loop_start_time >= 1.5:   
                        state = "get shape servo plan"    
                        _,init_pose_1 = get_pykdl_client(robot_1.get_arm_joint_positions())
                        _,init_pose_2 = get_pykdl_client(robot_2.get_arm_joint_positions())
                        mp_pose_1 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)[-3])
                        mp_pose_2 = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])  

                        

                        percent = compute_intersection_percent(backbone=get_point_cloud()[backbone_idxs], cylinder_shift=cylinder_shift, vis = False)
                        print(f"Percent intersect: {percent*100} %")


                        if move_to_centroid:
                            move_to_centroid = False


                    else:
                        gym.set_actor_dof_velocity_targets(robot_1.env_handle, robot_1.robot_handle, action_1.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot_2.env_handle, robot_2.robot_handle, action_2.get_joint_position())

                    # Terminal conditions

                    
                    condition_1 = (not move_to_centroid) and all(abs(desired_position) <= 0.005)
                    condition_2 = percent >= 0.95 and iter_count > 3
                    condition_3 = percent >= 0.99

                    
                    if condition_1 or condition_2 or condition_3:                         
                        final_full_pc = get_point_cloud()
                        # record_data(edges_idxs_1, edges_idxs_2, backbone_idxs, init_full_pc, final_full_pc, cylinder_shift, save_path)
                        # all_done = True
                        full_pc = get_point_cloud()     
                        print("distance 2 edges:", np.mean(full_pc[edges_idxs_1], axis=0)-np.mean(full_pc[edges_idxs_2], axis=0))      

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
            # gym.sync_frame_time(sim)


  


    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(get_point_cloud())
    # open3d.io.write_point_cloud("/home/baothach/Downloads/wrap_tissue_mesh.ply", pcd)

    # data = get_point_cloud()[backbone_idxs]
    # with open("/home/baothach/Downloads/wrap_tissue_backbone.pickle", 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)  

    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    # print("total data pt count: ", data_point_count)