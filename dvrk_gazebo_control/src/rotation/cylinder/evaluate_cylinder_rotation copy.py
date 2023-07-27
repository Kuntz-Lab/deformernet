#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import


import sys

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
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame, get_pykdl_client
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
# from PIL import Image
import transformations
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl, TaskVelocityControl2



sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

import torch



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        davinci_dof_states['pos'][4] = 0.21
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

def down_sampling(pc):
    farthest_indices,_ = farthest_point_sampling(pc, 1024)
    pc = pc[farthest_indices.squeeze()]  
    return pc

# def convert_full_to_partial_pc(full_pc):
#     # proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
#     # fu = 2/proj[0, 0]
#     # fv = 2/proj[1, 1]
    

#     u_s =[]
#     v_s = []
    
#     for point in full_pc:
#         if point[2] < 0.0:
#             continue
#         point = list(point) + [1]

#         point = np.expand_dims(np.array(point), axis=0)

#         point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs_obj[0], cam_handles[0]))
#         # print("point_cam_frame:", point_cam_frame)
#         # image_coordinates = (gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[0]) * point_cam_frame)
#         # print("image_coordinates:",image_coordinates)
#         # u_s.append(image_coordinates[1, 0]/image_coordinates[2, 0]*2)
#         # v_s.append(image_coordinates[0, 0]/image_coordinates[2, 0]*2)
#         # print("fu fv:", fu, fv)
#         u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
#         v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
#     centerU = cam_width/2
#     centerV = cam_height/2    
#     # print(centerU - np.array(u_s)*cam_width)
#     # y_s = (np.array(u_s)*cam_width).astype(int)
#     # x_s = (np.array(v_s)*cam_height).astype(int)
#     y_s = (centerU - np.array(u_s)*cam_width).astype(int)
#     x_s = (centerV + np.array(v_s)*cam_height).astype(int)    

#     # print(x_s, y_s)
#     # valid_idxs = np.array(list(set(np.where(x_s <= 255)[0]) & set(np.where(y_s <= 255)[0]) & \
#     #                     set(np.where(x_s >= 0)[0]) & set(np.where(y_s >= 0)[0])))
#     # y_s = y_s[valid_idxs]      
#     # x_s = x_s[valid_idxs]    
#     sorted_idx = np.argsort(x_s)
#     x_s = x_s[sorted_idx]
#     y_s = y_s[sorted_idx]

#     valid_idxs = []
#     used_coors = []
#     for id, (x, y) in enumerate(zip(x_s, y_s)):
#         if (x,y) not in used_coors:
#             valid_idxs.append(id)
#             used_coors.append((x,y))
#     return full_pc[valid_idxs]
    
#     # y_s = y_s[valid_idxs]      
#     # x_s = x_s[valid_idxs]      
#     # gym.render_all_camera_sensors(sim)

#     # points = []
#     # print("Converting Depth images to point clouds. Have patience...")
#     # # for c in range(len(cam_handles)):
    
#     # # print("Deprojecting from camera %d, %d" % i))
#     # # Retrieve depth and segmentation buffer
#     # depth_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_DEPTH)
#     # seg_buffer = gym.get_camera_image(sim, envs_obj[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)


#     # # Get the camera view matrix and invert it to transform points from camera to world
#     # # space
    
#     # vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs_obj[0], cam_handles[0])))

#     # # Get the camera projection matrix and get the necessary scaling
#     # # coefficients for deprojection
#     # proj = gym.get_camera_proj_matrix(sim, envs_obj[0], cam_handles[0])
#     # fu = 2/proj[0, 0]
#     # fv = 2/proj[1, 1]

#     # # Ignore any points which originate from ground plane or empty space
#     # # depth_buffer[seg_buffer == 11] = -10001

#     # centerU = cam_width/2
#     # centerV = cam_height/2
#     # # for k in y_s:
#     # #     for t in x_s:
#     # for (t,k) in zip(y_s,x_s):
#     #         if depth_buffer[t, k] < -3:
#     #             continue

#     #         u = -(k-centerU)/(cam_width)  # image-space coordinate
#     #         v = (t-centerV)/(cam_height)  # image-space coordinate
#     #         d = depth_buffer[t, k]  # depth buffer value
#     #         X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
#     #         p2 = X2*vinv  # Inverse camera view to get world coordinates
#     #         # print("p2:", p2)
#     #         if p2[0, 2] > 0.01:
#     #             points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

#     # # pcd = open3d.geometry.PointCloud()
#     # # pcd.points = open3d.utility.Vector3dVector(np.array(points))
#     # # open3d.visualization.draw_geometries([pcd]) 

#     # # return points
#     # return np.array(points).astype('float32')
           


    

#     # points = np.zeros((cam_width, cam_height, 3))
#     # for t in range(len(x_s)):
#     #     points[x_s[t], y_s[t]] = goal_pc[t]    
#     # # points[x_s, y_s] = goal_pc
#     # if get_depth_img == False:        

#     #     return points
#     # else:
#     #     img = np.zeros((cam_width, cam_height))
#     #     img[x_s, y_s] = 255
        
#     #     # print(img)
#     #     return points, img

#     # return x_s, y_s

def get_heuristic_goal():
    import trimesh
    # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
    # mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.3*0.5, transform=rot_mat)
    # rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
    # mesh2.apply_transform(rot_mat)


    # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
    # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
    # rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
    # second_mesh2.apply_transform(rot_mat)


    # T = trimesh.transformations.translation_matrix([0, 0, -0.5])
    # second_mesh2.apply_transform(T)
    # rot_mat = transformations.euler_matrix(np.pi/2, 0, 0)
    # second_mesh2.apply_transform(rot_mat)


    # T = trimesh.transformations.translation_matrix([0, -0.714+0.28*np.cos(np.pi/4), -0.15+0.28*np.sin(np.pi/4)])
    # mesh2.apply_transform(T)
    # T = trimesh.transformations.translation_matrix([0, -0.714-0.3*np.cos(np.pi/4), -0.15+0.3*np.sin(np.pi/4)])
    # second_mesh2.apply_transform(T)

    # goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])


    # goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]

    ####################################
    # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
    # mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.25*0.5, transform=rot_mat)



    # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
    # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
    # rot_mat = transformations.euler_matrix(np.pi/4, 0, 0)
    # second_mesh2.apply_transform(rot_mat)


    # T = trimesh.transformations.translation_matrix([0, 0.08, 0.03])
    # second_mesh2.apply_transform(T)

    # goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])
    # T = trimesh.transformations.translation_matrix([0, 0.4-0.90, 0])
    # goal_mesh.apply_transform(T)
    
    # rot_mat = transformations.euler_matrix(0, np.pi/3, 0)
    # # rot_mat[:3,3] = np.array([0.26,0,0])
    # goal_mesh.apply_transform(rot_mat)

    # rot_mat = transformations.euler_matrix(-np.pi/6, 0 , 0)
    # mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.25*0.5, transform=rot_mat)



    # rot_mat = transformations.euler_matrix(np.pi/2, 0 , 0)
    # second_mesh2 = trimesh.creation.cylinder(radius=0.02*0.5, height=0.15*0.5, transform=rot_mat)
    # rot_mat = transformations.euler_matrix(-np.pi/4, 0, 0)
    # second_mesh2.apply_transform(rot_mat)


    # T = trimesh.transformations.translation_matrix([0, 0.05, 0.02])
    # second_mesh2.apply_transform(T)

    # goal_mesh = trimesh.util.concatenate([mesh2, second_mesh2])
    # T = trimesh.transformations.translation_matrix([0, 0.4-0.90, 0])
    # goal_mesh.apply_transform(T)

    # rot_mat = transformations.euler_matrix(0, np.pi/4, np.pi/4)
    # rot_mat[:3,3] = np.array([-0.34,-0.15,0.05])
    # goal_mesh.apply_transform(rot_mat)


    #######################################
    import csv
    # csv file name
    filename = "/home/baothach/Downloads/curve.csv"
    
    # initializing the titles and rows list

    xs = []
    ys = [] 

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        fields = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            # rows.append(row)
            # print(row)
            xs.append(float(str(row[0])))
            ys.append(float(str(row[1])))
            # print(float(str(row[0])))
            # break
    
        # get total number of rows
        # print("Total no. of rows: %d"%(csvreader.line_num))

    # # print(data)
    xs = np.array(xs)
    ys = np.array(ys)
    sorted_idx = np.argsort(xs)
    xs = xs[sorted_idx]
    ys = ys[sorted_idx]

    # degree = 10
    # eqn = np.polyfit(xs, ys, deg=degree)
    # y_prime_s = np.zeros(xs.shape)
    # for i, coe in enumerate(eqn):
    #     y_prime_s += coe*(xs**(degree-i))



    meshes = []
    # ys = y_prime_s
    delta_x = 0.1/xs.shape[0]
    for i in range(xs.shape[0]-1):
        endpoints = np.array([[xs[i],ys[i],0],[xs[i+1],ys[i+1],0]])
        # endpoints = np.array([[xs[i],ys[i],delta_x*i],[xs[i+1],ys[i+1],delta_x*i]])
        # endpoints = np.array([[0,xs[i],ys[i]],[0,xs[i+1],ys[i+1]]])
        # endpoints = np.array([[delta_x*i,xs[i],ys[i]],[delta_x*i,xs[i+1],ys[i+1]]])
        mesh = trimesh.creation.cylinder(radius=0.02*0.5, segment=endpoints)
        meshes.append(mesh)

    goal_mesh = trimesh.util.concatenate(meshes)
    # T = trimesh.transformations.translation_matrix([0.00,soft_pose.p.y-0.1,0.01])
    T = trimesh.transformations.translation_matrix([-0.01,soft_pose.p.y-0.11,0.01])
    # T = trimesh.transformations.translation_matrix([0.00,soft_pose.p.y-0.1,0.0])
    goal_mesh.apply_transform(T)    

    goal_pc = trimesh.sample.sample_surface(goal_mesh, count=1024)[0]    
    

    return goal_pc

if __name__ == "__main__":

    main_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/evaluate"
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
            {"name": "--obj_type", "type": str, "default": 'cylinder_1k', "help": "box1k, box5k, etc."},
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
        data = pickle.load(handle)    
    r = 0.02 #r = data["radius"]
    h = data["height"]

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
    asset_options.thickness = 0.0005#0.0001


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
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(objects_path, "urdf", args.obj_type, "outside") 


    # soft_asset_file = args.obj_name + ".urdf"    
    asset_root = '/home/baothach/sim_data/Custom/Custom_urdf/multi_cylinders/'
    soft_asset_file = 'cylinder_2_attached.urdf'

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, r/2.0)
    # soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, 0.03)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    soft_thickness = 0.0005#0.0005    # important to add some thickness to the soft body to avoid interpenetrations





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

        # add kuka2
        kuka_2_handle = gym.create_actor(env, kuka_asset, pose_2, "kuka2", i, 1, segmentationId=11)        
        

        # add soft obj        
        env_obj = env
        env_obj = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)
        object_handles.append(soft_actor)


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
        cam_pos = gymapi.Vec3(0.5, -0.8, 0.5)
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
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2)) # official
    # cam_positions.append(gymapi.Vec3(0.25, -0.5, 0.3))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
  
    # cam_positions.append(gymapi.Vec3(0.20, -1/2, 0.2))
    # cam_targets.append(gymapi.Vec3(0.0, -1/2, 0.01))    
    
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
    rospy.logerr("======Loading object ... " + str(args.obj_name))  
    rospy.logerr(f"Object type ... {args.obj_type}; inside: {args.inside}") 
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True    


    # Set up DNN:
    device = torch.device("cuda")
    
    
    no_rot = False #False
    if args.obj_type == "cylinder_1k":
        if no_rot:
            sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
            from architecture import DeformerNet            
            model = DeformerNet(normal_channel=False).to(device)
            weight_path = "/home/baothach/shape_servo_data/generalization/multi_cylinders/weights/run1"
            model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 220")))             
        else:
            sys.path.append('/home/baothach/shape_servo_DNN/rotation')
            from architecture_2 import DeformerNetMP as DeformerNet
            model = DeformerNet(normal_channel=False).to(device)
            weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/weights/run1/"
            model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 150")))  

            # weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/weights/run2_single_thin/"
            # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 300")))  


    model.eval()

    goal_recording_path = os.path.join(main_path, "goal_data")
    results_recording_path = os.path.join(main_path, "results_no_rot")
    data_point_count = len(os.listdir(results_recording_path))
    # if args.inside:
    #     goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
    #     chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside")
    # else:
    #     goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 
    #     chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "outside")

    goal_count = 0 #0
    frame_count = 0
    max_goal_count =  goal_count+1  #10

    max_shapesrv_time = 1.5*60    # 2 mins
    if args.inside:
        min_chamfer_dist = 0.1 #0.2
    else:
        min_chamfer_dist = 0.2 #0.25
    fail_mtp = False
    saved_chamfers = []
    final_chamfer_distances = []      

    dc_client = GraspDataCollectionClient()   

   
    # Get 10 goal pc data for 1 object:
    # with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'rb') as handle:
    with open(os.path.join(goal_recording_path, f"cylinder_{data_point_count}.pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])   # first goal pc
    # goal_pc_numpy = get_heuristic_goal()
    # goal_pc_numpy = convert_full_to_partial_pc(get_heuristic_goal())
    # print("shape: ")
    # print(get_heuristic_goal().shape)
    # print(goal_pc_numpy.shape)
    goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)   
    pcd_goal.paint_uniform_color([1,0,0])

    if no_rot:
        goal_position = goal_datas[goal_count]["pos"].T.squeeze()
    else:
        goal_pos = goal_datas[goal_count]["pos"] 
        goal_rot = goal_datas[goal_count]["rot"] 
    saved_obj_contact_state = goal_datas[goal_count]["obj contact state"]
    saved_robot_contact_state = goal_datas[goal_count]["robot contact state"]
    print("=========vel_limits",vel_limits)
    
    start_time = timeit.default_timer()    
    close_viewer = False
    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])

    while (not close_viewer) and (not all_done): 



        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
 

        if state == "home" :   
            frame_count += 1
            # gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka", "psm_main_insertion_joint"), 0.103)
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state)
                

                # if first_time:                    
                #     gym.refresh_particle_state_tensor(sim)
                #     saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                #     init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                #     first_time = False
                
                # state = "get shape servo plan"
                
                desired_position = np.array([0.,0.,0.]) # Set intiial desired gripper position
                # frame_count = 0

                gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
                gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)  
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                
                shapesrv_start_time = timeit.default_timer()

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(get_point_cloud())   
                pcd.paint_uniform_color([0,0,0])
                # open3d.visualization.draw_geometries([pcd, pcd_goal])


                # anchor_pose = deepcopy(init_pose)
                # anchor_eulers = deepcopy(init_eulers)    


            elif frame_count == 6:
                frame_count = 0
                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)
                state = "get shape servo plan"
                # state = "generate preshape"
                initial_pc = get_point_cloud()
                print("====initital_pc.shape", initial_pc.shape)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET-0.01,
                            0, 0.707107, 0.707107, 0]


            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 1)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset" 
                fail_mtp = True               
            else:
                rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                state = "move to preshape"
                # rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))


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
        
                # current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                # print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] )
                anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])

                # switch to velocity mode
                # dof_props_2 = gym.get_asset_dof_properties(kuka_asset)
                dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                # dof_props_2["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
                dof_props_2["stiffness"][:8].fill(0.0)
                dof_props_2["damping"][:8].fill(200.0)
                gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)

                # Reset state
                gym.refresh_particle_state_tensor(sim)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                shapesrv_start_time = timeit.default_timer()

                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pc_numpy = down_sampling(get_partial_point_cloud(i))                  
                            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)  
            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            rospy.logwarn(f"chamfer distance: {chamfer_dist}")
            
            pcd.paint_uniform_color([0,0,0])
            open3d.visualization.draw_geometries([pcd, pcd_goal])

            if no_rot:
                current_pc_tensor = torch.from_numpy(current_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
            else:
                mani_point = init_pose[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET])
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc_numpy)
                _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
                mp_channel = np.zeros(current_pc_numpy.shape[0])
                mp_channel[nearest_idxs.flatten()] = 1
                modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel])
                current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)
                # print("current_pc_tensor.shape, goal_pc_tensor.shape:", current_pc_tensor.shape, goal_pc_tensor.shape)
                mp = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                mp.paint_uniform_color([0,0,1])                
                # open3d.visualization.draw_geometries([pcd, pcd_goal, mp.translate((tuple(mani_point)))])

            if True: #chamfer_dist >= min_chamfer_dist:
                with torch.no_grad():
                    if no_rot:
                        current_pc_tensor = torch.from_numpy(get_point_cloud()).permute(1,0).unsqueeze(0).float().to(device)
                        full_goal_pc_numpy = down_sampling(goal_datas[goal_count]["full pcs"][1])   # first goal pc
                        goal_pc_tensor = torch.from_numpy(full_goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 

                        desired_position = model(current_pc_tensor, goal_pc_tensor)[0].cpu().detach().numpy()*(0.001) 
                        tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits)
                        print("from model:", desired_position)
                        print("ground truth: ", goal_position)   
                    else:
                        pos, rot_mat = model(current_pc_tensor, goal_pc_tensor) 
                        pos *= 0.001
                        pos, rot_mat = pos.detach().cpu().numpy(), rot_mat.detach().cpu().numpy()
                        
                        # pos = np.array([0.07402597, 0.06079869, 0.02437857])
                        rot_mat = transformations.euler_matrix(*[0, 0, np.pi/2])[:3,:3]

                        temp1 = np.eye(4)
                        temp1[:3,:3] = rot_mat
                        temp2 = np.eye(4)
                        temp2[:3,:3] = goal_rot            
                        print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                        print("goal_pos, goal_rot:", goal_pos, transformations.euler_from_matrix(temp2)) 


                        # # pos[0][0] *= 1.5
                        # pos[0][2] = 0.00

                        # if max(abs(pos.squeeze())) >= 0.05:
                        #     pos *= (0.05/max(abs(pos.squeeze())))

                        desired_pos = (pos + init_pose[:3,3]).flatten()
                        desired_rot = rot_mat @ init_pose[:3,:3]



                        tvc_behavior = TaskVelocityControl2([*desired_pos, desired_rot], robot, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                            pos_threshold = 2e-3, ori_threshold=5e-2)


            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))

            state = "move to goal"


        if state == "move to goal":           
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                # final_chamfer_distances.append(999) 
                

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_sampling(get_partial_point_cloud(i)))   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(999+min(saved_chamfers)) 
                print("***final chamfer distance: ", min(saved_chamfers))

                goal_count += 1
            
            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    state = "reset" 
                    current_pc = down_sampling(get_partial_point_cloud(i))
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)   
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                    
                    saved_chamfers.append(chamfer_dist)
                    final_chamfer_distances.append(min(saved_chamfers)) 
                    print("***final chamfer distance: ", min(saved_chamfers))
                    
                    goal_count += 1

                else:
                    action = tvc_behavior.get_action()  
                    if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 1.5:   
                        _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                        init_eulers = transformations.euler_from_matrix(init_pose) 
                        state = "get shape servo plan"    
                        # sys.path.append('/home/baothach/shape_servo_DNN/rotation')
                        # from architecture_2 import DeformerNetMP as DeformerNet
                        # model = DeformerNet(normal_channel=False).to(device)
                        # weight_path = "/home/baothach/shape_servo_data/rotation_extension/multi_cylinders_1000Pa/weights/run2(single_thin)/"
                        # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 140")))  
                        # no_rot = False


                    else:
                        # print("desired vel:", action.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())        

                    # Terminal conditions
                    if no_rot:
                        converge = all(abs(desired_position) <= 0.005)
                    else:
                        converge = all(abs(pos.squeeze()) <= 0.005)
                    if converge or chamfer_dist < min_chamfer_dist:
                        
                        current_pc = down_sampling(get_partial_point_cloud(i))
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(current_pc)  
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        print("***final chamfer distance: ", chamfer_dist)
                        final_chamfer_distances.append(chamfer_dist) 
                        goal_count += 1

                        state = "reset" 



        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            saved_chamfers = []
            
            rospy.logwarn(("=== JUST ENDED goal_count " + str(goal_count)))

            
            # Go to next goal pc
            if goal_count < max_goal_count:
                goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])
                goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                if no_rot:
                    goal_position = goal_datas[goal_count]["pos"].T.squeeze()
                else:
                    goal_pos = goal_datas[goal_count]["pos"] 
                    goal_rot = goal_datas[goal_count]["rot"]         
                saved_obj_contact_state = goal_datas[goal_count]["obj contact state"]
                saved_robot_contact_state = goal_datas[goal_count]["robot contact state"]

            # gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
            # print("Sucessfully reset robot and object")
            # state = "get shape servo plan"

            shapesrv_start_time = timeit.default_timer()
            
            state = "home"


            if fail_mtp:
                state = "home"  
                fail_mtp = False
        

        if  goal_count >= max_goal_count:                    
            all_done = True 
            # final_data = final_chamfer_distances
           
            final_data = {"chamfer": final_chamfer_distances, "final full pc": get_point_cloud(), "goal full pc": goal_datas[0]["full pcs"][1]}           
           
            # with open(os.path.join(results_recording_path, f"cylinder_{data_point_count}.pickle"), 'wb') as handle:
            #     pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


        # step rendering
        gym.step_graphics(sim)
        if not args.headless:
            gym.draw_viewer(viewer, sim, False)
            # gym.sync_frame_time(sim)


    current_pc_numpy = down_sampling(get_partial_point_cloud(i))                  
                    
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)    
    # pcd.paint_uniform_color([0,0,0])
    # pcd_goal.paint_uniform_color([1,0,0])
    # open3d.visualization.draw_geometries([pcd, pcd_goal])      



    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    if not args.headless:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    # print("total data pt count: ", data_point_count)
