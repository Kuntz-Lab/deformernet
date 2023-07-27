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
from utils.isaac_utils import fix_object_frame
# from utils.record_data_h5 import RecordGraspData_sparse
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from core import Robot
from behaviors import MoveToPose, TaskVelocityControl


### KeyNet
# sys.path.remove("/home/baothach/shape_servo_DNN/learn_mp")
# sys.path.remove("/home/baothach/shape_servo_DNN/generalization_tasks")
# sys.path.remove("/home/baothach/shape_servo_DNN/")
sys.path.append("/home/baothach/keypoints")
from keypoints.models import keynet
from config import config
from point_cloud_utils import *

sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
from architecture import DeformerNet

sys.path.append('/home/baothach/shape_servo_DNN')
from farthest_point_sampling import *

# sys.path.append('/home/baothach/shape_servo_DNN/learn_mp')
# from architecture_seg import ManiPointSegment
# from architecture_classifier import ManiPointNet2
import cv2



import torch



ROBOT_Z_OFFSET = 0.25
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
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

def down_sampling(pc, num_points=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_points)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def convert_unordered_to_ordered_pc(goal_pc, i, get_depth_img = False):
    # proj = gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[i])
    # fu = 2/proj[0, 0]
    # fv = 2/proj[1, 1]
    

    u_s =[]
    v_s = []

    for point in goal_pc:
        point = list(point) + [1]

        point = np.expand_dims(np.array(point), axis=0)

        point_cam_frame = point * np.matrix(gym.get_camera_view_matrix(sim, envs_obj[i], cam_handles[0]))
        # print("point_cam_frame:", point_cam_frame)
        # image_coordinates = (gym.get_camera_proj_matrix(sim, envs_obj[i], cam_handles[0]) * point_cam_frame)
        # print("image_coordinates:",image_coordinates)
        # u_s.append(image_coordinates[1, 0]/image_coordinates[2, 0]*2)
        # v_s.append(image_coordinates[0, 0]/image_coordinates[2, 0]*2)
        # print("fu fv:", fu, fv)
        u_s.append(1/2 * point_cam_frame[0, 0]/point_cam_frame[0, 2])
        v_s.append(1/2 * point_cam_frame[0, 1]/point_cam_frame[0, 2])      
          
    centerU = cam_width/2
    centerV = cam_height/2    
    # print(centerU - np.array(u_s)*cam_width)
    # y_s = (np.array(u_s)*cam_width).astype(int)
    # x_s = (np.array(v_s)*cam_height).astype(int)
    y_s = (centerU - np.array(u_s)*cam_width).astype(int)
    x_s = (centerV + np.array(v_s)*cam_height).astype(int)    
    
    points = np.zeros((cam_width, cam_height, 3))
    for t in range(len(x_s)):
        points[x_s[t], y_s[t]] = goal_pc[t]    
    # points[x_s, y_s] = goal_pc
    if get_depth_img == False:        

        return points
    else:
        img = np.zeros((cam_width, cam_height))
        img[x_s, y_s] = 255
        
        # print(img)
        return points, img

    return x_s, y_s

def get_mp_classifer(classifer_model, partial_init_pc, partial_goal_pc, full_init_pc, 
                    num_candidates=400, batch_size=64, vis=False):
    pc_goal_tensor = torch.from_numpy(partial_goal_pc).permute(1,0).unsqueeze(0).float().to(device)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(partial_init_pc))
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(np.array(partial_goal_pc))    
    
    full_pcd = open3d.geometry.PointCloud()
    full_pcd.points = open3d.utility.Vector3dVector(np.array(full_init_pc))
    obb = full_pcd.get_oriented_bounding_box()
    center = full_pcd.get_center()    

    # Get width, height, depth
    points = np.asarray(obb.get_box_points())
    x_axis = (points[1]-points[0])
    y_axis = (points[2]-points[0])
    z_axis = (points[3]-points[0])
    width = np.linalg.norm(x_axis)  # Length of x axis (https://www.cs.utah.edu/gdc/projects/alpha1/help/man/html/shape_edit/primitives.html)
    height = np.linalg.norm(y_axis)
    # depth = np.linalg.norm(z_axis)
    if 0.9 * height <= width <= 1.1 * height:
        x_axis, y_axis = y_axis, x_axis                

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # z_axis /= np.linalg.norm(z_axis)    

    m = y_axis[1] / y_axis[0] 
    if abs(m) >= 1:
        m = y_axis[0] / y_axis[1]
    b = center[1] - m * center[0]

    mp_candidates_idxs = np.where(m*partial_init_pc[:,0] + b - partial_init_pc[:,1] <= 0)[0]
    mp_candidates_idxs = np.random.choice(mp_candidates_idxs, size = num_candidates)    # sub-sample 32 candidates
    mp_candidates = partial_init_pc[mp_candidates_idxs]  # points on the correct half

    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(partial_init_pc)
    _, nearest_idxs = neigh.kneighbors(mp_candidates)
    mp_channel = np.zeros((mp_candidates.shape[0], partial_init_pc.shape[0]))
    mp_channel[np.array([i // 50 for i in range(mp_candidates.shape[0]*50)]),nearest_idxs.flatten()] = 1

    pcs_tensor = torch.from_numpy(partial_init_pc).permute(1,0).unsqueeze(0).repeat(mp_candidates.shape[0],1,1)
    modified_pc_tensor = torch.cat((pcs_tensor, torch.from_numpy(mp_channel).unsqueeze(1)), dim=1).float().to(device)

    pcs_goal_tensor = pc_goal_tensor.repeat(mp_candidates.shape[0],1,1)
    # print("pcs_goal_tensor.shape, pcs_tensor.shape:", pcs_goal_tensor.shape, pcs_tensor.shape)

    with torch.no_grad():
        outputs = []
        for batch_pc, batch_pc_goal in zip(torch.split(modified_pc_tensor, num_candidates//batch_size), torch.split(pcs_goal_tensor, num_candidates//batch_size)):
            outputs.append(classifer_model(batch_pc, batch_pc_goal))

        output = torch.cat(tuple(outputs), dim=0)
        success_probs = np.exp(output.cpu().detach().numpy())[:,1]
        success_probs = (success_probs/max(success_probs))
        best_mp = mp_candidates[np.argmax(success_probs)]

        if vis:
            heats = np.array([[prob, 0, 0] for prob in success_probs])
            best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            best_mani_point.paint_uniform_color([0,1,0])  

            colors = np.zeros(partial_init_pc.shape)
            colors[mp_candidates_idxs] = heats  
            pcd.colors =  open3d.utility.Vector3dVector(colors) 
            
            mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point.paint_uniform_color([0,0,1])
           
            open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), best_mani_point.translate(tuple(best_mp)), \
                                                    pcd_goal.translate((0.2,0,0))])  
        

    return best_mp

def get_mp_keypoint(partial_init_pc, partial_goal_pc, vis=False):
    source_pc = partial_init_pc
    target_pc = partial_goal_pc
    with torch.no_grad():
        source_pc_tensor = torch.tensor(source_pc).permute(2,0,1).unsqueeze(0).float().to(device)
        target_pc_tensor = torch.tensor(target_pc).permute(2,0,1).unsqueeze(0).float().to(device)     

        _, _, k1, _, _, _ = kp_network(source_pc_tensor, source_pc_tensor)
        _, _, k2, _, _, _ = kp_network(target_pc_tensor, target_pc_tensor)

        k1 = k1.squeeze().detach().cpu().numpy()
        k2 = k2.squeeze().detach().cpu().numpy()


        heatmap, mani_point, mani_point_xyz = get_kp_pc_heatmap(k1, k2, source_pc, target_pc, get_mani_point = True, thickness = 1, N=20, rm_outliers=False)
        gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        gt_mani_point.paint_uniform_color([0,0,1])
        gt_mani_point.translate(tuple(gt_mp))

        if vis:
            open3d.visualization.draw_geometries([heatmap, mani_point, gt_mani_point])    
        return mani_point_xyz

def get_goal_projected_on_image(goal_pc, i, thickness = 0): 

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

    main_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success"

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
    goal_count = 0
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
        object_meshes_path = os.path.join(main_path, "meshes", args.obj_type, "inside")
    else:
        object_meshes_path = os.path.join(main_path, "meshes", args.obj_type, "outside") 

    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)    
    h = data["height"]
    w = data["width"]
    thickness = data["thickness"]

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
    asset_options.thickness = 0.0001#0.0001


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
        asset_root = os.path.join(main_path, "urdf", args.obj_type, "inside")
    else:
        asset_root = os.path.join(main_path, "urdf", args.obj_type, "outside") 


    soft_asset_file = args.obj_name + ".urdf"    


    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.7)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
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

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
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
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
  

    
    for i, env_obj in enumerate(envs_obj):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])

    vis_cam_positions = []
    vis_cam_targets = []
    vis_cam_handles = []
    vis_cam_width = 800
    vis_cam_height = 800
    vis_cam_props = gymapi.CameraProperties()
    vis_cam_props.width = vis_cam_width
    vis_cam_props.height = vis_cam_height

    vis_cam_positions.append(gymapi.Vec3(0.15, -0.35, 0.1))  # sample 4
    # vis_cam_positions.append(gymapi.Vec3(0.1, -0.3, 0.1)) # worst
    # vis_cam_positions.append(gymapi.Vec3(-0.1, -0.3, 0.1)) # 2nd worst
    # vis_cam_positions.append(gymapi.Vec3(-0.2, -0.2, 0.05))  # sample 2
    # vis_cam_positions.append(gymapi.Vec3(-0.2, -two_robot_offset/2-0.03, 0.15)) # sample 0
    # vis_cam_positions.append(gymapi.Vec3(-0.00, -0.27, 0.05))   # importance of MP
    # vis_cam_positions.append(gymapi.Vec3(0.05, -0.27, 0.1))
    vis_cam_targets.append(gymapi.Vec3(0.0, -0.42, 0.00))

    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
            gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])

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
    model = DeformerNet(normal_channel=False).to(device)
    
    if args.obj_type == "box_1k":
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_1000Pa/weights/run2(partial)"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))   
    elif args.obj_type == "box_5k":
        # weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run1"
        # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 250")))  
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_5kPa/weights/run3(partial)"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 240")))              
    elif args.obj_type == "box_10k":
        weight_path = "/home/baothach/shape_servo_data/generalization/multi_boxes_10kPa/weights/run1"
        model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 288"))) 

    model.eval()

    args_kp = config(use_gym = True, config = "/home/baothach/keypoints/configs/keypoints_celeba.yaml/")
    args_kp.load = "/home/baothach/keypoints/data/models/keypoints/F/run_9/best/"
    kp_network = keynet.make(args_kp).to(device)
    kp_network.eval()



    # # Set up manipulation point model
    # mp_seg_model = ManiPointSegment(num_classes=2).to(device)
    # weight_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/weights/seg/run1"
    # mp_seg_model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 150")))    
    # mp_seg_model.eval()

    # mp_classifier_model = ManiPointNet2(normal_channel=False).to(device)
    # weight_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/weights/classifer/run1"
    # mp_classifier_model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 300")))    
    # mp_classifier_model.eval()        

    if args.inside:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "inside")
        chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "inside")
    else:
        goal_recording_path = os.path.join(main_path, "goal_data", args.obj_type, "outside") 
        chamfer_recording_path = os.path.join(main_path, "chamfer_results", args.obj_type, "outside")

    # goal_count = 5#0 
    frame_count = 0
    max_goal_count =  10#10

    max_shapesrv_time = 2*60    # 2 mins
    if args.inside:
        min_chamfer_dist = 0.2
    else:
        min_chamfer_dist = 0.2 #0.25
    fail_mtp = False
    saved_chamfers = []
    final_chamfer_distances = []      

    dc_client = GraspDataCollectionClient()   

   
    # Get 10 goal pc data for 1 object:
    goal_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success/goal_data/box_1k/test_fail_cases/goals"
    with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])  # first goal pc
    goal_pc = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
    goal_position = goal_datas[goal_count]["positions"]    
    ordered_goal_pc_numpy = convert_unordered_to_ordered_pc(goal_datas[goal_count]["partial pcs"][1], i)
    pcd_goal.paint_uniform_color([1,0,0])

    init_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][0])  # first goal pc
    init_pc = torch.from_numpy(init_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)  

    full_pc_numpy = goal_datas[goal_count]["full pcs"][0]  
    
    gt_mp = np.array(list(goal_datas[goal_count]["mani_point"][0]))
    saved_object_state = goal_datas[goal_count]["init object state"]
    saved_frame_state = goal_datas[goal_count]["init frame state"]


    # Visualization stuff
    prepare_vis_cam = False#True
    start_vis_cam = False
    vis_frame_count = 0
    num_image = 0
    save_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/evaluate_success/goal_data/box_1k/test_fail_cases/visualization/1"       
    
    start_time = timeit.default_timer()    
    close_viewer = False
    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])

    while (not close_viewer) and (not all_done): 



        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if prepare_vis_cam:
            radius = 1 #1        
            # Red color in BGR
            color = (0, 0, 255)  #(200, 0, 0) #(0, 0, 255)
            thickness = 2 
            goal_xs, goal_ys = get_goal_projected_on_image(down_sampling(goal_datas[goal_count]["full pcs"][1], num_points=800), i, thickness = 0)
            points = np.column_stack((np.array(goal_ys), np.array(goal_xs)))
            prepare_vis_cam = False

 
        if start_vis_cam:   
            if vis_frame_count % 5 == 0:
                gym.render_all_camera_sensors(sim)
                im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))[:,:,:3]
                # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)
                # im[goal_xs, goal_ys, :] = [255,0,0]
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
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.203)            
            
            if frame_count == 1:
                gym.set_actor_rigid_body_states(envs_obj[i], object_handles[i], saved_frame_state, gymapi.STATE_ALL)
                gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state)
                

                if first_time:                    
                    # gym.refresh_particle_state_tensor(sim)
                    # saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    first_time = False

                # gym.set_actor_rigid_body_states(envs_obj[i], object_handles[i], saved_frame_state, gymapi.STATE_ALL)
                # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))

                state = "generate preshape"
                
                desired_position = np.array([0.,0.,0.]) # Set intiial desired gripper position
                frame_count = 0

                initial_pc = get_point_cloud()
                # print("====initital_pc.shape", initial_pc.shape)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(initial_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)
                saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 


        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            # preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            # cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            with torch.no_grad():
                # ### Seg
                # output = mp_seg_model(init_pc, goal_pc)
                # success_probs = np.exp(output.squeeze().cpu().detach().numpy())[1,:]
                # best_mp = gt_mp #init_pc_numpy[np.argmax(success_probs)]
            
                #-- Visualization:
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(np.array(init_pc_numpy))
                
                # best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # best_mani_point.paint_uniform_color([0,1,0])  
                # best_mani_point.translate(tuple(best_mp))
                
                # gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # gt_mani_point.paint_uniform_color([0,0,1])
                # gt_mani_point.translate(tuple(gt_mp))

                # heats = np.array([[prob, 0, 0] for prob in success_probs/max(success_probs)])
                # pcd.colors =  open3d.utility.Vector3dVector(heats) 

                # open3d.visualization.draw_geometries([pcd, best_mani_point, deepcopy(pcd_goal).translate((0.2,0,0)), gt_mani_point])
                
                ### Classifer:
                # best_mp = get_mp_classifer(mp_classifier_model, init_pc_numpy, goal_pc_numpy, full_pc_numpy, vis=False)
                # open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.2,0,0))])
                
                ordered_pc_numpy = convert_unordered_to_ordered_pc(get_partial_point_cloud(i), i)
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(ordered_pc_numpy.reshape(-1,3))
                # pcd_goal = open3d.geometry.PointCloud()
                # pcd_goal.points = open3d.utility.Vector3dVector(ordered_goal_pc_numpy.reshape(-1,3))
                # open3d.visualization.draw_geometries([pcd, deepcopy(pcd_goal).translate((0.2,0,0))])
                
                # data_recording_path = "/home/baothach/shape_servo_data/manipulation_points/multi_boxes_1000Pa/heatmaps/ordered_goal_data"
                # with open(os.path.join(data_recording_path, f"goal {0}.pickle"), 'rb') as handle:
                #     data = pickle.load(handle)                         
                # source_pc_ori = data["partial pcs"][0].reshape(-1,3)
                # target_pc_ori = data["partial pcs"][1].reshape(-1,3)
                # source_pc = convert_unordered_to_ordered_pc(source_pc_ori, i)
                # target_pc = convert_unordered_to_ordered_pc(target_pc_ori, i)
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(target_pc_ori)                
                # pcd2 = open3d.geometry.PointCloud()
                # pcd2.points = open3d.utility.Vector3dVector(target_pc.reshape(-1,3))   
                # open3d.visualization.draw_geometries([pcd, deepcopy(pcd2).translate((0.2,0,0))])             
                # best_mp = get_mp_keypoint(source_pc, target_pc, vis=True)

                best_mp = get_mp_keypoint(ordered_pc_numpy, ordered_goal_pc_numpy, vis=True)


            # target_pose = [-best_mp[0], -best_mp[1], best_mp[2] - ROBOT_Z_OFFSET, 0, 0.707107, 0.707107, 0]
            target_pose = [-best_mp[0], -best_mp[1], best_mp[2] - ROBOT_Z_OFFSET-0.02, 0, 0.707107, 0.707107, 0]


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
                # gym.refresh_particle_state_tensor(sim)
                # saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                # init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                shapesrv_start_time = timeit.default_timer()
                start_vis_cam = True


        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
            print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 


            # current_pc = get_point_cloud()
            current_pc = down_sampling(get_partial_point_cloud(i))                  
                            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc)  
            # open3d.visualization.draw_geometries([pcd, pcd_goal])  
            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            print("chamfer distance: ", chamfer_dist)

            if chamfer_dist >= min_chamfer_dist:
                current_pc = torch.from_numpy(current_pc).permute(1,0).unsqueeze(0).float()
                with torch.no_grad():
                    desired_position = model(current_pc.to(device), goal_pc)[0].cpu().detach().numpy()*(0.001) 
            print("from model:", desired_position)
            print("ground truth: ", goal_position)             

            tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits, error_threshold = 2e-3)     

            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))
            # test = timeit.default_timer()

            state = "move to goal"


        if state == "move to goal":           
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
                print("Lost contact with robot")
                state = "reset" 
                final_chamfer_distances.append(999) 
                goal_count += 1
            
            if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                print("Timeout")
                state = "reset" 
                current_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(current_pc)   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(min(saved_chamfers)) 
                goal_count += 1

            else:
                action = tvc_behavior.get_action()  
                # print("==test timer:", timeit.default_timer()-test)
                # print("complete:", tvc_behavior.is_complete_success()) 
                # print(action)
                if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 1.5:   
                    final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                    delta_x = -(final_pose["pose"]["p"]["x"] - anchor_pose["pose"]["p"]["x"])
                    delta_y = -(final_pose["pose"]["p"]["y"] - anchor_pose["pose"]["p"]["y"])
                    delta_z = final_pose["pose"]["p"]["z"] - anchor_pose["pose"]["p"]["z"]
                    print("delta x, y, z:", delta_x, delta_y, delta_z)
                    state = "get shape servo plan"    
                    # pcd = open3d.geometry.PointCloud()
                    # pcd.points = open3d.utility.Vector3dVector(get_partial_point_cloud(i)) 
                    # pcd.paint_uniform_color([0,0,0])
                    # open3d.visualization.draw_geometries([pcd, pcd_goal]) 
                else:
                    # print("desired vel:", action.get_joint_position())
                    gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())        

                # Terminal conditions
                if all(abs(desired_position) <= 0.005) \
                        or chamfer_dist < min_chamfer_dist:
                    
                    current_pc = get_point_cloud()
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)  
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                    print("final chamfer distance: ", chamfer_dist)
                    final_chamfer_distances.append(chamfer_dist) 
                    goal_count += 1

                    state = "reset" 



        if state == "reset":   
            rospy.loginfo("**Current state: " + state)
            frame_count = 0
            saved_chamfers = []
            
            rospy.logwarn(("=== JUST ENDED goal_count" + str(goal_count)))

            # gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            # gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.15,0,0,0,1.5,0.8]) 
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)  
            gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.15,0,0,0,1.5,0.8]) 

            print("Sucessfully reset robot and object")

            shapesrv_start_time = timeit.default_timer()
            # state = "get shape servo plan"
            state = "home"
            
            # Go to next goal pc
            if goal_count < max_goal_count:
                goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])
                goal_pc = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                goal_position = goal_datas[goal_count]["positions"]               
                gt_mp = np.array(list(goal_datas[goal_count]["mani_point"][0]))
                pcd_goal.paint_uniform_color([1,0,0])

            if fail_mtp:
                state = "home"  
                fail_mtp = False
        

        if  goal_count >= max_goal_count:                    
            all_done = True 
            final_data = final_chamfer_distances
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
