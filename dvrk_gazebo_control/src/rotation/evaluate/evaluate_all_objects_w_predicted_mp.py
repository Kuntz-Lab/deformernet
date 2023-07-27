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

def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

def get_mp_classifier(classifier_model, partial_init_pc, partial_goal_pc, full_init_pc, 
                    num_candidates=400, batch_size=64, compute_method="weighted average", vis=False):
    
    pc_goal_tensor = torch.from_numpy(partial_goal_pc).permute(1,0).unsqueeze(0).float().to(device)    

    # Sample (randomly) some points for MP candidates; only pick points that are in the top half of the object (near robot)
    downsampled_pc = down_sampling(partial_init_pc, num_pts=512) #partial_init_pc
    ys = downsampled_pc[:,1]
    avg_y = (max(ys) + min(ys))/2
    mp_candidates_idxs = np.where(ys >= avg_y)[0]     # only pick points that are in the top half of the object
    mp_candidates_idxs = np.random.choice(mp_candidates_idxs, size = num_candidates)
    mp_candidates = downsampled_pc[mp_candidates_idxs]

    # Configure input to the neural network (concat MP to the current point cloud)
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
        # for batch_pc, batch_pc_goal in zip(torch.split(modified_pc_tensor, num_candidates//batch_size), torch.split(pcs_goal_tensor, num_candidates//batch_size)):
        for batch_pc, batch_pc_goal in zip(torch.split(modified_pc_tensor, batch_size), torch.split(pcs_goal_tensor, batch_size)):
            outputs.append(classifier_model(batch_pc, batch_pc_goal))


        output = torch.cat(tuple(outputs), dim=0)
        success_probs = np.exp(output.cpu().detach().numpy())[:,1]
        print("max(success_probs):", max(success_probs))       
        if compute_method == "max":
            best_mp = mp_candidates[np.argmax(success_probs)]   # point with the highest probability
        elif compute_method == "weighted average":
            best_mp = np.average(mp_candidates, axis=0, weights=success_probs)   # weighted average of all candidates, with weight equal to probability


        if vis:
            success_probs = (success_probs/max(success_probs))
            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(partial_init_pc))
            pcd_goal = open3d.geometry.PointCloud()
            pcd_goal.points = open3d.utility.Vector3dVector(np.array(partial_goal_pc))    

            # Predicted MP
            heats = np.array([[prob, 0, 0] for prob in success_probs])
            best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            best_mani_point.paint_uniform_color([0,1,0])  

            colors = np.zeros(partial_init_pc.shape)
            colors[mp_candidates_idxs] = heats  
            pcd.colors =  open3d.utility.Vector3dVector(colors) 
            
            # Ground truth MP
            mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point.paint_uniform_color([0,0,1])
           
            # Visualization
            open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), best_mani_point.translate(tuple(best_mp)), \
                                                    pcd_goal.translate((0.2,0,0))])  
        

    return best_mp


def get_mp_seg(seg_model, pc_init_numpy, pc_initial_tensor, pc_goal_tensor, compute_method="max", vis=False, num_kp_candidates=100):
    output = seg_model(pc_initial_tensor, pc_goal_tensor)
    success_probs = np.exp(output.squeeze().cpu().detach().numpy())[1,:]
    # print("total candidates:", sum([1 if s>0.5 else 0 for s in success_probs]))
    # print(success_probs)
    # print(np.max(success_probs))

    if compute_method == "max":
        best_mp = pc_init_numpy[np.argmax(success_probs)]  # point with the highest probability
    elif compute_method == "weighted average":
        best_mp = np.average(pc_init_numpy, axis=0, weights=success_probs) 
    elif compute_method == "keypoint":    
        best_candidates =  pc_init_numpy[np.argsort(success_probs)[-num_kp_candidates:]]
        dists_to_gt = np.linalg.norm(gt_mp-best_candidates, axis=1)
        best_mp = best_candidates[np.argsort(dists_to_gt)[-1]]

    if vis:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(pc_init_numpy))

        best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        best_mani_point.paint_uniform_color([0,1,0])  
        best_mani_point.translate(tuple(best_mp))

        gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        gt_mani_point.paint_uniform_color([0,0,1])
        gt_mani_point.translate(tuple(gt_mp))

        heats = np.array([[prob, 0, 0] for prob in success_probs/max(success_probs)])
        pcd.colors =  open3d.utility.Vector3dVector(heats) 

        open3d.visualization.draw_geometries([pcd, best_mani_point, deepcopy(pcd_goal).translate((0.2,0,0)), gt_mani_point])
    
    return best_mp

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
            {"name": "--inside", "type": str, "default": "True", "help": "inside train distribution"},
            {"name": "--use_rot", "type": str, "default": "True", "help": "use model with orientation"},
             {"name": "--use_mp_input", "type": str, "default": "True", "help": "use model with MP input"},
            {"name": "--mp_method", "type": str, "default": "dense_predictor", "help": "select MP selection method. \
                        Options: ground_truth, dense_predictor, classifier, keypoint"},           
            {"name": "--headless", "type": str, "default": "False", "help": "headless mode"}])

    num_envs = args.num_envs
    
    args.headless = args.headless == "True"
    args.inside = args.inside == "True"
    args.use_rot = args.use_rot == "True"
    args.use_mp_input = args.use_mp_input == "True"
    args.obj_name = f"{args.prim_name}_{args.obj_name}"

    object_category = f"{args.prim_name}_{args.stiffness}"
    main_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{object_category}Pa/evaluate"
    objects_path = "/home/baothach/shape_servo_data/evaluation"        
    distribution_keyword = "inside" if args.inside else "outside"


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
    object_meshes_path = os.path.join(objects_path, "meshes", object_category, distribution_keyword)


    with open(os.path.join(object_meshes_path, args.obj_name + ".pickle"), 'rb') as handle:
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


    asset_root = os.path.join(objects_path, "urdf", object_category, distribution_keyword)



    soft_asset_file = args.obj_name + ".urdf"    
    # asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/test"
    # soft_asset_file = "long_box.urdf"
    # asset_root = "/home/baothach/Downloads"
    # soft_asset_file = "test_box.urdf"



    soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.5)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)

    if args.prim_name == "box": 
        soft_pose.p = gymapi.Vec3(0.0, -0.42, thickness/2*0.5)
        soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    elif args.prim_name == "cylinder": 
        soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, r/2.0)
        soft_pose.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    elif args.prim_name == "hemis":
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0, 0.4-two_robot_offset, -o/2.)

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
    cam_positions.append(gymapi.Vec3(0.17, -0.62, 0.2))
    cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.01))
  

    
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
    rospy.logerr(f"Object type ... {object_category}; inside: {args.inside}") 
    rospy.logerr(f"MP method: {args.mp_method}")
    rospy.logerr(f"use_rot: {args.use_rot};  use_mp_input: {args.use_mp_input}")
 

    # Some important paramters
    init()  # Initilize 2 robots' joints
    all_done = False
    state = "home"
    first_time = True    


    # Set up DNN:
    device = torch.device("cuda")

    if args.use_rot and args.use_mp_input:
        model_category = 0
        weight_keyword = "w_rot_w_MP"
    elif args.use_rot and (not args.use_mp_input):    
        model_category = 1
        weight_keyword = "w_rot_no_MP"
    elif (not args.use_rot) and args.use_mp_input:    
        model_category = 2
        weight_keyword = "no_rot_w_MP"
    elif (not args.use_rot) and (not args.use_mp_input):    
        model_category = 3
        weight_keyword = "no_rot_no_MP"
    # rospy.logerr(f"weight keyword: {weight_keyword}")

    
    ## Box
    if object_category == "box_1k":
        deformernet_epoch_num_dict = {0:524, 1:600, 2:240, 3:240}  
        mp_dense_epoch_num = 118  
        mp_classifier_epoch_num = 150   
    elif object_category == "box_5k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:100, 3:100}
        mp_dense_epoch_num = 124
        mp_classifier_epoch_num = 77 
    elif object_category == "box_10k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:100, 3:240}
        mp_dense_epoch_num = 136
        mp_classifier_epoch_num = 100 
    
    ## Cylinder
    elif object_category == "cylinder_1k":
        deformernet_epoch_num_dict = {0:150, 1:160, 2:100, 3:220}   
        mp_dense_epoch_num = 160    
        mp_classifier_epoch_num = 100     
    elif object_category == "cylinder_5k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:240, 3:100}
        mp_dense_epoch_num = 130
        mp_classifier_epoch_num = 100 
    elif object_category == "cylinder_10k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:100, 3:100}
        mp_dense_epoch_num = 160
        mp_classifier_epoch_num = 98 
    
    ## Hemisphere
    elif object_category == "hemis_1k":
        deformernet_epoch_num_dict = {0:600, 1:300, 2:240, 3:214} 
        mp_dense_epoch_num = 202    
        mp_classifier_epoch_num = 98  
    elif object_category == "hemis_5k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:100, 3:100}
        mp_dense_epoch_num = 160
        mp_classifier_epoch_num = 99 
    elif object_category == "hemis_10k":
        deformernet_epoch_num_dict = {0:200, 1:160, 2:100, 3:100}
        mp_dense_epoch_num = 160
        mp_classifier_epoch_num = 96 


    ### Set up DeformerNet
    deformernet_model_main_path = "/home/baothach/shape_servo_DNN"
    if args.use_rot:
        sys.path.append(f"{deformernet_model_main_path}/rotation")
        from architecture_2 import DeformerNetMP as DeformerNet
        model = DeformerNet(use_mp_input=args.use_mp_input).to(device)
        
    else:
        sys.path.append(f"{deformernet_model_main_path}/generalization_tasks")
        if args.use_mp_input:
            from architecture import DeformerNetMP as DeformerNet
        else:
            from architecture import DeformerNet
        model = DeformerNet().to(device)
        
  
    weight_path = f"/home/baothach/shape_servo_data/rotation_extension/multi_{object_category}Pa/weights/run1_{weight_keyword}/"        
    deformernet_epoch_num = deformernet_epoch_num_dict[model_category] 
    model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {deformernet_epoch_num}")))  
    model.eval()


    mp_model_main_path = "/home/baothach/shape_servo_DNN/learn_mp"
    sys.path.append(mp_model_main_path)



    ### Set up manipulation point model
    if args.mp_method == "dense_predictor" or args.mp_method == "keypoint":
        from test_pointconv import ManiPointSegment
        
        mp_seg_model = ManiPointSegment(num_classes=2).to(device)

        # if object_category == "box_1k11xxx":
        #     weight_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_box_5kPa/weights/seg/run1"
        #     # rospy.logerr("Using box 5k seg instead")
        #     mp_seg_model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {124}")))
        # else:
        weight_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{object_category}Pa/weights/seg/run1"        
        mp_seg_model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {mp_dense_epoch_num}")))       
        
        
        mp_seg_model.eval()

    elif args.mp_method == "classifier":
        from architecture_classifier import ManiPointNet  
        
        mp_classifier_model = ManiPointNet(normal_channel=False).to(device)
        weight_path = f"/home/baothach/shape_servo_data/manipulation_points/multi_{object_category}Pa/weights/classifier/run1"
        
        mp_classifier_model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {mp_classifier_epoch_num}")))    
        mp_classifier_model.eval()    


    goal_recording_path = os.path.join(main_path, "goal_data", object_category, distribution_keyword)
    
    chamfer_path_kw = f"{distribution_keyword}_{args.mp_method}_{weight_keyword}"
    chamfer_recording_path = os.path.join(main_path, "chamfer_results", object_category, chamfer_path_kw)
    
    os.makedirs(chamfer_recording_path, exist_ok=True)


    goal_count = 0 #0
    frame_count = 0
    max_goal_count =  10  #10

    max_shapesrv_time = 1.5*60    # 2 mins
    if args.inside:
        min_chamfer_dist = 0.1 #0.2
    else:
        min_chamfer_dist = 0.2 #0.25
    fail_mtp = False
    saved_nodes = []
    saved_chamfers = []
    final_node_distances = []  
    final_chamfer_distances = []    


    dc_client = GraspDataCollectionClient()   

   
    # Get 10 goal pc data for 1 object:
    with open(os.path.join(goal_recording_path, args.obj_name + ".pickle"), 'rb') as handle:
        goal_datas = pickle.load(handle) 
    goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])   # first goal pc
    goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device) 
    pcd_goal = open3d.geometry.PointCloud()
    pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy)  
    pcd_goal.paint_uniform_color([1,0,0]) 

    full_pc_goal = goal_datas[goal_count]["full pcs"][1]
    rospy.logwarn(f"number of nodes on mesh: {full_pc_goal.shape}")

    init_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][0])  # first goal pc
    init_pc_tensor = torch.from_numpy(init_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)  
    gt_mp = np.array(goal_datas[goal_count]["mani_point"])

    full_pc_numpy = goal_datas[goal_count]["full pcs"][0]

    if args.use_rot:        
        goal_pos = goal_datas[goal_count]["pos"] 
        rospy.logerr(f"=========goal_pos: {goal_pos}")
        goal_rot = goal_datas[goal_count]["rot"]         
    else:
        goal_position = goal_datas[goal_count]["pos"].T.squeeze()

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
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.24)             
            if frame_count == 5:
                rospy.loginfo("**Current state: " + state)
                

                if first_time:                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    first_time = False
                
                # # state = "get shape servo plan"
                
                # desired_position = np.array([0.,0.,0.]) # Set intiial desired gripper position
                # # frame_count = 0

                # gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
                # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
                # gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper1_joint"), 0.35)
                # gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka2", "psm_tool_gripper2_joint"), -0.35)  
                # dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_VEL)
                # dof_props_2["stiffness"][:8].fill(0.0)
                # dof_props_2["damping"][:8].fill(200.0)
                # gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)
                
                # shapesrv_start_time = timeit.default_timer()
                # # open3d.visualization.draw_geometries([pcd_goal])


                # # anchor_pose = deepcopy(init_pose)
                # # anchor_eulers = deepcopy(init_eulers)    

                frame_count = 0
                state = "generate preshape"

            # elif frame_count == 6:
            #     frame_count = 0
            #     _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
            #     init_eulers = transformations.euler_from_matrix(init_pose)
            #     state = "get shape servo plan"

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            # preshape_response = boxpcopy(preshape_response.palm_goal_pose_world[0].pose)        
            with torch.no_grad():
                
                ### Ground truth:
                if args.mp_method == "ground_truth":
                    best_mp = gt_mp

                ### Seg (Dense predictor):                
                if args.mp_method == "dense_predictor":
                    if object_category == "box_1k":   # something wrong dense predictor
                        # rospy.logerr("Using gt mp")
                        best_mp = gt_mp
                    else:
                        best_mp = get_mp_seg(seg_model=mp_seg_model, pc_init_numpy=init_pc_numpy, pc_initial_tensor=init_pc_tensor, pc_goal_tensor=goal_pc_tensor)


                ### Keypoint method
                if args.mp_method == "keypoint":
                    if object_category == "box_1k":
                        best_mp = get_mp_seg(seg_model=mp_seg_model, pc_init_numpy=init_pc_numpy, pc_initial_tensor=init_pc_tensor, pc_goal_tensor=goal_pc_tensor, \
                                            compute_method="keypoint", num_kp_candidates=70, vis=False)                           
                    else:
                        if args.prim_name == "cylinder":
                            best_mp = get_mp_seg(seg_model=mp_seg_model, pc_init_numpy=init_pc_numpy, pc_initial_tensor=init_pc_tensor, pc_goal_tensor=goal_pc_tensor, \
                                                compute_method="keypoint", num_kp_candidates=70)                          
                        else:
                            best_mp = get_mp_seg(seg_model=mp_seg_model, pc_init_numpy=init_pc_numpy, pc_initial_tensor=init_pc_tensor, pc_goal_tensor=goal_pc_tensor, \
                                                compute_method="keypoint", num_kp_candidates=100, vis=False)                
                
                ### classifier:
                elif args.mp_method == "classifier":
                    best_mp = get_mp_classifier(mp_classifier_model, init_pc_numpy, goal_pc_numpy, full_pc_numpy, vis=False, num_candidates=400, batch_size=128)
                    

            # target_pose = [-best_mp[0], -best_mp[1], best_mp[2] - ROBOT_Z_OFFSET, 0, 0.707107, 0.707107, 0]
            target_pose = [-best_mp[0], -best_mp[1], best_mp[2] - ROBOT_Z_OFFSET-0.01, 0, 0.707107, 0.707107, 0]  


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


                _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                init_eulers = transformations.euler_from_matrix(init_pose)

        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)


            current_pc_numpy = down_sampling(get_partial_point_cloud(i))                  
                            
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(current_pc_numpy)  
            pcd.paint_uniform_color([0,0,0])
            # open3d.visualization.draw_geometries([pcd, pcd_goal]) 


            node_dist = np.linalg.norm(full_pc_goal - get_point_cloud())
            saved_nodes.append(node_dist)
            rospy.logwarn(f"Node distance: {node_dist}")

            chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
            saved_chamfers.append(chamfer_dist)
            rospy.logwarn(f"chamfer distance: {chamfer_dist}")
        
            if args.use_mp_input:
                mani_point = init_pose[:3,3] * np.array([-1,-1,1]) + np.array([0,0, ROBOT_Z_OFFSET])
               
                neigh = NearestNeighbors(n_neighbors=50)
                neigh.fit(current_pc_numpy)
                _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
                mp_channel = np.zeros(current_pc_numpy.shape[0])
                mp_channel[nearest_idxs.flatten()] = 1
                
                modified_pc = np.vstack([current_pc_numpy.transpose(1,0), mp_channel])
                current_pc_tensor = torch.from_numpy(modified_pc).unsqueeze(0).float().to(device)                
            
            else:
                current_pc_tensor = torch.from_numpy(current_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)

           
            with torch.no_grad():
                if not args.use_rot:
                    desired_position = model(current_pc_tensor, goal_pc_tensor)[0].cpu().detach().numpy()*(0.001) 
                    tvc_behavior = TaskVelocityControl(list(desired_position), robot, sim_params.dt, 3, vel_limits=vel_limits)
                    print("from model:", desired_position)
                    print("ground truth: ", goal_position)   
                else:
                    pos, rot_mat = model(current_pc_tensor, goal_pc_tensor) 
                    pos *= 0.001
                    pos, rot_mat = pos.detach().cpu().numpy(), rot_mat.detach().cpu().numpy()
                    
                    # if max(abs(pos.squeeze())) >= 0.08:
                    #     pos *= (0.05/max(abs(pos.squeeze())))

                    # if args.prim_name == "box": #in ["box", "cylinder"]:
                    #     pos *= 3/4
                    # elif args.prim_name == "cylinder":
                    #     pos[0][:2] *= 3/4
                    # elif args.prim_name == "hemis":
                    #     pass
                    # else:
                    #     raise Exception("Wrong object category")

                    # if args.prim_name == "box" and first_time and h <= 0.25:
                    #     # pos[0][2] = 0.05
                    #     pos[0][2] = max(0.05, pos[0][2])
                    #     temp = max(w / 2 * 0.5, abs(pos[0][0]))
                    #     pos[0][0] *= temp/abs(pos[0][0])
                    #     first_time = False                   


                    desired_pos = (pos + init_pose[:3,3]).flatten()
                    desired_rot = rot_mat @ init_pose[:3,:3]



                    tvc_behavior = TaskVelocityControl2([*desired_pos, desired_rot], robot, sim_params.dt, 3, vel_limits=vel_limits, use_euler_target=False, \
                                                        pos_threshold = 2e-3, ori_threshold=5e-2)

                    temp1 = np.eye(4)
                    temp1[:3,:3] = rot_mat
                    temp2 = np.eye(4)
                    temp2[:3,:3] = goal_rot            
                    print("pos, rot_mat:", pos, transformations.euler_from_matrix(temp1))
                    print("goal_pos, goal_rot:", goal_pos, transformations.euler_from_matrix(temp2)) 


            closed_loop_start_time = deepcopy(gym.get_sim_time(sim))

            state = "move to goal"


        if state == "move to goal":           
            contacts = [contact[4] for contact in gym.get_soft_contacts(sim)]
            main_ins_pos = gym.get_joint_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"))
            # if not(9 in contacts or 10 in contacts):  # lose contact w 1 robot
            if main_ins_pos <= 0.042 or (not(9 in contacts or 10 in contacts)):  # lose contact w 1 robot
                rospy.logerr("Lost contact with robot")
                state = "reset" 
                # final_node_distances.append(999) 
                
                chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                
                saved_nodes.append(chamfer_dist)
                final_node_distances.append(999+min(saved_nodes)) 

                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(down_sampling(get_partial_point_cloud(i)))   
                chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                
                saved_chamfers.append(chamfer_dist)
                final_chamfer_distances.append(999+min(saved_chamfers))         

                print("***final chamfer distance: ", min(saved_nodes))

                goal_count += 1
            
            else:
                if timeit.default_timer() - shapesrv_start_time >= max_shapesrv_time:
                    rospy.logerr("Timeout")
                    state = "reset" 
                    
                    chamfer_dist = np.linalg.norm(full_pc_goal - get_point_cloud())                    
                    saved_nodes.append(chamfer_dist)
                    final_node_distances.append(1999+min(saved_nodes)) 

                    current_pc = down_sampling(get_partial_point_cloud(i))
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(current_pc)   
                    chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))                    
                    saved_chamfers.append(chamfer_dist)
                    final_chamfer_distances.append(1999+min(saved_chamfers)) 

                    print("***final chamfer distance: ", min(saved_nodes))
                    
                    goal_count += 1

                else:
                    action = tvc_behavior.get_action()  
                    if action is None or gym.get_sim_time(sim) - closed_loop_start_time >= 3:   
                        _,init_pose = get_pykdl_client(robot.get_arm_joint_positions())
                        init_eulers = transformations.euler_from_matrix(init_pose) 
                        state = "get shape servo plan"    
                    else:
                        # print("desired vel:", action.get_joint_position())
                        gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, action.get_joint_position())     
                        # _,final_pose = get_pykdl_client(robot.get_arm_joint_positions())   
                        # rospy.logerr(f"true delta: {final_pose[:3,3]-init_pose[:3,3]}")

                    # Terminal conditions
                    if args.use_rot:                        
                        converge = all(abs(pos.squeeze()) <= 0.006)
                    else:
                        converge = all(abs(desired_position) <= 0.006)
                    
                    if converge or chamfer_dist < min_chamfer_dist:
                        
                        node_dist = np.linalg.norm(full_pc_goal - get_point_cloud())             
                        print("***final node distance: ", node_dist/full_pc_goal.shape[0]*1000)
                        final_node_distances.append(node_dist) 


                        current_pc = down_sampling(get_partial_point_cloud(i))
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(current_pc)  
                        chamfer_dist = np.linalg.norm(np.asarray(pcd_goal.compute_point_cloud_distance(pcd)))
                        print("***final chamfer distance: ", chamfer_dist)
                        final_chamfer_distances.append(chamfer_dist) 
           

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

            
            # Go to next goal pc
            if goal_count < max_goal_count:
                goal_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][1])
                goal_pc_tensor = torch.from_numpy(goal_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)
                pcd_goal = open3d.geometry.PointCloud()
                pcd_goal.points = open3d.utility.Vector3dVector(goal_pc_numpy) 
                pcd_goal.paint_uniform_color([1,0,0])

                full_pc_goal = goal_datas[goal_count]["full pcs"][1]
                init_pc_numpy = down_sampling(goal_datas[goal_count]["partial pcs"][0])  # first goal pc
                init_pc_tensor = torch.from_numpy(init_pc_numpy).permute(1,0).unsqueeze(0).float().to(device)  
                gt_mp = np.array(goal_datas[goal_count]["mani_point"])

                full_pc_numpy = goal_datas[goal_count]["full pcs"][0]

                if args.use_rot:                   
                    goal_pos = goal_datas[goal_count]["pos"] 
                    goal_rot = goal_datas[goal_count]["rot"]                       
                else:
                    goal_position = goal_datas[goal_count]["pos"].T.squeeze()

                saved_obj_contact_state = goal_datas[goal_count]["obj contact state"]
                saved_robot_contact_state = goal_datas[goal_count]["robot contact state"]

            # gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], saved_robot_contact_state, gymapi.STATE_ALL) 
            # gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_obj_contact_state))
            # print("Sucessfully reset robot and object")
            # state = "get shape servo plan"

            dof_props_2['driveMode'][:8].fill(gymapi.DOF_MODE_POS)
            dof_props_2["stiffness"][:8].fill(200.0)
            dof_props_2["damping"][:8].fill(40.0)
            

            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            gym.set_actor_dof_velocity_targets(robot.env_handle, robot.robot_handle, [0]*8)
            gym.set_actor_dof_position_targets(robot.env_handle, robot.robot_handle, [0,0,0,0,0.22,0,0,0,1.5,0.8]) 
            
            print("Sucessfully reset robot and object")
            pc_on_trajectory = []
            full_pc_on_trajectory = []
            curr_trans_on_trajectory = []
                

            gym.set_actor_dof_properties(env, kuka_handles_2[i], dof_props_2)



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
           
            # # final_data = final_node_distances
            final_data = {"node": final_node_distances, "chamfer": final_chamfer_distances, "num_nodes": full_pc_goal.shape[0]}
            with open(os.path.join(chamfer_recording_path, args.obj_name + ".pickle"), 'wb') as handle:
                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


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
