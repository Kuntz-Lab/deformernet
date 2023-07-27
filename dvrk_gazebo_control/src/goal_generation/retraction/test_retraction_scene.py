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
from copy import copy
import rospy
# from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
from GraspDataCollectionClient import GraspDataCollectionClient
import open3d
# from utils import open3d_ros_helper as orh
# from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
# #import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
from utils.isaac_utils import fix_object_frame
import pickle
# from ShapeServo import *
# from sklearn.decomposition import PCA
import timeit
from copy import deepcopy
from goal_plane_backup import *
from core import Robot
from behaviors import MoveToPose

sys.path.append('/home/baothach/shape_servo_DNN/generalization_tasks')
# from pointcloud_recon_2 import PointNetShapeServo, PointNetShapeServo2
from architecture import DeformerNet
import torch
# from PIL import Image
import transformations


ROBOT_Z_OFFSET = 0.30
# angle_kuka_2 = -0.4
# init_kuka_2 = 0.15
two_robot_offset = 0.86



def init():
    for i in range(num_envs):
        # # Kuka 2
        davinci_dof_states = gym.get_actor_dof_states(envs[i], kuka_handles_2[i], gymapi.STATE_NONE)
        davinci_dof_states['pos'][8] = 1.5
        davinci_dof_states['pos'][9] = 0.8
        davinci_dof_states['pos'][4] = 0.22
        gym.set_actor_dof_states(envs[i], kuka_handles_2[i], davinci_dof_states, gymapi.STATE_POS)

def get_point_cloud():
    gym.refresh_particle_state_tensor(sim)
    particle_state_tensor = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim)))
    point_cloud = particle_state_tensor.numpy()[:, :3]  
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud))
    # open3d.visualization.draw_geometries([pcd])     

    # obb = pcd.get_oriented_bounding_box()

    # points = np.asarray(obb.get_box_points())
    # lines = [
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [1, 6],
    #     [1, 7],
    #     [2, 5], 
    #     [2, 7],
    #     [3, 5],
    #     [3, 6],
    #     [4, 5],
    #     [4, 6],
    #     [4, 7],
    # ]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = open3d.geometry.LineSet(
    #     points=open3d.utility.Vector3dVector(points),
    #     lines=open3d.utility.Vector2iVector(lines),
    # )
    # line_set.colors = open3d.utility.Vector3dVector(colors)
    # open3d.visualization.draw_geometries([pcd, line_set]) 



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
    seg_buffer = gym.get_camera_image(sim, envs[i], cam_handles[i], gymapi.IMAGE_SEGMENTATION)


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
    # depth_buffer[seg_buffer != 10] = -10001

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
            if p2[0, 2] > 0.03:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(np.array(points))
    # open3d.visualization.draw_geometries([pcd]) 

    # return points
    return np.array(points).astype('float32')

 

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

    # initialize gym
    gym = gymapi.acquire_gym()

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"},
            {"name": "--headless", "type": bool, "default": False, "help": "headless mode"}])

    num_envs = args.num_envs
    


    # configure sim
    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
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


    rigid_asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/goal_generation"
    rigid_asset_file = "kidney_rigid.urdf"
    rigid_pose = gymapi.Transform()
    rigid_pose.p = gymapi.Vec3(0.00, 0.40-two_robot_offset, 0.03)
    rigid_pose.r = gymapi.Quat(0.0, 0.0, -0.707107, 0.707107)
    kidney_angle = -np.pi/4.5
    eulers = [-np.pi/2+kidney_angle, 0, np.pi]
    quat = transformations.quaternion_from_euler(*eulers)
    rigid_pose.r = gymapi.Quat(*list(quat))
    
    asset_options.thickness = 0.003 # 0.002
    rigid_asset = gym.load_asset(sim, rigid_asset_root, rigid_asset_file, asset_options)



    
    # Load soft objects' assets

    asset_root = "/home/baothach/sim_data/Custom/Custom_urdf/goal_generation"
    soft_asset_file = 'thin_tissue_layer_attached.urdf'


    soft_pose = gymapi.Transform()
    # soft_pose.p = gymapi.Vec3(0.0, 0.50-two_robot_offset, 0.1)
    # soft_pose.p = gymapi.Vec3(0.05, 0.50-two_robot_offset, 0.1)
    soft_pose.p = gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.1)  # 0.081
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)   #for banana
    tissue_angle = -np.pi/5
    eulers = [np.pi/2+tissue_angle, 0, np.pi]
    quat = transformations.quaternion_from_euler(*eulers)
    soft_pose.r = gymapi.Quat(*list(quat))
    
    
    soft_thickness = 0.001    # important to add some thickness to the soft body to avoid interpenetrations





    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    # print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
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
        envs_obj.append(env_obj)        
        
        soft_actor = gym.create_actor(env_obj, soft_asset, soft_pose, "soft", i, 0)

        # add rigid obj
        rigid_actor = gym.create_actor(env, rigid_asset, rigid_pose, 'rigid', i, 0, segmentationId=10)
        color = gymapi.Vec3(1,0,0)
        gym.set_rigid_body_color(env, rigid_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        object_handles.append(soft_actor)


        kuka_handles_2.append(kuka_2_handle)



    dof_props_2 = gym.get_actor_dof_properties(envs[0], kuka_handles_2[0])
    dof_props_2["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props_2["stiffness"].fill(200.0)
    dof_props_2["damping"].fill(40.0)
    dof_props_2["stiffness"][8:].fill(1)
    dof_props_2["damping"][8:].fill(2)  
    

    # Camera setup
    if not args.headless:
        cam_pos = gymapi.Vec3(1, 0.5, 1)
        # cam_pos = gymapi.Vec3(0.3, -0.7, 0.3)
        # cam_pos = gymapi.Vec3(0.3, -0.1, 0.5)  # final setup for thin layer tissue
        # cam_pos = gymapi.Vec3(-0.5, -0.46, 0.4)
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
    # cam_positions.append(gymapi.Vec3(0.15, -0.6, 0.20))
    # cam_targets.append(gymapi.Vec3(0.0, 0.40-two_robot_offset, 0.05))
    # rigid_pose.p = gymapi.Vec3(0.00, 0.40-two_robot_offset, 0.03)

    cam_positions.append(gymapi.Vec3(0.15, 0.37-two_robot_offset, 0.25))
    cam_targets.append(gymapi.Vec3(0.035, 0.37-two_robot_offset, 0.05))

    
    for i, env_obj in enumerate(envs_obj):
        # for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env_obj, cam_props))
            gym.set_camera_location(cam_handles[i], env_obj, cam_positions[0], cam_targets[0])


    # # Camera for point cloud setup
    # vis_cam_positions = []
    # vis_cam_targets = []
    # vis_cam_handles = []
    # vis_cam_width = 400
    # vis_cam_height = 400
    # vis_cam_props = gymapi.CameraProperties()
    # vis_cam_props.width = vis_cam_width
    # vis_cam_props.height = vis_cam_height



    # vis_cam_positions.append(gymapi.Vec3(0.2, 0.4-two_robot_offset, 0.2))   # 1
    # # vis_cam_positions.append(gymapi.Vec3(0.15, 0.3-two_robot_offset, 0.2))   # 2
    # # vis_cam_positions.append(gymapi.Vec3(-0.1, 0.3-two_robot_offset, 0.2))   # 3
    # # vis_cam_positions.append(gymapi.Vec3(0.15, 0.2-two_robot_offset, 0.2))   # 4
    # vis_cam_targets.append(gymapi.Vec3(0.0, 0.3-two_robot_offset, 0.05))   #4
    # # vis_cam_targets.append(gymapi.Vec3(0.0, 0.4-two_robot_offset, 0.05))


    
    # for i, env_obj in enumerate(envs_obj):
    #     # for c in range(len(cam_positions)):
    #         vis_cam_handles.append(gym.create_camera_sensor(env_obj, vis_cam_props))
    #         gym.set_camera_location(vis_cam_handles[i], env_obj, vis_cam_positions[0], vis_cam_targets[0])
    
    
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

    
    sample_count = 0
    frame_count = 0


    final_point_clouds = []
    final_desired_positions = []
    first_time = True
    save_intial_pc = True
    get_goal_pc = True
    

    vis_frame_count = 0
    num_image = 0
    start_vis_cam = True
    prepare_vis_cam = True
    prepare_vis_goal_pc = True
    prepare_vis_shift_plane = True


    # Set up DNN:
    device = torch.device("cuda")
    model = DeformerNet(normal_channel=False)
    # model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run1/epoch 150"))
    model.load_state_dict(torch.load("/home/baothach/shape_servo_data/generalization/surgical_setup/weights/run2(on_ground)/epoch 128"))
    model.eval()

    dc_client = GraspDataCollectionClient()
    saved_plane_path = "/home/baothach/shape_servo_data/generalization/plane/computed_mani_points"
    data_recording_path = "/home/baothach/shape_servo_data/generalization/plane/results"   # success or nah

    
    execute_count = 0
    plane_count = 75
    num_new_goal = 0
    max_execute_count = 3
    max_plane_count = 100
    max_num_new_goal = 10    
    max_plane_time = 2 * 60 # 2 mins
    
    
    
    start_time = timeit.default_timer()    
    close_viewer = False
    robot = Robot(gym, sim, envs[0], kuka_handles_2[0])
    
    # constrain_plane = np.array([0, 1, 0, 0.50]) # 1
    # # constrain_plane = np.array([1, 1, 0, 0.4])  # 2
    # # constrain_plane = np.array([-1, 1, 0, 0.48])  # 3
    # # constrain_plane = np.array([0, 1, 0, 0.57])  # 4
    # constrain_plane = np.array([1, 1, 0, 0.38])
    # shift_plane = constrain_plane.copy()
    # shift_plane[3] += 0.03 

    while (not close_viewer) and (not all_done): 

        if not args.headless:
            close_viewer = gym.query_viewer_has_closed(viewer)  

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        t = gym.get_sim_time(sim)


        # if prepare_vis_cam:
        #     plane_points = visualize_plane(constrain_plane, num_pts=50000)
        #     plane_xs, plane_ys = get_goal_projected_on_image(plane_points, i, thickness = 0)
        #     valid_ind = []
        #     for t in range(len(plane_xs)):
        #         if 0 < plane_xs[t] < vis_cam_width and 0 < plane_ys[t] < vis_cam_height:
        #             valid_ind.append(t)
        #     plane_xs = np.array(plane_xs)[valid_ind]
        #     plane_ys = np.array(plane_ys)[valid_ind]
            


        #     prepare_vis_cam = False

        # if start_vis_cam: 
        #     if vis_frame_count % 20 == 0:
        #         gym.render_all_camera_sensors(sim)
        #         im = gym.get_camera_image(sim, envs_obj[i], vis_cam_handles[0], gymapi.IMAGE_COLOR).reshape((vis_cam_height,vis_cam_width,4))
        #         # goal_xs, goal_ys = get_goal_projected_on_image(data["full pcs"][1], i, thickness = 1)
                
        #         im[plane_xs, plane_ys, :] = [0,0,255,255]
             


        #         im = Image.fromarray(im)
                
        #         img_path =  os.path.join(save_path, "image" + f'{num_image:03}' + ".png")
                
        #         im.save(img_path)
        #         num_image += 1         
            
   

            # vis_frame_count += 1


        if state == "home" :   
            frame_count += 1
            gym.set_joint_target_position(envs[0], gym.get_joint_handle(envs[0], "kuka2", "psm_main_insertion_joint"), 0.22)            
            

            if frame_count == 10:
                rospy.logerr("**Current state: " + state + ", current plane count: " + str(plane_count))

                if first_time:
                    frame_state = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)                
                    frame_state['pose']['p']['z'] -= 0.05                
                    gym.set_actor_rigid_body_states(envs[i], object_handles[i], frame_state, gymapi.STATE_ALL) 
                    
                    gym.refresh_particle_state_tensor(sim)
                    saved_object_state = deepcopy(gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))) 
                    init_robot_state = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_ALL))
                    first_time = False



                # Get plane and computed manipulation point:
                with open(os.path.join(saved_plane_path, f"plane_{plane_count}.pickle"), 'rb') as handle:
                    plane_data = pickle.load(handle) 
                    constrain_plane = plane_data["plane"]
                    mani_point = plane_data["mani_point"]


                state = "generate preshape"
                
                frame_count = 0

                current_pc = get_point_cloud()
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(current_pc))
                open3d.io.write_point_cloud("/home/baothach/shape_servo_data/multi_grasps/1.pcd", pcd) # save_grasp_visual_data , point cloud of the object
                pc_ros_msg = dc_client.seg_obj_from_file_client(pcd_file_path = "/home/baothach/shape_servo_data/multi_grasps/1.pcd", align_obj_frame = False).obj
                pc_ros_msg = fix_object_frame(pc_ros_msg)

                constrain_plane = np.array([0, 1, 0, -(0.40-0.86)])
                rospy.logerr(f"constrain_plane: {constrain_plane}")
                
                # pcd = open3d.geometry.PointCloud()
                # pcd.points = open3d.utility.Vector3dVector(np.array(get_partial_point_cloud(i)))
                # plane_points = visualize_plane(constrain_plane, x_range=0.15, y_range=[-0.57,-0.27], z_range=[0.0,0.18], num_pts=2000)
                # pcd_plane = open3d.geometry.PointCloud()
                # pcd_plane.points = open3d.utility.Vector3dVector(plane_points)
                # pcd.paint_uniform_color([0,0,0]) 
                # pcd_plane.paint_uniform_color([0,1,0])

                constrain_plane_2 = np.array([-np.tan(kidney_angle), 1, 0, -(0.40-0.86)+0.02])
                pcd_plane_2 = pcd_ize(visualize_plane(constrain_plane_2, x_range=0.15, y_range=[-0.57,-0.27], z_range=[0.0,0.18], num_pts=2000))
                pcd_plane_2.paint_uniform_color([1,0,0])

                # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                # open3d.visualization.draw_geometries([pcd, pcd_plane, pcd_plane_2, coor])

                

        if state == "generate preshape":                   
            rospy.loginfo("**Current state: " + state)
            preshape_response = dc_client.gen_grasp_preshape_client(pc_ros_msg)               
            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[0].pose)        
            target_pose = [-cartesian_goal.position.x, -cartesian_goal.position.y, cartesian_goal.position.z-ROBOT_Z_OFFSET,
                            0, 0.707107, 0.707107, 0]
            # x, y, z = mani_point[0], mani_point[1], mani_point[2], 
            # target_pose = [-x, -y, z-ROBOT_Z_OFFSET, 0, 0.707107, 0.707107, 0]



            mtp_behavior = MoveToPose(target_pose, robot, sim_params.dt, 2)
            if mtp_behavior.is_complete_failure():
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "reset"                
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
        
                anchor_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                start_vis_cam = True

                plane_start_time = timeit.default_timer()




        if state == "get shape servo plan":
            rospy.loginfo("**Current state: " + state)

            current_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
            print("***Current x, y, z: ", current_pose["pose"]["p"]["x"], current_pose["pose"]["p"]["y"], current_pose["pose"]["p"]["z"] ) 

            current_pc = get_point_cloud()
            
            pcd = pcd_ize(get_partial_point_cloud(i))
            pcd.paint_uniform_color([0,0,0]) 
            open3d.visualization.draw_geometries([pcd, pcd_plane_2]) 
  
            
            num_points = current_pc.shape[0]         
            if save_intial_pc:
                initial_pc = deepcopy(current_pc)
                save_initial_pc = False
            
            if get_goal_pc:
                delta = 0.00
                goal_pc_numpy = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc)                         
                goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                get_goal_pc = False

            current_pc = torch.from_numpy(np.swapaxes(current_pc,0,1)).float()         

            with torch.no_grad():
                desired_position = model(current_pc.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001)  
                # desired_position = model(intial_pc_tensor.unsqueeze(0), goal_pc.unsqueeze(0))[0].detach().numpy()*(0.001) 

            print("from model:", desired_position)
          
            # delta_x = desired_position[0]   
            # delta_y = desired_position[1] 
            # delta_z = desired_position[2] 

            delta_x, delta_y, delta_z = get_action(tissue_angle, y_mag=0.1, z_mag=0.07)

          

            cartesian_pose = Pose()
            cartesian_pose.orientation.x = 0
            cartesian_pose.orientation.y = 0.707107
            cartesian_pose.orientation.z = 0.707107
            cartesian_pose.orientation.w = 0
            cartesian_pose.position.x = -current_pose["pose"]["p"]["x"] + delta_x
            cartesian_pose.position.y = -current_pose["pose"]["p"]["y"] + delta_y
            # cartesian_pose.position.z = current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z
            cartesian_pose.position.z = max(current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + 0.04, current_pose["pose"]["p"]["z"] - ROBOT_Z_OFFSET + delta_z)
            dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']

            plan_traj = dc_client.arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_pose, current_position=dof_states)
            state = "move to goal"
            traj_index = 0

        if state == "move to goal":           
            # Does plan exist?
            if (not plan_traj):
                rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                state = "get shape servo plan"
            else:            
                # if frame_count % 10 == 0:
                #     current_pc = get_point_cloud()
                #     num_points = current_pc.shape[0]                    
                #     num_failed_points = len([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]]) 
                #     rospy.logwarn(f"percentage passed: {1-float(num_failed_points)/float(num_points)}")
                #     print("num, num failed:", num_points, num_failed_points)
                # frame_count += 1

                # print(traj_index, len(plan_traj))

                if num_new_goal >= max_num_new_goal \
                        or timeit.default_timer() - plane_start_time >= max_plane_time:

                    current_pc = get_point_cloud()
                    num_points = current_pc.shape[0]                    
                    num_failed_points = len([p for p in current_pc if constrain_plane[0]*p[0] + constrain_plane[1]*p[1] + constrain_plane[2]*p[2] > -constrain_plane[3]]) 

                    final_data = {"success": False, "% passed": 1-float(num_failed_points)/float(num_points)}
                    
                    with open(os.path.join(data_recording_path, f"plane_{plane_count}.pickle"), 'wb') as handle:
                        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)                       

                    rospy.logerr("FAIL: plane count " + str(plane_count))                             
                    plane_count += 1
                    state = "reset"




                dof_states = gym.get_actor_dof_states(envs[0], kuka_handles_2[0], gymapi.STATE_POS)['pos']
                plan_traj_with_gripper = [plan+[0.15,-0.15] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[0], kuka_handles_2[0], pos_targets)                
                
                if np.allclose(dof_states[:8], pos_targets[:8], rtol=0, atol=0.02):
                    traj_index += 1 

        
                # if traj_index == 10 or traj_index == len(plan_traj):
                if traj_index == len(plan_traj):
                    traj_index = 0  
                    # final_pose = deepcopy(gym.get_actor_rigid_body_states(envs[i], kuka_handles_2[i], gymapi.STATE_POS)[-3])
                    # print("***Final x, y, z: ", final_pose[" pose"]["p"]["x"], final_pose["pose"]["p"]["y"], final_pose["pose"]["p"]["z"] ) 
                    # delta_x = -(final_pose["pose"]["p"]["x"] - anchor_pose["pose"]["p"]["x"])
                    # delta_y = -(final_pose["pose"]["p"]["y"] - anchor_pose["pose"]["p"]["y"])
                    # delta_z = final_pose["pose"]["p"]["z"] - anchor_pose["pose"]["p"]["z"]
                    # print("delta x, y, z:", delta_x, delta_y, delta_z)
                    
                    state = "get shape servo plan" 
                    execute_count += 1

                    if execute_count >= max_execute_count:
                        rospy.logwarn("Shift goal plane")
                        
                        delta += 0.02
                        new_goal = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc, check=True, delta=delta, current_pc=get_partial_point_cloud(i))
                        if new_goal is not None:
                            goal_pc_numpy = new_goal
                        
                        # goal_pc_numpy = get_goal_plane(constrain_plane=constrain_plane, initial_pc=initial_pc, check=True, delta=delta, current_pc=[])
                        
                        # delta = min(0.06, delta)
                        # shift_plane[3] = constrain_plane[3] + 0.03 + delta
                        # prepare_vis_goal_pc = True
                        # prepare_vis_shift_plane = True
                        
                        num_new_goal += 1
                        print("num_new_goal:", num_new_goal)

                        if goal_pc_numpy is 'success':
                            final_data = {"success": True}
                            with open(os.path.join(data_recording_path, f"plane_{plane_count}.pickle"), 'wb') as handle:
                                pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)                       

                            rospy.logerr("SUCCESS: plane count " + str(plane_count))                            
                            plane_count += 1
                            state = "reset"
                            
                    
                        else:
                            goal_pc = torch.from_numpy(np.swapaxes(goal_pc_numpy,0,1)).float() 
                            execute_count = 0
                            frame_count = 0
                            state = "get shape servo plan" 
                    state = "reset"
                                   

        if state == "reset":   
            # rospy.loginfo("**Current state: " + state)
            rospy.logwarn("==== RESETTING ====")
            frame_count = 0
            sample_count = 0
            execute_count = 0
            num_new_goal = 0
            gym.set_actor_rigid_body_states(envs[i], kuka_handles_2[i], init_robot_state, gymapi.STATE_ALL) 
            gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(saved_object_state))
            print("Sucessfully reset robot and object")           
            state = "home"
 
  


        
        if plane_count >= max_plane_count:             
            all_done = True    

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

