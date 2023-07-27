#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import copy, deepcopy
import rospy
from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
import GraspDataCollectionClient as dc_class
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped
import timeit


# Move robot up for better range of motion
ROBOT_Z_OFFSET = dc_class.ROBOT_Z_OFFSET


# Get current object id, and current object index (in the BB dataset)
BigBird_asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new"
BigBird_meshes_path = "/home/baothach/sim_data/BigBird/BigBird_mesh"
dc = dc_class.GraspDataCollectionClient() 
dc.get_last_object_id_name()
last_object_name = dc.last_object_name  
print("***last object name: " + str(last_object_name))
current_obj_id = dc.cur_object_id
print("***current_obj_id: " + str(current_obj_id))

BB_object_list = sorted(os.listdir(BigBird_meshes_path)) # List (sorted alphabetically) of BigBird dataset object names
if last_object_name == 'empty' or last_object_name == b'empty':
    current_obj_BB_index = 0
else:
    last_object_name = last_object_name.decode('UTF-8')     # Convert bytes literals to string
    current_obj_BB_index = BB_object_list.index(last_object_name) + 1 # index of the last obj name in the BigBird_meshes_path children directories + 1


# Parameters about number of objects and number of grasps per object
num_objects = 1
num_grasps_per_object = 2
num_envs = num_grasps_per_object*num_objects

def check_reach_desired_position(i, j, desired_position, error = 0.01 ):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    
    # absolute(a - b) <= (atol + rtol * absolute(b)) will return True
    # rtol: relative tolerance; atol: absolute tolerance 
    return np.allclose(current_position, desired_position, rtol=0, atol=error)

def get_current_joint_states(i, j):
    current_position = gym.get_actor_dof_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    return list(current_position)



if __name__ == "__main__":

    # initialize gym
    gym = gymapi.acquire_gym()
  

    # parse arguments
    args = gymutil.parse_arguments(
        description="Kuka Bin Test",
        custom_parameters=[
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
            {"name": "--num_objects", "type": int, "default": 10, "help": "Number of objects in the bin"},
            {"name": "--object_type", "type": int, "default": 0, "help": "Type of bjects to place in the bin: 0 - box, 1 - meat can, 2 - banana, 3 - mug, 4 - brick, 5 - random"}])

    # configure sim
    sim_type = args.physics_engine
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 50
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.8
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
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # load robot assets
    asset_root = "../../assets"

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, ROBOT_Z_OFFSET)
    #pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002


    asset_root = "./src/dvrk_env"
    kuka_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"


    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = False
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True

    if sim_type is gymapi.SIM_FLEX:
        asset_options.max_angular_velocity = 40.

    print("Loading asset '%s' from '%s'" % (kuka_asset_file, asset_root))
    kuka_asset = gym.load_asset(sim, asset_root, kuka_asset_file, asset_options)
  

    # create box asset
    box_size = 0.1
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box_pose = gymapi.Transform()

    # Load cube asset
    load_options = gymapi.AssetOptions()
    load_options.fix_base_link = True
    load_options.disable_gravity = True  
    load_options.thickness = 0.005
    asset_root = "../"
    cube_asset_file = "sim_data/BigBird/cube_2.urdf"
    cube_asset = gym.load_asset(sim, asset_root, cube_asset_file, load_options)

    
    # Load soft objects' assets
    asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new" # Current directory
    soft_asset_file = "soft_box/soft_box.urdf"

    soft_pose = gymapi.Transform()
    soft_pose.p = gymapi.Vec3(0., 0.4, 0.03)
    # soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
        
    
    # # Load bigbird syrup object assets
    # asset_root = "/home/baothach/sim_data/BigBird/BigBird_urdf_new" 
    # # syrup_asset_file = "aunt_jemima_original_syrup.urdf"
    # syrup_asset_file = "cholula_chipotle_hot_sauce.urdf"

    # syrup_pose = gymapi.Transform()
    # syrup_pose.p = gymapi.Vec3(0, 0.4, 0.0)
    # syrup_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    # asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    # asset_options.thickness = syrup_thickness
    # asset_options.disable_gravity = True
    # syrup_asset = gym.load_asset(sim, asset_root, syrup_asset_file, asset_options)
    
    # set up the env grid
    spacing = 0.75
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # use position drive for all dofs; override default stiffness and damping values
    dof_props = gym.get_asset_dof_properties(kuka_asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(1000.0)
    dof_props["damping"].fill(200.0)
    dof_props["stiffness"][8:].fill(0.5)
    dof_props["damping"][8:].fill(1)


    # get joint limits and ranges for kuka
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']
    ranges = upper_limits - lower_limits
    mids = 0.5 * (upper_limits + lower_limits)
    #num_dofs = len(kuka_dof_props)


    # default dof states and position targets
    num_dofs = gym.get_asset_dof_count(kuka_asset)
    default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
    default_dof_pos = upper_limits

    default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    # cache some common handles for later use
    envs = []
    kuka_handles = []
    object_handles = []
    dc_clients = []



    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_objects):
        object_handles_per_object = []
        kuka_handles_per_object = []
        envs_per_object = []
        dc_clients_per_object = []
        object_name = BB_object_list[current_obj_BB_index + i]  # Find the object name in the BigBird_meshes_path
        object_asset_file = object_name + ".urdf"
        
        for j in range(num_grasps_per_object):
            # create env
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            envs_per_object.append(env)                 

            # add dvrk
            kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i+j, 1, segmentationId=11)
            kuka_handles_per_object.append(kuka_handle)

            # Set up data collection client
            dc_client = dc_class.GraspDataCollectionClient()
            dc_client.cur_object_id = current_obj_id + i
            dc_client.object_name = object_name
            dc_client.grasp_id = j
            dc_clients_per_object.append(dc_client)

            # Set random pose for the object
            object_pose_stamped = dc_client.gen_object_pose()
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(object_pose_stamped.pose.position.x,object_pose_stamped.pose.position.y,object_pose_stamped.pose.position.z)
            object_pose.r = gymapi.Quat(object_pose_stamped.pose.orientation.x,object_pose_stamped.pose.orientation.y,\
                                            object_pose_stamped.pose.orientation.z,object_pose_stamped.pose.orientation.w)  
            print("object %d, %d intial pose:" %(i,j), object_pose_stamped)                                         
            # object_pose = gymapi.Transform()
            # object_pose.p = gymapi.Vec3(0, 0.4, 0.0)  
            # object_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)    
      
            
            # Load and add object actor
            object_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.thickness = object_thickness
            asset_options.disable_gravity = True
            object_asset = gym.load_asset(sim, BigBird_asset_root, object_asset_file, asset_options)
            soft_actor = gym.create_actor(env, object_asset, object_pose, "object " + str(i), i+j, 0)
            object_handles_per_object.append(soft_actor)     
            
            

        object_handles.append(object_handles_per_object)
        kuka_handles.append(kuka_handles_per_object)
        envs.append(envs_per_object)
        dc_clients.append(dc_clients_per_object)
        
        # # Add object:
        # cube_handle = gym.create_actor(env, cube_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.3, 0.1)), 'cube', i, 0)
        # object_handles.append(cube_handle)

        # # add soft obj        
        # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0, segmentationId=11)
        # object_handles.append(soft_actor)

        # # Test add bigbird obj        
        # syrup_actor = gym.create_actor(env, syrup_asset, syrup_pose, "soft", i, 0, segmentationId=11)
        # object_handles.append(syrup_actor)


    # Camera setup
    cam_pos = gymapi.Vec3(1, 0.5, 1)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    # middle_env = envs[num_objects // 2][num_grasps_per_object // 2]
    middle_env = envs[0][0]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 700
    cam_height = 700
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(0.5, 1.0, 0.5))
    cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))
    # cam_positions.append(gymapi.Vec3(-0.5, 1.0, 0.5))
    # cam_targets.append(gymapi.Vec3(0.0, 0.4, 0.0))    

    for i in range(num_objects):
        cam_handles_per_object = []
        for j in range(num_grasps_per_object):
            cam_handle = gym.create_camera_sensor(envs[i][j], cam_props)
            cam_handles_per_object.append(cam_handle)
            gym.set_camera_location(cam_handle, envs[i][j], cam_positions[0], cam_targets[0])

        cam_handles.append(cam_handles_per_object)

    # # Visual camera:
    # visual_cam_handles = []
    # visual_cam_width = 3000
    # visual_cam_height = 3000
    # visual_cam_props = gymapi.CameraProperties()
    # visual_cam_props.width = visual_cam_width
    # visual_cam_props.height = visual_cam_height
    # visual_cam_position = gymapi.Vec3(0.5, 0.7, 0.2)
    # visual_cam_target = gymapi.Vec3(0.0, 0.4, 0.0)

    # for i in range(num_objects):
    #     visual_cam_handles_per_object = []
    #     for j in range(num_grasps_per_object):
    #         visual_cam_handle = gym.create_camera_sensor(envs[i][j], visual_cam_props)
    #         visual_cam_handles_per_object.append(visual_cam_handle)
    #         gym.set_camera_location(visual_cam_handle, envs[i][j], visual_cam_position, visual_cam_target)
            
    #     visual_cam_handles.append(visual_cam_handles_per_object)

    # set dof properties
    for i in range(num_objects):
        for j in range(num_grasps_per_object):
            gym.set_actor_dof_properties(envs[i][j], kuka_handles[i][j], dof_props)




    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')    
    all_done = False    

    # get_last_object_id_name() and object name *** NEED FIX
    
    
    start_time = timeit.default_timer()
    while (not gym.query_viewer_has_closed(viewer)) and (not all_done):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    
        
        # Run sart state machine
        for i in range(num_objects):
            for j in range(num_grasps_per_object):
            
                if dc_clients[i][j].state == "done":
                    rospy.loginfo("object " + str(i) + ", grasp " + str(j) + " is done !!!" )                    
                    continue

                if dc_clients[i][j].state == "home":
                    rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                    rospy.loginfo('Object_id: %s' %str(dc_clients[i][j].cur_object_id))
                    rospy.loginfo('Grasp_id: %s' %str(dc_clients[i][j].grasp_id))

                    # Robot go home!
                    pos_targets = np.array([0.,0.,0.,0.,0.042,0.,0.,0.,0.,0.], dtype=np.float32)
                    gym.set_actor_dof_position_targets(envs[i][j], kuka_handles[i][j], pos_targets) 
                    if check_reach_desired_position(i, j, pos_targets):
                        dc_clients[i][j].state = "get point cloud"
                        dc_clients[i][j].frame_count = 0
               
                

                if dc_clients[i][j].state == "get point cloud":
                    dc_clients[i][j].frame_count += 1
                    if dc_clients[i][j].frame_count == 60:
                        rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                        # Array of RGB Colors, one per camera, for dots in the resulting
                        # point cloud. Points will have a color which indicates which camera's
                        # depth image created the point.
                        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

                        # Render all of the image sensors only when we need their output here
                        # rather than every frame.
                        gym.render_all_camera_sensors(sim)

                        points = []
                        color = []
                        print("Converting Depth images to point clouds. Have patience...")
                        # for c in range(len(cam_handles)):
                        
                        print("Deprojecting from camera %d, %d" % (i, j))
                        # Retrieve depth and segmentation buffer
                        depth_buffer = gym.get_camera_image(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_DEPTH)
                        seg_buffer = gym.get_camera_image(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_SEGMENTATION)

                        # Get the camera view matrix and invert it to transform points from camera to world
                        # space
                        vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs[0][0], cam_handles[0][0])))

                        # Get the camera projection matrix and get the necessary scaling
                        # coefficients for deprojection
                        proj = gym.get_camera_proj_matrix(sim, envs[i][j], cam_handles[i][j])
                        fu = 2/proj[0, 0]
                        fv = 2/proj[1, 1]

                        # Ignore any points which originate from ground plane or empty space
                        depth_buffer[seg_buffer == 1] = -10001

                        centerU = cam_width/2
                        centerV = cam_height/2
                        for k in range(cam_width):
                            for t in range(cam_height):
                                if depth_buffer[t, k] < -3:
                                    continue
                                if seg_buffer[t, k] == 0:
                                    u = -(k-centerU)/(cam_width)  # image-space coordinate
                                    v = (t-centerV)/(cam_height)  # image-space coordinate
                                    d = depth_buffer[t, k]  # depth buffer value
                                    X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                                    p2 = X2*vinv  # Inverse camera view to get world coordinates
                                    if p2[0, 2] > 0.005:
                                        points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                                        color.append(0)
                        
                        dc_clients[i][j].point_cloud = points
                        dc_clients[i][j].frame_count = 0
                        # v = pptk.viewer(points, color)
                        # v.color_map(color_map)
                        # # Sets a similar view to the gym viewer in the PPTK viewer
                        # v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0)
                        

                        
                        # pcd = open3d.geometry.PointCloud()
                        # pcd.points = open3d.utility.Vector3dVector(np.array(points))
                        # obb = pcd.get_oriented_bounding_box()
                        # print(obb.R)

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
                        # item = [x[0] for x in points]
                        # print(max(item))
                        # print(min(item))
                        # item = [x[1] for x in points]
                        # print(max(item))                
                        # item = [x[2] for x in points]
                        # print(max(item))    

                        dc_clients[i][j].state = "generate preshape"

                if dc_clients[i][j].state == "generate preshape":
                    rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(dc_clients[i][j].point_cloud))
                    
                    # Save scene cloud, RGB image, and depth image
                    open3d.io.write_point_cloud(dc_clients[i][j].get_scene_cloud_save_path(lift=False), pcd) # save_grasp_visual_data , point cloud of the object
                    gym.write_camera_image_to_file(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_COLOR, dc_clients[i][j].get_rgb_image_save_path(lift=False)) # Need fix, choose the right camera
                    gym.write_camera_image_to_file(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_DEPTH, dc_clients[i][j].get_depth_image_save_path(lift=False)) # Need fix, choose the right camera


                    pc_ros_msg = o3dpc_GO.o3dpc_to_GraspObject_msg(pcd)  # point cloud with GraspObject msg format
                    # print("bounding box pose: ", pc_ros_msg.pose)
                    # print("bounding box height: ", pc_ros_msg.height)
                    # print("bounding box width: ", pc_ros_msg.width)
                    # print("bounding box depth: ", pc_ros_msg.depth) 

                    dc_clients[i][j].object_world_seg_pose =  pc_ros_msg.pose # Save obb pose (in Pose() ros format)        
                    
                    cartesian_goal = None
                    preshape_response = dc_clients[i][j].gen_grasp_preshape_client(pc_ros_msg)               
                    for idx in range(len(preshape_response.palm_goal_pose_world)):  # Pick only top grasp
                        if preshape_response.is_top_grasp[idx] == True:
                            cartesian_goal = deepcopy(preshape_response.palm_goal_pose_world[idx].pose) # Need fix
                            dc_clients[i][j].top_grasp_preshape_idx = idx
                    if cartesian_goal == None:
                        dc_clients[i][j].state = "done"
                        rospy.logerr('NO CARTESIAN GOAL.\n') 
                        dc_clients[i][j].failed_grasp_plan = True
                    else:
                        cartesian_goal.position.z -= ROBOT_Z_OFFSET
                                    
                        # # Create MoveIt scene:
                        # dc_clients[i].create_moveit_scene_client(dc_clients[i].object_world_sim_pose, mesh_scaling_factor = 2)

                        # Get plan from MoveIt
                        dc_clients[i][j].plan_traj = dc_clients[i][j].arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_goal, current_position=get_current_joint_states(i, j))
                        
                        # # Clean MoveIt scene:
                        # dc_clients[i].clean_moveit_scene_client()

                        # Does plan exist?
                        if (not dc_clients[i][j].plan_traj):
                            rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                            dc_clients[i][j].state = "done"
                            dc_clients[i][j].failed_grasp_plan = True
                        else:
                            rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                            dc_clients[i][j].state = "move to preshape"
                            rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))

                if dc_clients[i][j].state == "move to preshape":
                    # rospy.loginfo("**Current state: " + dc_clients[i].state)
                    # print(plan_traj)
                    # rospy.loginfo('Moving to this preshape goal' + str(cartesian_goal))
                    
                    plan_traj_with_gripper = [plan+[1.5,1] for plan in dc_clients[i][j].plan_traj]
                    pos_targets = np.array(plan_traj_with_gripper[dc_clients[i][j].traj_index], dtype=np.float32)
                    gym.set_actor_dof_position_targets(envs[i][j], kuka_handles[i][j], pos_targets)        
                    if check_reach_desired_position(i, j, pos_targets):
                        dc_clients[i][j].traj_index += 1             
                    if dc_clients[i][j].traj_index == len(dc_clients[i][j].plan_traj):
                        dc_clients[i][j].traj_index = 0
                        dc_clients[i][j].state = "record true_palm_pose_world"   
                        rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!")


                if dc_clients[i][j].state == "record true_palm_pose_world":
                    dc_clients[i][j].frame_count += 1
                    if dc_clients[i][j].frame_count == 60:   # Wait to get better tool yaw link pose
                        rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                        robot_links_poses = gym.get_actor_rigid_body_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
                        dc_clients[i][j].true_palm_pose_world = to_PoseStamped(robot_links_poses[-3]) # save 'tool yaw link' pose in cartesian (before grasp)
                        dc_clients[i][j].frame_count = 0
                        dc_clients[i][j].state = "grasp object" 
                        print("true_palm_pose_world: ", dc_clients[i][j].true_palm_pose_world)

                if dc_clients[i][j].state == "grasp object":
                    dc_clients[i][j].frame_count += 1
                    rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                    gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_gripper1_joint"), 0.5)
                    gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_gripper2_joint"), -0.2)                      
                    dof_states = gym.get_actor_dof_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
                    grip_1_vel = gym.get_joint_velocity(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_gripper1_joint"))
                    grip_2_vel = gym.get_joint_velocity(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_gripper2_joint"))
                    print("grippers' velocity: %f and %f " %(grip_1_vel, grip_2_vel))
                    if dc_clients[i][j].frame_count >= 2:  #ignore initial velocities which are close to 0
                        if np.allclose(dof_states['pos'][8:], [0.5, -0.2], rtol=0, atol=0.04) or (grip_1_vel > -0.005 and grip_2_vel > -0.005):
                            dc_clients[i][j].state = "record close_palm_pose_world"
                            dc_clients[i][j].get_lift_moveit_plan = True
                            dc_clients[i][j].frame_count = 0
                        

                if dc_clients[i][j].state == "record close_palm_pose_world":
                    dc_clients[i][j].frame_count += 1
                    if dc_clients[i][j].frame_count == 50:   # Wait to get better tool yaw link pose   
                        rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                        robot_links_poses = gym.get_actor_rigid_body_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
                        dc_clients[i][j].close_palm_pose_world = to_PoseStamped(robot_links_poses[-3]) # save 'tool yaw link' pose in cartesian (already grasped and holding the object)                   
                        dc_clients[i][j].state = "lift object"
                        dc_clients[i][j].frame_count = 0
                        print("close_palm_pose_world: ", dc_clients[i][j].close_palm_pose_world)

                if dc_clients[i][j].state == "lift object":  
                    if dc_clients[i][j].get_lift_moveit_plan:                    
                        rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                        
                        dc_clients[i][j].plan_traj = dc_clients[i][j].lift_moveit_planner_client(current_position=get_current_joint_states(i, j))                            
                        if (not dc_clients[i][j].plan_traj):
                            rospy.logerr('Can not find moveit plan to LIFT. Ignore this LIFT.\n')  
                            dc_clients[i][j].state = "done" 
                            dc_clients[i][j].failed_grasp_plan = True
                        else:
                            rospy.loginfo('Sucesfully found a LIFT moveit plan to grasp.\n')
                            dc_clients[i][j].get_lift_moveit_plan = False
                    else:
                        pos_targets = np.array(dc_clients[i][j].plan_traj[dc_clients[i][j].traj_index], dtype=np.float32)       
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_yaw_joint"), pos_targets[0])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_pitch_back_joint"), pos_targets[1])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_pitch_bottom_joint"), pos_targets[2])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_pitch_end_joint"), pos_targets[3])                     
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_main_insertion_joint"), pos_targets[4])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_roll_joint"), pos_targets[5])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_pitch_joint"), pos_targets[6])
                        gym.set_joint_target_position(envs[i][j], gym.get_joint_handle(envs[i][j], "kuka", "psm_tool_yaw_joint"), pos_targets[7])
                    
                        
                        dof_states = gym.get_actor_dof_states(envs[i][j], kuka_handles[i][j], gymapi.STATE_POS)
                        if np.allclose(dof_states['pos'][:8], pos_targets, rtol=0, atol=0.01):
                            dc_clients[i][j].traj_index += 1     

                        if dc_clients[i][j].traj_index == len(dc_clients[i][j].plan_traj):
                            dc_clients[i][j].state = "record all grasp data"   
                            rospy.loginfo("Succesfully executed LIFT moveit plan. Done!")
                

                if dc_clients[i][j].state == "record all grasp data":
                    dc_clients[i][j].frame_count += 1
                    if dc_clients[i][j].frame_count == 40:                
                        rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                        
                        # Get grasp label 
                        dc_clients[i][j].grasp_label = 0
                        z_values = [x[2] for x in dc_clients[i][j].point_cloud]
                        dc_clients[i][j].height_before_lift = max(z_values)
                        gym.render_all_camera_sensors(sim)

                        points = []
                        print("Converting Depth images to point clouds. Have patience...")
                        # for c in range(len(cam_handles)):

                        print("Deprojecting from camera %d, %d" % (i, j))
                        # Retrieve depth and segmentation buffer
                        depth_buffer = gym.get_camera_image(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_DEPTH)
                        seg_buffer = gym.get_camera_image(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_SEGMENTATION)

                        # Get the camera view matrix and invert it to transform points from camera to world
                        # space
                        vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs[0][0], cam_handles[0][0])))

                        # Get the camera projection matrix and get the necessary scaling
                        # coefficients for deprojection
                        proj = gym.get_camera_proj_matrix(sim, envs[i][j], cam_handles[i][j])
                        fu = 2/proj[0, 0]
                        fv = 2/proj[1, 1]

                        # Ignore any points which originate from ground plane or empty space
                        depth_buffer[seg_buffer == 1] = -10001

                        centerU = cam_width/2
                        centerV = cam_height/2
                        for k in range(cam_width):
                            if dc_clients[i][j].grasp_label == 1:
                                break                            
                            for t in range(cam_height):

                                if depth_buffer[t, k] < -3:
                                    continue
                                if seg_buffer[t, k] == 0:
                                    u = -(k-centerU)/(cam_width)  # image-space coordinate
                                    v = (t-centerV)/(cam_height)  # image-space coordinate
                                    d = depth_buffer[t, k]  # depth buffer value
                                    X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                                    p2 = X2*vinv  # Inverse camera view to get world coordinates
                                    if p2[0, 2] > dc_clients[i][j].height_before_lift + 0.03:
                                        dc_clients[i][j].grasp_label = 1
                                        break        

                        print("This grasp is successful: ", dc_clients[i][j].grasp_label)                                  
                        dc_clients[i][j].record_grasp_data_client()    # Need fix grasp_preshape_idx
                        print("object " + str(i) + ", grasp " + str(j) + " has been RECORDED!" ) 
                        dc_clients[i][j].state = "save lift visual data"
                        dc_clients[i][j].frame_count = 0
                
                if dc_clients[i][j].state == "save lift visual data":
                    rospy.loginfo("**Current state: " + dc_clients[i][j].state)
                    gym.write_camera_image_to_file(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_COLOR, dc_clients[i][j].get_rgb_image_save_path(lift=True)) # Need fix, choose the right camera
                    gym.write_camera_image_to_file(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_DEPTH, dc_clients[i][j].get_depth_image_save_path(lift=True)) # Need fix, choose the right camera  
                    # Save RGB image into the right folder (/suc_grasps or /fail_grasps)
                    gym.write_camera_image_to_file(sim, envs[i][j], cam_handles[i][j], gymapi.IMAGE_COLOR, dc_clients[i][j].get_rgb_image_save_path_with_label()) # Need fix, choose the right camera                  
                    dc_clients[i][j].state = "done"                   

                
                all_done = all(dc_clients[x][y].state == "done" for x in range(num_objects) for y in range(num_grasps_per_object)) 
                  
            # step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)


    print("All done !")
    print("Elapsed time", timeit.default_timer() - start_time)
    print("-----------------------------------------")
    print("SUMMARY 1: ")
    print("-----------------------------------------")
    count = 0
    for i in range(num_objects):
        for j in range(num_grasps_per_object):
            if  dc_clients[i][j].state != "done":
                count += 1
                print("object " + str(i) + ", grasp " + str(j) + " is NOT DONE !!!" )  
    print("TOTAL:", count) 
    print("-----------------------------------------")
    print("SUMMARY 2: ")
    print("-----------------------------------------")
    count = 0
    for i in range(num_objects):
        for j in range(num_grasps_per_object):
            if  dc_clients[i][j].failed_grasp_plan == True:
                print("object " + str(i) + ", grasp " + str(j) + " FAILED PLAN" )  
                count += 1
    print("TOTAL:", count)

    print("-----------------------------------------")
    print("SUMMARY 3: ")
    print("-----------------------------------------")
    count = 0
    for i in range(num_objects):
        for j in range(num_grasps_per_object):
            if  dc_clients[i][j].grasp_label == 1:
                print("object " + str(i) + ", grasp " + str(j) + " SUCCESSFUL" )  
                count += 1
    print("TOTAL:", count)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

