#!/usr/bin/env python3
from __future__ import print_function, division, absolute_import

import os
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from copy import copy
import rospy
from dvrk_gazebo_control.srv import *
from geometry_msgs.msg import PoseStamped, Pose
import GraspDataCollectionClient as dc_class
import open3d
from utils import open3d_ros_helper as orh
from utils import o3dpc_to_GraspObject_msg as o3dpc_GO
#import pptk
from utils.isaac_utils import isaac_format_pose_to_PoseStamped as to_PoseStamped

ROBOT_Z_OFFSET = 0.3

def arm_moveit_planner_client(go_home=False, place_goal_pose=None, cartesian_pose=None):
    #bao123 
    '''
    return Is there any plan?
    calculate plan and assign to self.planning_response
    '''
    
    rospy.loginfo('Waiting for service moveit_cartesian_pose_planner.')
    rospy.wait_for_service('moveit_cartesian_pose_planner')
    rospy.loginfo('Calling service moveit_cartesian_pose_planner.')
    try:
        planning_proxy = rospy.ServiceProxy('moveit_cartesian_pose_planner', PalmGoalPoseWorld)
        planning_request = PalmGoalPoseWorldRequest()
        if go_home:
            planning_request.go_home = True
        elif place_goal_pose is not None:
            planning_request.palm_goal_pose_world = place_goal_pose
        else:
            # planning_request.palm_goal_pose_world = self.mount_desired_world.pose
            planning_request.palm_goal_pose_world = cartesian_pose
        planning_response = planning_proxy(planning_request) 
    except (rospy.ServiceException):
        rospy.loginfo('Service moveit_cartesian_pose_planner call failed')
    rospy.loginfo('Service moveit_cartesian_pose_planner is executed %s.'
            %str(planning_response.success))

    if not planning_response.success:
        rospy.loginfo('Does not have a plan to execute!')
    # print("-------------", planning_response.plan_traj)
    # Convert JointTrajectory message to list    
    plan_list = []
    for point in planning_response.plan_traj.points:
        plan_list.append(list(point.positions))
    return plan_list

def check_reach_desired_position(i, desired_position):
    '''
    Check if the robot has reached the desired goal positions
    '''
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
    current_position = [x[0] for x in current_position]
    
    # absolute(a - b) <= (atol + rtol * absolute(b)) will return True
    # rtol: relative tolerance; atol: absolute tolerance 
    return np.allclose(current_position, desired_position, rtol=0, atol=0.01)

def get_current_joint_states(i):
    current_position = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
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

    num_envs = args.num_envs


    # configure sim
    sim_type = args.physics_engine
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    if sim_type is gymapi.SIM_FLEX:
        sim_params.substeps = 4
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
        sim_params.flex.relaxation = 0.7
        sim_params.flex.warm_start = 0.1
        sim_params.flex.shape_collision_distance = 5e-6
        sim_params.flex.contact_regularization = 1.0e-6
        sim_params.flex.shape_collision_margin = 1.0e-6
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
    soft_pose.p = gymapi.Vec3(0, 0.3, 0.03)
    soft_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    soft_thickness = 0.005    # important to add some thickness to the soft body to avoid interpenetrations

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = soft_thickness
    asset_options.disable_gravity = True
    # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
    soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)
        
    # set up the env grid
    spacing = 1.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # use position drive for all dofs; override default stiffness and damping values
    dof_props = gym.get_asset_dof_properties(kuka_asset)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(1000.0)
    dof_props["damping"].fill(200.0)
    dof_props["stiffness"][8:].fill(1)
    dof_props["damping"][8:].fill(2)


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
    






    print("Creating %d environments" % num_envs)
    num_per_row = int(math.sqrt(num_envs))
    base_poses = []

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)


        # add kuka
        kuka_handle = gym.create_actor(env, kuka_asset, pose, "kuka", i, 1, segmentationId=11)
        
        # Add object:
        cube_handle = gym.create_actor(env, cube_asset, gymapi.Transform(p=gymapi.Vec3(0, 0.3, 0.1)), 'cube', i, 0)
        object_handles.append(cube_handle)

        # # add soft obj        
        # soft_actor = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 0, segmentationId=11)
        # object_handles.append(soft_actor)


        kuka_handles.append(kuka_handle)

    # Camera setup
    cam_pos = gymapi.Vec3(1, 0.5, 1)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.1)
    middle_env = envs[num_envs // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    # Camera for point cloud setup
    cam_positions = []
    cam_targets = []
    cam_handles = []
    cam_width = 600
    cam_height = 600
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height
    cam_positions.append(gymapi.Vec3(1, 1, 1))
    cam_targets.append(gymapi.Vec3(0.0, 0.3, 0.0))
    cam_positions.append(gymapi.Vec3(-1, 1, 1))
    cam_targets.append(gymapi.Vec3(0.0, 0.3, 0.0))    

    for env in envs:
        for c in range(len(cam_positions)):
            cam_handles.append(gym.create_camera_sensor(env, cam_props))
            gym.set_camera_location(cam_handles[c], env, cam_positions[c], cam_targets[c])

    # set dof properties
    for env in envs:
        gym.set_actor_dof_properties(env, kuka_handles[i], dof_props)





    '''
    Main stuff is here
    '''
    rospy.init_node('isaac_grasp_client')
    dc_clients = []
    grasp_ids = []
    grasp_plan_failures_nums = []
    for i in range(num_envs): 
        grasp_ids.append(0)
        grasp_plan_failures_nums.append(0)
        dc_client = dc_class.GraspDataCollectionClient()
        dc_clients.append(dc_client)
    
    all_done = False    
    grasp_failure_retry_times = dc_clients[0].num_grasps_per_object

    # get_last_object_id_name() and object name *** NEED FIX
    dc_clients[0].cur_object_id = 0
    dc_clients[0].object_name = "custom_box"
    
    while (not gym.query_viewer_has_closed(viewer)) and (not all_done):

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    
        
        # # Run sart state machine
        for i in range(num_envs):  
            
            if dc_clients[i].state == "done":
                continue

            if dc_clients[i].state == "home":
                rospy.loginfo("**Current state: " + dc_clients[i].state)
                rospy.loginfo('Grasp_id: %s' %str(grasp_ids[i]))

                # Robot go home!
                pos_targets = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets) 
                if check_reach_desired_position(i, pos_targets):
                    dc_clients[i].state = "get point cloud"
                    dc_clients[i].frame_count = 0

                    # Set random pose for the object
                    object_pose_stamped = dc_clients[i].gen_object_pose()
                    state = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_NONE)    
                    state['pose']['p'].fill((object_pose_stamped.pose.position.x,object_pose_stamped.pose.position.y,object_pose_stamped.pose.position.z))
                    state['pose']['r'].fill((object_pose_stamped.pose.orientation.x,object_pose_stamped.pose.orientation.y,\
                                            object_pose_stamped.pose.orientation.z,object_pose_stamped.pose.orientation.w))
                    gym.set_actor_rigid_body_states(envs[i], object_handles[i], state, gymapi.STATE_ALL) 

                    
                    # print(gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS))
                    # print("object state: ", gym.get_rigid_body_segmentation_id(env,object_handles[i], 0) )

            # if dc_clients[i].state == "set up parameters":
            #     # dc_client.set_up_object_name(object_name=object_name, object_mesh_path=None) # Need fix object_mesh_path
            #     dc_clients[i].set_up_grasp_id(grasp_ids[i])
            #     dc_clients[i].state = "get point cloud"

            if dc_clients[i].state == "get point cloud":
                dc_clients[i].frame_count += 1
                if dc_clients[i].frame_count == 2:
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
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
                    for c in range(len(cam_handles)):
                        print("Deprojecting from camera %d" % c)
                        # Retrieve depth and segmentation buffer
                        depth_buffer = gym.get_camera_image(sim, envs[i], cam_handles[c], gymapi.IMAGE_DEPTH)
                        seg_buffer = gym.get_camera_image(sim, envs[i], cam_handles[c], gymapi.IMAGE_SEGMENTATION)

                        # Get the camera view matrix and invert it to transform points from camera to world
                        # space
                        vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs[i], cam_handles[c])))

                        # Get the camera projection matrix and get the necessary scaling
                        # coefficients for deprojection
                        proj = gym.get_camera_proj_matrix(sim, envs[i], cam_handles[c])
                        fu = 2/proj[0, 0]
                        fv = 2/proj[1, 1]

                        # Ignore any points which originate from ground plane or empty space
                        depth_buffer[seg_buffer == 1] = -10001

                        centerU = cam_width/2
                        centerV = cam_height/2
                        for k in range(cam_width):
                            for j in range(cam_height):
                                if depth_buffer[j, k] < -3:
                                    continue
                                if seg_buffer[j, k] == 0:
                                    u = -(k-centerU)/(cam_width)  # image-space coordinate
                                    v = (j-centerV)/(cam_height)  # image-space coordinate
                                    d = depth_buffer[j, k]  # depth buffer value
                                    X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                                    p2 = X2*vinv  # Inverse camera view to get world coordinates
                                    if p2[0, 2] > 0.005:
                                        points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                                        color.append(c)
                    # # print("point cloud data: ", points)
                    dc_clients[i].point_cloud = points
                    dc_clients[i].frame_count = 0
                    v = pptk.viewer(points, color)
                    v.color_map(color_map)
                    # Sets a similar view to the gym viewer in the PPTK viewer
                    v.set(lookat=[0, 0, 0], r=5, theta=0.4, phi=0)

                    
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

                    dc_clients[i].state = "generate preshape"

            if dc_clients[i].state == "generate preshape":
                rospy.loginfo("**Current state: " + dc_clients[i].state)
                pcd = open3d.geometry.PointCloud()
                pcd.points = open3d.utility.Vector3dVector(np.array(dc_clients[i].point_cloud))
                
                # Save scene cloud, RGB image, and depth image
                open3d.io.write_point_cloud(dc_clients[i].get_scene_cloud_save_path(lift=False), pcd) # save_grasp_visual_data , point cloud of the object
                gym.write_camera_image_to_file(sim, envs[i], cam_handles[0], gymapi.IMAGE_COLOR, dc_clients[i].get_rgb_image_save_path(lift=False)) # Need fix, choose the right camera
                gym.write_camera_image_to_file(sim, envs[i], cam_handles[0], gymapi.IMAGE_DEPTH, dc_clients[i].get_depth_image_save_path(lift=False)) # Need fix, choose the right camera


                pc_ros_msg = o3dpc_GO.o3dpc_to_GraspObject_msg(pcd)  # point cloud with GraspObject msg format
                print("bounding box pose: ", pc_ros_msg.pose)
                print("bounding box height: ", pc_ros_msg.height)
                print("bounding box width: ", pc_ros_msg.width)
                print("bounding box depth: ", pc_ros_msg.depth) 

                dc_clients[i].object_world_seg_pose =  pc_ros_msg.pose # Save obb pose (in Pose() ros format)        
                
                preshape_response = dc_clients[i].gen_grasp_preshape_client(pc_ros_msg)                
                for idx in range(len(preshape_response.palm_goal_pose_world)):  # Pick only top grasp
                    if preshape_response.is_top_grasp[idx] == True:
                        cartesian_goal = preshape_response.palm_goal_pose_world[idx].pose # Need fix
                        dc_clients[i].top_grasp_preshape_idx = idx
                
                cartesian_goal.position.z -= ROBOT_Z_OFFSET
                
                plan_traj = dc_clients[i].arm_moveit_planner_client(go_home=False, cartesian_goal=cartesian_goal, current_position=get_current_joint_states(i))
                if (not plan_traj):
                    rospy.logerr('Can not find moveit plan to grasp. Ignore this grasp.\n')  
                    dc_clients[i].grasp_plan_failures_num += 1 
                    dc_clients[i].state = "home"
                else:
                    rospy.loginfo('Sucesfully found a PRESHAPE moveit plan to grasp.\n')
                    dc_clients[i].state = "move to preshape"
                    traj_index = 0
                    rospy.loginfo('Moving to this preshape goal: ' + str(cartesian_goal))

            if dc_clients[i].state == "move to preshape":
                # rospy.loginfo("**Current state: " + dc_clients[i].state)
                # print(plan_traj)
                # rospy.loginfo('Moving to this preshape goal' + str(cartesian_goal))
                plan_traj_with_gripper = [plan+[1,1] for plan in plan_traj]
                pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)        
                if check_reach_desired_position(i, pos_targets):
                    traj_index += 1             
                if traj_index == len(plan_traj):
                    dc_clients[i].state = "record true_palm_pose_world"   
                    rospy.loginfo("Succesfully executed PRESHAPE moveit arm plan. Let's fucking grasp it!!")


            if dc_clients[i].state == "record true_palm_pose_world":
                dc_clients[i].frame_count += 1
                if dc_clients[i].frame_count == 60:   # Wait to get better tool yaw link pose
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
                    robot_links_poses = gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
                    dc_clients[i].true_palm_pose_world = to_PoseStamped(robot_links_poses[-3]) # save 'tool yaw link' pose in cartesian (before grasp)
                    dc_clients[i].frame_count = 0
                    dc_clients[i].state = "grasp object" 
                    print("true_palm_pose_world: ", dc_clients[i].true_palm_pose_world)

            if dc_clients[i].state == "grasp object":
                rospy.loginfo("**Current state: " + dc_clients[i].state)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper1_joint"), 0.5)
                gym.set_joint_target_position(envs[i], gym.get_joint_handle(envs[i], "kuka", "psm_tool_gripper2_joint"), -0.2) 
                gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)                     
                dof_states = gym.get_actor_dof_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
                if np.allclose(dof_states['pos'][8:], [0.5, -0.2], rtol=0, atol=0.05):
                    dc_clients[i].state = "record close_palm_pose_world"
                    dc_clients[i].get_lift_moveit_plan = True
                        

            if dc_clients[i].state == "record close_palm_pose_world":
                dc_clients[i].frame_count += 1
                if dc_clients[i].frame_count == 50:   # Wait to get better tool yaw link pose   
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
                    robot_links_poses = gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
                    dc_clients[i].close_palm_pose_world = to_PoseStamped(robot_links_poses[-3]) # save 'tool yaw link' pose in cartesian (already grasped and holding the object)                   
                    dc_clients[i].state = "lift object"
                    dc_clients[i].frame_count = 0
                    print("close_palm_pose_world: ", dc_clients[i].close_palm_pose_world)

            if dc_clients[i].state == "lift object":                            
                if dc_clients[i].get_lift_moveit_plan:
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
                    plan_traj = dc_clients[i].lift_moveit_planner_client(current_position=get_current_joint_states(i))                            
                    if (not plan_traj):
                        rospy.logerr('Can not find moveit plan to LIFT. Ignore this LIFT.\n')  
                        dc_clients[i].grasp_plan_failures_num += 1 
                        dc_clients[i].state = "home" 
                    else:
                        rospy.loginfo('Sucesfully found a LIFT moveit plan to grasp.\n')
                        traj_index = 0
                        dc_clients[i].get_lift_moveit_plan = False
                else:
                    plan_traj_with_gripper = [plan+[0.5, -0.2] for plan in plan_traj]
                    pos_targets = np.array(plan_traj_with_gripper[traj_index], dtype=np.float32)
                    gym.set_actor_dof_position_targets(envs[i], kuka_handles[i], pos_targets)        
                    if check_reach_desired_position(i, pos_targets):
                        traj_index += 1             
                    if traj_index == len(plan_traj):
                        dc_clients[i].state = "record lift_palm_pose_world"   
                        rospy.loginfo("Succesfully executed LIFT moveit plan. Done!")
            
            if dc_clients[i].state == "record lift_palm_pose_world":
                dc_clients[i].frame_count += 1
                if dc_clients[i].frame_count == 50:   # Wait to get better tool yaw link pose   
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
                    robot_links_poses = gym.get_actor_rigid_body_states(envs[i], kuka_handles[i], gymapi.STATE_POS)
                    dc_clients[i].lift_palm_pose_world = to_PoseStamped(robot_links_poses[-3]) # save 'tool yaw link' pose in cartesian (already grasped and holding the object)                   
                    dc_clients[i].state = "record all grasp data"
                    dc_clients[i].frame_count = 0
                    print("lift_palm_pose_world: ", dc_clients[i].lift_palm_pose_world)

            if dc_clients[i].state == "record all grasp data":
                rospy.loginfo("**Current state: " + dc_clients[i].state)
                object_pose = gym.get_actor_rigid_body_states(envs[i], object_handles[i], gymapi.STATE_POS)
                dc_clients[i].grasp_label = 1 if object_pose["pose"]["p"]["z"] >= dc_clients[i].grasp_success_object_height else 0
                dc_clients[i].record_grasp_data_client()    # Need fix grasp_preshape_idx
                dc_clients[i].state = "save lift visual data"
            
            if dc_clients[i].state == "save lift visual data":
                dc_clients[i].frame_count += 1
                if dc_clients[i].frame_count == 2:
                    rospy.loginfo("**Current state: " + dc_clients[i].state)
                    color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])
                    gym.render_all_camera_sensors(sim)

                    points = []
                    color = []
                    print("Converting Depth images to point clouds. Have patience...")
                    for c in range(len(cam_handles)):
                        print("Deprojecting from camera %d" % c)
                        # Retrieve depth and segmentation buffer
                        depth_buffer = gym.get_camera_image(sim, envs[i], cam_handles[c], gymapi.IMAGE_DEPTH)
                        seg_buffer = gym.get_camera_image(sim, envs[i], cam_handles[c], gymapi.IMAGE_SEGMENTATION)

                        # Get the camera view matrix and invert it to transform points from camera to world
                        # space
                        vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, envs[i], cam_handles[c])))

                        # Get the camera projection matrix and get the necessary scaling
                        # coefficients for deprojection
                        proj = gym.get_camera_proj_matrix(sim, envs[i], cam_handles[c])
                        fu = 2/proj[0, 0]
                        fv = 2/proj[1, 1]

                        # Ignore any points which originate from ground plane or empty space
                        depth_buffer[seg_buffer == 0] = -10001

                        centerU = cam_width/2
                        centerV = cam_height/2
                        for k in range(cam_width):
                            for j in range(cam_height):
                                if depth_buffer[j, k] < -10000:
                                    continue
                                if seg_buffer[j, k] > 0:
                                    u = -(k-centerU)/(cam_width)  # image-space coordinate
                                    v = (j-centerV)/(cam_height)  # image-space coordinate
                                    d = depth_buffer[j, k]  # depth buffer value
                                    X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                                    p2 = X2*vinv  # Inverse camera view to get world coordinates
                                    points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                                    color.append(c)
                    # Save scene cloud, RGB image, and depth image (AFTER LIFT)
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(points)                    
                    open3d.io.write_point_cloud(dc_clients[i].get_scene_cloud_save_path(lift=True), pcd) # save_grasp_visual_data , point cloud of the object
                    gym.write_camera_image_to_file(sim, envs[i], cam_handles[0], gymapi.IMAGE_COLOR, dc_clients[i].get_rgb_image_save_path(lift=True)) # Need fix, choose the right camera
                    gym.write_camera_image_to_file(sim, envs[i], cam_handles[0], gymapi.IMAGE_DEPTH, dc_clients[i].get_depth_image_save_path(lift=True)) # Need fix, choose the right camera                    
                    dc_clients[i].grasp_id += 1
                    dc_clients[i].state = "home"    


            # Determine whether everything is done and it's time to exit out:
            if dc_clients[i].grasp_id >= dc_clients[i].num_grasps_per_object or dc_clients[i].grasp_plan_failures_num >= dc_clients[i].max_grasp_plan_failures_num:
                rospy.loginfo("reach max number of runs")
                dc_clients[i].state = "done"
                rospy.loginfo("env " + str(i) + " is done !!!" )             
            
            all_done = all(dc_clients[i].state == "done" for i in range(num_envs))  
                  
            # step rendering
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)



    print("All done !")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

