from isaacgym import gymtorch, gymapi

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import rospy
from copy import deepcopy

from geometry_msgs.msg import Pose, TransformStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from ll4ma_isaacgym.msg import IsaacGymState

from ll4ma_isaacgym.core import EnvironmentState, RobotAction
from ll4ma_isaacgym.core import util as gym_util
from ll4ma_isaacgym.robots import Robot
from ll4ma_util import ui_util, vis_util, ros_util, file_util


class Simulator:
    """
    Encapsulates instance of Isaac Gym simulator, handling environment creation and
    resets, data logging, publishing to ROS, stepping physics, etc.
    """

    def __init__(self, session_config, create_sim=True, reset_wait_steps=50):
        self.config = session_config
        self.reset_wait_steps = reset_wait_steps
        self.gym = gymapi.acquire_gym()

        self.collect_data = (session_config.data_root is not None
                             and len(session_config.data_root) > 0
                             and session_config.n_demos > 0)
        if self.collect_data:
            self.reset_dataset()

        if session_config.publish_ros:
            self.state_pub = rospy.Publisher('/isaacgym_state', IsaacGymState, queue_size=1)

        self.should_log = [True for _ in range(session_config.n_envs)]
        self.timestep = 0

        self.env_states = []
        for _ in range(session_config.n_envs):
            env_state = EnvironmentState()
            env_state.dt = session_config.sim.dt
            env_state.objects = session_config.env.objects
            env_state.object_colors = { k: ColorRGBA()
              for k in session_config.env.objects.keys() }
            self.env_states.append(env_state)


        # TODO for now just assume one camera. This is easy to change just need to make
        # rgb/depth dicts in dataset with sensor_name keys, but it will change data loading
        # so holding off for a moment
        if len(session_config.env.sensors) > 1:
            raise ValueError("Only one camera sensor currently supported")
        self.sensor_config = next(iter(session_config.env.sensors.values()))
        self.rgb_size = None
        if self.config.env.img_size is not None:
            self.rgb_size = self.config.env.img_size

        # TODO for now only one robot
        robot_config = next(iter(session_config.env.robots.values()))
        self.robot = Robot(robot_config)

        self.actions = [
          RobotAction( self.robot.arm.config, self.robot.end_effector.config )
            for _ in range(session_config.n_envs) ]

        if session_config.demo:
            # Initialize the scene the same as this previously recorded demo
            data, attrs = file_util.load_pickle(session_config.demo)
            self.robot.set_init_joint_position(data['joint_position'][0].tolist())
            for obj_name, obj_data in attrs['objects'].items():
                if obj_name in session_config.env.objects:
                    session_config.env.objects[obj_name].from_dict(obj_data)
                    session_config.env.objects[obj_name].rgb_color = obj_data['rgb_color']
                    pos = data['objects'][obj_name]['position'][0]
                    quat = data['objects'][obj_name]['orientation'][0]
                    session_config.env.objects[obj_name].position_x = pos[0]
                    session_config.env.objects[obj_name].position_y = pos[1]
                    session_config.env.objects[obj_name].position_z = pos[2]
                    session_config.env.objects[obj_name].orientation = quat

        if create_sim:
            self.create_sim()

    @property
    def num_dofs(self):
        return self.gym.get_sim_dof_count(self.sim)

    def step(self, post_physics=True):
        """
        One step in the simulation.
        """
        self.gym.simulate(self.sim) # Step physics
        if self.config.device == 'cpu' or self.collect_data or self.config.sim.render_graphics:
            self.gym.fetch_results(self.sim, True)
        if self.config.sim.render_graphics:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.timestep += 1

        # Compute observations, cache data, publish to ROS, etc.
        if post_physics:
            self._post_physics_step()

    def _post_physics_step(self):
        """
        Perform operations after a simulator step has been made. Handles rendering the
        visualizer and caching data in memory if data collection is active.
        """
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, False)
        if self.config.sim.render_graphics:
            self.gym.sync_frame_time(self.sim)
        if self.collect_data:
            self._cache_step_data()
        if self.config.publish_ros:
            self._publish_ros_data()

    def create_sim(self):
        """
        Creates simulation (including all envs) and prepares tensors.
        """
        self.sim = gym_util.get_default_sim(self.config.sim)

        if self.config.sim.render_graphics:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")
        else:
            self.viewer = None

        self.envs = []
        self.seg_id_dict = {}
        self.current_seg_id = 1
        self.obj_assets = {}

        self._create_envs()

        self.gym.prepare_sim(self.sim)  # Use tensor API

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        self.n_joints = self.robot.num_joints()
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view( self.config.n_envs, -1, 1 )
        self.dof_vel = self.dof_states[:, 1].view( self.config.n_envs, -1, 1 )

        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.config.n_envs, -1, 1)

        _fs_data = self.gym.acquire_force_sensor_tensor(self.sim)
        self.fs_data = gymtorch.wrap_tensor(_fs_data)

        # _jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.robot_config.name)
        # self.jacobian = gymtorch.wrap_tensor(_jacobian)
        # print("JAC", self.jacobian.shape)

        self.reset()

    def _create_envs(self):
        """
        Create simulation assets and assign things to each environment.
        """
        self._create_object_assets()
        self._create_robot_assets()

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        n_per_row = int(math.sqrt(self.config.n_envs))
        spacing = self.config.env.spacing
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Reset rigid-body indices in case they were loaded previously from file
        for obj_config in self.config.env.objects.values():
            obj_config.rb_indices = []

        for env_idx in range(self.config.n_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, n_per_row)
            self.envs.append(env)

            for obj_name, obj_config in self.config.env.objects.items():
                if obj_name not in self.seg_id_dict:
                    #print([obj_name, self.current_seg_id])
                    self.seg_id_dict[obj_name] = self.current_seg_id
                    self.current_seg_id += 1
                


                obj_handle = self.gym.create_actor(env, self.obj_assets[obj_name],
                                                   gymapi.Transform(), obj_name, env_idx, 0,
                                                   self.seg_id_dict[obj_name])

                # Get global index of object in rigid body state tensor
                rb_idx = self.gym.get_actor_rigid_body_index(env, obj_handle, 0, gymapi.DOMAIN_SIM)
                obj_config.rb_indices.append(rb_idx)
            self.reset_objects(env_idx)

            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(*self.robot.config.position)
            robot_pose.r = gymapi.Quat(*self.robot.config.orientation)
            robot_handle = self.gym.create_actor(env, self.robot_asset, robot_pose,
                                                 self.robot.config.name, env_idx, 2)
            self.gym.enable_actor_dof_force_sensors(env, robot_handle)
            self.gym.set_actor_dof_properties(env, robot_handle, self.robot_dof_props)

            self.robot_link_names = []
            for link_name, rb_idx in self.gym.get_actor_rigid_body_dict(env, robot_handle).items():
                # Semantic segmentation IDs
                #print([link_name, self.current_seg_id])
                if link_name not in self.seg_id_dict:
                    self.seg_id_dict[link_name] = self.current_seg_id
                    self.current_seg_id += 1
                self.gym.set_rigid_body_segmentation_id(env, robot_handle, rb_idx,
                                                        self.seg_id_dict[link_name])
                self.robot.add_rb_index(link_name, rb_idx) # Track RB indices (e.g. for contact info)
                self.robot_link_names.append(link_name)

            self.robot_joint_names = self.gym.get_actor_dof_names(env, robot_handle)
            # for env_state in self.env_states:
            #     env_state.joint_names = self.robot_joint_names
            self.reset_robots(env_idx)

            if self.robot.has_end_effector():
                ee_link = self.robot.end_effector.get_link()
                ee_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle, ee_link)
                ee_pose = self.gym.get_rigid_transform(env, ee_handle)
                ee_idx = self.gym.find_actor_rigid_body_index(env, robot_handle, ee_link,
                                                              gymapi.DOMAIN_SIM)
                self.robot.end_effector.add_rb_index(ee_idx)

            # TODO force sensors not yet properly tested
            # # TODO can better configure this to be robot agnostic
            # for s in ["left", "right"]:
            #     finger_handle = self.gym.find_actor_rigid_body_handle(env, robot_handle,
            #                                                           "panda_{}finger".format(s))
            #     fs_pose = gymapi.Transform()  # TODO define differently?
            #     fs = self.gym.create_force_sensor(env, finger_handle, fs_pose)

            # TODO will need to modify when there are multiple cameras supported (see __init__)
            if self.sensor_config.sensor_type == 'camera':
                sensor_props = gymapi.CameraProperties()
                sensor_props.width = self.sensor_config.width
                sensor_props.height = self.sensor_config.height
                sensor_handle = self.gym.create_camera_sensor(env, sensor_props)
                sensor_origin = gymapi.Vec3(*self.sensor_config.origin)
                sensor_target = gymapi.Vec3(*self.sensor_config.target)
                self.gym.set_camera_location(sensor_handle, env, sensor_origin, sensor_target)
                self.sensor_config.sim_handle = sensor_handle
            else:
                raise ValueError(f"Unknown sensor type: {self.sensor_config.sensor_type}")

            self.seg_ids = list(self.seg_id_dict.values())
            self.seg_colors = vis_util.random_colors(len(self.seg_ids))

        if self.viewer is not None:
            # Point viewer camera at middle env
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.config.n_envs // 2 + n_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

    def _create_object_assets(self):
        """
        Creates object assets specified in config.
        """
        for obj_name, cfg in self.config.env.objects.items():
            options = gymapi.AssetOptions()
            if cfg.density:
                options.density = cfg.density
            options.fix_base_link = cfg.fix_base_link
            if cfg.object_type == 'box':
                asset = self.gym.create_box(self.sim, cfg.extents[0], cfg.extents[1],
                                            cfg.extents[2], options)
            elif cfg.object_type == 'urdf':
                options.override_com = True
                options.override_inertia = True
                asset_root = ros_util.resolve_ros_package_path(cfg.asset_root)
                asset = self.gym.load_asset(self.sim, asset_root, cfg.asset_filename, options)
            else:
                raise ValueError(f"Unknown object type for simulation: {cfg.object_type}")
            self.obj_assets[obj_name] = asset

        # Track additional data that is env-specific (e.g. object colors) for logging
        obj_names = self.config.env.objects.keys()
        self.env_obj_data = [{obj_name: {} for obj_name in obj_names}
                             for _ in range(self.config.n_envs)]

    def _create_robot_assets(self):
        """
        Creates robot assets specified in config.

        TODO Assuming there is only one robot for now, need to modify for bimanual
        """
        asset_options = gymapi.AssetOptions()
        asset_options.armature = self.robot.config.armature
        asset_options.fix_base_link = self.robot.config.fix_base_link
        asset_options.disable_gravity = self.robot.config.disable_gravity
        asset_options.flip_visual_attachments = self.robot.config.flip_visual_attachments
        self.robot_asset = self.gym.load_asset(self.sim, self.config.sim.asset_root,
                                               self.robot.config.asset_filename, asset_options)

        self.robot_num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        n_arm_joints = self.robot.arm.num_joints()
        n_ee_joints = self.robot.end_effector.num_joints()
        self.robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        self.robot_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS) # TODO add option for modes

        self.dof_lower_lims = torch.tensor(self.robot_dof_props['lower'][:self.robot_num_dofs])
        self.dof_upper_lims = torch.tensor(self.robot_dof_props['upper'][:self.robot_num_dofs])
        self.dof_speed_scales = torch.ones_like(self.dof_lower_lims)
        # findx = self.robot_config.end_effector.close_finger_indices
        # # resistance metrics : TODO changing these seems to impact expert performance
        # self.dof_speed_scales[findx] = 0.25
        # self.robot_dof_props['effort'][findx] = 150

        self.robot_dof_props["stiffness"][:n_arm_joints] = self.robot.arm.config.stiffness
        self.robot_dof_props["damping"][:n_arm_joints] = self.robot.arm.config.damping
        if self.robot.config.end_effector:
            self.robot_dof_props["stiffness"][n_arm_joints:] = self.robot.end_effector.config.stiffness
            self.robot_dof_props["damping"][n_arm_joints:] = self.robot.end_effector.config.damping

        if self.robot_num_dofs != self.robot.num_joints():
            raise RuntimeError(f"Something is wrong, expected robot to have "
                               f"{self.robot.num_joints()} but simulator has {self.robot_num_dofs}")
        self.default_dof_pos = np.zeros(self.robot_num_dofs, dtype=np.float32)
        self.default_dof_pos[:n_arm_joints] = self.robot.arm.get_default_joint_position()
        if self.robot.has_end_effector():
            self.default_dof_pos[n_arm_joints:] = self.robot.end_effector.get_default_joint_position()
        self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        self.pos_target = torch.from_numpy(self.default_dof_pos).repeat(self.config.n_envs, 1)
        self.pos_target = self.pos_target.unsqueeze(-1).to(self.config.device)

    def destroy_sim(self):
        """
        Destroys the active simulator.
        """
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def reset(self):
        """
        Reset the simulator envs and dataset.
        """
        self.reset_environments()
        # Make sure env is setup by stepping a handful of timesteps
        for _ in range(self.reset_wait_steps):
            self.step()

        if self.collect_data:
            self.reset_dataset()
        self.should_log = [True for _ in range(self.config.n_envs)]
        self.timestep = 0

    def reset_dataset(self):
        """
        Resets the data caches used for saving datasets for each env.
        """
        obj_names = self.config.env.objects.keys()
        self.dataset = {
            'rgb': [[] for _ in range(self.config.n_envs)],
            'depth': [[] for _ in range(self.config.n_envs)],
            'projection_matrix': [[] for _ in range(self.config.n_envs)],
            'view_matrix': [[] for _ in range(self.config.n_envs)],
            'flow': [[] for _ in range(self.config.n_envs)],
            'segmentation': [[] for _ in range(self.config.n_envs)],
            'joint_position': [[] for _ in range(self.config.n_envs)],
            'joint_velocity': [[] for _ in range(self.config.n_envs)],
            'joint_torque': [[] for _ in range(self.config.n_envs)],
            'ee_position': [[] for _ in range(self.config.n_envs)],
            'ee_orientation': [[] for _ in range(self.config.n_envs)],
            'ee_velocity': [[] for _ in range(self.config.n_envs)],
            'objects': [
                {n: {'position': [], 'orientation': [], 'velocity': []} for n in obj_names}
                for _ in range(self.config.n_envs)
            ],
            'action': [
                {'joint_position': [],
                 'arm': {'joint_position': []},
                 'ee': {'joint_position': [], 'discrete': [], 'same_angle': []}}
                for _ in range(self.config.n_envs)
            ]
        }

    def reset_environments(self):
        """
        Reset the environments by setting object poses and robot configuration.
        """
        for env_idx, env in enumerate(self.envs):
            self.reset_objects(env_idx)
            self.reset_robots(env_idx)

    def reset_objects(self, env_idx):
        """
        Resets object poses and colors based on config (will be randomized if specified in config).
        """
        env = self.envs[env_idx]
        for obj_name, cfg in self.config.env.objects.items():

            pos = deepcopy(cfg.position)
            if cfg.position_ranges is not None:
                for i, pos_range in enumerate(cfg.position_ranges):
                    if pos_range is not None:
                        pos[i] = np.random.uniform(*pos_range)
            if cfg.orientation is None:
                axis = gymapi.Vec3(*cfg.sample_axis)
                angle = np.random.uniform(cfg.sample_angle_lower, cfg.sample_angle_upper)
                rot = gymapi.Quat.from_axis_angle(axis, angle)
            else:
                rot = gymapi.Quat(*cfg.orientation)

            obj_idx = self.gym.find_actor_index(env, obj_name, gymapi.DOMAIN_ENV)
            handle = self.gym.get_actor_handle(env, obj_idx)
            obj_state = np.copy(self.gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_ALL))
            obj_state['pose']['p'].fill(tuple(pos))
            obj_state['pose']['r'].fill((rot.x, rot.y, rot.z, rot.w))
            self.gym.set_actor_rigid_body_states(env, handle, obj_state, gymapi.STATE_ALL)

            if cfg.friction:
                self._set_object_property('friction', cfg.friction, env, handle)
            if cfg.restitution:
                self._set_object_property('restitution', cfg.restitution, env, handle)
                
            if cfg.set_color:
                cr = np.random.uniform(0, 1) if cfg.rgb_color is None else cfg.rgb_color[0]
                cg = np.random.uniform(0, 1) if cfg.rgb_color is None else cfg.rgb_color[1]
                cb = np.random.uniform(0, 1) if cfg.rgb_color is None else cfg.rgb_color[2]
                color = gymapi.Vec3(cr, cg, cb)
                self.env_obj_data[env_idx][obj_name]['color'] = [cr, cg, cb]
                self.gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            else:
                self.env_obj_data[env_idx][obj_name]['color'] = None

    def reset_robots(self, env_idx):
        """
        Resets robot joint configuration (will be randomized if specified in config).
        """
        env = self.envs[env_idx]
        dof_state = self.default_dof_state.copy()
        dof_pos = self.default_dof_pos.copy()

        # TODO I find managing all this kind of a mess, there's probably a more
        # parsimonious way to do this but I've been going off what the gym examples
        # did and haven't tried to look too much into how to make it more efficient
        n_arm_joints = self.robot.arm.num_joints()
        arm_joints = self.robot.arm.get_init_joint_position(self.config.randomize_robot)
        ee_joints = self.robot.end_effector.get_init_joint_position(self.config.randomize_robot)
        init_joints = np.concatenate([arm_joints, ee_joints])
        dof_state["pos"] = init_joints
        dof_pos[:] = init_joints
        self.pos_target[env_idx] = torch.from_numpy(init_joints).unsqueeze(-1)

        if self.robot.get_name() is None:
            raise RuntimeError("Robot name is not known. Was it specified in config?")
        robot_idx = self.gym.find_actor_index(env, self.robot.get_name(), gymapi.DOMAIN_ENV)
        robot_handle = self.gym.get_actor_handle(env, robot_idx)
        self.gym.set_actor_dof_states(env, robot_handle, dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_dof_position_targets(env, robot_handle, dof_pos)

        self.actions[env_idx].set_joint_position(arm_joints, ee_joints)
        self.actions[env_idx].clear_ee_discrete()
        self.actions[env_idx].clear_ee_same_angle()
        self.actions[env_idx].set_ee_discrete('open')

    def _publish_ros_data(self):
        """
        Publishes current sim state to ROS.
        """
        state = IsaacGymState()
        for env_idx in range(self.config.n_envs):
            joint_state = JointState()
            joint_state.name = self.robot_joint_names
            joint_state.position = self.dof_pos[env_idx].squeeze().cpu().numpy().tolist()
            joint_state.velocity = self.dof_vel[env_idx].squeeze().cpu().numpy().tolist()
            joint_state.effort = self.dof_force[env_idx].squeeze().cpu().numpy().tolist()
            state.joint_state.append(joint_state)

            if self.robot.has_end_effector():
                ee_state = self.rb_states[self.robot.end_effector.get_rb_index(env_idx)]
                ee_pose = Pose()
                ee_pose.position.x = ee_state[0]
                ee_pose.position.y = ee_state[1]
                ee_pose.position.z = ee_state[2]
                ee_pose.orientation.x = ee_state[3]
                ee_pose.orientation.y = ee_state[4]
                ee_pose.orientation.z = ee_state[5]
                ee_pose.orientation.w = ee_state[6]
                state.ee_pose.append(ee_pose)

            rgb = self._get_rgb_img(self.envs[env_idx], self.sensor_config)
            state.rgb.append(ros_util.rgb_to_msg(rgb))
            depth = self._get_depth_img(self.envs[env_idx], self.sensor_config)
            state.depth.append(ros_util.depth_to_msg(depth))

            tf_env = TFMessage()
            for obj_name, obj_config in self.config.env.objects.items():
                obj_state = self.rb_states[obj_config.rb_indices[env_idx]]
                tf = TransformStamped()
                tf.header.frame_id = 'world'
                tf.child_frame_id = obj_name
                tf.transform.translation.x = obj_state[0]
                tf.transform.translation.y = obj_state[1]
                tf.transform.translation.z = obj_state[2]
                tf.transform.rotation.x = obj_state[3]
                tf.transform.rotation.y = obj_state[4]
                tf.transform.rotation.z = obj_state[5]
                tf.transform.rotation.w = obj_state[6]
                tf_env.transforms.append(tf)
            state.tf.append(tf_env)

        self.state_pub.publish(state)

    def _cache_step_data(self):
        """
        Caches all relevant simulation state/observations to a running memory for saving data.

        NOTE: We currently take a very simple approach to saving data which is just hold onto
        everything in memory and then when all envs are done start saving the data. This could
        be made much more complicated to use asynchronous data buffers and such, but right now
        it's not so important as long as you have a decent memory size (e.g. 32GB is good).
        """
        for i in range(self.config.n_envs):
            if not self.should_log[i]:
                continue

            # Note for all vector data the flatten operation is making a copy of the data.
            # If you don't flatten make sure to still copy, otherwise you just store a
            # reference to the data that keeps getting updated, so when you save it you'll
            # just have the last recorded data point for all timesteps.
            self.dataset['joint_position'][i].append(self.dof_pos[i].cpu().numpy().flatten())
            self.dataset['joint_velocity'][i].append(self.dof_vel[i].cpu().numpy().flatten())
            self.dataset['joint_torque'][i].append(self.dof_force[i].cpu().numpy().flatten())

            # Log actions
            act = self.actions[i]
            self.dataset['action'][i]['joint_position'].append(act.get_joint_position())
            self.dataset['action'][i]['arm']['joint_position'].append(act.get_arm_joint_position())
            if self.robot.has_end_effector():
                self.dataset['action'][i]['ee']['joint_position'].append(act.get_ee_joint_position())
                # Tracking no-ops for these (i.e. appending None) simplifies indexing so that you
                # can easily mix action modes across different skills in the task
                discrete = act.get_ee_discrete() if act.has_ee_discrete() else None
                same_angle = act.get_same_angle() if act.has_ee_same_angle() else None
                self.dataset['action'][i]['ee']['discrete'].append(discrete)
                self.dataset['action'][i]['ee']['same_angle'].append(same_angle)

            if self.robot.has_end_effector():
                ee_idx = self.robot.end_effector.get_rb_index(i)
                ee_state = self.rb_states[ee_idx, :].cpu().numpy().flatten()
                self.dataset['ee_position'][i].append(ee_state[:3])
                self.dataset['ee_orientation'][i].append(ee_state[3:7])
                self.dataset['ee_velocity'][i].append(ee_state[7:])

            for obj_name, obj_config in self.config.env.objects.items():
                rb_idx = obj_config.rb_indices[i]
                obj_state = self.rb_states[rb_idx, :].cpu().numpy().flatten()
                self.dataset['objects'][i][obj_name]['position'].append(obj_state[:3])
                self.dataset['objects'][i][obj_name]['orientation'].append(obj_state[3:7])
                self.dataset['objects'][i][obj_name]['velocity'].append(obj_state[7:])

            if self.sensor_config.sensor_type == 'camera':
                self.dataset['rgb'][i].append(
                    self._get_rgb_img(self.envs[i], self.sensor_config))
                self.dataset['depth'][i].append(
                    self._get_depth_img(self.envs[i], self.sensor_config))
                #print(self._get_camera_intrinsics(self.envs[i], self.sensor_config))
                self.dataset['projection_matrix'][i].append(
                    self._get_camera_intrinsics(self.envs[i], self.sensor_config)[0])
                #print(self.dataset['projection_matrix'][i])
                self.dataset['view_matrix'][i].append(
                    self._get_camera_intrinsics(self.envs[i], self.sensor_config)[1])
                self.dataset['segmentation'][i].append(
                    self._get_seg_img(self.envs[i], self.sensor_config))
                # TODO leaving out flow for now, wasn't being read for some reason
                # self.dataset['flow'][i].append(self._get_flow_img(self.envs[i], sensor_config))
            else:
                raise ValueError(f"Unknown sensor type: {self.sensor_config.sensor_type}")

    def save_data(self, pbar=None):
        """
        Saves memory-cached data to disk for all envs.
        """
        # Save config as metadata so it's easy to recreate scene even if original config changes.
        # Note it's assumed per directory that every demo is for the same task
        metadata_filename = os.path.join(self.config.data_root, "metadata.yaml")
        if not os.path.exists(metadata_filename):
            file_util.save_yaml(self.config.to_dict(), metadata_filename)

        for env_idx in range(self.config.n_envs):
            # Don't even try saving ones that didn't move from init, if n_steps is low it
            # means we likely didn't even get a plan for first motion
            n_steps = len(self.dataset['joint_position'][env_idx])
            if n_steps < 10:
                ui_util.print_warning(f"Skipping env {env_idx}, too few timesteps")
                continue
            self._save_pickle_data(env_idx)
            if pbar:
                pbar.update(1)

    def _save_pickle_data(self, env_idx):
        """
        Save current data buffer to pickle files.
        """
        n_files = len(file_util.list_dir(self.config.data_root, '.pickle'))
        filename = os.path.join(self.config.data_root,
                                f"{self.config.data_prefix}_{n_files+1:04d}.pickle")

        act = self.dataset['action'][env_idx]
        data = {
            'rgb': np.array(self.dataset['rgb'][env_idx]),
            'depth': np.array(self.dataset['depth'][env_idx]),
            'projection_matrix': np.array(self.dataset['projection_matrix'][env_idx]),
            'view_matrix': np.array(self.dataset['view_matrix'][env_idx]),
            'segmentation': np.array(self.dataset['segmentation'][env_idx]),
            'joint_position': np.array(self.dataset['joint_position'][env_idx]),
            'joint_velocity': np.array(self.dataset['joint_velocity'][env_idx]),
            'joint_torque': np.array(self.dataset['joint_torque'][env_idx]),
            'ee_position': np.array(self.dataset['ee_position'][env_idx]),
            'ee_orientation': np.array(self.dataset['ee_orientation'][env_idx]),
            'ee_velocity': np.array(self.dataset['ee_velocity'][env_idx]),
            'objects': {},
            'action': {
                'joint_position': np.array(act['joint_position']),
                'arm': {'joint_position': np.array(act['arm']['joint_position'])},
                'ee': {
                    'joint_position': np.array(act['ee']['joint_position']),
                    'discrete': np.array(act['ee']['discrete']),
                    'same_angle': np.array(act['ee']['same_angle'])
                }
            }
        }

        attrs = {
            'segmentation_labels': list(self.seg_id_dict.keys()),
            'segmentation_ids': list(self.seg_id_dict.values()),
            'objects': {},
            'robot_joint_names': self.robot_joint_names,
            'robot_link_names': self.robot_link_names,
            'n_arm_joints': self.robot.arm.num_joints(),
            'n_ee_joints': self.robot.end_effector.num_joints()
        }

        for obj_name, obj_config in self.config.env.objects.items():
            data['objects'][obj_name] = {
                'position': np.array(self.dataset['objects'][env_idx][obj_name]['position']),
                'orientation': np.array(self.dataset['objects'][env_idx][obj_name]['orientation']),
                'velocity': np.array(self.dataset['objects'][env_idx][obj_name]['velocity'])
            }
            attrs['objects'][obj_name] = {
                'asset_filename': obj_config.asset_filename,
                'parent_frame': 'world',
                'x_extent': obj_config.extents[0],
                'y_extent': obj_config.extents[1],
                'z_extent': obj_config.extents[2],
                'rgb_color': self.env_obj_data[env_idx][obj_name]['color'],
                'fix_base_link': obj_config.fix_base_link,
                'density': obj_config.density
            }
        file_util.save_pickle((data, attrs), filename)

    def _get_rgb_img(self, env, config):
        img = self.gym.get_camera_image(self.sim, env, config.sim_handle, gymapi.IMAGE_COLOR)
        img = img.reshape(config.width, config.height, 4)[:,:,:3]
        if self.rgb_size is not None:
            img = torch.tensor( img, dtype=torch.float ).permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate( img, self.rgb_size,
              mode='bilinear', align_corners=False ) # (t, 3, 64, 64)
            img = img.squeeze(0).numpy()
        return img

    def _get_depth_img(self, env, config):
        img = self.gym.get_camera_image(self.sim, env, config.sim_handle, gymapi.IMAGE_DEPTH)
        img[img == -np.inf] = 0.0  # Filter no-depth values
        img[img < config.depth_min] = config.depth_min  # Clamp to min value
        return img

    def _get_camera_intrinsics(self, env, config):
        projection_matrix = np.matrix(self.gym.get_camera_proj_matrix(self.sim, env, config.sim_handle))
        view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env, config.sim_handle))
        return projection_matrix, view_matrix

    def _get_seg_img(self, env, config):
        img = self.gym.get_camera_image(self.sim, env, config.sim_handle, gymapi.IMAGE_SEGMENTATION)
        return img

    def _get_flow_img(self, env, config):
        flow = self.gym.get_camera_image(self.sim, env, config.sim_handle, gymapi.IMAGE_OPTICAL_FLOW)
        img = np.zeros((config.width, config.height, 3))
        img[0,0] = config.width * flow[0,0] / 2**15
        img[0,1] = config.height * flow[0,1] / 2**15
        return img

    def get_env_state(self, env_idx):
        """
        Populate env state with current values from simulator.

        Args:
            env_idx (int): Index of environment to populate state for
        """
        self.env_states[env_idx].joint_names = self.robot_joint_names
        self.env_states[env_idx].joint_position = self.dof_pos[env_idx,:]
        self.env_states[env_idx].joint_velocity = self.dof_vel[env_idx,:]
        self.env_states[env_idx].joint_torque = self.dof_force[env_idx,:]
        self.env_states[env_idx].n_arm_joints = self.robot.arm.num_joints()
        self.env_states[env_idx].n_ee_joints = self.robot.end_effector.num_joints()

        self.env_states[env_idx].timestep = self.timestep

        self.env_states[env_idx].prev_action = self.actions[env_idx]

        if self.robot.has_end_effector():
            ee_idx = self.robot.end_effector.get_rb_index(env_idx)
            self.env_states[env_idx].ee_state = self.rb_states[ee_idx, :]

        # self.env_states[env_idx].jacobian = self.jacobian[env_idx,:,:,:]

        for obj_name, obj_config in self.config.env.objects.items():
            rb_idx = obj_config.rb_indices[env_idx]
            self.env_states[env_idx].object_states[obj_name] = self.rb_states[rb_idx, :]
            if obj_config.set_color:
                obj_color = self.env_obj_data[env_idx][obj_name]['color']
                self.env_states[env_idx].object_colors[obj_name].r = obj_color[0]
                self.env_states[env_idx].object_colors[obj_name].g = obj_color[1]
                self.env_states[env_idx].object_colors[obj_name].b = obj_color[2]
                self.env_states[env_idx].object_colors[obj_name].a = 1

        if self.config.task.include_rgb_in_state:
            self.env_states[env_idx].rgb = self._get_rgb_img(self.envs[env_idx], self.sensor_config)
        if self.config.task.include_depth_in_state:
            self.env_states[env_idx].depth = self._get_depth_img(self.envs[env_idx], self.sensor_config)

        return self.env_states[env_idx]

    def get_joint_position(self, env_idx):
        """
        Returns the current robot joint position for the specified env.

        TODO right now returns flattened np array, can make returns more configurable
        """
        return self.dof_pos[env_idx].cpu().numpy().flatten()

    def get_ee_state(self, env_idx=None):
        if env_idx is None:
            return self.rb_states[self.robot.end_effector.rb_indices]
        else:
            return self.rb_states[self.robot.end_effector.rb_indices[env_idx]]

    def set_dof_state(self, dof_state):
        """
        Manually set the DOF state for the robots.

        Args:
            dof_state (Tensor): Tensor for DOF state, shape (n_envs, 2*n_dofs)
        """
        if isinstance(dof_state, torch.Tensor):
            dof_state = gymtorch.unwrap_tensor(dof_state)
        self.gym.set_dof_state_tensor(self.sim, dof_state)

    def get_dof_state(self):
        """
        Returns current DOF state (Tensor), shape (n_envs, 2*n_dofs)
        """
        dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        return dof_state.view(self.config.n_envs, self.n_joints, 2)

    def apply_action(self, action, env_idx):
        """
        TODO this should be updated when the action interface is updated to be tensorized
        so that you can opptionally send in env_idx, and if it's not passed in it will
        take the tensor version from the action interface
        """
        self.robot.update_action_joint_pos(action)
        self.set_target_joint_position(action.get_joint_position(), env_idx)
        self.actions[env_idx] = action  # Cache the action for logging and passing to behaviors

    def set_target_joint_position(self, joint_pos, env_idx):
        """
        Sets the target joint position for the specified env.
        """
        if isinstance(joint_pos, np.ndarray):
            joint_pos = torch.tensor(joint_pos, dtype=float)
        self.pos_target[env_idx] = joint_pos.unsqueeze(-1)
        self.set_target_joint_position_tensor(self.pos_target)

    def set_target_joint_position_tensor(self, joint_pos):
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(joint_pos.to(self.config.device)))

    def refresh_rigid_body_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _set_object_property(self, name, value, env, handle):
        props = self.gym.get_actor_rigid_shape_properties(env, handle)
        if not hasattr(props[0], name):
            raise ValueError(f"Unknown rigid shape property: {name}")
        setattr(props[0], name, value)
        self.gym.set_actor_rigid_shape_properties(env, handle, props)


    def forward_kinematics(self, joint_pos):
        """
        Computes forward kinematics in batch.

        Args:
            joint_pos (Tensor): Tensor of joint positions (n_batch, n_joints)
        """
        orig_dof_state = self.get_dof_state()
        
        if joint_pos.size(1) != orig_dof_state.size(1):
            ui_util.print_error(f"Joint state in FK request of size {joint_pos.size(1)} "
                                f"but expected size {orig_dof_state.size(1)}")
            return None
        
        batches = torch.split(joint_pos, self.config.n_envs)
        poses = []
        new_dof_state = orig_dof_state.clone()
        for batch in batches:
            n_batch = batch.size(0)
            new_dof_state[:n_batch,:,0] = batch
            self.set_dof_state(new_dof_state)
            self.refresh_rigid_body_state()
            ee_state = self.get_ee_state() # (n_envs, 13), last dim pos(3)/quat(4)/vel(6)
            poses.append(ee_state[:n_batch,:7])
        # Revert back to old state
        self.set_dof_state(orig_dof_state)
        self.refresh_rigid_body_state()
        
        return torch.cat(poses, dim=0)

    def get_env_contacts(self, env_idx, check_only=[], exclude_pairs=[]):
        """
        Filter contacts based for ones registering collision between the
        specified objects. Checks all pairwise connections between objects.
        """
        contacts = self.gym.get_env_rigid_contacts(self.envs[env_idx])
        if check_only:
            rb_idxs = [self.get_env_rb_index(env_idx, n) for n in check_only]
            contacts = [c for c in contacts if c['body0'] in rb_idxs and c['body1'] in rb_idxs]
        if exclude_pairs:
            rb_pairs = [(self.get_env_rb_index(env_idx, n1), self.get_env_rb_index(env_idx, n2))
                        for (n1, n2) in exclude_pairs]
            contacts = [c for c in contacts
                        if not any([self._is_contact_between(c, i1, i2) for (i1, i2) in rb_pairs])]
        return contacts

    def _is_contact_between(self, contact, idx1, idx2):
        return ((idx1 == contact['body0'] and idx2 == contact['body1']) or
                (idx1 == contact['body1'] and idx2 == contact['body0']))

    # def get_rb_name(self, env_idx, rb_idx):
    #     if rb_idx == -1:
    #         return 'ground'
    #     for obj_name, obj_config in self.config.env.objects.items():
    #         if rb_idx == obj_config.rb_indices[env_idx]:
    #             return obj_name
    #     for link_name, rb_idxs in self.robot_config.rb_indices.items():
    #         if rb_idx == rb_idxs[env_idx]:
    #             return link_name
    #     return None

    def get_env_rb_index(self, env_idx, rb_name):
        """
        TODO I think you should just cache these once they're looked up, because after
        creation they shouldn't ever change
        """
        rb_idx = None
        if rb_name == 'ground':
            return -1
        elif rb_name in self.config.env.objects:
            obj_handle = self.gym.find_actor_handle(self.envs[env_idx], rb_name)
            rb_idx = self.gym.get_actor_rigid_body_index(self.envs[env_idx], obj_handle, 0,
                                                         gymapi.DOMAIN_ENV)
        elif rb_name in self.robot_link_names:
            robot_idx = self.gym.find_actor_index(env, self.robot.get_name(), gymapi.DOMAIN_ENV)
            robot_handle = self.gym.get_actor_handle(env, robot_idx)
            rb_idx = self.gym.find_actor_rigid_body_index(self.envs[env_idx], robot_handle, rb_name,
                                                          gymapi.DOMAIN_ENV)
        return rb_idx
