"""
This file contains several dataclass classes (essentially acting as structs)
for various components used in the ll4ma_isaacgym framework. These are useful
as they can be instantiated from YAML configuration files recursively, provide
a way to set default configurations easily while overriding from configuration
files, and are notationally less verbose than passing dictionaries around.
"""
from isaacgym import gymapi

import os
import sys
import importlib  # Dynamic class creation
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Dict

from ll4ma_util import file_util, ros_util, func_util

BEHAVIOR_MODULES = ["ll4ma_isaacgym.behaviors", "multisensory_learning.planning"]
BEHAVIOR_CONFIG_MODULES = ["ll4ma_isaacgym.core.config"]

LL4MA_ISAACGYM_ROOT = os.path.join(ros_util.get_path('ll4ma_isaacgym'), 'src', 'll4ma_isaacgym')
DEFAULT_TASK_CONFIG_DIR = os.path.join(LL4MA_ISAACGYM_ROOT, 'config')
VALID_TASKS = ['stack_objects', 'pick_object', 'place_object', 'pick_place',
               'pick_drop_in_basket', 'move_to_object', 'push_blocks', 'push_object']
# Needs to be workspace path since urdfs are written for ROS which will do asset
# resolution (like mesh files) relative to package directory
DEFAULT_ASSET_ROOT = os.path.dirname(ros_util.get_path('ll4ma_robots_description'))


@dataclass
class ObjectConfig(func_util.Config):
    """
    Configuration dataclass for objects in the simulator.
    """
    object_type: str = None
    asset_root: str = None
    asset_filename: str = None
    name: str = None
    extents: List[float] = func_util.lambda_field([0., 0., 0.])
    density: float = None  # g/m^3
    friction: float = None  # Range [0,1]: 0 slippery, 1 sticky, defaults 1
    restitution: float = None  # Range [0,1]: 0 no-bounce, 1 bounces away on drop, defaults 0
    rgb_color: List[float] = None
    set_color: bool = True
    position: List[float] = func_util.lambda_field([None]*3)
    position_ranges: List[List[float]] = None
    orientation: List[float] = None
    sample_axis: List[float] = None
    sample_angle_lower: float = None
    sample_angle_upper: float = None
    fix_base_link: bool = False
    rb_indices: List[int] = field(default_factory=list) # For retrieving pose in sim
    frame_id: str = 'world'


@dataclass
class ArmConfig(func_util.Config):
    """
    Generic configuration dataclass for arms.
    """
    arm_type: str = None
    name: str = None
    n_joints: int = 0
    stiffness: List[float] = None
    damping: List[float] = None
    default_joint_pos: List[float] = None
    joint_pos_sample_range: float = 0.0
    rb_indices: Dict[str, int] = field(default_factory=dict)


@dataclass
class EndEffectorConfig(func_util.Config):
    """
    Generic configuration dataclass for end-effectors.
    """
    ee_type: str = None
    name: str = None
    n_joints: int = 0
    stiffness: List[float] = None
    damping: List[float] = None
    link: str = None
    default_joint_pos: List[float] = None
    close_finger_joint_pos: List[float] = None
    close_finger_indices: List[float] = None
    close_for_steps: int = 20  # Number of steps to repeat close-finger action
    open_for_steps: int = 20   # Number of steps to repeat open-finger action
    joint_pos_sample_range: float = 0.0
    obj_to_ee_offset: float = 0.0

    
@dataclass
class RobotConfig(func_util.Config):
    """
    Robot configuration dataclass that encapsulates configurations for the
    arm and the end-effector, as well as attributes that are applied to the
    robot system as a whole.
    """
    arm: ArmConfig = field(default_factory=ArmConfig)
    end_effector: EndEffectorConfig = field(default_factory=EndEffectorConfig)
    asset_filename: str = None
    name: str = None
    n_joints: int = 0
    armature: float = 0.01
    fix_base_link: bool = True
    disable_gravity: bool = True
    flip_visual_attachments: bool = False
    position: List[float] = func_util.lambda_field([0., 0., 0.])
    orientation: List[float] = func_util.lambda_field([0., 0., 0., 1.])

    def __post_init__(self):
        self.n_joints = self.arm.n_joints + self.end_effector.n_joints
        super().__post_init__()


@dataclass
class SensorConfig(func_util.Config):
    """
    Generic configuration dataclass for sensors.
    """
    sensor_type: str = None


@dataclass
class CameraConfig(func_util.Config):
    """
    Configuration dataclass for cameras.
    """
    sensor_type: str = "camera"
    origin: List[float] = None
    target: List[float] = None
    width: int = 512
    height: int = 512
    sim_handle: int = None
    depth_min: float = -3.0


@dataclass
class PandaGripperConfig(EndEffectorConfig):
    """
    Configuration dataclass for Panda two-finger gripper.
    """
    ee_type: str = "PandaGripper"
    name: str = "panda_gripper"
    n_joints: int = 2
    stiffness: List[float] = func_util.lambda_field([800.0] * 2)
    damping: List[float] = func_util.lambda_field([40.0] * 2)
    link: str = "end_effector_frame"
    default_joint_pos: List[float] = func_util.lambda_field([0.04] * 2)
    open_finger_joint_pos: List[float] = func_util.lambda_field([0.04] * 2)
    close_finger_joint_pos: List[float] = func_util.lambda_field([0.0] * 2)
    grip_finger_indices: List[int] = func_util.lambda_field([-2, -1])


@dataclass
class ReflexConfig(EndEffectorConfig):
    """
    Configuration dataclass for Reflex hand.
    """
    ee_type: str = "reflex"
    name: str = "reflex"
    n_joints: int = 5
    stiffness: List[float] = func_util.lambda_field([800.0] * 5)
    damping: List[float] = func_util.lambda_field([40.0] * 5)
    link: str = "reflex_palm_link"
    default_joint_pos: List[float] = func_util.lambda_field([0.0] * 5)
    # Close proximal joints and ignore preshape
    open_finger_joint_pos: List[float] = func_util.lambda_field([0.0] * 3)
    close_finger_joint_pos: List[float] = func_util.lambda_field([2.0] * 3)
    grip_finger_indices: List[int] = func_util.lambda_field([-4, -2, -1])
    obj_to_ee_offset: float = 0.05
    

@dataclass
class PandaConfig(ArmConfig):
    """
    Configuration dataclass for the Franka Emika Panda arm.
    """
    robot_type: str = "Panda"
    name: str = "panda"
    n_joints: int = 7
    stiffness: List[float] = func_util.lambda_field([400.0] * 7)
    damping: List[float] = func_util.lambda_field([40.0] * 7)
    default_joint_pos: List[float] = func_util.lambda_field([0, 0, 0, -0.9006, 0, 1.1205, 0])


@dataclass
class IiwaConfig(ArmConfig):
    """
    Configuration dataclass for the KUKA iiwa arm.
    """
    robot_type: str = "iiwa"
    name: str = "iiwa"
    n_joints: int = 7
    stiffness: List[float] = func_util.lambda_field([400.0] * 7)
    damping: List[float] = func_util.lambda_field([40.0] * 7)
    end_effector: EndEffectorConfig = field(default_factory=ReflexConfig)
    default_joint_pos: List[float] = func_util.lambda_field([0, 0.5, 0, -0.5, 0, 0.5, 0])


@dataclass
class EnvironmentConfig(func_util.Config):
    """
    Configuration dataclass for the simulation environment, which encapsulates
    objects, robots (arms + end-effectors), sensors, and general attributes to
    be applied to the simulation environment.
    """
    img_size: int = None
    spacing: float = 1.0
    objects: Dict[str, ObjectConfig] = field(default_factory=dict)
    robots: Dict[str, RobotConfig] = field(default_factory=dict)
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)

    def from_dict(self, dict_):
        for k, v in dict_['objects'].items():
            self.objects[k] = ObjectConfig(config_dict=v)
        del dict_['objects']

        for k, v in dict_['robots'].items():
            arm_config_name = f"{v['arm']['arm_type']}Config"
            ArmConfigClass = func_util.get_class(arm_config_name, ['ll4ma_isaacgym.core.config'])
            if ArmConfigClass is None:
                raise ValueError(f"No config known for arm type: {v['arm']['arm_type']}")
            arm_config  = ArmConfigClass(config_dict=v['arm'])

            ee_config_name = f"{v['end_effector']['ee_type']}Config"
            EEConfigClass = func_util.get_class(ee_config_name, ['ll4ma_isaacgym.core.config'])
            if EEConfigClass is None:
                raise ValueError(f"No config known for EE type: {v['end_effector']['ee_type']}")
            ee_config = EEConfigClass(config_dict=v['end_effector'])

            del v['arm']
            del v['end_effector']
            robot_config = RobotConfig(config_dict=v)
            robot_config.arm = arm_config
            robot_config.end_effector = ee_config
            robot_config.name = k
            self.robots[k] = robot_config

        del dict_['robots']

        for k, v in dict_['sensors'].items():
            if v['sensor_type'] == 'camera':
                self.sensors[k] = CameraConfig(config_dict=v)
            else:
                raise ValueError(f"Unknown sensor type: {v['sensor_type']}")
        del dict_['sensors']

        super().from_dict(dict_)


@dataclass
class BehaviorConfig(func_util.Config):
    """
    Configuration dataclass for behaviors.
    """
    # Each behavior can have arbitrarily many sub-behaviors (hierarchical). Note these are
    # type BehaviorConfig but seems dataclasses don't allow recursive typing
    behaviors: List = field(default_factory=list)
    behavior_type: str = ''
    name: str = ''
    max_plan_attempts: int = 1
    ignore_error: bool = False
    wait_after_behavior: int = 10

    def from_dict(self, dict_):
        if 'behaviors' in dict_:
            for v in dict_['behaviors']:
                if 'behavior_config_type' in v:
                    # Use the user-provided config type
                    BehaviorConfigClass = get_behavior_config_class(v['behavior_config_type'])
                elif 'behavior_type' in v:
                    # Try to use the behavior-specific config if we can find it, default
                    # to the base BehaviorConfig class
                    BehaviorConfigClass = get_behavior_config_class(f"{v['behavior_type']}Config")
                else:
                    raise ValueError("Could not resolve behavior class")
                behavior_config = BehaviorConfigClass(config_dict=v)
                self.behaviors.append(behavior_config)
            del dict_['behaviors']
        super().from_dict(dict_)

    def __postinit__(self):
        if not self.name:
            raise ValueError("Must specify unique name for behavior (none was provided)")


@dataclass
class MoveToPoseConfig(BehaviorConfig):
    """
    Configuration dataclass for move-to-target behavior.
    """
    behavior_type: str = 'MoveToPose'
    pose: List[float] = None
    offset_from_current: List[float] = func_util.lambda_field([0.0] * 3)
    offset_from_current_ranges: List[List[float]] = None
    offset_from_current_choices: List[List[float]] = None
    planning_time: float = 1.0
    max_vel_factor: float = 0.3
    max_acc_factor: float = 0.1
    disable_collisions: List[str] = field(default_factory=list)
    cartesian_path: bool = False

    def __postinit__(self):
        if max_vel_factor <= 0. or max_vel_factor > 1.:
            raise ValueError("max_vel_factor must be in range (0,1]")
        if max_acc_factor <= 0. or max_acc_factor > 1.:
            raise ValueError("max_acc_factor must be in range (0,1]")


@dataclass
class CloseFingersConfig(BehaviorConfig):
    """
    Configuration dataclass for closing EE fingers behavior.
    """
    behavior_type: str = 'CloseFingers'


@dataclass
class OpenFingersConfig(BehaviorConfig):
    """
    Configuration dataclass for opening EE fingers behavior.
    """
    behavior_type: str = 'OpenFingers'


@dataclass
class PickObjectConfig(BehaviorConfig):
    """
    Configuration dataclass for object picking behavior.
    """
    behavior_type: str = 'PickObject'
    target_object: str = ''
    lift_offset: List[float] = func_util.lambda_field([0.0] * 3)
    lift_offset_ranges: List[List[float]] = None
    lift_offset_choices: List[List[float]] = None
    allow_top_grasps: bool = True
    allow_side_grasps: bool = True
    allow_bottom_grasps: bool = False  # Usually in collision with supporting surface
    remove_aligned_short_bb_face: bool = False
    

@dataclass
class PlaceObjectConfig(BehaviorConfig):
    """
    Configuration dataclass for object picking behavior.
    """
    behavior_type: str = 'PlaceObject'
    target_object: str = ''
    place_position: List[float] = func_util.lambda_field([0.0] * 3)
    place_orientation: List[float] = None
    place_position_ranges: List[List[float]] = None
    place_approach_height: float = 0.1


@dataclass
class PickPlaceObjectConfig(BehaviorConfig):
    behavior_type: str = 'PickPlaceObject'
    target_object: str = ''
    lift_offset: List[float] = func_util.lambda_field([0.0, 0.0, 0.2]) # Default lift 20cm
    lift_offset_ranges: List[List[float]] = None
    lift_offset_choices: List[List[float]] = None
    allow_top_grasps: bool = True
    allow_side_grasps: bool = True
    allow_bottom_grasps: bool = False  # Usually in collision with supporting surface
    remove_aligned_short_bb_face: bool = False
    place_position: List[float] = func_util.lambda_field([0.0] * 3)
    place_orientation: List[float] = None
    place_position_ranges: List[List[float]] = None
    

@dataclass
class StackObjectsConfig(BehaviorConfig):
    behavior_type: str = 'StackObjects'
    objects: List[float] = field(default_factory=list)
    base_obj_position: List[float] = None
    base_obj_position_ranges: List[List[float]] = None
    # TODO need to handle these better, need to add to obj config so you can disallow some
    # outright based on obj geometry, but maybe still have options based on behavior
    allow_top_grasps: bool = True
    allow_bottom_grasps: bool = False  # Usually in collision with supporting surface
    allow_side_grasps: bool = True
    remove_aligned_short_bb_face: bool = False
    stack_buffer: float = 0.0

    
@dataclass
class PushObjectConfig(BehaviorConfig):
    target_object: str = ''
    target_position: List[float] = func_util.lambda_field([None] * 3)
    target_position_ranges: List[List[float]] = None
    obj_offset_dist: float = 0.0
    push_height_offset: float = 0.0
    push_height_offset_range: List[float] = None
    push_height_offset_choices: List[float] = None
    min_cartesian_pct: float = 0.7
    
    
@dataclass
class ROSTeleopConfig(BehaviorConfig):
    """
    Configuration dataclass for ROS teleoperation behavior.
    """
    joint_cmd_topic: str = ''


@dataclass
class TaskConfig(func_util.Config):
    """
    Configuration dataclass for task configuration including hierarchical behavior
    configurations and attributes applied to the task generally.
    """
    task_type: str = ''
    # Assume main entry point is a single behavior
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    # These are for passing sim state around to behaviors, you probably only want
    # to include the images in state if you're using a learning-based behavior
    include_rgb_in_state: bool = False
    include_depth_in_state: bool = False
    extra_steps: int = 100  # Extra steps to take after behaviors have completed
    

@dataclass
class SimulatorConfig(func_util.Config):
    """
    Configuration dataclass for the simulator including attributes for the physics
    enginge and torch settings and such.
    """
    device: str = 'cpu'
    dt: float = 1./60.
    substeps: int = 2
    asset_root: str = DEFAULT_ASSET_ROOT
    render_graphics: bool = True
    physics_engine: str = 'SIM_PHYSX'
    use_gpu: bool = False
    n_threads: int = 0
    compute_device_id: int = 0
    graphics_device_id: int = 0


@dataclass
class SessionConfig(func_util.Config):
    """
    Configuration dataclass for a session including task, simulation environment,
    and simulator configurations, as well as general attributes to be applied
    for the session, e.g. data logging, publishing to ROS, etc.
    """
    task: TaskConfig = field(default_factory=TaskConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    sim: SimulatorConfig = field(default_factory=SimulatorConfig)
    data_root: str = None
    data_prefix: str = 'demo'
    n_envs: int = 1
    n_demos: int = 1
    n_steps: int = -1
    open_loop: bool = False
    randomize_robot: bool = True
    run_forever: bool = False
    publish_ros: bool = False
    demo: str = None
    device: str = 'cpu'


def get_behavior_config_class(config_type):
    """
    Retrieves the behavior config class dynamically.
    """
    BehaviorConfigClass = None
    MODULES = BEHAVIOR_MODULES + BEHAVIOR_CONFIG_MODULES

    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        BehaviorConfigClass = getattr(module, config_type, None)
        if BehaviorConfigClass is not None:
            break

    if BehaviorConfigClass is None:
        raise ValueError(f"Unknown behavior config type: {config_type}")

    return BehaviorConfigClass



if __name__ == '__main__':
    # Simple sanity checks

    obj_config_1 = ObjectConfig(object_type='box', name='my_box', x_extent=1.0, y_extent=2.0,
                                z_extent=3.0, density=0.5, rgb_color=(0.1, 0.2, 0.3),
                                position_x=0.1, position_x_lower=-0.1, position_x_upper=0.1,
                                position_y=0.2, position_y_lower=-0.2, position_y_upper=0.2,
                                position_z=0.3, position_z_lower=-0.3, position_z_upper=0.3,
                                orientation=(0,1,0,0), sample_axis=(1,0,0), sample_angle_lower=-0.6,
                                sample_angle_upper=0.6, fix_base_link=True)
    obj_config_2 = ObjectConfig(config_dict=obj_config_1.to_dict())
    assert obj_config_1 == obj_config_2

    cam_config_1 = CameraConfig(origin=(1,2,3), target=(4,5,6), width=256, height=256, sim_handle=1)
    cam_config_2 = CameraConfig(config_dict=cam_config_1.to_dict())
    assert cam_config_1 == cam_config_2

    robot_config_1 = PandaConfig(name='my_panda', stiffness=[100.0]*7, damping=[10.0]*7)
    robot_config_2 = PandaConfig(config_dict=robot_config_1.to_dict())
    assert robot_config_1 == robot_config_2

    env_config_1 = EnvironmentConfig()
    env_config_1.robots = {'panda': robot_config_1}
    env_config_1.objects = {'box': obj_config_1}
    env_config_1.sensors = {'camera': cam_config_1}
    env_config_2 = EnvironmentConfig(config_dict=env_config_1.to_dict())
    assert env_config_1 == env_config_2

    task_config_1 = TaskConfig()
    task_config_1.task_type = 'grasp_object'
    task_config_2 = TaskConfig(config_dict=task_config_1.to_dict())
    assert task_config_1 == task_config_2

    session_config_1 = SessionConfig()
    session_config_1.task = task_config_1
    session_config_1.env = env_config_1
    session_config_2 = SessionConfig(config_dict=session_config_1.to_dict())
    assert session_config_1 == session_config_2

    behavior_config_1 = BehaviorConfig(behavior_type="b1", target_object="obj1")
    behavior_config_1.behaviors = [BehaviorConfig(behavior_type="b2", target_object="obj2")]
    behavior_config_2 = BehaviorConfig(config_dict=behavior_config_1.to_dict())
    assert behavior_config_1 == behavior_config_2

    print("\nAll tests passed\n")
