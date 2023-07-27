from isaacgym import gymapi

from ll4ma_util import func_util


def get_default_sim(sim_config):
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = sim_config.dt
    sim_params.substeps = sim_config.substeps
    sim_params.use_gpu_pipeline = sim_config.use_gpu
    if sim_config.physics_engine == gymapi.SIM_PHYSX.name:
        physics_engine = gymapi.SIM_PHYSX
        # TODO can move the rest of these to config
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = sim_config.n_threads
        sim_params.physx.use_gpu = sim_config.use_gpu
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(sim_config.compute_device_id, sim_config.graphics_device_id,
                         physics_engine, sim_params)
    if sim is None:
        raise Exception("Failed to create sim")
    return sim


def get_arm(arm_config, modules=['ll4ma_isaacgym.robots']):
    ArmClass = func_util.get_class(arm_config.arm_type, modules)
    if ArmClass is None:
        raise ValueError(f"Unknown arm type for class creation: {arm_config.arm_type}")
    arm = ArmClass(arm_config)
    return arm


def get_end_effector(ee_config, modules=['ll4ma_isaacgym.robots']):
    ee = None
    if ee_config is not None:
        EEClass = func_util.get_class(ee_config.ee_type, modules)
        if EEClass is None:
            raise ValueError(f"Unknown EE type for class creation: {ee_config.ee_type}")
        ee = EEClass(ee_config)
    return ee
