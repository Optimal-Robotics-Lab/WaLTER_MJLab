"""Adapt the pinned shared morphology to MJLab without owning robot geometry."""

from functools import partial
from typing import Literal

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# MuJoCo's compiled public exports are absent from its Python type information.
from mujoco import (
    MjModel,  # pyright: ignore[reportAttributeAccessIssue]
    MjSpec,  # pyright: ignore[reportAttributeAccessIssue]
    mj_name2id,  # pyright: ignore[reportAttributeAccessIssue]
    mjtJoint,  # pyright: ignore[reportAttributeAccessIssue]
    mjtObj,  # pyright: ignore[reportAttributeAccessIssue]
    mjtSensor,  # pyright: ignore[reportAttributeAccessIssue]
)
from WaLTER_Modeling import GenerateModel, config_dir

RobotProfile = Literal["FS", "WS"]

# Policy order is independent of the generator's body and actuator declaration order.
LEG_PREFIXES = ("head_left", "head_right", "torso_left", "torso_right")
HIP_JOINT_NAMES = tuple(f"{leg}_thigh_joint" for leg in LEG_PREFIXES)
KNEE_JOINT_NAMES = tuple(f"{leg}_thigh_shin_joint" for leg in LEG_PREFIXES)
WHEEL_JOINT_NAMES = tuple(
    f"{leg}_shin_{end}_wheel_joint" for leg in LEG_PREFIXES for end in ("front", "rear")
)
JOINT_NAMES = HIP_JOINT_NAMES + KNEE_JOINT_NAMES + WHEEL_JOINT_NAMES
WAIST_JOINT_NAME = "head_joint"

# Initial legacy controller tuning: joint-side Nm, Nm/rad and Nm/(rad/s).
# These static saturation limits are not a calibrated physical torque-speed model.
HIP_EFFORT_LIMIT = 504.8
KNEE_EFFORT_LIMIT = 308.75
WHEEL_EFFORT_LIMIT = 47.97


def validate_robot_model(model: MjModel, *, expected_actuators: int = 0) -> None:
    """Reject topology changes that invalidate the controller's named joint contract."""
    if model.nu != expected_actuators:
        raise ValueError(f"Expected {expected_actuators} actuators; received {model.nu}.")
    for name in (*JOINT_NAMES, WAIST_JOINT_NAME):
        joint_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Required WaLTER joint is missing: {name}.")
        if model.jnt_type[joint_id] != mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected a scalar hinge joint: {name}.")
    root_id = mj_name2id(model, mjtObj.mjOBJ_JOINT, "torso_joint")
    if root_id < 0 or model.jnt_type[root_id] != mjtJoint.mjJNT_FREE:
        raise ValueError("WaLTER requires the torso_joint free base.")
    if model.njnt != len(JOINT_NAMES) + 2:
        raise ValueError("Unexpected joints: revise the robot/control contract explicitly.")


def get_robot_spec(profile: RobotProfile = "FS") -> MjSpec:
    """Build a fresh robot-only spec; MJLab owns all active motor control.

    Shared armatures, passive damping, wheel contacts and the passive waist are
    preserved. The torso IMU uses body-aligned axes: acceleration is specific force
    (m/s²), gyro angular velocity is rad/s. No ground or lights are added here.
    """
    if profile not in ("FS", "WS"):
        raise ValueError(f"Unknown robot profile {profile!r}; choose FS or WS.")
    directory = config_dir(full_scale=profile == "FS")
    spec = GenerateModel(directory / "model_config.yaml", directory / "motor_config.yaml").spec
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    torso = spec.body("torso")
    torso.add_site(name="imu", pos=(0, 0, 0), size=(0.01, 0.01, 0.01))
    for name, sensor_type in (
        ("imu_acc", mjtSensor.mjSENS_ACCELEROMETER),
        ("imu_gyro", mjtSensor.mjSENS_GYRO),
    ):
        spec.add_sensor(name=name, type=sensor_type, objtype=mjtObj.mjOBJ_SITE, objname="imu")
    validate_robot_model(spec.compile())
    return spec


def get_robot_cfg(profile: RobotProfile = "FS") -> EntityCfg:
    """Return independent model and explicit PD configuration for each task instance.

    Shared YAML motor gains are intentionally superseded by the migration control
    profile. Its reflected rotor inertia and passive joint damping remain intact;
    the active derivative gain is additional to that physical passive damping.
    WS compiles for morphology inspection only: these FS gains were unstable in
    a 4 ms WS standstill smoke test and need a separate tuned control profile.
    """
    if profile not in ("FS", "WS"):
        raise ValueError(f"Unknown robot profile {profile!r}; choose FS or WS.")
    actuator_groups = (
        (HIP_JOINT_NAMES, 150.0, 50.0, HIP_EFFORT_LIMIT),
        (KNEE_JOINT_NAMES, 150.0, 50.0, KNEE_EFFORT_LIMIT),
        (WHEEL_JOINT_NAMES, 0.0, 12.0, WHEEL_EFFORT_LIMIT),
    )
    return EntityCfg(
        spec_fn=partial(get_robot_spec, profile),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4 if profile == "FS" else 0.25),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                IdealPdActuatorCfg(
                    target_names_expr=names,
                    stiffness=stiffness,
                    damping=damping,
                    effort_limit=effort_limit,
                )
                for names, stiffness, damping, effort_limit in actuator_groups
            )
        ),
    )
