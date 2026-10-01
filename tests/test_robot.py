"""CPU compilation checks for the shared model and MJLab actuator ownership."""

import mujoco
import numpy as np
import pytest
from mjlab.entity import Entity
from WaLTER_Modeling import GenerateModel, config_dir

from walter_mjlab.robots import JOINT_NAMES, get_robot_cfg, get_robot_spec
from walter_mjlab.robots.walter import WAIST_JOINT_NAME, validate_robot_model


@pytest.mark.parametrize("profile", ["FS", "WS"])
def test_robot_compiles_with_single_actuator_per_named_joint(profile: str) -> None:
    robot = Entity(get_robot_cfg(profile))
    model = robot.spec.compile()
    validate_robot_model(model, expected_actuators=16)
    targets = [model.joint(int(index)).name for index in model.actuator_trnid[:, 0]]
    assert set(targets) == set(JOINT_NAMES)
    assert len(targets) == len(set(targets))
    assert WAIST_JOINT_NAME not in targets
    assert model.nsensor == 2
    assert model.sensor("imu_acc").type == mujoco.mjtSensor.mjSENS_ACCELEROMETER
    assert model.sensor("imu_gyro").type == mujoco.mjtSensor.mjSENS_GYRO
    limits = dict(zip(targets, model.actuator_forcerange[:, 1], strict=True))
    for name in JOINT_NAMES:
        expected = 47.97 if "wheel" in name else 308.75 if "shin" in name else 504.8
        assert limits[name] == pytest.approx(expected)


@pytest.mark.parametrize("profile", ["FS", "WS"])
def test_adapter_preserves_shared_physics_and_has_no_scene(profile: str) -> None:
    directory = config_dir(full_scale=profile == "FS")
    source = GenerateModel(directory / "model_config.yaml", directory / "motor_config.yaml")
    spec = get_robot_spec(profile)
    model = spec.compile()
    np.testing.assert_allclose(model.dof_armature, source.mj_model.dof_armature)
    np.testing.assert_allclose(model.dof_damping, source.mj_model.dof_damping)
    np.testing.assert_allclose(model.jnt_stiffness, source.mj_model.jnt_stiffness)
    np.testing.assert_allclose(model.body_mass, source.mj_model.body_mass)
    assert model.nu == 0
    assert model.nlight == 0
    assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    assert not np.any(model.geom_bodyid == 0)
    second_spec = get_robot_spec(profile)
    assert second_spec is not spec


def test_unknown_model_profile_fails_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown robot profile"):
        get_robot_cfg("unknown")
    with pytest.raises(ValueError, match="Unknown robot profile"):
        get_robot_spec("unknown")


def test_fs_zero_command_settles_on_plane_with_explicit_pd() -> None:
    """Exercise contact and feedback at the task's 4 ms physics period on CPU."""
    cfg = get_robot_cfg("FS")
    robot = Entity(cfg)
    robot.spec.worldbody.add_geom(
        name="test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1)
    )
    robot.spec.option.timestep = 0.004
    robot.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_ids = model.actuator_trnid[:, 0]
    positions = model.jnt_qposadr[joint_ids]
    velocities = model.jnt_dofadr[joint_ids]
    assert cfg.articulation is not None
    gains = {
        name: (group.stiffness, group.damping)
        for group in cfg.articulation.actuators
        for name in group.target_names_expr
    }
    stiffness, damping = np.array([gains[model.joint(int(index)).name] for index in joint_ids]).T
    effort_limit = model.actuator_forcerange[:, 1]
    for _ in range(1250):
        data.ctrl[:] = np.clip(
            -stiffness * data.qpos[positions] - damping * data.qvel[velocities],
            -effort_limit,
            effort_limit,
        )
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not data.warning.number.any()
    assert 0.26 < data.qpos[2] < 0.30
    assert np.linalg.norm(data.qvel) < 0.01
    assert data.qpos[3] > 0.99
