"""Numerical controller contracts, independent of simulation and joystick hardware."""

from pathlib import Path

import numpy as np
import pytest

from walter_mjlab.controllers import (
    HeuristicJoystickController,
    JointTargets,
    JoystickInput,
    JoystickMapping,
    targets_to_actions,
)


def test_differential_wheels_follow_named_joint_order() -> None:
    controller = HeuristicJoystickController()
    targets = controller.step(JoystickInput(forward=0.25, yaw=0.5), 0.02)
    np.testing.assert_allclose(targets.wheel_velocity, [-15, -15, 45, 45, -15, -15, 45, 45])


def test_integrated_targets_are_independent_of_control_frequency() -> None:
    inputs = JoystickInput(knee_forward=0.2, knee_yaw=-0.1, hip_splay=0.1)
    slow = HeuristicJoystickController()
    fast = HeuristicJoystickController()
    slow_targets = slow.step(inputs, 0.1)
    for _ in range(10):
        fast_targets = fast.step(inputs, 0.01)
    np.testing.assert_allclose(slow_targets.knee_position, fast_targets.knee_position)
    np.testing.assert_allclose(slow_targets.hip_position, fast_targets.hip_position)


def test_posture_snaps_measured_knees_and_respects_training_hip_limit() -> None:
    controller = HeuristicJoystickController()
    measured = np.zeros(16)
    measured[4:8] = [3.0, -3.0, 6.0, -6.0]
    targets = controller.step(JoystickInput(posture="wide"), 0.02, joint_positions=measured)
    np.testing.assert_allclose(targets.knee_position, [np.pi, -np.pi, 2 * np.pi, -2 * np.pi])
    np.testing.assert_allclose(targets.hip_position, [-1.25, -1.25, 1.25, 1.25])


def test_action_inverse_preserves_absolute_hips_relative_knees_and_wheels() -> None:
    positions = np.arange(16, dtype=float)
    targets = JointTargets(np.full(4, 0.625), positions[4:8] + 2.75, np.full(8, -30.0))
    actions = targets_to_actions(targets, positions)
    np.testing.assert_allclose(actions, [0.5] * 8 + [-0.5] * 8)


def test_reset_and_output_ownership() -> None:
    controller = HeuristicJoystickController()
    controller.step(JoystickInput(knee_forward=1), 1)
    controller.reset(np.zeros(16))
    targets = controller.step(JoystickInput(), 0.02)
    targets.knee_position[:] = 99
    np.testing.assert_array_equal(controller.step(JoystickInput(), 0.02).knee_position, 0)


@pytest.mark.parametrize("dt", [0, -1, float("nan")])
def test_invalid_dt_fails(dt: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        HeuristicJoystickController().step(JoystickInput(), dt)


def test_invalid_action_state_fails() -> None:
    targets = JointTargets(np.zeros(4), np.zeros(4), np.zeros(8))
    with pytest.raises(ValueError, match="joint_positions"):
        targets_to_actions(targets, np.zeros(15))


def test_mapping_json_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "joystick.json"
    path.write_text('{"knee_forward_axis": 3, "knee_yaw_axis": 2}')
    mapping = JoystickMapping.from_json(path)
    assert mapping.knee_forward_axis == 3
    assert mapping.forward_axis == 1


def test_joystick_reset_edges_and_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    from walter_mjlab.controllers import PygameJoystick

    buttons = [0] * 8
    events: list[SimpleNamespace] = []
    device = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        get_instance_id=lambda: 12,
        get_numaxes=lambda: 5,
        get_numbuttons=lambda: 8,
        get_numhats=lambda: 1,
        get_axis=lambda index: 0.1,
        get_button=lambda index: buttons[index],
        get_hat=lambda index: (0, 0),
        get_init=lambda: True,
    )
    pygame = SimpleNamespace(
        display=SimpleNamespace(init=lambda: None, quit=lambda: None),
        joystick=SimpleNamespace(
            init=lambda: None, quit=lambda: None, get_count=lambda: 1, Joystick=lambda index: device
        ),
        event=SimpleNamespace(get=lambda: events),
        JOYDEVICEREMOVED=1,
        QUIT=2,
    )
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    joystick = PygameJoystick()
    try:
        buttons[7] = 1
        assert joystick.poll().reset
        assert not joystick.poll().reset
        assert joystick.poll().forward == 0
        buttons[7] = 0
        joystick.poll()
        buttons[7] = 1
        assert joystick.poll().reset
        events.append(SimpleNamespace(type=1, instance_id=12))
        with pytest.raises(RuntimeError, match="disconnected"):
            joystick.poll()
    finally:
        joystick.close()


def test_positive_yaw_has_positive_differential_drive_angular_velocity() -> None:
    targets = HeuristicJoystickController().step(JoystickInput(yaw=0.5), 0.02)
    left_speed = targets.wheel_velocity[[0, 1, 4, 5]].mean()
    right_speed = targets.wheel_velocity[[2, 3, 6, 7]].mean()
    wheel_radius = 0.1
    track_width = 0.5
    assert wheel_radius * (right_speed - left_speed) / track_width > 0
    assert JoystickMapping().yaw_sign == -1
