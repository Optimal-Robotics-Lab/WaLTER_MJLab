"""Shared-model integration and explicit controller joint ordering."""

from walter_mjlab.robots.walter import (
    HIP_JOINT_NAMES,
    JOINT_NAMES,
    KNEE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    get_robot_cfg,
    get_robot_spec,
)

__all__ = [
    "HIP_JOINT_NAMES",
    "JOINT_NAMES",
    "KNEE_JOINT_NAMES",
    "WHEEL_JOINT_NAMES",
    "get_robot_cfg",
    "get_robot_spec",
]
