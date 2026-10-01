"""Portable open-loop controls in joint-side radians and radians per second."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Posture = Literal["neutral", "wide", "splay", "crouch"]


@dataclass(frozen=True)
class JoystickInput:
    """Semantic inputs in [-1, 1]; reset and quit are adapter requests."""

    forward: float = 0.0
    yaw: float = 0.0
    knee_forward: float = 0.0
    knee_yaw: float = 0.0
    hip_splay: float = 0.0
    front_hip_fold: float = 0.0
    posture: Posture | None = None
    reset: bool = False
    quit: bool = False


@dataclass(frozen=True)
class JointTargets:
    """Single-robot targets: hips/knees FL FR BL BR, wheels FL1 FL2 FR1 FR2 BL1 BL2 BR1 BR2."""

    hip_position: FloatArray
    knee_position: FloatArray
    wheel_velocity: FloatArray


@dataclass(frozen=True)
class HeuristicControllerConfig:
    """Rates replace legacy per-physics-step increments; adjust for your experiment."""

    max_wheel_velocity: float = 60.0
    knee_rate: float = 5.0
    hip_rate: float = 0.5
    hip_limit: float = 1.25

    def __post_init__(self) -> None:
        values = (self.max_wheel_velocity, self.knee_rate, self.hip_rate, self.hip_limit)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Controller rates and limits must be finite and positive.")


def _validate_vector(value: FloatArray, size: int, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite array of shape ({size},).")
    return vector


class HeuristicJoystickController:
    """GenZero differential wheels, incremental knees/hips, and four posture presets.

    The controller owns target history, never simulation state. Call reset after each
    episode reset. Positive yaw is counterclockwise about body +Z (left wheels slower than right).
    """

    def __init__(self, config: HeuristicControllerConfig | None = None) -> None:
        self.config = config or HeuristicControllerConfig()
        self.reset(np.zeros(16))

    def reset(self, joint_positions: FloatArray) -> None:
        """Initialize targets from canonical measured joint positions in radians."""
        positions = _validate_vector(joint_positions, 16, "joint_positions")
        self._hips = np.clip(positions[:4], -self.config.hip_limit, self.config.hip_limit)
        self._knees = positions[4:8].copy()

    def step(
        self,
        inputs: JoystickInput,
        dt: float,
        *,
        joint_positions: FloatArray | None = None,
    ) -> JointTargets:
        """Advance target integration once per control period, expressed in seconds.

        Optional measured positions make posture snapping match the legacy behavior;
        without feedback, presets snap the previous commanded knee positions.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        values = np.array(
            [
                inputs.forward,
                inputs.yaw,
                inputs.knee_forward,
                inputs.knee_yaw,
                inputs.hip_splay,
                inputs.front_hip_fold,
            ]
        )
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
            raise ValueError("Analog inputs must be finite and within [-1, 1].")
        if inputs.posture not in (None, "neutral", "wide", "splay", "crouch"):
            raise ValueError(f"Unknown posture: {inputs.posture}")
        knees = self._knees.copy()
        if joint_positions is not None:
            knees = _validate_vector(joint_positions, 16, "joint_positions")[4:8]
        self._hips += (
            self.config.hip_rate
            * dt
            * np.array(
                [
                    -inputs.hip_splay + inputs.front_hip_fold,
                    -inputs.hip_splay + inputs.front_hip_fold,
                    inputs.hip_splay,
                    inputs.hip_splay,
                ]
            )
        )
        self._knees += (
            self.config.knee_rate
            * dt
            * np.array(
                [
                    inputs.knee_forward + inputs.knee_yaw,
                    inputs.knee_forward - inputs.knee_yaw,
                    inputs.knee_forward + inputs.knee_yaw,
                    inputs.knee_forward - inputs.knee_yaw,
                ]
            )
        )
        if inputs.posture is not None:
            angle = {"neutral": 0, "wide": np.pi / 2, "splay": np.pi / 3, "crouch": np.pi / 12}[
                inputs.posture
            ]
            self._hips[:] = [-angle, -angle, angle, angle]
            if inputs.posture != "splay":
                self._knees[:] = np.round(knees / np.pi) * np.pi
            if inputs.posture == "crouch":
                self._knees += np.array([-1, -1, 1, 1]) * np.pi / 4
        self._hips = np.clip(self._hips, -self.config.hip_limit, self.config.hip_limit)
        left = inputs.forward - inputs.yaw
        right = inputs.forward + inputs.yaw
        wheels = self.config.max_wheel_velocity * np.clip(
            [left, left, right, right, left, left, right, right], -1, 1
        )
        return JointTargets(self._hips.copy(), self._knees.copy(), wheels)


def targets_to_actions(
    targets: JointTargets,
    joint_positions: FloatArray,
    *,
    hip_scale: float = 1.25,
    knee_scale: float = 5.5,
    wheel_scale: float = 60.0,
) -> FloatArray:
    """Convert physical targets to the training action contract, clipping to [-1, 1].

    Hips are absolute positions; knees are measured-position deltas held for one
    policy step; wheels are velocities. Input and output are canonical (16,) arrays.
    """
    scales = np.array([hip_scale, knee_scale, wheel_scale])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Action scales must be finite and positive.")
    positions = _validate_vector(joint_positions, 16, "joint_positions")
    hips = _validate_vector(targets.hip_position, 4, "hip_position")
    knees = _validate_vector(targets.knee_position, 4, "knee_position")
    wheels = _validate_vector(targets.wheel_velocity, 8, "wheel_velocity")
    return np.clip(
        np.concatenate(
            (
                hips / hip_scale,
                (knees - positions[4:8]) / knee_scale,
                wheels / wheel_scale,
            )
        ),
        -1,
        1,
    )
