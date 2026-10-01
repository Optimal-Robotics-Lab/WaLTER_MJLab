"""Optional pygame input adapter; no joystick is initialized during import."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from walter_mjlab.controllers.heuristic import JoystickInput, Posture


@dataclass(frozen=True)
class JoystickMapping:
    """Default Logitech/XInput mapping; JSON keys override these fields.

    Left stick drives wheels, right stick integrates knees, hat Y splays hips,
    left bumper folds front hips. A/B/Y/X select neutral/wide/splay/crouch.
    Start requests reset on a rising edge; Back requests quit. Axis signs follow
    the GenZero driving feel: forward = minus left-stick Y, yaw = minus X.
    Semantic positive yaw follows body +Z; stick-right requests a negative yaw.
    """

    forward_axis: int = 1
    yaw_axis: int = 0
    knee_forward_axis: int = 4
    knee_yaw_axis: int = 3
    forward_sign: float = -1.0
    yaw_sign: float = -1.0
    knee_forward_sign: float = -1.0
    knee_yaw_sign: float = 1.0
    knee_forward_offset: float = 0.0
    hip_hat: int = 0
    front_hip_button: int = 4
    neutral_button: int = 0
    wide_button: int = 1
    crouch_button: int = 2
    splay_button: int = 3
    reset_button: int = 7
    quit_button: int = 6
    deadzone: float = 0.2

    def __post_init__(self) -> None:
        if not 0 <= self.deadzone < 1:
            raise ValueError("deadzone must be in [0, 1).")
        for name, value in vars(self).items():
            if name.endswith(("_axis", "_button", "_hat")):
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a nonnegative integer.")
            if name.endswith("_sign") and value not in (-1, 1):
                raise ValueError(f"{name} must be -1 or 1.")
        if not -1 <= self.knee_forward_offset <= 1:
            raise ValueError("knee_forward_offset must be finite and in [-1, 1].")

    @classmethod
    def from_json(cls, path: str | Path) -> "JoystickMapping":
        """Load a partial mapping; unknown fields fail instead of being ignored."""
        with Path(path).open() as mapping_file:
            values = json.load(mapping_file)
        if not isinstance(values, dict):
            raise ValueError("Joystick mapping JSON must contain an object.")
        return cls(**values)


class PygameJoystick:
    """Read a joystick, raising RuntimeError on removal instead of retaining commands."""

    def __init__(self, device_index: int = 0, mapping: JoystickMapping | None = None) -> None:
        # MJLab's Viser viewer owns the browser window. SDL still has to poll the
        # gamepad while that window is in the background.
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError(
                "Joystick input requires pygame: install the joystick extra."
            ) from error
        self.mapping = mapping or JoystickMapping()
        self._pygame = pygame
        pygame.display.init()
        pygame.joystick.init()
        if not 0 <= device_index < pygame.joystick.get_count():
            self.close()
            raise RuntimeError(f"Joystick index {device_index} is unavailable.")
        self._device = pygame.joystick.Joystick(device_index)
        self._device.init()
        self._instance_id = self._device.get_instance_id()
        self._reset_held = False
        for name, value in vars(self.mapping).items():
            count = None
            if name.endswith("_axis"):
                count = self._device.get_numaxes()
            elif name.endswith("_button"):
                count = self._device.get_numbuttons()
            elif name.endswith("_hat"):
                count = self._device.get_numhats()
            if count is not None and value >= count:
                self.close()
                raise ValueError(f"Mapping {name}={value} exceeds device count {count}.")

    @property
    def name(self) -> str:
        """Connected device name reported by SDL."""
        return self._device.get_name()

    def _axis(self, index: int, sign: float, offset: float = 0.0) -> float:
        value = sign * (self._device.get_axis(index) + offset)
        return 0.0 if abs(value) < self.mapping.deadzone else max(-1.0, min(1.0, value))

    def poll(self) -> JoystickInput:
        """Pump input events once; disconnects terminate the caller's rollout cleanly."""
        quit_requested = False
        for event in self._pygame.event.get():
            if (
                event.type == self._pygame.JOYDEVICEREMOVED
                and event.instance_id == self._instance_id
            ):
                raise RuntimeError("Joystick disconnected; stop the rollout.")
            if event.type == self._pygame.QUIT:
                quit_requested = True
        if not self._device.get_init():
            raise RuntimeError("Joystick is no longer initialized; stop the rollout.")
        mapping = self.mapping
        button = self._device.get_button
        reset_held = bool(button(mapping.reset_button))
        reset_requested = reset_held and not self._reset_held
        self._reset_held = reset_held
        posture: Posture | None = None
        presets: tuple[tuple[Posture, int], ...] = (
            ("neutral", mapping.neutral_button),
            ("wide", mapping.wide_button),
            ("splay", mapping.splay_button),
            ("crouch", mapping.crouch_button),
        )
        for candidate, index in presets:
            if button(index):
                posture = candidate
                break
        return JoystickInput(
            forward=self._axis(mapping.forward_axis, mapping.forward_sign),
            yaw=self._axis(mapping.yaw_axis, mapping.yaw_sign),
            knee_forward=self._axis(
                mapping.knee_forward_axis, mapping.knee_forward_sign, mapping.knee_forward_offset
            ),
            knee_yaw=self._axis(mapping.knee_yaw_axis, mapping.knee_yaw_sign),
            hip_splay=float(self._device.get_hat(mapping.hip_hat)[1]),
            front_hip_fold=float(button(mapping.front_hip_button)),
            posture=posture,
            reset=reset_requested,
            quit=quit_requested or bool(button(mapping.quit_button)),
        )

    def close(self) -> None:
        """Release the joystick and SDL subsystems opened by this adapter."""
        if hasattr(self, "_device"):
            self._device.quit()
        self._pygame.joystick.quit()
        self._pygame.display.quit()
