"""Atomic arm-and-hand frames for dataset recording and synchronization."""

from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np
import zmq


SCHEMA_VERSION = 1


def _finite_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _joint_payload(
    joint_names: Iterable[str],
    qpos: Optional[np.ndarray],
) -> dict:
    names = list(joint_names)
    if qpos is None:
        return {"valid": False, "joint_names": names, "qpos": []}

    values = _finite_array(qpos, "qpos").reshape(-1)
    if values.shape != (len(names),):
        raise ValueError(
            f"Expected {len(names)} joint values, got shape {values.shape}"
        )
    return {
        "valid": True,
        "joint_names": names,
        "qpos": values.tolist(),
    }


def build_synchronized_frame(
    *,
    frame_id: int,
    mode: str,
    robot: str,
    hand_type: str,
    timestamp_ns: int,
    monotonic_ns: int,
    capture_timestamp_ns: int,
    arm_joint_names: Iterable[str],
    arm_qpos: np.ndarray,
    arm_tauff: Optional[np.ndarray],
    left_hand_joint_names: Iterable[str],
    left_hand_qpos: Optional[np.ndarray],
    right_hand_joint_names: Iterable[str],
    right_hand_qpos: Optional[np.ndarray],
    left_wrist_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    left_hand_landmarks: np.ndarray,
    right_hand_landmarks: np.ndarray,
) -> dict:
    """Build one JSON-serializable frame after arm and hand retargeting."""
    if frame_id < 0:
        raise ValueError(f"frame_id must be non-negative, got {frame_id}")
    if mode not in ("sim2sim", "sim2real"):
        raise ValueError(f"Unsupported synchronized-frame mode: {mode}")

    arm = _joint_payload(arm_joint_names, arm_qpos)
    tauff = (
        []
        if arm_tauff is None
        else _finite_array(arm_tauff, "arm_tauff").reshape(-1).tolist()
    )
    arm["tauff"] = tauff

    return {
        "schema_version": SCHEMA_VERSION,
        "type": "synchronized_teleop_frame",
        "mode": mode,
        "frame_id": int(frame_id),
        "timestamp": timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": int(timestamp_ns),
        "monotonic_ns": int(monotonic_ns),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "robot": robot,
        "hand_type": hand_type,
        "arm": arm,
        "left_hand": _joint_payload(left_hand_joint_names, left_hand_qpos),
        "right_hand": _joint_payload(right_hand_joint_names, right_hand_qpos),
        "source": {
            "left_wrist_pose": _finite_array(
                left_wrist_pose, "left_wrist_pose"
            ).tolist(),
            "right_wrist_pose": _finite_array(
                right_wrist_pose, "right_wrist_pose"
            ).tolist(),
            "left_hand_landmarks": _finite_array(
                left_hand_landmarks, "left_hand_landmarks"
            ).tolist(),
            "right_hand_landmarks": _finite_array(
                right_hand_landmarks, "right_hand_landmarks"
            ).tolist(),
        },
    }


class SynchronizedFramePublisher:
    """Optional side-channel PUB socket; it does not replace control sockets."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 8560,
        connect_delay: float = 0.5,
    ) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        # This stream carries control targets: stale frames must be replaced,
        # never queued and replayed with increasing latency.
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.endpoint = f"tcp://{bind_host}:{port}"
        try:
            self.socket.bind(self.endpoint)
        except Exception:
            self.socket.close()
            self.context.term()
            raise
        if connect_delay > 0.0:
            time.sleep(connect_delay)

    def publish(self, frame: dict) -> bool:
        try:
            self.socket.send_json(frame, flags=zmq.NOBLOCK)
        except zmq.Again:
            return False
        return True

    def close(self) -> None:
        self.socket.close()
        self.context.term()
