"""VR hand retargeting and ZMQ publishing for OmniHandPro simulation."""

import logging
import time
from typing import Optional

import numpy as np
import zmq

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType


LOGGER = logging.getLogger(__name__)

# TeleVuer returns hand landmarks in the Unitree hand convention.  OmniHand's
# URDF/MJCF uses a different initial-pose basis:
#
#   Unitree +X -> OmniHand -X
#   Unitree +Y -> OmniHand -Z
#   Unitree +Z -> OmniHand -Y
#
# This is a proper rotation (det=+1), not a reflection.  The same basis change
# applies to both hands; their left/right geometry is already mirrored by the
# landmarks and the corresponding OmniHand model.
UNITREE_TO_OMNIHAND_ROTATION = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def unitree_to_omnihand_points(points: np.ndarray) -> np.ndarray:
    """Change 3-D points from TeleVuer/Unitree to the OmniHand model basis.

    The input may contain any leading dimensions but must end in XYZ.  A new
    float64 array is returned so callers cannot accidentally mutate TeleData.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError(
            "OmniHand coordinate conversion expects an array ending in XYZ; "
            f"got shape {points.shape}"
        )
    return points @ UNITREE_TO_OMNIHAND_ROTATION.T


class OmniHandController:
    """Retarget XR landmarks and publish the 12 active joints per hand."""

    def __init__(
        self,
        zmq_left_port: int = 5560,
        zmq_right_port: int = 5561,
        bind_host: str = "*",
        connect_delay: float = 0.5,
        retargeting_type: str = "dexpilot",
    ) -> None:
        retargeting_types = {
            "dexpilot": HandType.OMNIHAND,
            "vector": HandType.OMNIHAND_VECTOR,
        }
        try:
            hand_type = retargeting_types[retargeting_type.lower()]
        except KeyError as exc:
            raise ValueError(
                "OmniHand retargeting_type must be 'dexpilot' or 'vector'; "
                f"got {retargeting_type!r}"
            ) from exc
        self.hand_retargeting = HandRetargeting(hand_type)
        self.context = zmq.Context()
        self.left_socket = self.context.socket(zmq.PUB)
        self.right_socket = self.context.socket(zmq.PUB)
        self.left_socket.setsockopt(zmq.LINGER, 0)
        self.right_socket.setsockopt(zmq.LINGER, 0)
        self.closed = False
        self._target_hand_visualization: dict[str, dict] = {}

        try:
            self.left_socket.bind(f"tcp://{bind_host}:{zmq_left_port}")
            self.right_socket.bind(f"tcp://{bind_host}:{zmq_right_port}")
        except Exception:
            self.close()
            raise

        LOGGER.info("OmniHand left publisher bound to port %d", zmq_left_port)
        LOGGER.info("OmniHand right publisher bound to port %d", zmq_right_port)
        if connect_delay > 0.0:
            time.sleep(connect_delay)

    @staticmethod
    def _valid_hand_data(hand_data: np.ndarray) -> bool:
        return (
            hand_data.shape == (25, 3)
            and np.all(np.isfinite(hand_data))
            and not np.all(hand_data == 0.0)
        )

    def _retarget(self, hand_data: np.ndarray, side: str) -> Optional[np.ndarray]:
        hand_data = np.asarray(hand_data, dtype=np.float64)
        if not self._valid_hand_data(hand_data):
            getattr(self, "_target_hand_visualization", {}).pop(side, None)
            return None

        if side == "left":
            indices = self.hand_retargeting.left_indices
            retargeting = self.hand_retargeting.left_retargeting
            mapping = self.hand_retargeting.left_dex_retargeting_to_hardware
        elif side == "right":
            indices = self.hand_retargeting.right_indices
            retargeting = self.hand_retargeting.right_retargeting
            mapping = self.hand_retargeting.right_dex_retargeting_to_hardware
        else:
            raise ValueError(f"Unsupported hand side: {side}")

        # HandRetargeting's OmniHand kinematics lives in the OmniHand URDF
        # basis, while TeleData is deliberately normalized to Unitree's hand
        # basis in tv_wrapper.py.  Without this dedicated conversion, an open
        # human hand points across the robot palm and the optimizer resolves
        # the mismatch by driving most flexion joints to their grasp limits.
        omnihand_points = unitree_to_omnihand_points(hand_data)
        reference_vectors = (
            omnihand_points[indices[1, :]] - omnihand_points[indices[0, :]]
        )
        full_q = retargeting.retarget(reference_vectors)
        visualizations = getattr(self, "_target_hand_visualization", None)
        if visualizations is None:
            visualizations = {}
            self._target_hand_visualization = visualizations
        # Build this after retarget() so DexPilot's projected/eta state belongs
        # to the same frame as the joint target being published.
        visualizations[side] = self.hand_retargeting.target_hand_visualization(
            omnihand_points,
            side,
        )
        active_q = np.asarray(full_q[mapping], dtype=np.float64)
        if active_q.shape != (12,) or not np.all(np.isfinite(active_q)):
            LOGGER.warning("Ignoring invalid %s OmniHand retarget output", side)
            return None
        return active_q

    @staticmethod
    def _message(
        qpos: np.ndarray,
        joint_names: list[str],
        timestamp: float,
        target_hand: Optional[dict] = None,
    ) -> dict:
        message = {
            "timestamp": timestamp,
            "qpos": qpos.tolist(),
            "joint_names": joint_names,
            "type": "sim2sim",
        }
        if target_hand is not None:
            message["target_hand"] = target_hand
        return message

    def update(self, tele_data) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retarget and publish each valid hand independently."""
        left_q = self._retarget(np.asarray(tele_data.left_hand_pos), "left")
        right_q = self._retarget(np.asarray(tele_data.right_hand_pos), "right")
        timestamp = time.time()

        if left_q is not None:
            self.left_socket.send_json(
                self._message(
                    left_q,
                    self.hand_retargeting.left_omnihand_api_joint_names,
                    timestamp,
                    self._target_hand_visualization.get("left"),
                )
            )
        if right_q is not None:
            self.right_socket.send_json(
                self._message(
                    right_q,
                    self.hand_retargeting.right_omnihand_api_joint_names,
                    timestamp,
                    self._target_hand_visualization.get("right"),
                )
            )
        return left_q, right_q

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.left_socket.close()
        self.right_socket.close()
        self.context.term()
