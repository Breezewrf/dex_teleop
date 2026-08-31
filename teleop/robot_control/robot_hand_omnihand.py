"""VR hand retargeting and ZMQ publishing for OmniHandPro sim and hardware."""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Optional

import numpy as np
import zmq


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
        zmq_left_real_port: int = 5555,
        zmq_right_real_port: int = 5556,
        bind_host: str = "*",
        connect_delay: float = 0.5,
        retargeting_type: str = "dexpilot",
        publish_sim: bool = True,
        publish_real: bool = False,
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
        if not publish_sim and not publish_real:
            raise ValueError("At least one OmniHand ZMQ output must be enabled")
        self.context = zmq.Context()
        self.left_socket = None
        self.right_socket = None
        self.left_real_socket = None
        self.right_real_socket = None
        self.closed = False
        self._target_hand_visualization: dict[str, dict] = {}

        try:
            if publish_sim:
                self.left_socket = self._bind_publisher(bind_host, zmq_left_port)
                self.right_socket = self._bind_publisher(bind_host, zmq_right_port)
            if publish_real:
                self.left_real_socket = self._bind_publisher(
                    bind_host, zmq_left_real_port
                )
                self.right_real_socket = self._bind_publisher(
                    bind_host, zmq_right_real_port
                )
        except Exception:
            self.close()
            raise

        if publish_sim:
            LOGGER.info(
                "OmniHand simulation publishers bound to ports %d/%d",
                zmq_left_port,
                zmq_right_port,
            )
        if publish_real:
            LOGGER.info(
                "OmniHand physical-hand publishers bound to ports %d/%d",
                zmq_left_real_port,
                zmq_right_real_port,
            )
        if connect_delay > 0.0:
            time.sleep(connect_delay)

    def _bind_publisher(self, bind_host: str, port: int):
        socket = self.context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(f"tcp://{bind_host}:{port}")
        except Exception:
            socket.close()
            raise
        return socket

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
        message_type: str,
        target_hand: Optional[dict] = None,
    ) -> dict:
        message = {
            "timestamp": timestamp,
            "qpos": qpos.tolist(),
            "joint_names": joint_names,
            "type": message_type,
        }
        if target_hand is not None:
            message["target_hand"] = target_hand
        return message

    def update(self, tele_data) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retarget and publish each valid hand independently."""
        left_q = self._retarget(np.asarray(tele_data.left_hand_pos), "left")
        right_q = self._retarget(np.asarray(tele_data.right_hand_pos), "right")
        timestamp = time.time()

        if left_q is not None and self.left_socket is not None:
            self.left_socket.send_json(
                self._message(
                    left_q,
                    self.hand_retargeting.left_omnihand_api_joint_names,
                    timestamp,
                    "sim2sim",
                    self._target_hand_visualization.get("left"),
                )
            )
        if right_q is not None and self.right_socket is not None:
            self.right_socket.send_json(
                self._message(
                    right_q,
                    self.hand_retargeting.right_omnihand_api_joint_names,
                    timestamp,
                    "sim2sim",
                    self._target_hand_visualization.get("right"),
                )
            )
        if left_q is not None and self.left_real_socket is not None:
            self.left_real_socket.send_json(
                self._message(
                    left_q,
                    self.hand_retargeting.left_omnihand_api_joint_names,
                    timestamp,
                    "sim2real",
                )
            )
        if right_q is not None and self.right_real_socket is not None:
            self.right_real_socket.send_json(
                self._message(
                    right_q,
                    self.hand_retargeting.right_omnihand_api_joint_names,
                    timestamp,
                    "sim2real",
                )
            )
        return left_q, right_q

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for socket in (
            self.left_socket,
            self.right_socket,
            self.left_real_socket,
            self.right_real_socket,
        ):
            if socket is not None:
                socket.close()
        self.context.term()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the hand-only OmniHand teleoperation CLI."""
    parser = argparse.ArgumentParser(
        description="Standalone VR teleoperation for dual OmniHandPro hands."
    )
    parser.add_argument(
        "--retargeting",
        choices=("dexpilot", "vector"),
        default="dexpilot",
        help="Hand retargeting optimizer (default: dexpilot).",
    )
    parser.add_argument("--backend", choices=("mujoco", "real"), default="mujoco")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--zmq-left-port", type=int, default=5560)
    parser.add_argument("--zmq-right-port", type=int, default=5561)
    parser.add_argument("--zmq-left-real-port", type=int, default=5555)
    parser.add_argument("--zmq-right-real-port", type=int, default=5556)
    parser.add_argument(
        "--bind-host",
        default="*",
        help="ZMQ publisher bind host (default: all interfaces).",
    )
    parser.add_argument(
        "--connect-delay",
        type=float,
        default=0.5,
        help="Seconds to wait after binding publishers (default: 0.5).",
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Start tracking without waiting for Enter.",
    )
    return parser


def run_standalone(args: argparse.Namespace) -> None:
    """Run VR-to-OmniHand retargeting without initializing a robot arm."""
    if args.frequency <= 0.0:
        raise ValueError(f"frequency must be positive, got {args.frequency}")
    if args.connect_delay < 0.0:
        raise ValueError(
            f"connect-delay must be non-negative, got {args.connect_delay}"
        )

    from televuer import TeleVuerWrapper

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    tv_wrapper = None
    controller = None
    try:
        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=True,
            binocular=True,
            img_shape=(480, 1280),
            display_fps=args.frequency,
            display_mode="pass-through",
        )
        controller = OmniHandController(
            zmq_left_port=args.zmq_left_port,
            zmq_right_port=args.zmq_right_port,
            zmq_left_real_port=args.zmq_left_real_port,
            zmq_right_real_port=args.zmq_right_real_port,
            bind_host=args.bind_host,
            connect_delay=args.connect_delay,
            retargeting_type=args.retargeting,
            publish_sim=args.backend == "mujoco",
            publish_real=args.backend == "real",
        )

        if not args.start_immediately:
            input("Press Enter to start OmniHand VR tracking. Press Ctrl+C to stop.\n")

        period = 1.0 / args.frequency
        LOGGER.info(
            "Started standalone OmniHand teleop with retargeting=%s",
            args.retargeting,
        )
        while not stop:
            start = time.monotonic()
            tele_data = tv_wrapper.get_tele_data()
            if tele_data is not None:
                controller.update(tele_data)
            time.sleep(max(0.0, period - (time.monotonic() - start)))
    finally:
        if controller is not None:
            controller.close()
        if tv_wrapper is not None:
            tv_wrapper.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_standalone(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
