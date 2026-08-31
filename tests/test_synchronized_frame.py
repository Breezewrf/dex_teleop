import json
import unittest

import numpy as np

from teleop.robot_control.synchronized_frame_util import build_synchronized_frame
from teleop.robot_control.synchronized_frame_receiver import (
    build_arg_parser as build_receiver_arg_parser,
)
from teleop.robot_control.vr_arm_hand_teleop import build_arg_parser


class SynchronizedFrameTest(unittest.TestCase):
    def build_frame(self, mode="sim2sim", right_q=None):
        return build_synchronized_frame(
            frame_id=7,
            mode=mode,
            robot="x2",
            hand_type="omnihand",
            timestamp_ns=1_800_000_000,
            monotonic_ns=900_000_000,
            capture_timestamp_ns=1_700_000_000,
            arm_joint_names=("left_arm", "right_arm"),
            arm_qpos=np.array([0.1, -0.2]),
            arm_tauff=np.array([1.0, 2.0]),
            left_hand_joint_names=("L_joint_0", "L_joint_1"),
            left_hand_qpos=np.array([0.3, 0.4]),
            right_hand_joint_names=("R_joint_0", "R_joint_1"),
            right_hand_qpos=right_q,
            left_wrist_pose=np.eye(4),
            right_wrist_pose=np.eye(4),
            left_hand_landmarks=np.zeros((25, 3)),
            right_hand_landmarks=np.ones((25, 3)),
        )

    def test_frame_is_atomic_and_json_serializable(self):
        frame = self.build_frame(right_q=np.array([0.5, 0.6]))
        json.dumps(frame)
        self.assertEqual(frame["type"], "synchronized_teleop_frame")
        self.assertEqual(frame["schema_version"], 1)
        self.assertEqual(frame["frame_id"], 7)
        self.assertEqual(frame["mode"], "sim2sim")
        self.assertTrue(frame["arm"]["valid"])
        self.assertTrue(frame["left_hand"]["valid"])
        self.assertTrue(frame["right_hand"]["valid"])
        self.assertEqual(frame["arm"]["qpos"], [0.1, -0.2])
        self.assertEqual(frame["left_hand"]["qpos"], [0.3, 0.4])
        self.assertEqual(frame["right_hand"]["qpos"], [0.5, 0.6])

    def test_invalid_tracking_keeps_frame_structure(self):
        frame = self.build_frame(mode="sim2real")
        self.assertEqual(frame["mode"], "sim2real")
        self.assertFalse(frame["right_hand"]["valid"])
        self.assertEqual(frame["right_hand"]["qpos"], [])
        self.assertEqual(frame["right_hand"]["joint_names"], ["R_joint_0", "R_joint_1"])

    def test_joint_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_synchronized_frame(
                frame_id=0,
                mode="sim2sim",
                robot="x2",
                hand_type="none",
                timestamp_ns=1,
                monotonic_ns=1,
                capture_timestamp_ns=1,
                arm_joint_names=("a", "b"),
                arm_qpos=np.array([0.0]),
                arm_tauff=None,
                left_hand_joint_names=(),
                left_hand_qpos=None,
                right_hand_joint_names=(),
                right_hand_qpos=None,
                left_wrist_pose=np.eye(4),
                right_wrist_pose=np.eye(4),
                left_hand_landmarks=np.zeros((25, 3)),
                right_hand_landmarks=np.zeros((25, 3)),
            )

    def test_integrated_cli_keeps_sync_frame_stream_opt_in(self):
        defaults = build_arg_parser().parse_args([])
        self.assertFalse(defaults.sync_frame_enable_zmq)
        args = build_arg_parser().parse_args(
            [
                "--sync-frame-enable-zmq",
                "--sync-frame-zmq-bind-host",
                "127.0.0.1",
                "--sync-frame-zmq-port",
                "9000",
            ]
        )
        self.assertTrue(args.sync_frame_enable_zmq)
        self.assertEqual(args.sync_frame_zmq_bind_host, "127.0.0.1")
        self.assertEqual(args.sync_frame_zmq_port, 9000)

    def test_receiver_cli(self):
        args = build_receiver_arg_parser().parse_args(
            ["--endpoint", "tcp://127.0.0.1:9000", "--count", "2"]
        )
        self.assertEqual(args.endpoint, "tcp://127.0.0.1:9000")
        self.assertEqual(args.count, 2)


if __name__ == "__main__":
    unittest.main()
