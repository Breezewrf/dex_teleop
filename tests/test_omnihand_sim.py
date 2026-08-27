import os
import unittest
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.mujoco_receiver import MuJoCoReceiver
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.robot_control.robot_hand_omnihand import (
    UNITREE_TO_OMNIHAND_ROTATION,
    OmniHandController,
    unitree_to_omnihand_points,
)
from teleop.robot_control.vr_arm_hand_teleop import build_arg_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml"


class OmniHandRetargetingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_cwd = Path.cwd()
        os.chdir(PROJECT_ROOT)
        cls.retargeting = HandRetargeting(HandType.OMNIHAND)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.previous_cwd)

    def test_active_joint_mapping_has_twelve_joints_per_hand(self):
        self.assertEqual(len(self.retargeting.left_dex_retargeting_to_hardware), 12)
        self.assertEqual(len(self.retargeting.right_dex_retargeting_to_hardware), 12)
        self.assertEqual(len(self.retargeting.left_omnihand_api_joint_names), 12)
        self.assertEqual(len(self.retargeting.right_omnihand_api_joint_names), 12)
        self.assertFalse(
            any("dip" in name for name in self.retargeting.left_omnihand_api_joint_names)
        )

    def test_coordinate_conversion_is_a_proper_rotation(self):
        np.testing.assert_allclose(
            UNITREE_TO_OMNIHAND_ROTATION.T @ UNITREE_TO_OMNIHAND_ROTATION,
            np.eye(3),
        )
        self.assertAlmostEqual(np.linalg.det(UNITREE_TO_OMNIHAND_ROTATION), 1.0)

        unitree_axes = np.eye(3)
        expected_omnihand_axes = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
            ]
        )
        np.testing.assert_allclose(
            unitree_to_omnihand_points(unitree_axes),
            expected_omnihand_axes,
        )

    def test_omnihand_open_pose_does_not_retarget_to_a_grasp(self):
        # Generate exact open-pose tip landmarks from OmniHand FK, transform
        # them back to TeleVuer's Unitree convention, then exercise the public
        # controller boundary.  This catches the original frame-mismatch bug.
        retargeting = HandRetargeting(HandType.OMNIHAND)
        controller = OmniHandController.__new__(OmniHandController)
        controller.hand_retargeting = retargeting

        for side in ("left", "right"):
            seq = getattr(retargeting, f"{side}_retargeting")
            robot = seq.optimizer.robot
            prefix = "L" if side == "left" else "R"
            link_names = [f"{prefix}_palm"] + [
                f"{prefix}_{finger}_tip"
                for finger in ("thumb", "index", "middle", "ring", "pinky")
            ]
            robot.compute_forward_kinematics(robot.q0)
            omnihand_keypoints = np.stack(
                [
                    robot.get_link_pose(robot.get_link_index(name))[:3, 3]
                    for name in link_names
                ]
            )

            # Compensate the configured optimizer scale so its scaled target
            # vectors still describe the exact robot zero pose.
            wrist = omnihand_keypoints[0]
            target_keypoints = wrist + (
                omnihand_keypoints - wrist
            ) / seq.optimizer.scaling
            unitree_keypoints = target_keypoints @ UNITREE_TO_OMNIHAND_ROTATION
            landmarks = np.zeros((25, 3), dtype=np.float64)
            landmarks[[0, 4, 9, 14, 19, 24]] = unitree_keypoints

            active_q = controller._retarget(landmarks, side)
            self.assertIsNotNone(active_q)
            self.assertLess(
                np.max(np.abs(active_q)),
                0.06,
                msg=f"{side} open pose was incorrectly solved as a grasp: {active_q}",
            )

    def test_coordinate_conversion_rejects_non_xyz_input(self):
        with self.assertRaises(ValueError):
            unitree_to_omnihand_points(np.zeros((25, 2)))

    def test_synthetic_landmarks_produce_finite_active_targets(self):
        landmarks = np.zeros((25, 3), dtype=np.float64)
        directions = np.array(
            [
                [0.4, -0.7, 0.5],
                [0.2, -0.3, 1.0],
                [0.0, -0.1, 1.1],
                [-0.2, 0.1, 1.0],
                [-0.4, 0.3, 0.9],
            ]
        )
        for finger, direction in enumerate(directions):
            direction = direction / np.linalg.norm(direction)
            start = finger * 5
            for segment in range(1, 5):
                landmarks[start + segment] = direction * (0.025 * segment)

        controller = OmniHandController.__new__(OmniHandController)
        controller.hand_retargeting = self.retargeting
        left_q = controller._retarget(landmarks, "left")
        right_q = controller._retarget(landmarks, "right")
        self.assertEqual(left_q.shape, (12,))
        self.assertEqual(right_q.shape, (12,))
        self.assertTrue(np.all(np.isfinite(left_q)))
        self.assertTrue(np.all(np.isfinite(right_q)))
        for side in ("left", "right"):
            payload = controller._target_hand_visualization[side]
            self.assertEqual(np.asarray(payload["landmarks"]).shape, (25, 3))
            self.assertEqual(len(payload["skeleton_edges"]), 24)
            self.assertEqual(len(payload["constraint_vectors"]), 15)
            self.assertEqual(len(payload["constraint_projected"]), 15)
            np.testing.assert_allclose(payload["landmarks"][0], np.zeros(3))

    def test_cli_accepts_omnihand(self):
        args = build_arg_parser().parse_args(
            [
                "--robot",
                "x2",
                "--hand",
                "omnihand",
                "--omnihand-retargeting",
                "vector",
            ]
        )
        self.assertEqual(args.hand, "omnihand")
        self.assertEqual(args.omnihand_retargeting, "vector")

    def test_vector_config_builds_with_intermediate_links(self):
        vector_retargeting = HandRetargeting(HandType.OMNIHAND_VECTOR)
        for side in ("left", "right"):
            seq = getattr(vector_retargeting, f"{side}_retargeting")
            self.assertEqual(seq.optimizer.retargeting_type, "VECTOR")
            self.assertEqual(seq.optimizer.target_link_human_indices.shape, (2, 30))
            self.assertEqual(len(seq.optimizer.origin_link_names), 30)
            self.assertEqual(len(seq.optimizer.task_link_names), 30)
            self.assertEqual(
                len(getattr(vector_retargeting, f"{side}_dex_retargeting_to_hardware")),
                12,
            )
            full_q = seq.retarget(np.zeros((30, 3), dtype=np.float64))
            active_q = full_q[
                getattr(vector_retargeting, f"{side}_dex_retargeting_to_hardware")
            ]
            self.assertEqual(active_q.shape, (12,))
            self.assertTrue(np.all(np.isfinite(active_q)))
            payload = vector_retargeting.target_hand_visualization(
                np.zeros((25, 3), dtype=np.float64),
                side,
            )
            self.assertEqual(payload["optimizer_type"], "vector")
            self.assertEqual(len(payload["constraint_vectors"]), 30)
            self.assertEqual(len(payload["constraint_projected"]), 30)

    def test_visualization_payload_is_config_driven_for_casia(self):
        casia = HandRetargeting(HandType.CASIA_HAND)
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger_start in (0, 5, 10, 15, 20):
            for segment in range(1, 5):
                landmarks[finger_start + segment] = [
                    0.01 * finger_start,
                    0.02 * segment,
                    0.005 * segment,
                ]
        indices = casia.left_indices
        references = landmarks[indices[1]] - landmarks[indices[0]]
        casia.left_retargeting.retarget(references)
        payload = casia.target_hand_visualization(landmarks, "left")
        self.assertEqual(payload["anchor_body_name"], "left_base_link")
        self.assertEqual(
            len(payload["constraint_vectors"]),
            casia.left_indices.shape[1],
        )


class OmniHandMujocoTest(unittest.TestCase):
    def setUp(self):
        self.receiver = MuJoCoReceiver(
            str(SCENE_PATH),
            smoothing_alpha=1.0,
            subscribe_both=True,
            control_mode="position-actuator",
        )

    def tearDown(self):
        self.receiver.close()

    def _joint_qpos(self, name: str) -> float:
        joint_id = mujoco.mj_name2id(
            self.receiver.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        return float(self.receiver.data.qpos[self.receiver.model.jnt_qposadr[joint_id]])

    def test_active_target_does_not_directly_write_passive_qpos(self):
        passive_before = self._joint_qpos("L_index_dip_joint")
        self.receiver.update_target_qpos(
            np.array([0.8]),
            ["L_index_pip_joint"],
        )
        self.assertEqual(self._joint_qpos("L_index_dip_joint"), passive_before)
        actuator_id = self.receiver._joint_to_actuator_id("L_index_pip_joint")
        self.assertAlmostEqual(self.receiver.smoothed_ctrl[actuator_id], 0.8)

    def test_mjcf_non_linear_constraint_drives_passive_joint(self):
        self.receiver.update_target_qpos(
            np.array([0.8]),
            ["L_index_pip_joint"],
        )
        for _ in range(1500):
            self.receiver.step_simulation()

        pip = self._joint_qpos("L_index_pip_joint")
        dip = self._joint_qpos("L_index_dip_joint")
        expected_dip = (
            1.063 * pip
            + 0.08942 * pip**2
            + 0.1845 * pip**3
            - 0.2169 * pip**4
        )
        self.assertGreater(abs(dip), 0.05)
        self.assertAlmostEqual(dip, expected_dip, delta=2e-3)

    def test_invalid_message_is_rejected(self):
        with self.assertRaises(ValueError):
            self.receiver.update_target_qpos(
                np.array([0.1, 0.2]),
                ["L_index_pip_joint"],
            )
        with self.assertRaises(ValueError):
            self.receiver.update_target_qpos(
                np.array([np.nan]),
                ["L_index_pip_joint"],
            )

    def test_target_hand_three_render_modes_and_anchor(self):
        retargeting = HandRetargeting(HandType.OMNIHAND)
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger_start in (0, 5, 10, 15, 20):
            for segment in range(1, 5):
                landmarks[finger_start + segment] = [
                    0.01 * finger_start,
                    0.02 * segment,
                    0.005 * segment,
                ]
        indices = retargeting.left_indices
        omni_points = unitree_to_omnihand_points(landmarks)
        references = omni_points[indices[1]] - omni_points[indices[0]]
        retargeting.left_retargeting.retarget(references)
        payload = retargeting.target_hand_visualization(
            omni_points,
            "left",
        )
        joint_names = retargeting.left_omnihand_api_joint_names
        self.receiver.update_target_hand(payload, joint_names)
        anchor_id = self.receiver.target_hand_payloads[0]["anchor_body_id"]
        self.assertEqual(
            mujoco.mj_id2name(
                self.receiver.model,
                mujoco.mjtObj.mjOBJ_BODY,
                anchor_id,
            ),
            "L_palm",
        )

        scene = mujoco.MjvScene(self.receiver.model, maxgeom=100)
        self.receiver.target_hand_mode = "landmarks"
        self.receiver.render_target_hands(scene)
        self.assertEqual(scene.ngeom, 25)
        self.receiver.target_hand_mode = "skeleton"
        self.receiver.render_target_hands(scene)
        self.assertEqual(scene.ngeom, 25 + 24)
        self.receiver.target_hand_mode = "constraints"
        self.receiver.render_target_hands(scene)
        self.assertEqual(scene.ngeom, 25 + 15)

    def test_invalid_target_hand_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            self.receiver.update_target_hand(
                {"side": "left", "landmarks": [[0.0, 0.0, 0.0]]},
                ["L_index_pip_joint"],
            )

    def test_kinematic_coupled_mode_applies_mjcf_polynomials_immediately(self):
        receiver = MuJoCoReceiver(
            str(SCENE_PATH),
            smoothing_alpha=1.0,
            interpol_steps=1,
            subscribe_both=True,
            control_mode="kinematic-coupled",
        )
        try:
            self.assertEqual(len(receiver.coupled_joint_equalities), 14)

            def qpos(name: str) -> float:
                joint_id = mujoco.mj_name2id(
                    receiver.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
                return float(receiver.data.qpos[receiver.model.jnt_qposadr[joint_id]])

            initial_time = receiver.data.time
            receiver.update_target_qpos(
                np.array([0.8]),
                ["L_ring_mcp_joint"],
            )
            receiver.step_simulation()

            mcp = qpos("L_ring_mcp_joint")
            expected_pip = 0.7869 * mcp + 0.3884 * mcp**2 - 0.4545 * mcp**3 + 0.1578 * mcp**4
            expected_dip = 0.899 * mcp + 0.3138 * mcp**2 - 0.1728 * mcp**3 - 0.03666 * mcp**4
            self.assertAlmostEqual(mcp, 0.8)
            self.assertAlmostEqual(qpos("L_ring_pip_joint"), expected_pip)
            self.assertAlmostEqual(qpos("L_ring_dip_joint"), expected_dip)
            self.assertEqual(receiver.data.time, initial_time)

            receiver.update_target_qpos(
                np.array([0.0]),
                ["L_ring_mcp_joint"],
            )
            receiver.step_simulation()
            self.assertAlmostEqual(qpos("L_ring_mcp_joint"), 0.0)
            self.assertAlmostEqual(qpos("L_ring_pip_joint"), 0.0)
            self.assertAlmostEqual(qpos("L_ring_dip_joint"), 0.0)
        finally:
            receiver.close()


if __name__ == "__main__":
    unittest.main()
