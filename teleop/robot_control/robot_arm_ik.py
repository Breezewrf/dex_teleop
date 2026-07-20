import os
import pickle
import sys

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.utils.weighted_moving_filter import WeightedMovingFilter


class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False, urdf_path=None, model_dir=None, **kwargs):
        if kwargs:
            logger_mp.warning("[G1_29_ArmIK] Ignoring unsupported IK arguments: %s", sorted(kwargs.keys()))

        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        self.cache_path = os.path.join(parent2_dir, "g1_29_pin_model_cache.pkl")
        self.urdf_path = urdf_path or os.path.join(parent2_dir, "assets/g1/g1_body29_hand14.urdf")
        self.model_dir = model_dir or os.path.join(parent2_dir, "assets/g1")
        self.last_error = {
            "left_pos": np.inf,
            "right_pos": np.inf,
            "left_ori": np.inf,
            "right_ori": np.inf,
        }

        if os.path.exists(self.cache_path) and not self.Visualization:
            logger_mp.info("[G1_29_ArmIK] >>> Loading cached robot model: %s", self.cache_path)
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[G1_29_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
            ]
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            if not os.path.exists(self.cache_path) and not self.Visualization:
                self.save_cache()
                logger_mp.info("[G1_29_ArmIK] >>> Cache saved to %s", self.cache_path)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            frame_axis_positions = (
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            frame_axis_colors = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(position=0.1 * frame_axis_positions, color=frame_axis_colors),
                        mg.LineBasicMaterial(linewidth=20, vertexColors=True),
                    )
                )

    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def _update_last_error(self, q, left_wrist, right_wrist):
        trans = np.array(self.translational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        rot = np.array(self.rotational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        self.last_error = {
            "left_pos": float(np.linalg.norm(trans[:3])),
            "right_pos": float(np.linalg.norm(trans[3:])),
            "left_ori": float(np.linalg.norm(rot[:3])),
            "right_ori": float(np.linalg.norm(rot[3:])),
        }

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q = sol.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data.copy()

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            self._update_last_error(sol_q, left_wrist, right_wrist)

            if self.Visualization:
                self.vis.display(sol_q)

            return sol_q, sol_tauff

        except Exception as exc:
            logger_mp.error("ERROR in convergence, plotting debug info.%s", exc)
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data.copy()

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            self._update_last_error(sol_q, left_wrist, right_wrist)
            logger_mp.error(
                "sol_q:%s \nmotorstate: \n%s \nleft_pose: \n%s \nright_pose: \n%s",
                sol_q,
                current_lr_arm_motor_q,
                left_wrist,
                right_wrist,
            )
            if self.Visualization:
                self.vis.display(sol_q)

            if current_lr_arm_motor_q is not None:
                return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
            return sol_q, sol_tauff


class G1_23_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False, urdf_path=None, model_dir=None, **kwargs):
        if kwargs:
            logger_mp.warning("[G1_23_ArmIK] Ignoring unsupported IK arguments: %s", sorted(kwargs.keys()))

        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        self.cache_path = os.path.join(parent2_dir, "g1_23_pin_model_cache.pkl")
        self.urdf_path = urdf_path or os.path.join(parent2_dir, "assets/g1/g1_body23.urdf")
        self.model_dir = model_dir or os.path.join(parent2_dir, "assets/g1")
        self.last_error = {
            "left_pos": np.inf,
            "right_pos": np.inf,
            "left_ori": np.inf,
            "right_ori": np.inf,
        }

        if os.path.exists(self.cache_path) and not self.Visualization:
            logger_mp.info("[G1_23_ArmIK] >>> Loading cached robot model: %s", self.cache_path)
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[G1_23_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
            ]
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_roll_joint"),
                    pin.SE3(np.eye(3), np.array([0.20, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_roll_joint"),
                    pin.SE3(np.eye(3), np.array([0.20, 0.0, 0.0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            if not os.path.exists(self.cache_path) and not self.Visualization:
                self.save_cache()
                logger_mp.info("[G1_23_ArmIK] >>> Cache saved to %s", self.cache_path)

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + 0.5 * self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 10)
        self.vis = None

        if self.Visualization:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            frame_axis_positions = (
                np.array(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            frame_axis_colors = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(position=0.1 * frame_axis_positions, color=frame_axis_colors),
                        mg.LineBasicMaterial(linewidth=20, vertexColors=True),
                    )
                )

    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.75):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def _update_last_error(self, q, left_wrist, right_wrist):
        trans = np.array(self.translational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        rot = np.array(self.rotational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        self.last_error = {
            "left_pos": float(np.linalg.norm(trans[:3])),
            "right_pos": float(np.linalg.norm(trans[3:])),
            "left_ori": float(np.linalg.norm(rot[:3])),
            "right_ori": float(np.linalg.norm(rot[3:])),
        }

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            sol = self.opti.solve()
            sol_q = sol.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data.copy()

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            self._update_last_error(sol_q, left_wrist, right_wrist)

            if self.Visualization:
                self.vis.display(sol_q)

            return sol_q, sol_tauff

        except Exception as exc:
            logger_mp.error("ERROR in convergence, plotting debug info.%s", exc)
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data.copy()

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            self._update_last_error(sol_q, left_wrist, right_wrist)
            logger_mp.error(
                "sol_q:%s \nmotorstate: \n%s \nleft_pose: \n%s \nright_pose: \n%s",
                sol_q,
                current_lr_arm_motor_q,
                left_wrist,
                right_wrist,
            )
            if self.Visualization:
                self.vis.display(sol_q)

            if current_lr_arm_motor_q is not None:
                return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
            return sol_q, sol_tauff


class X2_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False, urdf_path=None, model_dir=None, **kwargs):
        if kwargs:
            logger_mp.warning("[X2_ArmIK] Ignoring unsupported IK arguments: %s", sorted(kwargs.keys()))

        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        self.cache_path = os.path.join(parent2_dir, "x2_ultra_wrist_pin_model_cache.pkl")
        self.urdf_path = urdf_path or os.path.join(parent2_dir, "assets/X2/x2_ultra.urdf")
        self.model_dir = model_dir or os.path.join(parent2_dir, "assets/X2")
        # TeleVuer's wrist frame points +X toward the fingers. The X2 wrist
        # chain points -Z in that direction, unlike G1 whose wrist +X already
        # follows the hand. Rotate the operational frame without moving its
        # origin away from the physical wrist pivot.
        self.ee_placement = pin.SE3(
            pin.rpy.rpyToMatrix(0.0, np.pi / 2.0, 0.0),
            np.zeros(3),
        )
        self.last_error = {
            "left_pos": np.inf,
            "right_pos": np.inf,
            "left_ori": np.inf,
            "right_ori": np.inf,
        }

        cache_loaded = False
        if os.path.exists(self.cache_path) and not self.Visualization:
            try:
                logger_mp.info("[X2_ArmIK] >>> Loading cached robot model: %s", self.cache_path)
                self.robot, self.reduced_robot = self.load_cache()
                cache_loaded = self._cache_has_expected_ee_frames()
                if not cache_loaded:
                    logger_mp.warning("[X2_ArmIK] >>> Cached end-effector frames are stale; rebuilding from URDF.")
            except Exception as exc:
                logger_mp.warning("[X2_ArmIK] >>> Failed to load cached model; rebuilding from URDF. %s", exc)

        if not cache_loaded:
            logger_mp.info("[X2_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            # Lock all joints except the arms, waist, and head.
            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
                "head_yaw_joint",
                "head_pitch_joint",
            ]

            # Build a simplified model with only the arm joints free for faster IK, and add Right EE and Left EE frames.
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_roll_joint"),
                    self.ee_placement,
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_roll_joint"),
                    self.ee_placement,
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.data = self.reduced_robot.model.createData()
            if not self.Visualization:
                self.save_cache()
                logger_mp.info("[X2_ArmIK] >>> Cache saved to %s", self.cache_path)

        # X2's all-zero arm pose is arms-down. Use the same bent-arm pose like G1's nominal pose as the IK prior.
        self.nominal_q = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.57,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.57,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        if self.nominal_q.shape[0] != self.reduced_robot.model.nq:
            raise ValueError(
                f"X2 nominal arm pose has {self.nominal_q.shape[0]} joints, "
                f"but reduced model has {self.reduced_robot.model.nq}"
            )

        # Set up CasADi symbolic model using Pinocchio model and optimization problem for IK.
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) # joint positions
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)                     # left wrist target pose 4x4 
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)                     # right wrist target pose 4x4
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Define CasADi functions for calculating translational and rotational errors.
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )

        # Set up CasADi optimization problem for inverse kinematics.
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)  # joint positions
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # last joint positions for smoothness cost
        self.param_q_nominal = self.opti.parameter(self.reduced_robot.model.nq)  # nominal joint positions for regularization cost
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        # Define cost functions for the optimization problem. Squared sum for the errors
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q - self.param_q_nominal)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )
        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,  # tolerance 
            "ipopt.acceptable_tol": 5e-4,  # acceptable tolerance
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = self.nominal_q.copy()
        self._has_solution = False
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_robot.model.nq)
        self.vis = None

        if self.Visualization:
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length=0.15, axis_width=5)
            self.vis.display(self.nominal_q)

    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def _cache_has_expected_ee_frames(self):
        expected_joints = {
            "L_ee": "left_wrist_roll_joint",
            "R_ee": "right_wrist_roll_joint",
        }
        model = self.reduced_robot.model
        for frame_name, joint_name in expected_joints.items():
            if not model.existFrame(frame_name):
                return False
            frame = model.frames[model.getFrameId(frame_name)]
            if frame.parentJoint != model.getJointId(joint_name):
                return False
            if not np.allclose(frame.placement.rotation, self.ee_placement.rotation, atol=1e-9):
                return False
            if not np.allclose(frame.placement.translation, self.ee_placement.translation, atol=1e-9):
                return False
        return True

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.62):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def _update_last_error(self, q, left_wrist, right_wrist):
        trans = np.array(self.translational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        rot = np.array(self.rotational_error(q, left_wrist, right_wrist), dtype=np.float64).reshape(-1)
        self.last_error = {
            "left_pos": float(np.linalg.norm(trans[:3])),
            "right_pos": float(np.linalg.norm(trans[3:])),
            "left_ori": float(np.linalg.norm(rot[:3])),
            "right_ori": float(np.linalg.norm(rot[3:])),
        }

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            current_q = np.asarray(current_lr_arm_motor_q, dtype=np.float64)
            if current_q.shape != self.init_data.shape:
                raise ValueError(f"Expected X2 arm q shape {self.init_data.shape}, got {current_q.shape}")
            if self._has_solution or not np.allclose(current_q, 0.0, atol=1e-6):
                self.init_data = current_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)
        self.opti.set_value(self.param_q_nominal, self.nominal_q)

        try:
            sol = self.opti.solve()
            sol_q = sol.value(self.var_q)
        except Exception as exc:
            logger_mp.error("ERROR in X2 IK convergence, using debug solution. %s", exc)
            sol_q = self.opti.debug.value(self.var_q)
            if current_lr_arm_motor_q is not None:
                sol_q = current_lr_arm_motor_q

        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data.copy()
        self.init_data = sol_q
        self._has_solution = True
        self._update_last_error(sol_q, left_wrist, right_wrist)

        if self.Visualization:
            self.vis.display(sol_q)

        return sol_q, np.zeros(self.reduced_robot.model.nv)


class H1_2_ArmIK:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("H1_2_ArmIK is not migrated in dex_teleop.")


class H1_ArmIK:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("H1_ArmIK is not migrated in dex_teleop.")
