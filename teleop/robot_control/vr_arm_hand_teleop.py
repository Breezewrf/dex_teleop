import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing import Array
from typing import Iterable, Optional

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TELEOP_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(TELEOP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TELEOP_DIR not in sys.path:
    sys.path.insert(0, TELEOP_DIR)

LOGGER = logging.getLogger(__name__)

DEFAULT_G1_MODEL = os.path.join(PROJECT_ROOT, "assets/g1/g1_body29_hand14.xml")
DEFAULT_G1_23_MODEL = os.path.join(PROJECT_ROOT, "assets/g1/g1_23dof_rev_1_0.xml")
DEFAULT_G1_CASIA_MODEL = os.path.join(PROJECT_ROOT, "assets/g1/g1_body29_casia_hand.xml")
DEFAULT_G1_23_CASIA_MODEL = os.path.join(PROJECT_ROOT, "assets/g1/g1_23dof_rev_1_0_casia_hand.xml")

G1_29_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_23_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

ROBOT_ARM_JOINT_NAMES = {
    "g1_29": G1_29_ARM_JOINT_NAMES,
    "g1_23": G1_23_ARM_JOINT_NAMES,
}

CASIA_LEFT_JOINT_NAMES = (
    "left_index_proximal",
    "left_index_intermediate",
    "left_index_distal",
    "left_pinky_proximal",
    "left_pinky_intermediate",
    "left_pinky_distal",
    "left_middle_proximal",
    "left_middle_intermediate",
    "left_middle_distal",
    "left_ring_proximal",
    "left_ring_intermediate",
    "left_ring_distal",
    "left_thumb_proximal",
    "left_thumb_intermediate",
)

CASIA_RIGHT_JOINT_NAMES = (
    "right_index_proximal",
    "right_index_intermediate",
    "right_index_distal",
    "right_pinky_proximal",
    "right_pinky_intermediate",
    "right_pinky_distal",
    "right_middle_proximal",
    "right_middle_intermediate",
    "right_middle_distal",
    "right_ring_proximal",
    "right_ring_intermediate",
    "right_ring_distal",
    "right_thumb_proximal",
    "right_thumb_intermediate",
)


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray


class MujocoG1ArmController:
    def __init__(
        self,
        model_path: str,
        joint_names: Iterable[str] = G1_29_ARM_JOINT_NAMES,
        render: bool = True,
        control_mode: str = "kinematic",
        kp: float = 80.0,
        kd: float = 3.0,
        enable_casia_hand: bool = False,
    ):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        self.control_mode = control_mode
        self.kp = kp
        self.kd = kd
        self.viewer = None
        self.enable_casia_hand = enable_casia_hand

        self.joint_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self.qpos_ids = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.dof_ids = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.ctrl_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self.ctrl_ranges = self.model.actuator_ctrlrange[self.ctrl_ids].copy()
        self.has_limited_ctrl = self.model.actuator_ctrllimited[self.ctrl_ids].astype(bool)
        self.left_casia_qpos_ids = None
        self.left_casia_dof_ids = None
        self.left_casia_ranges = None
        self.right_casia_qpos_ids = None
        self.right_casia_dof_ids = None
        self.right_casia_ranges = None
        if self.enable_casia_hand:
            self._setup_casia_hand_joints()

        if self.control_mode not in ("kinematic", "pd"):
            raise ValueError(f"Unsupported MuJoCo control mode: {self.control_mode}")

        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.793])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)

        if render:
            try:
                from mujoco import viewer

                self.viewer = viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                LOGGER.warning("MuJoCo viewer was not opened: %s", exc)

    def _name_to_id(self, obj_type, name: str) -> int:
        obj_id = self.mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return obj_id

    def _setup_casia_hand_joints(self) -> None:
        left_joint_ids = self._required_joint_ids(CASIA_LEFT_JOINT_NAMES)
        right_joint_ids = self._required_joint_ids(CASIA_RIGHT_JOINT_NAMES)
        self.left_casia_qpos_ids = self.model.jnt_qposadr[left_joint_ids]
        self.left_casia_dof_ids = self.model.jnt_dofadr[left_joint_ids]
        self.left_casia_ranges = self.model.jnt_range[left_joint_ids].copy()
        self.right_casia_qpos_ids = self.model.jnt_qposadr[right_joint_ids]
        self.right_casia_dof_ids = self.model.jnt_dofadr[right_joint_ids]
        self.right_casia_ranges = self.model.jnt_range[right_joint_ids].copy()

    def _required_joint_ids(self, names: Iterable[str]) -> np.ndarray:
        joint_ids = np.array(
            [self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            missing = [name for name, joint_id in zip(names, joint_ids) if joint_id < 0]
            raise ValueError(f"MuJoCo model does not contain Casia hand joints: {missing}")
        return joint_ids

    def get_state(self) -> ArmState:
        return ArmState(
            q=self.data.qpos[self.qpos_ids].copy(),
            dq=self.data.qvel[self.dof_ids].copy(),
        )

    def send(self, q_target: np.ndarray, tauff_target: Optional[np.ndarray] = None) -> None:
        q_target = np.asarray(q_target, dtype=np.float64)
        if q_target.shape != (len(self.joint_names),):
            raise ValueError(f"Expected {len(self.joint_names)} arm joints, got shape {q_target.shape}")

        if self.control_mode == "kinematic":
            self.data.qpos[self.qpos_ids] = q_target
            self.data.qvel[self.dof_ids] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
        else:
            self._step_pd(q_target, tauff_target)

        if self.viewer is not None:
            self.viewer.sync()

    def set_casia_hand_q(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        if self.left_casia_qpos_ids is None or self.right_casia_qpos_ids is None:
            return

        left_q = np.asarray(left_q, dtype=np.float64)
        right_q = np.asarray(right_q, dtype=np.float64)
        if left_q.shape != (len(CASIA_LEFT_JOINT_NAMES),):
            raise ValueError(f"Expected {len(CASIA_LEFT_JOINT_NAMES)} left Casia joints, got {left_q.shape}")
        if right_q.shape != (len(CASIA_RIGHT_JOINT_NAMES),):
            raise ValueError(f"Expected {len(CASIA_RIGHT_JOINT_NAMES)} right Casia joints, got {right_q.shape}")

        self.data.qpos[self.left_casia_qpos_ids] = np.clip(
            left_q,
            self.left_casia_ranges[:, 0],
            self.left_casia_ranges[:, 1],
        )
        self.data.qpos[self.right_casia_qpos_ids] = np.clip(
            right_q,
            self.right_casia_ranges[:, 0],
            self.right_casia_ranges[:, 1],
        )
        self.data.qvel[self.left_casia_dof_ids] = 0.0
        self.data.qvel[self.right_casia_dof_ids] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

    def _step_pd(self, q_target: np.ndarray, tauff_target: Optional[np.ndarray]) -> None:
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        ctrl = self.kp * (q_target - q) - self.kd * dq
        if tauff_target is not None:
            ctrl = ctrl + np.asarray(tauff_target, dtype=np.float64)[: len(self.ctrl_ids)]
        if np.any(self.has_limited_ctrl):
            low = self.ctrl_ranges[:, 0]
            high = self.ctrl_ranges[:, 1]
            ctrl = np.where(self.has_limited_ctrl, np.clip(ctrl, low, high), ctrl)
        self.data.ctrl[self.ctrl_ids] = ctrl

        non_arm_dofs = np.ones(self.model.nv, dtype=bool)
        non_arm_dofs[self.dof_ids] = False
        self.data.qvel[non_arm_dofs] = 0.0
        self.mujoco.mj_step(self.model, self.data)

    def go_home(self, steps: int = 120) -> None:
        start = self.get_state().q
        for alpha in np.linspace(0.0, 1.0, steps):
            self.send((1.0 - alpha) * start)
            time.sleep(0.005)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


class RealG1ArmController:
    def __init__(self, robot: str, network_interface: Optional[str], motion_mode: bool):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The dex_teleop venv does not have unitree_sdk2py installed. "
                "Install it before using --backend real."
            ) from exc

        from teleop.robot_control.robot_arm import G1_23_ArmController, G1_29_ArmController
        from teleop.utils.motion_switcher import MotionSwitcher

        ChannelFactoryInitialize(0, networkInterface=network_interface)
        self.motion_switcher = None
        if not motion_mode:
            self.motion_switcher = MotionSwitcher()
            status, result = self.motion_switcher.Enter_Debug_Mode()
            LOGGER.info("Enter debug mode: %s, %s", status, result)

        controller_cls = {
            "g1_29": G1_29_ArmController,
            "g1_23": G1_23_ArmController,
        }[robot]
        LOGGER.info("Using real arm controller: %s", controller_cls.__name__)
        self.arm = controller_cls(motion_mode=motion_mode, simulation_mode=False)
        self.arm.speed_gradual_max()

    def get_state(self) -> ArmState:
        return ArmState(q=self.arm.get_current_dual_arm_q(), dq=self.arm.get_current_dual_arm_dq())

    def send(self, q_target: np.ndarray, tauff_target: Optional[np.ndarray] = None) -> None:
        if tauff_target is None:
            tauff_target = np.zeros_like(q_target)
        self.arm.ctrl_dual_arm(q_target, tauff_target)

    def go_home(self) -> None:
        self.arm.ctrl_dual_arm_go_home()

    def close(self) -> None:
        pass


class CasiaHandBridge:
    def __init__(
        self,
        frequency: float,
        enable_zmq: bool,
        zmq_left_port: int,
        zmq_right_port: int,
        zmq_left_real_port: int,
        zmq_right_real_port: int,
    ):
        from teleop.robot_control.robot_hand_casia_v2 import Casia_Controller

        self.left_hand_pos_array = Array("d", 75, lock=True)
        self.right_hand_pos_array = Array("d", 75, lock=True)
        self.controller = Casia_Controller(
            self.left_hand_pos_array,
            self.right_hand_pos_array,
            fps=frequency,
            enable_zmq=enable_zmq,
            zmq_left_port=zmq_left_port,
            zmq_right_port=zmq_right_port,
            zmq_left_real_port=zmq_left_real_port,
            zmq_right_real_port=zmq_right_real_port,
        )

    def update(self, tele_data) -> None:
        with self.left_hand_pos_array.get_lock():
            self.left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
        with self.right_hand_pos_array.get_lock():
            self.right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()

    def close(self) -> None:
        pass


class CasiaMujocoRetargeter:
    def __init__(self):
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

        self.hand_retargeting = HandRetargeting(HandType.CASIA_HAND)

    def retarget(self, tele_data) -> Optional[tuple[np.ndarray, np.ndarray]]:
        left_hand_data = np.asarray(tele_data.left_hand_pos, dtype=np.float64).reshape(25, 3)
        right_hand_data = np.asarray(tele_data.right_hand_pos, dtype=np.float64).reshape(25, 3)
        if np.all(left_hand_data == 0.0) or np.all(right_hand_data == 0.0):
            return None

        ref_left_value = (
            left_hand_data[self.hand_retargeting.left_indices[1, :]]
            - left_hand_data[self.hand_retargeting.left_indices[0, :]]
        )
        ref_right_value = (
            right_hand_data[self.hand_retargeting.right_indices[1, :]]
            - right_hand_data[self.hand_retargeting.right_indices[0, :]]
        )
        left_q = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
            self.hand_retargeting.left_dex_retargeting_to_hardware
        ]
        right_q = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
            self.hand_retargeting.right_dex_retargeting_to_hardware
        ]
        return left_q, right_q


def default_mujoco_model(robot: str, hand: str) -> str:
    if hand == "casia":
        if robot == "g1_23" and os.path.exists(DEFAULT_G1_23_CASIA_MODEL):
            return DEFAULT_G1_23_CASIA_MODEL
        return DEFAULT_G1_CASIA_MODEL
    if robot == "g1_23" and os.path.exists(DEFAULT_G1_23_MODEL):
        return DEFAULT_G1_23_MODEL
    else:
        LOGGER.warning("Using default G1 model for robot=%s, hand=%s. Consider specifying --model.", robot, hand)
    return DEFAULT_G1_MODEL


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal VR wrist/hand teleop for G1 arms and optional Casia hand retargeting."
    )
    parser.add_argument("--robot", choices=("g1_29", "g1_23"), default="g1_29")
    parser.add_argument("--backend", choices=("mujoco", "real"), default="mujoco")
    parser.add_argument("--hand", choices=("none", "casia"), default="none")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument(
        "--model",
        default=None,
        help="MuJoCo XML path. Defaults to a G1 model matching --robot/--hand.",
    )
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--motion", action="store_true", help="Use rt/arm_sdk motion topic on real G1.")
    parser.add_argument("--no-render", action="store_true", help="Disable MuJoCo passive viewer.")
    parser.add_argument(
        "--mujoco-control",
        choices=("kinematic", "pd"),
        default="kinematic",
        help="MuJoCo arm control. kinematic mirrors IK qpos; pd uses torque motors.",
    )
    parser.add_argument("--mujoco-kp", type=float, default=80.0)
    parser.add_argument("--mujoco-kd", type=float, default=3.0)
    parser.add_argument("--start-immediately", action="store_true")
    parser.add_argument("--log-poses", action="store_true")
    parser.add_argument("--casia-enable-zmq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--casia-zmq-left-port", type=int, default=5560)
    parser.add_argument("--casia-zmq-right-port", type=int, default=5561)
    parser.add_argument("--casia-zmq-left-real-port", type=int, default=5555)
    parser.add_argument("--casia-zmq-right-real-port", type=int, default=5556)
    return parser


def run(args: argparse.Namespace) -> None:
    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK

    model_path = args.model
    if args.backend == "mujoco" and model_path is None:
        model_path = default_mujoco_model(args.robot, args.hand)

    arm_ik_cls = {
        "g1_29": G1_29_ArmIK,
        "g1_23": G1_23_ArmIK,
    }[args.robot]
    arm_joint_names = ROBOT_ARM_JOINT_NAMES[args.robot]

    tv_wrapper = None
    arm_ctrl = None
    hand_bridge = None
    mujoco_hand_retargeter = None
    try:
        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=True,
            binocular=True,
            img_shape=(480, 1280),
            display_fps=args.frequency,
            display_mode="pass-through",
        )

        arm_ik = arm_ik_cls()
        if args.backend == "mujoco":
            arm_ctrl = MujocoG1ArmController(
                model_path,
                joint_names=arm_joint_names,
                render=not args.no_render,
                control_mode=args.mujoco_control,
                kp=args.mujoco_kp,
                kd=args.mujoco_kd,
                enable_casia_hand=args.hand == "casia",
            )
            if args.hand == "casia":
                mujoco_hand_retargeter = CasiaMujocoRetargeter()
        else:
            arm_ctrl = RealG1ArmController(args.robot, args.network_interface, args.motion)

        if args.hand == "casia":
            hand_bridge = CasiaHandBridge(
                args.frequency,
                args.casia_enable_zmq,
                args.casia_zmq_left_port,
                args.casia_zmq_right_port,
                args.casia_zmq_left_real_port,
                args.casia_zmq_right_real_port,
            )
    except Exception:
        if hand_bridge is not None:
            hand_bridge.close()
        if arm_ctrl is not None:
            arm_ctrl.close()
        if tv_wrapper is not None:
            tv_wrapper.close()
        raise

    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if not args.start_immediately:
            input("Press Enter to start VR arm/hand tracking. Press Ctrl+C to stop.\n")

        dt = 1.0 / args.frequency
        LOGGER.info("Started VR tracking with robot=%s, backend=%s, hand=%s", args.robot, args.backend, args.hand)
        while not stop:
            start = time.time()
            tele_data = tv_wrapper.get_tele_data()

            if hand_bridge is not None:
                hand_bridge.update(tele_data)

            state = arm_ctrl.get_state()
            sol_q, sol_tauff = arm_ik.solve_ik(
                tele_data.left_wrist_pose,
                tele_data.right_wrist_pose,
                state.q,
                state.dq,
            )
            arm_ctrl.send(sol_q, sol_tauff)
            if mujoco_hand_retargeter is not None:
                hand_q = mujoco_hand_retargeter.retarget(tele_data)
                if hand_q is not None:
                    arm_ctrl.set_casia_hand_q(*hand_q)

            if args.log_poses:
                LOGGER.info(
                    "left=%s right=%s q=%s ik_error=%s",
                    np.round(tele_data.left_wrist_pose[:3, 3], 3),
                    np.round(tele_data.right_wrist_pose[:3, 3], 3),
                    np.round(sol_q, 3),
                    getattr(arm_ik, "last_error", None),
                )

            time.sleep(max(0.0, dt - (time.time() - start)))
    finally:
        try:
            arm_ctrl.go_home()
        except Exception as exc:
            LOGGER.warning("Failed to send arms home: %s", exc)
        if hand_bridge is not None:
            hand_bridge.close()
        if arm_ctrl is not None:
            arm_ctrl.close()
        if tv_wrapper is not None:
            tv_wrapper.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
