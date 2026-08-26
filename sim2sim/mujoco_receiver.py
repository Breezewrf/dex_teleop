"""
MuJoCo Receiver for sim2sim hand retargeting.
Receives retargeted joint positions via ZMQ and controls a MuJoCo robot.
"""
import json
import time
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import mujoco
import mujoco.viewer
import zmq
import tyro
from loguru import logger


class MuJoCoReceiver:
    """Receives joint states via ZMQ and controls a MuJoCo robot."""

    def __init__(
        self,
        xml_path: str,
        zmq_url: str = "localhost",
        zmq_port_left: int = 5560,
        zmq_port_right: int = 5561,
        speed: float = 1.0,
        smoothing_alpha: float = 0.2,
        interpol_steps: int = 5,
        subscribe_both: bool = False,
        control_mode: str = "qpos",
    ):
        """
        Initialize MuJoCo robot and ZMQ subscriber.
        
        Args:
            xml_path: Path to one XML file, or two comma-separated XML files
            zmq_url: ZMQ host or full tcp:// endpoint (default localhost)
            zmq_port_left: ZMQ socket port number for the left hand (default 5560 for sim2sim)
            zmq_port_right: ZMQ socket port number for the right hand (default 5561 for sim2sim)
            speed: Simulation speed multiplier (default 1.0)
            smoothing_alpha: Low-pass filter alpha (0-1, lower = more smoothing)
            interpol_steps: Number of steps to interpolate between commands
            subscribe_both: Subscribe to both hand ports for a single dual-hand XML
            control_mode: "qpos", "position-actuator", or "kinematic-coupled"
        """
        self.source_xml_paths = [path.strip() for path in xml_path.split(",") if path.strip()]
        if not self.source_xml_paths:
            raise ValueError("At least one MuJoCo XML path is required")
        if control_mode not in {"qpos", "position-actuator", "kinematic-coupled"}:
            raise ValueError(f"Unsupported control mode: {control_mode}")
        if not 0.0 <= smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1]")
        if interpol_steps < 1:
            raise ValueError("interpol_steps must be at least 1")

        self.dual_hand_mode = len(self.source_xml_paths) > 1 or subscribe_both
        self.control_mode = control_mode
        self._generated_model_path: str | None = None

        model_path = self._build_model_path(self.source_xml_paths)

        # Load MuJoCo model
        logger.info(f"Loading MuJoCo model from {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.speed = speed
        
        logger.info(f"Model loaded with {self.model.nq} DOFs")
        logger.info(f"Model has {self.model.nbody} bodies")
        
        # Print joint names for debugging
        logger.info("Joint names in model:")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            logger.info(f"  {i}: {joint_name}")

        if self.dual_hand_mode:
            self.zmq_endpoints = [
                self._build_zmq_endpoint(zmq_url, zmq_port_left, use_exact_url=False),
                self._build_zmq_endpoint(zmq_url, zmq_port_right, use_exact_url=False),
            ]
        else:
            if zmq_port_right is not None:
                logger.info(
                    "Using zmq_port_right as the single endpoint."
                )
                self.zmq_endpoints = [self._build_zmq_endpoint(zmq_url, zmq_port_right, use_exact_url=True)]
            elif zmq_port_left is not None:
                logger.info(
                    "Using zmq_port_left as the single endpoint."
                )
                self.zmq_endpoints = [self._build_zmq_endpoint(zmq_url, zmq_port_left, use_exact_url=True)]

        
        # Setup ZMQ subscriber
        self.context = zmq.Context()
        self.sockets = []
        for endpoint in self.zmq_endpoints:
            socket = self.context.socket(zmq.SUB)
            socket.connect(endpoint)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            self.sockets.append(socket)
            logger.info(f"ZMQ subscriber connected to {endpoint}")
        
        # Smoothing parameters
        self.smoothing_alpha = smoothing_alpha
        self.interpol_steps = interpol_steps
        
        # Target and current states
        self.target_qpos = np.zeros(self.model.nq)
        self.smoothed_qpos = np.zeros(self.model.nq)
        self.interpol_qpos = np.zeros(self.model.nq)
        self.target_ctrl = self.data.ctrl.copy()
        self.smoothed_ctrl = self.data.ctrl.copy()
        self.steps_to_target = 0
        self.joint_mapping = {}
        self.actuator_mapping = {}
        
        self.last_qpos = None
        self.joint_names = None

        self.coupled_joint_equalities = self._joint_equality_mappings()
        if self.control_mode == "kinematic-coupled":
            logger.info(
                "Kinematic coupled mode found {} joint equality mappings",
                len(self.coupled_joint_equalities),
            )
            if not self.coupled_joint_equalities:
                logger.warning(
                    "kinematic-coupled mode is active, but the model has no "
                    "joint-to-joint equality constraints"
                )

    def _build_zmq_endpoint(self, zmq_url: str, zmq_port: int, use_exact_url: bool) -> str:
        if zmq_url.startswith("tcp://"):
            if use_exact_url:
                return zmq_url

            parsed = urlparse(zmq_url)
            host = parsed.hostname or "localhost"
            return f"tcp://{host}:{zmq_port}"

        return f"tcp://{zmq_url}:{zmq_port}"

    def _build_model_path(self, xml_paths: list[str]) -> str:
        if len(xml_paths) == 1:
            return xml_paths[0]

        common_children: list[ET.Element] = []

        first_xml = Path(xml_paths[0]).resolve()
        first_tree = ET.parse(first_xml)
        first_root = first_tree.getroot()
        common_children = [
            deepcopy(child)
            for child in list(first_root)
            if child.tag not in {"asset", "worldbody"}
        ]

        root = ET.Element("mujoco", {"model": "CASIAHAND-M-Dual"})
        for child in common_children:
            root.append(child)

        asset = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")

        for index, xml_path in enumerate(xml_paths):
            xml_file = Path(xml_path).resolve()
            tree = ET.parse(xml_file)
            source_root = tree.getroot()
            base_dir = xml_file.parent

            source_asset = source_root.find("asset")
            if source_asset is not None:
                for child in list(source_asset):
                    asset_child = deepcopy(child)
                    self._rewrite_mesh_paths(asset_child, base_dir)
                    asset.append(asset_child)

            source_worldbody = source_root.find("worldbody")
            if source_worldbody is None:
                raise ValueError(f"Missing <worldbody> in {xml_path}")

            hand_wrapper = ET.SubElement(
                worldbody,
                "body",
                {
                    "name": f"hand_{index}_root",
                    "pos": self._hand_offset(index),
                    "quat": self._hand_rotation_quat(index),
                },
            )
            for child in list(source_worldbody):
                hand_wrapper.append(deepcopy(child))

        temp_file = tempfile.NamedTemporaryFile("w", suffix="_dual_hand.xml", delete=False)
        temp_file_path = Path(temp_file.name)
        temp_file.close()
        ET.ElementTree(root).write(temp_file_path, encoding="unicode")
        self._generated_model_path = str(temp_file_path)
        logger.info(f"Generated combined MuJoCo XML at {self._generated_model_path}")
        return self._generated_model_path

    def _rewrite_mesh_paths(self, element: ET.Element | None, base_dir: Path) -> None:
        if element is None:
            return

        file_attr = element.attrib.get("file")
        if file_attr:
            element.attrib["file"] = str((base_dir / file_attr).resolve())

        for child in list(element):
            self._rewrite_mesh_paths(child, base_dir)

    def _hand_offset(self, index: int) -> str:
        if index == 0:
            return "-0.12 0 0"
        return "0.12 0 0"

    def _hand_rotation_quat(self, index: int) -> str:
        return "0 0 0 1"

    def _joint_equality_mappings(self) -> list[tuple[int, int, np.ndarray]]:
        """Compile scalar joint equality constraints into qpos mappings.

        MuJoCo stores a joint equality as
        ``q_joint1 = poly(q_joint2)`` with five coefficients in ``eq_data``.
        Reading the compiled model keeps this mode synchronized with the MJCF
        nonlinear mappings without duplicating OmniHand-specific constants.
        """
        mappings: list[tuple[int, int, np.ndarray]] = []
        for equality_id in range(self.model.neq):
            if self.model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            if not self.model.eq_active0[equality_id]:
                continue
            joint1_id = int(self.model.eq_obj1id[equality_id])
            joint2_id = int(self.model.eq_obj2id[equality_id])
            if joint1_id < 0 or joint2_id < 0:
                continue
            scalar_types = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            if (
                self.model.jnt_type[joint1_id] not in scalar_types
                or self.model.jnt_type[joint2_id] not in scalar_types
            ):
                raise ValueError(
                    "kinematic-coupled only supports scalar hinge/slide joint equalities"
                )
            mappings.append(
                (
                    int(self.model.jnt_qposadr[joint1_id]),
                    int(self.model.jnt_qposadr[joint2_id]),
                    self.model.eq_data[equality_id, :5].copy(),
                )
            )
        return mappings

    def apply_coupled_joint_equalities(self) -> None:
        """Set passive qpos from the MJCF nonlinear joint equalities."""
        for passive_qpos_addr, source_qpos_addr, coefficients in self.coupled_joint_equalities:
            source_qpos = self.data.qpos[source_qpos_addr]
            self.data.qpos[passive_qpos_addr] = np.polynomial.polynomial.polyval(
                source_qpos,
                coefficients,
            )

    def update_target_qpos(self, target_qpos: np.ndarray, joint_names: list):
        """
        Update target joint positions with low-pass filtering.
        
        Args:
            target_qpos: Target joint positions
            joint_names: Names of joints in the received message
        """
        target_qpos = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
        if len(target_qpos) != len(joint_names):
            raise ValueError(
                f"Received {len(target_qpos)} positions for {len(joint_names)} joint names"
            )
        if not np.all(np.isfinite(target_qpos)):
            raise ValueError("Received non-finite joint target")

        if self.control_mode == "position-actuator":
            self.update_target_ctrl(target_qpos, joint_names)
            return
        
        # Apply low-pass filter to incoming commands
        for incoming_val, name in zip(target_qpos, joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                logger.warning(f"Joint '{name}' not found in model")
                continue
            joint_type = self.model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"Kinematic receiver only accepts 1-DOF joints, got '{name}'")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.smoothed_qpos[qpos_addr] = (
                self.smoothing_alpha * incoming_val
                + (1 - self.smoothing_alpha) * self.smoothed_qpos[qpos_addr]
            )
        
        # Copy smoothed values to target and start interpolation
        self.target_qpos[:] = self.smoothed_qpos
        self.interpol_qpos[:] = self.data.qpos
        self.steps_to_target = self.interpol_steps

    def _joint_to_actuator_id(self, joint_name: str) -> int:
        cached = self.actuator_mapping.get(joint_name)
        if cached is not None:
            return cached

        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model")

        for actuator_id in range(self.model.nu):
            if (
                self.model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
                and self.model.actuator_trnid[actuator_id, 0] == joint_id
            ):
                self.actuator_mapping[joint_name] = actuator_id
                return actuator_id
        raise ValueError(f"No joint actuator found for '{joint_name}'")

    def update_target_ctrl(self, target_qpos: np.ndarray, joint_names: list[str]) -> None:
        """Update position-actuator targets without writing passive joint qpos."""
        for incoming_val, joint_name in zip(target_qpos, joint_names):
            actuator_id = self._joint_to_actuator_id(joint_name)
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                incoming_val = np.clip(incoming_val, low, high)
            self.target_ctrl[actuator_id] = incoming_val
            self.smoothed_ctrl[actuator_id] = (
                self.smoothing_alpha * incoming_val
                + (1 - self.smoothing_alpha) * self.smoothed_ctrl[actuator_id]
            )

    def step_interpolation(self):
        """
        Perform one step of interpolation towards target position.
        This smooths out the motion to reduce jerking.
        """
        if self.steps_to_target > 0:
            # Linear interpolation
            alpha = 1.0 - ((self.steps_to_target - 1) / float(self.interpol_steps))
            self.data.qpos[:] = (
                (1 - alpha) * self.interpol_qpos + alpha * self.target_qpos
            )
            self.steps_to_target -= 1
        else:
            # Reached target, maintain position
            self.data.qpos[:] = self.target_qpos
        
        # Zero velocities for kinematic control
        self.data.qvel[:] = 0

    def step_simulation(self) -> None:
        if self.control_mode == "position-actuator":
            self.data.ctrl[:] = self.smoothed_ctrl
            mujoco.mj_step(self.model, self.data)
        elif self.control_mode == "kinematic-coupled":
            self.step_interpolation()
            self.apply_coupled_joint_equalities()
            # This mode is a kinematic player: update derived poses, contacts,
            # and sensors without integrating actuator dynamics or advancing
            # simulation time.
            mujoco.mj_forward(self.model, self.data)
        else:
            self.step_interpolation()
            mujoco.mj_step(self.model, self.data)

    def run(self):
        """Run the simulation loop."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            logger.info("MuJoCo viewer launched. Press Ctrl+C to exit.")
            logger.info(
                f"Smoothing parameters: alpha={self.smoothing_alpha}, "
                f"interpol_steps={self.interpol_steps}"
            )
            
            last_receive_times = {endpoint: time.time() for endpoint in self.zmq_endpoints}
            
            while viewer.is_running():
                for endpoint, socket in zip(self.zmq_endpoints, self.sockets):
                    try:
                        # Try to receive ZMQ message
                        message = socket.recv(zmq.NOBLOCK)
                        data = json.loads(message.decode("utf-8"))
                        
                        qpos = np.array(data["qpos"])
                        joint_names = data["joint_names"]
                        timestamp = data.get("timestamp", time.time())
                        
                        self.update_target_qpos(qpos, joint_names)
                        last_receive_times[endpoint] = time.time()
                        logger.info(f"Updated target state from {endpoint}: {qpos}")
                        
                    except zmq.Again:
                        # No message available, continue
                        pass
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode ZMQ message from {endpoint}: {e}")
                    except Exception as e:
                        logger.error(f"Error updating robot state from {endpoint}: {e}")
                
                # Check if we haven't received data for too long
                for endpoint, last_receive_time in last_receive_times.items():
                    if time.time() - last_receive_time > 5.0:
                        logger.warning(
                            f"No ZMQ messages received for 5 seconds from {endpoint}. "
                            "Check if sender is running."
                        )
                
                self.step_simulation()
                viewer.sync()

    def close(self):
        """Close ZMQ socket."""
        for socket in getattr(self, "sockets", []):
            socket.close()
        self.context.term()
        if self._generated_model_path is not None:
            try:
                Path(self._generated_model_path).unlink(missing_ok=True)
            except Exception:
                pass


def main(
    xml_path: str,
    zmq_url: str = "localhost",
    zmq_port_left: int = 5560,
    zmq_port_right: int = 5561,
    speed: float = 1.0,
    smoothing_alpha: float = 0.2,
    interpol_steps: int = 8,
    subscribe_both: bool = False,
    control_mode: str = "qpos",
):
    """
    Run MuJoCo simulation with ZMQ control (sim2sim).
    
    Args:
        xml_path: Path to one XML file, or two comma-separated XML files.
        zmq_url: ZMQ host or full tcp:// endpoint (default localhost).
        zmq_port_left: ZMQ socket port number for the left hand (default 5560 for sim2sim).
        zmq_port_right: ZMQ socket port number for the right hand (default 5561 for sim2sim).
        speed: Simulation speed multiplier (default 1.0).
        smoothing_alpha: Low-pass filter alpha for command smoothing (0-1).
            Lower values provide more smoothing (default 0.2).
        interpol_steps: Number of simulation steps to interpolate between
            command updates (default 8). Higher values produce smoother motion.
        subscribe_both: Subscribe to left and right ports for one dual-hand XML.
        control_mode: "qpos", "position-actuator", or "kinematic-coupled".
    """
    # Check if file exists
    xml_paths = [path.strip() for path in xml_path.split(",") if path.strip()]
    missing_paths = [path for path in xml_paths if not Path(path).exists()]
    if missing_paths:
        logger.error(f"XML file not found: {', '.join(missing_paths)}")
        return
    
    receiver = MuJoCoReceiver(
        xml_path,
        zmq_url=zmq_url,
        zmq_port_left=zmq_port_left,
        zmq_port_right=zmq_port_right,
        speed=speed,
        smoothing_alpha=smoothing_alpha,
        interpol_steps=interpol_steps,
        subscribe_both=subscribe_both,
        control_mode=control_mode,
    )
    try:
        receiver.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        receiver.close()


if __name__ == "__main__":
    tyro.cli(main)
