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
        """
        self.source_xml_paths = [path.strip() for path in xml_path.split(",") if path.strip()]
        self.dual_hand_mode = len(self.source_xml_paths) > 1
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
        self.steps_to_target = 0
        self.joint_mapping = {}  # Maps joint name to qpos indices
        
        self.last_qpos = None
        self.joint_names = None

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

    def update_target_qpos(self, target_qpos: np.ndarray, joint_names: list):
        """
        Update target joint positions with low-pass filtering.
        
        Args:
            target_qpos: Target joint positions
            joint_names: Names of joints in the received message
        """
        # Map received joint names to model joint indices
        joint_indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                joint_indices.append(joint_id)
            else:
                logger.warning(f"Joint '{name}' not found in model")
        
        # Apply low-pass filter to incoming commands
        qpos_idx = 0
        for i, joint_id in enumerate(joint_indices):
            if qpos_idx < len(target_qpos):
                # Get the address of this joint in qpos
                qpos_addr = self.model.jnt_qposadr[joint_id]
                
                # Determine the number of coordinates for this joint
                if joint_id < self.model.njnt - 1:
                    next_qpos_addr = self.model.jnt_qposadr[joint_id + 1]
                    qpos_size = next_qpos_addr - qpos_addr
                else:
                    qpos_size = self.model.nq - qpos_addr
                
                # Apply low-pass filter
                for j in range(qpos_size):
                    idx = qpos_addr + j
                    incoming_val = target_qpos[qpos_idx + j]
                    self.smoothed_qpos[idx] = (
                        self.smoothing_alpha * incoming_val
                        + (1 - self.smoothing_alpha) * self.smoothed_qpos[idx]
                    )
                
                qpos_idx += qpos_size
        
        # Copy smoothed values to target and start interpolation
        self.target_qpos[:] = self.smoothed_qpos
        self.interpol_qpos[:] = self.data.qpos
        self.steps_to_target = self.interpol_steps

    def step_interpolation(self):
        """
        Perform one step of interpolation towards target position.
        This smooths out the motion to reduce jerking.
        """
        if self.steps_to_target > 0:
            # Linear interpolation
            alpha = 1.0 - (self.steps_to_target / float(self.interpol_steps))
            self.data.qpos[:] = (
                (1 - alpha) * self.interpol_qpos + alpha * self.target_qpos
            )
            self.steps_to_target -= 1
        else:
            # Reached target, maintain position
            self.data.qpos[:] = self.target_qpos
        
        # Zero velocities for kinematic control
        self.data.qvel[:] = 0

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
                
                # Perform interpolation step
                self.step_interpolation()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
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
    )
    try:
        receiver.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        receiver.close()


if __name__ == "__main__":
    tyro.cli(main)
