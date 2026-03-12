"""
MuJoCo Receiver for sim2sim hand retargeting.
Receives retargeted joint positions via ZMQ and controls a MuJoCo robot.
"""
import json
import time
from pathlib import Path

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
        zmq_port: int = 5550,
        speed: float = 1.0,
        smoothing_alpha: float = 0.2,
        interpol_steps: int = 5,
    ):
        """
        Initialize MuJoCo robot and ZMQ subscriber.
        
        Args:
            xml_path: Path to MuJoCo XML model file
            zmq_port: ZMQ socket port number (default 5550 for sim2sim)
            speed: Simulation speed multiplier (default 1.0)
            smoothing_alpha: Low-pass filter alpha (0-1, lower = more smoothing)
            interpol_steps: Number of steps to interpolate between commands
        """
        # Load MuJoCo model
        logger.info(f"Loading MuJoCo model from {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.speed = speed
        
        logger.info(f"Model loaded with {self.model.nq} DOFs")
        logger.info(f"Model has {self.model.nbody} bodies")
        
        # Print joint names for debugging
        logger.info("Joint names in model:")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            logger.info(f"  {i}: {joint_name}")
        
        # Setup ZMQ subscriber
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{zmq_port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        logger.info(f"ZMQ subscriber connected to localhost:{zmq_port}")
        
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
            
            last_receive_time = time.time()
            
            while viewer.is_running():
                try:
                    # Try to receive ZMQ message
                    message = self.socket.recv(zmq.NOBLOCK)
                    data = json.loads(message.decode("utf-8"))
                    
                    qpos = np.array(data["qpos"])
                    joint_names = data["joint_names"]
                    timestamp = data.get("timestamp", time.time())
                    
                    self.update_target_qpos(qpos, joint_names)
                    last_receive_time = time.time()
                    logger.info(f"Updated target state: {qpos}")
                    
                except zmq.Again:
                    # No message available, continue
                    pass
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode ZMQ message: {e}")
                except Exception as e:
                    logger.error(f"Error updating robot state: {e}")
                
                # Check if we haven't received data for too long
                if time.time() - last_receive_time > 5.0:
                    logger.warning(f"No ZMQ messages received for 5 seconds for port {self.zmq_port}. Check if sender is running.")
                
                # Perform interpolation step
                self.step_interpolation()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def close(self):
        """Close ZMQ socket."""
        self.socket.close()
        self.context.term()


def main(
    xml_path: str,
    zmq_port: int = 5560,
    speed: float = 1.0,
    smoothing_alpha: float = 0.2,
    interpol_steps: int = 8,
):
    """
    Run MuJoCo simulation with ZMQ control (sim2sim).
    
    Args:
        xml_path: Path to MuJoCo XML model file.
        zmq_port: ZMQ socket port number (default 5550 for sim2sim).
        speed: Simulation speed multiplier (default 1.0).
        smoothing_alpha: Low-pass filter alpha for command smoothing (0-1).
            Lower values provide more smoothing (default 0.2).
        interpol_steps: Number of simulation steps to interpolate between
            command updates (default 8). Higher values produce smoother motion.
    """
    # Check if file exists
    if not Path(xml_path).exists():
        logger.error(f"XML file not found: {xml_path}")
        return
    
    receiver = MuJoCoReceiver(
        xml_path,
        zmq_port=zmq_port,
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
