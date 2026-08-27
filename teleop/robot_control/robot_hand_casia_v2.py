import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
import json
import zmq
from multiprocessing import Process, Array, Value, Lock

import logging_mp

try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError:
    # logging_mp may already be initialized by another module in the same process.
    pass

logger_mp = logging_mp.getLogger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

CASIA_Num_Motors = 14
CASIA_Num_Motors_Real = 10


class Casia_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, fps = 100.0, Unit_Test = False,
                 simulation_mode = False, enable_zmq = True, zmq_left_port = 5560,
                 zmq_right_port = 5561, zmq_left_real_port = 5555, zmq_right_real_port = 5556):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Casia_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.enable_zmq = enable_zmq
        self.zmq_left_port = zmq_left_port
        self.zmq_right_port = zmq_right_port
        self.zmq_left_real_port = zmq_left_real_port
        self.zmq_right_real_port = zmq_right_real_port

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.CASIA_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.CASIA_HAND_Unit_Test)

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,
                                                                          ))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Casia_Controller OK!")
    
    def control_process(self, left_hand_array_in, right_hand_array_in):
        self.running = True

        left_q_target  = np.full(CASIA_Num_Motors, 0)
        right_q_target = np.full(CASIA_Num_Motors, 0)
        left_q_target_real = np.full(CASIA_Num_Motors_Real, 0)
        right_q_target_real = np.full(CASIA_Num_Motors_Real, 0)
        left_target_hand = None
        right_target_hand = None

        # joint_names = ['index1', 'index2', 'index3', 'little1', 'little2', 'little3', 'middle1', 'middle2', 'middle3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2']
        # joint_names_real = ['thumb1', 'thumb2', 'index1', 'middle1', 'ring1', 'little1','index2', 'middle2', 'ring2', 'little2']
        left_joint_names = ['left_index_proximal', 'left_index_intermediate', 'left_index_distal',
                       'left_pinky_proximal', 'left_pinky_intermediate', 'left_pinky_distal',
                       'left_middle_proximal', 'left_middle_intermediate', 'left_middle_distal',
                       'left_ring_proximal', 'left_ring_intermediate', 'left_ring_distal',
                       'left_thumb_proximal', 'left_thumb_intermediate']
        left_joint_names_real = ['left_thumb_proximal', 'left_thumb_intermediate', 'left_index_proximal', 'left_middle_proximal', 'left_ring_proximal', 'left_pinky_proximal',
                            'left_index_intermediate', 'left_middle_intermediate', 'left_ring_intermediate', 'left_pinky_intermediate']

        right_joint_names = ['right_index_proximal', 'right_index_intermediate', 'right_index_distal',
                       'right_pinky_proximal', 'right_pinky_intermediate', 'right_pinky_distal',
                       'right_middle_proximal', 'right_middle_intermediate', 'right_middle_distal',
                       'right_ring_proximal', 'right_ring_intermediate', 'right_ring_distal',
                       'right_thumb_proximal', 'right_thumb_intermediate']
        right_joint_names_real = ['right_thumb_proximal', 'right_thumb_intermediate', 'right_index_proximal', 'right_middle_proximal', 'right_ring_proximal', 'right_pinky_proximal',
                            'right_index_intermediate', 'right_middle_intermediate', 'right_ring_intermediate', 'right_pinky_intermediate']
        
        left_robot_to_real_mapping = [left_joint_names.index(name) for name in left_joint_names_real]
        right_robot_to_real_mapping = [right_joint_names.index(name) for name in right_joint_names_real]

        context = None
        left_socket = None
        right_socket = None
        if self.enable_zmq:
            context = zmq.Context()

            left_socket = context.socket(zmq.PUB)
            left_socket.bind(f"tcp://*:{self.zmq_left_port}")
            logger_mp.info(f"ZMQ publisher (left hand) bound to port {self.zmq_left_port}")

            right_socket = context.socket(zmq.PUB)
            right_socket.bind(f"tcp://*:{self.zmq_right_port}")
            logger_mp.info(f"ZMQ publisher (right hand) bound to port {self.zmq_right_port}")

            left_real_socket = context.socket(zmq.PUB)
            left_real_socket.bind(f"tcp://*:{self.zmq_left_real_port}")
            logger_mp.info(f"ZMQ publisher (left hand real) bound to port {self.zmq_left_real_port}")

            right_real_socket = context.socket(zmq.PUB)
            right_real_socket.bind(f"tcp://*:{self.zmq_right_real_port}")
            logger_mp.info(f"ZMQ publisher (right hand real) bound to port {self.zmq_right_real_port}")

            time.sleep(1)

        logger_mp.info("Casia_Controller control process started.//////////////////////////////////////////")
        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array_in.get_lock():
                    left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                with right_hand_array_in.get_lock():
                    right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                left_target_hand = None
                right_target_hand = None
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    # Keep visualization in the exact landmark frame used by
                    # CASIA's optimizer. This same helper also supports other
                    # dex hands without hard-coded link or constraint lists.
                    left_target_hand = self.hand_retargeting.target_hand_visualization(
                        left_hand_data,
                        "left",
                    )
                    right_target_hand = self.hand_retargeting.target_hand_visualization(
                        right_hand_data,
                        "right",
                    )
                 
                    # Thumb_2 need abs() for sim2real
                    left_q_target_real = np.abs(left_q_target[left_robot_to_real_mapping])
                    right_q_target_real = np.abs(right_q_target[right_robot_to_real_mapping])

                    
                    # Potantial Normalization for sim2real
                    # In the official document, the angles are in the range [0, 1] ==> 0.0: fully open  1.0: fully closed
                    # The q_target now is in radians, ranges:
                    #     - idx 0:   0~1.52
                    #     - idx 1:   0~1.05
                    #     - idx 2~5: 0~1.47
                    # We normalize them using (max - value) / range
                    # def normalize(val, min_val, max_val):
                    #     return 1.0 - np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    # for idx in range(CASIA_Num_Motors):
                    #     if idx == 0:
                    #         left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.05)
                    #         right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.05)
                    #     elif idx == 1:
                    #         left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.05)
                    #         right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.05)
                    #     elif idx >= 2:
                    #         left_q_target[idx]  = normalize(left_q_target[idx], 0.0, 1.05)
                    #         right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.05)


                if self.enable_zmq and left_socket is not None and right_socket is not None:
                    timestamp = time.time()

                    data_left = {
                        "timestamp": timestamp,
                        "qpos": left_q_target.tolist(),
                        "joint_names": left_joint_names,
                        "type": "sim2sim",
                        "target_hand": left_target_hand,
                    }
                    left_socket.send(json.dumps(data_left).encode("utf-8"))

                    data_right = {
                        "timestamp": timestamp,
                        "qpos": right_q_target.tolist(),
                        "joint_names": right_joint_names,
                        "type": "sim2sim",
                        "target_hand": right_target_hand,
                    }
                    right_socket.send(json.dumps(data_right).encode("utf-8"))

                    data_left_real = {
                        "timestamp": timestamp,
                        "qpos": left_q_target_real.tolist(),
                        "joint_names": left_joint_names_real,
                        "type": "sim2real"
                    }
                    left_real_socket.send(json.dumps(data_left_real).encode("utf-8"))

                    data_right_real = {
                        "timestamp": timestamp,
                        "qpos": right_q_target_real.tolist(),
                        "joint_names": right_joint_names_real,
                        "type": "sim2real"
                    }
                    right_real_socket.send(json.dumps(data_right_real).encode("utf-8"))
                    
                    def fmt(arr):
                        return np.array2string(arr, precision=3, separator=', ', max_line_width=100000)
                    logger_mp.info(
                        "\n[ZMQ Publish]\n"
                        f"left_q_target       : {fmt(left_q_target)}\n"
                        f"right_q_target      : {fmt(right_q_target)}\n"
                        f"left_q_target_real  : {fmt(left_q_target_real)}\n"
                        f"right_q_target_real : {fmt(right_q_target_real)}"
                    )
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        except Exception as e:
            logger_mp.error(f"Error in control_process: {e}")
            raise 
        finally:
            if left_socket is not None:
                left_socket.close()
            if right_socket is not None:
                right_socket.close()
            if left_real_socket is not None:
                left_real_socket.close()
            if right_real_socket is not None:
                right_real_socket.close()
            if context is not None:
                context.term()
            logger_mp.info("Casia_Controller has been closed.")


if __name__ == "__main__":
    import argparse

    from televuer import TeleVuerWrapper

    parser = argparse.ArgumentParser(description='Casia Hand Controller with ZMQ publisher')
    parser.add_argument('--enable_zmq', type=bool, default=True, help='Enable ZMQ publishing')
    parser.add_argument('--zmq_left_port', type=int, default=5560, help='ZMQ port for left hand')
    parser.add_argument('--zmq_right_port', type=int, default=5561, help='ZMQ port for right hand')
    parser.add_argument('--zmq_left_real_port', type=int, default=5555, help='ZMQ port for left hand real')
    parser.add_argument('--zmq_right_real_port', type=int, default=5556, help='ZMQ port for right hand real')
    args = parser.parse_args()
    logger_mp.info(f"Command line arguments: {args}")

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(use_hand_tracking=True, 
                                 binocular=True, 
                                 img_shape=(480, 1280),
                                 display_fps=30.0,
                                 display_mode="pass-through", )

    left_hand_pos_array = Array('d', 75, lock = True)      # [input]
    right_hand_pos_array = Array('d', 75, lock = True)     # [input]
    hand_ctrl = Casia_Controller(left_hand_pos_array, right_hand_pos_array,
                                  enable_zmq=args.enable_zmq,
                                  zmq_left_port=args.zmq_left_port,
                                  zmq_right_port=args.zmq_right_port,
                                  zmq_left_real_port=args.zmq_left_real_port,
                                  zmq_right_real_port=args.zmq_right_real_port)
    
    logger_mp.info("Initialization complete. Waiting for user input to start control loop...")
    
    if input("Press Enter to start control loop...") == "":
        logger_mp.info(f"Received user input: start\n")
        while True:
            # head_img, head_img_fps = img_client.get_head_frame()
            # tv_wrapper.set_display_image(head_img)
            tele_data = tv_wrapper.get_tele_data()
            if tele_data is None:
                logger_mp.warning("tele_data is None, skipping this loop iteration.")
                time.sleep(0.01)
                continue
            
            with left_hand_pos_array.get_lock():
                left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
            with right_hand_pos_array.get_lock():
                right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()

            time.sleep(0.01)
