# dex_teleop
A Lightweight XR-based teleoperation for dexterous robot hands and humanoid arms.

The system receives hand and wrist tracking data from a Meta Quest headset, retargets the upper motion to the robot, and sends commands to either a MuJoCo simulation or physical hardware.

## System Overview
```
  XR headset
      |
      | Hand and wrist tracking
      v
  TeleVuer
      |
      | Retargeting and control
      v
  ZMQ publishers
      |
      +----> MuJoCo simulation
      |
      +----> Physical dex hands

  Default ZMQ ports: 

   Output         ARM  Left hand    Right hand  
  ━━━━━━━━━━━━    ━━━  ━━━━━━━━━    ━━━━━━━━━━
   Simulation     8559    5560          5561
  ────────────    ───  ─────────    ──────────
   Physical hand  8559    5555          5556
```

## Prepare Env
`conda create -n dex python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge`

**Also avaiable in python 3.11 for Robojudo-Plus**

Install the project dependencies into the active environment:
```sh
UV_PROJECT_ENVIRONMENT=PATH_TO_ANACONDA/envs/dex \
UV_PYTHON=$(which python) \
uv sync
```

## Set up XR Devices
For Quest3 XR Devices
1. Setup certificates
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
# Check the Generated Files
$ ls
build  cert.pem  key.pem  LICENSE  pyproject.toml  README.md  src  test

# this repo belongs to xr_teleoperate, so we use its config dir
mkdir -p ~/.config/xr_teleoperate/
cp cert.pem key.pem ~/.config/xr_teleoperate/
```

`sudo apt-get install -y libgl1-mesa-glx libglib2.0-0` 

The opencv-python wheel downloaded from PyPI requires the system's OpenGL core graphics library (libGL.so.1) and C language core library to run. However, because Ubuntu systems (or Docker images) are very streamlined by default and do not include these multimedia libraries, Python cannot find them.

2.  Allow Firewall Access
```bash
sudo ufw allow 8012
```

3. Test vuer
```sh
# run:
uv run teleop/televuer/example/test_tv_wrapper.py

# Connect XR device and your PC under identical WIFI
# Open browser on XR, open link: https://192.168.252.28:8012?ws=wss://192.168.252.28:8012
# Click the "pass-through" button in the bottom-left corner of the screen.
# Press Enter in the terminal to launch the program, after your hands have been detected by XR.
# Use the address reachable from the XR headset as <PC_IP> in the above instructions, `hostname -I`
```

## Sim2Sim
- G1 Arm Teleop:
```sh
python teleop/robot_control/vr_arm_hand_teleop.py --backend mujoco --hand casia --robot g1_23
```

- X2 Arm Teleop:
```sh
python teleop/robot_control/vr_arm_hand_teleop.py --backend mujoco --hand none --robot x2
```

For Dex hand, current only support CASIA hand
```sh
# Sim of left hand
uv run sim2sim/mujoco_receiver.py --xml-path assets/casia_hand_M/casia_left_hand.xml

# Sim of right hand
uv run sim2sim/mujoco_receiver.py --xml-path assets/casia_hand_M/casia_right_hand.xml

# Sim of both hands
uv run sim2sim/mujoco_receiver.py --xml-path assets/casia_hand_M/casia_left_hand.xml,assets/casia_hand_M/casia_right_hand.xml

# Teleop
uv run teleop/robot_control/robot_hand_casia_v2.py

```

## Sim2Real
- G1 Arm Teleop:
```sh
python teleop/robot_control/vr_arm_hand_teleop.py --backend real --hand casia --robot g1_23
```

- X2 Arm Teleop:
```sh
python teleop/robot_control/vr_arm_hand_teleop.py --backend real --hand none --robot x2
```

Following is for CAISA dex hands:
```sh
source /opt/zkgj_libs/setup.sh
cd ~/Desktop/toolkit/CH341SER_LINUX/driver/
sudo make install
lsusb 
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 002 Device 009: ID 04e8:4001 Samsung Electronics Co., Ltd PSSD T7
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 010: ID 045e:0745 Microsoft Corp. Nano Transceiver v1.0 for Bluetooth
Bus 001 Device 002: ID 5986:1193 Acer, Inc FHD Camera
Bus 001 Device 005: ID 8087:0036 Intel Corp. 
Bus 001 Device 004: ID 1038:1153 SteelSeries ApS SteelSeries ALC
Bus 001 Device 003: ID 1038:1122 SteelSeries ApS SteelSeries KLC
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub

sudo ln -s /dev/ttyCH341USB0 /dev/ttyUSB0

cd /home/breeze/Desktop/workplace/Humanoid/casia_hand_m_sdk_cpp/examples/teleop/build
./teleop


uv run teleop/robot_control/robot_hand_casia_v2.py
```

## Env setup problem
- This submodule cannot be run in NumPy 2.4.6 as it may crash, please downgrade to 'numpy<2' or try to upgrade the affected module.
- This submodule will automatically install a conda version scipy 1.17.1, which is not available in Robojudo, reinstall it using pypi 