# dex_teleop

## Prepare Env

`uv sync`

`uv run teleop/robot_control/robot_hand_casia_v2.py`

## Set up XR Devices
For Pico / Quest XR Devices
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

# or run:
uv run example/test_tv_wrapper.py

```

## Sim2Sim
```sh
# Sim of left hand
uv run sim2sim/mujoco_receiver.py --xml-path assets/casia_hand_M/casia_left_hand.xml --zmq_port 5560

# Sim of right hand
uv run sim2sim/mujoco_receiver.py --xml-path assets/casia_hand_M/casia_right_hand.xml --zmq_port 5561

# Teleop
uv run teleop/robot_control/robot_hand_casia_v2.py

```

Make sure left hand in the view

✅ Left only → left works
✅ Left + Right → both work
❌ Right only → nothing works

## Sim2Real
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