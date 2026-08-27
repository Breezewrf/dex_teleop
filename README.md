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
   Physical       8559    5555          5556
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

### OmniHandPro VR Retargeting

OmniHandPro is simulated as a separate dual-hand MuJoCo scene. The VR
retargeter publishes only the 12 active joints per hand. The remaining seven
joint coordinates are passive:

- thumb DIP follows thumb PIP;
- index and middle DIP follow their PIP joints;
- ring and pinky PIP/DIP follow their MCP joints.

The URDF mimic adaptor is used during retargeting, while playback uses the
nonlinear polynomial mappings defined by the MJCF.

#### Retargeting modes

`--omnihand-retargeting` selects how the 25 XR hand landmarks are converted to
the 12 active OmniHand targets. This is independent of the MuJoCo control mode.

| Mode | Target information | Advantages | Limitations |
| --- | --- | --- | --- |
| `dexpilot` (default) | Wrist-to-tip and fingertip-to-fingertip vectors | Stable pinch/grasp contact; DexPilot contact projection helps fingertips meet | Intermediate human finger joints are not directly constrained; contact projection has hysteresis |
| `vector` | 20 finger-chain vectors and 10 fingertip-pair vectors | MCP/PIP/DIP segment shape contributes to the optimization; usually better articulated poses | No DexPilot contact projection, so fingertip contact can be less sticky; more sensitive to human/robot proportions |

Start DexPilot retargeting:

```sh
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot x2 \
  --hand omnihand \
  --omnihand-retargeting dexpilot \
  --no-render
```

Start Vector retargeting:

```sh
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot x2 \
  --hand omnihand \
  --omnihand-retargeting vector \
  --no-render
```

#### MuJoCo playback modes

`--control-mode` belongs to `sim2sim/mujoco_receiver.py` and controls how the
received targets are played. It does not change the retarget optimizer.

| Mode | Active joints | Passive joints | Intended use |
| --- | --- | --- | --- |
| `position-actuator` | Sent to MuJoCo position actuators | Solved by MuJoCo equality constraints during `mj_step()` | Dynamics, actuator force limits, damping, contacts and physically meaningful response |
| `kinematic-coupled` | Written directly to `qpos` | Evaluated directly from the compiled MJCF equality polynomials, followed by `mj_forward()` | Low-latency retarget visualization and pose debugging |
| `qpos` (legacy default) | Written directly to `qpos` | Not explicitly reconstructed before stepping | CASIA and models whose commanded joint set already contains every displayed joint; not recommended for OmniHand |

For physics playback, start the receiver in one terminal:

```sh
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode position-actuator
```

This mode is intentionally slower: actuator `forcerange`, joint damping,
contacts and one actuator driving multiple coupled joints all affect response.

For low-latency visualization, start the kinematic coupled receiver instead:

```sh
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode kinematic-coupled \
  --smoothing-alpha 1.0 \
  --interpol-steps 1
```

Then start either retargeting command above in another terminal. `--no-render`
disables the separate X2 arm viewer so only the OmniHand scene is displayed.

#### VR target-hand visualization

The receiver can overlay the VR target hand on the corresponding simulated
palm. Add `--target-hand-mode` to either OmniHand playback command:

| Mode | Display | Intended use |
| --- | --- | --- |
| `landmarks` | 25 landmark spheres | Inspect tracking jitter, jumps, and invalid points |
| `skeleton` | 25 spheres and the complete human-hand skeleton | Compare the raw human pose with the dex hand |
| `constraints` | 25 spheres and the vectors actually passed to the optimizer | Inspect Vector/DexPilot objectives and DexPilot contact projection |
| `none` (default) | No overlay | Normal playback with no visualization overhead |

For example:

```sh
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode kinematic-coupled \
  --target-hand-mode constraints \
  --smoothing-alpha 1.0 \
  --interpol-steps 1
```

Left-hand geometry is cyan and right-hand geometry is orange. In
`constraints` mode, a red vector means DexPilot replaced the measured human
distance with its current projected contact distance (the eta1/eta2 contact
target). Non-red vectors are the scaled human reference vectors.

The overlay protocol is model-independent. The sender provides the 25 points,
the optimizer's `target_link_human_indices`, actual scaled/projected vectors,
and an optional palm anchor. The receiver first uses that anchor and otherwise
infers the common MuJoCo ancestor of the commanded joints. OmniHand and CASIA
publish this payload now; other dex-hand controllers can reuse
`HandRetargeting.target_hand_visualization()` after applying their own input
coordinate conversion and calling `retarget()`.

Recommended combinations:

| Goal | Retargeting | Playback |
| --- | --- | --- |
| Lowest latency and finger-shape debugging | `vector` | `kinematic-coupled` |
| Pinch/OK/contact-oriented visualization | `dexpilot` | `kinematic-coupled` |
| Evaluate physical response and coupling dynamics | `dexpilot` or `vector` | `position-actuator` |

#### Tuning and known limitations

- Both retarget configs currently use `low_pass_alpha: 0.2`. The receiver also
  defaults to `--smoothing-alpha 0.2`; using both creates two cascaded filters.
  For responsive kinematic playback, keep the retarget filter and use
  `--smoothing-alpha 1.0 --interpol-steps 1` on the receiver.
- `scaling_factor` is an isotropic scale: it changes finger length and lateral
  finger spacing together. Human and OmniHand proportions are different, so a
  value that matches finger length may still bias an ABAD joint.
- The active joint message order places `middle_abad_joint` at index 7. For the
  left hand, a persistent negative value turns the middle finger toward the
  ring finger; for the right hand, the corresponding direction is positive.
- Ring and pinky each have only one active flexion joint. Their PIP/DIP poses
  must remain on the MJCF coupling curve and cannot reproduce arbitrary human
  MCP/PIP/DIP combinations.
- `kinematic-coupled` preserves the nonlinear pose mapping but intentionally
  bypasses actuator force limits, dynamics and contact response. Use
  `position-actuator` whenever those effects matter.

Example: Vector retargeting with low-latency coupled playback uses these two
commands in separate terminals:

```sh
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode kinematic-coupled \
  --smoothing-alpha 1.0 \
  --interpol-steps 1
```

```sh
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot x2 \
  --hand omnihand \
  --omnihand-retargeting vector \
  --no-render
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
