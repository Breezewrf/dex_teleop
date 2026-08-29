# dex_teleop
A Lightweight XR-based teleoperation for dexterous robot hands and humanoid arms.

The system receives hand and wrist tracking data from a Meta Quest 3 headset, retargets the upper motion to the robot, and sends commands to either a MuJoCo simulation or physical hardware.

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

### Integrated Arm + Hand Teleop

The unified entry point controls the robot arms and dexterous hands together.
This is the recommended and most commonly used workflow. A separate
`mujoco_receiver.py` is not required.

```sh
# X2 arm with OmniHand using DexPilot
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot x2 \
  --hand omnihand \
  --omnihand-retargeting dexpilot

# X2 arm with OmniHand using Vector
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot x2 \
  --hand omnihand \
  --omnihand-retargeting vector

# G1 arm with CASIA hands
python teleop/robot_control/vr_arm_hand_teleop.py \
  --backend mujoco \
  --robot g1_23 \
  --hand casia
```

In the integrated viewer, X2 writes the 12 active OmniHand joints and
evaluates the 7 passive-joint polynomials. CASIA writes all 14 joints per
hand directly. `--mujoco-control {kinematic,pd}` controls the robot arms; it is
separate from the standalone hand receiver's `--control-mode`.

### Standalone hand simulation

Standalone hand teleoperation only initializes VR hand tracking and hand
retargeting. It does not initialize G1/X2 arm IK. Run the MuJoCo receiver and
the corresponding hand controller in separate terminals.

#### Standalone OmniHand

Start the dual-hand OmniHand MuJoCo scene. `kinematic-coupled` evaluates the
MJCF passive-joint polynomials immediately for low-latency playback:

```sh
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode kinematic-coupled \
  --smoothing-alpha 1.0 \
  --interpol-steps 1
```

Then start hand-only OmniHand retargeting in another terminal:

```sh
# DexPilot is the default
uv run teleop/robot_control/robot_hand_omnihand.py

# Or use Vector retargeting
uv run teleop/robot_control/robot_hand_omnihand.py --retargeting vector
```

#### Standalone CASIA

Start one of the CASIA MuJoCo scenes:

```sh
# Left hand only
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/casia_hand_M/casia_left_hand.xml \
  --control-mode qpos

# Right hand only
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/casia_hand_M/casia_right_hand.xml \
  --control-mode qpos

# Both hands
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/casia_hand_M/casia_left_hand.xml,assets/casia_hand_M/casia_right_hand.xml \
  --subscribe-both \
  --control-mode qpos
```

Then start the hand-only CASIA controller in another terminal:

```sh
uv run teleop/robot_control/robot_hand_casia_v2.py
```

### Retargeting optimizers

Retargeting converts the 25 XR landmarks into robot-hand joint targets. It is
independent of the MuJoCo playback mode.

| Hand | Available optimizer | Standalone option | Integrated option |
| --- | --- | --- | --- |
| OmniHand | DexPilot (default), Vector | `--retargeting` | `--omnihand-retargeting` |
| CASIA | DexPilot | Fixed by `casia.yml` | Fixed by `casia.yml` |

The two OmniHand optimizers use different target information:

| Mode | Target information | Advantages | Limitations |
| --- | --- | --- | --- |
| `dexpilot` | Wrist-to-tip and fingertip-to-fingertip vectors | Stable pinch/grasp contact; contact projection helps fingertips meet | Intermediate human finger joints are not directly constrained; contact projection has hysteresis |
| `vector` | 20 finger-chain vectors and 10 fingertip-pair vectors | MCP/PIP/DIP segment shape contributes to optimization; usually better articulated poses | No contact projection, so fingertip contact can be less sticky; more sensitive to human/robot proportions |

OmniHand publishes 12 active joints per hand. Its remaining seven joint
coordinates are passive: thumb DIP follows thumb PIP; index and middle DIP
follow their PIP joints; and ring/pinky PIP and DIP follow their MCP joints.
The URDF mimic adaptor represents these joints during retargeting, while the
MJCF contains the nonlinear polynomial mappings used for playback.

### MuJoCo standalone playback modes

`--control-mode` belongs to `sim2sim/mujoco_receiver.py`. It controls how a
standalone MuJoCo hand follows received joint targets and does not select the
retarget optimizer.

| Mode | Behavior | OmniHand | CASIA |
| --- | --- | --- | --- |
| `qpos` | Writes every commanded joint directly to `qpos` | Legacy mode; passive joints are not explicitly reconstructed before stepping | Recommended; all 14 displayed joints are commanded |
| `kinematic-coupled` | Writes active joints to `qpos`, evaluates compiled equality polynomials, then calls `mj_forward()` | Recommended for low-latency visualization and pose debugging | Works like direct `qpos` because CASIA has no coupled equalities |
| `position-actuator` | Sends targets to MuJoCo position actuators and advances with `mj_step()` | Use for actuator limits, damping, contacts and physical response | Not supported by the current actuator-free standalone CASIA XMLs |

`position-actuator` is intentionally slower because actuator `forcerange`,
joint damping, contacts and one actuator driving multiple coupled joints all
affect the response. `kinematic-coupled` preserves OmniHand's nonlinear pose
mapping but intentionally bypasses those physical effects.

Recommended OmniHand combinations:

| Goal | Retargeting | Playback |
| --- | --- | --- |
| Lowest latency and finger-shape debugging | `vector` | `kinematic-coupled` |
| Pinch/OK/contact-oriented visualization | `dexpilot` | `kinematic-coupled` |
| Evaluate physical response and coupling dynamics | `dexpilot` or `vector` | `position-actuator` |

### VR target-hand visualization

`mujoco_receiver.py` can overlay the VR target hand on the corresponding
simulated palm. This protocol is supported by both CASIA and OmniHand. Add
`--target-hand-mode` to a standalone receiver command:

| Mode | Display | Intended use |
| --- | --- | --- |
| `landmarks` | 25 landmark spheres | Inspect tracking jitter, jumps and invalid points |
| `skeleton` | 25 spheres and the complete human-hand skeleton | Compare the raw human pose with the dex hand |
| `constraints` | 25 spheres and the vectors passed to the optimizer | Inspect Vector/DexPilot objectives and DexPilot contact projection |
| `none` (default) | No overlay | Normal playback with no visualization overhead |

For example:

```sh
# OmniHand target constraints
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/o12_hand_description-o12_t3/assets/MJCF/scene.xml \
  --subscribe-both \
  --control-mode kinematic-coupled \
  --target-hand-mode constraints

# CASIA target constraints
uv run sim2sim/mujoco_receiver.py \
  --xml-path assets/casia_hand_M/casia_left_hand.xml,assets/casia_hand_M/casia_right_hand.xml \
  --subscribe-both \
  --control-mode qpos \
  --target-hand-mode constraints
```

Left-hand geometry is cyan and right-hand geometry is orange. In
`constraints` mode, **a red vector means DexPilot replaced the measured human
distance with its projected eta1/eta2 contact target**. Non-red vectors are the
scaled human reference vectors.

The sender provides the landmarks, optimizer indices, scaled/projected vectors
and an optional palm anchor. Other dex-hand controllers can reuse
`HandRetargeting.target_hand_visualization()` after applying their own input
coordinate conversion and calling `retarget()`.

The CASIA and OmniHand controllers used by `vr_arm_hand_teleop.py` also publish
this payload. An external `mujoco_receiver.py` can display it, but the
integrated G1/X2 viewer does not currently render the target-hand overlay.

### OmniHand tuning and known limitations

- Ring and pinky each have only one active flexion joint. Their PIP/DIP poses
  must remain on the MJCF coupling curve and cannot reproduce arbitrary human
  MCP/PIP/DIP combinations.
- `kinematic-coupled` preserves the nonlinear pose mapping but bypasses
  actuator force limits, dynamics and contact response. Use
  `position-actuator` whenever those effects matter.


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
