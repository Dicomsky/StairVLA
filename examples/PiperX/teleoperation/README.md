# PiperX VR teleoperation and data collection

This is the setup used to collect the PiperX demonstrations (Fruit25, PushBlock). A Meta Quest headset
streams the right controller pose to the robot PC over WebXR. The pose is mapped to an end-effector
target and solved with local IK, and the arm follows in joint space at 30 Hz. Each recorded frame
stores the measured joint state, the commanded joint target and both camera images as a
[LeRobot v3.0](https://github.com/huggingface/lerobot) dataset. The
[dataset tools](../dataset_tools/README.md) then convert the recordings into the EE-delta datasets
used for training.

Everything runs in the StairVLA environment. The `lerobot` package is not required.

## Setup

- **Hardware:** an AgileX PiperX on a CAN interface (`can0`), a top UVC camera and a wrist Intel
  RealSense (both 640×480 @ 30 fps), and a Meta Quest 2, 3 or Pro on the same network as the robot PC.
- **Packages:** `pip install -r examples/PiperX/requirements.txt`. You also need `openssl`, which
  is used once to create the local HTTPS certificate.
- **CAN:** bring the CAN interface up with the AgileX scripts, e.g. `bash can_activate.sh can0 1000000`.
- **Cameras:** set `--wrist-realsense-serial` to your RealSense serial (`rs-enumerate-devices`)
  and `--top-opencv-index` to the top camera's `/dev/videoN`.

## Record

```bash
python examples/PiperX/teleoperation/record.py \
    --root data/piperx/MyTask_raw \
    --task "Pick up the apple and place it in the red basket." \
    --num-episodes 50 \
    --wrist-realsense-serial <serial> --top-opencv-index 0
```

1. Open the printed `https://<robot-pc-ip>:8443` in the Quest browser. Accept the certificate
   warning once; the certificate is self-signed and stored in `~/.cache/stairvla/vr_tls/`.
2. Press **Start Controller Tracking** (passthrough keeps the real robot visible). As in XLeVR, each
   controller shows its model, RGB = XYZ axes and a Pos/Rot readout. Nothing floats in front of you:
   the current step and the next button are written on the right controller (red while recording),
   `Y: home  X: align` on the left one, and both hide while you hold the grip.
3. Run one episode:

   | Step | Press | What happens |
   |---|---|---|
   | Idle | **A** | The arm moves slowly to the home pose. Reset the scene meanwhile. |
   | Ready | **A** | Recording starts (red dot and timer). |
   | Recording | **A** / **B** | **A** saves the episode, **B** discards it so you can record it again. |

   While teleoperating, **hold the right grip** to move the arm. Release it to re-position your hand
   (clutch). The **right trigger** closes the gripper. **Left Y** sends the arm home and **left X**
   aligns the frame (point the right controller's red X axis at robot +X and press). Clicking the
   **right stick** shows alignment aids: the real gripper's orientation as translucent axes at the
   right controller (match your controller's axes to it) and the robot base axes above the left
   controller. The controller vibrates when recording starts,
   when an episode is saved or discarded, and when a target is out of reach.

The same steps work from the robot PC keyboard: <kbd>Space</kbd>/<kbd>→</kbd> = A,
<kbd>Backspace</kbd>/<kbd>←</kbd> = B, <kbd>h</kbd> = home. <kbd>q</kbd>/<kbd>Esc</kbd> quits and
saves an episode in progress; <kbd>Ctrl+C</kbd> aborts and discards it. The terminal keeps a single
status line updated.

To record several tasks into one dataset, run again with a new `--task` and `--resume`. Each
episode stores its own instruction.

### Other modes

```bash
# Teleoperate without recording (practice, checking the camera views)
python examples/PiperX/teleoperation/record.py --no-record

# No hardware at all: simulated arm and cameras, to try the headset setup and mapping
python examples/PiperX/teleoperation/record.py --mock-robot --root /tmp/vr_test --task "test"
```

## Recorded data

| Key | Shape | Content |
|---|---|---|
| `observation.state` | 7 | measured joints 1–6 (deg) and gripper opening (mm) |
| `action` | 7 | joint target sent to the arm (IK solution, deg) and gripper target (mm) |
| `observation.images.top`, `observation.images.wrist` | 480×640×3 | AV1 video, 30 fps |

The released raw datasets used the defaults: 30 Hz recording, `--speed-ratio 100`, high-follow
mode, and home pose `0 62.512 -66.452 83 0 0` (deg). Do not change them if you want new data to be
compatible. Training does not use these joint actions directly. `dataset_tools/convert_to_ee.py`
recomputes the actions from consecutive measured states (temporal EE deltas).

## Useful options

| Option | Default | Meaning |
|---|---|---|
| `--scale` | 600 | robot mm per metre of controller motion |
| `--max-linear-speed` / `--max-angular-speed` | 350 mm/s / 160 deg/s | per-step limits on the EE target |
| `--vr-timeout-s` | 0.5 | hold the arm if the headset stops sending |
| `--max-episode-s` | 0 (off) | auto-save after this duration |
| `--vcodec` | `av1` | `h264` encodes faster on slow CPUs |
| `--vr-port` | 8443 | port of the page and the WebSocket |

## Files

| File | Purpose |
|---|---|
| `record.py` | entry point: control loop, episode state machine, terminal UI |
| `vr_teleop.py` | controller → EE target → IK joint target, slow homing |
| `vr_mapping.py` | VR-to-robot frame mapping, filtering and speed limits |
| `vr_server.py` | HTTPS + WebSocket server for the headset page |
| `dataset_writer.py` | LeRobot v3.0 dataset writer (streams video while recording) |
| `web/` | WebXR page (A-Frame, vendored in `web/vendor/`): controller axes and readouts, status line, haptics |

The shared IK, URDF and robot driver live in [`../common/`](../common). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Troubleshooting

- **The page loads but "Start Controller Tracking" is disabled:** open it in the Quest browser itself, not a casting
  view. All page assets (A-Frame, font, controller models) are bundled in `web/vendor/`, so only the LAN connection to
  the robot PC is needed.
- **"Disconnected" on the page:** the robot PC firewall must allow TCP 8443.
- **The arm does not move:** you must hold the right grip. Check that the hint on the right controller says
  *Ready* or *Recording* and not *Homing*.
- **Motion feels rotated:** stand facing the robot, point the right controller along robot +X and
  press left X.
- **"video encoder falling behind":** use `--vcodec h264`.
