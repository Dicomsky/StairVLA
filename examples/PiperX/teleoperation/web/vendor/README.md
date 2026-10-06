# Vendored web assets

These are bundled so the headset needs only a LAN connection to the robot PC.

| File | Source | License |
|---|---|---|
| `aframe.min.js` | A-Frame 1.7.1, `https://aframe.io/releases/1.7.1/aframe.min.js` | MIT ([LICENSE-aframe](LICENSE-aframe)) |
| `Roboto-msdf.json`, `Roboto-msdf.png` | A-Frame's default text font, `https://cdn.aframe.io/fonts/` (MSDF atlas of Roboto) | Roboto: Apache License 2.0 |

The controller 3D models are not bundled: the page uses `model: false` and draws a small marker on each
controller instead.
