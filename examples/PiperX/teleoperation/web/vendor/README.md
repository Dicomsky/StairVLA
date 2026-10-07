# Vendored web assets

Bundled so the headset needs only a LAN connection to the robot PC. `index.html` sets
`window.AFRAME_CDN_ROOT = "vendor/cdn/"`, so A-Frame loads the files below instead of `cdn.aframe.io`.

| File | Source | License |
|---|---|---|
| `aframe.min.js` | A-Frame 1.7.1, `https://aframe.io/releases/1.7.1/aframe.min.js` | MIT ([LICENSE-aframe](LICENSE-aframe)) |
| `cdn/fonts/Roboto-msdf.*` | A-Frame's default text font, `https://cdn.aframe.io/fonts/` | Roboto: Apache License 2.0 |
| `cdn/controllers/{oculus,meta}/*` | Meta Quest Touch controller models used by `oculus-touch-controls`, `https://cdn.aframe.io/controllers/` (Quest 1/2/3/Pro) | distributed by the A-Frame project |
