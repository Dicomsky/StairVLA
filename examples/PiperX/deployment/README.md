# PiperX robot client (TODO)

This folder will hold the real-robot client used in the paper: camera capture, the websocket
client that queries `deployment/model_server/server_policy.py`, end-effector to joint-target
conversion through local inverse kinematics, and the safety limits (per-step translation ≤ 0.05 m,
rotation ≤ 0.20 rad, joint velocity ≤ 25°/s).
