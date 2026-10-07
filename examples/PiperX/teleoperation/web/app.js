// PiperX VR teleoperation client.
// Streams both Touch controllers to the robot PC over wss:// (same host/port as this page)
// and shows the session status sent back by record.py.
// The in-headset view follows XLeVR / telegrip (MIT, Copyright (c) 2025 Emil Rofors):
// controller models, RGB axes and a Pos/Rot readout on each controller, headset readout on top.

const $ = (id) => document.getElementById(id);
let ws = null;
let reconnectDelay = 500;
let status = {};
let aidsVisible = false; // gripper ghost + robot base axes (right stick click)
let lastRightStick = false;

// ----------------------------------------------------------------------------- websocket
function connect() {
  ws = new WebSocket(`wss://${location.host}`);
  ws.onopen = () => {
    reconnectDelay = 500;
    setConnection(true);
  };
  ws.onclose = () => {
    ws = null;
    setConnection(false);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, 4000);
  };
  ws.onerror = () => {};
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch (e) { return; }
    if (msg.type === 'status') { status = msg; renderStatus(); }
    if (msg.type === 'haptic') pulse(msg.hand, msg.intensity, msg.duration_ms);
  };
}

function send(message) {
  // Drop frames instead of queueing them if the network stalls: stale poses are useless.
  if (ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 16384) ws.send(JSON.stringify(message));
}

function setConnection(on) {
  const pill = $('connection');
  pill.textContent = on ? 'Connected' : 'Disconnected';
  pill.className = `pill ${on ? 'pill-on' : 'pill-off'}`;
  if (!on) setStatusText('Robot PC disconnected - reconnecting...', '#ff6060');
}

// ----------------------------------------------------------------------------- status
function formatTime(seconds) {
  if (seconds === undefined || seconds === null) return '';
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function warnings() {
  const out = [];
  if (status.tracking === false) out.push('right controller not tracked');
  if (status.ik_ok === false) out.push('target out of reach');
  if (status.encoder_backlog > 30) out.push('video encoder falling behind');
  return out;
}

function setFlag(id, on, bad = false) {
  $(id).className = `flag${on ? ' on' : ''}${bad ? ' bad' : ''}`;
}

// Short hint on the right controller: step + next button (+ warning). Hidden while gripping.
const NEXT = {
  idle: 'A: move home', homing: 'moving home...  B: cancel', ready: 'A: start recording',
  recording: 'A: save   B: discard', saving: 'saving...', stopped: 'done', teleop: 'grip: move',
};

function setStatusText(text, color) {
  const el = $('rightHandHint');
  if (!el) return;
  el.setAttribute('value', text);
  el.setAttribute('color', color);
}

function renderStatus() {
  const phase = status.phase || 'idle';
  const title = (status.title || phase).toUpperCase();
  const recording = phase === 'recording';
  const episode = status.record === false ? '' : `ep ${status.episode ?? '-'}/${status.target_episodes ?? '-'}`;

  // 2D page (browser / monitor)
  $('phase').textContent = title;
  $('phase').className = `phase phase-${phase}`;
  $('timer').textContent = recording ? formatTime(status.elapsed_s) : '';
  $('hint').textContent = status.hint || '';
  $('episode').textContent = status.record === false ? 'not recording' : `${status.episode ?? '-'} / ${status.target_episodes ?? '-'}`;
  $('saved').textContent = status.record === false ? '-' : `${status.saved ?? 0}`;
  $('task').textContent = status.task || '-';
  $('message').textContent = status.message || '';
  setFlag('flagTracking', !!status.tracking, status.tracking === false);
  setFlag('flagDeadman', !!status.deadman);
  setFlag('flagGripper', !!status.gripper_closed);
  setFlag('flagIk', status.ik_ok !== false, status.ik_ok === false);
  $('flagGripper').textContent = status.gripper_closed ? 'Gripper closed' : 'Gripper open';

  // Headset: a short hint lying on the right controller (no text in front of the eyes).
  const head = [recording ? `REC ${formatTime(status.elapsed_s)}` : title, episode].filter(Boolean).join('   ');
  const warn = warnings().join(' | ');
  const lines = [head, NEXT[phase] || ''];
  if (warn) lines.push(warn);
  setStatusText(lines.join('\n'), recording ? '#ff5050' : warn ? '#ffb020' : '#ffffff');
}

// ----------------------------------------------------------------------------- controllers
function gamepadOf(handEl) {
  const tracked = handEl && handEl.components && handEl.components['tracked-controls'];
  return tracked && tracked.controller ? tracked.controller.gamepad : null;
}

function pulse(hand, intensity = 0.5, durationMs = 60) {
  const gamepad = gamepadOf($(hand === 'left' ? 'leftHand' : 'rightHand'));
  const actuator = gamepad && gamepad.hapticActuators && gamepad.hapticActuators[0];
  if (actuator && actuator.pulse) actuator.pulse(intensity, durationMs).catch(() => {});
}

function readController(handEl, hand) {
  if (!handEl || !handEl.object3D) return null;
  const gamepad = gamepadOf(handEl);
  if (!gamepad) return null;
  const p = handEl.object3D.position;
  const q = handEl.object3D.quaternion;
  const pressed = (i) => !!(gamepad.buttons[i] && gamepad.buttons[i].pressed);
  const value = (i) => (gamepad.buttons[i] ? gamepad.buttons[i].value : 0);
  // WebXR "xr-standard" mapping on Touch controllers: 0 trigger, 1 squeeze, 3 stick press, 4 A/X, 5 B/Y.
  const face1 = pressed(4);
  const face2 = pressed(5);
  const buttons = { squeeze: pressed(1), thumbstick: pressed(3) };
  if (hand === 'left') Object.assign(buttons, { x: face1, y: face2 });
  else Object.assign(buttons, { a: face1, b: face2 });
  return {
    position: { x: p.x, y: p.y, z: p.z },
    quaternion: { x: q.x, y: q.y, z: q.z, w: q.w },
    trigger: value(0),
    gripActive: buttons.squeeze,
    buttons,
  };
}

function poseText(object3D) {
  const p = object3D.position;
  const r = object3D.rotation;
  const deg = (v) => (v * 180 / Math.PI).toFixed(0);
  return `Pos: ${p.x.toFixed(2)} ${p.y.toFixed(2)} ${p.z.toFixed(2)}\nRot: ${deg(r.x)} ${deg(r.y)} ${deg(r.z)}`;
}

// ----------------------------------------------------------------------------- axes
// Red = X, green = Y, blue = Z (same sizes as the original XLeVR indicators).
const AXES = [
  { color: '#ff0000', rotation: '0 0 -90', dir: [1, 0, 0] },
  { color: '#00ff00', rotation: '0 0 0', dir: [0, 1, 0] },
  { color: '#0000ff', rotation: '90 0 0', dir: [0, 0, 1] },
];

function addAxes(parent, { length = 0.08, radius = 0.003, tip = 0.015, opacity = 1 } = {}) {
  const material = opacity < 1 ? `material="transparent: true; opacity: ${opacity}"` : '';
  for (const { color, rotation, dir } of AXES) {
    const at = (d) => dir.map((v) => v * d).join(' ');
    parent.insertAdjacentHTML('beforeend', `
      <a-cylinder position="${at(length / 2)}" rotation="${rotation}" height="${length}" radius="${radius}" color="${color}" ${material}></a-cylinder>
      <a-cone position="${at(length + tip / 2)}" rotation="${rotation}" height="${tip}" radius-bottom="${radius * 2.7}" radius-top="0" color="${color}" ${material}></a-cone>`);
  }
}

function setQuaternion(el, q) {
  if (q && q.length === 4) el.object3D.quaternion.set(q[0], q[1], q[2], q[3]);
}

AFRAME.registerComponent('teleop-stream', {
  init() {
    this.left = $('leftHand');
    this.right = $('rightHand');
    this.head = $('head');
    this.ghost = $('gripperGhost');
    this.robotFrame = $('robotFrame');
    addAxes(this.left);
    addAxes(this.right);
    addAxes(this.ghost, { length: 0.12, radius: 0.005, tip: 0.02, opacity: 0.45 });
    addAxes(this.robotFrame, { length: 0.1, radius: 0.004, tip: 0.018 });
  },
  tick() {
    const left = readController(this.left, 'left');
    const right = readController(this.right, 'right');
    if (right) {
      if (right.buttons.thumbstick && !lastRightStick) aidsVisible = !aidsVisible;
      lastRightStick = right.buttons.thumbstick;
    }
    if (left || right) send({ type: 'controllers', t: Date.now(), left, right });

    $('leftHandInfo').setAttribute('value', poseText(this.left.object3D));
    $('rightHandInfo').setAttribute('value', poseText(this.right.object3D));
    $('headsetInfo').setAttribute('value', poseText(this.head.object3D));
    // Hide the hints while the right grip is held, i.e. while actually teleoperating.
    const gripping = !!(right && right.buttons.squeeze);
    $('leftHandHint').object3D.visible = !gripping;
    $('rightHandHint').object3D.visible = !gripping;
    this.updateAids(left, right);
  },
  updateAids(left, right) {
    // Optional alignment aids: the real gripper orientation (translucent) drawn at the right controller,
    // and the robot base axes as currently mapped above the left controller.
    const showGhost = aidsVisible && !!right && !!status.gripper_frame_q;
    const showRobot = aidsVisible && !!left && !!status.robot_frame_q;
    this.ghost.object3D.visible = showGhost;
    this.robotFrame.object3D.visible = showRobot;
    if (showGhost) {
      this.ghost.object3D.position.copy(this.right.object3D.position);
      setQuaternion(this.ghost, status.gripper_frame_q);
    }
    if (showRobot) {
      this.robotFrame.object3D.position.copy(this.left.object3D.position);
      this.robotFrame.object3D.position.y += 0.1;
      setQuaternion(this.robotFrame, status.robot_frame_q);
    }
  },
});

// ----------------------------------------------------------------------------- XR session
async function setupEnterButton() {
  const button = $('enterXr');
  const note = $('xrNote');
  const scene = $('scene');
  if (!navigator.xr) {
    note.textContent = 'WebXR is not available in this browser. Open this page in the Meta Quest browser.';
    return;
  }
  const ar = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  const vr = ar ? false : await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
  if (!ar && !vr) {
    note.textContent = 'No immersive WebXR session available here. This page is still useful as a status monitor.';
    return;
  }
  button.disabled = false;
  button.textContent = ar ? 'Start Controller Tracking' : 'Enter VR';
  note.textContent = 'Passthrough keeps the real robot visible.';
  button.onclick = () => {
    const start = ar && scene.enterAR ? scene.enterAR() : scene.enterVR();
    Promise.resolve(start).catch((err) => { note.textContent = `Could not start the XR session: ${err.message}`; });
  };
}

document.addEventListener('DOMContentLoaded', () => {
  const scene = $('scene');
  const attach = () => scene.setAttribute('teleop-stream', '');
  if (scene.hasLoaded) attach(); else scene.addEventListener('loaded', attach);
  setupEnterButton();
  connect();
});
