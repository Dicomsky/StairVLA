// PiperX VR teleoperation client.
// Streams both Touch controllers to the robot PC over wss:// (same host/port as this page)
// and renders the session status sent back by record.py on the page and in the headset.
// Controller reading follows XLeVR / telegrip (MIT, Copyright (c) 2025 Emil Rofors).

const PHASE_COLORS = {
  idle: '#3b4754', homing: '#b7791f', ready: '#1f7a4d', recording: '#c62828', saving: '#2f5fb3', stopped: '#2f5fb3',
};
const A_LABELS = {
  idle: 'A  move home', homing: 'A  (wait for home)', ready: 'A  start recording', recording: 'A  save episode',
  saving: 'A  (saving...)', stopped: 'A  -', teleop: 'A  -',
};

const $ = (id) => document.getElementById(id);
let ws = null;
let reconnectDelay = 500;
let status = {};
let labelsVisible = true;
let lastStickPressed = false;

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
  if (!on) setHud({ hint: 'Robot PC disconnected - reconnecting...' });
}

// ----------------------------------------------------------------------------- status rendering
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

function renderStatus() {
  const phase = status.phase || 'idle';
  $('phase').textContent = (status.title || phase).toUpperCase();
  $('phase').className = `phase phase-${phase}`;
  $('timer').textContent = phase === 'recording' ? formatTime(status.elapsed_s) : '';
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

  const info = status.record === false ? '' : `ep ${status.episode ?? '-'}/${status.target_episodes ?? '-'}  saved ${status.saved ?? 0}`;
  setHud({
    phase,
    title: (status.title || phase).toUpperCase(),
    hint: status.hint || '',
    task: status.task ? `Task: ${status.task}` : '',
    info: phase === 'recording' ? `${formatTime(status.elapsed_s)}   ${info}` : info,
    warn: warnings().join('  |  ') || status.message || '',
  });
  const labelA = $('labelA');
  if (labelA) labelA.setAttribute('value', A_LABELS[phase] || 'A');
  updateLabelVisibility();
}

function setHud(fields) {
  const set = (id, value) => { const el = $(id); if (el && value !== undefined) el.setAttribute('value', value); };
  set('hudPhase', fields.title);
  set('hudHint', fields.hint);
  set('hudTask', fields.task);
  set('hudInfo', fields.info);
  set('hudWarn', fields.warn);
  if (fields.phase) {
    $('hudPhaseBg').setAttribute('color', PHASE_COLORS[fields.phase] || PHASE_COLORS.idle);
    $('hudRec').setAttribute('visible', fields.phase === 'recording');
  }
}

function updateLabelVisibility() {
  // Labels stay out of the way while recording; the left stick click toggles them.
  const show = labelsVisible && status.phase !== 'recording';
  document.querySelectorAll('.labels').forEach((el) => el.setAttribute('visible', show));
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


// ----------------------------------------------------------------------------- frame axes
// Red = X, green = Y, blue = Z everywhere. Letters always face the viewer.
const AXES = [
  { name: 'X', color: '#ff4040', rotation: '0 0 -90', dir: [1, 0, 0] },
  { name: 'Y', color: '#40e040', rotation: '0 0 0', dir: [0, 1, 0] },
  { name: 'Z', color: '#4080ff', rotation: '90 0 0', dir: [0, 0, 1] },
];
const FONT = 'font="vendor/Roboto-msdf.json" font-image="vendor/Roboto-msdf.png"';
const billboards = [];
let framesVisible = true;
let lastRightStickPressed = false;

function makeTriad(parent, { length = 0.07, radius = 0.0025, opacity = 1.0, prefix = '', caption = '' } = {}) {
  const cone = length * 0.22;
  const material = `shader: flat; transparent: ${opacity < 1}; opacity: ${opacity}`;
  for (const axis of AXES) {
    const [x, y, z] = axis.dir;
    const shaft = length - cone;
    parent.insertAdjacentHTML('beforeend', `
      <a-cylinder position="${x * shaft / 2} ${y * shaft / 2} ${z * shaft / 2}" rotation="${axis.rotation}"
                  height="${shaft}" radius="${radius}" color="${axis.color}" material="${material}"></a-cylinder>
      <a-cone position="${x * (shaft + cone / 2)} ${y * (shaft + cone / 2)} ${z * (shaft + cone / 2)}" rotation="${axis.rotation}"
              height="${cone}" radius-bottom="${radius * 3}" radius-top="0" color="${axis.color}" material="${material}"></a-cone>
      <a-text class="axis-letter" ${FONT} value="${prefix}${axis.name}" align="center" width="${length * 2.2}" color="${axis.color}"
              position="${x * (length + 0.012)} ${y * (length + 0.012)} ${z * (length + 0.012)}"></a-text>`);
  }
  if (caption) {
    parent.insertAdjacentHTML('beforeend', `<a-text class="axis-letter" ${FONT} value="${caption}" align="center"
      width="${length * 3}" color="#ffffff" position="0 ${-0.02} 0"></a-text>`);
  }
  parent.querySelectorAll('.axis-letter').forEach((el) => billboards.push(el));
}

function faceCamera(camera) {
  // Keep axis letters readable: rotate each one to face the headset.
  const THREE = AFRAME.THREE;
  const camQ = new THREE.Quaternion();
  const parentQ = new THREE.Quaternion();
  camera.object3D.getWorldQuaternion(camQ);
  for (const el of billboards) {
    if (!el.object3D || !el.object3D.parent) continue;
    el.object3D.parent.getWorldQuaternion(parentQ);
    el.object3D.quaternion.copy(parentQ.invert().multiply(camQ));
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
    // Controller frames (what you hold), the real gripper's orientation (ghost) and the robot base axes.
    makeTriad(this.left, { length: 0.06 });
    makeTriad(this.right, { length: 0.06 });
    makeTriad(this.ghost, { length: 0.11, radius: 0.004, opacity: 0.45, caption: 'gripper' });
    makeTriad(this.robotFrame, { length: 0.09, radius: 0.003, prefix: '+', caption: 'robot base' });
  },
  tick() {
    const left = readController(this.left, 'left');
    const right = readController(this.right, 'right');
    if (left) {
      const stick = left.buttons.thumbstick;
      if (stick && !lastStickPressed) { labelsVisible = !labelsVisible; updateLabelVisibility(); }
      lastStickPressed = stick;
    }
    if (right) {
      const stick = right.buttons.thumbstick;
      if (stick && !lastRightStickPressed) framesVisible = !framesVisible;
      lastRightStickPressed = stick;
    }
    if (left || right) send({ type: 'controllers', t: Date.now(), left, right });
    this.updateFrames(left, right);
  },
  updateFrames(left, right) {
    const showGhost = framesVisible && !!right && !!status.gripper_frame_q;
    const showRobot = framesVisible && !!left && !!status.robot_frame_q;
    this.ghost.object3D.visible = showGhost;
    this.robotFrame.object3D.visible = showRobot;
    this.left.querySelectorAll('a-cylinder, a-cone, .axis-letter').forEach((el) => { el.object3D.visible = framesVisible; });
    this.right.querySelectorAll('a-cylinder, a-cone, .axis-letter').forEach((el) => { el.object3D.visible = framesVisible; });
    if (showGhost) {
      // Same position as the right controller, orientation of the real gripper (link6) in this frame.
      this.ghost.object3D.position.copy(this.right.object3D.position);
      setQuaternion(this.ghost, status.gripper_frame_q);
    }
    if (showRobot) {
      // Robot base axes as currently mapped (changes when you align with left X), above the left controller.
      this.robotFrame.object3D.position.copy(this.left.object3D.position);
      this.robotFrame.object3D.position.y += 0.08;
      setQuaternion(this.robotFrame, status.robot_frame_q);
    }
    faceCamera(this.head);
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
  button.textContent = ar ? 'Enter VR (passthrough)' : 'Enter VR';
  note.textContent = 'Passthrough keeps the real robot visible. The panel and labels follow your view.';
  button.onclick = () => {
    const start = ar && scene.enterAR ? scene.enterAR() : scene.enterVR();
    Promise.resolve(start).catch((err) => { note.textContent = `Could not start the XR session: ${err.message}`; });
  };
  scene.addEventListener('enter-vr', () => { $('hud').setAttribute('visible', true); });
  scene.addEventListener('exit-vr', () => { $('hud').setAttribute('visible', false); });
}

document.addEventListener('DOMContentLoaded', () => {
  const scene = $('scene');
  const attach = () => scene.setAttribute('teleop-stream', '');
  if (scene.hasLoaded) attach(); else scene.addEventListener('loaded', attach);
  setupEnterButton();
  connect();
});
