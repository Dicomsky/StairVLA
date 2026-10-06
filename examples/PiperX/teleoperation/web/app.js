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

AFRAME.registerComponent('teleop-stream', {
  init() {
    this.left = $('leftHand');
    this.right = $('rightHand');
  },
  tick() {
    const left = readController(this.left, 'left');
    const right = readController(this.right, 'right');
    if (left) {
      const stick = left.buttons.thumbstick;
      if (stick && !lastStickPressed) { labelsVisible = !labelsVisible; updateLabelVisibility(); }
      lastStickPressed = stick;
    }
    if (left || right) send({ type: 'controllers', t: Date.now(), left, right });
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
