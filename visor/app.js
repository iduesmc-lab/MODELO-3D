import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

const MODEL_URL = '../modelo/chibi_gamer.vrm';
const CHAIR_URL = '../modelo/silla_gamer.glb';
const MP_VERSION = '0.10.14';
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };

// ------------------------------------------------------------------ escena
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true,
                                           preserveDrawingBuffer: params.has('shot') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, innerWidth / innerHeight, 0.05, 50);
camera.position.set(0, 0.95, 2.6);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.82, 0);
controls.enableDamping = true;
controls.update();

const sun = new THREE.DirectionalLight(0xffffff, Math.PI * 0.9);
sun.position.set(0.6, 1.4, 1.6);
scene.add(sun);
scene.add(new THREE.AmbientLight(0xffffff, 0.9));

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

// ------------------------------------------------------------------ fondo
const BGS = { 'Oscuro': '#34363f', 'Verde': '#00ff00', 'Azul': '#0047ff', 'Blanco': '#f4f2fa',
              'Transparente': 'transparent' };
function setBg(name) {
  const c = BGS[name] ?? name;
  document.body.classList.toggle('transparent', c === 'transparent');
  document.body.style.background = c === 'transparent' ? 'transparent' : c;
  document.querySelectorAll('#bgs button').forEach((b) => b.classList.toggle('on', b.dataset.bg === name));
}
for (const name of Object.keys(BGS)) {
  const b = document.createElement('button');
  b.textContent = name; b.dataset.bg = name; b.onclick = () => setBg(name);
  $('bgs').appendChild(b);
}
const bgParam = params.get('bg');
setBg(bgParam === 'green' ? 'Verde' : bgParam === 'transparent' ? 'Transparente'
      : bgParam === 'blue' ? 'Azul' : bgParam ? '#' + bgParam.replace('#', '') : 'Oscuro');
if (params.get('ui') === '0' || params.has('shot')) document.body.classList.add('noui');
addEventListener('keydown', (e) => {
  if (e.key === 'h' || e.key === 'H') document.body.classList.toggle('noui');
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= EXPRS.length) setExpr(EXPRS[n - 1][1]);
});

// ------------------------------------------------------------------ silla (toon + contorno)
const chair = new THREE.Group();
scene.add(chair);
const gradient = new THREE.DataTexture(new Uint8Array([90, 90, 90, 255, 255, 255, 255, 255]), 2, 1);
gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
gradient.needsUpdate = true;
function outlineMaterial(width, color) {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: { w: { value: width }, c: { value: new THREE.Color(color) } },
    vertexShader: 'uniform float w; void main(){ vec3 p = position + normal * w;'
      + ' gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0); }',
    fragmentShader: 'uniform vec3 c; void main(){ gl_FragColor = vec4(c, 1.0); }',
  });
}
new GLTFLoader().load(CHAIR_URL, (gltf) => {
  const meshes = [];
  gltf.scene.traverse((o) => { if (o.isMesh) meshes.push(o); });
  for (const m of meshes) {
    const src = m.material;
    m.material = new THREE.MeshToonMaterial({ color: src.color, gradientMap: gradient });
    const ol = new THREE.Mesh(m.geometry, outlineMaterial(0.0035, 0x07060b));
    m.add(ol);
  }
  chair.add(gltf.scene);
});

// ------------------------------------------------------------------ VRM
let vrm = null;
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

async function loadVRM(url) {
  status('Cargando modelo…');
  const gltf = await loader.loadAsync(url);
  const v = gltf.userData.vrm;
  if (!v) throw new Error('El archivo no es un VRM válido');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons?.(gltf.scene);
  VRMUtils.rotateVRM0(v);
  if (vrm) { scene.remove(vrm.scene); VRMUtils.deepDispose(vrm.scene); }
  vrm = v;
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  scene.add(vrm.scene);
  vrm.lookAt.target = camera;
  applyPose();
  status(`${vrm.meta?.title ?? vrm.meta?.name ?? 'Modelo'} listo ✔`);
  window.__vrm = vrm;
  window.__ready = true;
}
loadVRM(params.get('model') ?? MODEL_URL).catch((e) => { status('Error: ' + e.message); console.error(e); });

// arrastrar y soltar otros .vrm
addEventListener('dragover', (e) => { e.preventDefault(); $('drop').style.display = 'flex'; });
addEventListener('dragleave', () => { $('drop').style.display = 'none'; });
addEventListener('drop', (e) => {
  e.preventDefault(); $('drop').style.display = 'none';
  const f = e.dataTransfer.files[0];
  if (f) loadVRM(URL.createObjectURL(f)).catch((err) => status('Error: ' + err.message));
});

// ------------------------------------------------------------------ pose
let seated = params.get('pose') !== 'stand';
const SEAT_OFFSET = new THREE.Vector3(0, 0.235, -0.04);
const POSE_SIT = {
  leftUpperLeg: [-1.65, 0.25, 0.32], rightUpperLeg: [-1.65, -0.25, -0.32],
  leftLowerLeg: [1.45, 0, 0], rightLowerLeg: [1.45, 0, 0],
  leftFoot: [0.1, 0, 0], rightFoot: [0.1, 0, 0],
  leftUpperArm: [0, 0.45, -0.85], rightUpperArm: [0, -0.45, 0.85],
  leftLowerArm: [0, -1.1, 0], rightLowerArm: [0, 1.1, 0],
  leftHand: [0, 0, -0.3], rightHand: [0, 0, 0.3],
};
const POSE_STAND = {
  leftUpperArm: [0, 0, -1.15], rightUpperArm: [0, 0, 1.15],
  leftLowerArm: [0, -0.25, 0], rightLowerArm: [0, 0.25, 0],
  leftUpperLeg: [0, 0, 0.03], rightUpperLeg: [0, 0, -0.03],
};
// Los VRM 0.x miran hacia -Z en su espacio normalizado: se invierten los ejes X y Z
// para que las mismas rotaciones sirvan para modelos VRM 0.x y 1.0.
function rot(name, x, y, z) {
  const node = vrm.humanoid.getNormalizedBoneNode(name);
  if (!node) return;
  const f = vrm.meta?.metaVersion === '0' ? -1 : 1;
  node.rotation.set(x * f, y, z * f);
}
function applyPose() {
  if (!vrm) return;
  for (const name of new Set([...Object.keys(POSE_SIT), ...Object.keys(POSE_STAND)])) {
    rot(name, 0, 0, 0);
  }
  const pose = seated ? POSE_SIT : POSE_STAND;
  for (const [name, r] of Object.entries(pose)) rot(name, ...r);
  vrm.scene.position.copy(seated ? SEAT_OFFSET : new THREE.Vector3());
  chair.visible = seated && $('btnChair').classList.contains('on');
  $('btnSit').classList.toggle('on', seated);
  $('btnStand').classList.toggle('on', !seated);
}
$('btnSit').onclick = () => { seated = true; applyPose(); };
$('btnStand').onclick = () => { seated = false; applyPose(); };
$('btnChair').onclick = (e) => { e.target.classList.toggle('on'); applyPose(); };
if (params.get('chair') === '0') $('btnChair').classList.remove('on');

// ------------------------------------------------------------------ expresiones
const EXPRS = [['😐 Normal', null], ['😊 Feliz', 'happy'], ['😌 Divertido', 'relaxed'],
               ['😠 Enojado', 'angry'], ['😢 Triste', 'sad'], ['😮 Sorpresa', 'Surprised']];
let currentExpr = params.get('expr');
function setExpr(name) {
  currentExpr = name;
  document.querySelectorAll('#exprs button').forEach((b) => b.classList.toggle('on', (b.dataset.e || null) === name));
}
EXPRS.forEach(([label, name]) => {
  const b = document.createElement('button');
  b.textContent = label; b.dataset.e = name ?? ''; b.onclick = () => setExpr(name);
  $('exprs').appendChild(b);
});
setExpr(currentExpr);

// ------------------------------------------------------------------ seguimiento facial (MediaPipe)
let landmarker = null, video = $('cam'), tracking = false, lastVideoTime = -1;
let mirror = true;
const face = { yaw: 0, pitch: 0, roll: 0, blinkL: 0, blinkR: 0, aa: 0, ih: 0, ou: 0, ee: 0, oh: 0,
               smile: 0, browUp: 0, browDown: 0, lookX: 0, lookY: 0, valid: false };
$('btnMirror').onclick = (e) => { mirror = !mirror; e.target.classList.toggle('on', mirror); };

$('btnCam').onclick = async () => {
  if (tracking) {
    tracking = false; video.srcObject?.getTracks().forEach((t) => t.stop());
    video.style.display = 'none'; $('btnCam').textContent = '📷 Activar cámara';
    $('btnCam').classList.remove('on'); face.valid = false; return;
  }
  try {
    status('Iniciando cámara…');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream; await video.play(); video.style.display = 'block';
    if (!landmarker) {
      status('Descargando detector facial…');
      const { FaceLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
        outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
        runningMode: 'VIDEO', numFaces: 1,
      });
    }
    tracking = true; $('btnCam').textContent = '⏹ Detener cámara'; $('btnCam').classList.add('on');
    status('Seguimiento activo ✔');
  } catch (e) {
    status('No se pudo usar la cámara: ' + e.message); console.error(e);
  }
};

const tmpM = new THREE.Matrix4(), tmpE = new THREE.Euler();
const clamp01 = (x) => Math.min(1, Math.max(0, x));
function trackFrame() {
  if (!tracking || !landmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = landmarker.detectForVideo(video, performance.now());
  if (!res.faceBlendshapes?.length) { face.valid = false; return; }
  const bs = {};
  for (const c of res.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
  if (res.facialTransformationMatrixes?.length) {
    tmpM.fromArray(res.facialTransformationMatrixes[0].data);
    tmpE.setFromRotationMatrix(tmpM, 'YXZ');
    const s = mirror ? -1 : 1;
    face.yaw = tmpE.y * s; face.pitch = tmpE.x; face.roll = tmpE.z * s;
  }
  const L = mirror ? 'Right' : 'Left', R = mirror ? 'Left' : 'Right';
  const blink = (v) => clamp01((v - 0.3) / 0.4);
  face.blinkL = blink(bs['eyeBlink' + L] ?? 0);   // ojo izquierdo del avatar
  face.blinkR = blink(bs['eyeBlink' + R] ?? 0);
  face.aa = clamp01((bs.jawOpen ?? 0) * 1.8 - 0.05);
  face.oh = clamp01((bs.mouthFunnel ?? 0) * 1.5);
  face.ou = clamp01((bs.mouthPucker ?? 0) * 1.2 - 0.2);
  face.smile = clamp01(((bs.mouthSmileLeft ?? 0) + (bs.mouthSmileRight ?? 0)) * 0.8);
  face.ee = clamp01(((bs.mouthStretchLeft ?? 0) + (bs.mouthStretchRight ?? 0)) * 1.2);
  face.ih = clamp01(face.smile * face.aa * 2);
  face.browUp = clamp01((bs.browInnerUp ?? 0) * 1.6 - 0.3);
  face.browDown = clamp01(((bs.browDownLeft ?? 0) + (bs.browDownRight ?? 0)) * 1.2 - 0.2);
  const lx = ((bs.eyeLookOutLeft ?? 0) - (bs.eyeLookInLeft ?? 0) + (bs.eyeLookInRight ?? 0) - (bs.eyeLookOutRight ?? 0)) / 2;
  const ly = (((bs.eyeLookUpLeft ?? 0) + (bs.eyeLookUpRight ?? 0)) - ((bs.eyeLookDownLeft ?? 0) + (bs.eyeLookDownRight ?? 0))) / 2;
  face.lookX = lx * (mirror ? -1 : 1); face.lookY = ly;
  face.valid = true;
}

// ------------------------------------------------------------------ micrófono (lipsync por volumen)
let analyser = null, micBuf = null, micLevel = 0;
$('btnMic').onclick = async (e) => {
  if (analyser) { analyser = null; e.target.classList.remove('on'); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext();
    analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    micBuf = new Float32Array(analyser.fftSize);
    e.target.classList.add('on');
  } catch (err) { status('No se pudo usar el micrófono: ' + err.message); }
};
function micFrame() {
  if (!analyser) { micLevel = 0; return; }
  analyser.getFloatTimeDomainData(micBuf);
  let s = 0; for (const v of micBuf) s += v * v;
  const rms = Math.sqrt(s / micBuf.length);
  micLevel += (clamp01((rms - 0.01) * 12) - micLevel) * 0.5;
}

// ------------------------------------------------------------------ animación
const clock = new THREE.Clock();
const sm = { yaw: 0, pitch: 0, roll: 0, values: {} };
let nextBlink = 2, blinkT = -1;
const lerp = (a, b, t) => a + (b - a) * t;

function setSmooth(name, target, k) {
  const v = lerp(sm.values[name] ?? 0, target, k);
  sm.values[name] = v;
  vrm.expressionManager?.setValue(name, v);
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;
  controls.update();
  trackFrame();
  micFrame();
  if (vrm) {
    const k = 1 - Math.pow(0.001, dt);  // suavizado independiente de los FPS
    const live = face.valid;
    // cabeza
    const idleYaw = Math.sin(t * 0.5) * 0.06, idlePitch = Math.sin(t * 0.8) * 0.03;
    sm.yaw = lerp(sm.yaw, live ? face.yaw : idleYaw, k);
    sm.pitch = lerp(sm.pitch, live ? face.pitch : idlePitch, k);
    sm.roll = lerp(sm.roll, live ? face.roll : Math.sin(t * 0.35) * 0.04, k);
    rot('neck', sm.pitch * 0.4, sm.yaw * 0.4, sm.roll * 0.4);
    rot('head', sm.pitch * 0.6, sm.yaw * 0.6, sm.roll * 0.6);
    rot('spine', 0, sm.yaw * 0.12, sm.roll * 0.1);
    rot('chest', Math.sin(t * 1.6) * 0.015, 0, 0);   // respiración
    // ojos
    if (live) {
      vrm.lookAt.target = null;
      vrm.lookAt.yaw = lerp(vrm.lookAt.yaw, face.lookX * 25, k);
      vrm.lookAt.pitch = lerp(vrm.lookAt.pitch, face.lookY * 20, k);
    } else if (params.has('nolook')) {
      vrm.lookAt.target = null; vrm.lookAt.yaw = 0; vrm.lookAt.pitch = 0;
    } else if (vrm.lookAt.target !== camera) {
      vrm.lookAt.target = camera;
    }
    // parpadeo
    let bl, br;
    if (live) { bl = face.blinkL; br = face.blinkR; } else {
      if (t > nextBlink) { blinkT = t; nextBlink = t + 2.5 + Math.random() * 3; }
      const p = blinkT < 0 ? 1 : (t - blinkT) / 0.16;
      const v = p < 1 ? Math.sin(p * Math.PI) : 0;
      bl = br = v;
    }
    const manual = currentExpr;
    const eyesBusy = manual === 'happy' || manual === 'relaxed';
    setSmooth('blinkLeft', eyesBusy ? 0 : bl, 0.6);
    setSmooth('blinkRight', eyesBusy ? 0 : br, 0.6);
    // boca
    const voice = Math.max(micLevel, 0);
    setSmooth('aa', live ? Math.max(face.aa, voice) : voice, 0.5);
    setSmooth('ih', live ? face.ih : 0, 0.5);
    setSmooth('ou', live ? face.ou : 0, 0.5);
    setSmooth('ee', live ? face.ee * 0.6 : 0, 0.5);
    setSmooth('oh', live ? face.oh : 0, 0.5);
    // expresiones (manual o detectada)
    const auto = live ? { relaxed: face.smile > 0.45 ? (face.smile - 0.45) * 2 : 0,
                          Surprised: face.browUp, angry: face.browDown } : {};
    for (const e of ['happy', 'relaxed', 'angry', 'sad', 'Surprised']) {
      setSmooth(e, manual === e ? 1 : clamp01(auto[e] ?? 0), 0.25);
    }
    vrm.update(dt);
  }
  renderer.render(scene, camera);
}
animate();

// ------------------------------------------------------------------ capturas (para documentación)
if (params.has('shot')) {
  const view = params.get('view') ?? 'front';
  const angles = { front: 0, left: 0.75, right: -0.75, side: 1.57, back: Math.PI, three: 0.5 };
  const a = angles[view] ?? parseFloat(view);
  const dist = parseFloat(params.get('dist') ?? (seated ? 2.9 : 2.6));
  const ty = parseFloat(params.get('ty') ?? (seated ? 0.8 : 0.62));
  camera.position.set(Math.sin(a) * dist, ty + 0.1, Math.cos(a) * dist);
  controls.target.set(0, ty, 0);
  if (params.has('fov')) { camera.fov = parseFloat(params.get('fov')); camera.updateProjectionMatrix(); }
  controls.update();
}
