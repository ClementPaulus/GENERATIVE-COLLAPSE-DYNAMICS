/**
 * WebGL Immersive Space Simulator — ESA-style Black Hole Environment
 *
 * Full-screen 3D space simulation with:
 *   - Schwarzschild-like black hole (event horizon shadow)
 *   - Gravitational lensing distortion of background starfield
 *   - Accretion disk with Doppler beaming
 *   - Photon ring glow
 *   - Orbital camera with mouse look + scroll zoom
 *   - Real-time HUD with GCD kernel readouts (Γ, ω, regime, redshift, v_esc)
 *
 * All physics derived from GCD kernel — Tier-0 Protocol.
 * No Tier-1 symbol is redefined.
 */

import { gammaOmega, computeKernel, classifyRegime } from './kernel';
import {
  dGamma, hawkingTemperature, gravitationalRedshift,
  escapeVelocity, OMEGA_ISCO, OMEGA_PHOTON_SPHERE,
  BLACK_HOLE_ENTITIES,
} from './spacetime';
import { EPSILON, P_EXPONENT } from './constants';

/* ═══════════════════════════════════════════════════════════════════
   §1  LINEAR ALGEBRA (column-major 4×4)
   ═══════════════════════════════════════════════════════════════════ */

type Mat4 = Float32Array;
type Vec3 = [number, number, number];

function mat4(): Mat4 { return new Float32Array(16); }

function identity(): Mat4 {
  const m = mat4();
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const m = mat4();
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mul(a: Mat4, b: Mat4): Mat4 {
  const o = mat4();
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      o[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] +
                      a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
  return o;
}

function rotX(a: number): Mat4 {
  const m = identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}

function rotY(a: number): Mat4 {
  const m = identity();
  const c = Math.cos(a), s = Math.sin(a);
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}

function translate(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

function normalize(v: Vec3): Vec3 {
  const l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = identity();
  m[0] = x[0]; m[4] = x[1]; m[8]  = x[2]; m[12] = -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);
  m[1] = y[0]; m[5] = y[1]; m[9]  = y[2]; m[13] = -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);
  m[2] = z[0]; m[6] = z[1]; m[10] = z[2]; m[14] = -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);
  m[3] = 0;    m[7] = 0;    m[11] = 0;    m[15] = 1;
  return m;
}

/* ═══════════════════════════════════════════════════════════════════
   §2  SHADERS
   ═══════════════════════════════════════════════════════════════════ */

// ── Full-screen quad for space background + gravitational lensing ──
const BG_VERT = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const BG_FRAG = `
precision highp float;
varying vec2 vUV;
uniform vec2 uBHScreen;      // black hole center in screen [0,1]
uniform float uBHRadius;     // angular radius of event horizon
uniform float uLensStrength;  // gravitational lensing magnitude
uniform float uTime;

// Pseudo-random hash for star generation
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// Procedural starfield
vec3 starfield(vec2 uv) {
  vec3 col = vec3(0.0);
  // Multiple layers at different scales
  for (int layer = 0; layer < 3; layer++) {
    float scale = 80.0 + float(layer) * 120.0;
    vec2 id = floor(uv * scale);
    vec2 f = fract(uv * scale) - 0.5;
    float h = hash(id + float(layer) * 100.0);
    if (h > 0.97) {
      float brightness = (h - 0.97) / 0.03;
      float r = length(f);
      float star = smoothstep(0.15, 0.0, r) * brightness;
      // Star color temperature variation
      float temp = hash(id * 2.0 + 7.0);
      vec3 starCol = mix(
        mix(vec3(0.6, 0.7, 1.0), vec3(1.0, 1.0, 0.95), temp),
        vec3(1.0, 0.8, 0.5),
        max(0.0, temp - 0.7) * 3.33
      );
      // Twinkling
      float twinkle = 0.7 + 0.3 * sin(uTime * (1.0 + h * 3.0) + h * 100.0);
      col += starCol * star * twinkle;
    }
  }
  // Subtle nebula glow
  float nebula = hash(floor(uv * 8.0)) * 0.015;
  col += vec3(0.15, 0.05, 0.2) * nebula;
  return col;
}

void main() {
  vec2 uv = vUV;
  vec2 centered = uv - uBHScreen;
  float dist = length(centered);
  vec2 dir = centered / max(dist, 0.001);

  // ── Gravitational lensing distortion ──
  // Einstein ring radius: R_E ∝ √(lensStrength)
  float rEinstein = sqrt(uLensStrength) * 0.15;
  // Deflection angle: θ = R_E² / dist (point mass approximation)
  float deflection = rEinstein * rEinstein / max(dist, 0.001);
  // Limit max deflection to avoid artifacts
  deflection = min(deflection, 0.5);
  // Apply radial deflection outward (light bends toward mass)
  vec2 lensedUV = uv + dir * deflection;

  // ── Event horizon shadow ──
  float horizonR = uBHRadius;
  // Photon sphere at 1.5× horizon
  float photonR = horizonR * 1.5;
  // Shadow is larger than horizon due to photon capture
  float shadowR = horizonR * 2.6;

  // ── Render starfield with lensing ──
  vec3 col = starfield(lensedUV);

  // ── Photon ring (bright ring at photon sphere) ──
  float ringDist = abs(dist - photonR);
  float photonRing = exp(-ringDist * ringDist / (0.0004 * photonR * photonR)) * 1.2;
  // Secondary ring (inner)
  float ring2Dist = abs(dist - photonR * 0.75);
  float ring2 = exp(-ring2Dist * ring2Dist / (0.0002 * photonR * photonR)) * 0.5;
  vec3 ringColor = vec3(1.0, 0.85, 0.4);
  col += ringColor * (photonRing + ring2);

  // ── Einstein ring glow ──
  float eRingDist = abs(dist - rEinstein);
  float eRing = exp(-eRingDist * eRingDist / (0.0008 * rEinstein * rEinstein)) * 0.4;
  col += vec3(0.5, 0.7, 1.0) * eRing;

  // ── Event horizon shadow (black disk) ──
  float shadowEdge = smoothstep(shadowR, shadowR * 0.85, dist);
  col *= (1.0 - shadowEdge);

  // ── Horizon edge glow (Hawking radiation analog) ──
  float edgeGlow = exp(-(dist - shadowR) * (dist - shadowR) / (0.001 * shadowR * shadowR));
  col += vec3(0.8, 0.3, 0.1) * edgeGlow * 0.3;

  gl_FragColor = vec4(col, 1.0);
}
`;

// ── Accretion disk shader ──
const DISK_VERT = `
attribute vec3 aPosition;
attribute vec2 aTexCoord;
uniform mat4 uMVP;
varying vec2 vTexCoord;
varying vec3 vWorldPos;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vTexCoord = aTexCoord;
  vWorldPos = aPosition;
}
`;

const DISK_FRAG = `
precision highp float;
varying vec2 vTexCoord;
varying vec3 vWorldPos;
uniform float uTime;
uniform float uInnerR;
uniform float uOuterR;

float noise(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  float r = length(vWorldPos.xz);
  float angle = atan(vWorldPos.z, vWorldPos.x);

  // Normalized radial position in disk
  float t = (r - uInnerR) / (uOuterR - uInnerR);
  t = clamp(t, 0.0, 1.0);

  // Temperature profile: hotter at inner edge
  float temp = 1.0 - t * 0.7;

  // Keplerian velocity for Doppler beaming
  float orbitalV = 1.0 / sqrt(max(r, 0.1));

  // Turbulent spiral structure
  float spiral = sin(angle * 4.0 - log(max(r, 0.01)) * 8.0 + uTime * orbitalV * 2.0)
               * 0.5 + 0.5;
  spiral = mix(0.6, 1.0, spiral);

  // Fine structure turbulence
  float turb = noise(vec2(angle * 20.0 + uTime * 0.5, r * 15.0)) * 0.2 + 0.8;

  // Color: hot inner = blue-white, outer = red-orange
  vec3 innerColor = vec3(0.9, 0.95, 1.0);    // blue-white
  vec3 midColor = vec3(1.0, 0.7, 0.2);       // yellow-orange
  vec3 outerColor = vec3(0.8, 0.15, 0.05);   // deep red

  vec3 diskColor = mix(innerColor, midColor, smoothstep(0.0, 0.4, t));
  diskColor = mix(diskColor, outerColor, smoothstep(0.4, 1.0, t));

  // Doppler beaming: approaching side brighter
  float doppler = 1.0 + 0.4 * sin(angle + uTime * 0.3) * orbitalV;

  float brightness = temp * spiral * turb * doppler;

  // Edge softness
  float innerEdge = smoothstep(0.0, 0.05, t);
  float outerEdge = smoothstep(1.0, 0.9, t);
  brightness *= innerEdge * outerEdge;

  // Opacity falloff
  float alpha = brightness * 0.85;
  alpha *= innerEdge * outerEdge;

  gl_FragColor = vec4(diskColor * brightness, alpha);
}
`;

// ── Particle shader (jets + infalling matter) ──
const PART_VERT = `
attribute vec3 aPosition;
attribute vec3 aColor;
attribute float aSize;
uniform mat4 uMVP;
varying vec3 vColor;

void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vColor = aColor;
  gl_PointSize = aSize;
}
`;

const PART_FRAG = `
precision mediump float;
varying vec3 vColor;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float glow = exp(-r * r * 8.0);
  gl_FragColor = vec4(vColor * glow, glow * 0.8);
}
`;

/* ═══════════════════════════════════════════════════════════════════
   §3  GEOMETRY GENERATORS
   ═══════════════════════════════════════════════════════════════════ */

// Accretion disk: flat annulus in the XZ plane
function generateDisk(innerR: number, outerR: number, segments: number, rings: number): {
  verts: Float32Array; indices: Uint16Array;
} {
  const count = (rings + 1) * segments;
  const verts = new Float32Array(count * 5); // xyz + uv
  let vi = 0;
  for (let ri = 0; ri <= rings; ri++) {
    const t = ri / rings;
    const r = innerR + t * (outerR - innerR);
    for (let si = 0; si < segments; si++) {
      const a = (si / segments) * Math.PI * 2;
      verts[vi++] = r * Math.cos(a);
      verts[vi++] = 0; // flat in XZ
      verts[vi++] = r * Math.sin(a);
      verts[vi++] = t;           // u = radial
      verts[vi++] = si / segments; // v = angular
    }
  }
  const idxCount = rings * segments * 6;
  const indices = new Uint16Array(idxCount);
  let ii = 0;
  for (let ri = 0; ri < rings; ri++) {
    for (let si = 0; si < segments; si++) {
      const c = ri * segments + si;
      const n = ri * segments + (si + 1) % segments;
      const a = (ri + 1) * segments + si;
      const an = (ri + 1) * segments + (si + 1) % segments;
      indices[ii++] = c; indices[ii++] = a; indices[ii++] = n;
      indices[ii++] = n; indices[ii++] = a; indices[ii++] = an;
    }
  }
  return { verts, indices };
}

/* ═══════════════════════════════════════════════════════════════════
   §4  PARTICLE SYSTEMS
   ═══════════════════════════════════════════════════════════════════ */

interface SpaceParticle {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  life: number;
  maxLife: number;
  size: number;
  type: 'accretion' | 'jet' | 'infalling';
  cr: number; cg: number; cb: number;
}

function spawnParticle(type: SpaceParticle['type'], innerR: number, outerR: number): SpaceParticle {
  if (type === 'accretion') {
    const r = innerR + Math.random() * (outerR - innerR);
    const a = Math.random() * Math.PI * 2;
    const orbV = 1.0 / Math.sqrt(r);
    return {
      x: r * Math.cos(a), y: (Math.random() - 0.5) * 0.05, z: r * Math.sin(a),
      vx: -orbV * Math.sin(a) * 0.3, vy: 0, vz: orbV * Math.cos(a) * 0.3,
      life: 0, maxLife: 3 + Math.random() * 5,
      size: 2 + Math.random() * 3,
      type, cr: 1, cg: 0.7 + Math.random() * 0.3, cb: 0.2 + Math.random() * 0.3,
    };
  } else if (type === 'jet') {
    const side = Math.random() > 0.5 ? 1 : -1;
    const spread = 0.1;
    return {
      x: (Math.random() - 0.5) * spread, y: side * innerR * 0.3,
      z: (Math.random() - 0.5) * spread,
      vx: (Math.random() - 0.5) * 0.05, vy: side * (1.5 + Math.random()),
      vz: (Math.random() - 0.5) * 0.05,
      life: 0, maxLife: 1.5 + Math.random() * 2,
      size: 1.5 + Math.random() * 2,
      type, cr: 0.4 + Math.random() * 0.3, cg: 0.5 + Math.random() * 0.4, cb: 1.0,
    };
  } else {
    // Infalling
    const a = Math.random() * Math.PI * 2;
    const r = outerR * (1.2 + Math.random() * 0.5);
    return {
      x: r * Math.cos(a), y: (Math.random() - 0.5) * 0.3, z: r * Math.sin(a),
      vx: -Math.cos(a) * 0.2, vy: 0, vz: -Math.sin(a) * 0.2,
      life: 0, maxLife: 4 + Math.random() * 4,
      size: 1.5 + Math.random() * 2.5,
      type, cr: 0.9, cg: 0.5, cb: 0.2,
    };
  }
}

/* ═══════════════════════════════════════════════════════════════════
   §5  HUD PANEL
   ═══════════════════════════════════════════════════════════════════ */

export interface HUDData {
  omega: number;
  gamma: number;
  regime: string;
  redshift: number;
  escapeV: number;
  hawkingT: number;
  distance: number;
  F: number;
  IC: number;
  kappa: number;
  S: number;
  C: number;
  delta: number;
}

function computeHUD(cameraDistance: number, bhOmega: number): HUDData {
  // Map camera distance to an ω for "observer position"
  // Closer → higher ω
  const maxDist = 20;
  const minOmega = 0.01;
  const observerOmega = Math.min(0.98, minOmega + (1 - Math.min(cameraDistance / maxDist, 1)) * (bhOmega - minOmega));

  const gamma = gammaOmega(observerOmega);
  const regime = classifyRegime(
    computeKernel(
      BLACK_HOLE_ENTITIES[0].c,
      BLACK_HOLE_ENTITIES[0].w,
    ),
  );
  const kr = computeKernel(BLACK_HOLE_ENTITIES[0].c, BLACK_HOLE_ENTITIES[0].w);

  return {
    omega: observerOmega,
    gamma,
    regime: regime.regime + (regime.isCritical ? ' (Critical)' : ''),
    redshift: gravitationalRedshift(observerOmega),
    escapeV: escapeVelocity(observerOmega),
    hawkingT: hawkingTemperature(kr.kappa),
    distance: cameraDistance,
    F: kr.F,
    IC: kr.IC,
    kappa: kr.kappa,
    S: kr.S,
    C: kr.C,
    delta: kr.delta,
  };
}

/* ═══════════════════════════════════════════════════════════════════
   §6  MAIN INIT
   ═══════════════════════════════════════════════════════════════════ */

export interface SpaceSimControls {
  destroy: () => void;
  getHUD: () => HUDData;
}

export function initSpaceSim(
  canvas: HTMLCanvasElement,
  hudCallback?: (data: HUDData) => void,
): SpaceSimControls {
  const glCtx = canvas.getContext('webgl', {
    antialias: true, alpha: false, premultipliedAlpha: false,
  });
  if (!glCtx) {
    console.error('WebGL not available');
    return { destroy: () => {}, getHUD: () => computeHUD(10, 0.95) };
  }
  const gl: WebGLRenderingContext = glCtx;

  // ── Extensions ──
  gl.getExtension('OES_standard_derivatives');

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);

  // ── Compile helpers ──
  function compileShader(type: number, src: string): WebGLShader {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      console.error('Shader:', gl.getShaderInfoLog(s));
    return s;
  }
  function linkProgram(vs: string, fs: string): WebGLProgram {
    const p = gl.createProgram()!;
    gl.attachShader(p, compileShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compileShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      console.error('Link:', gl.getProgramInfoLog(p));
    return p;
  }

  // ── Programs ──
  const bgProg = linkProgram(BG_VERT, BG_FRAG);
  const diskProg = linkProgram(DISK_VERT, DISK_FRAG);
  const partProg = linkProgram(PART_VERT, PART_FRAG);

  // ── Background quad ──
  const quadVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  // ── Accretion disk geometry ──
  const DISK_INNER = 0.6;
  const DISK_OUTER = 3.0;
  const disk = generateDisk(DISK_INNER, DISK_OUTER, 128, 32);

  const diskVBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, diskVBO);
  gl.bufferData(gl.ARRAY_BUFFER, disk.verts, gl.STATIC_DRAW);

  const diskIBO = gl.createBuffer()!;
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, diskIBO);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, disk.indices, gl.STATIC_DRAW);

  // ── Particles ──
  const MAX_PARTICLES = 400;
  const particles: SpaceParticle[] = [];
  // Seed initial particles
  for (let i = 0; i < 200; i++) {
    const type: SpaceParticle['type'] = i < 120 ? 'accretion' : i < 160 ? 'jet' : 'infalling';
    const p = spawnParticle(type, DISK_INNER, DISK_OUTER);
    p.life = Math.random() * p.maxLife; // stagger
    particles.push(p);
  }
  const particleVBO = gl.createBuffer()!;

  // ── Camera ──
  let azimuth = 0;
  let elevation = 0.35; // slightly above disk plane
  let camDist = 8.0;
  let dragging = false;
  let lastX = 0, lastY = 0;
  let autoOrbit = true;
  let targetAzimuth = 0;
  let targetElevation = 0.35;
  let targetDist = 8.0;

  // Mouse/touch controls
  canvas.addEventListener('mousedown', (e) => {
    dragging = true; autoOrbit = false;
    lastX = e.clientX; lastY = e.clientY;
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    targetAzimuth += (e.clientX - lastX) * 0.005;
    targetElevation += (e.clientY - lastY) * 0.005;
    targetElevation = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, targetElevation));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { dragging = false; canvas.style.cursor = 'grab'; });
  canvas.addEventListener('mouseleave', () => { dragging = false; canvas.style.cursor = 'grab'; });

  canvas.addEventListener('touchstart', (e) => {
    dragging = true; autoOrbit = false;
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    if (!dragging) return;
    targetAzimuth += (e.touches[0].clientX - lastX) * 0.005;
    targetElevation += (e.touches[0].clientY - lastY) * 0.005;
    targetElevation = Math.max(-Math.PI * 0.45, Math.min(Math.PI * 0.45, targetElevation));
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
    e.preventDefault();
  }, { passive: false });
  canvas.addEventListener('touchend', () => { dragging = false; });

  canvas.addEventListener('wheel', (e) => {
    targetDist += e.deltaY * 0.01;
    targetDist = Math.max(2.0, Math.min(25, targetDist));
    e.preventDefault();
  }, { passive: false });

  canvas.addEventListener('dblclick', () => {
    autoOrbit = true;
    targetElevation = 0.35;
    targetDist = 8.0;
  });

  canvas.style.cursor = 'grab';

  // ── BH params ──
  const bhOmega = 0.95; // deep collapse — near horizon

  // ── Uniform locations ──
  // Background
  const bgLocs = {
    aPos: gl.getAttribLocation(bgProg, 'aPos'),
    uBHScreen: gl.getUniformLocation(bgProg, 'uBHScreen'),
    uBHRadius: gl.getUniformLocation(bgProg, 'uBHRadius'),
    uLensStrength: gl.getUniformLocation(bgProg, 'uLensStrength'),
    uTime: gl.getUniformLocation(bgProg, 'uTime'),
  };
  // Disk
  const diskLocs = {
    aPosition: gl.getAttribLocation(diskProg, 'aPosition'),
    aTexCoord: gl.getAttribLocation(diskProg, 'aTexCoord'),
    uMVP: gl.getUniformLocation(diskProg, 'uMVP'),
    uTime: gl.getUniformLocation(diskProg, 'uTime'),
    uInnerR: gl.getUniformLocation(diskProg, 'uInnerR'),
    uOuterR: gl.getUniformLocation(diskProg, 'uOuterR'),
  };
  // Particles
  const partLocs = {
    aPosition: gl.getAttribLocation(partProg, 'aPosition'),
    aColor: gl.getAttribLocation(partProg, 'aColor'),
    aSize: gl.getAttribLocation(partProg, 'aSize'),
    uMVP: gl.getUniformLocation(partProg, 'uMVP'),
  };

  // ── Resize handler ──
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvas);

  // ── HUD state ──
  let hudData = computeHUD(camDist, bhOmega);

  // ── Animation ──
  let running = true;
  let lastTime = 0;

  function frame(now: number) {
    if (!running) return;
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    // Smooth camera interpolation
    if (autoOrbit) targetAzimuth += dt * 0.08;
    azimuth += (targetAzimuth - azimuth) * 0.05;
    elevation += (targetElevation - elevation) * 0.05;
    camDist += (targetDist - camDist) * 0.05;

    const cx = camDist * Math.cos(elevation) * Math.sin(azimuth);
    const cy = camDist * Math.sin(elevation);
    const cz = camDist * Math.cos(elevation) * Math.cos(azimuth);

    const aspect = canvas.width / canvas.height;
    const proj = perspective(0.9, aspect, 0.1, 100);
    const view = lookAt([cx, cy, cz], [0, 0, 0], [0, 1, 0]);
    const vp = mul(proj, view);

    // ── Project BH center to screen ──
    // BH is at origin; transform [0,0,0,1] through VP
    const clipX = vp[12] / vp[15];
    const clipY = vp[13] / vp[15];
    const screenBH: [number, number] = [clipX * 0.5 + 0.5, clipY * 0.5 + 0.5];

    // Apparent angular size of BH based on distance
    const bhAngularR = Math.atan(0.5 / camDist) / (0.9 / 2); // normalized to FOV
    // Lens strength varies with mass (|κ|) and distance
    const lensStrength = gammaOmega(bhOmega) / (camDist * 0.5);

    const t = now / 1000;

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // ═══ Pass 1: Background + Lensing ═══
    gl.depthMask(false);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(bgProg);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
    gl.enableVertexAttribArray(bgLocs.aPos);
    gl.vertexAttribPointer(bgLocs.aPos, 2, gl.FLOAT, false, 0, 0);

    gl.uniform2fv(bgLocs.uBHScreen, screenBH);
    gl.uniform1f(bgLocs.uBHRadius, bhAngularR);
    gl.uniform1f(bgLocs.uLensStrength, lensStrength);
    gl.uniform1f(bgLocs.uTime, t);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(bgLocs.aPos);

    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);

    // ═══ Pass 2: Accretion Disk ═══
    // Tilt disk slightly for visual interest
    const diskTilt = rotX(0.12);
    const diskMVP = mul(vp, diskTilt);

    gl.useProgram(diskProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, diskVBO);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, diskIBO);

    const stride = 5 * 4;
    gl.enableVertexAttribArray(diskLocs.aPosition);
    gl.vertexAttribPointer(diskLocs.aPosition, 3, gl.FLOAT, false, stride, 0);
    if (diskLocs.aTexCoord >= 0) {
      gl.enableVertexAttribArray(diskLocs.aTexCoord);
      gl.vertexAttribPointer(diskLocs.aTexCoord, 2, gl.FLOAT, false, stride, 12);
    }

    gl.uniformMatrix4fv(diskLocs.uMVP, false, diskMVP);
    gl.uniform1f(diskLocs.uTime, t);
    gl.uniform1f(diskLocs.uInnerR, DISK_INNER);
    gl.uniform1f(diskLocs.uOuterR, DISK_OUTER);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawElements(gl.TRIANGLES, disk.indices.length, gl.UNSIGNED_SHORT, 0);

    gl.disableVertexAttribArray(diskLocs.aPosition);
    if (diskLocs.aTexCoord >= 0) gl.disableVertexAttribArray(diskLocs.aTexCoord);

    // ═══ Pass 3: Particles ═══
    // Update particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life += dt;
      if (p.life > p.maxLife) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
        continue;
      }

      if (p.type === 'accretion') {
        // Orbit + slow infall
        const r = Math.sqrt(p.x * p.x + p.z * p.z) || 0.1;
        const orbSpeed = 1.0 / (r * Math.sqrt(r)) * 0.5;
        const ax = -p.z / r, az = p.x / r;
        p.vx = ax * orbSpeed - p.x / r * 0.01;
        p.vz = az * orbSpeed - p.z / r * 0.01;
      } else if (p.type === 'jet') {
        // Accelerate along y axis
        p.vy *= 1.01;
      } else {
        // Infalling: accelerate toward center
        const r = Math.sqrt(p.x * p.x + p.z * p.z) || 0.1;
        p.vx -= p.x / r * 0.05 * dt;
        p.vz -= p.z / r * 0.05 * dt;
      }

      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;

      // Respawn if too far or absorbed
      const dist = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (dist < 0.2 || dist > 15) {
        particles[i] = spawnParticle(p.type, DISK_INNER, DISK_OUTER);
      }
    }

    // Spawn new if needed
    while (particles.length < MAX_PARTICLES) {
      const roll = Math.random();
      const type: SpaceParticle['type'] = roll < 0.5 ? 'accretion' : roll < 0.75 ? 'jet' : 'infalling';
      particles.push(spawnParticle(type, DISK_INNER, DISK_OUTER));
    }

    // Upload particle data
    const pData = new Float32Array(particles.length * 7); // xyz rgb size
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      const fade = Math.min(1, Math.min(p.life / 0.3, (p.maxLife - p.life) / 0.5));
      const b = i * 7;
      pData[b] = p.x; pData[b + 1] = p.y; pData[b + 2] = p.z;
      pData[b + 3] = p.cr * fade; pData[b + 4] = p.cg * fade; pData[b + 5] = p.cb * fade;
      pData[b + 6] = p.size * (0.5 + fade * 0.5);
    }

    gl.useProgram(partProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleVBO);
    gl.bufferData(gl.ARRAY_BUFFER, pData, gl.DYNAMIC_DRAW);

    const pStride = 7 * 4;
    gl.enableVertexAttribArray(partLocs.aPosition);
    gl.vertexAttribPointer(partLocs.aPosition, 3, gl.FLOAT, false, pStride, 0);
    gl.enableVertexAttribArray(partLocs.aColor);
    gl.vertexAttribPointer(partLocs.aColor, 3, gl.FLOAT, false, pStride, 12);
    gl.enableVertexAttribArray(partLocs.aSize);
    gl.vertexAttribPointer(partLocs.aSize, 1, gl.FLOAT, false, pStride, 24);

    gl.uniformMatrix4fv(partLocs.uMVP, false, vp);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, particles.length);

    gl.disableVertexAttribArray(partLocs.aPosition);
    gl.disableVertexAttribArray(partLocs.aColor);
    gl.disableVertexAttribArray(partLocs.aSize);

    // Reset blend mode
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // ── Update HUD ──
    hudData = computeHUD(camDist, bhOmega);
    if (hudCallback) hudCallback(hudData);

    requestAnimationFrame(frame);
  }

  requestAnimationFrame((t) => { lastTime = t; frame(t); });

  return {
    destroy: () => {
      running = false;
      resizeObs.disconnect();
    },
    getHUD: () => hudData,
  };
}
