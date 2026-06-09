// surface3d.js - reusable Three.js engine that turns a strike x expiry x value
// matrix (an option ladder) into an interactive 3D mesh. No app logic lives
// here; the controller in app.js feeds it data and reads hover events back.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// ---- colormaps -------------------------------------------------------------
// Same ramps the old 2D heatmap used, so colors stay familiar.
export function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [[68, 1, 84], [59, 82, 139], [33, 144, 140], [93, 201, 99], [253, 231, 37]];
  const f = t * (stops.length - 1), i = Math.floor(f), r = f - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  return [a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r, a[2] + (b[2] - a[2]) * r];
}
// Diverging blue-white-red. t in [-1, 1]; 0 is white (no difference).
export function diverging(t) {
  t = Math.max(-1, Math.min(1, t));
  if (t < 0) { const u = 1 + t; return [54 + 201 * u, 120 + 135 * u, 255]; }
  const u = 1 - t; return [255, 107 + 148 * u, 129 + 126 * u];
}

// Visual extents of the plotted block (world units). Data is normalized into it.
const SIZE_X = 12;   // strike axis
const SIZE_Z = 9;    // expiry axis
const HEIGHT = 5;    // value axis (peak height for sequential, half-range for diverging)

// Build a canvas-textured sprite for an axis tick / title.
function makeLabel(text, { size = 13, color = '#8a93a8', weight = 400 } = {}) {
  const pad = 6, scale = 4; // supersample for crisp text
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = `${weight} ${size * scale}px ui-monospace, Menlo, Consolas, monospace`;
  const w = ctx.measureText(text).width;
  c.width = Math.ceil(w + pad * scale * 2);
  c.height = Math.ceil((size + pad) * scale);
  ctx.font = `${weight} ${size * scale}px ui-monospace, Menlo, Consolas, monospace`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, pad * scale, c.height / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sp = new THREE.Sprite(mat);
  sp.scale.set((c.width / c.height) * 0.7, 0.7, 1);
  sp.renderOrder = 10;
  return sp;
}

export class Surface3D {
  constructor(container, overlayEl) {
    this.container = container;
    this.overlay = overlayEl;
    this.data = null;
    this.opts = { wireframe: false, mode: 'sequential', heightScale: 1 };

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07080c');
    scene.fog = new THREE.Fog('#07080c', 26, 46);
    this.scene = scene;

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
    camera.position.set(13, 11, 15);
    this.camera = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    this.renderer = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 6;
    controls.maxDistance = 40;
    controls.maxPolarAngle = Math.PI * 0.495; // don't dive under the floor
    controls.target.set(0, HEIGHT * 0.35, 0);
    this.controls = controls;

    // Lighting tuned for a dark UI: cool key, warm fill, soft ambient.
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xbcd0ff, 1.05);
    key.position.set(8, 16, 10);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xff9f6b, 0.35);
    fill.position.set(-10, 6, -8);
    scene.add(fill);

    // Persistent scene furniture (floor grid). Data-driven groups get rebuilt.
    const grid = new THREE.GridHelper(SIZE_X * 1.6, 16, 0x1e2433, 0x141826);
    grid.position.y = -0.01;
    scene.add(grid);

    this.meshGroup = new THREE.Group();
    this.labelGroup = new THREE.Group();
    this.markerGroup = new THREE.Group();
    scene.add(this.meshGroup, this.labelGroup, this.markerGroup);

    // Hover picking.
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._hoverActive = false;
    renderer.domElement.addEventListener('pointermove', (e) => this._onPointerMove(e));
    renderer.domElement.addEventListener('pointerleave', () => {
      this._hoverActive = false;
      if (this.overlay) this.overlay.style.display = 'none';
    });

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);

    this._animate = this._animate.bind(this);
    this._raf = requestAnimationFrame(this._animate);
  }

  // --- coordinate helpers ---------------------------------------------------
  _xForCol(j, ns) { return ns <= 1 ? 0 : (j / (ns - 1) - 0.5) * SIZE_X; }
  _zForRow(i, ne) { return ne <= 1 ? 0 : (i / (ne - 1) - 0.5) * SIZE_Z; }

  // Map a raw value to world height + normalized [0..1] color stop, honoring mode.
  _yc(v) {
    const d = this.data, hs = this.opts.heightScale;
    if (this.opts.mode === 'diverging') {
      const m = d.maxAbs || 1;
      const tn = v / m;                      // -1..1
      return { y: tn * (HEIGHT / 2) * hs, color: diverging(tn) };
    }
    const span = (d.hi - d.lo) || 1;
    const tn = (v - d.lo) / span;            // 0..1
    return { y: tn * HEIGHT * hs, color: viridis(tn) };
  }

  // --- public API -----------------------------------------------------------
  // data: { matrix:[[]], strikes:[], expiries:[{days,date}|num], valueLabel, mode, S }
  setData(data) {
    const matrix = data.matrix;
    const ne = matrix.length, ns = matrix[0].length;
    let lo = Infinity, hi = -Infinity, maxAbs = 0;
    for (const row of matrix) for (const v of row) {
      if (v < lo) lo = v; if (v > hi) hi = v;
      if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }
    this.opts.mode = data.mode || 'sequential';
    this.data = { ...data, ne, ns, lo, hi, maxAbs };
    this._buildMesh();
    this._buildLabels();
    this.clearMarker();
    return { lo, hi, maxAbs };
  }

  setOption(key, val) {
    this.opts[key] = val;
    if (!this.data) return;
    if (key === 'wireframe') { this.wire.visible = !!val; }
    else { this._buildMesh(); this._buildLabels(); } // heightScale etc. reshape geometry
  }

  _buildMesh() {
    this.meshGroup.clear();
    const { matrix, ne, ns } = this.data;
    const positions = new Float32Array(ne * ns * 3);
    const colors = new Float32Array(ne * ns * 3);
    for (let i = 0; i < ne; i++) {
      for (let j = 0; j < ns; j++) {
        const idx = (i * ns + j) * 3;
        const { y, color } = this._yc(matrix[i][j]);
        positions[idx] = this._xForCol(j, ns);
        positions[idx + 1] = y;
        positions[idx + 2] = this._zForRow(i, ne);
        colors[idx] = color[0] / 255; colors[idx + 1] = color[1] / 255; colors[idx + 2] = color[2] / 255;
      }
    }
    const indices = [];
    for (let i = 0; i < ne - 1; i++) {
      for (let j = 0; j < ns - 1; j++) {
        const a = i * ns + j, b = a + 1, c = a + ns, d = c + 1;
        indices.push(a, c, b, b, c, d);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    this.geo = geo;

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.62, metalness: 0.12,
      side: THREE.DoubleSide, flatShading: false,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.meshGroup.add(this.mesh);

    // Wireframe overlay (thin lines on top of the shaded surface).
    const wireMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 });
    this.wire = new THREE.Mesh(geo, wireMat);
    this.wire.visible = !!this.opts.wireframe;
    this.meshGroup.add(this.wire);

    // Zero plane for diverging mode so above/below the BS baseline reads instantly.
    if (this.opts.mode === 'diverging') {
      const pg = new THREE.PlaneGeometry(SIZE_X * 1.08, SIZE_Z * 1.08);
      const pm = new THREE.MeshBasicMaterial({ color: 0x3a4660, transparent: true, opacity: 0.12, side: THREE.DoubleSide });
      const plane = new THREE.Mesh(pg, pm);
      plane.rotation.x = -Math.PI / 2;
      this.meshGroup.add(plane);
    }
  }

  _buildLabels() {
    this.labelGroup.clear();
    const { strikes, expiries, ns, ne, valueLabel, S } = this.data;
    const yTop = HEIGHT * this.opts.heightScale;

    // Strike ticks along the front edge (z = -SIZE_Z/2).
    const zEdge = -SIZE_Z / 2 - 0.8;
    const xstep = Math.max(1, Math.ceil(ns / 7));
    for (let j = 0; j < ns; j += xstep) {
      const sp = makeLabel(String(Math.round(strikes[j])));
      sp.position.set(this._xForCol(j, ns), -0.3, zEdge);
      this.labelGroup.add(sp);
    }
    const xTitle = makeLabel('STRIKE', { size: 12, color: '#4fe3c1', weight: 700 });
    xTitle.position.set(0, -0.3, zEdge - 1.2);
    this.labelGroup.add(xTitle);

    // ATM marker line at spot S (interpolate position along strike axis).
    if (S != null && strikes.length > 1 && S >= strikes[0] && S <= strikes[strikes.length - 1]) {
      let j = 0;
      while (j < strikes.length - 1 && strikes[j + 1] < S) j++;
      const frac = (S - strikes[j]) / (strikes[j + 1] - strikes[j] || 1);
      const x = this._xForCol(j + frac, ns);
      const mat = new THREE.LineBasicMaterial({ color: 0x36e2b4, transparent: true, opacity: 0.5 });
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 0, -SIZE_Z / 2), new THREE.Vector3(x, yTop * 0.5, SIZE_Z / 2),
      ]);
      this.labelGroup.add(new THREE.Line(g, mat));
      const atm = makeLabel('spot ' + S.toFixed(0), { size: 11, color: '#36e2b4' });
      atm.position.set(x, yTop * 0.55, SIZE_Z / 2 + 0.3);
      this.labelGroup.add(atm);
    }

    // Expiry ticks along the left edge (x = -SIZE_X/2).
    const xEdge = -SIZE_X / 2 - 1.0;
    for (let i = 0; i < ne; i++) {
      const e = expiries[i];
      const lbl = (e && e.days != null) ? e.days + 'd' : (e && e.date ? String(e.date).slice(5) : String(i));
      const sp = makeLabel(lbl);
      sp.position.set(xEdge, -0.3, this._zForRow(i, ne));
      this.labelGroup.add(sp);
    }
    const zTitle = makeLabel('EXPIRY', { size: 12, color: '#4fe3c1', weight: 700 });
    zTitle.position.set(xEdge - 1.0, -0.3, 0);
    this.labelGroup.add(zTitle);

    // Value axis ticks up the back-left corner.
    const vx = -SIZE_X / 2 - 0.6, vz = -SIZE_Z / 2 - 0.6;
    const ticks = 4;
    for (let k = 0; k <= ticks; k++) {
      const frac = k / ticks;
      let val;
      if (this.opts.mode === 'diverging') val = (frac - 0.5) * 2 * this.data.maxAbs;
      else val = this.data.lo + frac * (this.data.hi - this.data.lo);
      const y = this.opts.mode === 'diverging' ? frac * yTop : frac * yTop;
      const sp = makeLabel(val.toFixed(val >= 100 ? 0 : 1), { size: 11 });
      sp.position.set(vx, y, vz);
      this.labelGroup.add(sp);
    }
    const vTitle = makeLabel(valueLabel || 'VALUE', { size: 12, color: '#4fe3c1', weight: 700 });
    vTitle.position.set(vx, yTop + 0.5, vz);
    this.labelGroup.add(vTitle);
  }

  // Place a glowing marker at (strike K, expiry T-years) sitting on the surface.
  setMarker(K, T, value) {
    this.clearMarker();
    if (!this.data) return;
    const { strikes, expiries, ns, ne } = this.data;
    // interpolate column from K
    if (K < strikes[0] || K > strikes[strikes.length - 1]) return;
    let j = 0; while (j < strikes.length - 1 && strikes[j + 1] < K) j++;
    const cf = (K - strikes[j]) / (strikes[j + 1] - strikes[j] || 1);
    const x = this._xForCol(j + cf, ns);
    // interpolate row from T (years) using expiry T or days/365
    const tOf = (e) => (e && e.T != null) ? e.T : (e && e.days != null ? e.days / 365 : Number(e));
    let i = 0; const ts = expiries.map(tOf);
    while (i < ts.length - 1 && ts[i + 1] < T) i++;
    const rf = ts[i + 1] != null ? (T - ts[i]) / (ts[i + 1] - ts[i] || 1) : 0;
    const z = this._zForRow(i + Math.max(0, Math.min(1, rf)), ne);
    const { y } = this._yc(value);

    const sph = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xffd166 })
    );
    sph.position.set(x, y, z);
    this.markerGroup.add(sph);
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(0.34, 24, 24),
      new THREE.MeshBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.25 })
    );
    halo.position.copy(sph.position);
    this.markerGroup.add(halo);
    // drop line to floor
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, 0, z), new THREE.Vector3(x, y, z)]);
    this.markerGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.4 })));
  }
  clearMarker() { this.markerGroup.clear(); }

  resetView() {
    this.camera.position.set(13, 11, 15);
    this.controls.target.set(0, HEIGHT * 0.35, 0);
    this.controls.update();
  }

  // --- internals ------------------------------------------------------------
  _onPointerMove(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._hoverActive = true;
    this._hoverXY = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  _pick() {
    if (!this._hoverActive || !this.mesh || !this.overlay) return;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObject(this.mesh, false)[0];
    if (!hit) { this.overlay.style.display = 'none'; return; }
    const { ns, strikes, expiries, matrix, valueLabel } = this.data;
    const vi = hit.face.a;                 // nearest vertex of the hit triangle
    const i = Math.floor(vi / ns), j = vi % ns;
    const e = expiries[i];
    const exp = (e && e.days != null) ? e.days + 'd' : (e && e.date ? e.date : i);
    this.overlay.style.display = 'block';
    this.overlay.style.left = this._hoverXY.x + 14 + 'px';
    this.overlay.style.top = this._hoverXY.y + 14 + 'px';
    this.overlay.innerHTML =
      `<b>${(valueLabel || 'value')}</b> ${matrix[i][j].toFixed(3)}` +
      `<span>K ${Math.round(strikes[j])} &middot; ${exp}</span>`;
  }

  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate);
    this.controls.update();
    this._pick();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this._ro.disconnect();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
