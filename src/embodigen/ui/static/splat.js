/* 4D Gaussian-splat viewer (WebGL2): world.json + static.splat + dynamic/<frame>.splat.
 *
 * Each splat is drawn as a screen-space ellipse: its 3D covariance R S S^T R^T is
 * projected with the Jacobian of the perspective map (EWA splatting, Zwicker et al.;
 * the 3DGS renderer of Kerbl et al.), splats are sorted front to back on the CPU and
 * composited with "under" blending. Static splats are uploaded once; a frame change
 * only replaces the dynamic tail of the data texture.
 *
 * Editing: pick a splat, grow a sphere selection, delete or undo, export the frame
 * as a 3DGS PLY. The canvas can be recorded to WebM.
 */
(function (global) {
  "use strict";

  const TEX_W = 4096; // texels per row; 4 texels per splat
  const SH_C0 = 0.28209479177387814;

  const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform highp sampler2D u_data;
uniform mat4 u_view, u_proj;
uniform vec2 u_focal, u_viewport;
in vec2 a_corner;
in uint a_index;
out vec4 v_color;
out vec2 v_pos;
ivec2 at(uint i, int k) { int t = int(i) * 4 + k; return ivec2(t % ${TEX_W}, t / ${TEX_W}); }
void main() {
  vec4 t0 = texelFetch(u_data, at(a_index, 0), 0);
  vec4 t1 = texelFetch(u_data, at(a_index, 1), 0);
  vec4 t2 = texelFetch(u_data, at(a_index, 2), 0);
  vec4 t3 = texelFetch(u_data, at(a_index, 3), 0);
  vec4 cam = u_view * vec4(t0.xyz, 1.0);
  vec4 clip = u_proj * cam;
  if (cam.z > -1e-3 || abs(clip.x) > 1.3 * clip.w || abs(clip.y) > 1.3 * clip.w) { gl_Position = vec4(0, 0, 2, 1); return; }
  mat3 S = mat3(t2.x, t2.y, t2.z, t2.y, t2.w, t3.x, t2.z, t3.x, t3.y);
  float z = cam.z;
  mat3 J = mat3(u_focal.x / -z, 0.0, 0.0,
                0.0, u_focal.y / -z, 0.0,
                u_focal.x * cam.x / (z * z), u_focal.y * cam.y / (z * z), 0.0);
  mat3 T = J * mat3(u_view);
  mat3 C = T * S * transpose(T);
  float a = C[0][0] + 0.3, b = C[0][1], c = C[1][1] + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 <= 0.0) { gl_Position = vec4(0, 0, 2, 1); return; }
  vec2 dir = normalize(vec2(b, l1 - a) + vec2(1e-9, 0.0));
  vec2 major = min(sqrt(l1), 1024.0) * dir;
  vec2 minor = min(sqrt(l2), 1024.0) * vec2(dir.y, -dir.x);
  vec2 offset = (a_corner.x * major + a_corner.y * minor) * 2.0 / u_viewport;
  vec3 rgb = t1.w > 0.5 ? mix(t1.rgb, vec3(1.0, 0.25, 0.85), 0.65) : t1.rgb;
  v_color = vec4(rgb, t0.w);
  v_pos = a_corner;
  gl_Position = vec4(clip.xy / clip.w + offset, 0.0, 1.0);
}`;

  const FS = `#version 300 es
precision highp float;
in vec4 v_color;
in vec2 v_pos;
out vec4 frag;
void main() {
  float r2 = dot(v_pos, v_pos);
  if (r2 > 9.0) discard;
  float alpha = v_color.a * exp(-0.5 * r2);
  if (alpha < 1.0 / 255.0) discard;
  frag = vec4(v_color.rgb * alpha, alpha);
}`;

  /* ---------------------------------------------------------------- math */
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  function lookAt(eye, target, up) {
    const f = norm(sub(target, eye));
    let s = cross(f, up);
    if (Math.hypot(...s) < 1e-6) s = cross(f, [0, 1, 0]);
    s = norm(s);
    const u = cross(s, f);
    // column-major 4x4, camera looks down -z
    return new Float32Array([s[0], u[0], -f[0], 0, s[1], u[1], -f[1], 0, s[2], u[2], -f[2], 0,
      -dot(s, eye), -dot(u, eye), dot(f, eye), 1]);
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }

  /* ---------------------------------------------------------------- data */
  // .splat record: float32 xyz, float32 scale xyz, uint8 rgba, uint8 quat (w,x,y,z)
  function parseSplat(buffer) {
    const n = Math.floor(buffer.byteLength / 32);
    const f = new Float32Array(buffer, 0, n * 8);
    const u = new Uint8Array(buffer, 0, n * 32);
    const pos = new Float32Array(n * 3), tex = new Float32Array(n * 16);
    for (let i = 0; i < n; i++) {
      const x = f[i * 8], y = f[i * 8 + 1], z = f[i * 8 + 2];
      const sx = f[i * 8 + 3], sy = f[i * 8 + 4], sz = f[i * 8 + 5];
      const o = i * 32 + 24;
      let qw = (u[o + 4] - 128) / 128, qx = (u[o + 5] - 128) / 128, qy = (u[o + 6] - 128) / 128, qz = (u[o + 7] - 128) / 128;
      const ql = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= ql; qx /= ql; qy /= ql; qz /= ql;
      // R (row-major) from the quaternion, then M = R * diag(s), Sigma = M M^T
      const R = [
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
        2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
        2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
      ];
      const M = [R[0] * sx, R[1] * sy, R[2] * sz, R[3] * sx, R[4] * sy, R[5] * sz, R[6] * sx, R[7] * sy, R[8] * sz];
      const s00 = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
      const s01 = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];
      const s02 = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];
      const s11 = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];
      const s12 = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];
      const s22 = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];
      pos.set([x, y, z], i * 3);
      tex.set([x, y, z, u[o + 3] / 255, u[o] / 255, u[o + 1] / 255, u[o + 2] / 255, 0, s00, s01, s02, s11, s12, s22, 0, 0], i * 16);
    }
    return { n, pos, tex, raw: new Uint8Array(buffer.slice(0, n * 32)) };
  }

  // PLY in the 3DGS layout, from .splat records (degree-0 colour, logit opacity, log scale)
  function toPly(records) {
    const n = records.reduce((a, r) => a + r.length / 32, 0);
    const names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
      "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
    const head = `ply\nformat binary_little_endian 1.0\ncomment edited in the embodigen console\nelement vertex ${n}\n` +
      names.map((p) => `property float ${p}\n`).join("") + "end_header\n";
    const headBytes = new TextEncoder().encode(head);
    const out = new ArrayBuffer(headBytes.length + n * names.length * 4);
    new Uint8Array(out).set(headBytes);
    const dv = new DataView(out);
    let off = headBytes.length;
    const put = (v) => { dv.setFloat32(off, v, true); off += 4; };
    for (const rec of records) {
      const f = new DataView(rec.buffer, rec.byteOffset, rec.byteLength);
      for (let i = 0; i < rec.length / 32; i++) {
        const b = i * 32;
        for (let k = 0; k < 3; k++) put(f.getFloat32(b + k * 4, true));
        put(0); put(0); put(0);
        for (let k = 0; k < 3; k++) put((rec[b + 24 + k] / 255 - 0.5) / SH_C0);
        const a = Math.min(Math.max(rec[b + 27] / 255, 1e-4), 1 - 1e-4);
        put(Math.log(a / (1 - a)));
        for (let k = 0; k < 3; k++) put(Math.log(Math.max(f.getFloat32(b + 12 + k * 4, true), 1e-7)));
        const q = [0, 1, 2, 3].map((k) => (rec[b + 28 + k] - 128) / 128);
        const ql = Math.hypot(...q) || 1;
        q.forEach((v) => put(v / ql));
      }
    }
    return new Blob([out], { type: "application/octet-stream" });
  }

  /* ---------------------------------------------------------------- viewer */
  class SplatViewer {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.fetchBuffer = opts.fetchBuffer; // (relative path) -> Promise<ArrayBuffer>
      this.onFrame = opts.onFrame || (() => {});
      const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true });
      if (!gl) throw new Error("这个浏览器不支持 WebGL2");
      this.gl = gl;
      this._program();
      this.frame = 0;
      this.mode = "orbit";
      this.orbit = { az: -0.6, el: 0.45, dist: 2, target: [0, 0, 0] };
      this.dirty = true;
      this.layers = { static: true, dynamic: true };
      this.deleted = { static: new Set(), dynamic: new Map() };
      this.selected = new Set();
      this.history = [];
      this.cache = new Map();
      this._controls();
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }

    _program() {
      const gl = this.gl;
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      this.prog = p;
      gl.useProgram(p);
      this.u = Object.fromEntries(["u_data", "u_view", "u_proj", "u_focal", "u_viewport"].map((n) => [n, gl.getUniformLocation(p, n)]));
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      const corners = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, corners);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-3, -3, 3, -3, 3, 3, -3, 3]), gl.STATIC_DRAW);
      const ac = gl.getAttribLocation(p, "a_corner");
      gl.enableVertexAttribArray(ac);
      gl.vertexAttribPointer(ac, 2, gl.FLOAT, false, 0, 0);
      this.indexBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
      const ai = gl.getAttribLocation(p, "a_index");
      gl.enableVertexAttribArray(ai);
      gl.vertexAttribIPointer(ai, 1, gl.UNSIGNED_INT, 0, 0);
      gl.vertexAttribDivisor(ai, 1);
      this.tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE); // front to back
    }

    async load(world) {
      this.world = world;
      this.cache.clear();
      this.deleted = { static: new Set(), dynamic: new Map() };
      this.selected.clear();
      this.history = [];
      this.static = parseSplat(await this.fetchBuffer(world.static.file));
      const maxDyn = Math.max(0, ...world.per_frame.map((f) => f.count));
      this.staticRows = Math.ceil((this.static.n * 4) / TEX_W);
      this.dynBase = (this.staticRows * TEX_W) / 4; // first splat index of the dynamic tail
      const rows = this.staticRows + Math.max(1, Math.ceil((maxDyn * 4) / TEX_W));
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, rows, 0, gl.RGBA, gl.FLOAT, null);
      this._upload(this.static.tex, 0);
      const b = world.bounds;
      this.orbit = { az: -0.6, el: 0.45, dist: Math.max(b.radius * 2.2, 0.2), target: b.center.slice() };
      this.dyn = null;
      await this.setFrame(0);
    }

    _upload(tex, row) {
      if (!tex.length) return;
      const gl = this.gl;
      const rows = Math.ceil(tex.length / 4 / TEX_W);
      const padded = new Float32Array(rows * TEX_W * 4);
      padded.set(tex);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, row, TEX_W, rows, gl.RGBA, gl.FLOAT, padded);
    }

    async _dynamic(i) {
      const f = this.world.per_frame[i];
      if (!f || !f.dynamic) return { n: 0, pos: new Float32Array(0), tex: new Float32Array(0), raw: new Uint8Array(0) };
      if (!this.cache.has(i)) {
        if (this.cache.size > 64) this.cache.delete(this.cache.keys().next().value);
        this.cache.set(i, this.fetchBuffer(f.dynamic).then(parseSplat));
      }
      return this.cache.get(i);
    }

    async setFrame(i) {
      const n = this.world.frames;
      i = ((i % n) + n) % n;
      this.frame = i;
      const dyn = await this._dynamic(i);
      if (this.frame !== i) return;
      this.dyn = dyn;
      this._upload(this._marked(dyn.tex, this.selectedDyn(i)), this.staticRows);
      if (this.mode === "follow") this.follow();
      this.dirty = true;
      this.onFrame(i);
      // warm the next frames
      for (let k = 1; k <= 3; k++) this._dynamic((i + k) % n);
    }

    selectedDyn() {
      const out = new Set();
      for (const s of this.selected) if (s >= this.dynBase) out.add(s - this.dynBase);
      return out;
    }

    _marked(tex, marks) {
      if (!marks.size) return tex;
      const copy = tex.slice();
      for (const k of marks) copy[k * 16 + 7] = 1;
      return copy;
    }

    follow() {
      const cam = this.world.cameras[this.frame];
      if (cam) this.camera = cam;
      this.dirty = true;
    }

    setMode(mode) {
      this.mode = mode;
      if (mode === "follow") this.follow();
      this.dirty = true;
    }

    currentCamera() {
      if (this.mode === "follow" && this.camera) return this.camera;
      const o = this.orbit;
      const eye = add(o.target, [o.dist * Math.cos(o.el) * Math.cos(o.az), o.dist * Math.cos(o.el) * Math.sin(o.az), o.dist * Math.sin(o.el)]);
      return { eye, target: o.target, up: [0, 0, 1], fov_y: 50 };
    }

    _matrices() {
      const c = this.currentCamera();
      const w = this.canvas.width, h = this.canvas.height;
      const fovy = (c.fov_y * Math.PI) / 180;
      const r = this.world ? this.world.bounds.radius : 1;
      const dist = Math.hypot(...sub(c.eye, this.world ? this.world.bounds.center : [0, 0, 0]));
      const view = lookAt(c.eye, c.target, c.up);
      const proj = perspective(fovy, w / h, Math.max(1e-3, (dist + r) * 1e-4), (dist + r) * 4);
      const fy = h / (2 * Math.tan(fovy / 2));
      return { view, proj, focal: [fy, fy], c };
    }

    _visible() {
      const ids = [];
      if (this.layers.static) for (let i = 0; i < this.static.n; i++) if (!this.deleted.static.has(i)) ids.push(i);
      if (this.layers.dynamic && this.dyn) {
        const gone = this.deleted.dynamic.get(this.frame) || new Set();
        for (let i = 0; i < this.dyn.n; i++) if (!gone.has(i)) ids.push(this.dynBase + i);
      }
      return ids;
    }

    _pos(id) {
      if (id < this.dynBase) return [this.static.pos[id * 3], this.static.pos[id * 3 + 1], this.static.pos[id * 3 + 2]];
      const k = id - this.dynBase;
      return [this.dyn.pos[k * 3], this.dyn.pos[k * 3 + 1], this.dyn.pos[k * 3 + 2]];
    }

    _sort(view) {
      const ids = this._visible();
      const n = ids.length;
      const depth = new Float32Array(n);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const p = this._pos(ids[i]);
        const d = -(view[2] * p[0] + view[6] * p[1] + view[10] * p[2] + view[14]); // distance along the view axis
        depth[i] = d;
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
      const B = 65536, counts = new Uint32Array(B), keys = new Uint32Array(n);
      const scale = (B - 1) / Math.max(hi - lo, 1e-9);
      for (let i = 0; i < n; i++) { keys[i] = ((depth[i] - lo) * scale) | 0; counts[keys[i]]++; }
      for (let i = 1; i < B; i++) counts[i] += counts[i - 1];
      const order = new Uint32Array(n);
      for (let i = n - 1; i >= 0; i--) order[--counts[keys[i]]] = ids[i]; // nearest first
      return order;
    }

    _loop() {
      requestAnimationFrame(this._loop);
      if (!this.world || !this.static) return;
      const cv = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; this.dirty = true; }
      if (!this.dirty) return;
      this.dirty = false;
      const gl = this.gl;
      const { view, proj, focal } = this._matrices();
      const order = this._sort(view);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
      gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.uniform1i(this.u.u_data, 0);
      gl.uniformMatrix4fv(this.u.u_view, false, view);
      gl.uniformMatrix4fv(this.u.u_proj, false, proj);
      gl.uniform2f(this.u.u_focal, focal[0], focal[1]);
      gl.uniform2f(this.u.u_viewport, w, h);
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, order.length);
      this.drawn = order.length;
    }

    /* -------- controls: drag = orbit, shift/right drag = pan, wheel = zoom */
    _controls() {
      const cv = this.canvas;
      let last = null, button = 0, moved = 0;
      cv.addEventListener("contextmenu", (e) => e.preventDefault());
      cv.addEventListener("pointerdown", (e) => {
        last = [e.clientX, e.clientY]; button = e.shiftKey ? 2 : e.button; moved = 0;
        cv.setPointerCapture(e.pointerId);
      });
      cv.addEventListener("pointermove", (e) => {
        if (!last) return;
        const dx = e.clientX - last[0], dy = e.clientY - last[1];
        last = [e.clientX, e.clientY];
        moved += Math.abs(dx) + Math.abs(dy);
        if (this.mode === "follow") { this.orbitFromCamera(); this.mode = "orbit"; this.onMode && this.onMode("orbit"); }
        const o = this.orbit;
        if (button === 2) {
          const c = this.currentCamera();
          const f = norm(sub(c.target, c.eye)), s = norm(cross(f, [0, 0, 1])), u = cross(s, f);
          const k = o.dist / cv.clientHeight;
          o.target = add(o.target, add(mul(s, -dx * k), mul(u, dy * k)));
        } else {
          o.az -= dx * 0.008;
          o.el = Math.max(-1.5, Math.min(1.5, o.el + dy * 0.008));
        }
        this.dirty = true;
      });
      cv.addEventListener("pointerup", (e) => {
        const click = moved < 4 && last;
        last = null;
        if (click && this.onPick) this.onPick(this.pick(e.offsetX, e.offsetY));
      });
      cv.addEventListener("wheel", (e) => {
        e.preventDefault();
        if (this.mode === "follow") { this.orbitFromCamera(); this.mode = "orbit"; this.onMode && this.onMode("orbit"); }
        this.orbit.dist *= Math.exp(e.deltaY * 0.0012);
        this.dirty = true;
      }, { passive: false });
    }

    orbitFromCamera() {
      const c = this.currentCamera();
      const d = sub(c.eye, c.target);
      const dist = Math.hypot(...d) || 1;
      this.orbit = { az: Math.atan2(d[1], d[0]), el: Math.asin(Math.max(-1, Math.min(1, d[2] / dist))), dist, target: c.target.slice() };
    }

    /* -------- editing */
    pick(x, y) {
      const { view, proj } = this._matrices();
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      let best = null, bestD = Infinity;
      for (const id of this._visible()) {
        const p = this._pos(id);
        const cx = view[0] * p[0] + view[4] * p[1] + view[8] * p[2] + view[12];
        const cy = view[1] * p[0] + view[5] * p[1] + view[9] * p[2] + view[13];
        const cz = view[2] * p[0] + view[6] * p[1] + view[10] * p[2] + view[14];
        if (cz >= 0) continue;
        const sx = ((proj[0] * cx) / -cz * 0.5 + 0.5) * w, sy = (0.5 - (proj[5] * cy) / -cz * 0.5) * h;
        if (Math.hypot(sx - x, sy - y) < 6 && -cz < bestD) { bestD = -cz; best = id; }
      }
      return best == null ? null : { id: best, pos: this._pos(best) };
    }

    selectSphere(center, radius) {
      this.selected = new Set(this._visible().filter((id) => Math.hypot(...sub(this._pos(id), center)) <= radius));
      this._remark();
      return this.selected.size;
    }

    clearSelection() { this.selected.clear(); this._remark(); }

    _remark() {
      const marks = new Set();
      for (const s of this.selected) if (s < this.dynBase) marks.add(s);
      this._upload(this._marked(this.static.tex, marks), 0);
      if (this.dyn) this._upload(this._marked(this.dyn.tex, this.selectedDyn()), this.staticRows);
      this.dirty = true;
    }

    deleteSelection() {
      if (!this.selected.size) return 0;
      const step = { static: [], dynamic: [], frame: this.frame };
      const gone = this.deleted.dynamic.get(this.frame) || new Set();
      for (const s of this.selected) {
        if (s < this.dynBase) { this.deleted.static.add(s); step.static.push(s); }
        else { gone.add(s - this.dynBase); step.dynamic.push(s - this.dynBase); }
      }
      this.deleted.dynamic.set(this.frame, gone);
      this.history.push(step);
      const n = this.selected.size;
      this.clearSelection();
      return n;
    }

    undo() {
      const step = this.history.pop();
      if (!step) return false;
      step.static.forEach((s) => this.deleted.static.delete(s));
      const gone = this.deleted.dynamic.get(step.frame);
      if (gone) step.dynamic.forEach((s) => gone.delete(s));
      this.dirty = true;
      return true;
    }

    exportPly() {
      const keep = (raw, n, dropped) => {
        const out = new Uint8Array((n - dropped.size) * 32);
        let k = 0;
        for (let i = 0; i < n; i++) if (!dropped.has(i)) out.set(raw.subarray(i * 32, i * 32 + 32), (k++) * 32);
        return out;
      };
      const parts = [];
      if (this.layers.static) parts.push(keep(this.static.raw, this.static.n, this.deleted.static));
      if (this.layers.dynamic && this.dyn) parts.push(keep(this.dyn.raw, this.dyn.n, this.deleted.dynamic.get(this.frame) || new Set()));
      return toPly(parts);
    }

    setLayer(name, on) { this.layers[name] = on; this.dirty = true; }
  }

  global.SplatViewer = SplatViewer;
  global.SplatViewer.parseSplat = parseSplat;
  global.SplatViewer.toPly = toPly;
})(window);
