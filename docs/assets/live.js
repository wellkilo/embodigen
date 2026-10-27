/* EmbodiGen showcase — live 3D replay of a real episode.

   A tiny flat-shaded box renderer on Canvas2D (painter's algorithm, back-face
   culling). It draws only what the engine has: the bin, the cube and the two
   fingers, at the positions stored in the episode, plus annotations that are
   clearly overlays: the end-effector path, phase markers and the frustum of
   the camera the dataset was rendered from. No dependencies. */

(function () {
  const M = window.EMBODIGEN_METRICS;
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CAPTURE = new URLSearchParams(location.search).has("capture");

  /* ---------- vector helpers ---------- */
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const norm = (a) => {
    const n = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  };
  const lerp = (a, b, s) => a + (b - a) * s;
  const lerp3 = (a, b, s) => [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  const LIGHT = norm([-0.45, -0.55, 0.78]);
  let TARGET = [0.03, 0.0, 0.05];

  // Same ramp as `embodigen preview`: near is warm and bright.
  const RAMP = [
    [0.0, [252, 238, 170]],
    [0.3, [247, 152, 76]],
    [0.6, [196, 62, 104]],
    [0.85, [88, 40, 132]],
    [1.0, [24, 22, 60]],
  ];
  function ramp(x) {
    x = clamp(x, 0, 1);
    for (let i = 1; i < RAMP.length; i++) {
      const [x1, c1] = RAMP[i];
      const [x0, c0] = RAMP[i - 1];
      if (x <= x1) {
        const s = (x - x0) / (x1 - x0);
        return [lerp(c0[0], c1[0], s), lerp(c0[1], c1[1], s), lerp(c0[2], c1[2], s)];
      }
    }
    return RAMP[RAMP.length - 1][1];
  }

  const SEG = [
    [230, 96, 82], [86, 180, 233], [240, 200, 80], [120, 200, 140],
    [190, 120, 220], [240, 150, 70], [90, 210, 210], [220, 110, 170],
  ];

  function init() {
    const viewer = document.getElementById("viewer");
    const canvas = document.getElementById("scene3d");
    if (!viewer || !canvas || !M?.episode) return;
    const ctx = canvas.getContext("2d");
    const E = M.episode;
    const frames = E.frames;
    const N = frames.length;

    // Dataset camera: the one every label and pixel in the episode refers to.
    const dEye = E.camera.eye;
    const dFwd = norm(sub(E.camera.target, dEye));
    const dRight = norm(cross(dFwd, [0, 0, 1]));
    const dUp = cross(dRight, dFwd);
    const halfFov = E.camera.size / 2 / E.camera.fx;

    const statics = E.objects.filter((o) => o.name !== "cube");
    const cubeSpec = E.objects.find((o) => o.name === "cube");
    const segIndex = {};
    [...E.objects.map((o) => o.name), "finger_l", "finger_r"].forEach((n, i) => (segIndex[n] = i));

    // Phases as contiguous runs of frames.
    const phases = [];
    frames.forEach((f, i) => {
      const last = phases[phases.length - 1];
      if (!last || last.name !== f.phase) phases.push({ name: f.phase, start: i, end: i });
      else last.end = i;
    });

    const vmax = Math.max(
      1e-6,
      ...frames.flatMap((f) => f.action.slice(0, 3).map((v) => Math.abs(v)))
    );
    const gripOpen = Math.max(...frames.map((f) => f.gripper)) || 0.04;

    const st = {
      f: REDUCED || CAPTURE ? Math.floor(N * 0.45) : 0,
      playing: !(REDUCED || CAPTURE),
      hold: 0,
      yaw: -1.05,
      pitch: 0.5,
      dist: 1.25,
      mode: "material",
      drag: null,
      idle: 0,
      visible: false,
      scrubbing: false,
      W: 0,
      H: 0,
      dpr: 1,
    };

    /* ---------- HUD wiring ---------- */
    const $ = (id) => document.getElementById(id);
    const hud = {
      id: $("hud-id"), task: $("hud-task"), phase: $("hud-phase"),
      t: $("t-read"), fill: $("track-fill"), scrub: $("scrub"), play: $("play"),
      eef: $("tel-eef"), grip: $("tel-grip"), gripFill: $("grip-fill"),
      err: $("tel-err"), errFill: $("err-fill"), errThr: $("err-thr"),
      pip: $("pip"), pipMode: $("pip-mode"), bars: $("action-bars"), phases: $("phases"),
    };
    hud.id.textContent = `${E.id} · ${N} FRAMES · ${E.fps} FPS`;
    hud.task.textContent = E.instruction;

    hud.phases.innerHTML = phases
      .map((p) => `<span style="flex:${p.end - p.start + 1}" data-name="${p.name}"></span>`)
      .join("");
    const phaseEls = [...hud.phases.children];

    const labels = ["vx", "vy", "vz", "wx", "wy", "wz", "g"];
    hud.bars.innerHTML = labels.map((l) => `<div><span></span><em>${l}</em></div>`).join("");
    const barEls = [...hud.bars.querySelectorAll("span")];

    const errMax = E.threshold_mm * 1.5;
    hud.errThr.style.left = `${(E.threshold_mm / errMax) * 100}%`;

    /* ---------- camera ---------- */
    // Everything that must stay in frame: the bin, every end-effector
    // position and the cube's path. The orbit centres on their bounds.
    const keyPoints = [
      ...frames.map((f) => f.eef),
      ...frames.map((f) => f.cube),
      ...statics.flatMap((o) => [add(o.center, o.half_extent), sub(o.center, o.half_extent)]),
    ];
    const lo = [0, 1, 2].map((k) => Math.min(...keyPoints.map((p) => p[k])));
    const hi = [0, 1, 2].map((k) => Math.max(...keyPoints.map((p) => p[k])));
    TARGET = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];

    let eye, fwd, right, up, F, cx, cy;
    let fit = 1;
    function basis(yaw, pitch) {
      const cp = Math.cos(pitch);
      eye = add(TARGET, mul([cp * Math.cos(yaw), cp * Math.sin(yaw), Math.sin(pitch)], st.dist));
      fwd = norm(sub(TARGET, eye));
      right = norm(cross(fwd, [0, 0, 1]));
      up = cross(right, fwd);
    }
    function setupCamera() {
      basis(st.yaw, st.pitch);
      F = fit;
      cx = st.W / 2;
      // The timeline covers the bottom strip; keep the scene above it.
      cy = st.H * 0.45;
    }
    // Pick one focal length that keeps the key points in frame from every
    // orbit angle, so the scene never pops in and out while rotating.
    function fitFocal() {
      let need = 0;
      for (let yaw = 0; yaw < Math.PI * 2; yaw += Math.PI / 8) {
        for (const pitch of [0.25, 0.5, 0.9]) {
          basis(yaw, pitch);
          for (const p of keyPoints) {
            const r = sub(p, eye);
            const z = dot(r, fwd);
            const x = Math.abs(dot(r, right) / z) / (st.W * 0.42);
            const y = Math.abs(dot(r, up) / z) / (st.H * 0.35);
            need = Math.max(need, x, y);
          }
        }
      }
      fit = need > 0 ? 1 / need : 800;
    }
    function project(p) {
      const r = sub(p, eye);
      const z = dot(r, fwd);
      return [cx + (dot(r, right) / z) * F, cy - (dot(r, up) / z) * F, z];
    }

    /* ---------- scene at a fractional frame ---------- */
    function sample(f) {
      f = Number.isFinite(f) ? Math.max(0, f) : 0; // the first tick can arrive before the clock starts
      const i = Math.min(N - 1, Math.floor(f));
      const j = Math.min(N - 1, i + 1);
      const s = f - i;
      const a = frames[i];
      const b = frames[j];
      return {
        i: Math.min(N - 1, Math.round(f)),
        eef: lerp3(a.eef, b.eef, s),
        cube: lerp3(a.cube, b.cube, s),
        gripper: lerp(a.gripper, b.gripper, s),
        action: a.action.map((v, k) => lerp(v, b.action[k], s)),
        err: a.err_mm == null ? null : lerp(a.err_mm, b.err_mm ?? a.err_mm, s),
        phase: frames[Math.min(N - 1, Math.round(f))].phase,
        t: lerp(a.t, b.t, s),
      };
    }

    function boxesAt(S) {
      const boxes = statics.map((o) => ({ name: o.name, mat: o.material, albedo: o.albedo, c: o.center, h: o.half_extent }));
      boxes.push({ name: "cube", mat: "cube", albedo: cubeSpec.albedo, c: S.cube, h: cubeSpec.half_extent });
      // Fingers exactly as the exporter places them.
      const sep = S.gripper / 2 + 0.008;
      const hz = 0.018;
      const fh = [0.012, 0.008, hz];
      boxes.push({ name: "finger_l", mat: "gripper", c: [S.eef[0], S.eef[1] - sep, S.eef[2] + hz], h: fh });
      boxes.push({ name: "finger_r", mat: "gripper", c: [S.eef[0], S.eef[1] + sep, S.eef[2] + hz], h: fh });
      return boxes;
    }

    const AXES = [
      [[1, 0, 0], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]],
      [[-1, 0, 0], [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]]],
      [[0, 1, 0], [[1, 1, -1], [-1, 1, -1], [-1, 1, 1], [1, 1, 1]]],
      [[0, -1, 0], [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]]],
      [[0, 0, 1], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]],
      [[0, 0, -1], [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]],
    ];

    function baseColor(box) {
      if (st.mode === "seg") return SEG[segIndex[box.name] % SEG.length];
      if (st.mode === "clay") return [196, 200, 206];
      if (box.mat === "bin") return [44, 92, 94];
      if (box.mat === "gripper") return [170, 184, 214];
      return box.albedo || [200, 200, 200];
    }

    function faceColor(box, n, fc) {
      if (st.mode === "depth") {
        const z = dot(sub(fc, dEye), dFwd);
        const c = ramp((z - 0.5) / 0.42);
        const k = 0.82 + 0.18 * Math.max(0, dot(n, LIGHT));
        return `rgb(${c[0] * k | 0},${c[1] * k | 0},${c[2] * k | 0})`;
      }
      const base = baseColor(box);
      const lam = Math.max(0, dot(n, LIGHT));
      const k = st.mode === "seg" ? 0.7 + 0.3 * lam : 0.34 + 0.8 * lam;
      return `rgb(${Math.min(255, base[0] * k) | 0},${Math.min(255, base[1] * k) | 0},${Math.min(255, base[2] * k) | 0})`;
    }

    /* ---------- drawing ---------- */
    function drawFloor() {
      const ext = { x0: -0.34, x1: 0.42, y0: -0.3, y1: 0.32 };
      const corners = [
        [ext.x0, ext.y0, 0], [ext.x1, ext.y0, 0], [ext.x1, ext.y1, 0], [ext.x0, ext.y1, 0],
      ].map(project);
      const g = ctx.createRadialGradient(cx, cy + st.H * 0.12, 0, cx, cy + st.H * 0.12, st.W * 0.6);
      if (st.mode === "depth") {
        g.addColorStop(0, "rgba(196,62,104,0.35)");
        g.addColorStop(1, "rgba(24,22,60,0)");
      } else if (st.mode === "seg") {
        g.addColorStop(0, "rgba(40,82,88,0.5)");
        g.addColorStop(1, "rgba(20,46,52,0)");
      } else {
        g.addColorStop(0, st.mode === "clay" ? "rgba(150,156,166,0.28)" : "rgba(140,112,86,0.26)");
        g.addColorStop(1, "rgba(6,22,26,0)");
      }
      ctx.fillStyle = g;
      ctx.beginPath();
      corners.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fill();

      ctx.lineWidth = 1;
      const step = 0.04;
      for (let x = ext.x0; x <= ext.x1 + 1e-6; x += step) line3([x, ext.y0, 0], [x, ext.y1, 0], gridAlpha(x, ext.x0, ext.x1));
      for (let y = ext.y0; y <= ext.y1 + 1e-6; y += step) line3([ext.x0, y, 0], [ext.x1, y, 0], gridAlpha(y, ext.y0, ext.y1));
    }
    function gridAlpha(v, lo, hi) {
      const m = Math.min(v - lo, hi - v) / ((hi - lo) / 2);
      return 0.02 + 0.08 * Math.min(1, m * 1.6);
    }
    function line3(a, b, alpha, color = "143,182,255") {
      const p = project(a);
      const q = project(b);
      if (p[2] < 0.03 || q[2] < 0.03) return;
      ctx.strokeStyle = `rgba(${color},${alpha})`;
      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(q[0], q[1]);
      ctx.stroke();
    }

    function drawShadow(c, h, height) {
      const p = project([c[0], c[1], 0.0005]);
      const sx = Math.max(h[0], h[1]) * F / p[2] * 1.5;
      const alpha = clamp(0.5 - height * 2.2, 0.08, 0.5);
      const g = ctx.createRadialGradient(p[0], p[1], 0, p[0], p[1], sx);
      g.addColorStop(0, `rgba(0,0,0,${alpha})`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(p[0], p[1], sx, sx * 0.55, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    function drawBoxes(boxes) {
      const faces = [];
      for (const box of boxes) {
        for (const [n, verts] of AXES) {
          const fc = add(box.c, [n[0] * box.h[0], n[1] * box.h[1], n[2] * box.h[2]]);
          if (dot(n, sub(eye, fc)) <= 0) continue;
          const pts = verts.map((v) => project([box.c[0] + v[0] * box.h[0], box.c[1] + v[1] * box.h[1], box.c[2] + v[2] * box.h[2]]));
          if (pts.some((p) => p[2] < 0.03)) continue;
          const d = sub(fc, eye);
          faces.push({ pts, depth: dot(d, d), color: faceColor(box, n, fc) });
        }
      }
      faces.sort((a, b) => b.depth - a.depth);
      ctx.lineJoin = "round";
      for (const face of faces) {
        ctx.beginPath();
        face.pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.fillStyle = face.color;
        ctx.fill();
        ctx.strokeStyle = st.mode === "clay" ? "rgba(40,44,52,0.35)" : "rgba(255,255,255,0.10)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    function drawTrail(S) {
      const pts = frames.map((f) => project(f.eef));
      const cut = S.i;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      // Full planned path, dim and dashed.
      ctx.setLineDash([3, 5]);
      ctx.strokeStyle = "rgba(142,233,228,0.28)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
      ctx.setLineDash([]);
      // Travelled part, glowing.
      const head = project(S.eef);
      ctx.save();
      ctx.shadowColor = "rgba(15,217,210,0.9)";
      ctx.shadowBlur = 14;
      ctx.strokeStyle = "rgba(90,226,220,0.95)";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let k = 0; k <= cut; k++) (k ? ctx.lineTo(pts[k][0], pts[k][1]) : ctx.moveTo(pts[k][0], pts[k][1]));
      ctx.lineTo(head[0], head[1]);
      ctx.stroke();
      ctx.restore();

      // Phase start markers.
      ctx.font = "500 9.5px ui-monospace, 'SF Mono', Menlo, monospace";
      for (const p of phases) {
        const [x, y] = pts[p.start];
        const active = S.phase === p.name;
        ctx.fillStyle = active ? "#8ee9e4" : "rgba(142,233,228,0.45)";
        ctx.beginPath();
        ctx.arc(x, y, active ? 3.2 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      // End-effector head.
      const pulse = 6 + Math.sin(performance.now() / 260) * 2;
      ctx.strokeStyle = "rgba(142,233,228,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(head[0], head[1], pulse + 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "#effcfb";
      ctx.beginPath();
      ctx.arc(head[0], head[1], 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(239,252,251,0.85)";
      ctx.fillText(`EEF  z ${S.eef[2].toFixed(3)} m`, head[0] + 14, head[1] - 10);
    }

    function drawFrustum() {
      const e = project(dEye);
      const m = 40;
      if (e[2] < 0.2 || e[0] < -m || e[0] > st.W + m || e[1] < -m || e[1] > st.H + m) return;
      const d = 0.1;
      const c = add(dEye, mul(dFwd, d));
      const r = mul(dRight, d * halfFov);
      const u = mul(dUp, d * halfFov);
      const quad = [add(add(c, r), u), add(sub(c, r), u), sub(sub(c, r), u), sub(add(c, r), u)];
      ctx.lineWidth = 1;
      quad.forEach((q, k) => {
        line3(dEye, q, 0.55, "23,201,168");
        line3(q, quad[(k + 1) % 4], 0.7, "23,201,168");
      });
      // Dashed sight line to the scene.
      const p = project(dEye);
      const t = project(E.camera.target);
      if (p[2] > 0.03 && t[2] > 0.03) {
        ctx.setLineDash([2, 6]);
        ctx.strokeStyle = "rgba(122,189,106,0.3)";
        ctx.beginPath();
        ctx.moveTo(p[0], p[1]);
        ctx.lineTo(t[0], t[1]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(122,189,106,0.9)";
        ctx.font = "500 9.5px ui-monospace, 'SF Mono', Menlo, monospace";
        ctx.fillText("DATASET CAMERA", p[0] + 10, p[1] - 8);
      }
    }

    function render() {
      setupCamera();
      ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
      ctx.clearRect(0, 0, st.W, st.H);
      const S = sample(st.f);
      const boxes = boxesAt(S);
      drawFloor();
      drawShadow(S.cube, cubeSpec.half_extent, S.cube[2]);
      drawShadow(S.eef, [0.02, 0.03, 0], S.eef[2]);
      drawBoxes(boxes);
      drawTrail(S);
      drawFrustum();
      updateHud(S);
    }

    /* ---------- HUD ---------- */
    let lastPhase = "";
    function updateHud(S) {
      const progress = st.f / (N - 1);
      hud.fill.style.width = `${progress * 100}%`;
      if (!st.scrubbing) hud.scrub.value = String(Math.round(progress * 1000));
      hud.t.textContent = `${S.t.toFixed(2)} s`;
      hud.eef.textContent = S.eef.map((v) => (v >= 0 ? " " : "") + v.toFixed(3)).join(" ");
      hud.grip.textContent = `${S.gripper.toFixed(3)} m`;
      hud.gripFill.style.width = `${(S.gripper / gripOpen) * 100}%`;

      S.action.forEach((v, k) => {
        const n = k === 6 ? v / gripOpen : clamp(v / vmax, -1, 1);
        const el = barEls[k];
        el.classList.toggle("neg", n < 0);
        el.style.top = n >= 0 ? `${50 - n * 50}%` : "50%";
        el.style.height = `${Math.max(1.5, Math.abs(n) * 50)}%`;
      });

      if (S.err == null) {
        hud.err.textContent = "untracked";
        hud.errFill.style.width = "0";
      } else {
        hud.err.textContent = `${S.err.toFixed(2)} mm`;
        hud.errFill.style.width = `${clamp(S.err / errMax, 0.01, 1) * 100}%`;
        hud.errFill.parentElement.classList.toggle("bad", S.err > E.threshold_mm);
      }

      if (S.phase !== lastPhase) {
        lastPhase = S.phase;
        hud.phase.textContent = S.phase.toUpperCase().replace("_", " ");
        phaseEls.forEach((el) => el.classList.toggle("on", el.dataset.name === S.phase));
      }

      const key = st.mode === "clay" ? "white" : "photoreal";
      const sheet = M.sheets?.[key];
      if (sheet && hud.pip) {
        const rows = Math.ceil(sheet.frames / sheet.cols);
        const idx = Math.min(sheet.frames - 1, S.i);
        hud.pip.style.backgroundImage = `url("${sheet.src}")`;
        hud.pip.style.backgroundSize = `${sheet.cols * 100}% ${rows * 100}%`;
        hud.pip.style.backgroundPosition = `${((idx % sheet.cols) / (sheet.cols - 1)) * 100}% ${(Math.floor(idx / sheet.cols) / Math.max(1, rows - 1)) * 100}%`;
        hud.pipMode.textContent = key === "white" ? "· WHITE MODEL" : "· PHOTOREAL";
      }
    }

    /* ---------- loop ---------- */
    let prev = performance.now();
    function tick(now) {
      const dt = Math.min(0.05, (now - prev) / 1000);
      prev = now;
      if (st.playing && !st.scrubbing) {
        if (st.f >= N - 1) {
          st.hold += dt;
          if (st.hold > 1.1) {
            st.f = 0;
            st.hold = 0;
          }
        } else {
          st.f = Math.min(N - 1, st.f + dt * E.fps);
        }
      }
      if (!st.drag) {
        st.idle += dt;
        if (st.idle > 2.5) st.yaw += dt * 0.07;
      }
      render();
      if (st.visible && !REDUCED) requestAnimationFrame(tick);
      else running = false;
    }
    let running = false;
    function start() {
      if (running || REDUCED) return;
      running = true;
      prev = performance.now();
      requestAnimationFrame(tick);
    }

    function resize() {
      const rect = viewer.getBoundingClientRect();
      st.dpr = Math.min(2, window.devicePixelRatio || 1);
      st.W = rect.width;
      st.H = rect.height;
      canvas.width = Math.round(st.W * st.dpr);
      canvas.height = Math.round(st.H * st.dpr);
      fitFocal();
      render();
    }

    /* ---------- interaction ---------- */
    viewer.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button, input, .timeline")) return;
      st.drag = { x: e.clientX, y: e.clientY, yaw: st.yaw, pitch: st.pitch };
      viewer.setPointerCapture(e.pointerId);
    });
    viewer.addEventListener("pointermove", (e) => {
      if (!st.drag) return;
      st.yaw = st.drag.yaw - (e.clientX - st.drag.x) * 0.008;
      st.pitch = clamp(st.drag.pitch + (e.clientY - st.drag.y) * 0.006, 0.12, 1.35);
      st.idle = 0;
      if (!running) render();
    });
    const endDrag = () => {
      st.drag = null;
      st.idle = 0;
    };
    viewer.addEventListener("pointerup", endDrag);
    viewer.addEventListener("pointercancel", endDrag);

    hud.scrub.addEventListener("input", () => {
      st.scrubbing = true;
      st.f = (Number(hud.scrub.value) / 1000) * (N - 1);
      st.hold = 0;
      if (!running) render();
    });
    hud.scrub.addEventListener("change", () => (st.scrubbing = false));
    hud.scrub.addEventListener("pointerup", () => (st.scrubbing = false));

    const setPlaying = (on) => {
      st.playing = on;
      hud.play.textContent = on ? "❚❚" : "▶";
      hud.play.setAttribute("aria-label", on ? "暂停" : "播放");
    };
    setPlaying(st.playing);
    hud.play.addEventListener("click", () => {
      if (st.f >= N - 1) st.f = 0;
      setPlaying(!st.playing);
      start();
    });

    document.querySelectorAll("[data-mode]").forEach((btn) =>
      btn.addEventListener("click", () => {
        st.mode = btn.dataset.mode;
        document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b === btn));
        if (!running) render();
      })
    );

    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(viewer);
    else window.addEventListener("resize", resize);
    resize();

    new IntersectionObserver(
      (entries) => {
        st.visible = entries[0].isIntersecting;
        if (st.visible) start();
      },
      { threshold: 0.05 }
    ).observe(viewer);
  }

  init();
})();
