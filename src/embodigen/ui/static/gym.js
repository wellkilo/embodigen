/* Live MuJoCo poses -> browser viewer; all state transitions happen in the CLI worker. */
window.EmbodiGym = function ({ api, toast }) {
  const $ = (s) => document.querySelector(s);
  const canvas = $("#gym-canvas"), ctx = canvas.getContext("2d");
  let live = null, frames = [], running = false, demo = false, visible = false;
  let pending = null, resetting = false, timer = null, grip = 1, top = false;
  const held = new Set();
  const keys = { KeyW: [0, 1], KeyS: [0, -1], KeyA: [1, 1], KeyD: [1, -1], KeyR: [2, 1], KeyF: [2, -1] };
  const controls = [...document.querySelectorAll("[data-gym-move]")];
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (v) => v.map((x) => x / Math.hypot(...v));

  function draw(state) {
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = "#091a21"; ctx.fillRect(0, 0, w, h);
    if (!state) {
      ctx.fillStyle = "#8da6ae"; ctx.font = "20px sans-serif"; ctx.textAlign = "center";
      ctx.fillText("重置场景，开始交互", w / 2, h / 2); return;
    }
    const eye = top ? [.01, -.001, 1.05] : [.65, -.8, .66];
    const forward = norm(sub([.02, 0, .04], eye));
    const right = norm(cross(forward, [0, 0, 1])), up = cross(right, forward);
    function project(p) {
      const v = sub(p, eye), z = dot(v, forward);
      return [w / 2 + dot(v, right) * h * 1.5 / z, h * .53 - dot(v, up) * h * 1.5 / z, z];
    }
    function line(a, b, color, width = 1) {
      a = project(a); b = project(b);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    const faces = [];
    for (const g of state.geoms) {
      // Show the task workspace rather than the sampler's 5-metre backdrop.
      const half = g.name === "table" ? [.46, .46, .02] : g.half_extent;
      const vertices = [];
      for (let k = 0; k < 8; k++) {
        const v = half.map((s, i) => (k & (1 << i) ? 1 : -1) * s);
        vertices.push(project(g.position.map((p, i) => p + dot(g.rotation[i], v))));
      }
      const indices = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
      indices.forEach((ids, f) => {
        const points = ids.map((i) => vertices[i]);
        if (points.some((p) => p[2] <= .02)) return;
        const shade = [.62, 1, .68, .88, .75, .9][f];
        const color = g.rgba.slice(0, 3).map((c) => Math.round(c * 255 * shade));
        faces.push({ points, z: points.reduce((s, p) => s + p[2], 0) / 4, color });
      });
    }
    faces.sort((a, b) => b.z - a.z).forEach(({ points, color }) => {
      ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
      ctx.fillStyle = `rgb(${color.join(",")})`; ctx.fill();
      ctx.strokeStyle = "rgba(3,15,23,.3)"; ctx.lineWidth = .8; ctx.stroke();
    });
    const origin = [-.35, -.32, .004];
    [[.12, 0, 0], [0, .12, 0], [0, 0, .12]].forEach((v, i) => {
      const p = origin.map((x, k) => x + v[k]); line(origin, p, ["#ff8a75", "#7ed9a6", "#85baff"][i], 3);
      const q = project(p); ctx.fillStyle = ["#ff8a75", "#7ed9a6", "#85baff"][i];
      ctx.font = "15px monospace"; ctx.textAlign = "center"; ctx.fillText(["X", "Y", "Z"][i], q[0], q[1] - 7);
    });
    ctx.textAlign = "left"; ctx.font = "13px monospace"; ctx.fillStyle = "#b5d2d8";
    ctx.fillText(`MuJoCo  /  seed ${state.info.seed}  /  ${top ? "TOP" : "PERSPECTIVE"}`, 22, 28);
    ctx.fillStyle = "#6d959e"; ctx.fillText("CARTESIAN PARALLEL GRIPPER", 22, h - 22);
  }

  function status(text, kind = "") { $("#gym-status").textContent = text; $("#gym-status").className = `pill ${kind}`; }
  function paint(state) {
    draw(state);
    $("#gym-steps").textContent = `${state.steps} / ${state.max_steps}`;
    $("#gym-reward").textContent = Number(state.reward || 0).toFixed(3);
    $("#gym-return").textContent = Number(state.return).toFixed(3);
    $("#gym-opening").textContent = `${(state.opening * 1000).toFixed(0)} mm`;
    $("#gym-frame-label").textContent = `${state.steps} / ${frames.length - 1}`;
    const info = state.info;
    $("#gym-feedback").textContent = info.is_success ? "成功：方块已释放并稳定落在盒内。" :
      state.done ? (info.reason === "time_limit" ? "已达到步数上限，请重置后重试。" : "方块离开工作区，请重置后重试。") :
      info.collision ? "夹爪接触桌面或盒壁，请调整位置。" : info.grasped ? "双指已接触方块，可以尝试抬升。" :
      "W / S 前后 · A / D 左右 · R / F 升降 · 空格开合夹爪";
    if (state.done) status(info.is_success ? "放置成功" : "回合结束", info.is_success ? "ok" : "bad");
  }
  function buttons() {
    $("#gym-play").textContent = running ? "暂停" : "开始控制";
    for (const id of ["play", "step", "demo", "grip"]) $("#gym-" + id).disabled = !live || live.done || resetting;
    $("#gym-export").disabled = !live || resetting;
    $("#gym-reset").disabled = $("#gym-new").disabled = resetting;
    controls.forEach((b) => { b.disabled = !live || live.done || resetting; });
    $("#gym-grip").textContent = `夹爪：${grip > 0 ? "张开" : "闭合"}`;
  }
  function stop() {
    running = false; clearTimeout(timer); held.clear(); controls.forEach((b) => b.classList.remove("on"));
    if (live && !live.done) status("已暂停"); buttons();
  }
  function action() {
    const a = [0, 0, 0, grip];
    held.forEach((key) => { const [i, d] = keys[key] || key.split(",").map(Number); a[i] += d; });
    return a.map((v) => Math.max(-1, Math.min(1, v)));
  }
  async function step() {
    if (!live || live.done || resetting || pending) return;
    const start = performance.now();
    pending = (async () => {
      try {
        const state = await api("/api/gym/step", { episode: live.episode, action: action(), demo });
        live = state; frames.push(state); paint(state);
        $("#gym-frame").max = $("#gym-frame").value = frames.length - 1;
        if (state.done) stop();
      } catch (e) { stop(); toast(e.message); }
    })();
    await pending; pending = null; buttons();
    if (running) timer = setTimeout(step, Math.max(0, 50 - (performance.now() - start)));
  }
  function start() {
    if (!live || live.done || resetting || running) return;
    running = true; status(demo ? "演示策略运行中" : "手动控制中"); buttons();
    // If a prior manual step is in flight, it schedules the next one on completion.
    if (!pending) step();
  }
  async function reset() {
    if (resetting) return false;
    stop(); if (pending) await pending;
    resetting = true; buttons(); status("正在重置…");
    try {
      const state = await api("/api/gym/reset", {
        seed: Number($("#gym-seed").value), max_steps: Number($("#gym-limit").value),
        twin: $("#gym-twin").value.trim(), jitter: Number($("#gym-jitter").value),
        randomize: $("#gym-randomize").checked, reward_mode: $("#gym-reward-mode").value
      });
      live = state; frames = [state]; demo = false; grip = 1;
      $("#gym-frame").disabled = false; $("#gym-frame").max = $("#gym-frame").value = 0;
      paint(state); status("准备就绪"); return true;
    } catch (e) { status("重置失败", "bad"); toast(e.message); return false; }
    finally { resetting = false; buttons(); }
  }
  $("#gym-config").addEventListener("submit", (e) => { e.preventDefault(); reset(); });
  $("#gym-new").addEventListener("click", () => { $("#gym-seed").value = (Number($("#gym-seed").value) + 1) % 2147483648; reset(); });
  $("#gym-play").addEventListener("click", () => { if (running) stop(); else { demo = false; start(); canvas.focus(); } });
  $("#gym-step").addEventListener("click", async () => { stop(); demo = false; if (pending) await pending; step(); });
  $("#gym-demo").addEventListener("click", async () => { if (await reset()) { demo = true; start(); } });
  $("#gym-camera").addEventListener("click", () => { top = !top; draw(frames[Number($("#gym-frame").value)] || live); });
  $("#gym-grip").addEventListener("click", () => { demo = false; grip *= -1; buttons(); start(); });
  $("#gym-frame").addEventListener("input", async () => {
    stop(); if (pending) await pending;
    paint(frames[Number($("#gym-frame").value)]);
    status("回放中（环境已暂停）");
  });
  $("#gym-export").addEventListener("click", async () => {
    stop(); if (pending) await pending;
    try {
      const data = await api("/api/gym/recording", { episode: live.episode });
      const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" }));
      const a = document.createElement("a"); a.href = url; a.download = `pick-place-${data.seed}-${data.episode}.json`; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { toast(e.message); }
  });
  controls.forEach((b) => {
    b.addEventListener("pointerdown", (e) => {
      e.preventDefault(); if (!live || live.done) return;
      b.setPointerCapture(e.pointerId); held.add(b.dataset.gymMove); b.classList.add("on"); demo = false; start();
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(type, () => { held.delete(b.dataset.gymMove); b.classList.remove("on"); });
  });
  canvas.addEventListener("keydown", (e) => {
    if (!(e.code in keys) && e.code !== "Space") return;
    e.preventDefault(); if (!live || live.done || e.repeat) return;
    demo = false;
    if (e.code === "Space") { grip *= -1; buttons(); } else held.add(e.code);
    start();
  });
  window.addEventListener("keyup", (e) => held.delete(e.code));
  window.addEventListener("blur", stop);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  draw(null); buttons();
  return { visibility(on) { visible = on; if (!visible) stop(); } };
};
