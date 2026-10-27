/* embodigen ui: forms -> /api/run, one job at a time, log polled from /api/jobs/<id>. */
(function () {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  /* token: from the launch link's #token=..., kept for this tab only */
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("token");
  if (fromHash) {
    try { sessionStorage.setItem("embodigen-token", fromHash); } catch (e) { /* private mode */ }
    history.replaceState(null, "", location.pathname);
  }
  let TOKEN = fromHash;
  try { TOKEN = TOKEN || sessionStorage.getItem("embodigen-token"); } catch (e) { /* ignore */ }

  async function api(path, body) {
    const opts = { headers: { "X-Token": TOKEN || "" } };
    if (body !== undefined) {
      opts.method = "POST";
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({ error: res.statusText }));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }
  const fileUrl = (rel) => `/files/${rel.split("/").map(encodeURIComponent).join("/")}?t=${encodeURIComponent(TOKEN)}`;

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : String(kid));
    return n;
  }

  let toastTimer;
  function toast(msg, kind = "err") {
    const t = $("#toast");
    t.textContent = msg;
    t.className = `toast ${kind}`;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), kind === "err" ? 8000 : 3000);
  }

  /* ------------------------------------------------------------------ tabs */
  function show(tab) {
    $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
    $$(".page").forEach((p) => p.classList.toggle("on", p.dataset.page === tab));
    if (tab === "data") loadDatasets();
    try { localStorage.setItem("embodigen-tab", tab); } catch (e) { /* ignore */ }
  }
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

  /* ------------------------------------------------------------------ state */
  let STATE = null;
  async function loadState() {
    STATE = await api("/api/state");
    const x = STATE.extras;
    $("#env").replaceChildren(
      el("span", { title: STATE.root }, STATE.root.split("/").slice(-2).join("/")),
      el("span", {}, `Python ${STATE.python}`),
      el("span", { class: x.video ? "ok" : "off" }, `视频解码 ${x.video ? "✓" : "✗"}`),
      el("span", { class: x.zstd ? "ok" : "off" }, `D2E ${x.zstd ? "✓" : "✗"}`),
      el("span", { class: STATE.llm.key_set ? "ok" : "off" }, `云端模型 ${STATE.llm.key_set ? "✓" : "未配置"}`)
    );
    $$("[data-need]").forEach((n) => (n.hidden = !!x[n.dataset.need]));
    fillSelect($("#world-select"), STATE.worlds.map((w) => [w.label, `${w.label}${w.engine === "raycast" ? "" : ` · ${w.engine}`}`]));
    fillSelect($("#study-select"), STATE.studies.map((s) => [s.path, `${s.name} · 预算 ${s.budgets.join("/")} h`]),
      "configs/studies/reference.toml");
    fillSelect($("#robot-select"), STATE.robots.map((r) => [r, r]));
    fillLists(STATE.datasets);
    renderLLM(STATE.llm);
    renderHistory(STATE.jobs);
    $$("form[data-action]").forEach(planFor);
    if (STATE.current) follow(STATE.current);
    else if (STATE.jobs.length && !JOB) follow(STATE.jobs[0].id);
  }

  function fillSelect(sel, pairs, preferred) {
    const keep = new Set($$("option:checked", sel).map((o) => o.value));
    if (!keep.size && preferred) keep.add(preferred);
    sel.replaceChildren(...pairs.map(([v, t]) => el("option", { value: v, selected: keep.has(v) }, t)));
  }
  function fillLists(ds) {
    const opts = (rows) => rows.map((r) => el("option", { value: r.path }));
    $("#dl-inputs").replaceChildren(...opts(ds.inputs));
    $("#dl-raw").replaceChildren(...opts(ds.raw));
    $("#dl-curated").replaceChildren(...opts(ds.curated));
  }

  /* ------------------------------------------------------------------ forms */
  function params(form) {
    const p = {};
    for (const n of $$("input[name], select[name]", form)) {
      if (n.type === "checkbox") p[n.name] = n.checked;
      else if (n.multiple) p[n.name] = $$("option:checked", n).map((o) => o.value);
      else p[n.name] = n.value;
    }
    if (p.worlds && !p.worlds.length) p.worlds = ["all"];
    return p;
  }
  const planTimers = new WeakMap();
  function planFor(form) {
    const box = $("[data-plan]", form);
    if (!box) return;
    clearTimeout(planTimers.get(form));
    planTimers.set(form, setTimeout(async () => {
      try {
        const r = await api("/api/plan", { action: form.dataset.action, params: params(form) });
        if (r.error) throw new Error(r.error);
        box.classList.remove("bad");
        box.replaceChildren(...r.steps.map((s) => el("code", {}, s)));
      } catch (e) {
        box.classList.add("bad");
        box.replaceChildren(el("span", {}, e.message));
      }
    }, 150));
  }
  $$("form[data-action]").forEach((form) => {
    form.addEventListener("input", () => planFor(form));
    form.addEventListener("change", () => planFor(form));
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const btn = $("button.run", form);
      btn.disabled = true;
      try {
        const job = await api("/api/run", { action: form.dataset.action, params: params(form) });
        follow(job.id, true);
      } catch (e) {
        toast(e.message);
      } finally {
        btn.disabled = false;
      }
    });
  });
  /* ingest: picking a fetched folder also picks its source */
  $("form[data-action=ingest] [name=input]").addEventListener("change", (ev) => {
    const hit = STATE && STATE.datasets.inputs.find((r) => r.path === ev.target.value);
    if (hit) $("form[data-action=ingest] [name=source]").value = hit.source;
  });

  /* ------------------------------------------------------------------ job panel */
  let JOB = null, SINCE = 0, POLL = null;
  const STATUS = { running: "运行中", ok: "完成", failed: "失败", stopped: "已停止" };

  function follow(id, reveal) {
    if (reveal && matchMedia("(max-width: 1100px)").matches) $("#job").scrollIntoView({ behavior: "smooth" });
    if (JOB !== id) {
      JOB = id;
      SINCE = 0;
      $("#job-log").textContent = "";
      $("#job-art").replaceChildren();
    }
    clearTimeout(POLL);
    poll();
  }
  async function poll() {
    let j;
    try {
      j = await api(`/api/jobs/${JOB}?since=${SINCE}`);
    } catch (e) {
      POLL = setTimeout(poll, 3000);
      return;
    }
    const log = $("#job-log");
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
    if (j.skipped) log.append(`… 省略 ${j.skipped} 行 …\n`);
    if (j.lines.length) log.append(j.lines.join("\n") + "\n");
    SINCE = j.next;
    if (atEnd) log.scrollTop = log.scrollHeight;

    $("#job-title").textContent = j.title;
    const pill = $("#job-status");
    pill.textContent = STATUS[j.status] + (j.status === "running" && j.steps.length > 1 ? ` ${j.step + 1}/${j.steps.length}` : "");
    pill.className = `pill ${j.status}`;
    $("#job-stop").hidden = j.status !== "running";
    $("#job-steps").replaceChildren(...j.steps.map((s, i) =>
      el("li", { class: i < j.step || (j.status === "ok") ? "done" : i === j.step ? j.status : "" }, el("code", {}, s))));
    if (j.status === "running") {
      POLL = setTimeout(poll, 700);
    } else {
      if (j.status === "ok") j.artifacts.forEach((a) => $("#job-art").replaceChildren(el("img", { src: fileUrl(a) + `&v=${j.ended}`, alt: a })));
      if (!pollDoneFor.has(j.id)) {
        pollDoneFor.add(j.id);
        loadState().catch(() => {});
        if ($(".page.on").dataset.page === "data") loadDatasets();
      }
    }
  }
  const pollDoneFor = new Set();
  $("#job-stop").addEventListener("click", async () => {
    try { await api(`/api/jobs/${JOB}/stop`, {}); } catch (e) { toast(e.message); }
  });
  function renderHistory(jobs) {
    $("#job-history").replaceChildren(...jobs.map((j) =>
      el("li", {}, el("button", { class: "link", onclick: () => follow(j.id) },
        el("span", { class: `pill ${j.status}` }, STATUS[j.status]), ` ${j.title}`))));
  }

  /* ------------------------------------------------------------------ data */
  async function loadDatasets() {
    let ds;
    try { ds = await api("/api/datasets"); } catch (e) { toast(e.message); return; }
    if (STATE) STATE.datasets = ds;
    fillLists(ds);
    const box = $("#datasets");
    const group = (title, rows, line, open) =>
      rows.length ? el("div", { class: "group" }, el("h2", {}, title),
        el("ul", { class: "dsets" }, rows.map((r) => el("li", {}, el("button", { class: "ds", onclick: () => open(r) },
          el("b", {}, r.path), el("span", {}, line(r))))))) : null;
    const any = Object.values(ds).some((r) => r.length);
    box.replaceChildren(...[
      any ? null : el("p", { class: "empty" }, "还没有数据。去「快速开始」跑一次参考世界试试。"),
      group("训练集", ds.mixes, (r) => `${r.study} · ${r.datasets} 个训练集 · 标注 ${r.annotator}`, (r) => openLedger(r.path)),
      group("筛选结果", ds.curated, (r) => `${r.certified}/${r.recordings} 通过 · ${r.minutes} 分钟`, (r) => openLedger(r.path)),
      group("原始录像", ds.raw, (r) => `${r.recordings} 段录像`, (r) => openRecordings(r.path)),
      group("下载的公开数据", ds.inputs, (r) => `来源 ${r.source}，可在「分步操作」导入`, (r) => {
        show("steps");
        $("form[data-action=ingest] [name=input]").value = r.path;
        $("form[data-action=ingest] [name=source]").value = r.source;
        planFor($("form[data-action=ingest]"));
      }),
      group("机器人数据", ds.robot, (r) => `${r.episodes} 个通过的回合` + (r.rejected ? ` · ${r.rejected} 个被拒` : ""), (r) => openRobot(r.path)),
    ].filter(Boolean));
  }
  $("#rescan").addEventListener("click", loadDatasets);

  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const ALIGN = { aligned: "对齐", inconclusive: "无法验证", skipped: "跳过", drift: "漂移", misaligned: "错位", "": "-" };

  function table(head, rows, onRow) {
    return el("div", { class: "tbl" }, el("table", {},
      el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((r, i) => el("tr", onRow ? { class: "click", tabindex: "0", onclick: () => onRow(i),
        onkeydown: (e) => { if (e.key === "Enter") onRow(i); } } : {}, r.map((c) => el("td", {}, c)))))));
  }

  async function openLedger(path) {
    let r;
    try { r = await api(`/api/ledger?path=${encodeURIComponent(path)}`); } catch (e) { toast(e.message); return; }
    const L = r.ledger;
    const box = $("#detail");
    if (L.stage === "curate") {
      const s = L.summary;
      box.replaceChildren(
        el("h2", {}, `筛选结果 · ${r.path}`),
        el("div", { class: "kpis" },
          kpi(`${s.certified}/${s.recordings}`, "通过闸门"), kpi(`${(s.hours_certified * 60).toFixed(1)}`, "可用分钟"),
          kpi(String(s.realigned), "自动校正"),
          s.alignment ? kpi(Object.entries(s.alignment).map(([k, v]) => `${ALIGN[k] || k} ${v}`).join(" · "), "对齐检查") : null),
        table(["录像", "世界", "来源", "分钟", "对齐", "结论"], r.recordings.map((x) => [
          el("span", { class: "mono", title: x.id }, x.id), x.world, x.source.split(":")[0], x.minutes.toFixed(1),
          (ALIGN[x.alignment] || x.alignment) + (x.realigned_ms ? ` ${x.realigned_ms > 0 ? "+" : ""}${x.realigned_ms.toFixed(0)} ms` : ""),
          x.passed ? el("span", { class: "ok" }, "通过") : el("span", { class: "bad", title: x.reasons.join("; ") }, "拒绝：" + x.reasons.map((t) => t.split(":")[0]).join("、")),
        ])),
        el("h3", {}, "各世界"),
        table(["世界", "录像", "通过", "分钟", "鼠标截断"], Object.entries(L.worlds).map(([w, v]) => [
          w, v.recordings, v.certified, (v.hours_certified * 60).toFixed(1), pct(v.mouse_clipped_frac || 0, 1)]))
      );
    } else {
      const c = L.checks;
      box.replaceChildren(
        el("h2", {}, `训练集 · ${L.study}`),
        el("p", { class: "hint" }, `${r.path} · 配置 ${L.config_hash} · 标注 ${L.annotator}`),
        el("div", { class: "kpis" },
          kpi(`${L.pool.windows}`, "候选窗口"), kpi(`${(L.pool.footage_hours * 60).toFixed(1)}`, "素材分钟"),
          kpi(`${L.pool.diversity.worlds}`, "世界数"), kpi(L.pool.diversity.hill_d.toFixed(2), "Hill D")),
        el("p", { class: "checks" },
          chk(c.budgets_nested, "预算嵌套"), chk(c.train_val_disjoint, "训练/验证不重叠"),
          el("span", {}, `理由密度差 ${pct(c.rationale_density_spread, 2)}`)),
        table(["混合", "预算 h", "素材 h", "样本", "世界", "Hill D", "锚点", "理由", "缺口 h"], L.grid.map((g) => [
          g.mixture, g.budget_hours, g.footage_hours.toFixed(3), g.samples, g.worlds, g.hill_d.toFixed(2),
          bar(g.anchor_share), pct(g.rationale_density, 1), g.shortfall_hours ? el("span", { class: "bad" }, g.shortfall_hours.toFixed(3)) : "0"])),
        ...(L.notes || []).map((n) => el("p", { class: "hint" }, n))
      );
    }
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  const kpi = (v, label, cls) => el("div", { class: "kpi" }, el("b", cls ? { class: cls } : {}, v), el("span", {}, label));
  const chk = (ok, label) => el("span", { class: ok ? "ok" : "bad" }, `${ok ? "✓" : "✗"} ${label}`);
  function bar(x) {
    const fill = el("i");
    fill.style.width = pct(x, 1);
    return el("span", { class: "bar", title: pct(x, 1) }, el("span", { class: "track" }, fill), pct(x, 1));
  }

  async function openRecordings(dir) {
    let rows;
    try { rows = await api(`/api/recordings?dir=${encodeURIComponent(dir)}`); } catch (e) { toast(e.message); return; }
    const box = $("#detail");
    const view = el("div", { class: "preview" });
    box.replaceChildren(
      el("h2", {}, `原始录像 · ${dir}`),
      el("p", { class: "hint" }, "点「预览」生成一段 GIF（默认前 60 帧，约 7.5 秒）。"),
      view,
      table(["录像", "世界", "来源", "分钟", ""], rows.map((x) => [
        el("span", { class: "mono", title: x.path }, x.id), x.world, x.source.split(":")[0], x.minutes.toFixed(1),
        el("span", { class: "acts" },
          x.preview ? el("button", { class: "link", onclick: () => showGif(view, x.preview, x.id) }, "查看") : null,
          el("button", { class: "link", onclick: () => runPreview(x, view) }, x.preview ? "重新生成" : "预览")),
      ]))
    );
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function showGif(view, rel, id) {
    view.replaceChildren(el("figure", {}, el("img", { src: fileUrl(rel) + `&v=${Date.now()}`, alt: id }), el("figcaption", {}, id)));
  }
  async function runPreview(x) {
    try {
      const job = await api("/api/run", { action: "preview", params: { recording: x.path } });
      follow(job.id, true);
    } catch (e) { toast(e.message); }
  }

  /* ------------------------------------------------------------------ robot episodes */
  const fmtBytes = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} KB` : `${n} B`);
  const getText = (rel) => api(`/api/text?path=${encodeURIComponent(rel)}`);

  async function openRobot(dir) {
    let r;
    try { r = await api(`/api/robot?path=${encodeURIComponent(dir)}`); } catch (e) { toast(e.message); return; }
    const s = r.index.summary || {};
    const view = el("div", { class: "episode" });
    $("#detail").replaceChildren(
      el("h2", {}, `机器人数据 · ${r.path}`),
      el("div", { class: "kpis" },
        kpi(`${s.certified}/${s.total}`, "通过保真度闸门"), kpi(pct(s.yield || 0), "良率"),
        kpi(`${r.index.threshold_mm} mm`, "闸门阈值"), kpi(r.index.config_hash || "-", "配置哈希", "mono")),
      el("p", { class: "hint" }, "点一行查看回合：逐帧画面、深度、分割，以及每个文件的原始内容。"),
      table(["回合", "任务", "机器人", "帧", "p95 mm", "max mm", "尝试", "结论"], r.episodes.map((x) => [
        el("span", { class: "mono" }, x.episode_id), x.instruction, x.robot, x.frames, x.p95_error_mm.toFixed(2),
        x.max_error_mm.toFixed(2), x.attempts, x.passed ? el("span", { class: "ok" }, "通过") : el("span", { class: "bad" }, "拒绝")]),
        (i) => r.episodes[i].path ? openEpisode(r.episodes[i], view) : toast("找不到这个回合的目录")),
      el("details", { class: "raw" }, el("summary", {}, `${r.index_file} 原文`), el("pre", { class: "src" }, r.index_text)),
      view
    );
    $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function openEpisode(ep, view) {
    let tree, info, rows, fid, scene;
    try {
      tree = await api(`/api/tree?path=${encodeURIComponent(ep.path)}`);
      const has = (n) => tree.files.find((f) => f.name === n);
      const dataFile = tree.files.find((f) => /^data\/.*\.jsonl$/.test(f.name));
      [info, rows, fid, scene] = await Promise.all([
        has("meta/info.json") ? getText(has("meta/info.json").path).then((t) => JSON.parse(t.text)) : {},
        dataFile ? getText(dataFile.path) : { text: "" },
        has("meta/fidelity.json") ? getText(has("meta/fidelity.json").path).then((t) => JSON.parse(t.text)) : {},
        has("meta/scene.json") ? getText(has("meta/scene.json").path).then((t) => JSON.parse(t.text)) : {},
      ]);
    } catch (e) { toast(e.message); return; }
    const lines = rows.text.split("\n").filter((l) => l.trim());
    const frames = lines.map((l) => JSON.parse(l));
    const [h, w] = ((info.features || {})["observation.depth"] || {}).shape || [0, 0];
    const fps = info.fps || 8;
    const far = 0.999 * (((scene.camera || {}).far) || Infinity);  // rays that hit nothing read the far plane
    const base = ep.path;

    /* frame player: the four channels the pack writes, from the files themselves */
    const imgW = el("img", { alt: "white model" }), imgP = el("img", { alt: "photoreal" });
    const cvD = el("canvas", { width: w, height: h }), cvS = el("canvas", { width: w, height: h });
    const panel = (node, label, note) => el("figure", { class: "chan" }, node, el("figcaption", {}, el("b", {}, label), " ", note));
    const slider = el("input", { type: "range", min: 0, max: Math.max(frames.length - 1, 0), value: 0, "aria-label": "帧" });
    const play = el("button", { class: "ghost small" }, "播放");
    const meta = el("div", { class: "frame-meta" });
    const line = el("pre", { class: "src one" });
    let timer = null;
    const cache = new Map();
    async function buf(rel) {
      if (!cache.has(rel)) cache.set(rel, fetch(fileUrl(rel)).then((r) => r.arrayBuffer()));
      return cache.get(rel);
    }
    function paint(cv, n, color) {
      const ctx = cv.getContext("2d"), img = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) img.data.set([...color(i), 255], i * 4);
      ctx.putImageData(img, 0, 0);
    }
    async function show(i) {
      const f = frames[i];
      if (!f) return;
      slider.value = i;
      const at = (k) => `${base}/${f.frames[k]}`;
      imgW.src = fileUrl(at("white_model"));
      imgP.src = fileUrl(at("photoreal"));
      const [d, sg] = await Promise.all([buf(at("depth")), buf(at("segmentation"))]);
      if (+slider.value !== i) return;
      const dv = new DataView(d), sv = new DataView(sg);
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < w * h; k++) {
        const z = dv.getFloat32(k * 4, true);
        if (Number.isFinite(z) && z > 0 && z < far) { lo = Math.min(lo, z); hi = Math.max(hi, z); }
      }
      paint(cvD, w * h, (k) => {
        const z = dv.getFloat32(k * 4, true);
        if (!(z > 0 && z < far)) return [5, 19, 23];
        const t = 1 - (z - lo) / Math.max(hi - lo, 1e-6);
        return [15 + 200 * t * t, 40 + 177 * t, 50 + 160 * t];
      });
      paint(cvS, w * h, (k) => {
        const id = sv.getUint16(k * 2, true);
        if (!id) return [5, 19, 23];
        const x = Math.imul(id, 2654435761) >>> 0;
        return [64 + (x & 191), 64 + ((x >> 8) & 191), 64 + ((x >> 16) & 191)];
      });
      const err = (fid.per_frame_mm || [])[i];
      meta.replaceChildren(
        el("b", {}, `第 ${i + 1}/${frames.length} 帧`), ` · t = ${f.timestamp}s · 阶段 ${f.phase}`,
        err != null ? el("span", { class: err <= (fid.threshold_mm || 12) ? "ok" : "bad" }, ` · 误差 ${err.toFixed(2)} mm`) : "",
        ` · 深度 ${lo === Infinity ? "-" : `${lo.toFixed(2)}–${hi.toFixed(2)} m`}`);
      line.textContent = lines[i];
    }
    slider.addEventListener("input", () => show(+slider.value));
    play.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = "播放"; return; }
      play.textContent = "暂停";
      timer = setInterval(() => {
        if (!view.isConnected) { clearInterval(timer); return; }
        show((+slider.value + 1) % frames.length);
      }, 1000 / fps);
    });

    /* file browser: every file under the episode, shown as written */
    const viewer = el("div", { class: "viewer" }, el("p", { class: "hint" }, "选一个文件查看原始内容。"));
    async function openFile(f) {
      $$(".files .on", view).forEach((n) => n.classList.remove("on"));
      const btn = $(`.files [data-path="${CSS.escape(f.path)}"]`, view);
      if (btn) btn.classList.add("on");
      let t;
      try { t = await getText(f.path); } catch (e) { toast(e.message); return; }
      const head = el("p", { class: "hint" }, `${t.path} · ${fmtBytes(t.bytes)}` +
        (t.kind === "binary" ? ` · 二进制，下面是前 ${Math.min(512, t.bytes)} 字节的十六进制` : "") +
        (t.lines != null ? ` · ${t.lines} 行` : "") + (t.truncated && t.kind === "text" ? " · 只显示前 256 KB" : ""));
      const kids = [head];
      if (/\.(png|gif|jpe?g)$/i.test(f.name)) kids.push(el("img", { class: "pix", src: fileUrl(f.path), alt: f.name }));
      kids.push(el("pre", { class: t.kind === "text" ? "src" : "src hex" }, t.kind === "text" ? t.text : t.hex));
      viewer.replaceChildren(...kids);
    }
    const groups = new Map();
    for (const f of tree.files) {
      const dir = f.name.includes("/") ? f.name.slice(0, f.name.lastIndexOf("/")) : ".";
      if (!groups.has(dir)) groups.set(dir, []);
      groups.get(dir).push(f);
    }
    const fileBtn = (f) => el("button", { class: "file", "data-path": f.path, onclick: () => openFile(f) },
      el("span", {}, f.name.split("/").pop()), el("small", {}, fmtBytes(f.bytes)));
    const files = el("div", { class: "files" }, [...groups].map(([dir, fs]) => {
      const total = fs.reduce((a, f) => a + f.bytes, 0);
      return el("details", { open: fs.length <= 8 }, el("summary", {}, `${dir}/`, el("small", {}, ` ${fs.length} 个 · ${fmtBytes(total)}`)),
        el("div", { class: "file-list" }, fs.map(fileBtn)));
    }));

    view.replaceChildren(
      el("h3", {}, `${ep.episode_id} · ${ep.instruction}`),
      frames.length && w ? el("div", { class: "player" },
        el("div", { class: "chans" },
          panel(imgW, "白模", "frames/white/*.png"), panel(imgP, "写实", "frames/photoreal/*.png"),
          panel(cvD, "深度", "frames/depth/*.f32"), panel(cvS, "分割", "frames/seg/*.u16")),
        el("div", { class: "scrub" }, play, slider), meta,
        el("p", { class: "hint" }, "这一帧在 data/…/episode_*.jsonl 里的原始一行："), line) :
        el("p", { class: "hint" }, "这个回合没有逐帧数据。"),
      el("h3", {}, "文件"),
      el("div", { class: "browser" }, files, viewer)
    );
    if (frames.length && w) show(0);
    const first = tree.files.find((f) => f.name === "meta/fidelity.json") || tree.files[0];
    if (first) openFile(first);
    view.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ------------------------------------------------------------------ model */
  function renderLLM(s) {
    const sel = $("#provider-select");
    if (!sel.options.length) {
      sel.replaceChildren(...Object.keys(s.providers).map((p) => el("option", { value: p }, p)));
    }
    const f = $("#llm-form");
    if (!f.dataset.touched) {
      sel.value = s.provider || "dashscope";
      f.model.value = s.model || "";
      f.base_url.value = sel.value === "custom" ? s.base_url : "";
      f.images.checked = s.images;
    }
    $("[data-custom]", f).hidden = sel.value !== "custom";
    $("#key-input").placeholder = s.key_set ? "已保存（不会显示）。留空 = 保留原 key" : "粘贴 API key";
    const env = s.providers[sel.value] ? s.providers[sel.value].key_env : s.key_env;
    $("#key-note").textContent =
      `key 写入 ${s.config_path}（仅本人可读），不会出现在页面、日志、账本里。也可以用环境变量 ${env} 提供。`;
    $("#model-dot").className = `dot ${s.key_set && !s.problems.length ? "ok" : "off"}`;
    const st = $("#llm-status");
    if (!s.problems.length) {
      st.className = "status ok";
      st.replaceChildren(el("b", {}, "已就绪"), ` ${s.provider || "custom"} · ${s.model} · key 来自 ${s.key_source}`);
    } else {
      st.className = "status off";
      st.replaceChildren(el("b", {}, "未就绪"), el("ul", {}, s.problems.map((p) => el("li", {}, p))));
    }
  }
  $("#llm-form").addEventListener("input", (ev) => {
    ev.currentTarget.dataset.touched = "1";
    if (STATE) renderLLM(STATE.llm);
  });
  $("#llm-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.currentTarget;
    try {
      const s = await api("/api/llm", {
        provider: f.provider.value, model: f.model.value.trim(), base_url: f.base_url.value.trim(),
        images: f.images.checked, api_key: f.api_key.value,
      });
      f.api_key.value = "";
      delete f.dataset.touched;
      STATE.llm = s;
      renderLLM(s);
      loadState().catch(() => {});
      toast("已保存", "info");
    } catch (e) { toast(e.message); }
  });
  $("#llm-ping").addEventListener("click", async () => {
    try {
      const job = await api("/api/run", { action: "llm-check", params: {} });
      follow(job.id, true);
    } catch (e) { toast(e.message); }
  });

  /* ------------------------------------------------------------------ boot */
  let start = "start";
  try { start = localStorage.getItem("embodigen-tab") || start; } catch (e) { /* ignore */ }
  show(start);
  if (!TOKEN) {
    toast("缺少访问令牌：请用 embodigen ui 启动时打印的完整链接打开（含 #token=…）");
  } else {
    loadState().catch((e) => toast(e.message));
  }
})();
