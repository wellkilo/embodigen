/* EmbodiGen showcase — no dependencies, no build step. */

const METRICS = window.EMBODIGEN_METRICS || null;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
// ?capture renders the settled state immediately: used for README screenshots
// and for visual regression checks, where mid-transition frames are noise.
const CAPTURE = new URLSearchParams(location.search).has("capture");
if (CAPTURE) document.documentElement.classList.add("capture");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* ------------------------------------------------------------------
   1. The typographic field
   A faint grid of repeated words, parallaxed by the pointer. This is the
   one visual quote from the reference page: text as texture.
   ------------------------------------------------------------------ */

function initField() {
  const canvas = document.getElementById("field");
  if (!canvas) return;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let dpr = 1;
  let pointerX = 0;
  let pointerY = 0;
  let targetX = 0;
  let targetY = 0;
  let start = performance.now();

  const words = ["GROUND TRUTH", "LABEL", "DEPTH", "SEG", "POSE"];

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw(now) {
    const t = REDUCED ? 0 : (now - start) / 1000;
    pointerX += (targetX - pointerX) * 0.045;
    pointerY += (targetY - pointerY) * 0.045;

    ctx.clearRect(0, 0, width, height);

    const stepX = width < 760 ? 190 : 262;
    const stepY = width < 760 ? 74 : 96;
    const driftX = Math.sin(t * 0.08) * 16;
    const driftY = Math.cos(t * 0.06) * 10;
    const offsetX = pointerX * 18 + driftX;
    const offsetY = pointerY * 12 + driftY;

    ctx.font = "500 11px ui-monospace, 'SF Mono', Menlo, monospace";
    if ("letterSpacing" in ctx) ctx.letterSpacing = "3.2px";

    let row = 0;
    for (let y = -stepY; y < height + stepY; y += stepY) {
      const stagger = row % 2 === 0 ? 0 : stepX / 2;
      for (let x = -stepX; x < width + stepX; x += stepX) {
        const word = words[(row + Math.round(x / stepX)) % words.length];
        ctx.fillStyle =
          word === "GROUND TRUTH" ? "rgba(255,255,255,0.055)" : "rgba(255,255,255,0.026)";
        ctx.fillText(word, x + offsetX + stagger, y + offsetY);
      }
      row += 1;
    }

    // Fade the field out towards the edges so it reads as atmosphere.
    const cx = width / 2;
    const cy = height * 0.42;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(width, height) * 0.72);
    g.addColorStop(0, "rgba(8,9,12,0)");
    g.addColorStop(0.55, "rgba(8,9,12,0.35)");
    g.addColorStop(1, "rgba(8,9,12,0.98)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, width, height);

    if (!REDUCED) requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => {
    resize();
    if (REDUCED) draw(performance.now());
  });
  window.addEventListener(
    "pointermove",
    (e) => {
      targetX = (e.clientX / Math.max(1, width)) * 2 - 1;
      targetY = (e.clientY / Math.max(1, height)) * 2 - 1;
    },
    { passive: true }
  );

  resize();
  requestAnimationFrame(draw);
}

/* ------------------------------------------------------------------
   2. Pointer glow
   ------------------------------------------------------------------ */

function initGlow() {
  const glow = document.getElementById("glow");
  if (!glow || REDUCED) {
    if (glow) glow.style.display = "none";
    return;
  }
  let x = window.innerWidth / 2;
  let y = window.innerHeight * 0.35;
  let tx = x;
  let ty = y;

  window.addEventListener(
    "pointermove",
    (e) => {
      tx = e.clientX;
      ty = e.clientY;
    },
    { passive: true }
  );

  (function loop() {
    x += (tx - x) * 0.06;
    y += (ty - y) * 0.06;
    glow.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    requestAnimationFrame(loop);
  })();
}

/* ------------------------------------------------------------------
   3. Scroll reveal
   ------------------------------------------------------------------ */

function initReveal() {
  const items = [...document.querySelectorAll("[data-reveal]")];
  if (!items.length) return;

  if (REDUCED || CAPTURE) {
    items.forEach((el) => el.classList.add("is-in"));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const siblings = [...(el.parentElement?.children || [])].filter((n) =>
          n.hasAttribute("data-reveal")
        );
        const index = Math.max(0, siblings.indexOf(el));
        el.style.transitionDelay = `${Math.min(index, 5) * 80}ms`;
        el.classList.add("is-in");
        io.unobserve(el);
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -6% 0px" }
  );

  items.forEach((el) => io.observe(el));
}

/* ------------------------------------------------------------------
   4. Pipeline progress
   ------------------------------------------------------------------ */

function initPipeline() {
  const stages = [...document.querySelectorAll("[data-stage]")];
  const fill = document.getElementById("pipe-fill");
  const count = document.getElementById("pipe-count");
  if (!stages.length || !fill) return;

  let raf = 0;
  const update = () => {
    raf = 0;
    const line = window.innerHeight * 0.52;
    let active = 0;
    stages.forEach((stage, i) => {
      const rect = stage.getBoundingClientRect();
      if (rect.top + rect.height * 0.35 < line) active = i;
    });
    stages.forEach((stage, i) => stage.classList.toggle("is-active", i === active));
    fill.style.width = `${((active + 1) / stages.length) * 100}%`;
    if (count) count.textContent = String(active + 1).padStart(2, "0");
  };

  const onScroll = () => {
    if (!raf) raf = requestAnimationFrame(update);
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  update();
}

/* ------------------------------------------------------------------
   5. Frame film
   ------------------------------------------------------------------ */

function initFilm() {
  const el = document.getElementById("film");
  const badge = document.getElementById("film-badge");
  const counter = document.getElementById("film-count");
  const buttons = [...document.querySelectorAll(".switch button")];
  if (!el || !METRICS) return;

  const sheets = METRICS.sheets;
  const state = { key: "white", frame: 0, timer: 0, visible: false };

  const note = document.querySelector(".film-note");
  if (note && sheets.white) {
    note.textContent = `${sheets.white.size} px · ${sheets.white.frames} 帧 · 管线真实产出`;
  }

  function paint() {
    const sheet = sheets[state.key];
    const cols = sheet.cols;
    const frames = sheet.frames;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (!w || !h) return;
    const rows = Math.ceil(frames / cols);
    el.style.backgroundImage = `url("${sheet.src}")`;
    el.style.backgroundSize = `${cols * w}px ${rows * h}px`;
    const col = state.frame % cols;
    const row = Math.floor(state.frame / cols);
    el.style.backgroundPosition = `-${col * w}px -${row * h}px`;
    if (counter) {
      counter.textContent = `${String(state.frame + 1).padStart(2, "0")} / ${String(frames).padStart(2, "0")}`;
    }
  }

  function tick() {
    const frames = sheets[state.key].frames;
    state.frame = (state.frame + 1) % frames;
    paint();
  }

  function play() {
    stop();
    if (REDUCED) return;
    state.timer = window.setInterval(tick, 125);
  }

  function stop() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = 0;
  }

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.key = btn.dataset.sheet;
      state.frame = 0;
      buttons.forEach((b) => b.classList.toggle("on", b === btn));
      if (badge) badge.textContent = state.key === "white" ? "WHITE MODEL" : "PHOTOREAL";
      paint();
      if (state.visible) play();
    });
  });

  window.addEventListener("resize", paint);

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        state.visible = entry.isIntersecting;
        if (entry.isIntersecting) play();
        else stop();
      });
    },
    { threshold: 0.25 }
  );
  io.observe(el);

  requestAnimationFrame(paint);
}

/* ------------------------------------------------------------------
   6. Before / after slider
   ------------------------------------------------------------------ */

function initSlider() {
  const slider = document.getElementById("slider");
  const handle = document.getElementById("handle");
  if (!slider || !handle) return;

  let value = 52;

  const apply = (v) => {
    value = clamp(v, 0, 100);
    slider.style.setProperty("--x", value + "%");
    handle.setAttribute("aria-valuenow", String(Math.round(value)));
  };

  const fromEvent = (e) => {
    const rect = slider.getBoundingClientRect();
    return ((e.clientX - rect.left) / rect.width) * 100;
  };

  slider.addEventListener("pointerdown", (e) => {
    slider.setPointerCapture(e.pointerId);
    apply(fromEvent(e));
  });
  slider.addEventListener("pointermove", (e) => {
    if (e.buttons) apply(fromEvent(e));
  });
  slider.addEventListener("pointerup", (e) => {
    try {
      slider.releasePointerCapture(e.pointerId);
    } catch (_) {}
  });

  handle.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") apply(value - 2);
    else if (e.key === "ArrowRight") apply(value + 2);
    else if (e.key === "Home") apply(0);
    else if (e.key === "End") apply(100);
    else return;
    e.preventDefault();
  });

  apply(value);
}

/* ------------------------------------------------------------------
   7. Counters + error bars
   ------------------------------------------------------------------ */

function animateNumber(el) {
  const target = parseFloat(el.dataset.count || "0");
  const decimals = parseInt(el.dataset.decimals || "0", 10);
  if (REDUCED || CAPTURE) {
    el.textContent = target.toFixed(decimals);
    return;
  }
  const start = performance.now();
  const dur = 1100;
  const step = (now) => {
    const p = clamp((now - start) / dur, 0, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * eased).toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function renderBars(container, series) {
  if (!container || !series) return;
  const max = 30; // mm, shared scale for both panels
  series.forEach((value, i) => {
    const bar = document.createElement("span");
    bar.className = "bar";
    const h = value == null ? 0 : clamp((value / max) * 100, 3, 100);
    bar.style.height = h + "%";
    bar.style.transitionDelay = `${i * 22}ms`;
    container.appendChild(bar);
  });
}

function initVerdicts() {
  if (!METRICS) return;
  renderBars(document.getElementById("bars-bad"), METRICS.drifted?.per_frame_mm);
  renderBars(document.getElementById("bars-good"), METRICS.clean?.per_frame_mm);

  // Keep the hero and the verdict panels tied to the generated metrics so the
  // page can never quote a number the pipeline did not produce.
  const clean = METRICS.clean?.p95_mm;
  const drifted = METRICS.drifted?.p95_mm;
  const statP95 = document.getElementById("stat-p95");
  if (statP95 && clean != null) statP95.textContent = clean.toFixed(2);
  const goodNum = document.querySelector(".is-good [data-count]");
  if (goodNum && clean != null) goodNum.dataset.count = clean.toFixed(2);
  const badNum = document.querySelector(".is-bad [data-count]");
  if (badNum && drifted != null) badNum.dataset.count = drifted.toFixed(2);

  const targets = [...document.querySelectorAll("[data-count]")];
  if (!targets.length) return;

  if (REDUCED || CAPTURE) {
    targets.forEach((el) => animateNumber(el));
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        animateNumber(entry.target);
        io.unobserve(entry.target);
      });
    },
    { threshold: 0.4 }
  );
  targets.forEach((el) => io.observe(el));
}

/* ------------------------------------------------------------------
   8. Morphology cards
   ------------------------------------------------------------------ */

function initBodies() {
  const grid = document.getElementById("body-grid");
  if (!grid || !METRICS?.morphologies) return;

  grid.innerHTML = METRICS.morphologies
    .map(
      (m) => `
      <article class="body-card" data-reveal>
        <img src="${m.image}" alt="${m.name} 视角的渲染帧" />
        <div class="body-meta">
          <h3>${m.name}</h3>
          <p class="m-tag">${m.morphology} · ${m.frames} frames</p>
          <dl>
            <div><dt>ACTION</dt><dd>${m.action_dim}-D</dd></div>
            <div><dt>DOF</dt><dd>${m.dof}</dd></div>
            <div><dt>REACH</dt><dd>${m.reach_m} m</dd></div>
          </dl>
        </div>
      </article>`
    )
    .join("");
}

/* ------------------------------------------------------------------
   9. Copy button
   ------------------------------------------------------------------ */

function initCopy() {
  const btn = document.getElementById("copy");
  const cmd = document.getElementById("cmd");
  if (!btn || !cmd) return;

  btn.addEventListener("click", async () => {
    const text = cmd.textContent;
    const original = btn.textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "已复制";
    } catch (_) {
      const range = document.createRange();
      range.selectNodeContents(cmd);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      btn.textContent = "已选中";
    }
    window.setTimeout(() => (btn.textContent = original), 1600);
  });
}

/* ------------------------------------------------------------------ */

initField();
initGlow();
initBodies();
initReveal();
initPipeline();
initFilm();
initSlider();
initVerdicts();
initCopy();

if (CAPTURE && location.hash) {
  const target = document.querySelector(location.hash);
  if (target) target.scrollIntoView({ behavior: "auto", block: "start" });
}
