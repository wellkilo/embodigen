/* Virtual-world section: renders window.GAMEPLAY (scripts/build_gameplay_assets.py). */
(function () {
  const G = window.GAMEPLAY;
  if (!G) return;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
  const S = G.samples;

  /* hero stat */
  if ($("stat-real")) $("stat-real").textContent = `${S.summary.certified}/${S.summary.recordings}`;

  /* one real training row */
  const ex = S.example;
  if (ex && $("w-sample")) {
    const SOURCE = { d2e: "D2E", p2p: "Open-P2P", vpt: "VPT" };
    $("w-sample-src").textContent = `${ex.world} · ${SOURCE[ex.recording.split("-")[0]] || ""} · 回合 ${ex.start_turn}`;
    $("w-sample-img").textContent = ex.image.split("/").slice(1).join("/");
    const paint = (text) =>
      esc(text)
        .replace(/&lt;think&gt;|&lt;\/think&gt;/g, (m) => `<span class="tk-think">${m}</span>`)
        .replace(/&lt;\|action_(start|end)\|&gt;/g, (m) => `<span class="tk-act">${m}</span>`)
        .replace(/(-?\d+) ; (-?\d+) ; (-?\d+) ; /, '<span class="tk-mouse">$1 ; $2 ; $3</span> ; ');
    $("w-sample").innerHTML = ex.turns
      .map((t, i) => `<code><span class="tk-role">assistant · t+${i * 200} ms</span>\n${paint(t)}</code>`)
      .join("\n");
  }

  /* real public recordings */
  const SRC = { vpt: "VPT", d2e: "D2E", p2p: "P2P" };
  const ALIGN = { aligned: "对齐", inconclusive: "无法验证", skipped: "跳过", drift: "漂移", misaligned: "错位" };
  if ($("w-sources")) {
    $("w-real-sum").textContent =
      `${S.summary.certified}/${S.summary.recordings} 过闸 · ${(S.summary.hours_certified * 60).toFixed(1)} 分钟 · ` +
      Object.entries(S.summary.alignment || {}).map(([k, v]) => `${ALIGN[k] || k} ${v}`).join(" · ");
    const head = `<div class="w-row w-head" role="row"><span>录像</span><span>来源</span><span>世界</span><span>分钟</span><span>对齐</span></div>`;
    const rows = S.recordings
      .map((r) => {
        const id = r.recording.replace(/^(vpt|d2e|p2p)-/, "");
        const fix = r.realigned_ms ? ` <em>${r.realigned_ms > 0 ? "+" : ""}${r.realigned_ms.toFixed(0)} ms</em>` : "";
        return (
          `<div class="w-row" role="row"><span class="w-id" title="${esc(r.recording)}">${esc(id)}</span>` +
          `<span><i class="chip src-${r.source}">${SRC[r.source] || r.source}</i></span>` +
          `<span>${esc(r.world)}</span><span class="num">${r.minutes.toFixed(1)}</span>` +
          `<span class="al al-${r.alignment}">${ALIGN[r.alignment] || r.alignment}${fix}</span></div>`
        );
      })
      .join("");
    $("w-sources").innerHTML = head + rows;
  }

  /* mixtures at the largest budget */
  if ($("w-mix")) {
    const top = S.grid.filter((g) => g.budget_hours === S.top_budget);
    $("w-mix-budget").textContent = `${S.top_budget} h 档（${(S.top_budget * 60).toFixed(0)} 分钟画面）`;
    $("w-mix").innerHTML = top
      .map(
        (g) =>
          `<div class="mix-row"><span class="mix-name">${g.mixture}</span>` +
          `<div class="mix-bar"><span class="mix-anchor" style="width:${(g.anchor_share * 100).toFixed(2)}%"></span>` +
          `<span class="mix-rest" style="width:${((1 - g.anchor_share) * 100).toFixed(2)}%"></span></div>` +
          `<span class="mix-meta">锚点 ${pct(g.anchor_share)} · ${g.worlds} 个世界 · D ${g.hill_d.toFixed(2)}</span></div>`
      )
      .join("");
    $("w-mix-note").textContent =
      `锚点（青色）是 minecraft：VPT 与 D2E 的《我的世界》归为同一世界；其余为 D2E / P2P 的游戏。` +
      `D 是 Hill 多样性数 exp(−Σp·ln p)。各预算是同一优先顺序的前缀，S1–S5 共 ${S.grid.length} 个训练集。`;
  }

  /* injected lag on real recordings */
  if ($("w-align") && G.injection && G.injection.length) {
    const cases = G.injection[0].cases.map((c) => c.case);
    const head = `<div class="al-row al-head"><span></span>${cases.map((c) => `<span>${esc(c)}</span>`).join("")}</div>`;
    const rows = G.injection
      .map(
        (r) =>
          `<div class="al-row"><span class="al-src">${SRC[r.source] || r.source}</span>` +
          r.cases
            .map((c) => {
              if (c.status === "aligned" && c.realigned_ms) {
                const v = `${c.realigned_ms > 0 ? "+" : ""}${c.realigned_ms.toFixed(0)}`;
                return `<span class="al-cell ok"><b>${v}</b><small>已校正</small></span>`;
              }
              const label = { drift: "漂移", misaligned: "错位", inconclusive: "无法验证" }[c.status] || c.status;
              return `<span class="al-cell ${c.accepted ? "ok" : "bad"}"><b>${label}</b><small>${c.accepted ? "放行" : "拒绝"}</small></span>`;
            })
            .join("") +
          `</div>`
      )
      .join("");
    $("w-align").innerHTML = head + rows;
  }

  /* invariant checks */
  if ($("w-checks")) {
    const R = G.reference;
    const tick = (b) => (b ? "✓" : "✗");
    $("w-checks").innerHTML =
      `<span>budgets nested ${tick(S.checks.budgets_nested)}</span>` +
      `<span>train/val disjoint ${tick(S.checks.train_val_disjoint)}</span>` +
      `<span>rationale density spread ${pct(S.checks.rationale_density_spread, 2)}</span>` +
      `<span>参考世界 ${R.certified}/${R.recordings} 过闸：${R.rejected.map((x) => esc(x.reason)).join("、")} 被拒，${R.realigned} 条校正</span>`;
  }
})();
