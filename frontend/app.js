/* Weather Whiplash frontend.
   Vanilla JS on purpose: no build step, no CDN, no lockfile. `uvicorn app.main:app`
   serves this directly, so there is exactly one command between a judge and a
   running demo. The trend chart is hand-drawn on a canvas for the same reason. */

const $ = (id) => document.getElementById(id);

const state = {
  sessionId: (crypto.randomUUID && crypto.randomUUID()) || String(Math.random()).slice(2),
  steps: [],      // one entry per frame: {url, frame, trend, recommendation}
  cursor: -1,
  colors: { Dry: "#22c55e", Damp: "#eab308", Drying: "#38bdf8", Wet: "#ef4444" },
  classes: ["Dry", "Damp", "Drying", "Wet"],
  wetness: { Dry: 0, Damp: 1, Drying: 1.5, Wet: 2 },
  window: 12,
  playing: null,
};

/* ---------------------------------------------------------------- health */

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (h.class_colors) state.colors = h.class_colors;
    if (h.classes) state.classes = h.classes;
    if (h.wetness_scale) state.wetness = h.wetness_scale;
    if (h.window) state.window = h.window;

    const chip = $("backendChip");
    const finetuned = h.backend === "fine-tuned";
    chip.textContent = finetuned ? "fine-tuned ViT" : "CLIP zero-shot (fallback)";
    chip.className = "status-chip " + (finetuned ? "finetuned" : "fallback");
    chip.title = h.model || "";
    $("deviceChip").textContent = h.device || "";
    $("pulse").classList.add("live");
    renderLegend();
  } catch {
    $("backendChip").textContent = "backend offline";
    $("backendChip").className = "status-chip fallback";
  }
}

/* ---------------------------------------------------------------- upload */

async function send(files) {
  if (!files.length) return;
  const fd = new FormData();
  const hint = $("weather").value.trim();
  if (hint) fd.append("weather_hint", hint);
  fd.append("session_id", state.sessionId);

  const single = files.length === 1;
  if (single) {
    // One frame appends to the running session, so trends build up as you go.
    fd.append("file", files[0]);
  } else {
    // A burst is a self-contained sequence: reset, then replay it.
    for (const f of files) fd.append("files", f);
    fd.append("reset", "true");
    state.steps = [];
  }

  setBusy(true);
  try {
    const res = await fetch(single ? "/api/predict" : "/api/sequence", {
      method: "POST",
      body: fd,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");

    if (single) {
      pushStep(URL.createObjectURL(files[0]), data);
    } else {
      data.steps.forEach((step, i) => pushStep(URL.createObjectURL(files[i]), step));
      play();
    }
  } catch (err) {
    toast(err.message || "Could not reach the backend");
  } finally {
    setBusy(false);
  }
}

function pushStep(url, data) {
  state.steps.push({
    url,
    frame: data.frame,
    trend: data.trend,
    recommendation: data.recommendation,
  });
  renderFilmstrip();
  show(state.steps.length - 1);
  $("replay").disabled = state.steps.length < 2;
}

/* ---------------------------------------------------------------- render */

function show(index) {
  const step = state.steps[index];
  if (!step) return;
  state.cursor = index;

  $("empty").hidden = true;
  const img = $("frame");
  img.src = step.url;
  img.hidden = false;

  const trend = step.trend;
  const rec = step.recommendation;

  // condition badge
  const cond = $("condition");
  cond.textContent = trend.current_class;
  cond.dataset.c = trend.current_class;

  const pct = Math.round(trend.current_confidence * 100);
  $("confidenceFill").style.width = pct + "%";
  $("confidenceNum").textContent = pct + "%";
  $("confidenceFill").parentElement.style.color = state.colors[trend.current_class];

  // trend chip
  const chip = $("trendChip");
  const map = {
    DRYING: ["drying", "drying ↓"],
    WETTING: ["wetting", "wetting ↑"],
    STABLE: ["stable", "stable"],
    INSUFFICIENT_DATA: ["", `collecting ${trend.frames}/${state.window}`],
  };
  const [cls, text] = map[trend.trend] || ["", trend.trend];
  chip.className = "chip " + cls;
  chip.textContent = text;

  const tchip = $("transitionChip");
  tchip.hidden = !trend.transition;
  if (trend.transition) tchip.textContent = trend.transition.replace("->", " → ");

  // strategy call
  $("recMessage").textContent = rec.message;
  $("recTire").textContent = rec.tire;
  $("recAction").textContent = rec.action;
  $("recCard").dataset.u = rec.urgency;

  $("chartMeta").textContent =
    `${trend.frames} frame${trend.frames === 1 ? "" : "s"} · slope ${trend.slope > 0 ? "+" : ""}${trend.slope.toFixed(3)}`;

  renderProbs(step.frame.probs);
  renderFilmstrip();
  drawChart(trend);
}

function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";
  state.steps.forEach((step, i) => {
    const img = document.createElement("img");
    img.src = step.url;
    img.className = i === state.cursor ? "active" : "";
    img.title = `Frame ${i + 1}: ${step.frame.label} ${(step.frame.confidence * 100).toFixed(0)}%`;
    img.onclick = () => { stop(); show(i); };
    strip.appendChild(img);
  });
  const active = strip.querySelector(".active");
  if (active) active.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function renderProbs(probs) {
  const box = $("probs");
  box.innerHTML = "";
  state.classes.forEach((cls) => {
    const p = probs[cls] || 0;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML =
      `<span class="name">${cls}</span>` +
      `<span class="track"><span class="fill"></span></span>` +
      `<span class="val">${(p * 100).toFixed(0)}%</span>`;
    box.appendChild(row);
    const fill = row.querySelector(".fill");
    fill.style.background = state.colors[cls];
    requestAnimationFrame(() => { fill.style.width = (p * 100) + "%"; });
  });
}

function renderLegend() {
  $("legend").innerHTML = state.classes
    .map((c) => `<span><i style="background:${state.colors[c]}"></i>${c}</span>`)
    .join("");
}

/* ---------------------------------------------------------------- chart */

function drawChart(trend) {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 720;
  const h = 240;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 52, padR = 12, padT = 12, padB = 22;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Y axis is the wetness scale, wet at the top: the line falling means the
  // track is improving, which is the reading a pit wall wants at a glance.
  const yFor = (v) => padT + plotH * (1 - v / 2);

  ctx.strokeStyle = "#262b36";
  ctx.fillStyle = "#8b93a6";
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.lineWidth = 1;

  const ticks = [["Wet", 2], ["Drying", 1.5], ["Damp", 1], ["Dry", 0]];
  ticks.forEach(([name, v]) => {
    const y = yFor(v);
    ctx.beginPath();
    ctx.moveTo(padL, y + 0.5);
    ctx.lineTo(w - padR, y + 0.5);
    ctx.stroke();
    ctx.fillText(name, 8, y + 4);
  });

  const history = trend.wetness_history || [];
  if (!history.length) return;

  const n = Math.max(history.length, 2);
  const xFor = (i) => padL + (plotW * i) / (n - 1);

  // area under the curve
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, "rgba(56,189,248,.22)");
  grad.addColorStop(1, "rgba(56,189,248,0)");
  ctx.beginPath();
  ctx.moveTo(xFor(0), padT + plotH);
  history.forEach((v, i) => ctx.lineTo(xFor(i), yFor(v)));
  ctx.lineTo(xFor(history.length - 1), padT + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // the line itself
  ctx.beginPath();
  history.forEach((v, i) => (i ? ctx.lineTo(xFor(i), yFor(v)) : ctx.moveTo(xFor(i), yFor(v))));
  ctx.strokeStyle = "#7dd3fc";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // one dot per frame, coloured by that frame's predicted class
  const classes = trend.class_history || [];
  history.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(xFor(i), yFor(v), i === history.length - 1 ? 5 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = state.colors[classes[i]] || "#7dd3fc";
    ctx.fill();
    if (i === history.length - 1) {
      ctx.strokeStyle = "#0b0d12";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
}

/* ---------------------------------------------------------------- replay */

function play() {
  stop();
  if (state.steps.length < 2) return;
  let i = 0;
  show(0);
  state.playing = setInterval(() => {
    i += 1;
    if (i >= state.steps.length) return stop();
    show(i);
  }, 750);
}

function stop() {
  if (state.playing) clearInterval(state.playing);
  state.playing = null;
}

/* ---------------------------------------------------------------- demo */

async function loadDemo() {
  setBusy(true);
  try {
    const res = await fetch("/api/demo-sequence");
    if (!res.ok) throw new Error("No demo sequence bundled");
    const { files } = await res.json();
    const blobs = await Promise.all(
      files.map(async (f) => {
        const r = await fetch(f);
        const b = await r.blob();
        return new File([b], f.split("/").pop(), { type: b.type });
      })
    );
    await send(blobs);
  } catch (err) {
    toast(err.message || "Demo sequence unavailable");
  } finally {
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- misc */

function setBusy(busy) {
  ["pick", "pick2", "replay", "reset", "demo"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = busy || (id === "replay" && state.steps.length < 2);
  });
}

let toastTimer = null;
function toast(message) {
  document.querySelector(".toast")?.remove();
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  document.body.appendChild(el);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.remove(), 4200);
}

async function resetSession() {
  stop();
  const fd = new FormData();
  fd.append("session_id", state.sessionId);
  try { await fetch("/api/session/reset", { method: "POST", body: fd }); } catch {}
  state.steps = [];
  state.cursor = -1;
  $("frame").hidden = true;
  $("empty").hidden = false;
  $("filmstrip").innerHTML = "";
  $("probs").innerHTML = "";
  $("condition").textContent = "—";
  $("condition").removeAttribute("data-c");
  $("confidenceFill").style.width = "0%";
  $("confidenceNum").textContent = "—";
  $("trendChip").className = "chip";
  $("trendChip").textContent = "awaiting frames";
  $("transitionChip").hidden = true;
  $("recMessage").textContent = "Upload frames to begin.";
  $("recTire").textContent = "";
  $("recAction").textContent = "";
  $("recCard").dataset.u = "0";
  $("chartMeta").textContent = "";
  $("replay").disabled = true;
  drawChart({ wetness_history: [], class_history: [] });
}

/* ---------------------------------------------------------------- wiring */

$("pick").onclick = () => $("file").click();
$("pick2").onclick = () => $("file").click();
$("demo").onclick = loadDemo;
$("replay").onclick = play;
$("reset").onclick = resetSession;
$("file").onchange = (e) => {
  const files = Array.from(e.target.files);
  e.target.value = "";
  send(files);
};

const wrap = $("frameWrap");
["dragenter", "dragover"].forEach((ev) =>
  wrap.addEventListener(ev, (e) => { e.preventDefault(); wrap.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  wrap.addEventListener(ev, (e) => { e.preventDefault(); wrap.classList.remove("drag"); })
);
wrap.addEventListener("drop", (e) => {
  const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  send(files);
});

window.addEventListener("resize", () => {
  const step = state.steps[state.cursor];
  if (step) drawChart(step.trend);
});

loadHealth();
drawChart({ wetness_history: [], class_history: [] });
