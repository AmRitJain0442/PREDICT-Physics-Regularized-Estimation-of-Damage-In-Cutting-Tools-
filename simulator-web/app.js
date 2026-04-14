const els = {
  dataset: document.getElementById("datasetSelect"),
  spindle: document.getElementById("spindleInput"),
  feed: document.getElementById("feedInput"),
  depth: document.getElementById("depthInput"),
  time: document.getElementById("timeInput"),
  vibStress: document.getElementById("vibStress"),
  aeStress: document.getElementById("aeStress"),
  crestStress: document.getElementById("crestStress"),
  vibStressValue: document.getElementById("vibStressValue"),
  aeStressValue: document.getElementById("aeStressValue"),
  crestStressValue: document.getElementById("crestStressValue"),
  lifeText: document.getElementById("lifeText"),
  wearText: document.getElementById("wearText"),
  failureText: document.getElementById("failureText"),
  statusText: document.getElementById("statusText"),
  equationText: document.getElementById("equationText"),
  bestTableBody: document.getElementById("bestTableBody"),
  markersList: document.getElementById("markersList"),
  canvas: document.getElementById("trajectoryCanvas"),
};

let state = null;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function parseInputs() {
  return {
    dataset: els.dataset.value,
    spindle: Number(els.spindle.value),
    feed: Number(els.feed.value),
    depth: Number(els.depth.value),
    elapsed: Number(els.time.value),
    vibStress: Number(els.vibStress.value),
    aeStress: Number(els.aeStress.value),
    crestStress: Number(els.crestStress.value),
  };
}

function idwLife(feed, spindle, depth, rows) {
  let num = 0;
  let den = 0;
  for (const r of rows) {
    const df = (feed - Number(r.feed_per_tooth)) / 0.005;
    const ds = (spindle - Number(r.spindle_speed)) / 50;
    const dd = (depth - Number(r.axial_cutting_depth)) / 0.5;
    const dist = Math.sqrt(df * df + ds * ds + dd * dd);
    if (dist < 1e-7) {
      return Number(r.predicted_life_s);
    }
    const w = 1 / (dist * dist + 1e-8);
    num += w * Number(r.predicted_life_s);
    den += w;
  }
  return num / den;
}

function equationLife(feed, spindle, depth, eq) {
  const c = Number(eq.intercept_exp);
  const a = Number(eq.speed_exponent);
  const b = Number(eq.feed_exponent);
  const d = Number(eq.depth_exponent);
  return c * Math.pow(spindle, a) * Math.pow(feed, b) * Math.pow(depth, d);
}

function markerPenalty(vibStress, aeStress, crestStress) {
  const vib = Math.max(0, vibStress - 1.0);
  const ae = Math.max(0, aeStress - 1.0);
  const crest = Math.max(0, crestStress - 1.0);
  return 0.5 * vib + 0.35 * ae + 0.4 * crest;
}

function computeSimulation(input, data) {
  const tableLife = idwLife(input.feed, input.spindle, input.depth, data.nuaa_life_table);
  const eqLife = equationLife(input.feed, input.spindle, input.depth, data.taylor_like_equation);
  const blendedLife = 0.7 * tableLife + 0.3 * eqLife;
  const penalty = markerPenalty(input.vibStress, input.aeStress, input.crestStress);
  const effectiveLife = clamp(blendedLife / (1 + 0.45 * penalty), 120, 7200);

  const threshold = Number(data.thresholds_mm[input.dataset]);
  const wearShape = 0.92 + 0.2 * penalty;
  const wearRatio = Math.pow(input.elapsed / Math.max(effectiveLife, 1), wearShape);
  const estimatedWear = clamp(threshold * wearRatio, 0, threshold * 1.8);

  const pFail = clamp(sigmoid(5 * (input.elapsed / effectiveLife - 1) + 1.3 * penalty), 0.001, 0.999);
  const remaining = effectiveLife - input.elapsed;

  return {
    threshold,
    tableLife,
    eqLife,
    effectiveLife,
    estimatedWear,
    pFail,
    remaining,
  };
}

function statusText(prob) {
  if (prob < 0.35) return { label: "Healthy Zone", color: "#1e7f4a" };
  if (prob < 0.7) return { label: "Warning Zone", color: "#b37a1a" };
  return { label: "Failure Risk", color: "#b03a2e" };
}

function renderTable(data) {
  const sorted = [...data.nuaa_life_table].sort(
    (a, b) => Number(b.predicted_life_s) - Number(a.predicted_life_s),
  );
  const top = sorted.slice(0, 5);
  els.bestTableBody.innerHTML = top
    .map(
      (r) => `
      <tr>
        <td>${Number(r.feed_per_tooth).toFixed(3)}</td>
        <td>${Number(r.spindle_speed).toFixed(0)}</td>
        <td>${Number(r.axial_cutting_depth).toFixed(1)}</td>
        <td>${Number(r.predicted_life_s).toFixed(1)}</td>
      </tr>
    `,
    )
    .join("");
}

function renderMarkers(dataset, data) {
  const rows = data.top_markers[dataset] || [];
  els.markersList.innerHTML = rows
    .map((r) => {
      const dir = r.direction === "higher" ? "upward shift" : "downward shift";
      return `<li>${r.feature}: ${dir}, threshold ${Number(r.marker_threshold).toFixed(4)}</li>`;
    })
    .join("");
}

function drawTrajectory(canvas, sim, input) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = Math.round(cssW * 0.4);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = `${cssH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssW;
  const h = cssH;
  const pad = { l: 56, r: 16, t: 14, b: 38 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const tMax = Math.max(input.elapsed * 1.25, sim.effectiveLife * 1.4, 600);
  const yMax = sim.threshold * 1.9;
  const steps = 140;
  const penalty = markerPenalty(input.vibStress, input.aeStress, input.crestStress);
  const wearShape = 0.92 + 0.2 * penalty;

  function xScale(t) {
    return pad.l + (t / tMax) * plotW;
  }
  function yScale(y) {
    return pad.t + (1 - y / yMax) * plotH;
  }

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#d4deeb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#cc3f3f";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(pad.l, yScale(sim.threshold));
  ctx.lineTo(w - pad.r, yScale(sim.threshold));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#0f6ba8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * tMax;
    const wearRatio = Math.pow(t / Math.max(sim.effectiveLife, 1), wearShape);
    const y = clamp(sim.threshold * wearRatio, 0, sim.threshold * 1.8);
    const px = xScale(t);
    const py = yScale(y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  const xElapsed = xScale(input.elapsed);
  const wearNow = sim.estimatedWear;
  ctx.fillStyle = "#c5862f";
  ctx.beginPath();
  ctx.arc(xElapsed, yScale(wearNow), 4.4, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#22364f";
  ctx.font = '12px "IBM Plex Mono"';
  ctx.fillText("Time (s)", w / 2 - 26, h - 10);
  ctx.save();
  ctx.translate(14, h / 2 + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Wear (mm)", 0, 0);
  ctx.restore();
}

function render() {
  if (!state) return;
  const input = parseInputs();
  const sim = computeSimulation(input, state);
  const status = statusText(sim.pFail);

  els.vibStressValue.textContent = `${input.vibStress.toFixed(2)}x`;
  els.aeStressValue.textContent = `${input.aeStress.toFixed(2)}x`;
  els.crestStressValue.textContent = `${input.crestStress.toFixed(2)}x`;

  els.lifeText.textContent = `${sim.effectiveLife.toFixed(1)} s (${(sim.effectiveLife / 60).toFixed(1)} min)`;
  els.wearText.textContent = `${sim.estimatedWear.toFixed(4)} mm`;
  els.failureText.textContent = `${(sim.pFail * 100).toFixed(1)} %`;
  els.statusText.textContent = `${status.label} | remaining ${sim.remaining.toFixed(1)} s`;
  els.statusText.style.color = status.color;

  const eq = state.taylor_like_equation;
  els.equationText.textContent =
    `Life blend = 0.7*IDW(table) + 0.3*(C*s^a*f^b*d^c), ` +
    `C=${Number(eq.intercept_exp).toExponential(3)}, a=${eq.speed_exponent.toFixed(3)}, b=${eq.feed_exponent.toFixed(3)}, d=${eq.depth_exponent.toFixed(3)}`;

  renderMarkers(input.dataset, state);
  drawTrajectory(els.canvas, sim, input);
}

function bind() {
  [
    els.dataset,
    els.spindle,
    els.feed,
    els.depth,
    els.time,
    els.vibStress,
    els.aeStress,
    els.crestStress,
  ].forEach((el) => el.addEventListener("input", render));
  window.addEventListener("resize", render);
}

async function init() {
  try {
    const res = await fetch("simulator_data.json");
    state = await res.json();
  } catch (err) {
    document.body.innerHTML = `<p style="padding:20px;font-family:monospace;">Failed to load simulator_data.json. Run with a local server (example: python -m http.server 8000 inside simulator-web).</p>`;
    return;
  }

  const ds = Object.keys(state.thresholds_mm);
  els.dataset.innerHTML = ds.map((d) => `<option value="${d}">${d}</option>`).join("");
  els.dataset.value = "PHM2010";

  renderTable(state);
  bind();
  render();
}

init();
