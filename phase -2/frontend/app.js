const MODEL_URL = "../outputs/phase2_model_summary.json";

const els = {
  speed: document.getElementById("speedInput"),
  feed: document.getElementById("feedInput"),
  depth: document.getElementById("depthInput"),
  elapsed: document.getElementById("elapsedInput"),
  threshold: document.getElementById("thresholdInput"),
  initialWear: document.getElementById("initialWearInput"),
  material: document.getElementById("materialInput"),
  calibration: document.getElementById("calibrationInput"),
  observedTime: document.getElementById("observedTimeInput"),
  observedWear: document.getElementById("observedWearInput"),
  calibrateButton: document.getElementById("calibrateButton"),
  exportCsvButton: document.getElementById("exportCsvButton"),
  materialHint: document.getElementById("materialHint"),
  calibrationEstimate: document.getElementById("calibrationEstimate"),
  lifeMetric: document.getElementById("lifeMetric"),
  lifeSubMetric: document.getElementById("lifeSubMetric"),
  wearMetric: document.getElementById("wearMetric"),
  wearSubMetric: document.getElementById("wearSubMetric"),
  phiMetric: document.getElementById("phiMetric"),
  phiSubMetric: document.getElementById("phiSubMetric"),
  transitionMetric: document.getElementById("transitionMetric"),
  transitionSubMetric: document.getElementById("transitionSubMetric"),
  lateFactorMetric: document.getElementById("lateFactorMetric"),
  lateFactorSubMetric: document.getElementById("lateFactorSubMetric"),
  transitionWearChip: document.getElementById("transitionWearChip"),
  earlyExponentChip: document.getElementById("earlyExponentChip"),
  lateExponentChip: document.getElementById("lateExponentChip"),
  conditionEquation: document.getElementById("conditionEquation"),
  wearEquation: document.getElementById("wearEquation"),
  statusLine: document.getElementById("statusLine"),
  coeffTable: document.getElementById("coeffTable"),
  evidenceTable: document.getElementById("evidenceTable"),
  materialList: document.getElementById("materialList"),
  notesList: document.getElementById("notesList"),
  chart: document.getElementById("curveChart"),
};

const state = {
  model: null,
};

function coeffs() {
  return state.model.selected_coefficients;
}

function fallbackModel() {
  return window.__PHASE2_MODEL__ || null;
}

async function loadModel() {
  try {
    const response = await fetch(MODEL_URL, { cache: "no-store" });
    if (response.ok) {
      return await response.json();
    }
  } catch (error) {
    // Fall back to the embedded snapshot if the page was opened directly.
  }

  const embedded = fallbackModel();
  if (embedded) {
    return embedded;
  }

  throw new Error("Could not load the phase-2 model summary.");
}

function parseOptionalNumber(element) {
  const raw = element.value.trim();
  if (!raw) {
    return null;
  }
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function parseInputs() {
  return {
    speedRpm: Math.max(Number(els.speed.value), 0),
    feedMmTooth: Math.max(Number(els.feed.value), 0),
    depthMm: Math.max(Number(els.depth.value), 0),
    elapsedMinutes: Math.max(Number(els.elapsed.value), 0),
    thresholdWearMm: Math.max(Number(els.threshold.value), 0),
    initialWearMm: Math.max(Number(els.initialWear.value), 0),
    materialFamily: els.material.value,
    calibrationFactor: Math.max(Number(els.calibration.value), 0),
    observedTimeMin: parseOptionalNumber(els.observedTime),
    observedWearMm: parseOptionalNumber(els.observedWear),
  };
}

function lateFactor(materialFamily) {
  const factors = coeffs().late_stage_factor_by_material_family;
  if (Object.prototype.hasOwnProperty.call(factors, materialFamily)) {
    return Number(factors[materialFamily]);
  }
  return Number(factors.generic);
}

function conditionIntensity(input) {
  const c = coeffs();
  return (
    Math.pow(input.speedRpm / Number(c.reference_speed_rpm), Number(c.speed_exponent)) *
    Math.pow(input.feedMmTooth / Number(c.reference_feed_mm_tooth), Number(c.feed_exponent)) *
    Math.pow(input.depthMm / Number(c.reference_depth_mm), Number(c.depth_exponent))
  );
}

function earlyAmplitude(input) {
  return input.calibrationFactor * Number(coeffs().k_early) * conditionIntensity(input);
}

function transitionTimeMinutes(input) {
  const deltaWear = Number(state.model.transition_wear_mm) - input.initialWearMm;
  const amplitude = earlyAmplitude(input);

  if (deltaWear <= 0 || amplitude <= 0) {
    return 0;
  }

  return Math.pow(deltaWear / amplitude, 1 / Number(coeffs().early_exponent));
}

function wearAtMinutes(input, timeMinutes) {
  const c = coeffs();
  const earlyExp = Number(c.early_exponent);
  const lateExp = Number(c.late_exponent);
  const transitionWear = Number(state.model.transition_wear_mm);
  const amplitude = earlyAmplitude(input);
  const transitionTime = transitionTimeMinutes(input);

  if (timeMinutes <= 0) {
    return input.initialWearMm;
  }

  if (timeMinutes <= transitionTime) {
    return input.initialWearMm + amplitude * Math.pow(timeMinutes, earlyExp);
  }

  return (
    transitionWear +
    lateFactor(input.materialFamily) * amplitude * Math.pow(timeMinutes - transitionTime, lateExp)
  );
}

function lifeToThresholdMinutes(input) {
  const c = coeffs();
  const earlyExp = Number(c.early_exponent);
  const lateExp = Number(c.late_exponent);
  const transitionWear = Number(state.model.transition_wear_mm);
  const amplitude = earlyAmplitude(input);

  if (input.thresholdWearMm <= input.initialWearMm) {
    return 0;
  }

  if (amplitude <= 0) {
    return Number.POSITIVE_INFINITY;
  }

  if (input.thresholdWearMm <= transitionWear) {
    return Math.pow((input.thresholdWearMm - input.initialWearMm) / amplitude, 1 / earlyExp);
  }

  const deltaWear = transitionWear - input.initialWearMm;
  const transitionTime = deltaWear <= 0 ? 0 : Math.pow(deltaWear / amplitude, 1 / earlyExp);
  const lateAmplitude = lateFactor(input.materialFamily) * amplitude;

  if (lateAmplitude <= 0) {
    return Number.POSITIVE_INFINITY;
  }

  const lateMinutes = Math.pow((input.thresholdWearMm - transitionWear) / lateAmplitude, 1 / lateExp);
  return transitionTime + lateMinutes;
}

function calibrationFactorFromPoint(input) {
  const c = coeffs();
  const observedWear = input.observedWearMm;
  const observedTime = input.observedTimeMin;
  const phi = conditionIntensity(input);

  if (
    observedWear === null ||
    observedTime === null ||
    observedWear <= input.initialWearMm ||
    observedTime <= 0
  ) {
    return 1;
  }

  const numerator = observedWear - input.initialWearMm;
  const denominator =
    Number(c.k_early) * phi * Math.pow(observedTime, Number(c.early_exponent));

  if (denominator <= 0) {
    return 1;
  }

  return numerator / denominator;
}

function buildCurve(input, lifeMinutes, transitionTime) {
  const safeLife = Number.isFinite(lifeMinutes) ? lifeMinutes : input.elapsedMinutes + 60;
  const maxMinutes = Math.max(
    40,
    safeLife * 1.14,
    input.elapsedMinutes * 1.18,
    transitionTime * 1.15,
  );
  const pointCount = 180;
  const points = [];
  let maxWear = input.initialWearMm;

  for (let index = 0; index <= pointCount; index += 1) {
    const timeMinutes = (index / pointCount) * maxMinutes;
    const wearMm = wearAtMinutes(input, timeMinutes);
    points.push({ timeMinutes, wearMm });
    maxWear = Math.max(maxWear, wearMm);
  }

  maxWear = Math.max(maxWear, input.thresholdWearMm, Number(state.model.transition_wear_mm), 0.3);
  return { points, maxMinutes, maxWear };
}

function toLabel(text) {
  return text.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatNumber(value, digits = 4) {
  if (!Number.isFinite(value)) {
    return "Not defined";
  }
  return Number(value).toFixed(digits);
}

function formatMetric(value, digits, unit = "") {
  if (!Number.isFinite(value)) {
    return "Not defined";
  }
  return `${Number(value).toFixed(digits)}${unit}`;
}

function renderDefinitionTable(rows) {
  return rows
    .map(
      ([label, value]) => `
        <dl>
          <dt>${label}</dt>
          <dd>${value}</dd>
        </dl>
      `,
    )
    .join("");
}

function renderResearchSnapshot() {
  const c = coeffs();
  const evidence = state.model.evidence;

  els.transitionWearChip.textContent = `${formatNumber(state.model.transition_wear_mm, 3)} mm`;
  els.earlyExponentChip.textContent = formatNumber(c.early_exponent, 3);
  els.lateExponentChip.textContent = formatNumber(c.late_exponent, 3);

  els.coeffTable.innerHTML = renderDefinitionTable([
    ["k_early", formatNumber(c.k_early, 5)],
    ["speed exponent", formatNumber(c.speed_exponent, 4)],
    ["feed exponent", formatNumber(c.feed_exponent, 4)],
    ["depth exponent", formatNumber(c.depth_exponent, 4)],
    ["amplitude R^2", formatNumber(c.amplitude_r2, 4)],
    ["reference speed", `${formatNumber(c.reference_speed_rpm, 0)} rpm`],
    ["reference feed", `${formatNumber(c.reference_feed_mm_tooth, 3)} mm/tooth`],
    ["reference depth", `${formatNumber(c.reference_depth_mm, 1)} mm`],
  ]);

  els.evidenceTable.innerHTML = renderDefinitionTable([
    ["NUAA early exponent", formatNumber(evidence.nuaa_early_exponent_raw, 4)],
    ["PHM early exponent", formatNumber(evidence.phm_early_exponent_raw, 4)],
    ["NASA late exponent", formatNumber(evidence.nasa_late_exponent_raw, 4)],
    ["NUAA RMSE", `${formatNumber(evidence.nuaa_early_rmse_mm, 4)} mm`],
    ["PHM RMSE", `${formatNumber(evidence.phm_early_rmse_mm, 4)} mm`],
    ["NASA RMSE", `${formatNumber(evidence.nasa_late_rmse_mm, 4)} mm`],
  ]);

  els.materialList.innerHTML = Object.entries(c.late_stage_factor_by_material_family)
    .map(
      ([material, factor]) =>
        `<li><strong>${toLabel(material)}</strong>: late-stage multiplier ${formatNumber(factor, 4)}</li>`,
    )
    .join("");

  els.notesList.innerHTML = state.model.notes
    .map((note) => `<li>${note}</li>`)
    .join("");

  els.conditionEquation.textContent =
    `phi = (n / ${formatNumber(c.reference_speed_rpm, 0)})^${formatNumber(c.speed_exponent, 4)} ` +
    `* (fz / ${formatNumber(c.reference_feed_mm_tooth, 3)})^${formatNumber(c.feed_exponent, 4)} ` +
    `* (ap / ${formatNumber(c.reference_depth_mm, 1)})^${formatNumber(c.depth_exponent, 4)}`;

  els.wearEquation.textContent =
    `If VB <= ${formatNumber(state.model.transition_wear_mm, 2)} mm:\n` +
    `VB(t) = VB0 + lambda * ${formatNumber(c.k_early, 5)} * phi * t^${formatNumber(c.early_exponent, 2)}\n\n` +
    `If VB > ${formatNumber(state.model.transition_wear_mm, 2)} mm:\n` +
    `VB(t) = ${formatNumber(state.model.transition_wear_mm, 2)} + lambda * rho * ${formatNumber(c.k_early, 5)} * phi * (t - t_transition)^${formatNumber(c.late_exponent, 2)}`;
}

function createTicks(maxValue, count) {
  const ticks = [];
  for (let index = 0; index <= count; index += 1) {
    ticks.push((index / count) * maxValue);
  }
  return ticks;
}

function renderChart(curve, input, lifeMinutes, transitionTime) {
  const width = 900;
  const height = 430;
  const pad = { left: 72, right: 26, top: 22, bottom: 54 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xMax = curve.maxMinutes;
  const yMax = Math.max(curve.maxWear * 1.08, 0.3);
  const thresholdY = input.thresholdWearMm;
  const currentWear = wearAtMinutes(input, input.elapsedMinutes);

  const xScale = (value) => pad.left + (value / xMax) * plotWidth;
  const yScale = (value) => pad.top + (1 - value / yMax) * plotHeight;

  const linePath = curve.points
    .map((point, index) => {
      const x = xScale(point.timeMinutes).toFixed(2);
      const y = yScale(point.wearMm).toFixed(2);
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const fillPath =
    `${linePath} ` +
    `L ${xScale(curve.points[curve.points.length - 1].timeMinutes).toFixed(2)} ${yScale(0).toFixed(2)} ` +
    `L ${xScale(0).toFixed(2)} ${yScale(0).toFixed(2)} Z`;

  const yTicks = createTicks(yMax, 4)
    .map((tick) => {
      const y = yScale(tick);
      return `
        <line class="chart-grid" x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}"></line>
        <text class="chart-axis" x="${pad.left - 12}" y="${y + 4}" text-anchor="end">${tick.toFixed(2)}</text>
      `;
    })
    .join("");

  const xTicks = createTicks(xMax, 5)
    .map((tick) => {
      const x = xScale(tick);
      return `
        <line class="chart-grid" x1="${x}" y1="${pad.top}" x2="${x}" y2="${height - pad.bottom}"></line>
        <text class="chart-axis" x="${x}" y="${height - pad.bottom + 20}" text-anchor="middle">${tick.toFixed(0)}</text>
      `;
    })
    .join("");

  const thresholdLine = `
    <line class="chart-threshold" x1="${pad.left}" y1="${yScale(thresholdY)}" x2="${width - pad.right}" y2="${yScale(thresholdY)}"></line>
    <text class="chart-label" x="${width - pad.right - 8}" y="${yScale(thresholdY) - 8}" text-anchor="end">threshold</text>
  `;

  const transitionBand =
    transitionTime > 0 && transitionTime < xMax
      ? `<rect class="chart-band" x="${xScale(transitionTime)}" y="${pad.top}" width="${width - pad.right - xScale(transitionTime)}" height="${plotHeight}"></rect>`
      : "";

  const transitionLine =
    transitionTime > 0 && transitionTime < xMax
      ? `
        <line class="chart-transition" x1="${xScale(transitionTime)}" y1="${pad.top}" x2="${xScale(transitionTime)}" y2="${height - pad.bottom}"></line>
        <text class="chart-label" x="${xScale(transitionTime) + 6}" y="${pad.top + 16}">t_transition</text>
      `
      : "";

  const lifeLine =
    Number.isFinite(lifeMinutes) && lifeMinutes <= xMax
      ? `
        <line class="chart-life" x1="${xScale(lifeMinutes)}" y1="${pad.top}" x2="${xScale(lifeMinutes)}" y2="${height - pad.bottom}"></line>
        <text class="chart-label" x="${xScale(lifeMinutes) - 8}" y="${pad.top + 16}" text-anchor="end">tool life</text>
      `
      : "";

  els.chart.innerHTML = `
    <defs>
      <linearGradient id="curve-fill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#1e5f51" stop-opacity="0.26"></stop>
        <stop offset="100%" stop-color="#1e5f51" stop-opacity="0.02"></stop>
      </linearGradient>
    </defs>
    ${transitionBand}
    ${yTicks}
    ${xTicks}
    <rect class="chart-boundary" x="${pad.left}" y="${pad.top}" width="${plotWidth}" height="${plotHeight}" fill="none"></rect>
    ${thresholdLine}
    ${transitionLine}
    ${lifeLine}
    <path class="chart-fill" d="${fillPath}"></path>
    <path class="chart-line" d="${linePath}"></path>
    <circle class="chart-point" cx="${xScale(curve.points[curve.points.length - 1].timeMinutes)}" cy="${yScale(curve.points[curve.points.length - 1].wearMm)}" r="3.5"></circle>
    <circle class="chart-current" cx="${xScale(input.elapsedMinutes)}" cy="${yScale(currentWear)}" r="5"></circle>
    <text class="chart-label" x="${xScale(input.elapsedMinutes) + 8}" y="${yScale(currentWear) - 10}">elapsed</text>
    <text class="chart-axis" x="${pad.left + plotWidth / 2}" y="${height - 12}" text-anchor="middle">time (min)</text>
    <text class="chart-axis" transform="translate(18 ${pad.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">wear (mm)</text>
  `;
}

function renderCalibrationEstimate(input) {
  if (input.observedTimeMin === null || input.observedWearMm === null) {
    els.calibrationEstimate.textContent = "No observed point yet.";
    return;
  }

  if (input.observedTimeMin <= 0) {
    els.calibrationEstimate.textContent = "Observed time must be greater than zero.";
    return;
  }

  if (input.observedWearMm <= input.initialWearMm) {
    els.calibrationEstimate.textContent = "Observed wear must be above the initial wear state.";
    return;
  }

  const estimate = calibrationFactorFromPoint(input);
  els.calibrationEstimate.textContent =
    `Estimated lambda = ${formatNumber(estimate, 4)} from (${formatNumber(input.observedTimeMin, 1)} min, ${formatNumber(input.observedWearMm, 3)} mm).`;
}

function render() {
  const input = parseInputs();
  const phi = conditionIntensity(input);
  const lifeMinutes = lifeToThresholdMinutes(input);
  const transitionTime = transitionTimeMinutes(input);
  const currentWear = wearAtMinutes(input, input.elapsedMinutes);
  const currentRegime = input.elapsedMinutes <= transitionTime ? "early regime" : "late regime";
  const curve = buildCurve(input, lifeMinutes, transitionTime);
  const remaining = Number.isFinite(lifeMinutes) ? lifeMinutes - input.elapsedMinutes : Number.POSITIVE_INFINITY;
  const factor = lateFactor(input.materialFamily);

  els.lifeMetric.textContent = formatMetric(lifeMinutes, 2, " min");
  els.lifeSubMetric.textContent = Number.isFinite(remaining)
    ? `${formatMetric(Math.max(remaining, 0), 2, " min")} remaining`
    : "Threshold crossing not defined";

  els.wearMetric.textContent = formatMetric(currentWear, 4, " mm");
  els.wearSubMetric.textContent = `${currentRegime} at ${formatMetric(input.elapsedMinutes, 1, " min")}`;

  els.phiMetric.textContent = formatMetric(phi, 4);
  els.phiSubMetric.textContent = `relative to ${formatMetric(coeffs().reference_speed_rpm, 0)} rpm / ${formatMetric(coeffs().reference_feed_mm_tooth, 3)} / ${formatMetric(coeffs().reference_depth_mm, 1)}`;

  els.transitionMetric.textContent = formatMetric(transitionTime, 2, " min");
  els.transitionSubMetric.textContent = `${formatMetric(state.model.transition_wear_mm, 2, " mm")} transition wear`;

  els.lateFactorMetric.textContent = formatMetric(factor, 4);
  els.lateFactorSubMetric.textContent = `${toLabel(input.materialFamily)} material family`;

  els.materialHint.textContent =
    `Late-stage multiplier for ${toLabel(input.materialFamily)} is ${formatNumber(factor, 4)}. Higher values accelerate wear after the ${formatNumber(state.model.transition_wear_mm, 2)} mm transition.`;

  if (input.thresholdWearMm <= input.initialWearMm) {
    els.statusLine.textContent = "The selected threshold is already at or below the initial wear state, so predicted tool life is zero.";
  } else if (Number.isFinite(remaining) && remaining >= 0) {
    els.statusLine.textContent =
      `${formatMetric(remaining, 2, " min")} remain before the tool reaches ${formatMetric(input.thresholdWearMm, 3, " mm")} wear.`;
  } else if (Number.isFinite(remaining)) {
    els.statusLine.textContent =
      `The elapsed time is ${formatMetric(Math.abs(remaining), 2, " min")} past the predicted threshold crossing.`;
  } else {
    els.statusLine.textContent = "The selected parameters do not produce a finite threshold-crossing time with the current settings.";
  }

  renderCalibrationEstimate(input);
  renderChart(curve, input, lifeMinutes, transitionTime);
}

function applyEstimatedCalibration() {
  const input = parseInputs();
  if (
    input.observedTimeMin === null ||
    input.observedWearMm === null ||
    input.observedTimeMin <= 0 ||
    input.observedWearMm <= input.initialWearMm
  ) {
    renderCalibrationEstimate(input);
    return;
  }
  const estimate = calibrationFactorFromPoint(input);
  els.calibration.value = formatNumber(estimate, 4);
  render();
}

function exportCurveCsv() {
  const input = parseInputs();
  const lifeMinutes = lifeToThresholdMinutes(input);
  const transitionTime = transitionTimeMinutes(input);
  const curve = buildCurve(input, lifeMinutes, transitionTime);
  const csvLines = ["time_min,wear_mm"];

  curve.points.forEach((point) => {
    csvLines.push(`${point.timeMinutes.toFixed(6)},${point.wearMm.toFixed(6)}`);
  });

  const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const material = input.materialFamily.replace(/[^a-z0-9_-]+/gi, "_");
  link.href = url;
  link.download = `phase2_curve_${material}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bind() {
  [
    els.speed,
    els.feed,
    els.depth,
    els.elapsed,
    els.threshold,
    els.initialWear,
    els.material,
    els.calibration,
    els.observedTime,
    els.observedWear,
  ].forEach((element) => {
    element.addEventListener("input", render);
  });

  els.calibrateButton.addEventListener("click", applyEstimatedCalibration);
  els.exportCsvButton.addEventListener("click", exportCurveCsv);
}

async function init() {
  try {
    state.model = await loadModel();
  } catch (error) {
    document.body.innerHTML =
      `<div style="padding:24px;font-family:IBM Plex Mono,monospace;">${error.message}</div>`;
    return;
  }

  renderResearchSnapshot();
  bind();
  render();
}

init();
