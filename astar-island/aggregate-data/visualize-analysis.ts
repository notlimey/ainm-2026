import "@std/dotenv/load";
import type { Analysis } from "./client.ts";

const DATA_DIR = "data";

const CLASSES = [
	{ name: "Empty/Ocean/Plains", color: [200, 184, 138] },
	{ name: "Settlement", color: [212, 118, 10] },
	{ name: "Port", color: [14, 116, 144] },
	{ name: "Ruin", color: [127, 29, 29] },
	{ name: "Forest", color: [45, 90, 39] },
	{ name: "Mountain", color: [107, 114, 128] },
];

function terrainToClass(code: number): number {
	switch (code) {
		case 0: case 10: case 11: return 0;
		case 1: return 1;
		case 2: return 2;
		case 3: return 3;
		case 4: return 4;
		case 5: return 5;
		default: return 0;
	}
}

interface SeedData {
	round: number;
	seed: number;
	analysis: Analysis;
}

async function loadAllAnalysis(): Promise<SeedData[]> {
	const results: SeedData[] = [];
	const analysisDir = `${DATA_DIR}/analysis`;

	for await (const roundEntry of Deno.readDir(analysisDir)) {
		if (!roundEntry.isDirectory || !roundEntry.name.startsWith("r")) continue;
		const roundNum = parseInt(roundEntry.name.slice(1));

		for await (const seedEntry of Deno.readDir(`${analysisDir}/${roundEntry.name}`)) {
			if (!seedEntry.name.startsWith("s") || !seedEntry.name.endsWith(".json")) continue;
			const seedNum = parseInt(seedEntry.name.slice(1));

			const raw = await Deno.readTextFile(`${analysisDir}/${roundEntry.name}/${seedEntry.name}`);
			const analysis: Analysis = JSON.parse(raw);
			results.push({ round: roundNum, seed: seedNum, analysis });
		}
	}

	results.sort((a, b) => a.round - b.round || a.seed - b.seed);
	return results;
}

function generateHTML(seeds: SeedData[]): string {
	// Pre-compute JSON data for the page
	const seedsJson = seeds.map(s => ({
		round: s.round,
		seed: s.seed,
		score: s.analysis.score,
		width: s.analysis.width,
		height: s.analysis.height,
		ground_truth: s.analysis.ground_truth,
		prediction: s.analysis.prediction,
		initial_grid: s.analysis.initial_grid,
	}));

	return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Astar Island — Analysis Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
h1 { font-size: 18px; margin-bottom: 12px; color: #fff; }
h2 { font-size: 14px; margin: 16px 0 8px; color: #aaa; }

.controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
.controls label { font-size: 12px; color: #888; }
.controls select, .controls input { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 4px 8px; font-family: inherit; font-size: 12px; border-radius: 3px; }
button { background: #222; color: #e0e0e0; border: 1px solid #444; padding: 4px 12px; cursor: pointer; font-family: inherit; font-size: 12px; border-radius: 3px; }
button:hover { background: #333; }
button.active { background: #444; border-color: #888; }

.grid-row { display: flex; gap: 20px; margin-bottom: 16px; flex-wrap: wrap; }
.grid-box { display: flex; flex-direction: column; align-items: center; }
.grid-box .label { font-size: 11px; color: #666; margin-bottom: 4px; }
canvas { image-rendering: pixelated; border: 1px solid #333; cursor: crosshair; }

.info-panel { background: #111; border: 1px solid #333; padding: 12px; font-size: 12px; margin-top: 8px; border-radius: 4px; min-height: 80px; }
.info-panel .coord { color: #888; }
.info-panel .bar-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.info-panel .bar-label { width: 120px; font-size: 11px; }
.info-panel .bar-bg { width: 200px; height: 12px; background: #222; border-radius: 2px; overflow: hidden; position: relative; }
.info-panel .bar-fill { height: 100%; border-radius: 2px; }
.info-panel .bar-val { font-size: 11px; width: 50px; text-align: right; }
.bar-pair { display: flex; gap: 2px; }
.bar-pair .bar-bg { width: 100px; }

.legend { display: flex; gap: 12px; margin: 8px 0; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; }
.legend-swatch { width: 12px; height: 12px; border-radius: 2px; border: 1px solid #555; }

.score-badge { padding: 2px 8px; border-radius: 3px; font-weight: bold; }
.entropy-high { color: #f87171; }
.entropy-low { color: #4ade80; }

.stats { display: grid; grid-template-columns: repeat(auto-fill, 140px); gap: 8px; margin: 8px 0; }
.stat-card { background: #151515; border: 1px solid #282828; padding: 8px; border-radius: 4px; text-align: center; }
.stat-card .val { font-size: 18px; font-weight: bold; color: #fff; }
.stat-card .lbl { font-size: 10px; color: #666; margin-top: 2px; }
</style>
</head>
<body>
<h1>Astar Island — Analysis Viewer</h1>

<div class="controls">
  <label>Round/Seed:
    <select id="seedSelect"></select>
  </label>
  <label>View:
    <select id="viewMode">
      <option value="argmax">Argmax (most likely class)</option>
      <option value="entropy">Entropy (uncertainty)</option>
      <option value="diff">Diff (ground truth vs prediction)</option>
      <option value="class">Single class probability</option>
    </select>
  </label>
  <label id="classSelectWrap" style="display:none">Class:
    <select id="classSelect">
      <option value="0">Empty/Plains</option>
      <option value="1">Settlement</option>
      <option value="2">Port</option>
      <option value="3">Ruin</option>
      <option value="4">Forest</option>
      <option value="5">Mountain</option>
    </select>
  </label>
  <label>Zoom:
    <input id="zoom" type="range" min="8" max="24" value="14">
  </label>
</div>

<div class="legend" id="legend"></div>

<div class="stats" id="statsPanel"></div>

<div class="grid-row">
  <div class="grid-box">
    <div class="label">Initial State</div>
    <canvas id="cvInitial"></canvas>
  </div>
  <div class="grid-box">
    <div class="label">Ground Truth</div>
    <canvas id="cvTruth"></canvas>
  </div>
  <div class="grid-box">
    <div class="label">Your Prediction</div>
    <canvas id="cvPred"></canvas>
  </div>
</div>

<div class="info-panel" id="infoPanel">Hover over a cell to see probabilities</div>

<script>
const SEEDS = ${JSON.stringify(seedsJson)};
const CLASS_NAMES = ${JSON.stringify(CLASSES.map(c => c.name))};
const CLASS_COLORS = ${JSON.stringify(CLASSES.map(c => c.color))};

const seedSelect = document.getElementById('seedSelect');
const viewMode = document.getElementById('viewMode');
const classSelect = document.getElementById('classSelect');
const classSelectWrap = document.getElementById('classSelectWrap');
const zoomSlider = document.getElementById('zoom');
const infoPanel = document.getElementById('infoPanel');
const statsPanel = document.getElementById('statsPanel');
const legend = document.getElementById('legend');

const cvInitial = document.getElementById('cvInitial');
const cvTruth = document.getElementById('cvTruth');
const cvPred = document.getElementById('cvPred');

SEEDS.forEach((s, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  const sc = s.score !== null ? ' — score: ' + s.score.toFixed(1) : '';
  opt.textContent = 'r' + s.round + '.s' + s.seed + sc;
  seedSelect.appendChild(opt);
});

function terrainToClass(code) {
  if (code === 0 || code === 10 || code === 11) return 0;
  if (code >= 1 && code <= 5) return code;
  return 0;
}

function entropy(probs) {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

function klDiv(p, q) {
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      const qi = Math.max(q[i], 1e-10);
      kl += p[i] * Math.log(p[i] / qi);
    }
  }
  return kl;
}

function argmax(probs) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }
  return best;
}

function colorForClass(cls, alpha) {
  const c = CLASS_COLORS[cls];
  return [c[0] * alpha + 10 * (1-alpha), c[1] * alpha + 10 * (1-alpha), c[2] * alpha + 10 * (1-alpha)];
}

function drawGrid(canvas, w, h, cellSize, paintFn) {
  canvas.width = w * cellSize;
  canvas.height = h * cellSize;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w * cellSize, h * cellSize);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = paintFn(x, y);
      for (let dy = 0; dy < cellSize; dy++) {
        for (let dx = 0; dx < cellSize; dx++) {
          const idx = ((y * cellSize + dy) * w * cellSize + x * cellSize + dx) * 4;
          img.data[idx] = r;
          img.data[idx+1] = g;
          img.data[idx+2] = b;
          img.data[idx+3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function render() {
  const s = SEEDS[seedSelect.value];
  if (!s) return;
  const { width: w, height: h, ground_truth: gt, prediction: pred, initial_grid: ig } = s;
  const cellSize = parseInt(zoomSlider.value);
  const mode = viewMode.value;

  classSelectWrap.style.display = mode === 'class' ? '' : 'none';

  // Initial state
  drawGrid(cvInitial, w, h, cellSize, (x, y) => {
    if (!ig) return [30, 30, 30];
    const cls = terrainToClass(ig[y][x]);
    return CLASS_COLORS[cls];
  });

  // Compute entropy range for scaling
  let maxEntropy = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      maxEntropy = Math.max(maxEntropy, entropy(gt[y][x]));

  const classIdx = parseInt(classSelect.value);

  // Ground truth
  drawGrid(cvTruth, w, h, cellSize, (x, y) => {
    const probs = gt[y][x];
    if (mode === 'entropy') {
      const e = entropy(probs) / Math.max(maxEntropy, 0.01);
      return [Math.floor(e * 248 + 10), Math.floor((1-e) * 100 + 10), 30];
    }
    if (mode === 'class') {
      const p = probs[classIdx];
      const c = CLASS_COLORS[classIdx];
      return [Math.floor(c[0]*p + 10*(1-p)), Math.floor(c[1]*p + 10*(1-p)), Math.floor(c[2]*p + 10*(1-p))];
    }
    // argmax or diff: show argmax
    const cls = argmax(probs);
    const conf = probs[cls];
    return colorForClass(cls, 0.3 + 0.7 * conf);
  });

  // Prediction
  drawGrid(cvPred, w, h, cellSize, (x, y) => {
    const probs = pred[y][x];
    if (mode === 'diff') {
      const kl = klDiv(gt[y][x], probs);
      const intensity = Math.min(kl / 2, 1);
      return [Math.floor(255 * intensity), Math.floor(80 * (1-intensity)), Math.floor(30 * (1-intensity))];
    }
    if (mode === 'entropy') {
      const e = entropy(probs) / Math.max(maxEntropy, 0.01);
      return [Math.floor(e * 248 + 10), Math.floor((1-e) * 100 + 10), 30];
    }
    if (mode === 'class') {
      const p = probs[classIdx];
      const c = CLASS_COLORS[classIdx];
      return [Math.floor(c[0]*p + 10*(1-p)), Math.floor(c[1]*p + 10*(1-p)), Math.floor(c[2]*p + 10*(1-p))];
    }
    const cls = argmax(probs);
    const conf = probs[cls];
    return colorForClass(cls, 0.3 + 0.7 * conf);
  });

  // Legend
  legend.innerHTML = '';
  if (mode === 'entropy') {
    legend.innerHTML = '<span style="font-size:11px;color:#888">Red = high entropy (uncertain) &nbsp; Green = low entropy (confident)</span>';
  } else if (mode === 'diff') {
    legend.innerHTML = '<span style="font-size:11px;color:#888">Prediction canvas: Red = high KL divergence (bad), dark = good match</span>';
  } else {
    CLASS_NAMES.forEach((name, i) => {
      const c = CLASS_COLORS[i];
      legend.innerHTML += '<div class="legend-item"><div class="legend-swatch" style="background:rgb('+c+')"></div>'+name+'</div>';
    });
  }

  // Stats
  let totalKL = 0, totalEntropy = 0, dynamicCells = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const e = entropy(gt[y][x]);
      const kl = klDiv(gt[y][x], pred[y][x]);
      totalEntropy += e;
      totalKL += e * kl;
      if (e > 0.01) dynamicCells++;
    }
  }
  const weightedKL = totalEntropy > 0 ? totalKL / totalEntropy : 0;
  const calcScore = Math.max(0, Math.min(100, 100 * Math.exp(-3 * weightedKL)));

  statsPanel.innerHTML = [
    { val: s.score !== null ? s.score.toFixed(1) : '-', lbl: 'Score' },
    { val: calcScore.toFixed(1), lbl: 'Calc Score' },
    { val: weightedKL.toFixed(4), lbl: 'Weighted KL' },
    { val: dynamicCells, lbl: 'Dynamic Cells' },
    { val: (w*h - dynamicCells), lbl: 'Static Cells' },
    { val: maxEntropy.toFixed(3), lbl: 'Max Entropy' },
  ].map(s => '<div class="stat-card"><div class="val">'+s.val+'</div><div class="lbl">'+s.lbl+'</div></div>').join('');
}

function handleHover(e, canvas, dataKey) {
  const s = SEEDS[seedSelect.value];
  if (!s) return;
  const cellSize = parseInt(zoomSlider.value);
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / cellSize);
  const y = Math.floor((e.clientY - rect.top) / cellSize);
  if (x < 0 || x >= s.width || y < 0 || y >= s.height) return;

  const gt = s.ground_truth[y][x];
  const pr = s.prediction[y][x];
  const ig = s.initial_grid ? terrainToClass(s.initial_grid[y][x]) : -1;
  const cellEntropy = entropy(gt);
  const cellKL = klDiv(gt, pr);

  let html = '<span class="coord">Cell ('+x+', '+y+')';
  if (ig >= 0) html += ' — initial: ' + CLASS_NAMES[ig];
  html += ' — entropy: <span class="'+(cellEntropy > 0.5 ? 'entropy-high' : 'entropy-low')+'">'+cellEntropy.toFixed(3)+'</span>';
  html += ' — KL: '+cellKL.toFixed(4)+'</span><br>';

  for (let i = 0; i < 6; i++) {
    const c = CLASS_COLORS[i];
    const gtPct = (gt[i]*100).toFixed(1);
    const prPct = (pr[i]*100).toFixed(1);
    html += '<div class="bar-row">';
    html += '<div class="bar-label" style="color:rgb('+c+')">'+CLASS_NAMES[i]+'</div>';
    html += '<div class="bar-pair">';
    html += '<div class="bar-bg"><div class="bar-fill" style="width:'+gtPct+'%;background:rgb('+c+')"></div></div>';
    html += '<div class="bar-bg"><div class="bar-fill" style="width:'+prPct+'%;background:rgb('+c+');opacity:0.5"></div></div>';
    html += '</div>';
    html += '<div class="bar-val">'+gtPct+'%</div>';
    html += '<div class="bar-val" style="opacity:0.5">'+prPct+'%</div>';
    html += '</div>';
  }
  html += '<div style="font-size:10px;color:#555;margin-top:4px">Left bars = ground truth, right bars = your prediction</div>';
  infoPanel.innerHTML = html;
}

[cvInitial, cvTruth, cvPred].forEach(cv => {
  cv.addEventListener('mousemove', (e) => handleHover(e, cv, null));
});

seedSelect.addEventListener('change', render);
viewMode.addEventListener('change', render);
classSelect.addEventListener('change', render);
zoomSlider.addEventListener('input', render);

render();
</script>
</body>
</html>`;
}

console.log("Loading analysis data...");
const seeds = await loadAllAnalysis();

if (seeds.length === 0) {
	console.error("No analysis data found in data/analysis/");
	console.error("Run: deno run --allow-net --allow-env --allow-read --allow-write fetch-analysis.ts");
	Deno.exit(1);
}

console.log(`Found ${seeds.length} seed analyses`);

const html = generateHTML(seeds);
const outPath = `${DATA_DIR}/analysis-viewer.html`;
await Deno.writeTextFile(outPath, html);
console.log(`Written to ${outPath}`);
console.log(`Open: file://${Deno.cwd()}/${outPath}`);
