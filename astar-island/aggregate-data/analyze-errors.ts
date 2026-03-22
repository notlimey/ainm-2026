// Error analysis tool for sim predictions vs ground truth
// Usage: deno run -A analyze-errors.ts [round]

import { NUM_CLASSES, terrainToClass, entropy, readPrediction as readPredictionBin, readGridBin, readGroundTruthBin } from "./bin-io.ts";

const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";

const CLASS_NAMES = ["Empty/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"];
const CLASS_SHORT = ["Empty", "Settle", "Port", "Ruin", "Forest", "Mtn"];

function klDivergence(p: number[], q: number[]): number {
	let k = 0;
	for (let i = 0; i < p.length; i++) if (p[i] > 0) k += p[i] * Math.log(p[i] / Math.max(q[i], 1e-10));
	return k;
}

function computeScore(gt: number[][][], pred: number[][][], W: number, H: number): number {
	let tE = 0, tWK = 0;
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		const e = entropy(gt[y][x]);
		if (e < 1e-6) continue;
		tE += e;
		tWK += e * klDivergence(gt[y][x], pred[y][x]);
	}
	const wk = tE > 0 ? tWK / tE : 0;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wk)));
}

// --- Data structures ---

interface CellError {
	round: number;
	seed: number;
	x: number;
	y: number;
	initialTerrain: number;
	gt: number[];
	sim: number[];
	cellEntropy: number;
	cellKL: number;
	weightedKL: number;
	perClassKL: number[]; // p_i * log(p_i / q_i) for each class
}

// --- Formatting helpers ---

function pad(s: string, n: number, align: "left" | "right" = "right"): string {
	if (align === "left") return s.padEnd(n);
	return s.padStart(n);
}

function fmtPct(v: number, decimals = 1): string {
	return (v * 100).toFixed(decimals) + "%";
}

function fmtFloat(v: number, decimals = 4): string {
	return v.toFixed(decimals);
}

function printTable(headers: string[], rows: string[][], colWidths?: number[]) {
	const widths = colWidths ?? headers.map((h, i) => Math.max(h.length, ...rows.map(r => (r[i] || "").length)));
	const headerLine = headers.map((h, i) => pad(h, widths[i], i === 0 ? "left" : "right")).join("  ");
	console.log(headerLine);
	console.log("-".repeat(headerLine.length));
	for (const row of rows) {
		console.log(row.map((c, i) => pad(c, widths[i], i === 0 ? "left" : "right")).join("  "));
	}
}

function printSectionHeader(title: string) {
	console.log("");
	console.log("=".repeat(70));
	console.log(`  ${title}`);
	console.log("=".repeat(70));
}

// --- Main analysis ---

function loadData(roundFilter?: number) {
	const gridsPath = `${BIN_DIR}/grids.bin`;
	const gtPath = `${BIN_DIR}/ground_truth.bin`;

	// Discover rounds
	const gridsData = Deno.readFileSync(gridsPath);
	const gridsView = new DataView(gridsData.buffer);
	let offset = 6;
	const gridCount = gridsView.getUint32(offset, true); offset += 4;
	const roundSeedSet = new Set<string>();
	for (let i = 0; i < gridCount; i++) {
		const round = gridsView.getInt32(offset, true); offset += 4;
		const seed = gridsView.getInt32(offset, true); offset += 4;
		const W = gridsView.getInt32(offset, true); offset += 4;
		const H = gridsView.getInt32(offset, true); offset += 4;
		roundSeedSet.add(`${round}:${seed}`);
		offset += W * H * 4;
	}

	const allErrors: CellError[] = [];
	const roundsLoaded: number[] = [];

	const roundNums = [...new Set([...roundSeedSet].map(s => parseInt(s.split(":")[0])))].sort((a, b) => a - b);

	for (const roundNum of roundNums) {
		if (roundFilter !== undefined && roundNum !== roundFilter) continue;

		for (let seed = 0; seed < 5; seed++) {
			// Check if we have both sim prediction and ground truth
			let simPred: number[][][] | null = null;
			try {
				simPred = readPredictionBin(`${BIN_DIR}/pred_sim_r${roundNum}_s${seed}.bin`).prediction;
			} catch { continue; }

			const gt = readGroundTruthBin(gtPath, roundNum, seed);
			if (!gt) continue;

			const gridData = readGridBin(gridsPath, roundNum, seed);
			if (!gridData) continue;

			const { W, H, grid } = gridData;

			if (!roundsLoaded.includes(roundNum)) roundsLoaded.push(roundNum);

			for (let y = 0; y < H; y++) {
				for (let x = 0; x < W; x++) {
					const gtCell = gt[y][x];
					const simCell = simPred[y][x];
					const e = entropy(gtCell);
					if (e < 1e-6) continue; // skip static cells

					const kl = klDivergence(gtCell, simCell);
					const wkl = e * kl;

					// Per-class KL contributions
					const perClassKL: number[] = [];
					for (let c = 0; c < NUM_CLASSES; c++) {
						if (gtCell[c] > 0) {
							perClassKL.push(gtCell[c] * Math.log(gtCell[c] / Math.max(simCell[c], 1e-10)));
						} else {
							perClassKL.push(0);
						}
					}

					allErrors.push({
						round: roundNum,
						seed,
						x, y,
						initialTerrain: terrainToClass(grid[y][x]),
						gt: gtCell,
						sim: simCell,
						cellEntropy: e,
						cellKL: kl,
						weightedKL: wkl,
						perClassKL,
					});
				}
			}
		}
	}

	return { allErrors, roundsLoaded };
}

function analyzePerClassError(errors: CellError[]) {
	printSectionHeader("1. PER-CLASS KL CONTRIBUTION (which class hurts us most?)");

	// For each class, sum up weighted per-class KL: entropy_i * perClassKL_c
	const classWeightedKL = new Array(NUM_CLASSES).fill(0);
	const classTotalKL = new Array(NUM_CLASSES).fill(0);
	let totalWeightedKL = 0;

	for (const err of errors) {
		for (let c = 0; c < NUM_CLASSES; c++) {
			const wkl_c = err.cellEntropy * err.perClassKL[c];
			classWeightedKL[c] += wkl_c;
			classTotalKL[c] += err.perClassKL[c];
		}
		totalWeightedKL += err.weightedKL;
	}

	const totalEntropy = errors.reduce((s, e) => s + e.cellEntropy, 0);
	const avgWeightedKL = totalEntropy > 0 ? totalWeightedKL / totalEntropy : 0;
	const simScore = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWeightedKL)));

	console.log(`  Total dynamic cells: ${errors.length}`);
	console.log(`  Overall sim score: ${simScore.toFixed(1)}`);
	console.log(`  Avg weighted KL: ${avgWeightedKL.toFixed(4)}`);
	console.log("");

	const rows: string[][] = [];
	for (let c = 0; c < NUM_CLASSES; c++) {
		const pctOfTotal = totalWeightedKL > 0 ? classWeightedKL[c] / totalWeightedKL * 100 : 0;
		rows.push([
			CLASS_NAMES[c],
			fmtFloat(classWeightedKL[c], 4),
			pctOfTotal.toFixed(1) + "%",
			fmtFloat(classTotalKL[c] / errors.length, 6),
		]);
	}
	rows.push(["TOTAL", fmtFloat(totalWeightedKL, 4), "100.0%", fmtFloat(errors.reduce((s, e) => s + e.cellKL, 0) / errors.length, 6)]);

	printTable(
		["Class", "Sum wKL", "% of Total", "Avg KL/cell"],
		rows,
	);
}

function analyzeDirectionalBias(errors: CellError[]) {
	printSectionHeader("2. DIRECTIONAL BIAS (sim_prob - gt_prob, positive = over-predicting)");

	const classBias = new Array(NUM_CLASSES).fill(0);
	const classAbsBias = new Array(NUM_CLASSES).fill(0);
	const classGtMean = new Array(NUM_CLASSES).fill(0);
	const classSimMean = new Array(NUM_CLASSES).fill(0);

	for (const err of errors) {
		for (let c = 0; c < NUM_CLASSES; c++) {
			const diff = err.sim[c] - err.gt[c];
			classBias[c] += diff;
			classAbsBias[c] += Math.abs(diff);
			classGtMean[c] += err.gt[c];
			classSimMean[c] += err.sim[c];
		}
	}

	const n = errors.length;
	const rows: string[][] = [];
	for (let c = 0; c < NUM_CLASSES; c++) {
		const avgBias = classBias[c] / n;
		const avgAbsBias = classAbsBias[c] / n;
		const avgGt = classGtMean[c] / n;
		const avgSim = classSimMean[c] / n;
		const direction = avgBias > 0.001 ? "OVER" : avgBias < -0.001 ? "UNDER" : "~OK";
		rows.push([
			CLASS_NAMES[c],
			(avgBias >= 0 ? "+" : "") + fmtPct(avgBias, 2),
			fmtPct(avgAbsBias, 2),
			fmtPct(avgGt, 2),
			fmtPct(avgSim, 2),
			direction,
		]);
	}
	printTable(
		["Class", "Avg Bias", "|Avg Bias|", "GT Mean", "Sim Mean", "Dir"],
		rows,
	);
}

function analyzeByInitialTerrain(errors: CellError[]) {
	printSectionHeader("3. ERRORS BY INITIAL TERRAIN (which starting terrain is hardest?)");

	const groups: Map<number, CellError[]> = new Map();
	for (const err of errors) {
		const key = err.initialTerrain;
		if (!groups.has(key)) groups.set(key, []);
		groups.get(key)!.push(err);
	}

	const rows: string[][] = [];
	const sortedKeys = [...groups.keys()].sort((a, b) => {
		const aAvg = groups.get(a)!.reduce((s, e) => s + e.weightedKL, 0) / groups.get(a)!.length;
		const bAvg = groups.get(b)!.reduce((s, e) => s + e.weightedKL, 0) / groups.get(b)!.length;
		return bAvg - aAvg;
	});

	for (const key of sortedKeys) {
		const errs = groups.get(key)!;
		const avgWKL = errs.reduce((s, e) => s + e.weightedKL, 0) / errs.length;
		const avgKL = errs.reduce((s, e) => s + e.cellKL, 0) / errs.length;
		const avgEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0) / errs.length;
		const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const totalEntropyAll = errors.reduce((s, e) => s + e.cellEntropy, 0);
		const totalWKLAll = errors.reduce((s, e) => s + e.weightedKL, 0);

		// Compute equivalent score for just this group
		const groupTotalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const groupTotalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const groupAvgWKL = groupTotalEntropy > 0 ? groupTotalWKL / groupTotalEntropy : 0;
		const groupScore = Math.max(0, Math.min(100, 100 * Math.exp(-3 * groupAvgWKL)));

		rows.push([
			CLASS_NAMES[key],
			errs.length.toString(),
			fmtFloat(avgWKL, 4),
			fmtFloat(avgKL, 4),
			fmtFloat(avgEntropy, 2),
			groupScore.toFixed(1),
			(totalWKL / totalWKLAll * 100).toFixed(1) + "%",
		]);
	}

	printTable(
		["Init Terrain", "Cells", "Avg wKL", "Avg KL", "Avg Ent", "Score", "% Total wKL"],
		rows,
	);

	// Sub-analysis: for each initial terrain, show per-class bias
	console.log("");
	console.log("  Per-class bias by initial terrain (sim - gt, in percentage points):");
	console.log("");

	const biasHeaders = ["Init Terrain", ...CLASS_SHORT];
	const biasRows: string[][] = [];
	for (const key of sortedKeys) {
		const errs = groups.get(key)!;
		const biases: string[] = [CLASS_NAMES[key]];
		for (let c = 0; c < NUM_CLASSES; c++) {
			const avgBias = errs.reduce((s, e) => s + (e.sim[c] - e.gt[c]), 0) / errs.length;
			const formatted = (avgBias >= 0 ? "+" : "") + (avgBias * 100).toFixed(1);
			biases.push(formatted);
		}
		biasRows.push(biases);
	}
	printTable(biasHeaders, biasRows);
}

function analyzeSpatialPatterns(errors: CellError[]) {
	printSectionHeader("4. SPATIAL PATTERNS (error by grid quadrant)");

	// Group by 4x4 quadrants (each cell is in quadrant x/10, y/10 for a 40x40 grid)
	const quadrants: Map<string, CellError[]> = new Map();
	for (const err of errors) {
		const qx = Math.floor(err.x / 10);
		const qy = Math.floor(err.y / 10);
		const key = `${qx},${qy}`;
		if (!quadrants.has(key)) quadrants.set(key, []);
		quadrants.get(key)!.push(err);
	}

	// Display as a 4x4 grid of average weighted KL
	console.log("  Average weighted KL by quadrant (rows=y, cols=x):");
	console.log("");

	const header = "       " + [0, 1, 2, 3].map(i => pad(`x${i * 10}-${(i + 1) * 10 - 1}`, 10)).join("  ");
	console.log(header);
	console.log("  " + "-".repeat(header.length - 2));

	for (let qy = 0; qy < 4; qy++) {
		let line = `  y${qy * 10}-${(qy + 1) * 10 - 1}`.padEnd(7);
		for (let qx = 0; qx < 4; qx++) {
			const key = `${qx},${qy}`;
			const errs = quadrants.get(key) || [];
			if (errs.length > 0) {
				const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
				const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
				const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
				const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));
				line += pad(`${score.toFixed(1)} (${errs.length})`, 10) + "  ";
			} else {
				line += pad("-", 10) + "  ";
			}
		}
		console.log(line);
	}

	// Also show as sorted list
	console.log("");
	console.log("  Quadrants ranked by error (worst first):");
	const qRows: string[][] = [];
	const sortedQ = [...quadrants.entries()].sort((a, b) => {
		const aWKL = a[1].reduce((s, e) => s + e.weightedKL, 0);
		const bWKL = b[1].reduce((s, e) => s + e.weightedKL, 0);
		return bWKL - aWKL;
	});

	for (const [key, errs] of sortedQ) {
		const [qx, qy] = key.split(",").map(Number);
		const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
		const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));
		qRows.push([
			`(${qx * 10}-${(qx + 1) * 10 - 1}, ${qy * 10}-${(qy + 1) * 10 - 1})`,
			errs.length.toString(),
			score.toFixed(1),
			fmtFloat(avgWKL, 4),
			fmtFloat(totalWKL, 4),
		]);
	}
	printTable(["Quadrant (x, y)", "Cells", "Score", "Avg wKL", "Total wKL"], qRows);
}

function analyzeWorstCells(errors: CellError[], topN = 20) {
	printSectionHeader(`5. TOP ${topN} WORST CELLS (highest weighted KL)`);

	const sorted = [...errors].sort((a, b) => b.weightedKL - a.weightedKL);
	const worst = sorted.slice(0, topN);

	const rows: string[][] = [];
	for (const err of worst) {
		const gtStr = err.gt.map((v, i) => v > 0.01 ? `${CLASS_SHORT[i]}:${fmtPct(v, 0)}` : "").filter(Boolean).join(" ");
		const simStr = err.sim.map((v, i) => v > 0.01 ? `${CLASS_SHORT[i]}:${fmtPct(v, 0)}` : "").filter(Boolean).join(" ");
		rows.push([
			`R${err.round}S${err.seed}`,
			`(${err.x},${err.y})`,
			CLASS_SHORT[err.initialTerrain],
			fmtFloat(err.weightedKL, 4),
			fmtFloat(err.cellKL, 4),
			fmtFloat(err.cellEntropy, 2),
			gtStr,
			simStr,
		]);
	}
	printTable(
		["Round", "Pos", "Init", "wKL", "KL", "Ent", "GT Distribution", "Sim Distribution"],
		rows,
	);
}

function analyzeRoundByRound(errors: CellError[]) {
	printSectionHeader("6. ROUND-BY-ROUND ANALYSIS");

	const roundGroups: Map<number, CellError[]> = new Map();
	for (const err of errors) {
		if (!roundGroups.has(err.round)) roundGroups.set(err.round, []);
		roundGroups.get(err.round)!.push(err);
	}

	const roundNums = [...roundGroups.keys()].sort((a, b) => a - b);

	// Overview table
	console.log("  Overview:");
	const overviewRows: string[][] = [];
	for (const r of roundNums) {
		const errs = roundGroups.get(r)!;
		const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
		const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));
		overviewRows.push([
			`R${r}`,
			errs.length.toString(),
			score.toFixed(1),
			fmtFloat(avgWKL, 4),
			fmtFloat(totalWKL, 4),
		]);
	}
	printTable(["Round", "Dyn Cells", "Score", "Avg wKL", "Total wKL"], overviewRows);

	// Per-round, per-class weighted KL contribution
	console.log("");
	console.log("  Per-class weighted KL % by round:");
	const classHeaders = ["Round", ...CLASS_SHORT, "Score"];
	const classRows: string[][] = [];
	for (const r of roundNums) {
		const errs = roundGroups.get(r)!;
		const classWKL = new Array(NUM_CLASSES).fill(0);
		let totalWKL = 0;
		for (const err of errs) {
			for (let c = 0; c < NUM_CLASSES; c++) {
				classWKL[c] += err.cellEntropy * err.perClassKL[c];
			}
			totalWKL += err.weightedKL;
		}
		const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
		const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));

		const row = [`R${r}`];
		for (let c = 0; c < NUM_CLASSES; c++) {
			const pct = totalWKL > 0 ? classWKL[c] / totalWKL * 100 : 0;
			row.push(pct.toFixed(1) + "%");
		}
		row.push(score.toFixed(1));
		classRows.push(row);
	}
	printTable(classHeaders, classRows);

	// Per-round directional bias
	console.log("");
	console.log("  Per-class bias by round (sim - gt, pct points):");
	const biasHeaders = ["Round", ...CLASS_SHORT];
	const biasRows: string[][] = [];
	for (const r of roundNums) {
		const errs = roundGroups.get(r)!;
		const row = [`R${r}`];
		for (let c = 0; c < NUM_CLASSES; c++) {
			const avgBias = errs.reduce((s, e) => s + (e.sim[c] - e.gt[c]), 0) / errs.length;
			row.push((avgBias >= 0 ? "+" : "") + (avgBias * 100).toFixed(1));
		}
		biasRows.push(row);
	}
	printTable(biasHeaders, biasRows);
}

function analyzeConfusionPatterns(errors: CellError[]) {
	printSectionHeader("7. CONFUSION PATTERNS (where does sim probability go vs GT?)");
	console.log("  For cells where GT has a dominant class (>50%), where does sim put probability?");
	console.log("");

	for (let gtClass = 0; gtClass < NUM_CLASSES; gtClass++) {
		// Cells where GT has this class as dominant (>50%)
		const relevant = errors.filter(e => e.gt[gtClass] > 0.5);
		if (relevant.length < 5) continue;

		const avgGt = new Array(NUM_CLASSES).fill(0);
		const avgSim = new Array(NUM_CLASSES).fill(0);
		for (const err of relevant) {
			for (let c = 0; c < NUM_CLASSES; c++) {
				avgGt[c] += err.gt[c];
				avgSim[c] += err.sim[c];
			}
		}
		const n = relevant.length;
		const avgWKL = relevant.reduce((s, e) => s + e.weightedKL, 0) / n;

		console.log(`  GT dominant: ${CLASS_NAMES[gtClass]} (${n} cells, avg wKL=${fmtFloat(avgWKL, 4)})`);
		const rows: string[][] = [];
		for (let c = 0; c < NUM_CLASSES; c++) {
			const gt = avgGt[c] / n;
			const sim = avgSim[c] / n;
			if (gt < 0.005 && sim < 0.005) continue;
			rows.push([
				CLASS_SHORT[c],
				fmtPct(gt, 1),
				fmtPct(sim, 1),
				(sim - gt >= 0 ? "+" : "") + fmtPct(sim - gt, 1),
			]);
		}
		printTable(["Class", "GT Avg", "Sim Avg", "Diff"], rows);
		console.log("");
	}
}

function analyzeEntropyBands(errors: CellError[]) {
	printSectionHeader("8. ERROR BY ENTROPY BAND (easy vs hard cells)");

	const bands = [
		{ label: "Low (0-0.5)", min: 0, max: 0.5 },
		{ label: "Med-Low (0.5-1.0)", min: 0.5, max: 1.0 },
		{ label: "Medium (1.0-1.5)", min: 1.0, max: 1.5 },
		{ label: "Med-High (1.5-2.0)", min: 1.5, max: 2.0 },
		{ label: "High (2.0+)", min: 2.0, max: Infinity },
	];

	const rows: string[][] = [];
	for (const band of bands) {
		const errs = errors.filter(e => e.cellEntropy >= band.min && e.cellEntropy < band.max);
		if (errs.length === 0) {
			rows.push([band.label, "0", "-", "-", "-", "-"]);
			continue;
		}
		const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
		const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));
		const avgKL = errs.reduce((s, e) => s + e.cellKL, 0) / errs.length;
		const allTotalWKL = errors.reduce((s, e) => s + e.weightedKL, 0);

		rows.push([
			band.label,
			errs.length.toString(),
			score.toFixed(1),
			fmtFloat(avgKL, 4),
			fmtFloat(totalWKL, 4),
			(totalWKL / allTotalWKL * 100).toFixed(1) + "%",
		]);
	}
	printTable(["Entropy Band", "Cells", "Score", "Avg KL", "Total wKL", "% Total wKL"], rows);
}

// --- Main ---

const roundArg = Deno.args[0] ? parseInt(Deno.args[0]) : undefined;

if (roundArg !== undefined) {
	console.log(`Analyzing sim errors for round ${roundArg}`);
} else {
	console.log("Analyzing sim errors for all rounds with ground truth (R1-R9)");
}

const { allErrors, roundsLoaded } = loadData(roundArg);

if (allErrors.length === 0) {
	console.error("No data found! Make sure sim predictions and ground truth exist.");
	Deno.exit(1);
}

console.log(`Loaded ${allErrors.length} dynamic cells across rounds: ${roundsLoaded.join(", ")}`);

// Also show per-seed scores for context
if (roundArg === undefined) {
	console.log("");
	console.log("Per-seed sim scores:");
	const roundSeedMap: Map<string, CellError[]> = new Map();
	for (const err of allErrors) {
		const key = `${err.round}:${err.seed}`;
		if (!roundSeedMap.has(key)) roundSeedMap.set(key, []);
		roundSeedMap.get(key)!.push(err);
	}
	for (const [key, errs] of [...roundSeedMap.entries()].sort()) {
		const totalEntropy = errs.reduce((s, e) => s + e.cellEntropy, 0);
		const totalWKL = errs.reduce((s, e) => s + e.weightedKL, 0);
		const avgWKL = totalEntropy > 0 ? totalWKL / totalEntropy : 0;
		const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * avgWKL)));
		const [r, s] = key.split(":");
		process.stdout.write(`  R${r}S${s}: ${score.toFixed(1)}  `);
	}
	console.log("");
}

analyzePerClassError(allErrors);
analyzeDirectionalBias(allErrors);
analyzeByInitialTerrain(allErrors);
analyzeSpatialPatterns(allErrors);
analyzeWorstCells(allErrors, 20);
analyzeRoundByRound(allErrors);
analyzeConfusionPatterns(allErrors);
analyzeEntropyBands(allErrors);

console.log("");
console.log("Done.");
