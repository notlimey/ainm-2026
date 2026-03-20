/**
 * Per-class KL divergence analysis: which model is best at which terrain class?
 * Helps determine optimal per-class ensemble weights.
 */

const BIN_DIR = "../simulation/data";
const NUM_CLASSES = 6;
const CLASS_NAMES = ["Empty/Ocean", "Settlement", "Port", "Ruin", "Forest", "Mountain"];

function readPredictionBin(path: string) {
	const data = Deno.readFileSync(path);
	const view = new DataView(data.buffer);
	const round = view.getInt32(6, true);
	const seed = view.getInt32(10, true);
	const W = view.getInt32(14, true);
	const H = view.getInt32(18, true);
	const prediction: number[][][] = [];
	let offset = 22;
	for (let y = 0; y < H; y++) {
		const row: number[][] = [];
		for (let x = 0; x < W; x++) {
			const cell: number[] = [];
			for (let c = 0; c < NUM_CLASSES; c++) { cell.push(view.getFloat32(offset, true)); offset += 4; }
			row.push(cell);
		}
		prediction.push(row);
	}
	return { round, seed, W, H, prediction };
}

function readGroundTruthBin(path: string, wantRound: number, wantSeed: number) {
	try {
		const data = Deno.readFileSync(path);
		const view = new DataView(data.buffer);
		let offset = 6;
		const count = view.getUint32(offset, true); offset += 4;
		for (let i = 0; i < count; i++) {
			const round = view.getInt32(offset, true); offset += 4;
			const seed = view.getInt32(offset, true); offset += 4;
			const W = view.getInt32(offset, true); offset += 4;
			const H = view.getInt32(offset, true); offset += 4;
			if (round === wantRound && seed === wantSeed) {
				const gt: number[][][] = [];
				for (let y = 0; y < H; y++) { const row: number[][] = []; for (let x = 0; x < W; x++) { const cell: number[] = []; for (let c = 0; c < NUM_CLASSES; c++) { cell.push(view.getFloat32(offset, true)); offset += 4; } row.push(cell); } gt.push(row); }
				return { W, H, gt };
			}
			offset += W * H * NUM_CLASSES * 4;
		}
	} catch { /* */ }
	return null;
}

// Per-class weighted KL: for each cell, weight by ground truth entropy,
// but break down KL contribution by which class dominates ground truth
interface ClassStats {
	totalWeightedKL: number;
	totalEntropy: number;
	cellCount: number;
}

function analyzePerClass(
	gt: number[][][], pred: number[][][], W: number, H: number
): ClassStats[] {
	const stats: ClassStats[] = Array.from({ length: NUM_CLASSES }, () => ({
		totalWeightedKL: 0, totalEntropy: 0, cellCount: 0
	}));

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const p = gt[y][x];
			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++) if (p[c] > 0) ent -= p[c] * Math.log(p[c]);
			if (ent < 1e-6) continue; // skip static cells

			// Dominant class = argmax of ground truth
			let dominant = 0;
			for (let c = 1; c < NUM_CLASSES; c++) if (p[c] > p[dominant]) dominant = c;

			let kl = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				if (p[c] > 0) kl += p[c] * Math.log(p[c] / Math.max(pred[y][x][c], 1e-10));
			}

			stats[dominant].totalWeightedKL += ent * kl;
			stats[dominant].totalEntropy += ent;
			stats[dominant].cellCount++;
		}
	}
	return stats;
}

// Discover rounds
const rounds: number[] = [];
for (let r = 1; r <= 15; r++) {
	const gt = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, r, 0);
	if (gt) rounds.push(r);
}
console.log(`Rounds with ground truth: ${rounds.join(", ")}\n`);

// Aggregate per-class stats across all rounds/seeds for each model
const models = ["bucket", "mlp", "sim"] as const;
type ModelName = typeof models[number];

const globalStats: Record<ModelName, ClassStats[]> = {
	bucket: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
	mlp: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
	sim: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
};

// Also track per-round for detail
const perRoundScores: Record<number, Record<ModelName, ClassStats[]>> = {};

for (const r of rounds) {
	perRoundScores[r] = {
		bucket: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
		mlp: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
		sim: Array.from({ length: NUM_CLASSES }, () => ({ totalWeightedKL: 0, totalEntropy: 0, cellCount: 0 })),
	};

	for (let s = 0; s < 5; s++) {
		const gtData = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, r, s);
		if (!gtData) continue;

		for (const model of models) {
			try {
				const pred = readPredictionBin(`${BIN_DIR}/pred_${model}_r${r}_s${s}.bin`);
				const classStats = analyzePerClass(gtData.gt, pred.prediction, gtData.W, gtData.H);
				for (let c = 0; c < NUM_CLASSES; c++) {
					globalStats[model][c].totalWeightedKL += classStats[c].totalWeightedKL;
					globalStats[model][c].totalEntropy += classStats[c].totalEntropy;
					globalStats[model][c].cellCount += classStats[c].cellCount;
					perRoundScores[r][model][c].totalWeightedKL += classStats[c].totalWeightedKL;
					perRoundScores[r][model][c].totalEntropy += classStats[c].totalEntropy;
					perRoundScores[r][model][c].cellCount += classStats[c].cellCount;
				}
			} catch { /* prediction file missing */ }
		}
	}
}

// Print results
function klToScore(totalWK: number, totalE: number): number {
	if (totalE === 0) return -1;
	const wk = totalWK / totalE;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wk)));
}

console.log("=== GLOBAL PER-CLASS SCORES (all rounds combined) ===\n");
const header = ["Class", "Cells", "Entropy%", "Bucket", "MLP", "Sim", "Best"].map(s => s.padEnd(14)).join("");
console.log(header);
console.log("-".repeat(header.length));

let totalEntropy = 0;
for (let c = 0; c < NUM_CLASSES; c++) totalEntropy += globalStats.bucket[c].totalEntropy;

for (let c = 0; c < NUM_CLASSES; c++) {
	const cells = globalStats.bucket[c].cellCount;
	const entPct = totalEntropy > 0 ? (100 * globalStats.bucket[c].totalEntropy / totalEntropy).toFixed(1) + "%" : "0%";
	const scores = models.map(m => klToScore(globalStats[m][c].totalWeightedKL, globalStats[m][c].totalEntropy));
	const validScores = scores.filter(s => s >= 0);
	const bestIdx = validScores.length > 0 ? scores.indexOf(Math.max(...validScores)) : -1;
	const best = bestIdx >= 0 ? models[bestIdx] : "N/A";

	const row = [
		CLASS_NAMES[c].padEnd(14),
		String(cells).padEnd(14),
		entPct.padEnd(14),
		...scores.map(s => s >= 0 ? s.toFixed(1).padEnd(14) : "N/A".padEnd(14)),
		best.toUpperCase().padEnd(14),
	].join("");
	console.log(row);
}

// Overall
console.log("-".repeat(header.length));
const overallScores = models.map(m => {
	let twk = 0, te = 0;
	for (let c = 0; c < NUM_CLASSES; c++) { twk += globalStats[m][c].totalWeightedKL; te += globalStats[m][c].totalEntropy; }
	return klToScore(twk, te);
});
const bestOverall = models[overallScores.indexOf(Math.max(...overallScores))];
console.log([
	"OVERALL".padEnd(14), "".padEnd(14), "100%".padEnd(14),
	...overallScores.map(s => s.toFixed(1).padEnd(14)),
	bestOverall.toUpperCase().padEnd(14),
].join(""));

// Per-round breakdown
console.log("\n\n=== PER-ROUND SCORES BY DOMINANT CLASS ===\n");
for (const r of rounds) {
	console.log(`--- Round ${r} ---`);
	const rHeader = ["Class", "Bucket", "MLP", "Sim", "Best"].map(s => s.padEnd(14)).join("");
	console.log(rHeader);
	for (let c = 0; c < NUM_CLASSES; c++) {
		const cells = perRoundScores[r].bucket[c].cellCount;
		if (cells === 0) continue;
		const scores = models.map(m => klToScore(perRoundScores[r][m][c].totalWeightedKL, perRoundScores[r][m][c].totalEntropy));
		const validScores2 = scores.filter(s => s >= 0);
		const bestIdx = validScores2.length > 0 ? scores.indexOf(Math.max(...validScores2)) : -1;
		const bestModel = bestIdx >= 0 ? models[bestIdx] : "N/A";
		console.log([
			CLASS_NAMES[c].padEnd(14),
			...scores.map(s => s >= 0 ? s.toFixed(1).padEnd(14) : "N/A".padEnd(14)),
			bestModel.toUpperCase().padEnd(14),
		].join(""));
	}
	console.log("");
}

// Optimal per-class weights via grid search
console.log("\n=== OPTIMAL PER-CLASS BLEND (grid search) ===\n");
console.log("Testing blend weights [bucket, mlp, sim] per class...\n");

// We need per-cell data for blending, so let's do a proper blend test
// Load all predictions and ground truth, try different per-class weights
const WEIGHT_STEPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

interface CellData {
	gt: number[];
	bucket: number[];
	mlp: number[];
	sim: number[];
	entropy: number;
}

const allCells: CellData[] = [];

for (const r of rounds) {
	for (let s = 0; s < 5; s++) {
		const gtData = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, r, s);
		if (!gtData) continue;
		let bucketPred: number[][][] | null = null;
		let mlpPred: number[][][] | null = null;
		let simPred: number[][][] | null = null;
		try { bucketPred = readPredictionBin(`${BIN_DIR}/pred_bucket_r${r}_s${s}.bin`).prediction; } catch { /**/ }
		try { mlpPred = readPredictionBin(`${BIN_DIR}/pred_mlp_r${r}_s${s}.bin`).prediction; } catch { /**/ }
		try { simPred = readPredictionBin(`${BIN_DIR}/pred_sim_r${r}_s${s}.bin`).prediction; } catch { /**/ }
		if (!bucketPred || !mlpPred || !simPred) continue;

		for (let y = 0; y < gtData.H; y++) {
			for (let x = 0; x < gtData.W; x++) {
				const p = gtData.gt[y][x];
				let ent = 0;
				for (let c = 0; c < NUM_CLASSES; c++) if (p[c] > 0) ent -= p[c] * Math.log(p[c]);
				if (ent < 1e-6) continue;
				allCells.push({
					gt: p,
					bucket: bucketPred[y][x],
					mlp: mlpPred[y][x],
					sim: simPred[y][x],
					entropy: ent,
				});
			}
		}
	}
}

console.log(`Dynamic cells loaded: ${allCells.length}\n`);

// Test uniform blend first
function evalBlend(wb: number, wm: number, ws: number, cells: CellData[]): number {
	let tE = 0, tWK = 0;
	for (const cell of cells) {
		const q: number[] = [];
		let total = 0;
		for (let c = 0; c < NUM_CLASSES; c++) {
			const v = wb * cell.bucket[c] + wm * cell.mlp[c] + ws * cell.sim[c];
			q.push(Math.max(v, 1e-10));
			total += Math.max(v, 1e-10);
		}
		for (let c = 0; c < NUM_CLASSES; c++) q[c] /= total;

		let kl = 0;
		for (let c = 0; c < NUM_CLASSES; c++) {
			if (cell.gt[c] > 0) kl += cell.gt[c] * Math.log(cell.gt[c] / q[c]);
		}
		tE += cell.entropy;
		tWK += cell.entropy * kl;
	}
	return tE > 0 ? Math.max(0, Math.min(100, 100 * Math.exp(-3 * tWK / tE))) : 0;
}

// Grid search for best uniform weights
let bestScore = 0;
let bestW = [0, 0, 0];
for (const wb of WEIGHT_STEPS) {
	for (const wm of WEIGHT_STEPS) {
		const ws = Math.round((1 - wb - wm) * 10) / 10;
		if (ws < 0 || ws > 1) continue;
		const score = evalBlend(wb, wm, ws, allCells);
		if (score > bestScore) {
			bestScore = score;
			bestW = [wb, wm, ws];
		}
	}
}

console.log(`Best UNIFORM blend: bucket=${bestW[0]} mlp=${bestW[1]} sim=${bestW[2]} → score ${bestScore.toFixed(2)}`);
console.log(`  vs pure bucket: ${evalBlend(1, 0, 0, allCells).toFixed(2)}`);
console.log(`  vs pure mlp:    ${evalBlend(0, 1, 0, allCells).toFixed(2)}`);
console.log(`  vs pure sim:    ${evalBlend(0, 0, 1, allCells).toFixed(2)}`);
console.log(`  vs old 70/30:   ${evalBlend(0.7, 0.3, 0, allCells).toFixed(2)}`);

// Per-class optimal: group cells by dominant GT class, find best blend per class
console.log("\n--- Per-class optimal weights ---");
for (let cls = 0; cls < NUM_CLASSES; cls++) {
	const classCells = allCells.filter(cell => {
		let dominant = 0;
		for (let c = 1; c < NUM_CLASSES; c++) if (cell.gt[c] > cell.gt[dominant]) dominant = c;
		return dominant === cls;
	});
	if (classCells.length === 0) continue;

	let best = 0;
	let bw = [0, 0, 0];
	for (const wb of WEIGHT_STEPS) {
		for (const wm of WEIGHT_STEPS) {
			const ws = Math.round((1 - wb - wm) * 10) / 10;
			if (ws < 0 || ws > 1) continue;
			const score = evalBlend(wb, wm, ws, classCells);
			if (score > best) { best = score; bw = [wb, wm, ws]; }
		}
	}
	console.log(`  ${CLASS_NAMES[cls].padEnd(16)} → bucket=${bw[0]} mlp=${bw[1]} sim=${bw[2]}  score=${best.toFixed(1)}  (${classCells.length} cells)`);
}
