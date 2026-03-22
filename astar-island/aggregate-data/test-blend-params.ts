/**
 * Test different Bayesian blending parameters on historical rounds with GT.
 * Helps find optimal pseudocount / alpha settings.
 */
import { NUM_CLASSES, TERRAIN_MOUNTAIN, TERRAIN_OCEAN, terrainToClass, readPrediction, readGridBin, readGroundTruthBin } from "./bin-io.ts";

const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";

interface StoredQuery {
	query_index: number;
	seed_index: number;
	viewport: { x: number; y: number; w: number; h: number };
	grid: number[][];
	settlements: unknown[];
	timestamp: string;
}

async function loadStoredQueries(roundNum: number, seedIdx: number): Promise<StoredQuery[]> {
	try {
		const text = await Deno.readTextFile(`${DATA_DIR}/queries/r${roundNum}/s${seedIdx}_queries.json`);
		return JSON.parse(text);
	} catch { return []; }
}

function accumulateQueries(queries: StoredQuery[], W: number, H: number) {
	const sampleCounts: number[][] = Array.from({ length: H }, () => new Array(W).fill(0));
	const classCounts: number[][][] = Array.from({ length: H }, () =>
		Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0)));
	for (const q of queries) {
		const vp = q.viewport;
		for (let gy = 0; gy < q.grid.length; gy++)
			for (let gx = 0; gx < q.grid[gy].length; gx++) {
				const mapX = vp.x + gx, mapY = vp.y + gy;
				if (mapX >= W || mapY >= H) continue;
				classCounts[mapY][mapX][terrainToClass(q.grid[gy][gx])]++;
				sampleCounts[mapY][mapX]++;
			}
	}
	return { sampleCounts, classCounts };
}

function blendAndScore(
	modelPred: number[][][], gt: number[][][],
	sampleCounts: number[][], classCounts: number[][][],
	W: number, H: number, initialGrid: number[][] | null,
	pseudoRare: number, pseudoCommon: number, alphaCap: number
): number {
	const blended = modelPred.map(row => row.map(cell => [...cell]));

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const n = sampleCounts[y][x];
			if (n === 0) continue;
			const empirical = classCounts[y][x].map(c => c / n);
			for (let c = 0; c < NUM_CLASSES; c++) {
				const pseudoCount = (c >= 1 && c <= 3) ? pseudoRare : pseudoCommon;
				const alpha = Math.min(n / (n + pseudoCount), alphaCap);
				blended[y][x][c] = (1 - alpha) * modelPred[y][x][c] + alpha * empirical[c];
			}
			// Smart floor
			const terrain = initialGrid ? initialGrid[y][x] : -1;
			let total = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				const reachable = terrain === 5 ? (c === 5)
					: terrain === 10 ? (c === 0)
					: terrain >= 0 ? (c !== 5) : true;
				if (reachable) { if (blended[y][x][c] < 0.003) blended[y][x][c] = 0.003; }
				else blended[y][x][c] = 0.0;
				total += blended[y][x][c];
			}
			for (let c = 0; c < NUM_CLASSES; c++) blended[y][x][c] /= total;
		}
	}

	// Score
	let totalEntropy = 0, totalWeightedKL = 0;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++) if (gt[y][x][c] > 0) ent -= gt[y][x][c] * Math.log(gt[y][x][c]);
			if (ent < 1e-6) continue;
			let kl = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (gt[y][x][c] > 0) kl += gt[y][x][c] * Math.log(gt[y][x][c] / Math.max(blended[y][x][c], 1e-10));
			totalEntropy += ent;
			totalWeightedKL += ent * kl;
		}
	}
	const wkl = totalEntropy > 0 ? totalWeightedKL / totalEntropy : 0;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// Test on rounds that have both queries and GT
// Test all rounds that have stored queries
const allRoundDirs: number[] = [];
for (let r = 1; r <= 15; r++) {
	try { Deno.statSync(`${DATA_DIR}/queries/r${r}`); allRoundDirs.push(r); } catch { /**/ }
}
const rounds = allRoundDirs;

const configs = [
	{ pseudoRare: 3, pseudoCommon: 5, alphaCap: 0.85, label: "old (3/5, cap=0.85)" },
	{ pseudoRare: 8, pseudoCommon: 12, alphaCap: 0.70, label: "current (8/12, cap=0.70)" },
	{ pseudoRare: 5, pseudoCommon: 8, alphaCap: 0.80, label: "moderate (5/8, cap=0.80)" },
	{ pseudoRare: 2, pseudoCommon: 3, alphaCap: 0.90, label: "aggressive (2/3, cap=0.90)" },
	{ pseudoRare: -1, pseudoCommon: -1, alphaCap: -1, label: "ADAPTIVE" }, // special marker
];

// Adaptive config: compute from model disagreement on queried cells
function adaptiveParams(modelPred: number[][][], sampleCounts: number[][], W: number, H: number, otherModels: number[][][][] = []) {
	let avgDisagreement = 0;
	if (otherModels.length > 0) {
		const allModels = [modelPred, ...otherModels];
		let totalD = 0, cnt = 0;
		for (let y = 0; y < H; y++) {
			for (let x = 0; x < W; x++) {
				if (sampleCounts[y][x] === 0) continue;
				let pairKL = 0, pairs = 0;
				for (let i = 0; i < allModels.length; i++) {
					for (let j = i + 1; j < allModels.length; j++) {
						for (let c = 0; c < NUM_CLASSES; c++) {
							const p = allModels[i][y][x][c], q = allModels[j][y][x][c];
							if (p > 0.001) pairKL += p * Math.log(p / Math.max(q, 0.001));
							if (q > 0.001) pairKL += q * Math.log(q / Math.max(p, 0.001));
						}
						pairs++;
					}
				}
				totalD += pairs > 0 ? pairKL / pairs : 0;
				cnt++;
			}
		}
		avgDisagreement = cnt > 0 ? totalD / cnt : 0;
	} else {
		// Fallback to entropy
		let queriedEntropy = 0, queriedCount = 0;
		for (let y = 0; y < H; y++) {
			for (let x = 0; x < W; x++) {
				if (sampleCounts[y][x] === 0) continue;
				let ent = 0;
				for (let c = 0; c < NUM_CLASSES; c++) { const p = modelPred[y][x][c]; if (p > 0) ent -= p * Math.log(p); }
				queriedEntropy += ent; queriedCount++;
			}
		}
		avgDisagreement = queriedCount > 0 ? queriedEntropy / queriedCount : 1.0;
	}
	const t = Math.min(1.0, avgDisagreement / 1.5);
	return {
		pseudoRare: Math.round(8 - 5 * t),
		pseudoCommon: Math.round(12 - 7 * t),
		alphaCap: 0.70 + 0.15 * t,
		t, avgDisagreement,
	};
}

for (const round of rounds) {
	console.log(`\n=== Round ${round} ===`);

	for (let seedIdx = 0; seedIdx < 5; seedIdx++) {
		const queries = await loadStoredQueries(round, seedIdx);
		if (queries.length === 0) continue;

		// Load blend prediction (base model)
		let modelPred: number[][][] | null = null;
		let W = 40, H = 40;
		// Use pure sim prediction as base (not query-contaminated blend)
		try {
			const p = readPrediction(`${BIN_DIR}/pred_sim_r${round}_s${seedIdx}.bin`);
			modelPred = p.prediction; W = p.W; H = p.H;
		} catch { /**/ }
		if (!modelPred) continue;

		// Load GT
		const gtData = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, round, seedIdx);
		if (!gtData) continue;

		const gridData = readGridBin(`${BIN_DIR}/grids.bin`, round, seedIdx);
		const initialGrid = gridData ? gridData.grid : null;

		const { sampleCounts, classCounts } = accumulateQueries(queries, W, H);

		// Base score (no query blending)
		const baseScore = blendAndScore(modelPred, gtData, sampleCounts.map(r => r.map(() => 0)),
			classCounts, W, H, initialGrid, 1, 1, 0);

		if (seedIdx === 0) {
			console.log(`  Base sim (no queries): ${baseScore.toFixed(1)}`);
			console.log(`  Queries available: ${queries.length}`);
			console.log(`  ${"Config".padEnd(38)} ${[...Array(5)].map((_, i) => `S${i}`).join("     ")}   Avg`);
		}
	}

	// Now run all configs across all seeds
	for (const cfg of configs) {
		const scores: number[] = [];
		for (let seedIdx = 0; seedIdx < 5; seedIdx++) {
			const queries = await loadStoredQueries(round, seedIdx);
			if (queries.length === 0) { scores.push(-1); continue; }

			let modelPred: number[][][] | null = null;
			let W = 40, H = 40;
			try {
				const p = readPrediction(`${BIN_DIR}/pred_sim_r${round}_s${seedIdx}.bin`);
				modelPred = p.prediction; W = p.W; H = p.H;
			} catch { /**/ }
			if (!modelPred) { scores.push(-1); continue; }

			const gtData = readGroundTruthBin(`${BIN_DIR}/ground_truth.bin`, round, seedIdx);
			if (!gtData) { scores.push(-1); continue; }

			const gridData = readGridBin(`${BIN_DIR}/grids.bin`, round, seedIdx);
			const initialGrid = gridData ? gridData.grid : null;

			const { sampleCounts, classCounts } = accumulateQueries(queries, W, H);
			let pr = cfg.pseudoRare, pc = cfg.pseudoCommon, ac = cfg.alphaCap;
			if (pr < 0) {
				// Adaptive — load other models for disagreement
				const others: number[][][][] = [];
				for (const name of ["bucket", "mlp", "cnn"]) {
					try {
						const p = readPrediction(`${BIN_DIR}/pred_${name}_r${round}_s${seedIdx}.bin`);
						others.push(p.prediction);
					} catch { /**/ }
				}
				const ap = adaptiveParams(modelPred, sampleCounts, W, H, others);
				pr = ap.pseudoRare; pc = ap.pseudoCommon; ac = ap.alphaCap;
			}
			const score = blendAndScore(modelPred, gtData, sampleCounts, classCounts, W, H,
				initialGrid, pr, pc, ac);
			scores.push(score);
		}
		const valid = scores.filter(s => s >= 0);
		const avg = valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
		const seedStrs = scores.map(s => s >= 0 ? s.toFixed(1).padStart(5) : "  N/A").join("  ");
		// For adaptive, also show the params used
		let label = cfg.label;
		if (cfg.pseudoRare < 0 && valid.length > 0) {
			// Show params for seed 0
			const queries0 = await loadStoredQueries(round, 0);
			let mp: number[][][] | null = null;
			try { const p = readPrediction(`${BIN_DIR}/pred_sim_r${round}_s0.bin`); mp = p.prediction; } catch {/**/}
			if (mp && queries0.length > 0) {
				const { sampleCounts: sc0 } = accumulateQueries(queries0, 40, 40);
				const others0: number[][][][] = [];
				for (const name of ["bucket", "mlp", "cnn"]) {
					try { others0.push(readPrediction(`${BIN_DIR}/pred_${name}_r${round}_s0.bin`).prediction); } catch {/**/}
				}
				const ap = adaptiveParams(mp, sc0, 40, 40, others0);
				label = `ADAPTIVE (d=${ap.avgDisagreement.toFixed(2)} t=${ap.t.toFixed(2)} → ${ap.pseudoRare}/${ap.pseudoCommon},${ap.alphaCap.toFixed(2)})`;
			}
		}
		console.log(`  ${label.padEnd(38)} ${seedStrs}  ${avg.toFixed(1).padStart(5)}`);
	}
}
