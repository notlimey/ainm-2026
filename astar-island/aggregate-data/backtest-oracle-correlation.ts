// backtest-oracle-correlation.ts — Does the query-based oracle actually pick the GT-best model?
// For each round with queries + GT, compare query ranking vs GT ranking.
//
// Usage: deno run -A backtest-oracle-correlation.ts

import { readPrediction, readGroundTruthBin, terrainToClass, NUM_CLASSES } from "./bin-io.ts";

const MODELS = ["sim", "tuned", "bucket", "mlp", "cnn", "blend"] as const;
const GT_PATH = "../simulation/data/ground_truth.bin";
const DATA_DIR = "../simulation/data";

interface Query {
	viewport: { x: number; y: number; w: number; h: number };
	grid: number[][];
}

function loadQueries(round: number, seed: number): Query[] {
	try { return JSON.parse(Deno.readTextFileSync(`data/queries/r${round}/s${seed}_queries.json`)); }
	catch { return []; }
}

function scoreVsQueries(pred: number[][][], queries: Query[], W: number, H: number): number {
	const counts = Array.from({ length: H }, () => Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0)));
	const samp = Array.from({ length: H }, () => new Array(W).fill(0));
	for (const q of queries) {
		const { x, y, w, h } = q.viewport;
		for (let r = 0; r < h && y + r < H; r++)
			for (let c = 0; c < w && x + c < W; c++) {
				counts[y + r][x + c][terrainToClass(q.grid[r][c])]++;
				samp[y + r][x + c]++;
			}
	}
	let twkl = 0, tent = 0;
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		if (samp[y][x] < 2) continue;
		const obs = counts[y][x].map(c => c / samp[y][x]);
		let ent = 0;
		for (let c = 0; c < NUM_CLASSES; c++) if (obs[c] > 0) ent -= obs[c] * Math.log(obs[c]);
		if (ent < 1e-6) continue;
		let kl = 0;
		for (let c = 0; c < NUM_CLASSES; c++)
			if (obs[c] > 1e-10) kl += obs[c] * Math.log(obs[c] / Math.max(pred[y][x][c], 1e-10));
		twkl += ent * kl; tent += ent;
	}
	return tent < 1e-10 ? 100 : Math.max(0, Math.min(100, 100 * Math.exp(-3 * twkl / tent)));
}

function scoreVsGT(pred: number[][][], gt: number[][][], W: number, H: number): number {
	let twkl = 0, tent = 0;
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		let ent = 0;
		for (let c = 0; c < NUM_CLASSES; c++) if (gt[y][x][c] > 0) ent -= gt[y][x][c] * Math.log(gt[y][x][c]);
		if (ent < 1e-6) continue;
		let kl = 0;
		for (let c = 0; c < NUM_CLASSES; c++)
			if (gt[y][x][c] > 0) kl += gt[y][x][c] * Math.log(gt[y][x][c] / Math.max(pred[y][x][c], 1e-10));
		twkl += ent * kl; tent += ent;
	}
	return tent < 1e-10 ? 100 : Math.max(0, Math.min(100, 100 * Math.exp(-3 * twkl / tent)));
}

console.log("\n=== Oracle Correlation: Query Score vs GT Score ===\n");
console.log("Round Seed  QueryBest    GTBest     Match?  QueryPick→GT  GTBest→GT  Gap");
console.log("-".repeat(85));

let matches = 0, total = 0;
let oracleGTsum = 0, gtBestSum = 0, blendGTsum = 0;

for (let round = 1; round <= 30; round++) {
	for (let s = 0; s < 5; s++) {
		const gt = readGroundTruthBin(GT_PATH, round, s);
		const queries = loadQueries(round, s);
		if (!gt || queries.length === 0) continue;

		const qScores: [string, number][] = [];
		const gtScores: [string, number][] = [];

		for (const model of MODELS) {
			try {
				const { prediction, W, H } = readPrediction(`${DATA_DIR}/pred_${model}_r${round}_s${s}.bin`);
				const qs = scoreVsQueries(prediction, queries, W, H);
				const gs = scoreVsGT(prediction, gt, W, H);
				qScores.push([model, qs]);
				gtScores.push([model, gs]);
			} catch { /* not available */ }
		}

		if (qScores.length < 2) continue;

		qScores.sort((a, b) => b[1] - a[1]);
		gtScores.sort((a, b) => b[1] - a[1]);

		const queryBest = qScores[0][0];
		const gtBest = gtScores[0][0];
		const match = queryBest === gtBest;
		if (match) matches++;
		total++;

		// What GT score does the query pick achieve?
		const queryPickGT = gtScores.find(([m]) => m === queryBest)?.[1] ?? 0;
		const gtBestGT = gtScores[0][1];
		const blendGT = gtScores.find(([m]) => m === "blend")?.[1] ?? 0;
		oracleGTsum += queryPickGT;
		gtBestSum += gtBestGT;
		blendGTsum += blendGT;

		const gap = (queryPickGT - gtBestGT).toFixed(1);
		console.log(
			`R${String(round).padEnd(3)} S${s}   ${queryBest.padEnd(10)}  ${gtBest.padEnd(10)} ${match ? "YES" : "no "}    ${queryPickGT.toFixed(1).padStart(6)}       ${gtBestGT.toFixed(1).padStart(6)}    ${gap}`
		);
	}
}

console.log("-".repeat(85));
console.log(`\nQuery oracle picks GT-best: ${matches}/${total} seeds (${(100*matches/total).toFixed(0)}%)`);
console.log(`Avg GT score of query pick: ${(oracleGTsum / total).toFixed(1)}`);
console.log(`Avg GT score of GT-best:    ${(gtBestSum / total).toFixed(1)}`);
console.log(`Avg GT score of blend:      ${(blendGTsum / total).toFixed(1)}`);
console.log(`Oracle regret vs GT-best:   ${((gtBestSum - oracleGTsum) / total).toFixed(1)} pts`);
console.log(`Oracle gain vs always-blend: ${((oracleGTsum - blendGTsum) / total).toFixed(1)} pts`);
console.log();
