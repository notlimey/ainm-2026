// backtest-models.ts — Score all models against ground truth for historical rounds.
// Answers: does the oracle selector (which picks based on queries) actually pick the
// model that scores best on ground truth?
//
// Usage: deno run -A backtest-models.ts [--data-dir ../simulation/data]

import { readPrediction, readGroundTruthBin, NUM_CLASSES } from "./bin-io.ts";

const MODELS = ["sim", "tuned", "bucket", "mlp", "cnn", "blend"] as const;
const GT_PATH = "../simulation/data/ground_truth.bin";

function scoreVsGT(pred: number[][][], gt: number[][][], W: number, H: number): number {
	let totalWeightedKL = 0;
	let totalEntropy = 0;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (gt[y][x][c] > 0) ent -= gt[y][x][c] * Math.log(gt[y][x][c]);
			if (ent < 1e-6) continue;
			let kl = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (gt[y][x][c] > 0)
					kl += gt[y][x][c] * Math.log(gt[y][x][c] / Math.max(pred[y][x][c], 1e-10));
			totalWeightedKL += ent * kl;
			totalEntropy += ent;
		}
	}
	if (totalEntropy < 1e-10) return 100;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * totalWeightedKL / totalEntropy)));
}

let dataDir = "../simulation/data";
for (let i = 0; i < Deno.args.length; i++) {
	if (Deno.args[i] === "--data-dir" && Deno.args[i + 1]) dataDir = Deno.args[++i];
}

// Discover which rounds have ground truth
const rounds: number[] = [];
for (let r = 1; r <= 30; r++) {
	const gt = readGroundTruthBin(GT_PATH, r, 0);
	if (gt) rounds.push(r);
}

console.log(`\n=== Model Backtest vs Ground Truth (${rounds.length} rounds) ===\n`);

// Header
const modelNames = [...MODELS];
console.log(
	"Round  " +
	modelNames.map((m) => m.padStart(7)).join("") +
	"  Best     Oracle(blend)"
);
console.log("-".repeat(80));

// Track totals
const modelTotals: Record<string, { sum: number; count: number }> = {};
for (const m of MODELS) modelTotals[m] = { sum: 0, count: 0 };

let blendWins = 0;
let bestWins = 0;
let totalRounds = 0;
let blendBetterSum = 0;
let bestBetterSum = 0;

for (const round of rounds) {
	const seedScores: Record<string, number[]> = {};
	for (const m of MODELS) seedScores[m] = [];

	let validSeeds = 0;
	for (let s = 0; s < 5; s++) {
		const gt = readGroundTruthBin(GT_PATH, round, s);
		if (!gt) continue;
		validSeeds++;

		for (const model of MODELS) {
			const path = `${dataDir}/pred_${model}_r${round}_s${s}.bin`;
			try {
				const { prediction, W, H } = readPrediction(path);
				const score = scoreVsGT(prediction, gt, W, H);
				seedScores[model].push(score);
			} catch {
				seedScores[model].push(-1); // not available
			}
		}
	}

	if (validSeeds === 0) continue;
	totalRounds++;

	// Compute round averages (only counting available seeds)
	const roundAvg: Record<string, number> = {};
	for (const m of MODELS) {
		const valid = seedScores[m].filter((s) => s >= 0);
		roundAvg[m] = valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : -1;
		if (roundAvg[m] >= 0) {
			modelTotals[m].sum += roundAvg[m];
			modelTotals[m].count++;
		}
	}

	// Find best model for this round
	let bestModel = "sim";
	let bestScore = -1;
	for (const m of MODELS) {
		if (roundAvg[m] > bestScore) { bestScore = roundAvg[m]; bestModel = m; }
	}

	const blendScore = roundAvg["blend"] ?? -1;
	if (blendScore >= 0 && bestScore >= 0) {
		if (bestModel === "blend") blendWins++;
		else bestWins++;
		blendBetterSum += blendScore;
		bestBetterSum += bestScore;
	}

	// Format row
	const scores = modelNames.map((m) =>
		roundAvg[m] >= 0 ? roundAvg[m].toFixed(1).padStart(7) : "    N/A"
	);
	const gap = blendScore >= 0 ? (blendScore - bestScore).toFixed(1) : "N/A";
	const marker = bestModel === "blend" ? "  blend" : `  ${bestModel} (${gap})`;
	console.log(`R${String(round).padEnd(4)} ${scores.join("")}${marker}`);
}

console.log("-".repeat(80));

// Averages
const avgRow = modelNames.map((m) => {
	const t = modelTotals[m];
	return t.count > 0 ? (t.sum / t.count).toFixed(1).padStart(7) : "    N/A";
});
console.log(`Avg   ${avgRow.join("")}`);

console.log(`\nBlend is best model: ${blendWins}/${totalRounds} rounds`);
console.log(`Other model is best: ${bestWins}/${totalRounds} rounds`);
if (blendWins + bestWins > 0) {
	console.log(`Avg blend score: ${(blendBetterSum / (blendWins + bestWins)).toFixed(1)}`);
	console.log(`Avg best-model score: ${(bestBetterSum / (blendWins + bestWins)).toFixed(1)}`);
	console.log(`Oracle gain over always-blend: ${((bestBetterSum - blendBetterSum) / (blendWins + bestWins)).toFixed(1)} pts avg`);
}
console.log();
