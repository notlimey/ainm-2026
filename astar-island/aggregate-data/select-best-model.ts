// select-best-model.ts — Score all available models against query observations, pick best per seed.
// Replaces the sim-vs-tuned comparison in auto-round.sh step 7 with full model oracle selection.
//
// Usage: deno run -A select-best-model.ts <round> [--data-dir ../simulation/data] [--per-class]
//
// Models checked: sim, tuned, bucket, mlp, cnn, blend
// Output: copies best prediction to pred_r{R}_s{S}.bin for each seed

import { readPrediction, terrainToClass, NUM_CLASSES } from "./bin-io.ts";

const MODELS = ["sim", "tuned", "bucket", "mlp", "cnn", "blend"] as const;

interface Query {
	viewport: { x: number; y: number; w: number; h: number };
	grid: number[][];
}

function loadQueries(round: number, seed: number): Query[] {
	try {
		return JSON.parse(
			Deno.readTextFileSync(`data/queries/r${round}/s${seed}_queries.json`)
		);
	} catch {
		return [];
	}
}

// Build observed class distribution from multiple query observations
function buildObserved(
	queries: Query[],
	W: number,
	H: number
): { counts: number[][][]; samples: number[][] } {
	const counts = Array.from({ length: H }, () =>
		Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0))
	);
	const samples = Array.from({ length: H }, () => new Array(W).fill(0));
	for (const q of queries) {
		const { x, y, w, h } = q.viewport;
		for (let r = 0; r < h && y + r < H; r++) {
			for (let c = 0; c < w && x + c < W; c++) {
				counts[y + r][x + c][terrainToClass(q.grid[r][c])]++;
				samples[y + r][x + c]++;
			}
		}
	}
	return { counts, samples };
}

// Score a prediction against observed distributions (entropy-weighted KL)
function scorePrediction(
	pred: number[][][],
	counts: number[][][],
	samples: number[][],
	W: number,
	H: number
): number {
	let totalWeightedKL = 0;
	let totalEntropy = 0;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			if (samples[y][x] < 2) continue;
			const n = samples[y][x];
			const obs = counts[y][x].map((c) => c / n);
			// Entropy
			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (obs[c] > 0) ent -= obs[c] * Math.log(obs[c]);
			if (ent < 1e-6) continue;
			// KL divergence
			let kl = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (obs[c] > 1e-10)
					kl += obs[c] * Math.log(obs[c] / Math.max(pred[y][x][c], 1e-10));
			totalWeightedKL += ent * kl;
			totalEntropy += ent;
		}
	}
	if (totalEntropy < 1e-10) return 100;
	const wkl = totalWeightedKL / totalEntropy;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// Per-class scoring: for each class, score how well a model predicts that class
function scorePerClass(
	pred: number[][][],
	counts: number[][][],
	samples: number[][],
	W: number,
	H: number
): number[] {
	const classKL = new Array(NUM_CLASSES).fill(0);
	const classWeight = new Array(NUM_CLASSES).fill(0);
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			if (samples[y][x] < 2) continue;
			const n = samples[y][x];
			const obs = counts[y][x].map((c) => c / n);
			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++)
				if (obs[c] > 0) ent -= obs[c] * Math.log(obs[c]);
			if (ent < 1e-6) continue;
			for (let c = 0; c < NUM_CLASSES; c++) {
				if (obs[c] > 1e-10) {
					const kl_c = obs[c] * Math.log(obs[c] / Math.max(pred[y][x][c], 1e-10));
					classKL[c] += ent * kl_c;
					classWeight[c] += ent * obs[c];
				}
			}
		}
	}
	return classKL.map((kl, c) => (classWeight[c] > 1e-10 ? kl / classWeight[c] : 0));
}

// Build a per-class blended prediction from best models
function buildPerClassPred(
	preds: Map<string, number[][][]>,
	bestModelPerClass: string[],
	W: number,
	H: number
): number[][][] {
	const out: number[][][] = Array.from({ length: H }, () =>
		Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0))
	);
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			let sum = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				const model = bestModelPerClass[c];
				const p = preds.get(model)!;
				out[y][x][c] = p[y][x][c];
				sum += out[y][x][c];
			}
			// Renormalize
			if (sum > 0) {
				for (let c = 0; c < NUM_CLASSES; c++) out[y][x][c] /= sum;
			}
		}
	}
	return out;
}

// --- Main ---

const args = Deno.args;
const round = parseInt(args[0]);
if (isNaN(round)) {
	console.error("Usage: deno run -A select-best-model.ts <round> [--data-dir path] [--per-class]");
	Deno.exit(1);
}

let dataDir = "../simulation/data";
let perClass = false;
for (let i = 1; i < args.length; i++) {
	if (args[i] === "--data-dir" && args[i + 1]) { dataDir = args[++i]; }
	if (args[i] === "--per-class") perClass = true;
}

const CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"];
console.log(`\n=== Oracle Model Selection — Round ${round} ===\n`);

let totalScores: Record<string, number> = {};
let seedWinners: string[] = [];

for (let s = 0; s < 5; s++) {
	const queries = loadQueries(round, s);
	if (queries.length === 0) {
		console.log(`  S${s}: no queries, skipping`);
		seedWinners.push("sim");
		continue;
	}

	// Load all available model predictions
	const preds = new Map<string, { W: number; H: number; prediction: number[][][] }>();
	for (const model of MODELS) {
		const path = `${dataDir}/pred_${model}_r${round}_s${s}.bin`;
		try {
			preds.set(model, readPrediction(path));
		} catch {
			// Model not available for this round/seed
		}
	}

	if (preds.size === 0) {
		console.log(`  S${s}: no predictions found`);
		seedWinners.push("sim");
		continue;
	}

	const first = preds.values().next().value!;
	const { W, H } = first;
	const { counts, samples } = buildObserved(queries, W, H);

	// Score each model
	const scores: [string, number][] = [];
	for (const [model, { prediction }] of preds) {
		const score = scorePrediction(prediction, counts, samples, W, H);
		scores.push([model, score]);
		totalScores[model] = (totalScores[model] || 0) + score;
	}
	scores.sort((a, b) => b[1] - a[1]);

	const winner = scores[0][0];
	seedWinners.push(winner);

	// Display
	const scoreStr = scores
		.map(([m, sc]) => `${m}=${sc.toFixed(1)}${m === winner ? "*" : ""}`)
		.join("  ");
	console.log(`  S${s}: ${scoreStr}`);

	// Per-class analysis if requested
	if (perClass) {
		const classScores = new Map<string, number[]>();
		for (const [model, { prediction }] of preds) {
			classScores.set(model, scorePerClass(prediction, counts, samples, W, H));
		}
		const bestPerClass: string[] = [];
		for (let c = 0; c < NUM_CLASSES; c++) {
			let bestModel = "sim";
			let bestKL = Infinity;
			for (const [model, kls] of classScores) {
				if (kls[c] < bestKL) { bestKL = kls[c]; bestModel = model; }
			}
			bestPerClass.push(bestModel);
		}
		console.log(`    Per-class best: ${bestPerClass.map((m, i) => `${CLASS_NAMES[i]}=${m}`).join(", ")}`);

		// Score the per-class composite
		const composite = buildPerClassPred(
			new Map([...preds].map(([k, v]) => [k, v.prediction])),
			bestPerClass, W, H
		);
		const compositeScore = scorePrediction(composite, counts, samples, W, H);
		console.log(`    Composite score: ${compositeScore.toFixed(1)} (vs winner ${scores[0][1].toFixed(1)})`);
	}

	// Copy winner to submission path
	const srcPath = `${dataDir}/pred_${winner}_r${round}_s${s}.bin`;
	const dstPath = `${dataDir}/pred_r${round}_s${s}.bin`;
	Deno.copyFileSync(srcPath, dstPath);
}

// Summary
console.log(`\n  Winners: ${seedWinners.map((w, i) => `S${i}=${w}`).join("  ")}`);
const avgScores = Object.entries(totalScores)
	.map(([m, t]) => [m, t / 5] as [string, number])
	.sort((a, b) => b[1] - a[1]);
console.log(`  Avg scores: ${avgScores.map(([m, s]) => `${m}=${s.toFixed(1)}`).join("  ")}`);
console.log();
