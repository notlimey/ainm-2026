/**
 * Per-round parameter tuning using query observations.
 *
 * Strategy: Use query observations as partial ground truth to find
 * round-specific sim parameter adjustments. Optimizes a few key params
 * (expansion_prob, collapse_pop, winter_base_loss, raid_damage, growth_rate)
 * using grid search over multiplicative factors.
 *
 * Usage: deno run -A tune-round.ts <round> [--rollouts 100] [--dry-run]
 */

import { NUM_CLASSES, terrainToClass, readPrediction, writePrediction, readGridBin } from "./bin-io.ts";

const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";
const SIM_BIN = "../simulation/simulate";

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
		return JSON.parse(await Deno.readTextFile(`${DATA_DIR}/queries/r${roundNum}/s${seedIdx}_queries.json`));
	} catch { return []; }
}

// Build observation map from queries: for each observed cell, compute empirical distribution
function buildObservationMap(queries: StoredQuery[], W: number, H: number) {
	const counts: number[][][] = Array.from({ length: H }, () =>
		Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0)));
	const nobs: number[][] = Array.from({ length: H }, () => new Array(W).fill(0));

	for (const q of queries) {
		for (let gy = 0; gy < q.grid.length; gy++) {
			for (let gx = 0; gx < q.grid[gy].length; gx++) {
				const mx = q.viewport.x + gx, my = q.viewport.y + gy;
				if (mx >= W || my >= H) continue;
				counts[my][mx][terrainToClass(q.grid[gy][gx])]++;
				nobs[my][mx]++;
			}
		}
	}
	return { counts, nobs };
}

// Score a prediction against query observations (entropy-weighted KL on observed cells)
function scoreAgainstObs(pred: number[][][], counts: number[][][], nobs: number[][], W: number, H: number): number {
	let totalEntropy = 0, totalWeightedKL = 0;
	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const n = nobs[y][x];
			if (n < 2) continue; // need at least 2 observations for meaningful empirical
			const emp = counts[y][x].map(c => c / n);

			let ent = 0;
			for (let c = 0; c < NUM_CLASSES; c++) if (emp[c] > 0) ent -= emp[c] * Math.log(emp[c]);
			if (ent < 1e-6) continue;

			let kl = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				if (emp[c] > 0) kl += emp[c] * Math.log(emp[c] / Math.max(pred[y][x][c], 1e-10));
			}
			totalEntropy += ent;
			totalWeightedKL += ent * kl;
		}
	}
	if (totalEntropy === 0) return 50; // no data
	const wkl = totalWeightedKL / totalEntropy;
	return Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));
}

// Run simulator with given params file and return prediction
async function runSim(round: number, seed: number, rollouts: number, paramsFile: string, outFile: string): Promise<number[][][] | null> {
	const cmd = new Deno.Command(SIM_BIN, {
		args: ["data/grids.bin", String(round), String(seed), outFile,
			"--rollouts", String(rollouts), "--params", paramsFile],
		cwd: "../simulation",
		stdout: "piped", stderr: "piped",
	});
	const result = await cmd.output();
	if (!result.success) return null;
	try {
		return readPrediction(outFile).prediction;
	} catch { return null; }
}

// Write a tweaked params file by modifying specific floats in the binary
async function writeTweakedParams(baseParamsPath: string, outPath: string, tweaks: Record<number, number>) {
	const data = await Deno.readFile(baseParamsPath);
	const buf = new Uint8Array(data);
	const view = new DataView(buf.buffer);
	// Skip 4-byte "SIMP" magic
	for (const [offset, value] of Object.entries(tweaks)) {
		view.setFloat32(4 + Number(offset) * 4, value, true);
	}
	await Deno.writeFile(outPath, buf);
}

// Key param indices in SimParams struct (float offsets after magic):
// 0: init_population, 1: init_food, 2: init_defense, 3: init_tech
// 4: food_per_forest, 5: food_per_plains, 6: food_per_coastal
// 7: growth_threshold, 8: growth_rate
// 9: expansion_pop, 10: expansion_range (int!), 11: expansion_prob
// 12: port_threshold, 13: port_prob
// 14: longship_threshold, 15: longship_prob
// 16: raid_range_land, 17: raid_range_sea
// 18: raid_prob_base, 19: raid_prob_desperate, 20: desperation_food
// 21: raid_damage, 22: raid_loot_frac, 23: conquest_prob
// 24: trade_range, 25: trade_food, 26: trade_wealth, 27: tech_diffusion
// 28: winter_base_loss, 29: winter_variance
// 30: winter_catastrophe_prob, 31: winter_catastrophe_mult
// 32: collapse_pop, 33: collapse_food, 34: collapse_defense
// 35: ruin_reclaim_range, 36: ruin_reclaim_prob
// 37: ruin_forest_prob, 38: ruin_plains_prob, 39: forest_adj_bonus

// Tune these key params (most impact on settlement dynamics):
const TUNE_PARAMS = [
	{ name: "expansion_prob", idx: 11, factors: [0.8, 0.9, 1.0, 1.1, 1.2] },
	{ name: "collapse_pop", idx: 32, factors: [0.8, 0.9, 1.0, 1.1, 1.2] },
	{ name: "winter_base_loss", idx: 28, factors: [0.85, 0.95, 1.0, 1.05, 1.15] },
	{ name: "raid_damage", idx: 21, factors: [0.85, 0.95, 1.0, 1.05, 1.15] },
];

async function main() {
	const roundNum = parseInt(Deno.args[0]);
	if (isNaN(roundNum)) { console.log("Usage: tune-round.ts <round> [--rollouts N]"); Deno.exit(1); }

	const rollouts = parseInt(Deno.args.find((_, i) => Deno.args[i - 1] === "--rollouts") || "100");
	const dryRun = Deno.args.includes("--dry-run");

	console.log(`Per-round tuning: R${roundNum}, ${rollouts} rollouts`);

	// Load base params
	const baseParams = `${BIN_DIR}/params.bin`;
	const baseData = await Deno.readFile(baseParams);
	const baseView = new DataView(new Uint8Array(baseData).buffer);

	for (let seedIdx = 0; seedIdx < 5; seedIdx++) {
		const queries = await loadStoredQueries(roundNum, seedIdx);
		if (queries.length === 0) { console.log(`  S${seedIdx}: no queries, skipping`); continue; }

		const W = 40, H = 40;
		const { counts, nobs } = buildObservationMap(queries, W, H);

		// Count observed dynamic cells
		let obsCells = 0;
		for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (nobs[y][x] >= 2) obsCells++;
		console.log(`  S${seedIdx}: ${queries.length} queries, ${obsCells} cells with 2+ observations`);

		// Score baseline
		const basePred = readPrediction(`${BIN_DIR}/pred_sim_r${roundNum}_s${seedIdx}.bin`).prediction;
		const baseScore = scoreAgainstObs(basePred, counts, nobs, W, H);
		console.log(`    Baseline sim score on obs: ${baseScore.toFixed(1)}`);

		if (dryRun) continue;

		// Grid search over param combinations
		// To keep it fast, do sequential 1D sweeps (coordinate descent)
		let bestFactors: number[] = TUNE_PARAMS.map(() => 1.0);
		let bestScore = baseScore;
		const tmpParams = `${BIN_DIR}/_tune_tmp_params.bin`;
		const tmpPred = `${BIN_DIR}/_tune_tmp_pred.bin`;

		for (let iter = 0; iter < 2; iter++) { // 2 passes of coordinate descent
			for (let pi = 0; pi < TUNE_PARAMS.length; pi++) {
				const param = TUNE_PARAMS[pi];
				const baseVal = baseView.getFloat32(4 + param.idx * 4, true);
				let bestFactor = bestFactors[pi];

				for (const factor of param.factors) {
					if (factor === bestFactors[pi]) continue; // already tested

					// Build tweaked params
					const tweaks: Record<number, number> = {};
					for (let j = 0; j < TUNE_PARAMS.length; j++) {
						const p = TUNE_PARAMS[j];
						const bv = baseView.getFloat32(4 + p.idx * 4, true);
						const f = j === pi ? factor : bestFactors[j];
						tweaks[p.idx] = bv * f;
					}
					await writeTweakedParams(baseParams, tmpParams, tweaks);

					const pred = await runSim(roundNum, seedIdx, rollouts, tmpParams, tmpPred);
					if (!pred) continue;

					const score = scoreAgainstObs(pred, counts, nobs, W, H);
					if (score > bestScore) {
						bestScore = score;
						bestFactor = factor;
						bestFactors[pi] = factor;
					}
				}
				bestFactors[pi] = bestFactor;
			}
		}

		// Report and save best
		const improved = bestScore > baseScore;
		const changes = TUNE_PARAMS.map((p, i) => bestFactors[i] !== 1.0 ? `${p.name}×${bestFactors[i]}` : null).filter(Boolean);
		console.log(`    Best: ${bestScore.toFixed(1)} (${improved ? "+" : ""}${(bestScore - baseScore).toFixed(1)}) ${changes.length > 0 ? changes.join(", ") : "(no change)"}`);

		if (improved) {
			// Generate final prediction with best params at higher rollouts
			const tweaks: Record<number, number> = {};
			for (let j = 0; j < TUNE_PARAMS.length; j++) {
				const p = TUNE_PARAMS[j];
				const bv = baseView.getFloat32(4 + p.idx * 4, true);
				tweaks[p.idx] = bv * bestFactors[j];
			}
			await writeTweakedParams(baseParams, tmpParams, tweaks);
			const pred = await runSim(roundNum, seedIdx, rollouts * 2, tmpParams, `${BIN_DIR}/pred_tuned_r${roundNum}_s${seedIdx}.bin`);
			if (pred) {
				console.log(`    Written tuned prediction: pred_tuned_r${roundNum}_s${seedIdx}.bin`);
			}
		}

		// Cleanup
		try { await Deno.remove(tmpParams); } catch { /**/ }
		try { await Deno.remove(tmpPred); } catch { /**/ }
	}
}

await main();
