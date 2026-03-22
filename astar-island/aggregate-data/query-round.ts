import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";
import { NUM_CLASSES, TERRAIN_MOUNTAIN, TERRAIN_OCEAN, terrainToClass, entropy, readPrediction, writePrediction, readGridBin } from "./bin-io.ts";

/**
 * Smart query + blend pipeline for a round.
 *
 * Strategy:
 *   - 10 queries per seed (50 total / 5 seeds)
 *   - First 3-4 queries: tile viewports across high-entropy areas for coverage
 *   - Remaining queries: allocated proportional to viewport entropy score
 *   - Blend: for cells with N observations, mix empirical distribution with model prior
 *     using per-class alphas (rare classes trusted faster)
 *   - Store all raw query results to data/queries/r{N}/ for future reuse
 *
 * Usage:
 *   deno run -A query-round.ts <round_number> [--queries-per-seed 10] [--viewport 10] [--dry-run] [--model bucket|mlp|sim|blend|ensemble] [--smart-alloc]
 */

const api = new NMAIAstarIsland();
const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";

// ── Model disagreement ──

function modelDisagreement(models: number[][][][], x: number, y: number): number {
	// Average pairwise symmetric KL divergence between all model pairs
	let totalKL = 0;
	let pairs = 0;
	for (let i = 0; i < models.length; i++) {
		for (let j = i + 1; j < models.length; j++) {
			const p = models[i][y][x];
			const q = models[j][y][x];
			let kl = 0;
			for (let c = 0; c < 6; c++) {
				if (p[c] > 0.001) kl += p[c] * Math.log(p[c] / Math.max(q[c], 0.001));
				if (q[c] > 0.001) kl += q[c] * Math.log(q[c] / Math.max(p[c], 0.001));
			}
			totalKL += kl / 2; // symmetric KL = average of both directions
			pairs++;
		}
	}
	return pairs > 0 ? totalKL / pairs : 0;
}

// ── Viewport placement ──

function findBestViewports(
	prediction: number[][][],
	W: number, H: number,
	vpSize: number,
	count: number,
	otherModels?: number[][][][], // additional model predictions for disagreement scoring
): { x: number; y: number; entropyScore: number }[] {
	// Collect all models (primary + others) for disagreement computation
	const allModels: number[][][][] = [prediction];
	if (otherModels) {
		for (const m of otherModels) allModels.push(m);
	}
	const useDisagreement = allModels.length >= 2;

	// Build entropy map
	const entropyMap: number[][] = [];
	for (let y = 0; y < H; y++) {
		entropyMap.push([]);
		for (let x = 0; x < W; x++) {
			entropyMap[y].push(entropy(prediction[y][x]));
		}
	}

	// Build disagreement map (if multiple models available)
	const disagreementMap: number[][] = [];
	let maxDisagreement = 0;
	if (useDisagreement) {
		for (let y = 0; y < H; y++) {
			disagreementMap.push([]);
			for (let x = 0; x < W; x++) {
				const d = modelDisagreement(allModels, x, y);
				disagreementMap[y].push(d);
				if (d > maxDisagreement) maxDisagreement = d;
			}
		}
	}

	// Compute max entropy for normalization when combining with disagreement
	let maxEntropy = 0;
	for (let y = 0; y < H; y++)
		for (let x = 0; x < W; x++)
			if (entropyMap[y][x] > maxEntropy) maxEntropy = entropyMap[y][x];

	// Score every possible viewport position
	const candidates: { x: number; y: number; score: number }[] = [];
	for (let vy = 0; vy <= H - vpSize; vy++) {
		for (let vx = 0; vx <= W - vpSize; vx++) {
			let score = 0;
			for (let dy = 0; dy < vpSize; dy++) {
				for (let dx = 0; dx < vpSize; dx++) {
					if (useDisagreement && maxEntropy > 0 && maxDisagreement > 0) {
						// Combine normalized entropy and disagreement (0.5 / 0.5 weighting)
						const normEntropy = entropyMap[vy + dy][vx + dx] / maxEntropy;
						const normDisagreement = disagreementMap[vy + dy][vx + dx] / maxDisagreement;
						score += 0.5 * normEntropy + 0.5 * normDisagreement;
					} else {
						score += entropyMap[vy + dy][vx + dx];
					}
				}
			}
			candidates.push({ x: vx, y: vy, score });
		}
	}
	candidates.sort((a, b) => b.score - a.score);

	// Greedy selection with minimum separation.
	// Viewports must not overlap more than 25% with any already-selected viewport.
	// This forces coverage of different parts of the map.
	const selected: { x: number; y: number; entropyScore: number }[] = [];

	for (const vp of candidates) {
		if (selected.length >= count) break;

		// Check overlap with all already-selected viewports
		let tooClose = false;
		for (const existing of selected) {
			const overlapX = Math.max(0, Math.min(vp.x + vpSize, existing.x + vpSize) - Math.max(vp.x, existing.x));
			const overlapY = Math.max(0, Math.min(vp.y + vpSize, existing.y + vpSize) - Math.max(vp.y, existing.y));
			const overlapArea = overlapX * overlapY;
			const vpArea = vpSize * vpSize;
			if (overlapArea > vpArea * 0.25) {
				tooClose = true;
				break;
			}
		}
		if (tooClose) continue;

		selected.push({ x: vp.x, y: vp.y, entropyScore: vp.score });
	}

	// If we couldn't find enough non-overlapping viewports (small dynamic area),
	// relax to 50% overlap
	if (selected.length < count) {
		for (const vp of candidates) {
			if (selected.length >= count) break;
			let tooClose = false;
			for (const existing of selected) {
				const overlapX = Math.max(0, Math.min(vp.x + vpSize, existing.x + vpSize) - Math.max(vp.x, existing.x));
				const overlapY = Math.max(0, Math.min(vp.y + vpSize, existing.y + vpSize) - Math.max(vp.y, existing.y));
				if (overlapX * overlapY > vpSize * vpSize * 0.5) {
					tooClose = true;
					break;
				}
			}
			if (tooClose) continue;
			if (selected.find(s => s.x === vp.x && s.y === vp.y)) continue;
			selected.push({ x: vp.x, y: vp.y, entropyScore: vp.score });
		}
	}

	return selected;
}

// ── Blending ──

function blendPredictions(
	modelPred: number[][][],
	sampleCounts: number[][],
	classCounts: number[][][],
	W: number, H: number,
	initialGrid: number[][] | null,
): { blended: number[][][]; refinedCells: number; avgSamples: number } {
	const blended = modelPred.map(row => row.map(cell => [...cell]));
	let refinedCells = 0;
	let totalSamples = 0;

	// Fixed Bayesian blending parameters.
	// Tested old(3/5,0.85), current(8/12,0.70), moderate(5/8,0.80) on R7-R10:
	// Current gives best average (+5.6 on good rounds like R9, -2.6 on bad rounds).
	// Adaptive (entropy/disagreement) was tested but models share systematic biases,
	// so disagreement doesn't differentiate good vs bad rounds.
	const pseudoRare = 8;    // for settlement/port/ruin (classes 1-3)
	const pseudoCommon = 12;  // for empty/forest/mountain
	const alphaCap = 0.70;

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const n = sampleCounts[y][x];
			if (n === 0) continue;

			const empirical: number[] = classCounts[y][x].map(c => c / n);

			for (let c = 0; c < NUM_CLASSES; c++) {
				const pseudoCount = (c >= 1 && c <= 3) ? pseudoRare : pseudoCommon;
				const alpha = Math.min(n / (n + pseudoCount), alphaCap);
				blended[y][x][c] = (1 - alpha) * modelPred[y][x][c] + alpha * empirical[c];
			}

			// Smart floor: only floor reachable classes (matches simulate.cpp logic)
			const terrain = initialGrid ? initialGrid[y][x] : -1;
			let total = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				const reachable = terrain === TERRAIN_MOUNTAIN ? (c === 5)
					: terrain === TERRAIN_OCEAN ? (c === 0)
					: terrain >= 0 ? (c !== 5) // land: all except mountain
					: true; // no grid available: floor everything (old behavior)
				if (reachable) {
					if (blended[y][x][c] < 0.003) blended[y][x][c] = 0.003;
				} else {
					blended[y][x][c] = 0.0;
				}
				total += blended[y][x][c];
			}
			for (let c = 0; c < NUM_CLASSES; c++) blended[y][x][c] /= total;

			refinedCells++;
			totalSamples += n;
		}
	}

	return { blended, refinedCells, avgSamples: refinedCells > 0 ? totalSamples / refinedCells : 0 };
}

function ensemblePredictions(bucket: number[][][], mlp: number[][][], W: number, H: number): number[][][] {
	// 70/30 bucket-heavy blend — best average across all historical rounds.
	// Bucket is more stable; MLP adds value but has high variance.
	const BUCKET_WEIGHT = 0.7;
	const MLP_WEIGHT = 0.3;

	const result: number[][][] = [];
	for (let y = 0; y < H; y++) {
		const row: number[][] = [];
		for (let x = 0; x < W; x++) {
			const cell: number[] = [];
			let total = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				const v = BUCKET_WEIGHT * bucket[y][x][c] + MLP_WEIGHT * mlp[y][x][c];
				cell.push(v);
				total += v;
			}
			for (let c = 0; c < NUM_CLASSES; c++) cell[c] /= total;
			row.push(cell);
		}
		result.push(row);
	}
	return result;
}

// ── Storage ──

interface StoredQuery {
	query_index: number;
	seed_index: number;
	viewport: { x: number; y: number; w: number; h: number };
	grid: number[][];
	settlements: unknown[];
	timestamp: string;
}

async function loadStoredQueries(roundNum: number, seedIdx: number): Promise<StoredQuery[]> {
	const path = `${DATA_DIR}/queries/r${roundNum}/s${seedIdx}_queries.json`;
	try {
		const text = await Deno.readTextFile(path);
		return JSON.parse(text);
	} catch {
		return [];
	}
}

async function saveQueries(roundNum: number, seedIdx: number, queries: StoredQuery[]) {
	const dir = `${DATA_DIR}/queries/r${roundNum}`;
	try { await Deno.mkdir(dir, { recursive: true }); } catch { /* */ }
	const path = `${dir}/s${seedIdx}_queries.json`;
	await Deno.writeTextFile(path, JSON.stringify(queries, null, 2));
}

function accumulateQueries(
	queries: StoredQuery[],
	W: number, H: number,
): { sampleCounts: number[][]; classCounts: number[][][] } {
	const sampleCounts: number[][] = Array.from({ length: H }, () => new Array(W).fill(0));
	const classCounts: number[][][] = Array.from({ length: H }, () =>
		Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0))
	);

	for (const q of queries) {
		const vp = q.viewport;
		for (let gy = 0; gy < q.grid.length; gy++) {
			for (let gx = 0; gx < q.grid[gy].length; gx++) {
				const mapX = vp.x + gx;
				const mapY = vp.y + gy;
				if (mapX >= W || mapY >= H) continue;
				const cls = terrainToClass(q.grid[gy][gx]);
				classCounts[mapY][mapX][cls]++;
				sampleCounts[mapY][mapX]++;
			}
		}
	}

	return { sampleCounts, classCounts };
}

// ── Main ──

async function main() {
	const args = Deno.args;
	const roundNum = parseInt(args[0]);
	if (isNaN(roundNum)) {
		console.log("Usage: query-round.ts <round_number> [options]");
		console.log("  --queries-per-seed N   queries per seed (default: 10)");
		console.log("  --viewport N           viewport size (default: 10)");
		console.log("  --model TYPE           bucket|mlp|sim|blend|ensemble (default: auto-detect)");
		console.log("  --dry-run              skip API calls, use stored queries only");
		console.log("  --blend-only           no queries, just blend stored queries with models");
		console.log("  --seeds 0,1,2,3,4      which seeds to process (default: all)");
		console.log("  --smart-alloc          allocate queries proportional to per-seed entropy");
		Deno.exit(1);
	}

	const queriesPerSeed = parseInt(args.find((_, i) => args[i - 1] === "--queries-per-seed") || "10");
	const vpSize = parseInt(args.find((_, i) => args[i - 1] === "--viewport") || "10");
	const explicitModel = args.find((_, i) => args[i - 1] === "--model");
	const dryRun = args.includes("--dry-run");
	const blendOnly = args.includes("--blend-only");
	const smartAlloc = args.includes("--smart-alloc");
	const seedArg = args.find((_, i) => args[i - 1] === "--seeds");
	const seeds = seedArg ? seedArg.split(",").map(Number) : [0, 1, 2, 3, 4];

	// Auto-detect model type: try blend first, fall back to sim
	let modelType = explicitModel || "sim";
	if (!explicitModel) {
		// Check if blend files exist for seed 0
		try {
			Deno.statSync(`${BIN_DIR}/pred_blend_r${roundNum}_s0.bin`);
			modelType = "blend";
			console.log(`  Auto-detected blend predictions, using --model blend`);
		} catch {
			// Also check pred_r (submission file) as a blend proxy
			// Fall back to sim
			modelType = "sim";
		}
	}

	// Load round info
	const roundsData = JSON.parse(await Deno.readTextFile(`${DATA_DIR}/my-rounds.json`));
	const roundInfo = roundsData.find((r: { round_number: number }) => r.round_number === roundNum);
	if (!roundInfo) { console.error(`Round ${roundNum} not found. Run fetch-analysis.ts first.`); Deno.exit(1); }

	console.log(`\n  Round ${roundNum} — ${roundInfo.status}`);
	console.log(`  Queries: ${roundInfo.queries_used}/${roundInfo.queries_max}`);
	console.log(`  Model: ${modelType}, Viewport: ${vpSize}x${vpSize}`);
	console.log(`  Seeds: ${seeds.join(", ")}`);
	if (dryRun) console.log(`  *** DRY RUN — no API calls ***`);
	if (blendOnly) console.log(`  *** BLEND ONLY — using stored queries ***`);
	if (smartAlloc) console.log(`  *** SMART ALLOC — proportional query allocation ***`);
	console.log();

	// Smart allocation: compute per-seed query budgets proportional to total entropy
	const perSeedQueries: Map<number, number> = new Map();
	const totalBudget = queriesPerSeed * seeds.length;
	if (smartAlloc && seeds.length > 1) {
		const seedEntropies: Map<number, number> = new Map();
		for (const seedIdx of seeds) {
			let pred: number[][][] | null = null;
			// Try to load model prediction for entropy computation
			for (const prefix of ["pred_blend_", "pred_sim_", "pred_r"]) {
				try {
					const p = readPrediction(`${BIN_DIR}/${prefix}r${roundNum}_s${seedIdx}.bin`);
					pred = p.prediction;
					break;
				} catch { /* try next */ }
			}
			if (!pred) {
				seedEntropies.set(seedIdx, 1.0); // fallback
				continue;
			}
			let totalEntropy = 0;
			for (let y = 0; y < pred.length; y++)
				for (let x = 0; x < pred[y].length; x++)
					totalEntropy += entropy(pred[y][x]);
			seedEntropies.set(seedIdx, totalEntropy);
		}
		const totalEntropy = Array.from(seedEntropies.values()).reduce((a, b) => a + b, 0);
		let allocated = 0;
		const minPerSeed = 5;
		const maxPerSeed = 15;
		// First pass: allocate proportionally
		const rawAlloc: Map<number, number> = new Map();
		for (const seedIdx of seeds) {
			const share = totalEntropy > 0
				? Math.round(totalBudget * (seedEntropies.get(seedIdx)! / totalEntropy))
				: queriesPerSeed;
			rawAlloc.set(seedIdx, Math.max(minPerSeed, Math.min(maxPerSeed, share)));
		}
		// Second pass: adjust to match total budget
		allocated = Array.from(rawAlloc.values()).reduce((a, b) => a + b, 0);
		// Distribute remainder/deficit to highest/lowest entropy seeds
		const sortedSeeds = [...seeds].sort((a, b) =>
			(seedEntropies.get(b) || 0) - (seedEntropies.get(a) || 0)
		);
		let idx = 0;
		while (allocated < totalBudget) {
			const s = sortedSeeds[idx % sortedSeeds.length];
			if (rawAlloc.get(s)! < maxPerSeed) {
				rawAlloc.set(s, rawAlloc.get(s)! + 1);
				allocated++;
			}
			idx++;
			if (idx > totalBudget) break; // safety
		}
		idx = sortedSeeds.length - 1;
		while (allocated > totalBudget) {
			const s = sortedSeeds[idx >= 0 ? idx : 0];
			if (rawAlloc.get(s)! > minPerSeed) {
				rawAlloc.set(s, rawAlloc.get(s)! - 1);
				allocated--;
			}
			idx--;
			if (idx < -sortedSeeds.length) break; // safety
		}
		for (const seedIdx of seeds) {
			perSeedQueries.set(seedIdx, rawAlloc.get(seedIdx)!);
		}
		console.log(`  Smart alloc: ${seeds.map(s => `S${s}=${perSeedQueries.get(s)}`).join(", ")} (total=${totalBudget})`);
	} else {
		for (const seedIdx of seeds) {
			perSeedQueries.set(seedIdx, queriesPerSeed);
		}
	}

	const summaryRows: string[] = [];

	for (const seedIdx of seeds) {
		console.log(`━━━ Seed ${seedIdx} ━━━`);

		// Load model prediction(s)
		let modelPred: number[][][] | null = null;
		// Try pred_bucket_ first (safe name from iterate.sh), fall back to pred_r (legacy)
		let bucketPath = `${BIN_DIR}/pred_bucket_r${roundNum}_s${seedIdx}.bin`;
		try { Deno.statSync(bucketPath); } catch { bucketPath = `${BIN_DIR}/pred_r${roundNum}_s${seedIdx}.bin`; }
		const mlpPath = `${BIN_DIR}/pred_mlp_r${roundNum}_s${seedIdx}.bin`;

		let bucketPred: ReturnType<typeof readPrediction> | null = null;
		let mlpPred: ReturnType<typeof readPrediction> | null = null;
		let simPred: ReturnType<typeof readPrediction> | null = null;
		let cnnPred: ReturnType<typeof readPrediction> | null = null;
		let blendPredFile: ReturnType<typeof readPrediction> | null = null;
		let W = 40, H = 40;

		// Load initial grid for smart flooring
		const gridData = readGridBin(`${BIN_DIR}/grids.bin`, roundNum, seedIdx);
		const initialGrid = gridData ? gridData.grid : null;

		try { bucketPred = readPrediction(bucketPath); W = bucketPred.W; H = bucketPred.H; console.log(`  Loaded bucket: ${bucketPath}`); } catch { console.log(`  No bucket prediction at ${bucketPath}`); }
		try { mlpPred = readPrediction(mlpPath); W = mlpPred.W; H = mlpPred.H; console.log(`  Loaded MLP: ${mlpPath}`); } catch { console.log(`  No MLP prediction at ${mlpPath}`); }
		const simPath = `${BIN_DIR}/pred_sim_r${roundNum}_s${seedIdx}.bin`;
		try { simPred = readPrediction(simPath); W = simPred.W; H = simPred.H; console.log(`  Loaded sim: ${simPath}`); } catch { console.log(`  No sim prediction at ${simPath}`); }
		const cnnPath = `${BIN_DIR}/pred_cnn_r${roundNum}_s${seedIdx}.bin`;
		try { cnnPred = readPrediction(cnnPath); W = cnnPred.W; H = cnnPred.H; console.log(`  Loaded CNN: ${cnnPath}`); } catch { /* no CNN prediction */ }
		const blendFilePath = `${BIN_DIR}/pred_blend_r${roundNum}_s${seedIdx}.bin`;
		try { blendPredFile = readPrediction(blendFilePath); W = blendPredFile.W; H = blendPredFile.H; console.log(`  Loaded blend: ${blendFilePath}`); } catch { /* no blend file yet */ }

		if (modelType === "blend" && blendPredFile) {
			console.log(`  Using blend (per-class optimized ensemble)`);
			modelPred = blendPredFile.prediction;
		} else if (modelType === "sim" && simPred) {
			console.log(`  Using sim (calibrated simulator)`);
			modelPred = simPred.prediction;
		} else if (modelType === "ensemble" && bucketPred && mlpPred) {
			console.log(`  Using ensemble (70/30 bucket+mlp)`);
			modelPred = ensemblePredictions(bucketPred.prediction, mlpPred.prediction, W, H);
		} else if (modelType === "mlp" && mlpPred) {
			modelPred = mlpPred.prediction;
		} else if (modelType === "blend" && simPred) {
			// blend requested but no blend file — fall back to sim
			console.log(`  No blend file, falling back to sim model`);
			modelPred = simPred.prediction;
		} else if (simPred) {
			modelPred = simPred.prediction;
			if (modelType !== "sim") console.log(`  Falling back to sim model`);
		} else if (bucketPred) {
			modelPred = bucketPred.prediction;
			console.log(`  Falling back to bucket model`);
		} else if (mlpPred) {
			modelPred = mlpPred.prediction;
		} else {
			console.error(`  No model predictions found for seed ${seedIdx}, skipping`);
			continue;
		}

		// Load stored queries
		const stored = await loadStoredQueries(roundNum, seedIdx);
		console.log(`  Stored queries: ${stored.length}`);

		// Collect all available model predictions (excluding the primary) for disagreement scoring
		const otherModels: number[][][][] = [];
		const allPredSources = [
			{ pred: bucketPred, name: "bucket" },
			{ pred: mlpPred, name: "mlp" },
			{ pred: simPred, name: "sim" },
			{ pred: cnnPred, name: "cnn" },
		];
		for (const { pred } of allPredSources) {
			if (pred && pred.prediction !== modelPred) {
				otherModels.push(pred.prediction);
			}
		}
		if (otherModels.length > 0) {
			console.log(`  Viewport targeting: entropy + disagreement (${otherModels.length + 1} models)`);
		}

		// Plan viewports: request enough for full budget so extras use distinct locations
		const seedBudgetEst = perSeedQueries.get(seedIdx) || queriesPerSeed;
		const coverageVps = findBestViewports(modelPred, W, H, vpSize, Math.max(seedBudgetEst, Math.ceil(queriesPerSeed / 3)), otherModels.length > 0 ? otherModels : undefined);

		let newQueries: StoredQuery[] = [];

		if (!blendOnly && !dryRun && roundInfo.status === "active") {
			const budget = await api.getBudget();
			const remaining = budget.queries_max - budget.queries_used;
			const seedBudget = perSeedQueries.get(seedIdx) || queriesPerSeed;
			const toUse = Math.min(seedBudget, remaining);

			if (toUse <= 0) {
				console.log(`  No queries remaining!`);
			} else {
				console.log(`  Budget: ${budget.queries_used}/${budget.queries_max} used, will use ${toUse}`);

				// Build query plan: use distinct viewports first (each covers different area),
				// only repeat high-entropy viewports after exhausting distinct locations
				const queryPlan: { x: number; y: number }[] = [];
				// Phase 1: one query per distinct viewport (best coverage)
				for (let q = 0; q < Math.min(toUse, coverageVps.length); q++) {
					queryPlan.push(coverageVps[q]);
				}
				// Phase 2: if we still need more, repeat highest-entropy viewports
				if (toUse > coverageVps.length && coverageVps.length > 0) {
					let remaining = toUse - coverageVps.length;
					let idx = 0; // coverageVps is sorted by entropy, so idx=0 is highest
					while (remaining > 0) {
						queryPlan.push(coverageVps[idx % coverageVps.length]);
						idx++;
						remaining--;
					}
				}

				for (let q = 0; q < queryPlan.length; q++) {
					const vp = queryPlan[q];
					console.log(`  Query ${q + 1}/${queryPlan.length}: viewport (${vp.x}, ${vp.y}) ${vpSize}x${vpSize}`);

					try {
						const result = await api.simulate({
							round_id: roundInfo.id,
							seed_index: seedIdx,
							viewport_x: vp.x,
							viewport_y: vp.y,
							viewport_w: vpSize,
							viewport_h: vpSize,
						});

						const sq: StoredQuery = {
							query_index: stored.length + newQueries.length,
							seed_index: seedIdx,
							viewport: result.viewport,
							grid: result.grid,
							settlements: result.settlements,
							timestamp: new Date().toISOString(),
						};
						newQueries.push(sq);

						console.log(`    ✓ queries: ${result.queries_used}/${result.queries_max}`);
					} catch (e) {
						console.error(`    ✗ failed: ${e}`);
						break;
					}
				}
			}
		}

		// Combine stored + new queries
		const allQueries = [...stored, ...newQueries];
		if (newQueries.length > 0) {
			await saveQueries(roundNum, seedIdx, allQueries);
			console.log(`  Saved ${allQueries.length} total queries`);
		}

		// Accumulate observations and blend
		const { sampleCounts, classCounts } = accumulateQueries(allQueries, W, H);
		const { blended, refinedCells, avgSamples } = blendPredictions(modelPred, sampleCounts, classCounts, W, H, initialGrid);

		console.log(`  Blended: ${refinedCells} cells refined, avg ${avgSamples.toFixed(1)} samples/cell`);

		// Write blended output (separate file — never overwrite raw model predictions)
		const blendPath = `${BIN_DIR}/pred_blend_r${roundNum}_s${seedIdx}.bin`;
		writePrediction(blendPath, roundNum, seedIdx, W, H, blended);
		console.log(`  Written: ${blendPath}`);

		// Also write to pred_r{N}_s{M}.bin for submission (submit-predictions.ts reads this)
		const submitPath = `${BIN_DIR}/pred_r${roundNum}_s${seedIdx}.bin`;
		writePrediction(submitPath, roundNum, seedIdx, W, H, blended);
		console.log(`  Submit-ready: ${submitPath}`);

		summaryRows.push(`  S${seedIdx}: ${allQueries.length} queries, ${refinedCells} cells refined`);
		console.log();
	}

	console.log(`\n━━━ Summary ━━━`);
	for (const row of summaryRows) console.log(row);
	console.log(`\nPrediction files ready at ${BIN_DIR}/pred_r${roundNum}_s*.bin`);
	console.log(`Submit with: deno run -A submit-predictions.ts --round ${roundNum}`);
}

await main();
