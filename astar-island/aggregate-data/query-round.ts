import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";

/**
 * Smart query + blend pipeline for a round.
 *
 * Strategy:
 *   - 10 queries per seed (50 total / 5 seeds)
 *   - First 3-4 queries: tile viewports across high-entropy areas for coverage
 *   - Remaining 6-7 queries: re-query the highest-entropy viewports for better empirical estimates
 *   - Blend: for cells with N observations, mix empirical distribution with model prior
 *   - Store all raw query results to data/queries/r{N}/ for future reuse
 *
 * Usage:
 *   deno run -A query-round.ts <round_number> [--queries-per-seed 10] [--viewport 15] [--dry-run] [--model bucket|mlp|ensemble]
 */

const api = new NMAIAstarIsland();
const BIN_DIR = "../simulation/data";
const DATA_DIR = "data";
const NUM_CLASSES = 6;

// ── Helpers ──

function terrainToClass(code: number): number {
	switch (code) {
		case 1: return 1;
		case 2: return 2;
		case 3: return 3;
		case 4: return 4;
		case 5: return 5;
		default: return 0;
	}
}

function entropy(probs: number[]): number {
	let h = 0;
	for (const p of probs) if (p > 0) h -= p * Math.log2(p);
	return h;
}

function readPrediction(path: string) {
	const data = Deno.readFileSync(path);
	const view = new DataView(data.buffer);
	const magic = String.fromCharCode(data[0], data[1], data[2], data[3]);
	if (magic !== "ASTP") throw new Error(`Bad magic: ${magic}`);
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

function writePrediction(path: string, round: number, seed: number, W: number, H: number, prediction: number[][][]) {
	const buf = new ArrayBuffer(22 + H * W * NUM_CLASSES * 4);
	const view = new DataView(buf);
	const u8 = new Uint8Array(buf);
	u8.set([0x41, 0x53, 0x54, 0x50]); // "ASTP"
	view.setUint16(4, 1, true);
	view.setInt32(6, round, true);
	view.setInt32(10, seed, true);
	view.setInt32(14, W, true);
	view.setInt32(18, H, true);
	let offset = 22;
	for (let y = 0; y < H; y++)
		for (let x = 0; x < W; x++)
			for (let c = 0; c < NUM_CLASSES; c++) {
				view.setFloat32(offset, prediction[y][x][c], true);
				offset += 4;
			}
	Deno.writeFileSync(path, new Uint8Array(buf));
}

// ── Viewport placement ──

function findBestViewports(
	prediction: number[][][],
	W: number, H: number,
	vpSize: number,
	count: number,
): { x: number; y: number; entropyScore: number }[] {
	// Build entropy map
	const entropyMap: number[][] = [];
	for (let y = 0; y < H; y++) {
		entropyMap.push([]);
		for (let x = 0; x < W; x++) {
			entropyMap[y].push(entropy(prediction[y][x]));
		}
	}

	// Score every possible viewport position
	const candidates: { x: number; y: number; score: number }[] = [];
	for (let vy = 0; vy <= H - vpSize; vy++) {
		for (let vx = 0; vx <= W - vpSize; vx++) {
			let score = 0;
			for (let dy = 0; dy < vpSize; dy++)
				for (let dx = 0; dx < vpSize; dx++)
					score += entropyMap[vy + dy][vx + dx];
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
): { blended: number[][][]; refinedCells: number; avgSamples: number } {
	const blended = modelPred.map(row => row.map(cell => [...cell]));
	let refinedCells = 0;
	let totalSamples = 0;

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const n = sampleCounts[y][x];
			if (n === 0) continue;

			const empirical: number[] = classCounts[y][x].map(c => c / n);

			// Bayesian-style blending: more samples → trust empirical more
			// With 1 sample: alpha=0.17, 5: 0.50, 10: 0.67, 20: 0.80
			const alpha = Math.min(n / (n + 5), 0.85);

			for (let c = 0; c < NUM_CLASSES; c++)
				blended[y][x][c] = (1 - alpha) * modelPred[y][x][c] + alpha * empirical[c];

			// Floor + renormalize
			let total = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				if (blended[y][x][c] < 0.005) blended[y][x][c] = 0.005;
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
		console.log("  --viewport N           viewport size (default: 15)");
		console.log("  --model TYPE           bucket|mlp|sim|ensemble (default: sim)");
		console.log("  --dry-run              skip API calls, use stored queries only");
		console.log("  --blend-only           no queries, just blend stored queries with models");
		console.log("  --seeds 0,1,2,3,4      which seeds to process (default: all)");
		Deno.exit(1);
	}

	const queriesPerSeed = parseInt(args.find((_, i) => args[i - 1] === "--queries-per-seed") || "10");
	const vpSize = parseInt(args.find((_, i) => args[i - 1] === "--viewport") || "15");
	const modelType = args.find((_, i) => args[i - 1] === "--model") || "sim";
	const dryRun = args.includes("--dry-run");
	const blendOnly = args.includes("--blend-only");
	const seedArg = args.find((_, i) => args[i - 1] === "--seeds");
	const seeds = seedArg ? seedArg.split(",").map(Number) : [0, 1, 2, 3, 4];

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
	console.log();

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
		let W = 40, H = 40;

		try { bucketPred = readPrediction(bucketPath); W = bucketPred.W; H = bucketPred.H; console.log(`  Loaded bucket: ${bucketPath}`); } catch { console.log(`  No bucket prediction at ${bucketPath}`); }
		try { mlpPred = readPrediction(mlpPath); W = mlpPred.W; H = mlpPred.H; console.log(`  Loaded MLP: ${mlpPath}`); } catch { console.log(`  No MLP prediction at ${mlpPath}`); }
		const simPath = `${BIN_DIR}/pred_sim_r${roundNum}_s${seedIdx}.bin`;
		try { simPred = readPrediction(simPath); W = simPred.W; H = simPred.H; console.log(`  Loaded sim: ${simPath}`); } catch { console.log(`  No sim prediction at ${simPath}`); }

		if (modelType === "sim" && simPred) {
			console.log(`  Using sim (calibrated simulator)`);
			modelPred = simPred.prediction;
		} else if (modelType === "ensemble" && bucketPred && mlpPred) {
			console.log(`  Using ensemble (70/30 bucket+mlp)`);
			modelPred = ensemblePredictions(bucketPred.prediction, mlpPred.prediction, W, H);
		} else if (modelType === "mlp" && mlpPred) {
			modelPred = mlpPred.prediction;
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

		// Plan viewports based on model prediction entropy
		const coverageVps = findBestViewports(modelPred, W, H, vpSize, Math.ceil(queriesPerSeed / 3));

		let newQueries: StoredQuery[] = [];

		if (!blendOnly && !dryRun && roundInfo.status === "active") {
			const budget = await api.getBudget();
			const remaining = budget.queries_max - budget.queries_used;
			const toUse = Math.min(queriesPerSeed, remaining);

			if (toUse <= 0) {
				console.log(`  No queries remaining!`);
			} else {
				console.log(`  Budget: ${budget.queries_used}/${budget.queries_max} used, will use ${toUse}`);

				// Build query plan:
				// First round through coverage viewports, then repeat for density
				const queryPlan: { x: number; y: number }[] = [];
				for (let q = 0; q < toUse; q++) {
					queryPlan.push(coverageVps[q % coverageVps.length]);
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
		const { blended, refinedCells, avgSamples } = blendPredictions(modelPred, sampleCounts, classCounts, W, H);

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
