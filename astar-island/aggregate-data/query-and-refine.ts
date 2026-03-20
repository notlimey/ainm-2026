import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";

const api = new NMAIAstarIsland();
const BIN_DIR = "../simulation/data";
const NUM_CLASSES = 6;

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

function readPrediction(path: string) {
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
			for (let c = 0; c < NUM_CLASSES; c++) {
				cell.push(view.getFloat32(offset, true));
				offset += 4;
			}
			row.push(cell);
		}
		prediction.push(row);
	}
	return { round, seed, W, H, prediction };
}

function entropy(probs: number[]): number {
	let h = 0;
	for (const p of probs) {
		if (p > 0) h -= p * Math.log2(p);
	}
	return h;
}

function findBestViewports(
	prediction: number[][][],
	W: number, H: number,
	viewportSize: number,
	count: number,
): { x: number; y: number }[] {
	const entropyMap: number[][] = [];
	for (let y = 0; y < H; y++) {
		entropyMap.push([]);
		for (let x = 0; x < W; x++) {
			entropyMap[y].push(entropy(prediction[y][x]));
		}
	}

	const viewports: { x: number; y: number; score: number }[] = [];

	for (let vy = 0; vy <= H - viewportSize; vy++) {
		for (let vx = 0; vx <= W - viewportSize; vx++) {
			let score = 0;
			for (let dy = 0; dy < viewportSize; dy++)
				for (let dx = 0; dx < viewportSize; dx++)
					score += entropyMap[vy + dy][vx + dx];
			viewports.push({ x: vx, y: vy, score });
		}
	}

	viewports.sort((a, b) => b.score - a.score);

	const selected: { x: number; y: number }[] = [];
	const covered = new Set<string>();

	for (const vp of viewports) {
		if (selected.length >= count) break;

		let newCells = 0;
		for (let dy = 0; dy < viewportSize; dy++)
			for (let dx = 0; dx < viewportSize; dx++) {
				const key = `${vp.x + dx},${vp.y + dy}`;
				if (!covered.has(key) && entropyMap[vp.y + dy][vp.x + dx] > 0.1)
					newCells++;
			}

		if (newCells < 10 && selected.length > 0) continue;

		selected.push({ x: vp.x, y: vp.y });
		for (let dy = 0; dy < viewportSize; dy++)
			for (let dx = 0; dx < viewportSize; dx++)
				covered.add(`${vp.x + dx},${vp.y + dy}`);
	}

	return selected;
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

async function main() {
	const args = Deno.args;
	if (args.length < 2) {
		console.log("Usage: query-and-refine.ts <round_number> <seed> [queries_per_seed=10] [viewport_size=15]");
		Deno.exit(1);
	}

	const roundNum = parseInt(args[0]);
	const seedIdx = parseInt(args[1]);
	const queriesPerSeed = parseInt(args[2] || "10");
	const viewportSize = parseInt(args[3] || "15");

	const roundsData = JSON.parse(await Deno.readTextFile("data/my-rounds.json"));
	const roundInfo = roundsData.find((r: { round_number: number }) => r.round_number === roundNum);
	if (!roundInfo) { console.error(`Round ${roundNum} not found`); Deno.exit(1); }
	if (roundInfo.status !== "active") {
		console.error(`Round ${roundNum} is ${roundInfo.status}, not active`);
		Deno.exit(1);
	}

	const predPath = `${BIN_DIR}/pred_r${roundNum}_s${seedIdx}.bin`;
	console.log(`Loading model prediction: ${predPath}`);
	const pred = readPrediction(predPath);

	const viewports = findBestViewports(pred.prediction, pred.W, pred.H, viewportSize, Math.ceil(queriesPerSeed / 3));
	console.log(`Placed ${viewports.length} viewports for ${queriesPerSeed} queries:`);
	for (const vp of viewports)
		console.log(`  viewport (${vp.x}, ${vp.y}) ${viewportSize}x${viewportSize}`);

	const sampleCounts: number[][] = Array.from({ length: pred.H }, () => new Array(pred.W).fill(0));
	const classCounts: number[][][] = Array.from({ length: pred.H }, () =>
		Array.from({ length: pred.W }, () => new Array(NUM_CLASSES).fill(0))
	);

	const queryDir = `data/queries/r${roundNum}`;
	try { await Deno.mkdir(queryDir, { recursive: true }); } catch { /* */ }

	const rawResults: { query: number; viewport: { x: number; y: number }; grid: number[][]; settlements: unknown[] }[] = [];

	let queriesUsed = 0;
	for (let q = 0; q < queriesPerSeed; q++) {
		const vp = viewports[q % viewports.length];
		console.log(`  Query ${q + 1}/${queriesPerSeed}: viewport (${vp.x}, ${vp.y})`);

		try {
			const result = await api.simulate({
				round_id: roundInfo.id,
				seed_index: seedIdx,
				viewport_x: vp.x,
				viewport_y: vp.y,
				viewport_w: viewportSize,
				viewport_h: viewportSize,
			});

			rawResults.push({
				query: q,
				viewport: { x: vp.x, y: vp.y },
				grid: result.grid,
				settlements: result.settlements,
			});

			for (let gy = 0; gy < result.grid.length; gy++) {
				for (let gx = 0; gx < result.grid[gy].length; gx++) {
					const mapX = vp.x + gx;
					const mapY = vp.y + gy;
					if (mapX >= pred.W || mapY >= pred.H) continue;
					const cls = terrainToClass(result.grid[gy][gx]);
					classCounts[mapY][mapX][cls]++;
					sampleCounts[mapY][mapX]++;
				}
			}

			queriesUsed++;
			console.log(`    queries: ${result.queries_used}/${result.queries_max}`);
		} catch (e) {
			console.error(`    failed: ${e}`);
			break;
		}
	}

	const queryFile = `${queryDir}/s${seedIdx}_queries.json`;
	await Deno.writeTextFile(queryFile, JSON.stringify(rawResults, null, 2));
	console.log(`Saved ${rawResults.length} raw query results to ${queryFile}`);

	console.log(`\nBlending ${queriesUsed} query results with model prediction...`);

	const blended = pred.prediction.map(row => row.map(cell => [...cell]));
	let refinedCells = 0;

	for (let y = 0; y < pred.H; y++) {
		for (let x = 0; x < pred.W; x++) {
			const n = sampleCounts[y][x];
			if (n === 0) continue;

			const empirical: number[] = classCounts[y][x].map(c => c / n);
			const alpha = Math.min(n / (n + 5), 0.8); // more samples = trust empirical more, cap at 0.8

			for (let c = 0; c < NUM_CLASSES; c++)
				blended[y][x][c] = (1 - alpha) * pred.prediction[y][x][c] + alpha * empirical[c];

			// Floor + renormalize
			let total = 0;
			for (let c = 0; c < NUM_CLASSES; c++) {
				if (blended[y][x][c] < 0.01) blended[y][x][c] = 0.01;
				total += blended[y][x][c];
			}
			for (let c = 0; c < NUM_CLASSES; c++) blended[y][x][c] /= total;

			refinedCells++;
		}
	}

	console.log(`Refined ${refinedCells} cells with empirical data`);

	const outPath = `${BIN_DIR}/pred_r${roundNum}_s${seedIdx}.bin`;
	writePrediction(outPath, roundNum, seedIdx, pred.W, pred.H, blended);
	console.log(`Written refined prediction to ${outPath}`);
}

await main();
