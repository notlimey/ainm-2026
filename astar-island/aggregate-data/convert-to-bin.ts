import "@std/dotenv/load";

const DATA_DIR = "data";
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

function isLand(code: number): boolean {
	return code !== 10;
}

function writeU8(buf: number[], val: number) { buf.push(val & 0xFF); }
function writeU16(buf: number[], val: number) { buf.push(val & 0xFF, (val >> 8) & 0xFF); }
function writeU32(buf: number[], val: number) {
	buf.push(val & 0xFF, (val >> 8) & 0xFF, (val >> 16) & 0xFF, (val >> 24) & 0xFF);
}
function writeI32(buf: number[], val: number) { writeU32(buf, val); }
function writeF32(buf: number[], val: number) {
	const f = new Float32Array([val]);
	const b = new Uint8Array(f.buffer);
	buf.push(b[0], b[1], b[2], b[3]);
}

const DX = [0, 1, 1, 1, 0, -1, -1, -1];
const DY = [-1, -1, 0, 1, 1, 1, 0, -1];

interface CellFeatures {
	x: number; y: number;
	terrain_class: number; is_land: number;
	neighbors: number[];
	adj_ocean: number; adj_forest: number; adj_mountain: number;
	adj_settlement: number; adj_plains: number;
	dist_nearest_settlement: number; dist_nearest_port: number;
	settlements_r3: number; settlements_r5: number;
	is_coastal: number; dist_coast: number;
	land_cells_5x5: number;
}

function extractFeatures(grid: number[][]): CellFeatures[] {
	const H = grid.length;
	const W = grid[0].length;

	const settlements: [number, number][] = [];
	const ports: [number, number][] = [];
	const oceans: [number, number][] = [];

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const t = grid[y][x];
			if (t === 1) settlements.push([x, y]);
			if (t === 2) { ports.push([x, y]); settlements.push([x, y]); }
			if (t === 10) oceans.push([x, y]);
		}
	}

	const features: CellFeatures[] = [];

	for (let y = 0; y < H; y++) {
		for (let x = 0; x < W; x++) {
			const code = grid[y][x];
			const f: CellFeatures = {
				x, y,
				terrain_class: terrainToClass(code),
				is_land: isLand(code) ? 1 : 0,
				neighbors: new Array(8),
				adj_ocean: 0, adj_forest: 0, adj_mountain: 0,
				adj_settlement: 0, adj_plains: 0,
				dist_nearest_settlement: 999, dist_nearest_port: 999,
				settlements_r3: 0, settlements_r5: 0,
				is_coastal: 0, dist_coast: 999,
				land_cells_5x5: 0,
			};

			for (let d = 0; d < 8; d++) {
				const nx = x + DX[d], ny = y + DY[d];
				if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
					f.neighbors[d] = 255;
					continue;
				}
				f.neighbors[d] = terrainToClass(grid[ny][nx]);
				const nt = grid[ny][nx];
				if (nt === 10) f.adj_ocean++;
				if (nt === 4) f.adj_forest++;
				if (nt === 5) f.adj_mountain++;
				if (nt === 1 || nt === 2) f.adj_settlement++;
				if (nt === 11 || nt === 0) f.adj_plains++;
			}

			f.is_coastal = (f.is_land && f.adj_ocean > 0) ? 1 : 0;

			for (const [sx, sy] of settlements) {
				const d = Math.sqrt((x - sx) ** 2 + (y - sy) ** 2);
				if (d < f.dist_nearest_settlement) f.dist_nearest_settlement = d;
				const md = Math.abs(x - sx) + Math.abs(y - sy);
				if (md <= 3) f.settlements_r3++;
				if (md <= 5) f.settlements_r5++;
			}
			if (settlements.length === 0) f.dist_nearest_settlement = 99;

			for (const [px, py] of ports) {
				const d = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
				if (d < f.dist_nearest_port) f.dist_nearest_port = d;
			}
			if (ports.length === 0) f.dist_nearest_port = 99;

			for (const [ox, oy] of oceans) {
				const d = Math.sqrt((x - ox) ** 2 + (y - oy) ** 2);
				if (d < f.dist_coast) f.dist_coast = d;
			}
			if (oceans.length === 0) f.dist_coast = 99;

			for (let dy = -2; dy <= 2; dy++) {
				for (let dx = -2; dx <= 2; dx++) {
					const nx = x + dx, ny = y + dy;
					if (nx >= 0 && nx < W && ny >= 0 && ny < H && isLand(grid[ny][nx]))
						f.land_cells_5x5++;
				}
			}

			features.push(f);
		}
	}

	return features;
}

function writeCellFeatures(buf: number[], f: CellFeatures) {
	writeU8(buf, f.x);
	writeU8(buf, f.y);
	writeU8(buf, f.terrain_class);
	writeU8(buf, f.is_land);
	for (let d = 0; d < 8; d++) writeU8(buf, f.neighbors[d]);
	writeU8(buf, f.adj_ocean);
	writeU8(buf, f.adj_forest);
	writeU8(buf, f.adj_mountain);
	writeU8(buf, f.adj_settlement);
	writeU8(buf, f.adj_plains);
	writeU8(buf, 0); writeU8(buf, 0); writeU8(buf, 0); // padding to offset 20
	writeF32(buf, f.dist_nearest_settlement);
	writeF32(buf, f.dist_nearest_port);
	writeU8(buf, f.settlements_r3);
	writeU8(buf, f.settlements_r5);
	writeU8(buf, f.is_coastal);
	writeU8(buf, 0); // padding to offset 32
	writeF32(buf, f.dist_coast);
	writeU8(buf, f.land_cells_5x5);
	writeU8(buf, 0); writeU8(buf, 0); writeU8(buf, 0); // padding to 40 bytes
}

async function buildTraining() {
	const analysisDir = `${DATA_DIR}/analysis`;
	const buf: number[] = [];

	buf.push(0x41, 0x53, 0x54, 0x46); // "ASTF"
	writeU16(buf, 2); // v2: round+seed per sample
	writeU32(buf, 0); // num_samples placeholder (offset 6)
	writeU16(buf, 40);
	writeU16(buf, NUM_CLASSES);

	let totalSamples = 0;
	let dynamicCount = 0;

	for await (const roundEntry of Deno.readDir(analysisDir)) {
		if (!roundEntry.isDirectory || !roundEntry.name.startsWith("r")) continue;
		const roundNum = parseInt(roundEntry.name.slice(1));

		for await (const seedEntry of Deno.readDir(`${analysisDir}/${roundEntry.name}`)) {
			if (!seedEntry.name.startsWith("s") || !seedEntry.name.endsWith(".json")) continue;
			const seedNum = parseInt(seedEntry.name.slice(1));

			const raw = await Deno.readTextFile(`${analysisDir}/${roundEntry.name}/${seedEntry.name}`);
			const analysis = JSON.parse(raw);

			if (!analysis.initial_grid) {
				console.log(`  [r${roundNum}.s${seedNum}] skipped (no initial_grid)`);
				continue;
			}

			const grid: number[][] = analysis.initial_grid;
			const gt: number[][][] = analysis.ground_truth;
			const H = grid.length, W = grid[0].length;
			const features = extractFeatures(grid);

			let dyn = 0;
			for (let y = 0; y < H; y++) {
				for (let x = 0; x < W; x++) {
					const f = features[y * W + x];
					writeU16(buf, roundNum);
					writeU16(buf, seedNum);
					writeCellFeatures(buf, f);
					for (let c = 0; c < NUM_CLASSES; c++)
						writeF32(buf, gt[y][x][c]);

					let isDynamic = false;
					for (let c = 1; c < NUM_CLASSES; c++)
						if (gt[y][x][c] > 0.01) isDynamic = true;
					if (f.is_land && gt[y][x][0] < 0.95) isDynamic = true;
					if (isDynamic) dyn++;

					totalSamples++;
				}
			}

			dynamicCount += dyn;
			console.log(`  [r${roundNum}.s${seedNum}] ${W}x${H} = ${W * H} cells (${dyn} dynamic)`);
		}
	}

	const arr = new Uint8Array(buf);
	new DataView(arr.buffer).setUint32(6, totalSamples, true);

	const outPath = `${BIN_DIR}/training.bin`;
	await Deno.writeFile(outPath, arr);
	console.log(`\nTraining: ${totalSamples} samples (${dynamicCount} dynamic) -> ${outPath}`);
}

async function buildGrids() {
	const buf: number[] = [];
	buf.push(0x41, 0x53, 0x54, 0x47); // "ASTG"
	writeU16(buf, 1);
	writeU32(buf, 0);

	let count = 0;

	const analysisDir = `${DATA_DIR}/analysis`;
	try {
		for await (const roundEntry of Deno.readDir(analysisDir)) {
			if (!roundEntry.isDirectory || !roundEntry.name.startsWith("r")) continue;
			const roundNum = parseInt(roundEntry.name.slice(1));

			for await (const seedEntry of Deno.readDir(`${analysisDir}/${roundEntry.name}`)) {
				if (!seedEntry.name.startsWith("s") || !seedEntry.name.endsWith(".json")) continue;
				const seedNum = parseInt(seedEntry.name.slice(1));

				const raw = await Deno.readTextFile(`${analysisDir}/${roundEntry.name}/${seedEntry.name}`);
				const analysis = JSON.parse(raw);
				if (!analysis.initial_grid) continue;

				const grid: number[][] = analysis.initial_grid;
				const H = grid.length, W = grid[0].length;
				writeI32(buf, roundNum);
				writeI32(buf, seedNum);
				writeI32(buf, W);
				writeI32(buf, H);
				for (let y = 0; y < H; y++)
					for (let x = 0; x < W; x++)
						writeI32(buf, grid[y][x]);
				count++;
			}
		}
	} catch { /* */ }

	const initDir = `${DATA_DIR}/initial`;
	try {
		for await (const entry of Deno.readDir(initDir)) {
			if (!entry.name.startsWith("r") || !entry.name.endsWith(".json")) continue;
			const roundNum = parseInt(entry.name.slice(1));

			const raw = await Deno.readTextFile(`${initDir}/${entry.name}`);
			const detail = JSON.parse(raw);
			if (!detail.initial_states) continue;

			for (let seed = 0; seed < detail.initial_states.length; seed++) {
				const grid: number[][] = detail.initial_states[seed].grid;
				const H = grid.length, W = grid[0].length;
				writeI32(buf, roundNum);
				writeI32(buf, seed);
				writeI32(buf, W);
				writeI32(buf, H);
				for (let y = 0; y < H; y++)
					for (let x = 0; x < W; x++)
						writeI32(buf, grid[y][x]);
				count++;
			}
		}
	} catch { /* */ }

	const arr = new Uint8Array(buf);
	new DataView(arr.buffer).setUint32(6, count, true);

	const outPath = `${BIN_DIR}/grids.bin`;
	await Deno.writeFile(outPath, arr);
	console.log(`Grids: ${count} entries -> ${outPath}`);
}

async function buildGroundTruth() {
	const analysisDir = `${DATA_DIR}/analysis`;
	const buf: number[] = [];
	buf.push(0x41, 0x53, 0x54, 0x54); // "ASTT"
	writeU16(buf, 1);
	writeU32(buf, 0);

	let count = 0;

	for await (const roundEntry of Deno.readDir(analysisDir)) {
		if (!roundEntry.isDirectory || !roundEntry.name.startsWith("r")) continue;
		const roundNum = parseInt(roundEntry.name.slice(1));

		for await (const seedEntry of Deno.readDir(`${analysisDir}/${roundEntry.name}`)) {
			if (!seedEntry.name.startsWith("s") || !seedEntry.name.endsWith(".json")) continue;
			const seedNum = parseInt(seedEntry.name.slice(1));

			const raw = await Deno.readTextFile(`${analysisDir}/${roundEntry.name}/${seedEntry.name}`);
			const analysis = JSON.parse(raw);

			const gt: number[][][] = analysis.ground_truth;
			const H = gt.length, W = gt[0].length;

			writeI32(buf, roundNum);
			writeI32(buf, seedNum);
			writeI32(buf, W);
			writeI32(buf, H);
			for (let y = 0; y < H; y++)
				for (let x = 0; x < W; x++)
					for (let c = 0; c < NUM_CLASSES; c++)
						writeF32(buf, gt[y][x][c]);
			count++;
		}
	}

	const arr = new Uint8Array(buf);
	new DataView(arr.buffer).setUint32(6, count, true);

	const outPath = `${BIN_DIR}/ground_truth.bin`;
	await Deno.writeFile(outPath, arr);
	console.log(`Ground truth: ${count} entries -> ${outPath}`);
}

try { await Deno.mkdir(BIN_DIR, { recursive: true }); } catch { /* */ }

console.log("Converting JSON -> bin...\n");
await buildTraining();
console.log();
await buildGrids();
await buildGroundTruth();
console.log("\nDone. Files in simulation/data/:");
console.log("  training.bin     — features + ground truth (for model)");
console.log("  grids.bin        — initial state grids (for prediction input)");
console.log("  ground_truth.bin — raw ground truth (for validation)");
