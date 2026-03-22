// Shared binary I/O utilities for ASTP/grid/ground-truth formats.
// Used by: query-round.ts, preview-prediction.ts, analyze-errors.ts, analyze-per-class.ts

export const NUM_CLASSES = 6;

// Terrain codes (matching database.hpp)
export const TERRAIN_MOUNTAIN = 5;
export const TERRAIN_OCEAN = 10;

export function terrainToClass(code: number): number {
	switch (code) {
		case 1: return 1;
		case 2: return 2;
		case 3: return 3;
		case 4: return 4;
		case 5: return 5;
		default: return 0; // 0, 10, 11 -> class 0
	}
}

export function entropy(probs: number[]): number {
	let h = 0;
	for (const p of probs) if (p > 0) h -= p * Math.log2(p);
	return h;
}

// Read ASTP prediction binary (with magic validation)
export function readPrediction(path: string) {
	const data = Deno.readFileSync(path);
	const view = new DataView(data.buffer);
	const magic = String.fromCharCode(data[0], data[1], data[2], data[3]);
	if (magic !== "ASTP") throw new Error(`Bad magic in ${path}: ${magic}`);
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

// Write ASTP prediction binary
export function writePrediction(path: string, round: number, seed: number, W: number, H: number, prediction: number[][][]) {
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

// Read initial grid from grids.bin
export function readGridBin(path: string, wantRound: number, wantSeed: number) {
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
			const grid: number[][] = [];
			for (let y = 0; y < H; y++) { const row: number[] = []; for (let x = 0; x < W; x++) { row.push(view.getInt32(offset, true)); offset += 4; } grid.push(row); }
			return { W, H, grid };
		}
		offset += W * H * 4;
	}
	return null;
}

// Read ground truth from ground_truth.bin
export function readGroundTruthBin(path: string, wantRound: number, wantSeed: number) {
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
				return gt;
			}
			offset += W * H * NUM_CLASSES * 4;
		}
	} catch { /* file not found or parse error */ }
	return null;
}
