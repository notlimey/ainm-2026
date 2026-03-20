import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";

const api = new NMAIAstarIsland();
const BIN_DIR = "../simulation/data";
const NUM_CLASSES = 6;

function readF32(view: DataView, offset: number): number {
	return view.getFloat32(offset, true);
}

function loadPrediction(path: string): { round: number; seed: number; W: number; H: number; prediction: number[][][] } {
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
			for (let c = 0; c < NUM_CLASSES; c++) {
				cell.push(readF32(view, offset));
				offset += 4;
			}
			row.push(cell);
		}
		prediction.push(row);
	}

	return { round, seed, W, H, prediction };
}

async function main() {
	const roundsData = JSON.parse(await Deno.readTextFile("data/my-rounds.json"));
	const roundMap = new Map<number, string>();
	for (const r of roundsData) {
		roundMap.set(r.round_number, r.id);
	}

	const args = Deno.args;
	if (args.length === 0) {
		console.log("Usage: submit-predictions.ts <prediction.bin> [prediction2.bin ...]");
		console.log("   or: submit-predictions.ts --round <N>  (submits all data/pred_r{N}_s*.bin)");
		Deno.exit(1);
	}

	let files: string[] = [];

	if (args[0] === "--round") {
		const roundNum = parseInt(args[1]);
		for (let seed = 0; seed < 5; seed++) {
			const path = `${BIN_DIR}/pred_r${roundNum}_s${seed}.bin`;
			try {
				await Deno.stat(path);
				files.push(path);
			} catch {
				console.log(`  Skipping seed ${seed} (${path} not found)`);
			}
		}
	} else {
		files = args;
	}

	for (const file of files) {
		const pred = loadPrediction(file);
		const roundId = roundMap.get(pred.round);
		if (!roundId) {
			console.error(`  No round ID for round ${pred.round}, skipping`);
			continue;
		}

		console.log(`Submitting [r${pred.round}.s${pred.seed}] ${pred.W}x${pred.H} -> ${roundId}`);

		try {
			const result = await api.submit({
				round_id: roundId,
				seed_index: pred.seed,
				prediction: pred.prediction,
			});
			console.log(`  ${result.status}`);
		} catch (e) {
			console.error(`  Failed: ${e}`);
		}
	}
}

await main();
