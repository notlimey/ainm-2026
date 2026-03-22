/**
 * Comprehensive data dump — fetch everything from the AINM API before it goes away.
 *
 * Fetches: rounds, my-rounds, leaderboard, my-predictions, analysis, initial states.
 */
import { NMAIAstarIsland } from "./client.ts";
import "@std/dotenv/load";

const api = new NMAIAstarIsland();
const DATA_DIR = "data";
const DUMP_DIR = `${DATA_DIR}/dump`;

async function ensureDir(path: string) {
	try {
		await Deno.mkdir(path, { recursive: true });
	} catch (e) {
		if (!(e instanceof Deno.errors.AlreadyExists)) throw e;
	}
}

async function saveJSON(path: string, data: unknown) {
	await Deno.writeTextFile(path, JSON.stringify(data, null, 2));
	console.log(`  Saved: ${path}`);
}

// 1. Public rounds list
async function fetchRounds() {
	console.log("\n=== Fetching public rounds list ===");
	const rounds = await api.getRounds();
	await saveJSON(`${DUMP_DIR}/rounds.json`, rounds);
	console.log(`  ${rounds.length} rounds total`);
	return rounds;
}

// 2. My rounds (refreshed)
async function fetchMyRounds() {
	console.log("\n=== Fetching my-rounds (refreshed) ===");
	const rounds = await api.getMyRounds();
	await saveJSON(`${DUMP_DIR}/my-rounds.json`, rounds);
	// Also update the existing one
	await saveJSON(`${DATA_DIR}/my-rounds.json`, rounds);
	console.log(`  ${rounds.length} rounds`);
	return rounds;
}

// 3. Leaderboard
async function fetchLeaderboard() {
	console.log("\n=== Fetching leaderboard ===");
	const lb = await api.getLeaderboard();
	await saveJSON(`${DUMP_DIR}/leaderboard.json`, lb);
	console.log(`  ${lb.length} teams`);
	return lb;
}

// 4. My predictions for each round
async function fetchMyPredictions(myRounds: Awaited<ReturnType<typeof api.getMyRounds>>) {
	console.log("\n=== Fetching my predictions ===");
	const predDir = `${DUMP_DIR}/my-predictions`;
	await ensureDir(predDir);

	for (const round of myRounds) {
		const outPath = `${predDir}/r${round.round_number}.json`;
		try {
			await Deno.stat(outPath);
			console.log(`  [r${round.round_number}] already exists, skipping`);
			continue;
		} catch { /* doesn't exist */ }

		try {
			const preds = await api.getMyPredictions(round.id);
			if (preds && preds.length > 0) {
				await saveJSON(outPath, preds);
				console.log(`  [r${round.round_number}] ${preds.length} seed predictions`);
			} else {
				console.log(`  [r${round.round_number}] no predictions submitted`);
			}
		} catch (e) {
			console.error(`  [r${round.round_number}] failed: ${e}`);
		}
	}
}

// 5. Round details (initial states) — for any we might have missed
async function fetchRoundDetails(rounds: Awaited<ReturnType<typeof api.getRounds>>) {
	console.log("\n=== Fetching round details (initial states) ===");
	const initDir = `${DATA_DIR}/initial`;
	await ensureDir(initDir);

	for (const round of rounds) {
		const outPath = `${initDir}/r${round.round_number}.json`;
		try {
			await Deno.stat(outPath);
			console.log(`  [r${round.round_number}] already exists, skipping`);
			continue;
		} catch { /* doesn't exist */ }

		try {
			const detail = await api.getRound(round.id);
			await saveJSON(outPath, detail);
			console.log(`  [r${round.round_number}] saved — ${detail.seeds_count} seeds`);
		} catch (e) {
			console.error(`  [r${round.round_number}] failed: ${e}`);
		}
	}
}

// 6. Analysis/ground truth for completed rounds
async function fetchAllAnalysis(myRounds: Awaited<ReturnType<typeof api.getMyRounds>>) {
	console.log("\n=== Fetching analysis/ground truth ===");
	const analysisDir = `${DATA_DIR}/analysis`;
	await ensureDir(analysisDir);

	const completed = myRounds.filter((r) => r.status === "completed");
	for (const round of completed) {
		const roundDir = `${analysisDir}/r${round.round_number}`;
		await ensureDir(roundDir);

		for (let seed = 0; seed < round.seeds_count; seed++) {
			const outPath = `${roundDir}/s${seed}.json`;
			try {
				await Deno.stat(outPath);
				console.log(`  [r${round.round_number}.s${seed}] already exists, skipping`);
				continue;
			} catch { /* doesn't exist */ }

			try {
				const analysis = await api.getAnalysis(round.id, seed);
				await Deno.writeTextFile(outPath, JSON.stringify(analysis));
				console.log(`  [r${round.round_number}.s${seed}] saved — score: ${analysis.score?.toFixed(2) ?? "n/a"}`);
			} catch (e) {
				console.error(`  [r${round.round_number}.s${seed}] failed: ${e}`);
			}
		}
	}
}

// Main
async function main() {
	await ensureDir(DUMP_DIR);
	console.log("=== AINM Astar Island — Full Data Dump ===");
	console.log(`Timestamp: ${new Date().toISOString()}`);

	const [rounds, myRounds, _lb] = await Promise.all([
		fetchRounds(),
		fetchMyRounds(),
		fetchLeaderboard(),
	]);

	await fetchMyPredictions(myRounds);
	await fetchRoundDetails(rounds);
	await fetchAllAnalysis(myRounds);

	console.log("\n=== Done! ===");
	console.log(`All data saved to ${DATA_DIR}/ and ${DUMP_DIR}/`);
}

main();
