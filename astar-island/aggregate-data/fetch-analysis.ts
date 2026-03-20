import { NMAIAstarIsland, type Analysis, type MyRound } from "./client.ts";
import "@std/dotenv/load";

const api = new NMAIAstarIsland();
const DATA_DIR = "data";

async function ensureDir(path: string) {
	try {
		await Deno.mkdir(path, { recursive: true });
	} catch (e) {
		if (!(e instanceof Deno.errors.AlreadyExists)) throw e;
	}
}

async function fetchMyRounds(): Promise<MyRound[]> {
	console.log("Fetching my-rounds...\n");
	const rounds = await api.getMyRounds();

	console.log("Round  Status      Score   Rank     Seeds  Queries");
	console.log("─────  ──────────  ──────  ───────  ─────  ───────");
	for (const r of rounds) {
		const score = r.round_score !== null ? r.round_score.toFixed(1) : "  -";
		const rank =
			r.rank !== null ? `${r.rank}/${r.total_teams}` : "  -";
		console.log(
			`  ${String(r.round_number).padEnd(3)}  ${r.status.padEnd(10)}  ${String(score).padStart(6)}  ${String(rank).padStart(7)}  ${r.seeds_submitted}/${r.seeds_count}  ${r.queries_used}/${r.queries_max}`,
		);
	}
	console.log();

	const path = `${DATA_DIR}/my-rounds.json`;
	await Deno.writeTextFile(path, JSON.stringify(rounds, null, 2));
	console.log(`Saved to ${path}\n`);

	return rounds;
}

async function fetchAnalysis(rounds: MyRound[]) {
	const completed = rounds.filter((r) => r.status === "completed");

	if (completed.length === 0) {
		console.log("No completed rounds found — nothing to fetch.\n");
		return;
	}

	console.log(
		`Fetching analysis for ${completed.length} completed round(s)...\n`,
	);

	const analysisDir = `${DATA_DIR}/analysis`;
	await ensureDir(analysisDir);

	for (const round of completed) {
		const roundDir = `${analysisDir}/r${round.round_number}`;
		await ensureDir(roundDir);

		for (let seed = 0; seed < round.seeds_count; seed++) {
			const outPath = `${roundDir}/s${seed}.json`;

			try {
				await Deno.stat(outPath);
				console.log(
					`  [r${round.round_number}.s${seed}] already exists, skipping`,
				);
				continue;
			} catch { /* doesn't exist */ }

			try {
				console.log(
					`  [r${round.round_number}.s${seed}] fetching analysis...`,
				);
				const analysis: Analysis = await api.getAnalysis(
					round.id,
					seed,
				);

				await Deno.writeTextFile(
					outPath,
					JSON.stringify(analysis),
				);

				const gt = analysis.ground_truth;
				const h = gt.length;
				const w = gt[0].length;
				const score =
					analysis.score !== null
						? analysis.score.toFixed(2)
						: "n/a";
				console.log(
					`  [r${round.round_number}.s${seed}] saved — ${w}x${h} ground truth, score: ${score}`,
				);
			} catch (e) {
				console.error(
					`  [r${round.round_number}.s${seed}] failed: ${e}`,
				);
			}
		}
	}
	console.log();
}

async function fetchInitialStates(rounds: MyRound[]) {
	const available = rounds.filter((r) => r.status === "completed" || r.status === "active");

	const initDir = `${DATA_DIR}/initial`;
	await ensureDir(initDir);

	for (const round of available) {
		const outPath = `${initDir}/r${round.round_number}.json`;

		try {
			await Deno.stat(outPath);
			console.log(
				`  [r${round.round_number}] initial states already exist, skipping`,
			);
			continue;
		} catch { /* doesn't exist */ }

		try {
			console.log(
				`  [r${round.round_number}] fetching round detail...`,
			);
			const detail = await api.getRound(round.id);
			await Deno.writeTextFile(
				outPath,
				JSON.stringify(detail, null, 2),
			);
			console.log(
				`  [r${round.round_number}] saved — ${detail.seeds_count} seeds, ${detail.map_width}x${detail.map_height}`,
			);
		} catch (e) {
			console.error(`  [r${round.round_number}] failed: ${e}`);
		}
	}
	console.log();
}

const rounds = await fetchMyRounds();
await fetchInitialStates(rounds);
await fetchAnalysis(rounds);

console.log("Done. Training data structure:");
console.log("  data/initial/r{N}.json    — initial states (input)");
console.log("  data/analysis/r{N}/s{M}.json — ground truth (target)");
