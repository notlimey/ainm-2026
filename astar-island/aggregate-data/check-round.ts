#!/usr/bin/env -S deno run -A
/**
 * Lightweight round status checker. Outputs JSON for the bash watcher.
 *
 * Output: { "active_round": number|null, "round_id": string|null,
 *           "queries_used": number, "queries_max": number,
 *           "seeds_submitted": number, "closes_at": string|null,
 *           "minutes_remaining": number|null,
 *           "completed_unscored": number[] }
 */
import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";

try {
	const api = new NMAIAstarIsland();
	const myRounds = await api.getMyRounds();

	// Find active round
	const active = myRounds.find(r => r.status === "active");

	// Find completed rounds we might want to fetch GT for
	const completed = myRounds
		.filter(r => r.status === "completed")
		.map(r => r.round_number);

	if (active) {
		const closesAt = new Date(active.closes_at);
		const minutesRemaining = Math.max(0, (closesAt.getTime() - Date.now()) / 60000);

		console.log(JSON.stringify({
			active_round: active.round_number,
			round_id: active.id,
			queries_used: active.queries_used,
			queries_max: active.queries_max,
			seeds_submitted: active.seeds_submitted,
			closes_at: active.closes_at,
			minutes_remaining: Math.round(minutesRemaining),
			completed_rounds: completed,
		}));
	} else {
		console.log(JSON.stringify({
			active_round: null,
			round_id: null,
			queries_used: 0,
			queries_max: 0,
			seeds_submitted: 0,
			closes_at: null,
			minutes_remaining: null,
			completed_rounds: completed,
		}));
	}
} catch (e) {
	console.error(`API error: ${e}`);
	Deno.exit(1);
}
