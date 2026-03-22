import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";

const api = new NMAIAstarIsland();
const lb = await api.getLeaderboard();

console.log("Rank  Score  HotStrk  Rounds  Team");
console.log("─".repeat(65));
for (const e of lb.slice(0, 30)) {
	const us = e.team_slug.includes("entropy") || e.team_slug.includes("mkmy") ? " ◀◀◀" : "";
	console.log(
		String(e.rank).padStart(4) + "  " +
		e.weighted_score.toFixed(1).padStart(5) + "  " +
		e.hot_streak_score.toFixed(1).padStart(7) + "  " +
		String(e.rounds_participated).padStart(6) + "  " +
		e.team_name + us
	);
}
