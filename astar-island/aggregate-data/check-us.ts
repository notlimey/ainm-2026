import "@std/dotenv/load";
import { NMAIAstarIsland } from "./client.ts";
const api = new NMAIAstarIsland();

const [lb, myRounds] = await Promise.all([api.getLeaderboard(), api.getMyRounds()]);

// Find us
const us = lb.find(e => e.team_slug.includes("entropy") || e.team_slug.includes("mkmy"));
if (us) {
  console.log(`\nUs: Rank ${us.rank} | Score ${us.weighted_score.toFixed(1)} | HotStreak ${us.hot_streak_score.toFixed(1)} | Rounds ${us.rounds_participated}`);
  console.log(`Team: ${us.team_name} (${us.team_slug})`);
} else {
  console.log("\nNot found in leaderboard, checking all entries...");
  for (const e of lb) {
    console.log(`  ${e.rank}. ${e.team_name} (${e.team_slug}): ${e.weighted_score.toFixed(1)}`);
  }
}

// Our per-round scores
console.log("\nOur rounds:");
for (const r of myRounds.sort((a,b) => a.round_number - b.round_number)) {
  if (r.round_score !== null || r.seeds_submitted > 0) {
    console.log(`  R${r.round_number}: score=${r.round_score?.toFixed(1) ?? 'pending'} rank=${r.rank ?? '-'} weight=${r.round_weight} weighted=${r.round_score ? (r.round_score * r.round_weight).toFixed(1) : '-'} seeds=${r.seeds_submitted}/5`);
  }
}

// What do we need for top 5?
const top5score = lb[4]?.weighted_score ?? 0;
console.log(`\nTop 5 cutoff: ${top5score.toFixed(1)}`);
console.log(`Gap to top 5: need ${top5score.toFixed(1)} weighted score`);

// Check upcoming round weights
const rounds = await api.getRounds();
for (const r of rounds.filter(r => r.round_number >= 10)) {
  console.log(`  R${r.round_number}: weight=${r.round_weight} status=${r.status} | need raw score >= ${(top5score / r.round_weight).toFixed(1)} for top 5`);
}
