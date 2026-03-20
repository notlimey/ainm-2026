import { NMAIAstarIsland } from "./client.ts";

export const fetchAndStoreRounds = async () => {
	const api = new NMAIAstarIsland();
	const rounds = await api.getRounds();

	const FOLDER = ["data", "rounds"];

	for (const round of rounds) {
		const details = await api.getRound(round.id);
		console.log(
			`Round ${round.round_number} (${round.id}): ${details.seeds_count} seeds`,
		);

		for (let i = 0; i < details.seeds_count; i++) {
			// roundNumber_seedIndex_year
			const initial = details.initial_states[i];

			const id = `r${round.round_number}_s${i}_y0`; // Since its initial state, we can use 0 for year
			const filename = `${id}.json`;
			const path = [...FOLDER, filename].join("/");
			await Deno.writeTextFile(path, JSON.stringify(initial));
		}
	}
};
