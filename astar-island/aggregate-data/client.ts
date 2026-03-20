import "@std/dotenv/load";

const BASE_URL = "https://api.ainm.no";
function getToken(): string {
	const token = Deno.env.get("NMAI_BEARER_TOKEN");
	if (!token) throw new Error("NMAI_BEARER_TOKEN not set in environment");
	return token;
}

// --- Types ---

export interface Round {
	id: string;
	round_number: number;
	event_date: string;
	status: "pending" | "active" | "scoring" | "completed";
	map_width: number;
	map_height: number;
	prediction_window_minutes: number;
	started_at: string;
	closes_at: string;
	round_weight: number;
	created_at: string;
}

export interface Settlement {
	x: number;
	y: number;
	has_port: boolean;
	alive: boolean;
}

export interface InitialState {
	grid: number[][];
	settlements: Settlement[];
}

export interface RoundDetail {
	id: string;
	round_number: number;
	status: Round["status"];
	map_width: number;
	map_height: number;
	seeds_count: number;
	initial_states: InitialState[];
}

export interface Budget {
	round_id: string;
	queries_used: number;
	queries_max: number;
	active: boolean;
}

export interface SimulateRequest {
	round_id: string;
	seed_index: number;
	viewport_x?: number;
	viewport_y?: number;
	viewport_w?: number;
	viewport_h?: number;
}

export interface SimulatedSettlement {
	x: number;
	y: number;
	population: number;
	food: number;
	wealth: number;
	defense: number;
	has_port: boolean;
	alive: boolean;
	owner_id: number;
}

export interface SimulateResponse {
	grid: number[][];
	settlements: SimulatedSettlement[];
	viewport: { x: number; y: number; w: number; h: number };
	width: number;
	height: number;
	queries_used: number;
	queries_max: number;
}

export interface SubmitRequest {
	round_id: string;
	seed_index: number;
	prediction: number[][][];
}

export interface SubmitResponse {
	status: string;
	round_id: string;
	seed_index: number;
}

export interface MyRound extends Round {
	seeds_count: number;
	round_score: number | null;
	seed_scores: number[] | null;
	seeds_submitted: number;
	rank: number | null;
	total_teams: number | null;
	queries_used: number;
	queries_max: number;
	initial_grid: number[][];
}

export interface MyPrediction {
	seed_index: number;
	argmax_grid: number[][];
	confidence_grid: number[][];
	score: number | null;
	submitted_at: string | null;
}

export interface Analysis {
	prediction: number[][][];
	ground_truth: number[][][];
	score: number | null;
	width: number;
	height: number;
	initial_grid: number[][] | null;
}

export interface LeaderboardEntry {
	team_id: string;
	team_name: string;
	team_slug: string;
	weighted_score: number;
	rounds_participated: number;
	hot_streak_score: number;
	rank: number;
	is_verified: boolean;
}

// --- Client ---

export class NMAIAstarIsland {
	private token: string;

	constructor(token: string = getToken()) {
		this.token = token;
	}

	private async request<T>(
		method: string,
		path: string,
		body?: unknown,
	): Promise<T> {
		const res = await fetch(`${BASE_URL}${path}`, {
			method,
			headers: {
				Authorization: `Bearer ${this.token}`,
				"Content-Type": "application/json",
			},
			body: body ? JSON.stringify(body) : undefined,
		});
		if (!res.ok) {
			const text = await res.text();
			throw new Error(`${method} ${path} failed (${res.status}): ${text}`);
		}
		return res.json() as Promise<T>;
	}

	/** List all rounds with status and timing. (Public) */
	getRounds(): Promise<Round[]> {
		return this.request("GET", "/astar-island/rounds");
	}

	/** Get round details including initial map states for all seeds. (Public) */
	getRound(roundId: string): Promise<RoundDetail> {
		return this.request("GET", `/astar-island/rounds/${roundId}`);
	}

	/** Check remaining query budget for the active round. */
	getBudget(): Promise<Budget> {
		return this.request("GET", "/astar-island/budget");
	}

	/** Run one stochastic simulation and observe a viewport window. Costs 1 query. */
	simulate(params: SimulateRequest): Promise<SimulateResponse> {
		return this.request("POST", "/astar-island/simulate", params);
	}

	/** Submit a prediction tensor (H x W x 6) for one seed. Resubmitting overwrites. */
	submit(params: SubmitRequest): Promise<SubmitResponse> {
		return this.request("POST", "/astar-island/submit", params);
	}

	/** Get all rounds enriched with your team's scores, rank, and budget. */
	getMyRounds(): Promise<MyRound[]> {
		return this.request("GET", "/astar-island/my-rounds");
	}

	/** Get your submitted predictions for a round with argmax/confidence grids. */
	getMyPredictions(roundId: string): Promise<MyPrediction[]> {
		return this.request("GET", `/astar-island/my-predictions/${roundId}`);
	}

	/** Post-round ground truth comparison. Only available after round completes. */
	getAnalysis(roundId: string, seedIndex: number): Promise<Analysis> {
		return this.request(
			"GET",
			`/astar-island/analysis/${roundId}/${seedIndex}`,
		);
	}

	/** Public leaderboard — best round score per team. */
	getLeaderboard(): Promise<LeaderboardEntry[]> {
		return this.request("GET", "/astar-island/leaderboard");
	}
}
