import { scrapeSimulation } from "./scraper.ts";

export function add(a: number, b: number): number {
	return a + b;
}

if (!import.meta.main)
	throw new Error("This module is not meant to be imported");

await scrapeSimulation(
	"https://app.ainm.no/submit/astar-island/replay?round=71451d74-be9f-471f-aacd-a41f3b68a9cd",
);
