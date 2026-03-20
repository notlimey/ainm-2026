import { scrapeSimulation } from "./scraper.ts";

if (!import.meta.main)
	throw new Error("This module is not meant to be imported");

const url = Deno.args[0];
if (!url) {
	console.log("Usage: deno run main.ts <replay_url>");
	Deno.exit(1);
}

await scrapeSimulation(url);
