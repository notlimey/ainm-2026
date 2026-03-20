import { fetchAndStoreRounds } from "./rounds.ts";
export function add(a: number, b: number): number {
	return a + b;
}

if (!import.meta.main)
	throw new Error("This module is not meant to be imported");

await fetchAndStoreRounds();
