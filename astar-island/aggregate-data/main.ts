import { NMAIAstarIsland } from "./client.ts";

export function add(a: number, b: number): number {
	return a + b;
}

if (!import.meta.main)
	throw new Error("This module is not meant to be imported");

const api = new NMAIAstarIsland();
const rounds = await api.getRounds();
console.log("Rounds:", rounds);
