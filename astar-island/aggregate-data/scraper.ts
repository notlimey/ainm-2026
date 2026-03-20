/// <reference lib="deno.ns" />
import "@std/dotenv/load";
import { chromium, type Page } from "npm:playwright";
import { colorToTerrain } from "./colors.ts";

type Settlement = {
	x: number;
	y: number;
	has_port: boolean;
	alive: boolean;
	population?: number;
	food?: number;
	wealth?: number;
	defense?: number;
	faction?: number;
};

type StepData = {
	grid: number[][];
	settlements: Settlement[];
};

type RoundMeta = {
	round_number: number;
	seed_index: number;
	sim_seed: string;
	url: string;
};

function prompt(message: string): string {
	const buf = new Uint8Array(256);
	Deno.stdout.writeSync(new TextEncoder().encode(message));
	const n = Deno.stdin.readSync(buf);
	return new TextDecoder().decode(buf.subarray(0, n ?? 0)).trim();
}

export async function scrapeSimulation(url: string) {
	const roundNumber = parseInt(prompt("Round number: "), 10);
	const seedIndex = parseInt(prompt("Seed index: "), 10);

	const outDir = `./data/rounds/${roundNumber}/${seedIndex}`;
	Deno.mkdirSync(outDir, { recursive: true });

	const token = Deno.env.get("NMAI_BEARER_TOKEN");
	if (!token) throw new Error("NMAI_BEARER_TOKEN not set in .env");

	const browser = await chromium.launch({
		headless: false,
		slowMo: 50,
	});

	const context = await browser.newContext();
	await context.addCookies([
		{
			name: "access_token",
			value: token,
			domain: ".ainm.no",
			path: "/",
			httpOnly: true,
			secure: true,
			sameSite: "Lax",
		},
	]);

	const page = await context.newPage();
	await page.setViewportSize({ width: 1400, height: 1200 });
	await page.goto(url);
	await page.waitForTimeout(1000);

	// Read sim_seed from the UI
	const seedText = await page.getByText("sim_seed:").textContent();
	const simSeed = seedText?.match(/sim_seed:\s*(\d+)/)?.[1] ?? "unknown";
	console.log(`sim_seed: ${simSeed}`);

	// Save round metadata
	const meta: RoundMeta = {
		round_number: roundNumber,
		seed_index: seedIndex,
		sim_seed: simSeed,
		url,
	};
	Deno.writeTextFileSync(`${outDir}/meta.json`, JSON.stringify(meta, null, 4));
	console.log(`Saved ${outDir}/meta.json`);

	let step = 0;
	while (true) {
		console.log(`\n=== Step ${step} ===`);

		const data = await scrapeStep(page);
		const filename = `${outDir}/step_${step}.json`;
		Deno.writeTextFileSync(filename, JSON.stringify(data, null, 4));
		console.log(`  Saved ${filename}`);

		if (step >= 50) break;

		// Next step — click the skip-forward button
		const nextBtn = page.locator(".lucide-skip-forward").locator("..");
		if (await nextBtn.isDisabled()) break;

		await nextBtn.click();
		await page.waitForTimeout(400);
		step++;
	}

	console.log(`\nDone! ${step + 1} steps scraped for round ${roundNumber}, seed ${seedIndex}.`);
	await browser.close();
}

async function scrapeStep(page: Page): Promise<StepData> {
	const cells = await page.evaluate(() => {
		const gridContainer = document.querySelector(
			'.border.rounded-lg.overflow-hidden[style*="680px"]',
		);
		if (!gridContainer) return [];
		const rows = gridContainer.querySelectorAll(":scope > .flex");
		const result: { color: string | null; hasIcon: boolean; row: number; col: number }[] = [];
		let rowIdx = 0;
		for (const row of rows) {
			let colIdx = 0;
			for (const cell of row.children) {
				const style = (cell as HTMLElement).style.backgroundColor;
				const icon = cell.querySelector(".absolute");
				result.push({
					color: style || null,
					hasIcon: icon !== null,
					row: rowIdx,
					col: colIdx,
				});
				colIdx++;
			}
			rowIdx++;
		}
		return result;
	});

	console.log(`  Found ${cells.length} cells (${cells.filter(c => c.hasIcon).length} with icons)`);

	const grid: number[][] = [];
	const settlements: Settlement[] = [];

	// Build a locator for grid cells: row N, cell M
	const gridContainer = page.locator('.border.rounded-lg.overflow-hidden[style*="680px"]');

	for (let i = 0; i < cells.length; i++) {
		const { color, hasIcon, row, col } = cells[i];

		const terrain = colorToTerrain.get(color ?? "");
		if (!terrain) throw new Error(`Unknown color: ${color} at (${col},${row})`);

		if (row >= grid.length) grid.push([]);
		grid[row].push(terrain.code);

		if (hasIcon) {
			// Hover the actual cell element to trigger the panel
			const cellLocator = gridContainer.locator(":scope > .flex").nth(row).locator("> div").nth(col);
			const box = await cellLocator.boundingBox();
			if (box) {
				await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
				try {
					await page.locator("h3:text-matches('Settlement|Port|Ruin')").waitFor({ timeout: 2000 });
				} catch {
					console.log(`  Warning: panel didn't appear for (${col},${row})`);
				}

				const panel = await readSettlementPanel(page);
				settlements.push({
					x: col,
					y: row,
					has_port: panel.port ?? false,
					alive: terrain.code !== 3,
					...panel.population !== undefined && { population: panel.population },
					...panel.food !== undefined && { food: panel.food },
					...panel.wealth !== undefined && { wealth: panel.wealth },
					...panel.defense !== undefined && { defense: panel.defense },
					...panel.faction !== undefined && { faction: panel.faction },
				});
			}
		}
	}

	return { grid, settlements };
}

async function readSettlementPanel(page: Page): Promise<Partial<Omit<Settlement, "x" | "y" | "has_port" | "alive"> & { port?: boolean }>> {
	return await page.evaluate(() => {
		// Find the settlement heading like "Settlement (35, 3)"
		const heading = Array.from(document.querySelectorAll("h3")).find((h) =>
			h.textContent?.match(/Settlement|Port|Ruin/)
		);
		if (!heading) return {};

		// The stats are in the next sibling div with .flex.justify-between rows
		const statsContainer = heading.nextElementSibling;
		if (!statsContainer) return {};

		const rows = statsContainer.querySelectorAll(".flex.justify-between");
		const data: Record<string, string> = {};
		for (const row of rows) {
			const spans = row.querySelectorAll("span");
			if (spans.length >= 2) {
				const label = spans[0].textContent?.trim() ?? "";
				const value = spans[spans.length - 1].textContent?.trim() ?? "";
				data[label] = value;
			}
		}

		return {
			population: data["Population"] ? parseFloat(data["Population"]) : undefined,
			food: data["Food"] ? parseFloat(data["Food"]) : undefined,
			wealth: data["Wealth"] ? parseFloat(data["Wealth"]) : undefined,
			defense: data["Defense"] ? parseFloat(data["Defense"]) : undefined,
			port: data["Port"] ? data["Port"].toLowerCase() === "yes" : undefined,
			faction: data["Faction"] ? parseInt(data["Faction"], 10) : undefined,
		};
	});
}
