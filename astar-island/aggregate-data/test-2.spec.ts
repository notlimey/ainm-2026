import { test } from "@playwright/test";

test("test", async ({ page }) => {
	await page.goto(
		"https://app.ainm.no/submit/astar-island/replay?round=71451d74-be9f-471f-aacd-a41f3b68a9cd",
	);
	await page
		.getByText(
			"EventsLeaderboardTasksRulesPrizesDocsSign inAstar Island ReplayWatch the",
		)
		.click({
			button: "right",
		});
	await page.goto(
		"https://app.ainm.no/submit/astar-island/replay?round=71451d74-be9f-471f-aacd-a41f3b68a9cd",
	);
	await page
		.locator("div")
		.filter({ hasText: "RoundSelect a round...Round 5" })
		.nth(4)
		.click({
			button: "right",
		});
	await page.goto(
		"https://app.ainm.no/submit/astar-island/replay?round=71451d74-be9f-471f-aacd-a41f3b68a9cd",
	);
	await page.getByText("⌂").nth(2).click();
	await page.getByText("⌂").nth(1).click();
	await page.getByRole("button").filter({ hasText: /^$/ }).nth(4).click();
	await page.getByRole("button").filter({ hasText: /^$/ }).nth(4).click();
	await page.getByText("sim_seed:").click();
});
