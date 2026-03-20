import { defineConfig } from "@playwright/test";

export default defineConfig({
	testMatch: "scraper.ts",
	use: {
		headless: false,
		actionTimeout: 0,
		navigationTimeout: 0,
	},
	timeout: 0,
});
