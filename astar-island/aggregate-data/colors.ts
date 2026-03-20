export const colorToTerrain = new Map<
	string,
	{ code: number; terrain: string; classIndex: number; description: string }
>([
	[
		"rgb(30, 58, 95)",
		{
			code: 10,
			terrain: "Ocean",
			classIndex: 0,
			description: "Impassable water, borders the map",
		},
	],
	[
		"rgb(200, 184, 138)",
		{
			code: 11,
			terrain: "Plains",
			classIndex: 0,
			description: "Flat land, buildable",
		},
	],
	[
		"rgb(212, 118, 10)",
		{
			code: 1,
			terrain: "Settlement",
			classIndex: 1,
			description: "Active Norse settlement",
		},
	],
	[
		"rgb(14, 116, 144)",
		{
			code: 2,
			terrain: "Port",
			classIndex: 2,
			description: "Coastal settlement with harbour",
		},
	],
	[
		"rgb(127, 29, 29)",
		{
			code: 3,
			terrain: "Ruin",
			classIndex: 3,
			description: "Collapsed settlement",
		},
	],
	[
		"rgb(45, 90, 39)",
		{
			code: 4,
			terrain: "Forest",
			classIndex: 4,
			description: "Provides food to adjacent settlements",
		},
	],
	[
		"rgb(107, 114, 128)",
		{
			code: 5,
			terrain: "Mountain",
			classIndex: 5,
			description: "Impassable terrain",
		},
	],
]);

// Terrain name → RGB color string
export const terrainToColor = new Map<string, string>([
	["Ocean", "rgb(30, 58, 95)"],
	["Plains", "rgb(200, 184, 138)"],
	["Settlement", "rgb(212, 118, 10)"],
	["Port", "rgb(14, 116, 144)"],
	["Ruin", "rgb(127, 29, 29)"],
	["Forest", "rgb(45, 90, 39)"],
	["Mountain", "rgb(107, 114, 128)"],
]);
