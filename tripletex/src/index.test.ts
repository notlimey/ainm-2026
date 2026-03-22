import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";

vi.mock("ai", async (importOriginal) => {
	const mod = await importOriginal<typeof import("ai")>();
	return { ...mod, generateText: vi.fn() };
});

import { generateText } from "ai";
import app from "./index";
import { buildTools } from "./index";

const mockGenerateText = vi.mocked(generateText);

const ENV = { GOOGLE_GENERATIVE_AI_API_KEY: "test-key" };

const BASE_URL = "https://tx-proxy.ainm.no/v2";
const SESSION_TOKEN = "abc123-test-token";

function solveRequest(body: unknown) {
	return app.request(
		"/solve",
		{
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
		},
		ENV,
	);
}

function validBody(overrides: Record<string, unknown> = {}) {
	return {
		prompt:
			"Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
		tripletex_credentials: {
			base_url: BASE_URL,
			session_token: SESSION_TOKEN,
		},
		...overrides,
	};
}

// ─── Endpoint tests ──────────────────────────────────────────

describe("/solve endpoint", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		mockGenerateText.mockResolvedValue({ text: "", steps: [], finishReason: "stop" });
	});

	it("returns { status: 'completed' } for a valid request", async () => {
		const resp = await solveRequest(validBody());
		expect(resp.status).toBe(200);
		expect(await resp.json()).toEqual({ status: "completed" });
	});

	it("returns { status: 'completed' } even when generateText throws", async () => {
		mockGenerateText.mockRejectedValue(new Error("LLM timeout"));
		const resp = await solveRequest(validBody());
		expect(resp.status).toBe(200);
		expect(await resp.json()).toEqual({ status: "completed" });
	});

	it("passes the prompt as a text content part", async () => {
		const prompt = "Lag en kunde med navn Acme AS";
		await solveRequest(validBody({ prompt }));

		const call = mockGenerateText.mock.calls[0][0];
		const userMessage = call.messages[0];
		expect(userMessage.role).toBe("user");
		expect(userMessage.content[0]).toEqual({ type: "text", text: prompt });
	});

	it("includes a system prompt", async () => {
		await solveRequest(validBody());
		const call = mockGenerateText.mock.calls[0][0];
		expect(call.system).toContain("Tripletex");
	});

	it("provides all four tools to the model", async () => {
		await solveRequest(validBody());
		const call = mockGenerateText.mock.calls[0][0];
		const toolNames = Object.keys(call.tools);
		expect(toolNames).toContain("tripletex_get");
		expect(toolNames).toContain("tripletex_post");
		expect(toolNames).toContain("tripletex_put");
		expect(toolNames).toContain("tripletex_delete");
	});

	it("sets maxSteps for agentic loop", async () => {
		await solveRequest(validBody());
		const call = mockGenerateText.mock.calls[0][0];
		expect(call.maxSteps).toBeGreaterThanOrEqual(10);
	});
});

// ─── File handling tests ─────────────────────────────────────

describe("file handling", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		mockGenerateText.mockResolvedValue({ text: "", steps: [], finishReason: "stop" });
	});

	it("attaches image files as image content parts", async () => {
		await solveRequest(
			validBody({
				files: [
					{
						filename: "receipt.png",
						content_base64: "iVBORw0KGgo=",
						mime_type: "image/png",
					},
				],
			}),
		);

		const content = mockGenerateText.mock.calls[0][0].messages[0].content;
		expect(content).toHaveLength(2); // text + image
		expect(content[1]).toEqual({
			type: "image",
			image: "iVBORw0KGgo=",
			mimeType: "image/png",
		});
	});

	it("attaches PDF files as file content parts", async () => {
		await solveRequest(
			validBody({
				files: [
					{
						filename: "faktura.pdf",
						content_base64: "JVBERi0xLjQg",
						mime_type: "application/pdf",
					},
				],
			}),
		);

		const content = mockGenerateText.mock.calls[0][0].messages[0].content;
		expect(content).toHaveLength(2);
		expect(content[1]).toEqual({
			type: "file",
			data: "JVBERi0xLjQg",
			mimeType: "application/pdf",
		});
	});

	it("handles multiple mixed files", async () => {
		await solveRequest(
			validBody({
				files: [
					{
						filename: "faktura.pdf",
						content_base64: "JVBERi0=",
						mime_type: "application/pdf",
					},
					{
						filename: "photo.jpg",
						content_base64: "/9j/4AAQ",
						mime_type: "image/jpeg",
					},
				],
			}),
		);

		const content = mockGenerateText.mock.calls[0][0].messages[0].content;
		expect(content).toHaveLength(3); // text + pdf + jpg
		expect(content[1].type).toBe("file");
		expect(content[2].type).toBe("image");
	});

	it("works with no files array", async () => {
		await solveRequest({
			prompt: "Test",
			tripletex_credentials: {
				base_url: BASE_URL,
				session_token: SESSION_TOKEN,
			},
		});

		const content = mockGenerateText.mock.calls[0][0].messages[0].content;
		expect(content).toHaveLength(1);
		expect(content[0].type).toBe("text");
	});
});

// ─── Tool tests ──────────────────────────────────────────────

describe("buildTools", () => {
	const authHeader = `Basic ${btoa(`0:${SESSION_TOKEN}`)}`;
	let fetchSpy: Mock;

	beforeEach(() => {
		fetchSpy = vi.fn();
		vi.stubGlobal("fetch", fetchSpy);
	});

	describe("tripletex_get", () => {
		it("calls the correct URL with auth header", async () => {
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify({ values: [] })),
			);
			const tools = buildTools(BASE_URL, authHeader);

			await tools.tripletex_get.execute(
				{ path: "/employee", params: undefined },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(fetchSpy).toHaveBeenCalledWith(
				`${BASE_URL}/employee`,
				expect.objectContaining({
					headers: { Authorization: authHeader },
				}),
			);
		});

		it("appends query params to the URL", async () => {
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify({ values: [] })),
			);
			const tools = buildTools(BASE_URL, authHeader);

			await tools.tripletex_get.execute(
				{
					path: "/customer",
					params: { fields: "id,name", count: "10" },
				},
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			const calledUrl = fetchSpy.mock.calls[0][0];
			expect(calledUrl).toContain("/customer");
			expect(calledUrl).toContain("fields=id%2Cname");
			expect(calledUrl).toContain("count=10");
		});

		it("returns parsed JSON response", async () => {
			const data = {
				fullResultSize: 1,
				values: [{ id: 1, firstName: "Ola" }],
			};
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify(data)),
			);
			const tools = buildTools(BASE_URL, authHeader);

			const result = await tools.tripletex_get.execute(
				{ path: "/employee", params: { fields: "id,firstName" } },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(result).toEqual(data);
		});
	});

	describe("tripletex_post", () => {
		it("sends POST with JSON body and correct headers", async () => {
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify({ value: { id: 42 } })),
			);
			const tools = buildTools(BASE_URL, authHeader);
			const body = {
				firstName: "Ola",
				lastName: "Nordmann",
				email: "ola@example.org",
			};

			await tools.tripletex_post.execute(
				{ path: "/employee", body },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(fetchSpy).toHaveBeenCalledWith(`${BASE_URL}/employee`, {
				method: "POST",
				headers: {
					Authorization: authHeader,
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
			});
		});

		it("returns the created entity", async () => {
			const created = { value: { id: 42, firstName: "Ola" } };
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify(created)),
			);
			const tools = buildTools(BASE_URL, authHeader);

			const result = await tools.tripletex_post.execute(
				{
					path: "/employee",
					body: { firstName: "Ola", lastName: "Nordmann" },
				},
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(result).toEqual(created);
		});
	});

	describe("tripletex_put", () => {
		it("sends PUT with JSON body to path with ID", async () => {
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify({ value: { id: 42 } })),
			);
			const tools = buildTools(BASE_URL, authHeader);
			const body = { id: 42, email: "new@example.org" };

			await tools.tripletex_put.execute(
				{ path: "/employee/42", body },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(fetchSpy).toHaveBeenCalledWith(`${BASE_URL}/employee/42`, {
				method: "PUT",
				headers: {
					Authorization: authHeader,
					"Content-Type": "application/json",
				},
				body: JSON.stringify(body),
			});
		});
	});

	describe("tripletex_delete", () => {
		it("sends DELETE to the correct path", async () => {
			fetchSpy.mockResolvedValue(new Response(null, { status: 204 }));
			const tools = buildTools(BASE_URL, authHeader);

			await tools.tripletex_delete.execute(
				{ path: "/travelExpense/99" },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(fetchSpy).toHaveBeenCalledWith(
				`${BASE_URL}/travelExpense/99`,
				{
					method: "DELETE",
					headers: { Authorization: authHeader },
				},
			);
		});

		it("returns { success: true } on 204", async () => {
			fetchSpy.mockResolvedValue(new Response(null, { status: 204 }));
			const tools = buildTools(BASE_URL, authHeader);

			const result = await tools.tripletex_delete.execute(
				{ path: "/travelExpense/99" },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(result).toEqual({ success: true });
		});

		it("returns JSON body on non-204 response", async () => {
			const errorBody = { error: "Not found" };
			fetchSpy.mockResolvedValue(
				new Response(JSON.stringify(errorBody), { status: 404 }),
			);
			const tools = buildTools(BASE_URL, authHeader);

			const result = await tools.tripletex_delete.execute(
				{ path: "/travelExpense/999" },
				{ toolCallId: "1", messages: [], abortSignal: undefined as any },
			);

			expect(result).toEqual(errorBody);
		});
	});
});

// ─── Auth tests ──────────────────────────────────────────────

describe("authentication", () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it("constructs Basic auth header as base64(0:session_token)", async () => {
		let capturedTools: any;
		mockGenerateText.mockImplementation(async (opts: any) => {
			capturedTools = opts.tools;
			return { text: "", steps: [] };
		});

		const fetchSpy = vi.fn().mockResolvedValue(
			new Response(JSON.stringify({ values: [] })),
		);
		vi.stubGlobal("fetch", fetchSpy);

		await solveRequest(validBody());

		// Invoke a tool to see what auth header it uses
		await capturedTools.tripletex_get.execute(
			{ path: "/employee", params: undefined },
			{ toolCallId: "1", messages: [], abortSignal: undefined as any },
		);

		const expected = `Basic ${btoa(`0:${SESSION_TOKEN}`)}`;
		const calledHeaders = fetchSpy.mock.calls[0][1].headers;
		expect(calledHeaders.Authorization).toBe(expected);
	});
});

// ─── API key auth tests ──────────────────────────────────────

describe("API key auth", () => {
	beforeEach(() => {
		vi.clearAllMocks();
		mockGenerateText.mockResolvedValue({ text: "", steps: [], finishReason: "stop" });
	});

	it("rejects requests with wrong API key", async () => {
		const resp = await app.request(
			"/solve",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: "Bearer wrong-key",
				},
				body: JSON.stringify(validBody()),
			},
			{ ...ENV, API_KEY: "correct-key" },
		);
		expect(resp.status).toBe(401);
	});

	it("accepts requests with correct API key", async () => {
		const resp = await app.request(
			"/solve",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: "Bearer correct-key",
				},
				body: JSON.stringify(validBody()),
			},
			{ ...ENV, API_KEY: "correct-key" },
		);
		expect(resp.status).toBe(200);
	});

	it("skips auth check when API_KEY is not configured", async () => {
		const resp = await solveRequest(validBody());
		expect(resp.status).toBe(200);
	});
});

// ─── Health check ────────────────────────────────────────────

describe("GET /", () => {
	it("returns 200 OK", async () => {
		const resp = await app.request("/");
		expect(resp.status).toBe(200);
		expect(await resp.text()).toBe("Tripletex Agent OK");
	});
});
