import { Hono } from "hono";
import { generateText, tool } from "ai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { z } from "zod";

type Bindings = {
	GOOGLE_GENERATIVE_AI_API_KEY: string;
	API_KEY?: string;
};

const app = new Hono<{ Bindings: Bindings }>();

const AGENT_TIMEOUT_MS = 270_000; // 4.5 minutes — leave buffer for the 5 min competition timeout
const MAX_API_CALLS = 20; // Circuit breaker — cap total Tripletex API calls to avoid subrequest limits

const SYSTEM_PROMPT = `You are an AI accounting agent for Tripletex. You receive a task prompt describing an accounting operation and must complete it using the Tripletex REST API via the provided tools.

Authentication is already handled by the tools — just call them.

## API conventions
- GET list responses: {"fullResultSize": N, "values": [...]}
- POST/PUT responses: {"value": {...}}
- Use ?fields=* to see all fields, or ?fields=id,name for specific ones
- Pagination: ?count=100&from=0
- DELETE uses ID in path: DELETE /employee/123
- Object references use {"id": N} format, e.g. "customer": {"id": 123}
- Action endpoints (/:payment, /:createCreditNote, /:invoice) use **PUT** not POST

## Entity schemas — exact field names for POST

### POST /employee
{"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org", "dateOfBirth": "1990-01-15", "userType": "EXTENDED", "department": {"id": N}}
- userType is REQUIRED — must be "STANDARD", "EXTENDED", or "NO_ACCESS". Cannot be empty.
- department is REQUIRED — first GET /department to find existing departments, or use the default one
- To make admin/kontoadministrator: create with userType "EXTENDED", then:
  PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId={id}&template=ALL_PRIVILEGES (no body needed, use empty {})
- Available templates: NONE_PRIVILEGES, ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER

### POST /customer
{"name": "Acme AS", "email": "post@acme.no", "isCustomer": true}
- Optional: organizationNumber, phoneNumber, invoiceEmail, language, isSupplier, isPrivateIndividual
- Address: postalAddress, physicalAddress as {"addressLine1": "...", "postalCode": "...", "city": "..."}

### POST /product
{"name": "Widget", "priceExcludingVatCurrency": 100.0, "vatType": {"id": N}}
- Optional: number, description, costExcludingVatCurrency, productUnit ({"id": N}), isStockItem
- Get VAT types: GET /ledger/vatType?fields=id,name,number,percentage

### POST /order (required before creating invoices)
{"customer": {"id": N}, "orderDate": "2026-03-22", "deliveryDate": "2026-03-22"}
- Optional: receiverEmail, reference, ourContactEmployee ({"id": N}), department, project, invoiceComment
- You can embed orderLines directly: "orderLines": [{"description": "...", "count": 1, "unitPriceExcludingVatCurrency": 100, "vatType": {"id": N}}]

### POST /order/orderline (add lines to existing order — NOT /orderline, that does not exist!)
{"order": {"id": N}, "description": "Service", "count": 1, "unitPriceExcludingVatCurrency": 500.0, "vatType": {"id": N}}
- For project billing with hours: use "count": HOURS and "unitPriceExcludingVatCurrency": HOURLY_RATE

### POST /project/orderline (add order line linked to a project)
{"project": {"id": N}, "description": "Consulting", "count": 30, "unitPriceExcludingVatCurrency": 1550, "vatType": {"id": N}}

### POST /timesheet/entry (register hours worked)
{"employee": {"id": N}, "project": {"id": N}, "activity": {"id": N}, "date": "2026-03-22", "hours": 7.5}
- First GET /activity?fields=id,name to find the activity ID by name
- For project billing tasks: register timesheet entries, then create project invoice

### POST /timesheet/entry/list (register multiple timesheet entries)
Array of timesheet entry objects. Use when registering many hours across multiple days.

### POST /invoice
{"invoiceDate": "2026-03-22", "invoiceDueDate": "2026-04-22", "orders": [{"id": N}]}
- The order must have order lines before invoicing

### Shortcut: PUT /order/{id}/:invoice (create invoice directly from order)
Query params: invoiceDate (required), sendToCustomer (optional bool), paymentTypeId (optional), paidAmount (optional)
This is simpler than POST /invoice when you have one order.

### PUT /invoice/{id}/:payment (register payment — THIS IS PUT, NOT POST)
Query params: paymentDate (required), paymentTypeId (required), paidAmount (required), paidAmountCurrency (optional)
- paidAmount = amount in NOK (the payment type's account currency)
- paidAmountCurrency = amount in the INVOICE's currency (e.g. EUR). Required when invoice is in foreign currency. This is an AMOUNT, NOT a currency ID!
- Get payment types: GET /invoice/paymentType?fields=id,description (NOT /ledger/paymentType — that endpoint does NOT exist!)
- Use empty body {}
- Only call this ONCE per payment — do not retry if it returns 200

### PUT /invoice/{id}/:createCreditNote (THIS IS PUT, NOT POST)
Query params: date (required), comment (optional), sendToCustomer (optional bool)
- Use empty body {}

### POST /project
{"name": "Project X", "projectManager": {"id": N}, "startDate": "2026-03-22"}
- Optional: customer ({"id": N}), department, description, endDate, isInternal, isFixedPrice

### POST /department
{"name": "Sales", "departmentNumber": "100"}

### POST /contact (contact person on a customer)
{"firstName": "Per", "lastName": "Hansen", "email": "per@acme.no", "customer": {"id": N}}
- Optional: phoneNumberMobile, phoneNumberWork

### POST /travelExpense
{"employee": {"id": N}, "title": "Business trip", "date": "2026-03-22", "travelDetails": {"departureDate": "2026-03-20", "returnDate": "2026-03-22", "departureFrom": "Oslo", "destination": "Bergen", "purpose": "Client meeting"}}
- Optional: project ({"id": N}), department ({"id": N})
- DELETE /travelExpense/{id} to remove
- PUT /travelExpense/{id} to update

### POST /travelExpense/cost (add expense/receipt to a travel expense)
{"travelExpense": {"id": N}, "date": "2026-03-21", "amountCurrencyIncVat": 450.0, "comments": "Dinner", "isPaidByEmployee": true, "isChargeable": false, "paymentType": {"id": N}}
- paymentType is REQUIRED — get it from GET /travelExpense/paymentType?fields=id,description

### POST /travelExpense/mileageAllowance (add mileage to travel expense)
{"travelExpense": {"id": N}, "date": "2026-03-21", "departureLocation": "Oslo", "destination": "Bergen", "km": 460, "isCompanyCar": false}

### POST /travelExpense/perDiemCompensation (per diem / diet allowance)
{"travelExpense": {"id": N}, "count": 2, "overnightAccommodation": "HOTEL", "location": "Bergen"}
- overnightAccommodation: "HOTEL", "NONE", "BOARDING_HOUSE_WITHOUT_COOKING", "BOARDING_HOUSE_WITH_COOKING"

### PUT /employee/{id} (update employee)
Include the full object with id and version from the GET response, plus your changes.

### PUT /customer/{id} (update customer)
Include the full object with id and version from the GET response, plus your changes.

### POST /bank/statement/import (import bank statement from CSV)
This is a multipart form upload. Use the file data from the task's attached files.

### POST /bank/reconciliation (create bank reconciliation)
{"account": {"id": N}, "date": "2026-03-22"}

### PUT /bank/reconciliation/match/:suggest (auto-suggest matches)
Query param: bankReconciliationId (required). Use empty body {}.

### POST /ledger/voucher (journal entries / accounting postings)
This is how you create ANY accounting entry — depreciation, tax provisions, prepaid expenses, corrections, etc.
A voucher contains postings (debit/credit lines). Postings MUST balance (sum to zero).
CRITICAL: Each posting MUST have "row" >= 1. Row 0 is system-reserved and will cause a 422 error.
{
  "date": "2025-12-31",
  "description": "Depreciation - Inventar",
  "postings": [
    {"row": 1, "account": {"id": DEBIT_ACCOUNT_ID}, "amount": 20000, "description": "Depreciation charge"},
    {"row": 2, "account": {"id": CREDIT_ACCOUNT_ID}, "amount": -20000, "description": "Accumulated depreciation"}
  ]
}
- First GET /ledger/account?fields=id,number,name to find account IDs by account number
- Debit = positive amount, Credit = negative amount
- Each voucher's postings MUST sum to zero
- Use ?sendToLedger=true (default) to post immediately
- For depreciation: debit expense account (e.g. 6010), credit accumulated depreciation account (e.g. 1209)
- For tax provision: debit tax expense (8700), credit tax payable (2920)
- For prepaid expense reversal: debit expense account, credit prepaid account (1700)
- Create separate vouchers when the task says "separate vouchers" or "separate entries"

### PUT /ledger/voucher/{id}/:reverse (reverse a voucher)
Query param: date (REQUIRED) — the date for the reversal voucher. Use empty body {}.
Example: PUT /ledger/voucher/123/:reverse?date=2026-03-22 with body {}

### GET /ledger/account (look up account IDs by account number)
GET /ledger/account?number=6010&fields=id,number,name
- You MUST look up account IDs before creating vouchers — use the account number from the task prompt
- Use exact number search: GET /ledger/account?number=6010&fields=id,number,name (returns exact match)
- WARNING: numberFrom/numberTo returns ALL accounts in range — avoid using it, use exact ?number= instead
- Some accounts (like 1500 Kundefordringer) require a "customer" reference on the posting

### GET /ledger/voucherType
GET /ledger/voucherType?fields=id,name — list available voucher types

## Common GET endpoints
- GET /employee?fields=id,firstName,lastName,email
- GET /customer?fields=id,name,email&name=SearchTerm
- GET /product?fields=id,name,number
- GET /ledger/vatType?fields=id,name,number,percentage
- GET /invoice/paymentType?fields=id,description (for invoice payments — NOT /ledger/paymentType which does not exist)
- GET /ledger/account?fields=id,number,name&number=6010
- GET /currency?fields=id,code
- GET /travelExpense?fields=id,title,date,employee(*)
- GET /department?fields=id,name,departmentNumber (needed before creating employees!)
- GET /travelExpense/paymentType?fields=id,description (needed for travel expense costs)

## REQUIRED date ranges — these GET endpoints WILL FAIL without both date params:
- GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,invoiceNumber,invoiceDueDate,amount,amountOutstanding,customer(*)
- GET /order?orderDateFrom=2020-01-01&orderDateTo=2030-12-31&fields=id,number,customer(*)
- GET /ledger/voucher?dateFrom=2020-01-01&dateTo=2030-12-31&fields=id,number,date,description,postings(*)
- GET /ledger/posting?dateFrom=2020-01-01&dateTo=2030-12-31&fields=id,date,description,amount,account(*)
Always use wide date ranges (2020-01-01 to 2030-12-31) to catch all data.

## Correction / reversal tasks
- To reverse an invoice: PUT /invoice/{id}/:createCreditNote
- To reverse a voucher: PUT /ledger/voucher/{id}/:reverse?date=2026-03-22 (use empty body {})
- To delete a travel expense: DELETE /travelExpense/{id}
- To find existing entities to correct: GET with search params, e.g. GET /invoice?invoiceNumber=10001
- To find existing vouchers: GET /ledger/voucher?dateFrom=2026-01-01&dateTo=2026-12-31&fields=id,number,description,date,postings(*)

## Processing attached files
- PDF/image files are provided as base64 in the prompt. Read the content to extract data like invoice numbers, amounts, dates, product names, etc.
- For CSV files (bank statements): the data is base64-encoded, decode and parse to extract transactions
- Use extracted data to create the appropriate entities in Tripletex

## Norwegian accounting rules and common accounts
- Account 8060 = Valutagevinst (agio) — use when exchange rate FAVORS you (received more NOK than expected)
- Account 8070 = Valutatap (disagio) — use when exchange rate is AGAINST you (received less NOK than expected)
- Currency difference example: Invoice 10000 EUR at 11.00 NOK/EUR = 110,000 NOK booked. Customer pays at 10.50 = 105,000 NOK received. Disagio = 5,000 NOK. Debit 8070, credit 1500.
- For foreign currency payments: paidAmount = NOK amount received, paidAmountCurrency = original invoice currency amount (e.g. EUR amount)
- Account 1500 = Kundefordringer (Accounts receivable) — always needs customer reference in voucher postings
- Account 2400 = Leverandørgjeld (Accounts payable) — needs supplier reference
- Account 2920 = Betalbar skatt (Tax payable)
- Account 8700 = Skattekostnad (Tax expense)
- Depreciation: Debit expense (6010), credit accumulated depreciation (e.g. 1209)
- Prepaid expenses: Account 17xx, reverse by debiting the expense account

## Important rules
- Prompts come in 7 languages (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French)
- "kontoadministrator" / "administrator" / "admin" = ALL_PRIVILEGES template with userType EXTENDED
- Minimize API calls — fewer = better efficiency score
- Avoid 4xx errors — each reduces your efficiency bonus
- After POST, you already have the ID from the response — don't GET it again
- Read error messages carefully and fix in one retry
- Plan all needed calls before starting
- You CAN do any accounting task — use /ledger/voucher for journal entries, /bank/* for bank reconciliation, etc.
- NEVER give up — every task can be completed with the available tools
- Each submission uses a FRESH sandbox account — any existing data was pre-created for the task, so search with wide date ranges (e.g. 2020-01-01 to 2030-12-31)
- Today's date is 2026-03-22 — use this for any "today" references

## Workflow
1. Read and understand the task prompt (handle any language)
2. Identify entities to create/modify/delete
3. Plan minimal API call sequence
4. Execute using the tools
5. Stop when done — no unnecessary verification calls`;

interface TripletexFile {
	filename: string;
	content_base64: string;
	mime_type: string;
}

interface SolveRequest {
	prompt: string;
	files?: TripletexFile[];
	tripletex_credentials: {
		base_url: string;
		session_token: string;
	};
}

interface TraceEntry {
	step: number;
	tool: string;
	args: Record<string, unknown>;
	status: number;
	response: unknown;
	durationMs: number;
}

interface SolveTrace {
	prompt: string;
	files: string[];
	startedAt: string;
	entries: TraceEntry[];
	modelSteps: { text?: string; toolCalls: string[] }[];
	finishReason?: string;
	error?: string;
	durationMs?: number;
}

let currentTrace: SolveTrace | null = null;
let apiCallCount = 0;

function logApiCall(method: string, path: string, status: number, args: Record<string, unknown>, response: unknown, durationMs: number) {
	const level = status >= 400 ? "WARN" : "INFO";
	const respPreview = JSON.stringify(response).slice(0, 500);
	console.log(`[${level}] ${method} ${path} -> ${status} (${durationMs}ms) ${respPreview}`);

	if (currentTrace) {
		currentTrace.entries.push({
			step: currentTrace.entries.length + 1,
			tool: `${method} ${path}`,
			args,
			status,
			response,
			durationMs,
		});
	}
}

function checkBudget() {
	if (apiCallCount >= MAX_API_CALLS) {
		console.warn(`[BUDGET] Hit ${MAX_API_CALLS} API call limit — refusing further calls`);
		return { error: `API call budget exhausted (${MAX_API_CALLS} calls). Stop making calls and let the task complete.`, budgetExhausted: true };
	}
	apiCallCount++;
	return null;
}

export function buildTools(baseUrl: string, authHeader: string) {
	return {
		tripletex_get: tool({
			description:
				"Make a GET request to the Tripletex API. Use for listing/searching entities.",
			parameters: z.object({
				path: z
					.string()
					.describe("API path, e.g. /employee or /customer"),
				params: z
					.record(z.string())
					.optional()
					.describe(
						'Query parameters as key-value pairs, e.g. {"fields": "id,name", "count": "100"}'
					),
			}),
			execute: async ({ path, params }) => {
				const budgetError = checkBudget();
				if (budgetError) return budgetError;
				const t0 = Date.now();
				const url = new URL(`${baseUrl}${path}`);
				if (params) {
					for (const [k, v] of Object.entries(params)) {
						url.searchParams.set(k, v);
					}
				}
				const resp = await fetch(url.toString(), {
					headers: { Authorization: authHeader },
				});
				const data = await resp.json();
				logApiCall("GET", path + (params ? "?" + new URLSearchParams(params).toString() : ""), resp.status, { params }, data, Date.now() - t0);
				return data;
			},
		}),

		tripletex_post: tool({
			description:
				"Make a POST request to the Tripletex API. Use for creating new entities.",
			parameters: z.object({
				path: z.string().describe("API path, e.g. /employee, /order"),
				body: z
					.record(z.unknown())
					.describe("JSON body to send in the request"),
			}),
			execute: async ({ path, body }) => {
				const budgetError = checkBudget();
				if (budgetError) return budgetError;
				const t0 = Date.now();
				const resp = await fetch(`${baseUrl}${path}`, {
					method: "POST",
					headers: {
						Authorization: authHeader,
						"Content-Type": "application/json",
					},
					body: JSON.stringify(body),
				});
				const data = await resp.json();
				logApiCall("POST", path, resp.status, { body }, data, Date.now() - t0);
				return data;
			},
		}),

		tripletex_put: tool({
			description:
				"Make a PUT request to the Tripletex API. Use for updating entities AND for action endpoints like /invoice/123/:payment, /invoice/123/:createCreditNote, /order/123/:invoice, /employee/entitlement/:grantEntitlementsByTemplate. For action endpoints, pass parameters as query params in the path (e.g. /invoice/123/:payment?paymentDate=2026-03-22&paymentTypeId=1&paidAmount=1000) and use empty body {}.",
			parameters: z.object({
				path: z
					.string()
					.describe("API path, e.g. /employee/123 or /invoice/123/:payment?paymentDate=2026-03-22&paymentTypeId=1&paidAmount=1000"),
				body: z
					.record(z.unknown())
					.describe("JSON body for the request. Use empty {} for action endpoints where params are in the URL."),
			}),
			execute: async ({ path, body }) => {
				const budgetError = checkBudget();
				if (budgetError) return budgetError;
				const t0 = Date.now();
				const resp = await fetch(`${baseUrl}${path}`, {
					method: "PUT",
					headers: {
						Authorization: authHeader,
						"Content-Type": "application/json",
					},
					body: JSON.stringify(body),
				});
				const text = await resp.text();
				let data: unknown;
				if (!text) {
					data = { success: true, status: resp.status };
				} else {
					try { data = JSON.parse(text); } catch { data = { rawResponse: text, status: resp.status }; }
				}
				logApiCall("PUT", path, resp.status, { body }, data, Date.now() - t0);
				return data;
			},
		}),

		tripletex_delete: tool({
			description:
				"Make a DELETE request to the Tripletex API. Use for removing entities.",
			parameters: z.object({
				path: z
					.string()
					.describe("API path with ID, e.g. /travelExpense/123"),
			}),
			execute: async ({ path }) => {
				const budgetError = checkBudget();
				if (budgetError) return budgetError;
				const t0 = Date.now();
				const resp = await fetch(`${baseUrl}${path}`, {
					method: "DELETE",
					headers: { Authorization: authHeader },
				});
				let data: unknown;
				if (resp.status === 204) {
					data = { success: true };
				} else {
					data = await resp.json();
				}
				logApiCall("DELETE", path, resp.status, {}, data, Date.now() - t0);
				return data;
			},
		}),
	};
}

app.post("/solve", async (c) => {
	// Verify API key if configured
	const expectedKey = c.env.API_KEY;
	if (expectedKey) {
		const auth = c.req.header("authorization");
		if (auth !== `Bearer ${expectedKey}`) {
			return c.json({ error: "Unauthorized" }, 401);
		}
	}

	const body = await c.req.json<SolveRequest>();
	const { prompt, files = [], tripletex_credentials } = body;
	const { base_url, session_token } = tripletex_credentials;

	const t0 = Date.now();
	apiCallCount = 0;

	// Initialize trace
	const trace: SolveTrace = {
		prompt,
		files: files.map((f) => `${f.filename} (${f.mime_type})`),
		startedAt: new Date().toISOString(),
		entries: [],
		modelSteps: [],
	};
	currentTrace = trace;

	console.log("════════════════════════════════════════════════════════════");
	console.log(`[SOLVE] START ${trace.startedAt}`);
	console.log(`[SOLVE] PROMPT: ${prompt}`);
	if (files.length) console.log(`[SOLVE] FILES: ${trace.files.join(", ")}`);
	console.log("────────────────────────────────────────────────────────────");

	const google = createGoogleGenerativeAI({
		apiKey: c.env.GOOGLE_GENERATIVE_AI_API_KEY,
	});

	const authHeader = `Basic ${btoa(`0:${session_token}`)}`;
	const tools = buildTools(base_url, authHeader);

	// Build multimodal content parts
	const content: Array<
		| { type: "text"; text: string }
		| { type: "image"; image: string; mimeType: string }
		| { type: "file"; data: string; mimeType: string }
	> = [{ type: "text", text: prompt }];

	for (const file of files) {
		if (file.mime_type.startsWith("image/")) {
			content.push({
				type: "image",
				image: file.content_base64,
				mimeType: file.mime_type,
			});
		} else {
			content.push({
				type: "file",
				data: file.content_base64,
				mimeType: file.mime_type,
			});
		}
	}

	const abortController = new AbortController();
	const timeout = setTimeout(() => abortController.abort(), AGENT_TIMEOUT_MS);

	try {
		const result = await generateText({
			model: google("gemini-3.1-pro-preview"),
			system: SYSTEM_PROMPT,
			messages: [{ role: "user", content }],
			tools,
			maxSteps: 15,
			abortSignal: abortController.signal,
			onStepFinish: ({ text, toolCalls, toolResults }) => {
				const stepNum = trace.modelSteps.length + 1;
				const calls = toolCalls.map((tc) => `${tc.toolName}(${JSON.stringify(tc.args).slice(0, 200)})`);

				if (text) console.log(`[STEP ${stepNum}] MODEL: ${text.slice(0, 300)}`);
				for (const tc of toolCalls) {
					console.log(`[STEP ${stepNum}] CALL: ${tc.toolName} ${JSON.stringify(tc.args).slice(0, 400)}`);
				}
				for (const tr of toolResults) {
					const preview = JSON.stringify(tr.result).slice(0, 400);
					console.log(`[STEP ${stepNum}] RESULT(${tr.toolName}): ${preview}`);
				}

				trace.modelSteps.push({ text: text || undefined, toolCalls: calls });
			},
		});

		trace.finishReason = result.finishReason;
		trace.durationMs = Date.now() - t0;

		console.log("────────────────────────────────────────────────────────────");
		console.log(`[SOLVE] DONE in ${trace.durationMs}ms | steps=${result.steps.length} | finishReason=${result.finishReason}`);
		if (result.text) console.log(`[SOLVE] FINAL TEXT: ${result.text.slice(0, 500)}`);
	} catch (err) {
		trace.durationMs = Date.now() - t0;
		if (err instanceof Error && err.name === "AbortError") {
			trace.error = `timeout after ${AGENT_TIMEOUT_MS}ms`;
			console.error(`[SOLVE] ABORTED — hit ${AGENT_TIMEOUT_MS}ms timeout`);
		} else {
			trace.error = err instanceof Error ? err.message : String(err);
			console.error("[SOLVE] ERROR:", err);
		}
	} finally {
		clearTimeout(timeout);

		// Print trace summary
		const errors = trace.entries.filter((e) => e.status >= 400);
		console.log("────────────────────────────────────────────────────────────");
		console.log(`[TRACE] API calls: ${trace.entries.length} | errors: ${errors.length} | model steps: ${trace.modelSteps.length} | total: ${trace.durationMs}ms`);
		for (const e of trace.entries) {
			const marker = e.status >= 400 ? "✗" : "✓";
			console.log(`[TRACE] ${marker} ${e.tool} -> ${e.status} (${e.durationMs}ms)`);
		}
		if (errors.length) {
			console.log("[TRACE] ERRORS:");
			for (const e of errors) {
				console.log(`[TRACE]   ${e.tool}: ${JSON.stringify(e.response).slice(0, 300)}`);
			}
		}
		console.log("════════════════════════════════════════════════════════════");

		lastTrace = { ...trace };
		currentTrace = null;
	}

	return c.json({ status: "completed" });
});

let lastTrace: SolveTrace | null = null;

app.get("/trace", (c) => {
	if (!lastTrace) return c.json({ message: "No traces yet" });
	return c.json(lastTrace);
});

app.get("/", (c) => c.text("Tripletex Agent OK"));

export default app;
