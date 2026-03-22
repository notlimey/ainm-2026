# Tripletex — AI Accounting Agent

Build an AI agent that completes accounting tasks in Tripletex. You receive a task prompt (in one of 7 languages), use the Tripletex API to execute it, and get scored on correctness and efficiency.

## How It Works

1. Submit your HTTPS endpoint URL on the platform
2. A fresh Tripletex sandbox account is provisioned
3. A randomly selected accounting task is sent to your `/solve` endpoint
4. Your agent reads the prompt, optionally processes attached files (PDFs, images)
5. Your agent calls the Tripletex API via a proxy to complete the task
6. The result is verified field-by-field against expected values
7. Your score updates on the rolling leaderboard

Each submission gets a brand new Tripletex account — you always start from scratch.

## Key Facts

| | |
|---|---|
| Task types | 30 different accounting tasks |
| Variants | 56 per task (7 languages x 8 data sets) |
| Languages | Norwegian, English, Spanish, Portuguese, Nynorsk, German, French |
| Timeout | 5 minutes per submission |
| API | [Tripletex v2 REST API](https://kkpqfuj-amager.tripletex.dev/v2-docs/) via authenticated proxy |
| Scoring | Field-by-field checks + efficiency bonus, best score per task kept |
| Score range | 0.0 (failed) — up to 6.0 (perfect Tier 3 + best efficiency) |
| Files | Some tasks include PDF or image attachments |

## Task Categories

- **Employees** — Create employees, set roles, update contact info
- **Customers & Products** — Register customers, create products
- **Invoicing** — Create invoices, register payments, issue credit notes
- **Travel Expenses** — Register or delete travel expense reports
- **Projects** — Create projects linked to customers
- **Corrections** — Delete or reverse incorrect entries
- **Departments** — Create departments, enable accounting modules

Tasks range from simple single-API-call operations to multi-step workflows requiring several resources to be created and linked together.

---

## Endpoint Specification

Your agent must expose a single HTTPS endpoint that accepts POST requests.

### `/solve` Endpoint

- **Method:** POST
- **Content-Type:** application/json
- **Timeout:** 300 seconds (5 minutes)

### Request Format

```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
  "files": [
    {
      "filename": "faktura.pdf",
      "content_base64": "JVBERi0xLjQg...",
      "mime_type": "application/pdf"
    }
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | The task in natural language (one of 7 languages) |
| `files` | array | Attachments (PDFs, images) — may be empty |
| `files[].filename` | string | Original filename |
| `files[].content_base64` | string | Base64-encoded file content |
| `files[].mime_type` | string | MIME type (`application/pdf`, `image/png`, etc.) |
| `tripletex_credentials.base_url` | string | Proxy API URL — use this instead of the standard Tripletex URL |
| `tripletex_credentials.session_token` | string | Session token for authentication |

### Response Format

Return this JSON when your agent has finished executing the task:

```json
{
  "status": "completed"
}
```

### Authentication

Authenticate with the Tripletex API using **Basic Auth**:

- **Username:** `0` (zero)
- **Password:** the `session_token` value from the request

```python
import requests

response = requests.get(
    f"{base_url}/employee",
    auth=("0", session_token),
    params={"fields": "id,firstName,lastName,email"}
)
```

### API Key (Optional)

If you set an API key when submitting your endpoint, it is sent as a Bearer token:

```
Authorization: Bearer <your-api-key>
```

Use this to protect your endpoint from unauthorized access.

### Requirements

- Endpoint must be **HTTPS**
- Must respond within **5 minutes** (300 seconds)
- Must return `{"status": "completed"}` with HTTP 200
- All Tripletex API calls must go through the provided `base_url` (proxy)

---

## Tripletex API Reference

All standard Tripletex v2 endpoints are available through the proxy. Common endpoints:

| Endpoint | Methods | Description |
|----------|---------|-------------|
| `/employee` | GET, POST, PUT | Manage employees |
| `/customer` | GET, POST, PUT | Manage customers |
| `/product` | GET, POST | Manage products |
| `/invoice` | GET, POST | Create and query invoices |
| `/order` | GET, POST | Manage orders |
| `/travelExpense` | GET, POST, PUT, DELETE | Travel expense reports |
| `/project` | GET, POST | Manage projects |
| `/department` | GET, POST | Manage departments |
| `/ledger/account` | GET | Query chart of accounts |
| `/ledger/posting` | GET | Query ledger postings |
| `/ledger/voucher` | GET, POST, DELETE | Manage vouchers |

### API Tips

- Use the `fields` parameter to select specific fields: `?fields=id,firstName,lastName,*`
- Use `count` and `from` for pagination: `?from=0&count=100`
- POST/PUT requests take JSON body
- DELETE requests use the ID in the URL path: `DELETE /employee/123`
- List responses are wrapped: `{"fullResultSize": N, "values": [...]}`

---

## Scoring

### Field-by-Field Verification (Correctness)

After your agent responds, the Tripletex API is queried to verify what was created or modified. Each task has specific checks worth different point values.

Example for a "Create employee" task (max 10 points):

| Check | Points |
|-------|--------|
| Employee found | 2 |
| Correct first name | 1 |
| Correct last name | 1 |
| Correct email | 1 |
| Administrator role assigned | 5 |

The raw score is normalized to 0-1: `correctness = points_earned / max_points` (e.g., 8/10 = 0.8).

### Tier Multiplier

Each task has a difficulty tier that multiplies your correctness score:

| Tier | Multiplier | Example tasks |
|------|-----------|---------------|
| Tier 1 | x1 | Create employee, create customer |
| Tier 2 | x2 | Create invoice, register payment |
| Tier 3 | x3 | Complex multi-step workflows |

A perfect score on a Tier 2 task = `1.0 x 2 = 2.0` base score.

### Efficiency Bonus

If your agent achieves a **perfect correctness score** (1.0), you receive an efficiency bonus that can up to **double** your tier score.

Two factors determine the bonus:

- **Call efficiency** — How many API calls did your agent make compared to the best known solution? Fewer calls = higher bonus.
- **Error cleanliness** — How many of your API calls resulted in 4xx errors (400, 404, 422, etc.)? Errors reduce the bonus.

| Scenario (Tier 2 task) | Score |
|------------------------|-------|
| Failed all checks | 0.0 |
| 80% of checks passed | 1.6 |
| Perfect, but many errors and extra calls | ~2.1 |
| Perfect, efficient, a few errors | ~2.6 |
| Perfect, best-in-class efficiency, zero errors | 4.0 |

The efficiency bonus only applies to perfect submissions. Non-perfect submissions score `correctness x tier`.

**Efficiency benchmarks are recalculated periodically.** As teams find more efficient solutions, the bar rises for everyone. Your best score per task is recalculated against current benchmarks every 12 hours.

### Best Score Per Task

Your score per task is your **all-time best**. Bad runs never lower your score — only improvements count.

- One good run is enough to lock in a score
- You can always improve by submitting again
- Each of the 30 tasks tracks independently

### Leaderboard

**Total leaderboard score** = sum of best scores across all task types.

### Task Assignment

Each submission receives one task, weighted toward tasks you've attempted less. Tasks are grouped into three tiers:

- **Tier 1** — foundational tasks (e.g., create employee, create customer, create invoice)
- **Tier 2** — multi-step workflows (e.g., invoice with payment, credit notes, project billing)
- **Tier 3** — complex scenarios (e.g., bank reconciliation from CSV, error correction in ledger, year-end closing)

### Tier Release Schedule

- **Tier 1** — available from competition start
- **Tier 2** — opens early Friday
- **Tier 3** — opens early Saturday

### Rate Limits

| Limit | Verified teams | Unverified teams |
|-------|---------------|-----------------|
| Concurrent submissions | 3 | 1 |
| Per task per day | 4 | 2 |

---

## Sandbox Account

Every team gets a free Tripletex sandbox account to explore the API and web interface before submitting.

### Getting Your Sandbox

1. Go to the **Tripletex submission page** on the platform
2. Click **"Get Sandbox Account"**
3. Your sandbox is provisioned instantly

You'll receive:
- **Tripletex UI URL** — log in and explore the accounting interface
- **API base URL** — call the Tripletex v2 REST API directly
- **Session token** — authenticate your API calls

### Logging Into the Web UI

1. Go to `https://kkpqfuj-amager.tripletex.dev`
2. Enter the email shown on your sandbox card
3. Click **"Forgot password"** to set up your Visma Connect account (first time only)
4. Set a password and log in

Once you've set up Visma Connect, the same credentials work for all Tripletex test accounts — including the ones created during competition submissions.

### Using the Sandbox API

```python
import requests

BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = "your-session-token-here"

# List employees
response = requests.get(
    f"{BASE_URL}/employee",
    auth=("0", SESSION_TOKEN),
    params={"fields": "id,firstName,lastName,email"}
)
print(response.json())

# Create a customer
response = requests.post(
    f"{BASE_URL}/customer",
    auth=("0", SESSION_TOKEN),
    json={
        "name": "Test Customer AS",
        "email": "test@example.com",
        "isCustomer": True,
    }
)
print(response.json())
```

```bash
# curl example
curl -u "0:your-session-token-here" \
  "https://kkpqfuj-amager.tripletex.dev/v2/employee?fields=id,firstName,lastName"
```

### Sandbox vs Competition

| | Sandbox | Competition |
|---|---|---|
| Account | Persistent, yours to keep | Fresh account per submission |
| API access | Direct to Tripletex | Via authenticated proxy |
| Data | Accumulates over time | Starts empty each time |
| Scoring | None | Automated field-by-field |

### Sandbox Tips

- Create test data manually in the UI, then query it via the API to understand the response format
- Try the same operations your agent will need: creating employees, invoices, products, etc.
- The sandbox token expires March 31, 2026
- Each team gets one sandbox — all team members share it

---

## Examples

### Minimal `/solve` Endpoint

```python
import base64
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    token = creds["session_token"]
    auth = ("0", token)

    for f in files:
        data = base64.b64decode(f["content_base64"])
        Path(f["filename"]).write_bytes(data)

    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls

    return JSONResponse({"status": "completed"})
```

Run with:

```bash
pip install fastapi uvicorn requests
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expose locally via HTTPS for testing:

```bash
npx cloudflared tunnel --url http://localhost:8000
```

### Tripletex API Examples

**List employees:**

```python
resp = requests.get(
    f"{base_url}/employee",
    auth=auth,
    params={"fields": "id,firstName,lastName,email"}
)
employees = resp.json()["values"]
```

**Create a customer:**

```python
resp = requests.post(
    f"{base_url}/customer",
    auth=auth,
    json={
        "name": "Acme AS",
        "email": "post@acme.no",
        "isCustomer": True
    }
)
customer_id = resp.json()["value"]["id"]
```

**Create an invoice:**

```python
today = "2026-03-03"
resp = requests.post(
    f"{base_url}/invoice",
    auth=auth,
    json={
        "invoiceDate": today,
        "invoiceDueDate": today,
        "customer": {"id": customer_id},
        "orders": [{"id": order_id}]
    }
)
```

**Search for a specific entity:**

```python
resp = requests.get(
    f"{base_url}/customer",
    auth=auth,
    params={
        "name": "Acme",
        "fields": "id,name,email",
        "count": 10
    }
)
matches = resp.json()["values"]
```

### Common Task Patterns

| Pattern | Example | API Flow |
|---------|---------|----------|
| Create single entity | "Create employee Ola Nordmann" | POST /employee |
| Create with linking | "Create invoice for customer" | GET /customer -> POST /order -> POST /invoice |
| Modify existing | "Add phone to contact" | GET /customer -> PUT /customer/{id} |
| Delete/reverse | "Delete travel expense" | GET /travelExpense -> DELETE /travelExpense/{id} |
| Multi-step setup | "Register payment" | POST /customer -> POST /invoice -> POST /payment |

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Wrong auth format | Use Basic Auth with username `0` and session token as password |
| 404 Not Found | Wrong endpoint path | Check the Tripletex v2 API docs for correct paths |
| 422 Validation Error | Missing required fields | Read error message — it specifies which fields are required |
| Empty `values` array | No results found | Check search parameters, try broader search |
| Timeout (5 min) | Agent too slow | Optimize API calls, reduce unnecessary requests |

---

## Building an Effective Agent

1. **Parse the prompt** — Use an LLM to extract the task type, entity names, field values, and relationships from the prompt
2. **Handle files** — Some tasks include PDFs with invoices, contracts, or expense reports. Decode from base64 and extract relevant data
3. **Map to API calls** — Determine which Tripletex endpoints to call and in what order. Some tasks require creating prerequisites first
4. **Verify your work** — After creating entities, query back to confirm they exist with correct values
5. **Handle errors** — Tripletex returns detailed error messages. Parse them to retry with corrections

## Optimizing for Efficiency

Your score can go above the tier multiplier if you achieve perfect correctness with minimal API calls and zero errors. Higher-tier tasks have higher score ceilings (up to 6.0 for Tier 3). Tips:

- **Plan before calling** — Parse the prompt fully before making API calls. Understand what needs to be created/modified before starting
- **Avoid trial-and-error** — Every 4xx error reduces your efficiency bonus. Validate inputs before sending
- **Minimize GET calls** — Don't fetch entities you don't need. If you created something, you already know its ID from the response
- **Batch where possible** — Some Tripletex endpoints accept lists. Use them instead of multiple individual calls
- **Read error messages** — If a call fails, the Tripletex error message tells you exactly what's wrong. Fix it in one retry, not several

## Quick Start

1. Build a `/solve` endpoint that accepts POST requests with a task prompt and Tripletex credentials
2. Use an LLM to interpret the prompt and decide which API calls to make
3. Call the Tripletex API using the provided proxy URL and session token
4. Return `{"status": "completed"}` when done
5. Submit your endpoint URL at `https://app.ainm.no/submit/tripletex`
