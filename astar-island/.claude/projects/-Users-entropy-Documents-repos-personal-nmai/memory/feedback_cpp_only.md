---
name: cpp-only-for-ml
description: User strongly prefers C++ for all computation — no Python, no TypeScript for ML/model code
type: feedback
---

All model training, inference, and data processing must be in C++. Never suggest Python or TypeScript for ML tasks.

**Why:** User has a clear preference for C++ as the compute language. TypeScript/Deno is only for API interaction and JSON conversion.

**How to apply:** When building models, training loops, or any computational code, always use C++. The Deno layer is only for fetching data from the API and converting JSON → bin.
