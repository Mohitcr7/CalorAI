# CalorAI — Engineering Report

A detailed account of what was built, how each requirement was satisfied, how it
was verified, and what it would take to run this at scale.

This is the companion to [README.md](README.md). The README is for someone who
wants to run the thing; this is for someone who wants to evaluate the
engineering behind it.

---

## Contents

- [Executive summary](#executive-summary)
- [Core feature 1 — Conversational agent with tool calling](#core-feature-1--conversational-agent-with-tool-calling)
- [Core feature 2 — Persistent database](#core-feature-2--persistent-database)
- [Core feature 3 — Running daily totals](#core-feature-3--running-daily-totals)
- [Core feature 4 — Image input on a separate model](#core-feature-4--image-input-on-a-separate-model)
- [Core feature 5 — Persistent memory](#core-feature-5--persistent-memory)
- [Core feature 6 — Multi-turn ambiguity handling](#core-feature-6--multi-turn-ambiguity-handling)
- [Evals](#evals)
- [Failover tests](#failover-tests)
- [Clean clone verification](#clean-clone-verification)
- [Latency](#latency)
- [Bonus features](#bonus-features)
- [Bugs found, and how](#bugs-found-and-how)
- [Scaling this up](#scaling-this-up)

---

## Executive summary

| Area | Status | Evidence |
|---|---|---|
| 6 core features | all implemented | eval suite + manual verification below |
| Eval suite | 15 cases, DB-level assertions | 1 failure in 7 full runs |
| Failover tests | 6/6 stub tests, **+ 2/2 live on Gemini** | `tests/test_failover.py` + a real `CALORAI_PROVIDER=google` run |
| Clean clone | verified end to end | fresh clone + venv + `requirements.txt` |
| Text latency | **p50 1.97s / p95 2.80s** | n=20, mixed message shapes |
| Image latency | **p50 6.12s / p95 10.07s** | n=8, Sonnet 5 vision |
| Bonuses | 6 delivered | streaming, isolation, evals, failover, caching, tracing |

**Scale of the code:** 8 application modules, 10 tools, 15 eval cases, 6 failover
tests, 14 commits.

**The two decisions the whole system rests on:**

1. **Daily totals are never stored.** They are `SUM()`'d at read time over
   per-unit macros × quantity. This makes double-counting on a correction
   structurally impossible rather than merely unlikely.
2. **Memory is three different things, not one.** Facts, aliases and episodic
   history have different lifetimes, different retrieval paths and different
   costs. Conflating them is the standard failure mode.

---

## Core feature 1 — Conversational agent with tool calling

> *"Built in LangChain or LangGraph. At minimum the agent needs tools to log a
> meal, retrieve past meals, look up nutrition data, and return current totals.
> How you split up the tool surface is your call and is part of what we're
> evaluating."*

### What was built

A LangGraph `StateGraph` with three nodes:

```
perceive ──→ assistant ──→ tools ──┐
                  ▲                │
                  └────────────────┘
```

`perceive` is the image branch — it returns immediately on text-only turns, so it
costs nothing on the common path. `assistant` and `tools` form a standard
tool-calling loop, capped at 4 iterations because a stuck agent is worse than a
short answer.

### The tool split

Ten tools, divided by **side effect** rather than by topic.

| Category | Tools |
|---|---|
| Create | `log_meal` |
| Edit | `correct_item`, `scale_last_meal`, `delete_last_meal` |
| Read | `today_summary`, `meals_on_day`, `lookup_nutrition` |
| Memory | `recall_memory`, `remember`, `save_shortcut` |

**The reasoning.** The model's hardest job is not choosing between "log" and
"look up" — it is not corrupting the day's totals. So every operation that
mutates a logged meal is its own narrow tool whose name states exactly what it
does to the data. `correct_item` *cannot create a meal*; it locates an existing
item and overwrites its quantity.

A single `update_meal` tool with a flexible signature would be one bad model
decision away from logging five rotis instead of correcting two to three. No
amount of prompt wording reliably prevents that. **Making the wrong action
unrepresentable is cheaper than making it unlikely.**

Reads are separate and cheap so the model can inspect state before mutating it.
Memory gets two write paths for a deliberate reason — see feature 5.

### Security detail

`user_id` is **not** a tool argument. It comes from a `ContextVar` set once per
turn, so the model cannot address another user's data even if it constructs a
call that tries. This is verified by the `session_isolation` eval.

**Files:** [`agent.py`](calorai/agent.py), [`tools.py`](calorai/tools.py)

---

## Core feature 2 — Persistent database

> *"Postgres, SQLite, or Supabase. Schema is your call. Meals must persist
> across turns and across sessions."*

SQLite, six tables:

| Table | Purpose |
|---|---|
| `meals` | one row per logged meal; soft-delete flag |
| `meal_items` | per-unit macros + quantity, stored separately |
| `foods` | nutrition cache — seeded, plus LLM results written back |
| `memories` | facts and aliases, `UNIQUE(user_id, key)` |
| `memory_history` | audit trail for overwritten memories |
| `messages` | conversation transcript for cross-process rehydration |

### Verification

Tested across **three separate OS processes** sharing one database file:

```
process 1:  "had 3 rotis and dal for dinner"   → Logged, 492 kcal
process 1:  "i am vegetarian btw"              → Got it, noted.
process 2:  "how am I doing on calories?"      → You're at 492 kcal so far.
process 3:  "can I have chicken biryani?"      → You had chicken? Thought you
                                                  were vegetarian.
```

Process 2 proves **meals** persist. Process 3 proves **memory** persists *and is
actually used in reasoning* — which is a stronger claim than the data merely
being on disk.

Conversation history is rehydrated from `messages` (last 8) at the start of every
turn, which is why a fresh process continues mid-thread rather than starting
cold.

**Files:** [`db.py`](calorai/db.py)

---

## Core feature 3 — Running daily totals

> *"Calories and macros for the current day, updated as meals come in, correctly
> reflecting edits and deletions."*

### The invariant

> **Daily totals are never stored. They are always `SUM()`'d at read time.**

`meal_items` stores macros **per one unit** and **quantity** as separate columns:

```
name=paratha   quantity=2   kcal_per_unit=210
```

`daily_totals()` is `SUM(quantity × kcal_per_unit)` filtered by user, date and
the soft-delete flag.

### Why this makes corrections safe

"actually that was 3 parathas not 2" becomes a single `UPDATE` of `quantity`:
420 → 630. There is nothing to recompute, no stored counter to drift, and no
path by which the original 2 could be added to the new 3. Deletion sets a flag
the same query already filters.

Verified end to end:

```
"had 2 parathas and chai"           → 525 kcal   (DB: 525)
"actually that was 3 parathas not 2" → 735 kcal   (DB: 735)
photo + "half of this was my brother's" → +437    (DB: 1172)
"how am I doing on calories?"       → 1172 kcal  (DB: 1172)
```

735 + 437 = 1172 exactly, because that number is a database `SUM()` rather than
anything the model computed.

### The bug that made this harder than it looks

The database was correct from the first run. **The product still wasn't.**

On the first live test the agent told the user **1050 kcal** while SQLite
correctly held **525**. The day's total appeared *twice* in the model's context —
once in the tool result, once in the rebuilt system prompt — and the model added
them together.

The first fix made it worse. Removing the total from the tool result caused the
model to invent **786** against a stored **735**. Small models weight what is
adjacent to the generation point far more than what sits at the top of a
2600-token prompt.

The fix that worked has three parts:

1. The total is stated **once**, in the tool result, adjacent to generation.
2. It is worded to explicitly supersede the copy in the prompt.
3. The prompt forbids arithmetic outright: *"Never add, subtract or compute a
   total yourself. Every number you say must be copied from somewhere you were
   given it."*

This is the most instructive bug in the project: a correct database is necessary
but not sufficient, because the user reads the sentence, not the schema.

---

## Core feature 4 — Image input on a separate model

> *"Route images to a vision model of your choice and text to whatever you're
> using for conversation — do not run everything through one model. Document why
> you picked each and how you handle the handoff, including what happens when the
> vision model returns something ambiguous or wrong."*

### The split

| Path | Model | Config |
|---|---|---|
| text + tools | `claude-haiku-4-5` | 10s timeout, `max_retries=0`, temperature 0.2 |
| vision | `claude-sonnet-5` | 15s timeout, `max_retries=1`, no temperature |

Different model families, different configs, different timeouts, different retry
policies. Not a nominal split.

**Why Sonnet for vision:** recognising a half-eaten paratha under kitchen
lighting is a genuinely harder perceptual task than routing "had 2 rotis", and it
runs at most once per turn so the cost is bounded. Measured on the sample plate,
Sonnet identifies *bajra roti* and *laddu* correctly where Haiku guessed *yogurt*
and *pakora*.

### The handoff contract

**The vision model identifies food and portion only. It is never asked for
calories.**

Two reasons:

1. A meal photographed and a meal typed produce **byte-identical numbers**,
   because both resolve through the same nutrition table. A user's daily total
   does not depend on how they happened to report the meal.
2. Vision models are much better at "that's a paratha" than at "that's 210 kcal".
   Asking for the second invites confident, unverifiable numbers into the
   database.

Vision does not act. It writes a `[VISION]` note into the system prompt; the text
model does all the reasoning and makes exactly one tool call.

### Ambiguous or wrong vision output

`VisionResult.needs_clarification` is true when **any** of these hold:

- an API error occurred
- the image is not food
- no items were identified
- overall confidence is below `VISION_MIN_CONFIDENCE` (0.45)

In every one of those cases the agent **asks instead of logging**. A wrong meal
in the database is worse than one extra question.

### Photo + caption must resolve to one meal

The note explicitly disambiguates:

> *The user's message this turn is a caption on THIS photo. It is not a
> correction to anything logged earlier, even if it is worded like one. 'half of
> this was my brother's' means: multiply the quantities above and log the result
> ONCE. Do NOT call scale_last_meal — there is nothing earlier to correct.*

Two independent passes would produce two meals, or a meal plus a correction —
either way a wrong total and two replies to one message.

### A calibration finding

Sonnet reports confidence **0.50–0.55 while being correct**; Haiku reported
**0.79–0.82 while being wrong**. Haiku is confidently incorrect; Sonnet is
accurately unsure.

This means the single `VISION_MIN_CONFIDENCE` threshold is weaker than it looks —
the same number means different things on different models. It catches
catastrophic failures but is close to meaningless as a quality signal. Documented
as a known weakness rather than presented as a working safeguard.

**Files:** [`vision.py`](calorai/vision.py)

---

## Core feature 5 — Persistent memory

> *"The agent should remember things worth remembering across sessions... You
> decide what's worth storing, how it gets written, and how it gets retrieved
> into context without bloating every prompt. **This is the part we're most
> interested in.**"*

### Three kinds of memory, deliberately separated

| Kind | Example | Stored in | Retrieved by | Prompt cost |
|---|---|---|---|---|
| **Fact** | "Is vegetarian" | `memories`, kind=fact | injected every prompt | ~150–250 tok, capped |
| **Alias** | "my usual" → 2 parathas + chai | `memories`, kind=alias + JSON | on demand via `recall_memory` | name only (~5 tok) |
| **Episodic** | what they ate Tuesday | `meals` table | SQL query | 0 |

**Episodic history is not memory.** "same as yesterday" is
`SELECT ... WHERE meal_date = today-1`. Treating it as a recall problem is slower
*and* less correct. Conflating these three is the standard failure mode: dump
every message into a vector store and you get a memory system that is enormous,
slow, and still wrong about today's calories.

### What gets stored — a bounded key namespace

Twelve keys, and nothing else:

```
diet.restriction   diet.allergy      diet.dislikes
goal.calories      goal.protein_g    goal.weight
profile.name       profile.household
habit.breakfast    habit.schedule    habit.cooking
pref.detail
```

Plus `alias.<phrase>` for shortcuts.

**This is the mechanism that stops memory growing without bound.** Writes are
upserts on `(user_id, key)`, so the tenth time a user mentions being vegetarian
it overwrites one row rather than adding a tenth. An unbounded namespace is what
turns a memory store into a junk drawer.

The extractor is explicitly forbidden from storing episodic facts ("had 2 rotis
today") or transient states ("I'm full"). A confidence floor of 0.55 applies —
a guess is not a memory. Overwrites are recorded in `memory_history`, so a
changed memory is explainable rather than silently replaced.

### When it is written — off the critical path

After the reply has streamed, a background thread runs a cheap extraction call
over the turn. **The user never waits for it.** Worth ~300–600ms per turn.

The cost: a fact stated in message N may not be retrievable until N+1. Mitigated
by the `remember` tool, which the model calls synchronously when a user states
something outright — "i'm vegetarian btw" applies *this* turn, not next. The
background extractor catches what is implied rather than declared.

### How it is retrieved — the anti-bloat design

**Tier 1 — always injected.** Facts only, one short line each, hard-capped at 15
and priority-ordered so the cap drops the least useful first. Plus alias
**names**, a few tokens each.

**Tier 2 — on demand.** Alias **payloads** are fetched only when the model calls
`recall_memory`.

> **Names are cheap and go in the prompt. Bodies are expensive and stay in the
> database until asked for.**

A user with 40 saved shortcuts pays the same prompt cost as a user with two. This
is the property that makes the design scale — prompt size is bounded by the
*cap*, not by how long the user has been using the product.

### Verification

- `memory_write_fact` — a durable fact stated in passing reaches the store
- `memory_recall_alias` — "my usual" set in a *previous session* resolves with no
  question asked; nothing in the message says "paratha"
- `memory_teach_alias` — unknown shortcut → ask once → log **and** save
- `veg_awareness` — a stored restriction visibly changes behaviour
- Three-process test — memory survives and is used two processes later

**Files:** [`memory.py`](calorai/memory.py)

---

## Core feature 6 — Multi-turn ambiguity handling

> *"The agent decides when it has enough information to log and when it needs to
> ask. Over-asking kills the experience; under-asking produces garbage data.
> Where you draw that line is a product decision as much as a technical one."*

### The policy, stated explicitly

**Log without asking** when the food is identifiable and the portion inferable.
"some almonds" logs as a handful — being wrong by 30 kcal does not matter, and
asking teaches the user that logging is expensive.

**Ask** only when:
- the food itself is unnamed — "grazed all afternoon" (on *what*?)
- the portion is unknown **and** would swing calories by more than ~40%

**One question maximum**, specific, answerable in three words.

**Never ask permission.** "Want me to log that?" is always wrong — they told you
what they ate, which is the entire reason they messaged.

`"skipped lunch"` logs nothing. A skipped meal is zero calories, not an entry.

### How the policy is delivered

Prose rules plus ~1700 tokens of **worked examples** covering the boundary cases
— vague amounts that should be logged, missing foods that should trigger a
question, corrections that must edit rather than add, captions on photos, and
memory recall before questioning.

Those examples were added for a second reason (crossing the prompt-cache
threshold — see [Latency](#latency)), but they earn their place on behaviour
alone. **One of them immediately fixed a real bug**: after being told what "my
usual" meant, the agent saved the shortcut and then asked *"Want me to log it
now?"* instead of logging. That is the exact over-asking failure the brief warns
about, and it was caught by an eval going red.

### Verification

- `underspecified_asks` — "skipped lunch but grazed all afternoon" must ask and
  must log nothing
- `fractional_portion` — "two thirds of the box" logs as 0.67, not 2 or 3
- `basic_log` — "had 2 parathas and chai" asks nothing at all
- `memory_teach_alias` — asks exactly once, then never again

**Files:** [`prompts.py`](calorai/prompts.py)

---

## Evals

`evals/run_evals.py` — 15 cases.

### The definition of "correct"

Cases are graded on **the state of the database after each turn**, not on the
wording of the reply. A meal logger is correct when the right foods, the right
quantities and the right running total are stored. Reply-text assertions are used
only where the product behaviour *is* the reply — the agent must **ask** rather
than invent food for "grazed all afternoon".

Each case runs against a clean slate under its own user id, so cases cannot
contaminate each other, and multi-turn behaviour is tested as a **sequence**:
log then correct; teach a shortcut then use it.

### Coverage

| Case | What it protects |
|---|---|
| `basic_log` | the common path; must ask nothing |
| `correction_no_double_count` | 3-not-2 updates rather than adds |
| `fractional_portion` | "two thirds" → 0.67 |
| `underspecified_asks` | must ask, must invent nothing |
| `totals_query` | answers from the database |
| `protein_query` | macro-specific queries |
| `memory_write_fact` | durable facts reach the store |
| `memory_recall_alias` | cross-session recall without asking |
| `memory_teach_alias` | ask once, then log **and** save |
| `same_as_yesterday` | episodic recall via SQL |
| `image_only` | vision → log, one meal |
| `image_with_caption` | two models → **one** meal, halved |
| `veg_awareness` | stored restriction changes behaviour |
| `delete_meal` | deletions pull totals back down |
| `session_isolation` | one user's data never reaches another |

### Reliability

**One failure across seven full runs**, on a case that passed on every retry.

Every case drives a real model, so the suite is nondeterministic. This is
reported rather than rounded to a clean 15/15, because a suite that passes 6
times in 7 is a different claim from one that passes always.

The fix is *not* to loosen assertions until everything is green — the assertions
are the point. It is to run borderline cases N times and require a pass rate.
That is on the roadmap.

### An eval failure worth admitting

The image case originally asserted only `meal_count(1)`. It passed green while
the vision model was reporting **"idli, dosa, sambar"** for a plate of roti, curd
and laddu. One meal *was* logged, so the assertion held.

**A weak eval is worse than no eval, because it tells you you're fine.** Image
cases now assert against the photo's actual contents.

---

## Failover tests

`tests/test_failover.py` — **6/6 passing, no API keys required.** Stub runnables
make the routing policy testable deterministically and for free.

### Architecture

Claude primary, Gemini failover, across all three roles (text, vision,
extractor). Either provider alone is a **supported** configuration, not a
degraded one — with only one key configured, the fallback machinery is not
attached at all.

### What fails over, and what deliberately does not

| Failure | Behaviour | Why |
|---|---|---|
| rate limit, 529, timeout, connection, 5xx | **fail over** | transient; a second provider genuinely helps |
| auth, permission, bad request | **raise loudly** | config bug; failing over hides it forever |

That second row is not hypothetical. The Sonnet `temperature` bug (see below) was
a **400**. With a naive fallback, the app would have silently run on Gemini
forever and looked perfectly healthy while the shipped primary was broken.

### Three implementation details that matter

**`max_retries=0` on the text path.** The SDK retries twice with exponential
backoff by default — seconds burned *before* the fallback is even reached. On a
messaging product, a fast answer from the backup beats a slow one from the
primary. The extractor keeps its retries; nobody waits on it.

**Tools bind to each provider before the fallback is composed.**
`with_fallbacks()` returns a plain `Runnable` with no `.bind_tools()`, so binding
after composing fails outright — and binding only the primary would hand the user
a **toolless agent** the moment a failover fired. It would chat happily and log
nothing.

**Streaming fails over only before the first token.**
`RunnableWithFallbacks.stream()` pulls the first chunk inside the try block;
after that a mid-stream error propagates. A user must never see half a Claude
reply followed by a fresh, different Gemini reply. This was confirmed by reading
the library source, and the test pins the behaviour.

### Timeout budgets

Failover is only useful if it is fast. A naive fallback makes the user wait the
full primary timeout *and then* the full failover latency.

```
text:   10s primary + 12s failover = 22s worst case
vision: 15s primary + 20s failover = 35s worst case
```

The 10s figure is measured, not guessed — see [Latency](#latency).

### Verified live

`CALORAI_PROVIDER=google` forces every role onto Gemini with Claude excluded
entirely, confirmed via `llm.have_anthropic()` returning `False` before any
model call — so the following did not fall back to Claude by accident.
`basic_log` and `correction_no_double_count` ran end to end on Gemini alone and
both passed: correct tool calls, and 525 → 735 kcal on the correction, not 1155
— the no-double-count invariant held on a provider that had never actually been
exercised.

Getting a passing run took two fixes, both found by the live key rather than
reasoned about in the abstract — see [Bugs found, and how](#bugs-found-and-how).
The result also carries one honest caveat: these Gemini calls ran **11–15s** end
to end against Claude's sub-3s on the same cases. Failover buys availability, not
comparable speed. The timeout budgets above bound a single HTTP request, and a
full turn is at least two model calls, so total turn time on the failover path
runs well above what those single-call figures alone would suggest.

### What is still not proven

The live test confirms Gemini produces *correct* tool calls against this
system's actual prompt and tool schema. It does not simulate a real Claude
outage triggering the fallback mid-session in production — that path is still
proven by the stub tests only, not by a live failure injection.

---

## Clean clone verification

The brief requires the project to run locally from a clean clone with documented
setup. Verified rather than assumed:

```bash
git clone <repo> cleanclone && cd cleanclone
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Results:

- **Install succeeded** from `requirements.txt` alone, exit 0
- **All imports resolve** — no hidden dependency on the dev environment
- **With no `.env`**, the app fails with an actionable message rather than a
  stack trace:

  > `MissingKey: No provider configured for the text model. Set ANTHROPIC_API_KEY
  > (primary) and/or GEMINI_API_KEY (failover) in .env.` — with both key URLs.

One dependency (`langsmith`) arrived transitively through `langchain-core`; it is
now pinned explicitly, because tracing is a documented feature and a transitive
dependency is not a promise.

**Placeholder handling:** values left as `your_gemini_key_here` are treated as
absent. Without this, a half-filled `.env` makes the app report a failover it
does not have — and fail on a 401 at exactly the moment it is needed.

---

## Latency

Measured over a mix of message shapes — plain logs, corrections, queries,
fractional portions — not just easy cases.

| Path | p50 | p95 | mean | min | max | n |
|---|---|---|---|---|---|---|
| **text, total** | **1.97s** | **2.80s** | 1.91s | 0.98s | 2.93s | 20 |
| text, first token | 1.61s | 2.65s | — | — | — | 20 |
| **image, total** (Sonnet) | **6.12s** | **10.07s** | 6.88s | 4.95s | 10.47s | 8 |
| image, total (Haiku) | 6.39s | 7.41s | 6.31s | 4.46s | 7.51s | 10 |

### What was done to get there

**1. Disabled Gemini's internal reasoning pass** (`thinking_budget=0`). For "had
2 rotis" it costs more wall-clock than the rest of the turn and buys nothing.

**2. Pre-loaded today's totals into the system prompt.** "How am I doing?" is
answered with **zero tool calls** — an entire model round trip saved on one of
the most frequent messages a user sends.

**3. Memory writes moved off the critical path.** ~300–600ms per turn.

**4. Local nutrition table.** No network call on the hot path; only a genuine
miss costs a model call, and the result is cached back permanently.

**5. `max_retries=0` on the text path.** Immediate failover instead of seconds of
backoff.

**6. Timeouts set from measurement.** An earlier 6s default was cutting off a
*healthy* 8.8s call — "leftover biryani, maybe two thirds of the box", which the
model gets **right**, it just deliberates longer — turning a correct answer into
a hard failure. Now 10s.

**7. Prompt caching**, breakpoint placed after the static instructions and before
per-user state so that eating a meal cannot invalidate the prefix.

### Prompt caching, including a wrong conclusion I corrected

Anthropic enforces a per-model **minimum cacheable prefix**. I initially concluded
caching was unavailable on the API key. That was wrong. Measuring the boundary
directly:

```
claude-haiku-4-5    4059 tok → not cached    4509 tok → cached   (floor 4096)
claude-sonnet-5     4803 tok → cached
```

The prefix was ~2583 tokens — *just* under Haiku's floor. Not the key; the prompt
was too small.

Rather than pad with filler to cross the threshold, the ~1500-token gap was
filled with worked examples that earn their place independently. Prefix is now
4282 tokens and caching engages.

Per-call input cost on Haiku 4.5 ($1.00/MTok input, 1.25× write, 0.1× read):

| | tokens | cost |
|---|---|---|
| before | 2583 uncached | $0.002583 |
| cache write | 4282 write + 350 | $0.005703 |
| cache read | 4282 read + 350 | **$0.000778** |

Cache reads are **70% cheaper than the original**, while carrying 66% more
content. Break-even is ~2 hits per write; each tool-using turn makes two model
calls sharing one prefix, so a single multi-turn exchange clears it. **But the
default TTL is 5 minutes** — a user who logs breakfast and goes quiet for four
hours pays the write and never collects. Bursty sessions win; sparse ones lose
about 25%.

Latency was unchanged by the extra tokens: p50 2.33s vs 2.38s.

### What is slow, and why it was not fixed

**The image path at 6.1s p50 is not messaging speed.** Vision takes ~2.6s and the
text turn runs *after* it, serially — because the text model needs to know what
is on the plate before it can decide anything. The fix (speculative nutrition
resolution overlapping the vision call) is real work with modest payoff, and time
went to correctness work instead. It is the top item on the roadmap.

**Streaming buys almost nothing here.** First token p50 1.61s against a total of
1.97s — the user-visible text is generated *after* the tool round trip, so there
is little to stream. Implemented and working, but not the latency story.

---

## Bonus features

| Bonus | Status | Notes |
|---|---|---|
| Streaming responses | delivered | token-by-token; honest about limited benefit |
| Multi-user / session isolation | delivered | contextvar-scoped, eval-verified |
| Eval set with a definition of correct | delivered | 15 cases, DB-level assertions |
| LangSmith tracing | wired, unverified | env-var only; no key to confirm with |
| **Provider failover** | delivered | beyond the brief |
| **Prompt caching** | delivered | with measured cost analysis |

### Session isolation

`user_id` flows from a `ContextVar`, never a tool argument. Verified by an eval
that plants a second user's 1400 kcal meal *and* a conflicting "my usual"
shortcut in the same database, then asserts neither reaches the user under test.

This was originally verified by hand and later codified — precisely because it is
the kind of thing that breaks silently and is never noticed until it is someone
else's calorie count.

### LangSmith

`LANGSMITH_TRACING=true` plus a key is sufficient; LangGraph emits traces with no
application changes. **Not verified end to end** — no LangSmith key was
available. Reported as present-and-correctly-named rather than working.

---

## Bugs found, and how

The most useful artefact in this repo may be the commit history, because six
significant bugs were invisible until the code met a real model. All six are in
the log with their diagnosis.

| Bug | Symptom | Found by |
|---|---|---|
| Doubled totals | told user 1050 kcal, DB held 525 | first live call |
| Streaming emitted nothing | Anthropic returns content *blocks*, code assumed a string | latency bench showing empty replies |
| Vision default 400'd on every call | Sonnet 5 rejects `temperature`; evals ran on Haiku, which accepts it | insisting on testing the shipped default |
| Placeholder caption poisoned vision | `[sent a photo of their food]` passed to the vision model as a caption | a real photo |
| Shipped Gemini fallback models 404'd | `gemini-2.5-flash(-lite)` still listed by `models.list()`, but generation blocked for new-user cohorts | a live failover run on a freshly issued key |
| `thinking_budget=0` rejected outright | bare 400, no detail, only on that exact value — 1, -1, 128 all fine | bisecting kwargs on a bare call after the error string said nothing useful |

Three meta-lessons, all encoded in the code and the docs:

**The tested configuration and the shipped configuration must be the same one.**
The vision bug existed precisely because evals pinned Haiku for cost while
Sonnet shipped. Nothing else would have caught it. The Gemini model-deprecation
bug is the same lesson at the account level: an old developer key kept working
and grandfathered access to a model a brand-new key silently can't reach, so
"it works on my key" was never sufficient evidence.

**A passing eval is not evidence until you have checked what it asserts.** The
image eval was green while the model logged idli and dosa for a plate of roti.

**The "disable reasoning" knob is not portable.** `thinking_budget=0` was a
correct, working value for the Gemini 2.5 line and a rejected one for 3.5, with
an error message that gave no hint why. A latency optimisation built around one
model generation's specific parameter contract is a live liability the moment
that contract changes underneath it — and the only way this was caught was
running the actual call, not reading documentation about it.

---

## Scaling this up

Everything above is a single-process CLI backed by SQLite. This section is what
it would take to run CalorAI as a real WhatsApp product, with an actual budget.

Assume a target of **100,000 monthly active users**, averaging **8 text turns and
1 photo per day**.

### Phase 1 — Make it a service (weeks 1–4)

**Replace the CLI with an HTTP surface.** FastAPI + async LangGraph, deployed
behind the WhatsApp Business Cloud API. WhatsApp delivers webhooks and expects a
fast 200; the actual agent work goes on a queue.

```
WhatsApp → webhook (ack in <500ms) → queue → worker pool → agent → send API
```

This decouples user-perceived latency from agent latency, which matters because
the image path is 6s and WhatsApp will retry a slow webhook.

**SQLite → Postgres.** The schema ports directly; the invariant (totals as
`SUM()`) is unchanged and benefits from a proper index on
`(user_id, meal_date, deleted)`. Add connection pooling (PgBouncer) and read
replicas for the totals queries, which are read-heavy and latency-sensitive.

**Idempotency.** WhatsApp redelivers on timeout. Every inbound message needs a
dedupe key on its message id, or users get double-logged meals — which would
defeat the entire correctness design.

**Effort:** 2 engineers × 4 weeks. **Infra:** ~$800/month at this stage.

### Phase 2 — Latency (weeks 4–8)

The image path is the weak number. Three fixes, in order of payoff:

1. **Parallelise perception and preparation.** Start nutrition resolution for
   likely candidates and pre-warm the prompt cache while vision is still running.
   Expected: 6.1s → ~4.5s.
2. **Stream the vision result into the text turn.** Begin the text call as soon as
   the first identified item arrives rather than waiting for the full payload.
   Expected: ~4.5s → ~3s.
3. **Downscale images before upload.** A 4MB phone photo becomes ~200KB at
   1024px with no measurable accuracy loss, cutting both upload and prefill time.

**Prompt caching becomes properly economic at scale.** The 5-minute TTL loses on
sparse traffic, but with a worker pool serving thousands of concurrent users the
shared prefix is hit constantly. Consider a **1-hour TTL** — 2× write cost, but at
scale the hit rate justifies it.

**Effort:** 1 engineer × 4 weeks.

### Phase 3 — Memory at scale (weeks 8–14)

The current design is deliberately bounded and will hold much further than a
naive one. What changes at scale:

**Semantic recall for aliases.** Keyword matching works for "my usual"; it will
not work for a user with 50 shortcuts phrased loosely. Add pgvector over alias
descriptions, retrieved into tier 2 only — the tier-1 budget stays fixed.

**Memory consolidation and decay.** Facts currently only ever overwrite. Over
months you want history: "was vegetarian, started eating fish in March" is a
timeline, not a value. Add a periodic consolidation job (batch API, off-peak) that
summarises `memory_history` into a narrative and expires stale facts.

**Per-user memory budgets.** A hard token ceiling per user on tier 1, with
eviction by the existing priority ordering. Already partially built — the cap
exists; the eviction policy needs to be smarter than "drop the lowest priority".

**Learn portions per user.** If someone's "bowl of dal" is consistently corrected
upward, store that as a per-user portion multiplier. This is the highest-value
accuracy work available and it needs no new models — just mining the corrections
already in the database.

**Effort:** 2 engineers × 6 weeks.

### Phase 4 — Nutrition accuracy (weeks 10–16)

The 67-food table is right for a prototype and wrong for 100k users.

- **Ingest a real corpus** — IFCT 2017 for Indian foods, USDA FoodData Central
  for the rest. Tens of thousands of entries.
- **Embedding-based food matching** instead of fuzzy string matching, which will
  mis-hit badly at scale.
- **Regional variants.** A Gujarati dal and a Punjabi dal are not the same
  calories. Use the user's own history to disambiguate.
- **Human review queue** for LLM-estimated foods that exceed a usage threshold —
  if 5,000 users log "misal pav" from an LLM guess, that guess deserves a human
  check once.

**Effort:** 1 engineer + 1 nutritionist × 6 weeks.

### Cost model

Per turn on Haiku 4.5, with caching active and ~150 output tokens:

```
input  (cache read)   4282 × $0.10/MTok  = $0.00043
input  (dynamic)       350 × $1.00/MTok  = $0.00035
output                 150 × $5.00/MTok  = $0.00075
                                          ─────────
per text turn                             ≈ $0.0015
```

Image turns on Sonnet 5 ($2/$10 per MTok) with a downscaled image (~1.5k tokens)
run roughly **$0.02** each.

| Item | Daily | Monthly (100k MAU) |
|---|---|---|
| 8 text turns/user | $0.012 | **$36,000** |
| 1 photo/user | $0.020 | **$60,000** |
| Memory extraction (1/turn, cheap) | $0.004 | $12,000 |
| **LLM subtotal** | | **~$108,000** |
| Infra (Postgres, workers, queue, storage) | | ~$8,000 |
| **Total** | | **~$116,000/month** |

That is **~$1.16 per user per month** — viable against a subscription, tight
against ads.

**The three levers that move it most:**

1. **Route photos by difficulty.** Most plates are easy. A cheap first pass on
   Haiku with escalation to Sonnet only on low confidence could cut the image
   line by half — ~$30k/month. This is exactly what the calibration finding above
   argues for, and it needs the per-model thresholds to be fixed first.
2. **Batch the memory extractor.** It is already off the critical path, so it can
   go through the Batch API at 50% cost — ~$6k/month for a one-line change.
3. **1-hour cache TTL** at high concurrency — meaningful once the hit rate is
   measured.

Together those are roughly a **third off the bill** without touching quality.

### Quality and safety at scale

**Online evals.** The 15-case suite becomes a CI gate plus a sampled production
eval — score a random 1% of real turns nightly against the same DB-level
assertions. The flakiness problem gets solved properly here: run each case N
times, require a pass rate, alert on drift.

**Correction mining as a quality signal.** Every `correct_item` call is a labelled
error. Aggregated, corrections tell you exactly which foods the nutrition table
gets wrong and which phrasings the agent misparses. This is free training data the
product generates by being used.

**Guardrails the prototype does not have:**

- **Medical boundary.** A calorie tracker attracts users with eating disorders.
  Hard rules against commenting on weight, refusing to log, or moralising about
  food. The current prompt forbids opinions on healthiness — at scale that needs
  to be an enforced classifier, not a prompt instruction.
- **PII and deletion.** Food logs are health data. GDPR/DPDP compliance,
  encryption at rest, and a working delete-my-account path that actually purges
  `memories`, `memory_history` and `messages` — not just `meals`.
- **Prompt injection.** Users can type anything, including instructions. The
  contextvar design already prevents cross-user access; the tool surface needs an
  audit against a user who is actively trying to break it.

**Observability.** LangSmith for traces, structured logs keyed by turn id, and
dashboards on the numbers that actually matter: p50/p95 per path, tool-call error
rate, clarifying-question rate (over-asking is a silent product failure), and
cache hit rate.

### What I would *not* do

**Not fine-tune a model.** The behaviour is prompt-shaped and the examples are
working. Fine-tuning would freeze judgement that still needs to move.

**Not replace the nutrition table with an LLM.** Deterministic lookups are the
reason totals are reproducible. Estimation stays the fallback, never the default.

**Not move totals into a cached counter**, however tempting at scale. The `SUM()`
invariant is what makes corrections safe, and a denormalised counter is the single
change most likely to reintroduce the double-counting class of bug. If read load
demands it, use a materialised view with the `SUM()` as its definition — so the
invariant remains the source of truth rather than being replaced by one.
