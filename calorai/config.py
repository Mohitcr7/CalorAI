"""Central configuration: providers, model selection, timeouts, thresholds.

TWO PROVIDERS, THREE ROLES
--------------------------
Claude is primary. Gemini is failover. Each of the three roles resolves to a
model on both providers, so a provider outage degrades quality slightly rather
than taking the product down.

  role       primary (Anthropic)      failover (Google)
  ---------  -----------------------  -----------------------
  text       claude-haiku-4-5         gemini-2.5-flash-lite
  vision     claude-sonnet-5          gemini-2.5-flash
  extractor  claude-haiku-4-5         gemini-2.5-flash-lite

Why these:

* TEXT runs the conversation and every tool call, on the critical path for every
  single message. Haiku 4.5 is the fastest Claude model with reliable tool
  calling, which is exactly the trade this role wants. Routing "had 2 rotis" to
  a frontier model would be slower and no more correct.
* VISION only ever sees plate photos. Recognising a half-eaten paratha under bad
  kitchen lighting is a genuinely harder perceptual task than routing text, and
  it runs at most once per turn, so it gets Sonnet 5. This is also what keeps
  the two paths on genuinely different models rather than nominally so.
* EXTRACTOR writes long-term memory after the reply has already been streamed.
  Its latency is invisible to the user, so it takes the cheap model.

TIMEOUT BUDGETS
---------------
Failover is only useful if it is fast. A naive fallback makes the user wait the
full primary timeout AND then the full failover latency, so a 20s timeout on
both sides means a 40s turn -- worse than simply failing. The text path
therefore runs a deliberately tight primary timeout, so the worst case stays
bounded:

    text:   6s primary  + 12s failover  = 18s worst case
    vision: 15s primary + 20s failover  = 35s worst case

See llm.py for the retry policy, which matters just as much.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("CALORAI_DB", ROOT / "data" / "calorai.db"))

# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Lets you force one provider for testing or a demo, e.g. to show the failover
# path working without having to break anything.
#   auto      -> Claude primary, Gemini failover (default)
#   anthropic -> Claude only, no failover
#   google    -> Gemini only
PROVIDER_MODE = os.getenv("CALORAI_PROVIDER", "auto").lower()

# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
TEXT_MODEL = os.getenv("CALORAI_TEXT_MODEL", "claude-haiku-4-5-20251001")
VISION_MODEL = os.getenv("CALORAI_VISION_MODEL", "claude-sonnet-5")
EXTRACTOR_MODEL = os.getenv("CALORAI_EXTRACTOR_MODEL", "claude-haiku-4-5-20251001")

FALLBACK_TEXT_MODEL = os.getenv("CALORAI_FALLBACK_TEXT_MODEL", "gemini-2.5-flash-lite")
FALLBACK_VISION_MODEL = os.getenv("CALORAI_FALLBACK_VISION_MODEL", "gemini-2.5-flash")
FALLBACK_EXTRACTOR_MODEL = os.getenv("CALORAI_FALLBACK_EXTRACTOR_MODEL", "gemini-2.5-flash-lite")

# --------------------------------------------------------------------------
# timeouts (seconds)
# --------------------------------------------------------------------------
TEXT_TIMEOUT = float(os.getenv("CALORAI_TEXT_TIMEOUT", "6"))
TEXT_FALLBACK_TIMEOUT = float(os.getenv("CALORAI_TEXT_FALLBACK_TIMEOUT", "12"))
VISION_TIMEOUT = float(os.getenv("CALORAI_VISION_TIMEOUT", "15"))
VISION_FALLBACK_TIMEOUT = float(os.getenv("CALORAI_VISION_FALLBACK_TIMEOUT", "20"))
EXTRACTOR_TIMEOUT = float(os.getenv("CALORAI_EXTRACTOR_TIMEOUT", "15"))

# Gemini 2.5 runs an internal reasoning pass by default. Zero disables it; on
# the text path that pass costs more wall clock than the rest of the turn and
# buys nothing for "had 2 rotis". Only applies to the Gemini side.
TEXT_THINKING_BUDGET = int(os.getenv("CALORAI_TEXT_THINKING", "0"))
VISION_THINKING_BUDGET = int(os.getenv("CALORAI_VISION_THINKING", "0"))

# --------------------------------------------------------------------------
# behaviour thresholds
# --------------------------------------------------------------------------
# Tier-1 memory injection budget: how many facts may enter the system prompt.
MAX_TIER1_FACTS = int(os.getenv("CALORAI_MAX_TIER1_FACTS", "15"))

# Ambiguity policy: if a portion guess could swing the meal's calories by more
# than this fraction, ask instead of assuming.
PORTION_AMBIGUITY_THRESHOLD = 0.40

# Vision confidence below this is treated as "the model is unsure" and routed
# to a clarifying question rather than a silent log.
VISION_MIN_CONFIDENCE = 0.45
