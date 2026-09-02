"""Central configuration: model selection, paths, tunable thresholds.

Model split rationale (see README for the long version):

* TEXT_MODEL runs the conversation and every tool call. It is on the critical
  path for every single user message, so it is the fastest model available and
  runs with thinking disabled.
* VISION_MODEL only ever sees images. Food recognition from a plate photo is a
  harder perceptual task than routing "had 2 rotis", so it gets a stronger
  model and a small thinking budget. It never gets tools -- it returns one
  structured payload and hands off.
* EXTRACTOR_MODEL writes long-term memory. It runs *after* the reply has been
  streamed to the user, so its latency is invisible and it can be cheap.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("CALORAI_DB", ROOT / "data" / "calorai.db"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

TEXT_MODEL = os.getenv("CALORAI_TEXT_MODEL", "gemini-2.5-flash-lite")
VISION_MODEL = os.getenv("CALORAI_VISION_MODEL", "gemini-2.5-flash")
EXTRACTOR_MODEL = os.getenv("CALORAI_EXTRACTOR_MODEL", "gemini-2.5-flash-lite")

# Thinking budgets. 0 disables Gemini's internal reasoning pass entirely, which
# is the single largest latency win on the text path.
TEXT_THINKING_BUDGET = int(os.getenv("CALORAI_TEXT_THINKING", "0"))
VISION_THINKING_BUDGET = int(os.getenv("CALORAI_VISION_THINKING", "0"))

# Tier-1 memory injection budget. Facts are one short line each; this caps how
# much of the system prompt long-term memory is allowed to occupy.
MAX_TIER1_FACTS = int(os.getenv("CALORAI_MAX_TIER1_FACTS", "15"))

# Ambiguity policy: if a portion guess could swing the meal's calories by more
# than this fraction, ask instead of assuming.
PORTION_AMBIGUITY_THRESHOLD = 0.40

# Vision confidence below this is treated as "the vision model is unsure" and
# routed to a clarifying question rather than a silent log.
VISION_MIN_CONFIDENCE = 0.45
