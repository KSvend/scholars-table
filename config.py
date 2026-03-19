import os

# LLM Backend
DEFAULT_TIER = "free"
FREE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FREE_FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# API behavior
API_RETRY_DELAY_SECONDS = 2
API_CALL_DELAY_SECONDS = 1.5
MAX_RESPONSE_TOKENS = 1024

# Paths
PERSONAS_DIR = os.path.join(os.path.dirname(__file__), "scholars", "personas")

# Scholar IDs (order matters — used for turn order in later phases)
SCHOLAR_IDS = [
    "peacegrave",
    "ironhelm",
    "silencio",
    "flickerstone",
]
