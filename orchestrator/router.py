import logging
import time
from huggingface_hub import InferenceClient
import config

logger = logging.getLogger(__name__)


class LLMRouter:
    """Routes LLM requests to the appropriate backend."""

    VALID_TIERS = ("free",)  # Premium added in Phase 2

    def __init__(self, tier: str = "free"):
        if tier not in self.VALID_TIERS:
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: {self.VALID_TIERS}")

        self.tier = tier
        self.model = config.FREE_MODEL
        self.fallback_model = config.FREE_FALLBACK_MODEL
        self.client = InferenceClient(token=config.HF_TOKEN or None)

    def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Generate a response given a system prompt and message history."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=full_messages,
                max_tokens=config.MAX_RESPONSE_TOKENS,
            )
            return response.choices[0].message.content
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            # Retry with fallback model on transient errors
            logger.warning(f"Primary model failed ({type(e).__name__}: {e}), falling back to {self.fallback_model}")
            time.sleep(config.API_RETRY_DELAY_SECONDS)
            response = self.client.chat_completion(
                model=self.fallback_model,
                messages=full_messages,
                max_tokens=config.MAX_RESPONSE_TOKENS,
            )
            return response.choices[0].message.content
