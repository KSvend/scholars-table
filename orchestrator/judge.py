# orchestrator/judge.py
import json
import logging
import config

logger = logging.getLogger(__name__)


class Judge:
    """LLM-as-judge for orchestrating multi-scholar discussions."""

    def __init__(self, router):
        self.router = router

    def _call_judge(self, prompt: str) -> dict | None:
        """Make a judge call and parse JSON response. Returns None on failure."""
        try:
            system = (
                "You are a discussion moderator analyzing an academic debate. "
                "Respond ONLY with valid JSON, no other text."
            )
            response = self.router.generate(
                system_prompt=system,
                messages=[{"role": "user", "content": prompt}],
                model_override=config.JUDGE_MODEL,
                max_tokens_override=config.JUDGE_MAX_TOKENS,
            )
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Judge call failed: {e}")
            return None

    def analyze_tensions(self, responses: dict[str, str], question: str) -> list[dict]:
        response_text = "\n\n".join(f"[{sid}]: {text}" for sid, text in responses.items())
        prompt = (
            f"Original question: {question}\n\n"
            f"Scholar responses:\n{response_text}\n\n"
            f"Identify the 2-4 strongest intellectual disagreements between scholars. "
            f"Return JSON: {{\"tensions\": [{{\"responder\": \"scholar_id who should rebut\", "
            f"\"target\": \"scholar_id being rebutted\", \"summary\": \"what the disagreement is about\", "
            f"\"score\": 0.0-1.0}}]}}. Sort by score descending."
        )
        result = self._call_judge(prompt)
        if result and "tensions" in result:
            return sorted(result["tensions"], key=lambda t: t.get("score", 0), reverse=True)
        return []

    def select_next_speaker(self, scholars, conversation, turn_counts, scholar_names) -> str:
        last_messages = conversation[-3:] if len(conversation) >= 3 else conversation
        conv_text = "\n".join(m["content"] for m in last_messages)
        counts_text = ", ".join(f"{scholar_names.get(s, s)}: {turn_counts.get(s, 0)} turns" for s in scholars)
        prompt = (
            f"Recent debate exchanges:\n{conv_text}\n\n"
            f"Turn counts: {counts_text}\n\n"
            f"Available scholars: {', '.join(scholars)}\n\n"
            f"Which scholar should speak next? Consider: whose tradition is most provoked "
            f"by the last response, and who has spoken least. "
            f"Return JSON: {{\"next_speaker\": \"scholar_id\", \"reason\": \"brief reason\"}}"
        )
        result = self._call_judge(prompt)
        if result and "next_speaker" in result and result["next_speaker"] in scholars:
            return result["next_speaker"]
        return min(scholars, key=lambda s: turn_counts.get(s, 0))

    def check_convergence(self, last_responses: list[dict]) -> bool:
        texts = "\n\n".join(m["content"] for m in last_responses[-3:])
        prompt = (
            f"Last 3 debate responses:\n{texts}\n\n"
            f"Are these responses introducing genuinely new concepts or arguments, "
            f"or are they repeating/rephrasing earlier points? "
            f"Return JSON: {{\"new_concepts\": [\"list of new concepts if any\"], "
            f"\"converged\": true/false}}"
        )
        result = self._call_judge(prompt)
        if result and "converged" in result:
            return bool(result["converged"])
        return False
