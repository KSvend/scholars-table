# orchestrator/modes.py


class PrivateConsultation:
    """Mode 1: 1:1 conversation with a single scholar."""

    def __init__(self, scholar_id: str, engine, router):
        self.scholar_id = scholar_id
        self.engine = engine
        self.router = router
        self.history: list[dict] = []

    def send_message(self, user_message: str) -> str:
        """Send a message and get the scholar's response."""
        self.history.append({"role": "user", "content": user_message})

        system_prompt = self.engine.build_system_prompt(self.scholar_id, mode="private")
        response = self.router.generate(
            system_prompt=system_prompt,
            messages=self.history,
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        """Clear conversation history."""
        self.history = []
