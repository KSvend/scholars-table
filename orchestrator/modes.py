# orchestrator/modes.py
import time
import logging
import config
from orchestrator.judge import Judge

logger = logging.getLogger(__name__)


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


class PanelDiscussion:
    """Mode 2: Structured panel where all scholars respond, then rebut each other."""

    def __init__(self, scholar_ids: list[str], engine, router):
        self.scholar_ids = scholar_ids
        self.engine = engine
        self.router = router
        self.judge = Judge(router)
        self.conversation: list[dict] = []
        self.initial_responses: dict[str, str] = {}
        self.phase = "waiting"
        self.question = ""

    def start_discussion(self, question: str):
        """
        Generator that yields response dicts for each scholar's initial answer.
        Sets phase to 'rebuttals' when all scholars have responded.
        """
        self.question = question
        self.conversation.append({"role": "user", "content": question})
        scholar_names = self.engine.get_scholar_names()

        for sid in self.scholar_ids:
            name = scholar_names.get(sid, sid)
            try:
                system_prompt = self.engine.build_system_prompt(sid, mode="multi")
                response_text = self.router.generate(
                    system_prompt=system_prompt,
                    messages=self.conversation,
                )
                self.initial_responses[sid] = response_text
                labeled = f"**{name}:**\n\n{response_text}"
                msg = {"role": "assistant", "content": labeled, "scholar_id": sid}
                self.conversation.append(msg)
                yield msg
            except Exception as e:
                logger.warning(f"Scholar {sid} failed in start_discussion: {e}")
                error_msg = {"role": "assistant", "content": f"**{name}:** *(unavailable)*", "scholar_id": sid}
                self.conversation.append(error_msg)
                yield error_msg

            time.sleep(config.API_CALL_DELAY_SECONDS)

        self.phase = "rebuttals"

    def generate_rebuttals(self):
        """
        Generator that yields rebuttal response dicts.
        Uses judge to identify tensions, then generates targeted rebuttals.
        Sets phase to 'open' when done.
        """
        scholar_names = self.engine.get_scholar_names()
        tensions = self.judge.analyze_tensions(self.initial_responses, self.question)

        for tension in tensions[: config.PANEL_MAX_REBUTTALS]:
            responder = tension.get("responder")
            target = tension.get("target")
            if not responder or not target:
                continue
            if responder not in self.scholar_ids:
                continue

            name = scholar_names.get(responder, responder)
            try:
                system_prompt = self.engine.build_system_prompt(
                    responder, mode="multi", responding_to=target
                )
                response_text = self.router.generate(
                    system_prompt=system_prompt,
                    messages=self.conversation,
                )
                labeled = f"**{name}:**\n\n{response_text}"
                msg = {"role": "assistant", "content": labeled, "scholar_id": responder}
                self.conversation.append(msg)
                yield msg
            except Exception as e:
                logger.warning(f"Scholar {responder} failed rebuttal: {e}")

            time.sleep(config.API_CALL_DELAY_SECONDS)

        self.phase = "open"

    def add_interjection(self, message: str):
        """Append a user interjection to the conversation."""
        self.conversation.append({"role": "user", "content": message})

    def continue_discussion(self, user_message: str = None):
        """
        For follow-up exchanges after rebuttals.
        Optionally appends a user message, then uses judge to pick who responds next.
        Returns a single response dict.
        """
        if user_message:
            self.add_interjection(user_message)

        scholar_names = self.engine.get_scholar_names()
        turn_counts = {}
        for msg in self.conversation:
            sid = msg.get("scholar_id")
            if sid:
                turn_counts[sid] = turn_counts.get(sid, 0) + 1

        sid = self.judge.select_next_speaker(
            self.scholar_ids, self.conversation, turn_counts, scholar_names
        )
        name = scholar_names.get(sid, sid)
        try:
            system_prompt = self.engine.build_system_prompt(sid, mode="multi")
            response_text = self.router.generate(
                system_prompt=system_prompt,
                messages=self.conversation,
            )
            labeled = f"**{name}:**\n\n{response_text}"
            msg = {"role": "assistant", "content": labeled, "scholar_id": sid}
            self.conversation.append(msg)
            return msg
        except Exception as e:
            logger.warning(f"Scholar {sid} failed continue_discussion: {e}")
            return None

    def reset(self):
        """Clear all state."""
        self.conversation = []
        self.initial_responses = {}
        self.phase = "waiting"
        self.question = ""


class FreeDebate:
    """Mode 3: Open-ended debate where scholars take turns directed by a judge."""

    def __init__(self, scholar_ids: list[str], engine, router, max_exchanges: int = None):
        self.scholar_ids = scholar_ids
        self.engine = engine
        self.router = router
        self.judge = Judge(router)
        self.max_exchanges = max_exchanges if max_exchanges is not None else config.FREE_DEBATE_MAX_EXCHANGES
        self.conversation: list[dict] = []
        self.turn_counts: dict[str, int] = {sid: 0 for sid in scholar_ids}
        self.exchange_count = 0
        self.running = False

    def start(self, topic: str):
        """
        Generator that yields opening-round response dicts for each scholar.
        Sets running=True when the opening round completes.
        """
        self.conversation.append({"role": "user", "content": topic})
        scholar_names = self.engine.get_scholar_names()

        for sid in self.scholar_ids:
            name = scholar_names.get(sid, sid)
            try:
                system_prompt = self.engine.build_system_prompt(sid, mode="multi")
                response_text = self.router.generate(
                    system_prompt=system_prompt,
                    messages=self.conversation,
                )
                labeled = f"**{name}:**\n\n{response_text}"
                msg = {"role": "assistant", "content": labeled, "scholar_id": sid}
                self.conversation.append(msg)
                self.turn_counts[sid] = self.turn_counts.get(sid, 0) + 1
                self.exchange_count += 1
                yield msg
            except Exception as e:
                logger.warning(f"Scholar {sid} failed opening round: {e}")

            time.sleep(config.API_CALL_DELAY_SECONDS)

        self.running = True

    def next_turn(self):
        """
        Generate the next debate turn. Returns a response dict or None if the debate
        should end (max exchanges reached or convergence detected).
        """
        if not self.running or self.exchange_count >= self.max_exchanges:
            self.running = False
            return None

        scholar_names = self.engine.get_scholar_names()

        # Periodic convergence check
        if (
            self.exchange_count > 0
            and self.exchange_count % config.FREE_DEBATE_CONVERGENCE_CHECK_INTERVAL == 0
        ):
            assistant_msgs = [m for m in self.conversation if m["role"] == "assistant"]
            if self.judge.check_convergence(assistant_msgs):
                logger.info("Debate convergence detected.")
                return {
                    "role": "assistant",
                    "content": "*The discussion appears to be reaching consensus. You may wish to introduce a new angle, or stop the debate.*",
                    "convergence": True,
                }

        # Detect silent scholars every 5 turns
        silent = []
        if self.exchange_count % config.FREE_DEBATE_CONVERGENCE_CHECK_INTERVAL == 0:
            threshold = config.FREE_DEBATE_SILENCE_THRESHOLD
            silent = [
                sid for sid in self.scholar_ids
                if self.turn_counts.get(sid, 0) == 0
                or (self.exchange_count - self.turn_counts.get(sid, 0)) >= threshold
            ]
        if silent:
            sid = silent[0]
        else:
            sid = self.judge.select_next_speaker(
                self.scholar_ids, self.conversation, self.turn_counts, scholar_names
            )

        name = scholar_names.get(sid, sid)
        try:
            system_prompt = self.engine.build_system_prompt(sid, mode="multi")
            response_text = self.router.generate(
                system_prompt=system_prompt,
                messages=self.conversation,
            )
            labeled = f"**{name}:**\n\n{response_text}"
            msg = {"role": "assistant", "content": labeled, "scholar_id": sid}
            self.conversation.append(msg)
            self.turn_counts[sid] = self.turn_counts.get(sid, 0) + 1
            self.exchange_count += 1
            return msg
        except Exception as e:
            logger.warning(f"Scholar {sid} failed next_turn: {e}")
            return None

    def add_interjection(self, message: str):
        """Append a user interjection to the conversation."""
        self.conversation.append({"role": "user", "content": message})

    def stop(self):
        """Stop the debate."""
        self.running = False

    def reset(self):
        """Clear all state."""
        self.conversation = []
        self.turn_counts = {sid: 0 for sid in self.scholar_ids}
        self.exchange_count = 0
        self.running = False
