# Phase 2: Panel Discussion & Free Debate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mode 2 (Panel Discussion with LLM-as-judge rebuttals) and Mode 3 (Free Debate with orchestrator balancing and convergence detection), plus update the Gradio UI with mode selection.

**Architecture:** New `judge.py` module handles LLM-as-judge calls for tension analysis, speaker selection, and convergence detection using a lighter model. `PanelDiscussion` and `FreeDebate` classes in `modes.py` orchestrate multi-scholar conversations. The Gradio UI adds a mode selector and multi-scholar picker. All responses stream sequentially with progress indicators via Gradio's yield-based updates.

**Tech Stack:** Python 3.11+, Gradio 6.x, huggingface_hub, existing ScholarEngine + LLMRouter

**Spec:** `docs/superpowers/specs/2026-03-19-scholars-table-design.md` (sections: Orchestrator Logic, Interaction Modes, Error Handling)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | Modify | Add judge model, debate config constants |
| `orchestrator/judge.py` | Create | LLM-as-judge: tension analysis, speaker selection, convergence detection |
| `orchestrator/modes.py` | Modify | Add PanelDiscussion and FreeDebate classes |
| `app.py` | Modify | Mode selector, multi-scholar picker, streaming debate view |
| `tests/test_judge.py` | Create | Judge module tests |
| `tests/test_modes.py` | Modify | Add PanelDiscussion and FreeDebate tests |
| `tests/test_app.py` | Modify | Add mode switching smoke tests |

---

### Task 1: Config Updates

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add Phase 2 config constants**

Add to `config.py`:

```python
# Judge model (lighter, for orchestrator decisions)
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MAX_TOKENS = 256

# Debate settings
FREE_DEBATE_MAX_EXCHANGES = 20
FREE_DEBATE_SILENCE_THRESHOLD = 3  # turns before nudge
FREE_DEBATE_CONVERGENCE_CHECK_INTERVAL = 5
PANEL_MAX_REBUTTALS = 4  # max rebuttal pairs per round
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add Phase 2 config — judge model, debate settings"
```

---

### Task 2: LLM-as-Judge Module

**Files:**
- Create: `orchestrator/judge.py`
- Create: `tests/test_judge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_judge.py
import pytest
import json
from unittest.mock import MagicMock, patch
from orchestrator.judge import Judge


class TestAnalyzeTensions:
    def test_returns_tension_tuples(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "tensions": [
                {"responder": "silencio", "target": "ironhelm", "summary": "Colonial power critique vs realist power analysis", "score": 0.9},
                {"responder": "flickerstone", "target": "ironhelm", "summary": "Constructed vs fixed interests", "score": 0.7},
            ]
        })
        judge = Judge(router=mock_router)
        responses = {
            "peacegrave": "Structural violence analysis...",
            "ironhelm": "Power dynamics dictate...",
            "silencio": "Colonial legacies persist...",
            "flickerstone": "These identities are constructed...",
        }
        tensions = judge.analyze_tensions(responses, "How should we understand the Sudan conflict?")
        assert len(tensions) == 2
        assert tensions[0]["responder"] == "silencio"
        assert tensions[0]["target"] == "ironhelm"
        assert tensions[0]["score"] > tensions[1]["score"]

    def test_handles_malformed_json(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "not valid json"
        judge = Judge(router=mock_router)
        tensions = judge.analyze_tensions({"a": "response"}, "question")
        assert tensions == []


class TestSelectNextSpeaker:
    def test_selects_speaker(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "next_speaker": "silencio",
            "reason": "Post-colonial lens most relevant to last point"
        })
        judge = Judge(router=mock_router)
        speaker = judge.select_next_speaker(
            scholars=["peacegrave", "ironhelm", "silencio", "flickerstone"],
            conversation=[{"role": "assistant", "content": "[Ironhelm] Power is what matters."}],
            turn_counts={"peacegrave": 2, "ironhelm": 2, "silencio": 1, "flickerstone": 2},
            scholar_names={"peacegrave": "Prof Peacegrave", "ironhelm": "Col Ironhelm",
                          "silencio": "Dr Silencio", "flickerstone": "Dr Flickerstone"},
        )
        assert speaker == "silencio"

    def test_falls_back_to_least_spoken_on_error(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "broken"
        judge = Judge(router=mock_router)
        speaker = judge.select_next_speaker(
            scholars=["peacegrave", "ironhelm", "silencio"],
            conversation=[],
            turn_counts={"peacegrave": 3, "ironhelm": 2, "silencio": 1},
            scholar_names={"peacegrave": "P", "ironhelm": "I", "silencio": "S"},
        )
        assert speaker == "silencio"


class TestCheckConvergence:
    def test_detects_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "new_concepts": ["none significant"],
            "converged": True
        })
        judge = Judge(router=mock_router)
        result = judge.check_convergence([
            {"role": "assistant", "content": "[Peacegrave] Same point again..."},
            {"role": "assistant", "content": "[Ironhelm] I agree for once..."},
            {"role": "assistant", "content": "[Silencio] We seem to be circling..."},
        ])
        assert result is True

    def test_no_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = json.dumps({
            "new_concepts": ["deterrence theory", "hybrid peace", "epistemic justice"],
            "converged": False
        })
        judge = Judge(router=mock_router)
        result = judge.check_convergence([
            {"role": "assistant", "content": "New point 1"},
            {"role": "assistant", "content": "New point 2"},
            {"role": "assistant", "content": "New point 3"},
        ])
        assert result is False

    def test_handles_error_as_no_convergence(self):
        mock_router = MagicMock()
        mock_router.generate.return_value = "broken"
        judge = Judge(router=mock_router)
        result = judge.check_convergence([{"role": "assistant", "content": "x"}] * 3)
        assert result is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_judge.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement judge.py**

```python
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
            # Strip markdown code fences if present
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
        """Analyze initial panel responses to find strongest disagreements.

        Args:
            responses: {scholar_id: response_text}
            question: The original question posed

        Returns:
            List of {responder, target, summary, score} sorted by score descending
        """
        response_text = "\n\n".join(
            f"[{sid}]: {text}" for sid, text in responses.items()
        )
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

    def select_next_speaker(
        self,
        scholars: list[str],
        conversation: list[dict],
        turn_counts: dict[str, int],
        scholar_names: dict[str, str],
    ) -> str:
        """Select the next speaker for free debate mode.

        Falls back to least-spoken scholar on error.
        """
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

        # Fallback: pick scholar with fewest turns
        return min(scholars, key=lambda s: turn_counts.get(s, 0))

    def check_convergence(self, last_responses: list[dict]) -> bool:
        """Check if the last 3 responses introduce new concepts.

        Returns True if debate has converged (no new ideas).
        """
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
        return False  # Default to not converged on error
```

- [ ] **Step 4: Add model_override and max_tokens_override to router**

The judge needs to use a different model than the scholars. Modify `orchestrator/router.py` `generate` method signature:

```python
def generate(self, system_prompt: str, messages: list[dict],
             model_override: str | None = None,
             max_tokens_override: int | None = None) -> str:
```

Use `model_override or self.model` and `max_tokens_override or config.MAX_RESPONSE_TOKENS` in the method body. Apply the same pattern in the fallback path.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_judge.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add orchestrator/judge.py orchestrator/router.py tests/test_judge.py
git commit -m "feat: LLM-as-judge for tension analysis, speaker selection, convergence"
```

---

### Task 3: Panel Discussion Mode

**Files:**
- Modify: `orchestrator/modes.py`
- Modify: `tests/test_modes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_modes.py`:

```python
import json
from orchestrator.modes import PanelDiscussion


class TestPanelDiscussion:
    def test_init(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm", "silencio"],
            engine=mock_engine,
            router=mock_router,
        )
        assert panel.scholar_ids == ["peacegrave", "ironhelm", "silencio"]
        assert panel.phase == "waiting"
        assert panel.conversation == []

    def test_initial_round_generates_all_responses(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = ["Peace response", "Realist response"]

        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
        )
        responses = list(panel.start_discussion("What about Sudan?"))
        assert len(responses) == 2
        assert "Peace response" in responses[0]["content"]
        assert "Realist response" in responses[1]["content"]
        assert panel.phase == "rebuttals"

    def test_initial_round_skips_failed_scholar(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "System prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = [Exception("API down"), "Realist response"]

        panel = PanelDiscussion(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
        )
        responses = list(panel.start_discussion("Question"))
        # Should get 1 response + 1 skip notice
        scholar_responses = [r for r in responses if "unavailable" not in r["content"]]
        assert len(scholar_responses) == 1

    def test_user_interjection(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {"peacegrave": "P"}
        mock_router = MagicMock()
        mock_router.generate.return_value = "Response"

        panel = PanelDiscussion(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        list(panel.start_discussion("Initial question"))
        panel.add_interjection("But what about power?")
        assert any(m["role"] == "user" for m in panel.conversation)

    def test_reset(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        panel = PanelDiscussion(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        panel.conversation = [{"role": "user", "content": "test"}]
        panel.reset()
        assert panel.conversation == []
        assert panel.phase == "waiting"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_modes.py::TestPanelDiscussion -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement PanelDiscussion**

Add to `orchestrator/modes.py`:

```python
import time
import config
from orchestrator.judge import Judge


class PanelDiscussion:
    """Mode 2: Multiple scholars respond to a question, then rebut each other."""

    def __init__(self, scholar_ids: list[str], engine, router):
        self.scholar_ids = scholar_ids
        self.engine = engine
        self.router = router
        self.judge = Judge(router=router)
        self.conversation: list[dict] = []
        self.initial_responses: dict[str, str] = {}
        self.phase = "waiting"  # waiting -> initial -> rebuttals -> open
        self.question = ""
        self._names = engine.get_scholar_names()

    def start_discussion(self, question: str):
        """Run the initial round. Yields response dicts as they arrive."""
        self.question = question
        self.conversation.append({"role": "user", "content": question})
        self.phase = "initial"
        self.initial_responses = {}

        for sid in self.scholar_ids:
            name = self._names.get(sid, sid)
            try:
                prompt = self.engine.build_system_prompt(sid, mode="multi")
                response = self.router.generate(
                    system_prompt=prompt,
                    messages=self.conversation,
                )
                labeled = f"**{name}:**\n\n{response}"
                self.initial_responses[sid] = response
                self.conversation.append({"role": "assistant", "content": labeled})
                yield {"role": "assistant", "content": labeled}
            except Exception:
                skip_msg = f"*{name} is momentarily unavailable.*"
                self.conversation.append({"role": "assistant", "content": skip_msg})
                yield {"role": "assistant", "content": skip_msg}

            time.sleep(config.API_CALL_DELAY_SECONDS)

        self.phase = "rebuttals"

    def generate_rebuttals(self):
        """Analyze tensions and generate rebuttals. Yields response dicts."""
        if not self.initial_responses or len(self.initial_responses) < 2:
            return

        tensions = self.judge.analyze_tensions(self.initial_responses, self.question)
        rebuttal_count = 0

        for tension in tensions[:config.PANEL_MAX_REBUTTALS]:
            responder = tension.get("responder")
            target = tension.get("target")
            if responder not in self.scholar_ids or target not in self.scholar_ids:
                continue

            name = self._names.get(responder, responder)
            target_name = self._names.get(target, target)

            try:
                prompt = self.engine.build_system_prompt(
                    responder, mode="multi", responding_to=target
                )
                meta = (
                    f"You are responding specifically to {target_name}'s argument. "
                    f"Tension: {tension.get('summary', 'disagreement')}. "
                    f"Engage directly with their points."
                )
                messages = self.conversation + [{"role": "user", "content": meta}]
                response = self.router.generate(
                    system_prompt=prompt,
                    messages=messages,
                )
                labeled = f"**{name}** *(responding to {target_name}):*\n\n{response}"
                self.conversation.append({"role": "assistant", "content": labeled})
                yield {"role": "assistant", "content": labeled}
                rebuttal_count += 1
            except Exception:
                skip_msg = f"*{name} is momentarily unavailable.*"
                self.conversation.append({"role": "assistant", "content": skip_msg})
                yield {"role": "assistant", "content": skip_msg}

            time.sleep(config.API_CALL_DELAY_SECONDS)

        self.phase = "open"

    def add_interjection(self, message: str):
        """Add a user interjection to the conversation."""
        self.conversation.append({"role": "user", "content": message})

    def continue_discussion(self, user_message: str | None = None):
        """Continue after rebuttals — any scholar can respond to the user or prior points."""
        if user_message:
            self.add_interjection(user_message)

        # Pick the most relevant scholar to respond next
        turn_counts = {}
        for msg in self.conversation:
            if msg["role"] == "assistant":
                for sid in self.scholar_ids:
                    name = self._names.get(sid, sid)
                    if msg["content"].startswith(f"**{name}"):
                        turn_counts[sid] = turn_counts.get(sid, 0) + 1

        next_sid = self.judge.select_next_speaker(
            scholars=self.scholar_ids,
            conversation=self.conversation,
            turn_counts=turn_counts,
            scholar_names=self._names,
        )

        name = self._names.get(next_sid, next_sid)
        prompt = self.engine.build_system_prompt(next_sid, mode="multi")
        response = self.router.generate(system_prompt=prompt, messages=self.conversation)
        labeled = f"**{name}:**\n\n{response}"
        self.conversation.append({"role": "assistant", "content": labeled})
        return {"role": "assistant", "content": labeled}

    def reset(self):
        """Clear all state."""
        self.conversation = []
        self.initial_responses = {}
        self.phase = "waiting"
        self.question = ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_modes.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator/modes.py tests/test_modes.py
git commit -m "feat: Mode 2 panel discussion with initial round and rebuttals"
```

---

### Task 4: Free Debate Mode

**Files:**
- Modify: `orchestrator/modes.py`
- Modify: `tests/test_modes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_modes.py`:

```python
from orchestrator.modes import FreeDebate


class TestFreeDebate:
    def test_init(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=10,
        )
        assert debate.max_exchanges == 10
        assert debate.exchange_count == 0
        assert debate.running is False

    def test_opening_round(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "Prof Peacegrave",
            "ironhelm": "Col Ironhelm",
        }
        mock_router = MagicMock()
        mock_router.generate.side_effect = ["Opening 1", "Opening 2"]

        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=10,
        )
        responses = list(debate.start("What is peace?"))
        assert len(responses) == 2
        assert debate.exchange_count == 2
        assert debate.running is True

    def test_next_turn_uses_judge(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "Prompt"
        mock_engine.get_scholar_names.return_value = {
            "peacegrave": "P", "ironhelm": "I",
        }
        mock_router = MagicMock()
        mock_router.generate.return_value = "Next response"

        debate = FreeDebate(
            scholar_ids=["peacegrave", "ironhelm"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=20,
        )
        debate.running = True
        debate.exchange_count = 2
        debate.conversation = [
            {"role": "user", "content": "Topic"},
            {"role": "assistant", "content": "R1"},
            {"role": "assistant", "content": "R2"},
        ]
        debate.turn_counts = {"peacegrave": 1, "ironhelm": 1}

        # Mock judge to select peacegrave
        with patch.object(debate.judge, 'select_next_speaker', return_value="peacegrave"):
            result = debate.next_turn()
            assert result is not None
            assert debate.exchange_count == 3

    def test_stops_at_max_exchanges(self):
        mock_engine = MagicMock()
        mock_engine.build_system_prompt.return_value = "P"
        mock_engine.get_scholar_names.return_value = {"peacegrave": "P"}
        mock_router = MagicMock()
        mock_router.generate.return_value = "R"

        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
            max_exchanges=2,
        )
        debate.running = True
        debate.exchange_count = 2
        debate.conversation = [{"role": "assistant", "content": "x"}] * 2
        debate.turn_counts = {"peacegrave": 2}

        with patch.object(debate.judge, 'select_next_speaker', return_value="peacegrave"):
            result = debate.next_turn()
            assert result is None
            assert debate.running is False

    def test_interjection(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        debate.running = True
        debate.add_interjection("What about gender?")
        assert debate.conversation[-1]["role"] == "user"

    def test_stop(self):
        mock_engine = MagicMock()
        mock_router = MagicMock()
        debate = FreeDebate(
            scholar_ids=["peacegrave"],
            engine=mock_engine,
            router=mock_router,
        )
        debate.running = True
        debate.stop()
        assert debate.running is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_modes.py::TestFreeDebate -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement FreeDebate**

Add to `orchestrator/modes.py`:

```python
class FreeDebate:
    """Mode 3: Scholars discuss organically with orchestrator balancing."""

    def __init__(self, scholar_ids: list[str], engine, router, max_exchanges: int = None):
        self.scholar_ids = scholar_ids
        self.engine = engine
        self.router = router
        self.judge = Judge(router=router)
        self.max_exchanges = max_exchanges or config.FREE_DEBATE_MAX_EXCHANGES
        self.conversation: list[dict] = []
        self.turn_counts: dict[str, int] = {sid: 0 for sid in scholar_ids}
        self.exchange_count = 0
        self.running = False
        self._names = engine.get_scholar_names()

    def start(self, topic: str):
        """Opening round: each scholar gives initial take. Yields responses."""
        self.conversation.append({"role": "user", "content": topic})
        self.running = True

        for sid in self.scholar_ids:
            name = self._names.get(sid, sid)
            try:
                prompt = self.engine.build_system_prompt(sid, mode="multi")
                meta_msg = (
                    f"Give your initial perspective on this topic. Be concise but substantive. "
                    f"You are in a panel with other scholars — make your distinctive contribution."
                )
                messages = self.conversation + [{"role": "user", "content": meta_msg}]
                response = self.router.generate(system_prompt=prompt, messages=messages)
                labeled = f"**{name}:**\n\n{response}"
                self.conversation.append({"role": "assistant", "content": labeled})
                self.turn_counts[sid] = self.turn_counts.get(sid, 0) + 1
                self.exchange_count += 1
                yield {"role": "assistant", "content": labeled}
            except Exception:
                skip_msg = f"*{name} is momentarily unavailable.*"
                self.conversation.append({"role": "assistant", "content": skip_msg})
                yield {"role": "assistant", "content": skip_msg}

            time.sleep(config.API_CALL_DELAY_SECONDS)

    def next_turn(self) -> dict | None:
        """Generate the next turn in the debate. Returns None if debate should stop."""
        if not self.running or self.exchange_count >= self.max_exchanges:
            self.running = False
            return None

        # Convergence check
        if (self.exchange_count > 0 and
                self.exchange_count % config.FREE_DEBATE_CONVERGENCE_CHECK_INTERVAL == 0):
            assistant_msgs = [m for m in self.conversation if m["role"] == "assistant"]
            if len(assistant_msgs) >= 3 and self.judge.check_convergence(assistant_msgs[-3:]):
                self.running = False
                return {
                    "role": "assistant",
                    "content": "*The discussion appears to be reaching consensus. Further perspectives may be needed to break new ground.*"
                }

        # Silence nudge: check if any scholar has been quiet for too long
        nudge_sid = None
        for sid in self.scholar_ids:
            if self.turn_counts.get(sid, 0) <= self.exchange_count / len(self.scholar_ids) - config.FREE_DEBATE_SILENCE_THRESHOLD:
                nudge_sid = sid
                break

        if nudge_sid:
            next_sid = nudge_sid
        else:
            next_sid = self.judge.select_next_speaker(
                scholars=self.scholar_ids,
                conversation=self.conversation,
                turn_counts=self.turn_counts,
                scholar_names=self._names,
            )

        name = self._names.get(next_sid, next_sid)

        # Determine if this is a nudged response
        if nudge_sid:
            meta = f"You've been quiet in this discussion. What does your theoretical framework make of the points raised so far?"
        else:
            meta = "Respond to the points that most engage your theoretical framework. Engage directly with what others have said."

        prompt = self.engine.build_system_prompt(next_sid, mode="multi")
        messages = self.conversation + [{"role": "user", "content": meta}]

        try:
            response = self.router.generate(system_prompt=prompt, messages=messages)
        except Exception:
            return {"role": "assistant", "content": f"*{name} is momentarily unavailable.*"}

        labeled = f"**{name}:**\n\n{response}"
        self.conversation.append({"role": "assistant", "content": labeled})
        self.turn_counts[next_sid] = self.turn_counts.get(next_sid, 0) + 1
        self.exchange_count += 1
        return {"role": "assistant", "content": labeled}

    def add_interjection(self, message: str):
        """Add user interjection to the conversation."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/test_modes.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add orchestrator/modes.py tests/test_modes.py
git commit -m "feat: Mode 3 free debate with speaker selection and convergence"
```

---

### Task 5: Update Gradio UI for All Three Modes

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Add smoke tests**

Add to `tests/test_app.py`:

```python
    def test_app_has_mode_selector(self):
        import app
        assert hasattr(app, "MODE_CHOICES")
        assert len(app.MODE_CHOICES) == 3
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Rewrite app.py with mode switching**

The updated `app.py` must:
- Add a `MODE_CHOICES` list: `["Private Consultation", "Panel Discussion", "Free Debate"]`
- Add a mode Radio selector in the sidebar
- Change scholar selector to `Dropdown` (Mode 1) or `CheckboxGroup` (Modes 2/3) based on mode
- For Mode 2: after initial round, show "Generate Rebuttals" button, then allow user follow-ups
- For Mode 3: add max_exchanges slider, "Start Debate" button, "Next Turn" button (or auto-advance), "Stop" button
- All multi-scholar responses prefixed with scholar name (already handled by modes.py)
- Keep the nordic minimalist theme and CSS

Key UI structure:

```python
MODE_CHOICES = ["Private Consultation", "Panel Discussion", "Free Debate"]

# Sidebar
mode_radio = gr.Radio(choices=MODE_CHOICES, value="Private Consultation", label="Mode")
scholar_dropdown = gr.Dropdown(...)      # Mode 1
scholar_checkboxes = gr.CheckboxGroup(...)  # Modes 2/3
max_exchanges_slider = gr.Slider(...)    # Mode 3 only
start_btn = gr.Button("Begin")
next_turn_btn = gr.Button("Next Turn")  # Mode 3
stop_btn = gr.Button("Stop Debate")     # Mode 3

# Show/hide controls based on mode
mode_radio.change(fn=switch_mode, ...)
```

State management:
```python
session_state = {
    "mode": None,  # "private" | "panel" | "debate"
    "consultation": None,  # PrivateConsultation instance
    "panel": None,  # PanelDiscussion instance
    "debate": None,  # FreeDebate instance
}
```

Mode 2 flow:
1. User selects scholars + types question + clicks Begin
2. `start_panel()` runs `panel.start_discussion()`, yielding responses to chatbot
3. "Generate Rebuttals" button appears → runs `panel.generate_rebuttals()`
4. After rebuttals, user can type follow-up → runs `panel.continue_discussion()`

Mode 3 flow:
1. User selects scholars + sets max exchanges + types topic + clicks Begin
2. `start_debate()` runs `debate.start()`, yielding opening responses
3. "Next Turn" button advances one turn via `debate.next_turn()`
4. User can type interjection anytime → `debate.add_interjection()` then `debate.next_turn()`
5. "Stop" button or max exchanges ends debate

IMPORTANT: Since Gradio processes events synchronously and we can't do true streaming of multi-turn debates, use a simpler UX:
- For opening rounds (Modes 2/3), collect all responses then display at once
- For Mode 3 "Next Turn", generate one response per click
- This avoids complex async/generator patterns in Gradio

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Manual test — launch locally**

Run: `cd /Users/kmini/Github/scholars-table && python app.py`
Test each mode:
- Mode 1: Select one scholar, chat
- Mode 2: Select 2-3 scholars, ask question, generate rebuttals
- Mode 3: Select 2-3 scholars, start debate, click Next Turn several times

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: Gradio UI with mode selector for all three discussion modes"
```

---

### Task 6: Push to HF Spaces and Verify

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/kmini/Github/scholars-table && python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Push to both remotes**

```bash
git push origin master
git push hf master:main
```

- [ ] **Step 3: Verify on HF Spaces**

Visit https://huggingface.co/spaces/KSvendsen/scholars-table
- Test Mode 1 (should still work as before)
- Test Mode 2 with 2-3 scholars
- Test Mode 3 with 2-3 scholars
