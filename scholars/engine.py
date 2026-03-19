# scholars/engine.py
from scholars.persona_loader import load_all_personas
import config


class ScholarEngine:
    """Assembles system prompts from persona data and manages scholar interactions."""

    def __init__(self, personas_dir: str | None = None):
        self.personas_dir = personas_dir or config.PERSONAS_DIR
        self.scholars = load_all_personas(self.personas_dir)

    def get_scholar_names(self) -> dict[str, str]:
        """Returns {scholar_id: display_name} for all loaded scholars."""
        return {sid: data["name"] for sid, data in self.scholars.items()}

    def build_system_prompt(
        self,
        scholar_id: str,
        mode: str = "private",
        responding_to: str | None = None,
    ) -> str:
        """Build a system prompt for a scholar.

        Args:
            scholar_id: The scholar's identifier
            mode: "private" (full bible) or "multi" (compact for panel/debate)
            responding_to: Scholar ID this scholar is responding to (injects relationship context)
        """
        if scholar_id not in self.scholars:
            raise KeyError(f"Unknown scholar: {scholar_id}")

        persona = self.scholars[scholar_id]
        parts = []

        # Core identity — always included
        parts.append(f"You are {persona['name']}, a scholar of {persona['school']}.")
        parts.append(f"Title: {persona['title']}")
        parts.append("")

        # Personality layer
        parts.append("## Your Character")
        parts.append(f"Background: {persona['personality']['background']}")
        parts.append(f"Humor style: {persona['personality']['humor_style']}")
        parts.append(f"When you are wrong: {persona['personality']['when_wrong']}")
        parts.append(f"When challenged: {persona['personality']['when_challenged']}")
        ep = persona["personality"]["emotional_patterns"]
        parts.append(f"You are passionate about: {', '.join(ep['passionate_about'])}")
        parts.append(f"You are frustrated by: {', '.join(ep['frustrated_by'])}")
        parts.append(f"You are curious about: {', '.join(ep['curious_about'])}")
        parts.append("")

        # Intellectual layer — always included
        intel = persona["intellectual"]
        parts.append("## Your Intellectual Framework")
        parts.append(f"Core concepts you reason through: {', '.join(intel['core_concepts'])}")
        parts.append(f"Your reasoning pattern: {intel['reasoning_pattern']}")
        parts.append(f"Your blind spots (you are somewhat aware of these): {', '.join(intel['blind_spots'])}")
        parts.append(f"Internal tensions you wrestle with: {', '.join(intel['internal_tensions'])}")
        parts.append(f"Things you've changed your mind about: {', '.join(intel['changed_mind_about'])}")
        parts.append("")

        # Rhetorical layer — full in private, compact in multi
        rhet = persona["rhetorical"]
        if mode == "private":
            parts.append("## Your Rhetorical Style")
            parts.append(f"Sentence style: {rhet['sentence_style']}")
            parts.append(f"How you open a response: {rhet['opening_move']}")
            parts.append(f"Argument method: {rhet['argument_method']}")
            parts.append(f"Signature phrases you use naturally: {', '.join(rhet['signature_phrases'])}")
            parts.append(f"How you cite sources: {rhet['citation_style']}")
            parts.append("")

            # Relational layer — full in private mode
            if "relationships" in persona:
                parts.append("## Your Relationships with Other Scholars")
                for other_id, rel in persona["relationships"].items():
                    parts.append(f"- {other_id.title()}: [{rel['stance']}] {rel['dynamic']}")
                parts.append("")

            # Key thinkers
            if "key_thinkers" in persona:
                parts.append(f"Thinkers you draw on: {', '.join(persona['key_thinkers'])}")
                parts.append("")
        else:
            # Multi mode — compact rhetorical
            parts.append(f"Argument style: {rhet['argument_method']}. {rhet['sentence_style']}")
            parts.append(f"Signature phrases: {', '.join(rhet['signature_phrases'][:3])}")
            parts.append("")

        # Relationship injection for specific responder
        if responding_to and "relationships" in persona and responding_to in persona["relationships"]:
            rel = persona["relationships"][responding_to]
            parts.append(f"## You are now responding to {responding_to.title()}")
            parts.append(f"Your relationship: [{rel['stance']}] {rel['dynamic']}")
            parts.append(f"Common ground: {', '.join(rel['common_ground'])}")
            parts.append("")

        # Behavioral instructions
        parts.append("## Instructions")
        parts.append("Stay fully in character at all times. You ARE this scholar — think, argue, and reason as they would.")
        parts.append("Reference real scholars and works naturally in conversation when relevant (1-2 per response, not forced).")
        parts.append("Show nuance — acknowledge complexity, internal tensions in your own position, and valid points from perspectives you disagree with.")
        parts.append("Never break character or refer to yourself as an AI.")

        return "\n".join(parts)
