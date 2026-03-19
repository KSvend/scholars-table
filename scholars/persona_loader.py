# scholars/persona_loader.py
import os
import yaml

REQUIRED_TOP_LEVEL = ["name", "school", "title", "personality", "intellectual", "rhetorical"]
REQUIRED_PERSONALITY = ["background", "emotional_patterns", "humor_style", "when_wrong", "when_challenged"]
REQUIRED_INTELLECTUAL = ["core_concepts", "reasoning_pattern", "blind_spots", "internal_tensions",
                         "changed_mind_about", "borrowed_concepts"]
REQUIRED_RHETORICAL = ["sentence_style", "opening_move", "argument_method", "signature_phrases", "citation_style"]


def validate_persona(data: dict) -> list[str]:
    """Validate persona data against schema. Returns list of error strings (empty = valid)."""
    errors = []

    for field in REQUIRED_TOP_LEVEL:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "personality" in data:
        for field in REQUIRED_PERSONALITY:
            if field not in data["personality"]:
                errors.append(f"Missing personality.{field}")

    if "intellectual" in data:
        for field in REQUIRED_INTELLECTUAL:
            if field not in data["intellectual"]:
                errors.append(f"Missing intellectual.{field}")
        if "core_concepts" in data.get("intellectual", {}) and len(data["intellectual"]["core_concepts"]) == 0:
            errors.append("intellectual.core_concepts must not be empty")

    if "rhetorical" in data:
        for field in REQUIRED_RHETORICAL:
            if field not in data["rhetorical"]:
                errors.append(f"Missing rhetorical.{field}")

    return errors


def load_persona(filepath: str) -> dict:
    """Load and validate a single persona YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Persona file not found: {filepath}")

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    errors = validate_persona(data)
    if errors:
        raise ValueError(f"Invalid persona in {filepath}: {'; '.join(errors)}")

    return data


def load_all_personas(directory: str) -> dict[str, dict]:
    """Load all persona YAML files from a directory. Returns {scholar_id: persona_data}."""
    personas = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            scholar_id = os.path.splitext(filename)[0]
            filepath = os.path.join(directory, filename)
            personas[scholar_id] = load_persona(filepath)
    return personas
