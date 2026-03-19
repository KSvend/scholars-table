import gradio as gr
from scholars.engine import ScholarEngine
from orchestrator.router import LLMRouter
from orchestrator.modes import PrivateConsultation, PanelDiscussion, FreeDebate

# Initialize
engine = ScholarEngine()
scholar_names = engine.get_scholar_names()
scholar_choices = {name: sid for sid, name in scholar_names.items()}

MODE_CHOICES = ["Private Consultation", "Panel Discussion", "Free Debate"]

MODE_DESCRIPTIONS = {
    "Private Consultation": "One-on-one dialogue with a single scholar.",
    "Panel Discussion": "Multiple scholars respond to your question, then rebut each other.",
    "Free Debate": "Organic debate among scholars, guided by an orchestrator.",
}

# Nordic minimalist theme
nordic_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#FAFAF8",
    body_text_color="#2C2C2C",
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E8E6E1",
    block_shadow="none",
    block_title_text_color="#2C2C2C",
    block_label_text_color="#6B6B6B",
    input_background_fill="#FAFAF8",
    input_border_color="#D4D2CD",
    input_border_width="1px",
    button_primary_background_fill="#2C2C2C",
    button_primary_text_color="#FAFAF8",
    button_secondary_background_fill="#FAFAF8",
    button_secondary_border_color="#D4D2CD",
    button_secondary_text_color="#2C2C2C",
    border_color_primary="#E8E6E1",
    shadow_spread="0px",
)

CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}
h1 {
    font-weight: 300 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 1.4rem !important;
    color: #2C2C2C !important;
    border-bottom: 1px solid #E8E6E1 !important;
    padding-bottom: 12px !important;
    margin-bottom: 8px !important;
}
.subtitle {
    color: #8A8A8A !important;
    font-size: 0.85rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 24px !important;
}
.disclaimer {
    color: #AAAAAA !important;
    font-size: 0.7rem !important;
    font-weight: 300 !important;
    border-top: 1px solid #E8E6E1 !important;
    padding-top: 12px !important;
    margin-top: 24px !important;
}
.mode-label {
    color: #8A8A8A !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}
"""


def create_router():
    return LLMRouter(tier="free")


# Session state
session_state = {
    "mode": "private",
    "consultation": None,
    "panel": None,
    "debate": None,
}


def _reset_state():
    """Reset all mode instances."""
    session_state["consultation"] = None
    session_state["panel"] = None
    session_state["debate"] = None


# --- Mode switching ---

def on_mode_change(mode_name):
    """Handle mode radio change — show/hide controls, reset state."""
    _reset_state()
    is_private = mode_name == "Private Consultation"
    is_panel = mode_name == "Panel Discussion"
    is_debate = mode_name == "Free Debate"

    session_state["mode"] = (
        "private" if is_private else "panel" if is_panel else "debate"
    )

    return (
        # scholar_dropdown visible
        gr.update(visible=is_private),
        # scholar_checkboxes visible
        gr.update(visible=(is_panel or is_debate)),
        # max_exchanges_slider visible
        gr.update(visible=is_debate),
        # begin_btn visible
        gr.update(visible=(is_panel or is_debate)),
        # rebuttals_btn visible
        gr.update(visible=False),
        # next_turn_btn visible
        gr.update(visible=False),
        # stop_btn visible
        gr.update(visible=False),
        # clear chatbot
        [],
        # mode description
        MODE_DESCRIPTIONS.get(mode_name, ""),
    )


# --- Mode 1: Private Consultation ---

def select_scholar(scholar_name):
    """Handle scholar selection — create consultation, show intro."""
    if not scholar_name:
        return []
    scholar_id = scholar_choices[scholar_name]
    router = create_router()
    session_state["consultation"] = PrivateConsultation(
        scholar_id=scholar_id, engine=engine, router=router
    )
    persona = engine.scholars[scholar_id]
    intro = (
        f"**{persona['name']}** — *{persona['school']}*\n\n"
        f"{persona['personality']['background'][:200].strip()}...\n\n"
        f"*Ask me anything about international relations, conflict, or peace.*"
    )
    return [{"role": "assistant", "content": intro}]


# --- Mode 2: Panel Discussion ---

def start_panel(selected_scholars, question, history):
    """Start a panel discussion: all scholars respond to the question."""
    if not selected_scholars or len(selected_scholars) < 2:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please select at least 2 scholars."},
        ], gr.update(visible=False), ""
    if not question.strip():
        return history, gr.update(visible=False), ""

    scholar_ids = [scholar_choices[name] for name in selected_scholars]
    router = create_router()
    panel = PanelDiscussion(scholar_ids=scholar_ids, engine=engine, router=router)
    session_state["panel"] = panel

    history = history + [{"role": "user", "content": question}]
    for msg in panel.start_discussion(question):
        history = history + [{"role": "assistant", "content": msg["content"]}]

    return history, gr.update(visible=True), ""


def generate_rebuttals(history):
    """Generate rebuttals for the current panel discussion."""
    panel = session_state.get("panel")
    if panel is None or panel.phase != "rebuttals":
        return history, gr.update(visible=False)

    for msg in panel.generate_rebuttals():
        history = history + [{"role": "assistant", "content": msg["content"]}]

    return history, gr.update(visible=False)


# --- Mode 3: Free Debate ---

def start_debate(selected_scholars, topic, max_exchanges, history):
    """Start a free debate: opening round from each scholar."""
    if not selected_scholars or len(selected_scholars) < 2:
        return history + [
            {"role": "user", "content": topic},
            {"role": "assistant", "content": "Please select at least 2 scholars."},
        ], gr.update(visible=False), gr.update(visible=False), ""
    if not topic.strip():
        return history, gr.update(visible=False), gr.update(visible=False), ""

    scholar_ids = [scholar_choices[name] for name in selected_scholars]
    router = create_router()
    debate = FreeDebate(
        scholar_ids=scholar_ids, engine=engine, router=router,
        max_exchanges=int(max_exchanges),
    )
    session_state["debate"] = debate

    history = history + [{"role": "user", "content": topic}]
    for msg in debate.start(topic):
        history = history + [{"role": "assistant", "content": msg["content"]}]

    return history, gr.update(visible=True), gr.update(visible=True), ""


def next_turn(history):
    """Advance the free debate by one turn."""
    debate = session_state.get("debate")
    if debate is None or not debate.running:
        return history, gr.update(visible=False), gr.update(visible=False)

    msg = debate.next_turn()
    if msg is None:
        return history + [
            {"role": "assistant", "content": "*The debate has concluded.*"}
        ], gr.update(visible=False), gr.update(visible=False)

    history = history + [{"role": "assistant", "content": msg["content"]}]
    return history, gr.update(visible=True), gr.update(visible=True)


def stop_debate(history):
    """Stop the free debate."""
    debate = session_state.get("debate")
    if debate:
        debate.stop()
    return history + [
        {"role": "assistant", "content": "*Debate stopped.*"}
    ], gr.update(visible=False), gr.update(visible=False)


# --- Unified message handler ---

def chat(message, history):
    """Handle message input across all modes."""
    if not message.strip():
        return history, ""

    mode = session_state["mode"]

    if mode == "private":
        consultation = session_state.get("consultation")
        if consultation is None:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please select a scholar first."},
            ], ""
        try:
            response = consultation.send_message(message)
        except Exception as e:
            response = f"*Scholar is momentarily unavailable. Please try again.* ({type(e).__name__})"
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ], ""

    elif mode == "panel":
        panel = session_state.get("panel")
        if panel is None or panel.phase not in ("open", "rebuttals"):
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please start a panel discussion first."},
            ], ""
        history = history + [{"role": "user", "content": message}]
        result = panel.continue_discussion(message)
        if result:
            history = history + [{"role": "assistant", "content": result["content"]}]
        return history, ""

    elif mode == "debate":
        debate = session_state.get("debate")
        if debate is None or not debate.running:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please start a debate first."},
            ], ""
        debate.add_interjection(message)
        history = history + [{"role": "user", "content": message}]
        msg = debate.next_turn()
        if msg:
            history = history + [{"role": "assistant", "content": msg["content"]}]
        return history, ""

    return history, ""


def reset_chat():
    """Clear conversation and reset state."""
    _reset_state()
    return []


# --- Build UI ---

with gr.Blocks(
    title="The Scholar's Table",
) as demo:
    gr.Markdown("# The Scholar's Table")
    gr.Markdown(
        "Scholars of International Relations and Conflict Transformation. "
        "Choose a tradition. Begin the dialogue.",
        elem_classes=["subtitle"],
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            mode_selector = gr.Radio(
                choices=MODE_CHOICES,
                value="Private Consultation",
                label="Mode",
            )

            scholar_dropdown = gr.Dropdown(
                choices=list(scholar_choices.keys()),
                label="Scholar",
                value=None,
                visible=True,
            )

            scholar_checkboxes = gr.CheckboxGroup(
                choices=list(scholar_choices.keys()),
                label="Scholars",
                visible=False,
            )

            max_exchanges_slider = gr.Slider(
                minimum=5, maximum=30, step=1, value=20,
                label="Max Exchanges",
                visible=False,
            )

            begin_btn = gr.Button("Begin", variant="primary", visible=False)
            rebuttals_btn = gr.Button(
                "Generate Rebuttals", variant="secondary", visible=False
            )
            next_turn_btn = gr.Button("Next Turn", variant="secondary", visible=False)
            stop_btn = gr.Button("Stop", variant="stop", visible=False)

            reset_btn = gr.Button("Clear", variant="secondary", size="sm")

            mode_description = gr.Markdown(
                MODE_DESCRIPTIONS["Private Consultation"],
                elem_classes=["mode-label"],
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                show_label=False,
            )
            msg_input = gr.Textbox(
                label="Message",
                placeholder="Pose a question or scenario...",
                lines=1,
                show_label=False,
            )

    gr.Markdown(
        "This platform discusses conflict, violence, and political theory "
        "for educational purposes. Scholars are AI personas.",
        elem_classes=["disclaimer"],
    )

    # --- Events ---

    # Mode switching
    mode_selector.change(
        fn=on_mode_change,
        inputs=[mode_selector],
        outputs=[
            scholar_dropdown,
            scholar_checkboxes,
            max_exchanges_slider,
            begin_btn,
            rebuttals_btn,
            next_turn_btn,
            stop_btn,
            chatbot,
            mode_description,
        ],
    )

    # Mode 1: scholar selection
    scholar_dropdown.change(
        fn=select_scholar,
        inputs=[scholar_dropdown],
        outputs=[chatbot],
    )

    # Begin button — dispatches based on current mode
    def on_begin(selected_scholars, question, max_exchanges, history):
        mode = session_state["mode"]
        if mode == "panel":
            hist, rebut_vis, cleared = start_panel(
                selected_scholars, question, history
            )
            return hist, rebut_vis, gr.update(visible=False), gr.update(visible=False), cleared
        elif mode == "debate":
            hist, next_vis, stop_vis, cleared = start_debate(
                selected_scholars, question, max_exchanges, history
            )
            return hist, gr.update(visible=False), next_vis, stop_vis, cleared
        return history, gr.update(), gr.update(), gr.update(), question

    begin_btn.click(
        fn=on_begin,
        inputs=[scholar_checkboxes, msg_input, max_exchanges_slider, chatbot],
        outputs=[chatbot, rebuttals_btn, next_turn_btn, stop_btn, msg_input],
    )

    # Generate rebuttals (Mode 2)
    rebuttals_btn.click(
        fn=generate_rebuttals,
        inputs=[chatbot],
        outputs=[chatbot, rebuttals_btn],
    )

    # Next turn (Mode 3)
    next_turn_btn.click(
        fn=next_turn,
        inputs=[chatbot],
        outputs=[chatbot, next_turn_btn, stop_btn],
    )

    # Stop debate (Mode 3)
    stop_btn.click(
        fn=stop_debate,
        inputs=[chatbot],
        outputs=[chatbot, next_turn_btn, stop_btn],
    )

    # Message input
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    # Clear
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch(theme=nordic_theme, css=CSS)
