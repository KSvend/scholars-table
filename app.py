import gradio as gr
from scholars.engine import ScholarEngine
from orchestrator.router import LLMRouter
from orchestrator.modes import PrivateConsultation

# Initialize
engine = ScholarEngine()
scholar_names = engine.get_scholar_names()
scholar_choices = {name: sid for sid, name in scholar_names.items()}

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


def create_consultation(scholar_display_name: str):
    """Create a new PrivateConsultation for the selected scholar."""
    scholar_id = scholar_choices[scholar_display_name]
    router = LLMRouter(tier="free")
    return PrivateConsultation(scholar_id=scholar_id, engine=engine, router=router)


# State management
current_mode = {"consultation": None}


def select_scholar(scholar_name: str):
    """Handle scholar selection — reset conversation."""
    current_mode["consultation"] = create_consultation(scholar_name)
    scholar_id = scholar_choices[scholar_name]
    persona = engine.scholars[scholar_id]
    intro = (
        f"**{persona['name']}** — *{persona['school']}*\n\n"
        f"{persona['personality']['background'][:200].strip()}...\n\n"
        f"*Ask me anything about international relations, conflict, or peace.*"
    )
    return [{"role": "assistant", "content": intro}]


def chat(message: str, history: list):
    """Handle a chat message."""
    if current_mode["consultation"] is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please select a scholar first."},
        ], ""

    try:
        response = current_mode["consultation"].send_message(message)
    except Exception as e:
        response = f"*Scholar is momentarily unavailable. Please try again.* ({type(e).__name__})"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return history, ""


def reset_chat():
    """Clear the conversation."""
    if current_mode["consultation"]:
        current_mode["consultation"].reset()
    return []


# Build UI
with gr.Blocks(
    title="The Scholar's Table",
    theme=nordic_theme,
    css=CSS,
) as demo:
    gr.Markdown("# The Scholar's Table")
    gr.Markdown(
        "Scholars of International Relations and Conflict Transformation. "
        "Choose a tradition. Begin the dialogue.",
        elem_classes=["subtitle"],
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            scholar_dropdown = gr.Dropdown(
                choices=list(scholar_choices.keys()),
                label="Scholar",
                value=None,
            )
            reset_btn = gr.Button("Clear", variant="secondary", size="sm")
            gr.Markdown("Private consultation", elem_classes=["mode-label"])

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

    # Events
    scholar_dropdown.change(
        fn=select_scholar,
        inputs=[scholar_dropdown],
        outputs=[chatbot],
    )
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )
    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch()
