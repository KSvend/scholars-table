import gradio as gr
from scholars.engine import ScholarEngine
from orchestrator.router import LLMRouter
from orchestrator.modes import PrivateConsultation

# Initialize
engine = ScholarEngine()
scholar_names = engine.get_scholar_names()
scholar_choices = {name: sid for sid, name in scholar_names.items()}

DISCLAIMER = (
    "This platform discusses conflict, violence, and political theory for "
    "educational purposes. The scholars are AI personas representing distinct "
    "theoretical traditions in International Relations and Conflict Transformation."
)


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
) as demo:
    gr.Markdown("# The Scholar's Table")
    gr.Markdown(
        "Ten scholars of International Relations and Conflict Transformation, "
        "each representing a distinct theoretical tradition. Choose one and begin."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=1):
            scholar_dropdown = gr.Dropdown(
                choices=list(scholar_choices.keys()),
                label="Select a Scholar",
                value=None,
            )
            reset_btn = gr.Button("New Conversation", variant="secondary")
            gr.Markdown("### Mode: Private Consultation")
            gr.Markdown("*1:1 conversation with your selected scholar.*")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
            )
            msg_input = gr.Textbox(
                label="Your message",
                placeholder="Ask a question or present a scenario...",
                lines=2,
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
    demo.launch(theme=gr.themes.Soft())
