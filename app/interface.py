"""Clean, minimal Gradio interface for the fine-tuned multilingual QA model."""
import gradio as gr


CUSTOM_CSS = """
:root {
  --bb-ink: #17212b;
  --bb-muted: #667085;
  --bb-line: #e5e9ef;
  --bb-accent: #287b68;
  --bb-accent-hover: #216756;
  --bb-surface: #ffffff;
  --bb-soft: #f7f9fa;
}
.gradio-container {
  max-width: 980px !important;
  margin: 0 auto !important;
  background: var(--bb-soft) !important;
  color: var(--bb-ink) !important;
}
#app-header {
  padding: 28px 30px 22px;
  border: 1px solid var(--bb-line);
  border-radius: 18px;
  background: var(--bb-surface);
  margin-bottom: 18px;
}
#app-header h1 {
  margin: 0 0 8px;
  font-size: 29px;
  font-weight: 650;
  letter-spacing: -0.7px;
  color: var(--bb-ink);
}
#app-header p {
  margin: 0;
  color: var(--bb-muted);
  font-size: 14px;
}
#input-panel, #answer-panel {
  border: 1px solid var(--bb-line);
  border-radius: 16px;
  background: var(--bb-surface);
  padding: 20px;
}
#generate-btn {
  background: var(--bb-accent) !important;
  border: 1px solid var(--bb-accent) !important;
  color: white !important;
  font-weight: 600 !important;
}
#generate-btn:hover {
  background: var(--bb-accent-hover) !important;
}
#clear-btn {
  border: 1px solid var(--bb-line) !important;
  background: white !important;
}
#answer-box textarea {
  font-size: 16px !important;
  line-height: 1.65 !important;
}
footer { display: none !important; }
@media (max-width: 640px) {
  #app-header { padding: 22px 18px; }
  #app-header h1 { font-size: 24px; }
  #input-panel, #answer-panel { padding: 14px; }
}
"""


def create_interface(inference_engine):
    """Build the UI and connect it to the loaded fine-tuned model."""

    def generate(question, context, language):
        # Keep the function signature aligned with QAInference.answer_question.
        return inference_engine.answer_question(
            question=question,
            context=context,
            language=language,
        )

    with gr.Blocks(
        title="BharatBhasha EduAI",
    ) as demo:
        gr.HTML(
            """
            <div id="app-header">
              <h1>BharatBhasha EduAI</h1>
              <p>Ask a question and get an answer grounded in the context you provide.</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, elem_id="input-panel"):
                gr.Markdown("### Ask your question")
                language = gr.Dropdown(
                    choices=["English", "Hindi", "Telugu"],
                    value="English",
                    label="Answer language",
                    info="Choose the language used for your question.",
                    interactive=True,
                )
                context = gr.Textbox(
                    label="Context",
                    placeholder="Paste the passage or information the answer should be based on...",
                    lines=8,
                    max_lines=14,
                    buttons=["copy"],
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="Type your question here...",
                    lines=2,
                    max_lines=4,
                )
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate answer",
                        variant="primary",
                        elem_id="generate-btn",
                        scale=3,
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="clear-btn",
                        scale=1,
                    )

            with gr.Column(scale=1, elem_id="answer-panel"):
                gr.Markdown("### Answer")
                answer = gr.Textbox(
                    label="Generated answer",
                    placeholder="Your answer will appear here...",
                    lines=9,
                    max_lines=14,
                    interactive=False,
                    buttons=["copy"],
                    elem_id="answer-box",
                )
                details = gr.Markdown(
                    "<span style='color:#667085'>Your answer will be generated using the fine-tuned multilingual model.</span>"
                )

        generate_btn.click(
            fn=generate,
            inputs=[question, context, language],
            outputs=[answer, details],
            show_progress="full",
        )
        clear_btn.click(
            fn=lambda: ("", "", "", "Your answer will appear here."),
            inputs=[],
            outputs=[context, question, answer, details],
            queue=False,
        )

        with gr.Accordion("About this model", open=False):
            gr.Markdown(
                """
                - **Base model:** `google/mt5-small`
                - **Fine-tuning:** LoRA adapter
                - **Languages available in this project:** English, Hindi and Telugu
                - **How to use:** Select a language, paste a context, enter a question, then generate an answer.

                The model is designed for context-based question answering. Its response may be incomplete or incorrect, so verify important information against the supplied passage.
                """
            )

    return demo
