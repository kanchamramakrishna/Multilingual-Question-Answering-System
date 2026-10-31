"""Simple Gradio interface for the fine-tuned multilingual QA model."""
import gradio as gr

def create_interface(inference_engine):
    with gr.Blocks(title="Multilingual Question Answering System") as demo:
        gr.Markdown("# Multilingual Question Answering System\nFine-tuned locally on the supplied English, Hindi and Telugu data.")
        with gr.Tab("Ask a question"):
            language = gr.Radio(["English", "Hindi", "Telugu"], value="English", label="Question language")
            question = gr.Textbox(label="Question", lines=2, placeholder="Enter your question in the selected language")
            context = gr.Textbox(label="Context / passage", lines=8, placeholder="Paste the passage the answer should be based on")
            with gr.Row():
                ask = gr.Button("Generate answer", variant="primary")
                clear = gr.Button("Clear")
            answer = gr.Textbox(label="Generated answer", lines=4)
            details = gr.Markdown()
            ask.click(inference_engine.answer_question, inputs=[question, context, language], outputs=[answer, details])
            clear.click(lambda: ("", "", "", ""), outputs=[question, context, answer, details])
        with gr.Tab("About"):
            gr.Markdown(
                "### Model and training\n"
                "- Base model: `google/mt5-small` (downloaded automatically on first use)\n"
                "- Fine-tuning: LoRA adapter training\n"
                "- Supported data languages in the supplied archive: English, Hindi and Telugu\n"
                "- The Hindi and Telugu data includes unanswerable examples; those without a reference answer are excluded from supervised training/evaluation.\n\n"
                "**Note:** The supplied archive does not contain Malayalam data, so this version is not fine-tuned for Malayalam."
            )
    return demo
