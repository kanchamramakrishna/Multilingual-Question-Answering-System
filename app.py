"""Launch the local Gradio application after fine-tuning."""
from pathlib import Path
import sys
<<<<<<< HEAD
import gradio as gr
=======
>>>>>>> 0cdebc9e97207b8fce2d9ef34862d436bc6a45a4
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app.model_loader import ModelLoader
from app.inference import QAInference
<<<<<<< HEAD
from app.interface import CUSTOM_CSS, create_interface
=======
from app.interface import create_interface
>>>>>>> 0cdebc9e97207b8fce2d9ef34862d436bc6a45a4

def main():
    loader = ModelLoader(model_path=ROOT / "models" / "multilingual_model")
    model, tokenizer = loader.load()
    inference = QAInference(model, tokenizer, loader.device)
    demo = create_interface(inference)
<<<<<<< HEAD
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CUSTOM_CSS,
    )
=======
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
>>>>>>> 0cdebc9e97207b8fce2d9ef34862d436bc6a45a4

if __name__ == "__main__":
    main()
