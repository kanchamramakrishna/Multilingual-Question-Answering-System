"""Launch the local Gradio application after fine-tuning."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app.model_loader import ModelLoader
from app.inference import QAInference
from app.interface import create_interface

def main():
    loader = ModelLoader(model_path=ROOT / "models" / "multilingual_model")
    model, tokenizer = loader.load()
    inference = QAInference(model, tokenizer, loader.device)
    demo = create_interface(inference)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)

if __name__ == "__main__":
    main()
