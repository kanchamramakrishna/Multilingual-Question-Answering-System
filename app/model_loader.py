"""Load the locally fine-tuned mT5 LoRA adapter and its base model."""
import gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import try_to_load_from_cache
from peft import PeftModel

class ModelLoader:
    def __init__(self, model_path=None, base_model="google/mt5-small"):
        self.model_path = str(model_path or "models/multilingual_model")
        self.base_model_name = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        path = Path(self.model_path)
        if not path.exists() or not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Fine-tuned adapter not found at '{path}'. Run scripts/prepare_data.py, then training/train.py first."
            )
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        base_name = self.base_model_name
        config_file = path / "adapter_config.json"
        try:
            import json
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
            base_name = cfg.get("base_model_name_or_path") or base_name
        except Exception:
            pass
        cached_config = try_to_load_from_cache(base_name, "config.json")
        cache_has_config = (
            isinstance(cached_config, str) and Path(cached_config).is_file()
        )
        cached_weights = any(
            isinstance(cached_file, str) and Path(cached_file).is_file()
            for cached_file in (
                try_to_load_from_cache(base_name, "model.safetensors"),
                try_to_load_from_cache(base_name, "pytorch_model.bin"),
            )
        )
        base = AutoModelForSeq2SeqLM.from_pretrained(
            base_name,
            local_files_only=cache_has_config and cached_weights,
        )
        self.model = PeftModel.from_pretrained(base, str(path)).to(self.device)
        self.model.eval()
        print(f"Loaded base model {base_name} + LoRA adapter from {path} on {self.device}")
        return self.model, self.tokenizer

    def get_model_info(self):
        if self.model is None: return None
        return {"device": str(self.device), "model_path": self.model_path, "base_model": self.base_model_name}
