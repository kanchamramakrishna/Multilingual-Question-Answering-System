"""LoRA fine-tuning for the supplied English, Hindi and Telugu QA datasets."""
import argparse
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
)
from peft import LoraConfig, TaskType, get_peft_model
from data_utils import load_split

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DEFAULT = "google/mt5-small"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=BASE_MODEL_DEFAULT)
    parser.add_argument("--output-dir", default=str(ROOT / "models" / "multilingual_model"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-input-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--limit-per-language", type=int, default=0, help="Debug/smoke test; 0 uses all usable records.")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not (ROOT / "data" / "Hindi" / "train.json").exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_data.py")])
    set_seed(args.seed)
    train_rows, train_counts = load_split("train", args.data_root, limit_per_language=args.limit_per_language)
    val_rows, val_counts = load_split("validation", args.data_root, limit_per_language=max(0, min(args.limit_per_language, 1000)) if args.limit_per_language else 0)
    print("Train counts:", train_counts)
    print("Validation counts:", val_counts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q", "v"], bias="none"
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    def tokenize(batch):
        x = tokenizer(batch["input_text"], max_length=args.max_input_length, truncation=True)
        y = tokenizer(text_target=batch["target_text"], max_length=args.max_target_length, truncation=True)
        x["labels"] = y["input_ids"]
        return x
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names, desc="Tokenizing train")
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names, desc="Tokenizing validation")
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=args.logging_steps,
        predict_with_generate=False,
        generation_max_length=args.max_target_length,
        fp16=use_fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        dataloader_num_workers=0,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8 if use_fp16 else None)
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tokenizer
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    # Save explicit metadata for the app and future reproducibility.
    (output / "bharatbhasha_training_config.json").write_text(
        __import__("json").dumps({
            "base_model": args.model_name, "languages": ["English", "Hindi", "Telugu"],
            "method": "LoRA", "epochs": args.epochs, "train_counts": train_counts,
            "validation_counts": val_counts, "note": "Adapter weights; base model is also required at inference."
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFine-tuning finished. Adapter and tokenizer saved to: {output}")
    print(f"Base model: {args.model_name} (cached separately by Hugging Face)")

if __name__ == "__main__":
    main()
