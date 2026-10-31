# Multilingual Question Answering System

This package adapts the original project to the dataset archive you supplied. It fine-tunes **`google/mt5-small` with LoRA** and provides a local Gradio app.

## Important dataset notes

The archive contains:
- English: multiple-choice science QA (train 4,957; validation 299; test 1,172 records)
- Hindi: SQuAD-style QA (train 118,516; validation 11,873; test 11,803 records)
- Telugu: SQuAD-style QA (train 118,516; validation 11,873; test 11,803 records)

Hindi/Telugu have some records with empty answer lists. The loader filters those out because they cannot provide a supervised target answer. English has answer choices and an answer key; the loader converts the correct key to the answer text and uses the available fact/options as context.

**There is no Malayalam data in the supplied archive.** The code currently fine-tunes English, Hindi and Telugu only. Add a Malayalam train/validation/test dataset and implement its data mapping before claiming Malayalam support.

## 1. Requirements

- Python 3.10 or 3.11 (recommended)
- 16 GB RAM recommended for the full dataset preprocessing
- NVIDIA GPU with around 8 GB+ VRAM recommended; smaller GPUs may require batch size 1 and more gradient accumulation. CPU training is possible but very slow.
- Free disk space: allow several GB for extracted data, model cache, tokenized datasets, checkpoints and outputs.
- Internet for the first download of `google/mt5-small` and Python packages.

## 2. Unzip the project and create a virtual environment

Open a terminal/PowerShell in the extracted project folder (the folder containing `requirements.txt`).

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build suited to your computer from https://pytorch.org/get-started/locally/ if the default pip install does not detect your NVIDIA GPU.

## 3. Extract the bundled dataset

The project ZIP contains `data_bundle/dataset_bundle.zip`. Extract it into the expected `data/` folder with:
```bash
python scripts/prepare_data.py
```
A folder `data` should be created
Data should be collected and stored in the `data` folder 
This expands the dataset archive and creates `data/English`, `data/Hindi`, and `data/Telugu`.
When we run `python scripts/prepare_data.py` then the `data` is setup into the `data_bundle` folder

## 4. Run a small smoke test first

This uses up to 200 usable examples per language for training and up to 200 per language for validation:
```bash
python training/train.py --limit-per-language 200 --epochs 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation 8
```
The smoke test still downloads the pretrained mT5-small model the first time. Check that it completes and writes adapter files under `models/multilingual_model/`.

To start fresh after a smoke test, remove `models/multilingual_model/` and `checkpoints/` before full training. If you want to continue training an existing adapter, that requires a deliberate resume workflow; do not simply delete it.

## 5. Full fine-tuning

A conservative starting command for a GPU:
```bash
python training/train.py --epochs 1 --batch-size 2 --eval-batch-size 2 --gradient-accumulation 8
```
This uses all answerable training records in the three languages. One epoch can take hours depending on GPU and disk speed. After a successful run, the adapter and tokenizer are saved in:
- `models/multilingual_model/` — final LoRA adapter + tokenizer/config
- `models/multilingual_model/checkpoint-*` or `checkpoints/` — trainer checkpoints are stored under the model output folder `models/multilingual_model/`
- `logs/` — test metrics are written here by the evaluation script

To reduce memory use, use `--batch-size 1 --eval-batch-size 1 --gradient-accumulation 16`. If you get CUDA out-of-memory, close other GPU applications and retry with smaller batches. The `--no-fp16` flag disables mixed precision if your hardware does not support it.

## 6. Evaluate on the held-out test split

Do this after training. Test data is not used to update weights:
```bash
python training/evaluate.py
```
For a quick check:
```bash
python training/evaluate.py --max-examples 100
```
The script writes `logs/test_metrics.json` with exact-match and token F1 per language. These are simple normalized metrics, not a comprehensive benchmark; inspect predictions too for quality.

## 7. Launch the application

After fine-tuning completes:
```bash
python app.py
```
Open http://127.0.0.1:7860 in your browser. Paste a question and its context in the selected language.

## Model files and downloads

- Base pretrained weights (`google/mt5-small`) download automatically into the Hugging Face cache on first run. They are not copied into the project directory by default.
- Fine-tuned LoRA adapter and tokenizer are saved in `models/multilingual_model/`.
- The adapter is not a complete standalone model: keep access to the base model cache, or download the base model again on another computer.
- Do not expect model files to exist before training succeeds.

## Troubleshooting

- **`Dataset file not found`**: run `python scripts/prepare_data.py` from the project root.
- **`Fine-tuned adapter not found`**: finish training first; check for `models/multilingual_model/adapter_config.json`.
- **Model download error**: verify internet access and Hugging Face availability; first download requires internet.
- **CUDA out of memory**: batch size 1, gradient accumulation 16, close other GPU jobs. Full training on CPU is very slow.
- **Dependency errors**: use Python 3.10/3.11 in a fresh virtual environment and reinstall requirements.
