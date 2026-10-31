"""Generate answers on the held-out test split and report simple EM/token F1."""
import argparse, json, os, re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from data_utils import load_split

ROOT = Path(__file__).resolve().parents[1]

def norm(s):
    return " ".join(re.findall(r"\w+", s.lower(), flags=re.UNICODE))
def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return float(p == g)
    common = {}
    for t in p: common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g:
        if common.get(t, 0): overlap += 1; common[t] -= 1
    if not overlap: return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ROOT / "models" / "multilingual_model"))
    ap.add_argument("--base-model", default="google/mt5-small")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-input-length", type=int, default=384)
    ap.add_argument("--max-target-length", type=int, default=96)
    args = ap.parse_args()
    os.chdir(ROOT)
    if not (ROOT / "data" / "Hindi" / "test.json").exists():
        import subprocess, sys
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "prepare_data.py")])
    model_path = Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_results = {}
    for lang in ["English", "Hindi", "Telugu"]:
        rows, _ = load_split("test", args.data_root, languages=[lang])
        if args.max_examples: rows = rows[:args.max_examples]
        ems, f1s, predictions = [], [], []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            enc = tokenizer([r["input_text"] for r in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_input_length).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_target_length, num_beams=2)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            for row, pred in zip(batch, preds):
                gold = row["target_text"]
                ems.append(float(norm(pred) == norm(gold)))
                f1s.append(token_f1(pred, gold))
                predictions.append({"language": lang, "prediction": pred, "reference": gold})
        all_results[lang] = {"examples": len(rows), "exact_match": sum(ems)/max(1,len(ems)), "token_f1": sum(f1s)/max(1,len(f1s))}
        print(lang, json.dumps(all_results[lang], ensure_ascii=False))
    outdir = ROOT / "logs"; outdir.mkdir(exist_ok=True)
    (outdir / "test_metrics.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Metrics saved to logs/test_metrics.json")
if __name__ == "__main__":
    main()
