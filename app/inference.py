"""Question answering generation for English, Hindi and Telugu."""
import torch

class QAInference:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device

    def answer_question(self, question, context, language="English", max_length=96):
        if not question or not question.strip() or not context or not context.strip():
            return "Please provide both a question and context.", ""
        if language not in {"English", "Hindi", "Telugu"}:
            language = "English"
        prompt = f"answer the question in {language}. question: {question.strip()} context: {context.strip()}"
        try:
            inputs = self.tokenizer(prompt, max_length=384, truncation=True, return_tensors="pt").to(self.device)
            self.model.eval()
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_length, num_beams=4, early_stopping=True)
            answer = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            confidence = "Low"
            if answer and answer.casefold() in context.casefold(): confidence = "High"
            elif len(answer.split()) >= 2: confidence = "Medium"
            info = f"**Language:** {language}  \n**Answer length:** {len(answer.split())} words  \n**Confidence heuristic:** {confidence}  \n**Model:** mT5-small + LoRA"
            return answer or "No answer generated.", info
        except Exception as exc:
            return f"Error generating answer: {exc}", ""
