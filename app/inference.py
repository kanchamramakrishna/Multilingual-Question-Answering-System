"""Question answering generation for English, Hindi and Telugu."""
import torch


class QAInference:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def answer_question(self, question, context, language="English", max_length=96):
        question = (question or "").strip()
        context = (context or "").strip()

        if not context:
            return "Please enter a context passage.", ""
        if not question:
            return "Please enter a question.", ""
        if language not in {"English", "Hindi", "Telugu"}:
            language = "English"

        # This prompt intentionally matches the format used by training/data_utils.py.
        prompt = f"answer the question in {language}. question: {question} context: {context}"

        try:
            inputs = self.tokenizer(
                prompt,
                max_length=384,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            self.model.eval()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                )

            answer = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            ).strip()

            if not answer:
                return "The model did not generate an answer. Try a shorter context or rephrase the question.", ""

            info = (
                f"**Language:** {language}  ·  "
                "**Model:** mT5-small with your fine-tuned LoRA adapter"
            )
            return answer, info

        except Exception as exc:
            return f"Unable to generate an answer: {exc}", ""
