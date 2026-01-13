import torch
import difflib
from agents.llms.singleton import get_qwen


class TextCleaner:
    def __init__(self):
        self.model, self.tokenizer, self.device, self.lock = get_qwen()

    def clean(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a text cleaning system. "
                    "Correct spelling, grammar, expand short forms, "
                    "remove extra spaces and line breaks, "
                    "and output everything in one single line. "
                    "Do not add or remove information. "
                    "Do not summarize the text."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = inputs.to(self.device)

        with self.lock, torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=200,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = output_ids[:, inputs.shape[-1]:]

        cleaned_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return cleaned_text

    def clean_to_json(self, text: str) -> dict:
        cleaned_text = self.clean(text)

        similarity = difflib.SequenceMatcher(
            None,
            text.lower(),
            cleaned_text.lower()
        ).ratio()

        return {
            "raw_text": cleaned_text,
            "confidence": round(similarity, 2)
        }
