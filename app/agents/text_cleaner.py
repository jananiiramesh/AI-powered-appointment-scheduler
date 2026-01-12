import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextCleaner:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def clean(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a text normalization system. "
                    "Correct spelling, grammar, expand short forms, "
                    "remove extra spaces and line breaks, "
                    "and output everything in one single line. "
                    "Do not add or remove information."
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
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
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

# unit testing code 
# cleaner = TextCleaner()
# user_text = "Book an appt wit Pedeatician tmrw @ 5pm. Do not forget to buy vegetables tmrw."
# cleaned_text = cleaner.clean(user_text)

# print(cleaned_text)
# status ok
