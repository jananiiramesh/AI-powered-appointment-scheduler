import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from paddleocr import PaddleOCR

class OCRTextCleaner:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def _build_prompt(self, ocr_text: str) -> str:
        return f"""
You are a text normalization system.

Your task:
- Correct spelling mistakes
- Fix grammatical errors
- Expand short forms (e.g., appt → appointment, tmrw → tomorrow)
- Remove unnecessary line breaks
- Remove extra spaces
- Keep meaning EXACTLY the same
- Output everything in ONE SINGLE LINE

RULES:
- Do NOT summarize
- Do NOT add new information
- Do NOT explain
- Output ONLY the corrected text
- No quotes, no markdown

INPUT TEXT:
\"\"\"
{ocr_text}
\"\"\"
"""

    # def clean(self, ocr_text: str) -> str:
    #     prompt = self._build_prompt(ocr_text)

    #     messages = [
    #         {
    #             "role": "system",
    #             "content": prompt
    #         }
    #     ]

    #     inputs = self.tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=True,
    #         add_generation_prompt=False,
    #         return_tensors="pt"
    #     ).to(self.model.device)

    #     with torch.no_grad():
    #         output_ids = self.model.generate(
    #             inputs,
    #             max_new_tokens=200,
    #             do_sample=False,
    #             eos_token_id=self.tokenizer.eos_token_id
    #         )

    #     generated_ids = output_ids[:, inputs.shape[-1]:]

    #     cleaned_text = self.tokenizer.batch_decode(
    #         generated_ids,
    #         skip_special_tokens=True
    #     )[0].strip()

    #     return cleaned_text

    def clean(self, ocr_text: str) -> str:
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
                "content": ocr_text
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

text = """"Book an
Appointment
wit
Pedeatician"""

ocr_cleaner = OCRTextCleaner()
print(ocr_cleaner.clean(text))

# ocr = PaddleOCR(
#     lang="en",
#     use_textline_orientation=True
# )

# cleaner = OCRTextCleaner()

# result = ocr.predict("test3.jpeg")

# page = result[0]

# boxes = (
#     page.get("dt_polys")
#     or page.get("det_polygons")
#     or page.get("det_boxes")
# )

# if boxes is None:
#     raise RuntimeError(f"No detection boxes found. Keys: {page.keys()}")

# structured = [
#     {
#         "text": t,
#         "confidence": float(s),
#         "box": b
#     }
#     for t, s, b in zip(
#         page["rec_texts"],
#         page["rec_scores"],
#         boxes
#     )
# ]

# # Sort top-to-bottom, left-to-right
# structured.sort(key=lambda x: (x["box"][0][1], x["box"][0][0]))

# # Raw OCR text (with line breaks)
# raw_text = "\n".join(item["text"] for item in structured)

# avg_conf = sum(item["confidence"] for item in structured) / len(structured)

# print("----- RAW OCR TEXT -----")
# print(raw_text)

# # ✅ Clean using LLM
# cleaned_text = cleaner.clean(raw_text)

# print("\n----- CLEANED TEXT (ONE LINE) -----")
# print(cleaned_text)

# print(f"\nAverage OCR confidence: {avg_conf:.3f}")

