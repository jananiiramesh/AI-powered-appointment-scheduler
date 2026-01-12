import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextExtractor:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def _extract_json(self, text: str) -> dict | None:
        """
        Extracts and parses the first valid JSON object from model output.
        Returns dict if successful, else None.
        """

        if not text:
            return None

        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            return None

        json_str = text[start:end + 1].strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


    def _build_prompt(self, ocr_text: str) -> str:
        return f"""
You are an information extraction system.

Extract ONLY appointment-related information.

APPOINTMENT DEFINITION:
An appointment is a scheduled meeting, visit, or consultation.

JSON SCHEMA (return EXACTLY only this):
{{
  "entities": {{
    "date_phrase": "<string or 'not specified'>",
    "time_phrase": "<string or 'not specified'>",
    "department": "<string or 'not specified'>"
  }},
  "entities_confidence": <number between 0 and 1>
}}

RULES:
- Do NOT hallucinate
- Use "not specified" if missing
- If multiple dates or times exist, extract ONLY the one related to the APPOINTMENT
- Ignore dates related to deadlines, shipping, dispatch, or tasks
- department = who or what the appointment is with (e.g., dentist, team)
- Output ONLY valid JSON
- Start the response with {{ and end with }}
- Do NOT add explanations or extra text

EXAMPLE:

Text:
"Book an appointment with the dentist at 5pm next Friday"

Output:
{{
  "entities": {{
    "date_phrase": "next Friday",
    "time_phrase": "5pm",
    "department": "dentist"
  }},
  "entities_confidence": 0.85
}}

NOW EXTRACT FROM THIS TEXT:

\"\"\"
{ocr_text}
\"\"\"
"""

    def extract(self, ocr_text: str) -> dict:
        prompt = self._build_prompt(ocr_text)

        messages = [
            {
                "role": "system",
                "content": prompt
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
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )

        generated_ids = output_ids[:, inputs.shape[-1]:]

        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        decoded = decoded.strip()
        result = self._extract_json(decoded)
        return result


extractor = TextExtractor()

text = """
I have a lot of work this week. But my tooth has been aching so much. I need to book an appointment to the dentist for next Wednesday at 10am.
"""

result = extractor.extract(text)
print(result)
