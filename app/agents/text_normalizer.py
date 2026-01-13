import torch
import json
from datetime import datetime
from agents.llms.singleton import get_qwen


class TextNormalizer:
    def __init__(self):
        # 🔹 shared singleton model
        self.model, self.tokenizer, self.device, self.lock = get_qwen()

    def _extract_json(self, text: str) -> dict | None:
        if not text:
            return None

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            return None

        json_str = text[start:end + 1].strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def build_prompt(self, date_phrase: str, time_phrase: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        day = datetime.now().strftime("%A")

        return f"""
You are an information normalization system.

Your ONLY task is to normalize date and time expressions.
You must NOT reinterpret, infer intent, or change meaning.

CONTEXT:
- Local timezone: Asia/Kolkata (IST, UTC+05:30)
- Today's date (reference) : {today}
- Today's day (reference) : {day}

NORMALIZATION RULES:
- Convert date_phrase to ISO format: YYYY-MM-DD
- Convert time_phrase to 24-hour format: HH:MM
- Resolve relative dates (e.g., today, tomorrow, next Friday) using the reference date
- If the date cannot be uniquely resolved, return "Ambiguous date"
- If the time cannot be uniquely resolved, return "Ambiguous time"
- If mentioned afternoon or evening, consider as PM. If mentioned morning, consider as AM.
- Do NOT guess missing AM/PM
- Do NOT shift dates across timezones
- Do NOT change or add information
- The "normalization_confidence" field should represent the confidence in your answer

OUTPUT FORMAT:
Return ONLY valid JSON.
Do NOT include explanations, markdown, or any extra text.

JSON SCHEMA (return EXACTLY this):
{{
  "normalized": {{
    "date": "<YYYY-MM-DD or 'Ambiguous date'>",
    "time": "<HH:MM or 'Ambiguous Time'>",
    "tz": "Asia/Kolkata"
  }},
  "normalization_confidence": <float between 0 and 1>
}}

NOW NORMALIZE THE FOLLOWING:
date_phrase: "{date_phrase}"
time_phrase: "{time_phrase}"
"""

    def normalize(self, extracted_info: dict) -> dict:
        entities = extracted_info.get("entities", {})

        date_phrase = entities.get("date_phrase")
        time_phrase = entities.get("time_phrase")
        department = entities.get("department")

        error = {
            "status": "needs_clarification",
            "message": "Ambiguous date/time or department"
        }

        if department == "not specified":
            return error

        prompt = self.build_prompt(date_phrase, time_phrase)

        messages = [
            {
                "role": "system",
                "content": prompt
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
                max_new_tokens=120,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = output_ids[:, inputs.shape[-1]:]

        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        result = self._extract_json(decoded)
        if not result:
            return error

        normalized = result["normalized"]
        if (
            normalized["date"].lower() == "ambiguous date"
            or normalized["time"].lower() == "ambiguous time"
        ):
            return error

        return result
