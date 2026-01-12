### After we receive the extracted text/text from the previous endpoint, we need to clean it
# extract relevant parts
# remove whatever isn't related to the appointment booking part
# extract the fine details of the appointment
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

class AppointmentTextExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def build_prompt(self, ocr_text: str) -> str:
        return f"""
    You are an information extraction system.

    The text below was extracted using OCR and may contain spelling mistakes,
    broken lines, or minor grammatical errors.

    Your job is to understand the meaning of the text and extract
    appointment-related information only.

    TASK:
    Extract appointment-related entities and return ONLY valid JSON.

    JSON SCHEMA (follow exactly):
    {{
  "entities": {{
    "date_phrase": "<string or 'not specified'>",
    "time_phrase": "<string or 'not specified'>",
    "department": "<string or 'not specified'>"
  }},
  "entities_confidence": <number between 0 and 1>
    }}

    RULES:
    - Do NOT hallucinate information
    - If an entity is missing or unclear, use "not specified"
    - Correct obvious OCR spelling or grammar errors
    - If a day is mentioned without a date, keep the day as text
    - If a date can be inferred, use format DD-MM-YYYY
    - entities_confidence must reflect how confident you are overall
    - Output MUST be raw JSON only
    - Do NOT add explanations, markdown, or extra text

    OCR TEXT:
    \"\"\"
    {ocr_text}
    \"\"\"
    """

    
    def extract(self, ocr_text:str) -> dict:
        prompt = self.build_prompt(ocr_text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        print(">>> LLM: starting generation")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False
            )

        print(">>> LLM: generation finished")

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(response)

        json_start = response.find("{")
        json_output = response[json_start:]

        return json.loads(json_output)
