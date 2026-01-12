from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def build_prompt(ocr_text):
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
- If multiple dates or times are mentioned, extract ONLY the one that refers to the ACTUAL MEETING
- Ignore dates related to deadlines, dispatches, or non-meeting events
- If a day is mentioned without a date, keep it as text
- entities_confidence must reflect overall confidence
- Output MUST be raw JSON only
- Start the response with {{ and end with }}
- Do NOT add explanations or extra text

    OCR TEXT:
    \"\"\"
    {ocr_text}
    \"\"\"
"""

final_text= """As discussed, we need to schedule the dispatch of the latest batch of goods by Friday atleast. I'd like to discuss with you and Aditi       
accordingly. Please do schedule a meeting with the team at around 3 in the afternoon tomorrow regarding the same. Please do keep this as you   
highest priority task.
Thank you"""

prompt = build_prompt(final_text)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=False
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(response)