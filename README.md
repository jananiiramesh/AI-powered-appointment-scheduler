# AI-powered-appointment-scheduler-assistant

This project is a Flask-based API that accepts **image and text inputs** from users, extracts appointment or meeting-related information, and normalizes it into **local ISO date/time formats and local time zones**.  
The system is designed as a foundational component for a larger, intelligent appointment scheduling platform.

---

## 🛠️ Setup Instructions

Follow the steps to run the project
## 1️⃣ Clone the Repository
```bash
git clone "https://github.com/jananiiramesh/AI-powered-appointment-scheduler"
cd AI-powered-appointment-scheduler
```
## 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
## 3️⃣ Install all dependencies
```bash
pip install -r requirements.txt
```
## 4️⃣ Download and Configure ngrok
Create an account, download and install ngrok on your system. Then authenticate ngrok using your auth token
```bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```
## 5️⃣ Run the Flask Application
```bash
cd app
python main.py
```
## 6️⃣ Expose the API using ngrok
```bash
ngrok http 5000
```
## 7️⃣ Test the API
Test the API using Postman or any http client by sending requests to the ngrok generated URL (since the api endpoint is '/appointment', add it to the url). Sample input images are given in app/sample_inputs.

---

## System and Architecture

The system depends particularly on 4 modules
- OCRProcessor
- TextCleaner
- TextExtractor
- TextNormalizer

The working pipeline has been illustrated in the figure 
![Architecture Diagram](app/Architecture.png)

# Brief summary on the separate modules:
- OCRProcessor:
The OCRProcessor module is responsible for performing optical character recognition (OCR) on image inputs represented as NumPy arrays. It acts as a wrapper around a singleton OCR engine and converts low-level OCR predictions into a structured, readable format. The key highlights are the usage of a singleton (shared) OCR instance. The PaddleOCR module (known for being able to handle noisy images) has been used.

- TextCleaner:
The TextCleaner module is responsible for cleaning raw text (decoding short forms, special characters, etc) using a large language model. It improves OCR or user-provided text quality while preserving the original meaning. Uses a singleton Qwen language model for consistent and efficient inference. Ensures no information is added, removed, or summarized.

- TextExtractor
The TextExtractor module is responsible for structured information extraction from cleaned OCR or user-provided text. It uses a large language model to identify appointment-related entities and return them in a strict JSON format. Uses a singleton Qwen language model to ensure consistent and efficient inference. Extracts only appointment-relevant information: Date phrase, Time phrase, Department or meeting context

- TextNormalizer
The TextNormalizer module is responsible for converting extracted appointment date and time phrases into precise, machine-readable formats. It transforms natural language expressions into ISO-standard date and 24-hour time values, grounded in the local timezone. Uses a shared singleton Qwen language model for deterministic and consistent normalization. Normalizes relative and absolute date/time phrases using a fixed reference: Local timezone: Asia/Kolkata (IST), Current date and day as context

---
## Key Architecture Highlights
### Singleton Architecture for Model Management
This project uses a singleton architecture for all large machine learning models (OCR and LLMs) to ensure high performance, memory efficiency, and system stability.
Each heavy model (OCR engine and language models) is loaded exactly once per process and then shared across all pipeline components, thus saving memory and reducing overhead.

### JSON schema and Guardrails
The outputs of all the three text processing modules are in strict json format making it easy to process and proceed to next step. A key exit condition has been defined in the TextNormalizer component, where insufficient date/time/department details leads to an exit condition (alerting the user to provide sufficient details).

---
### Sample Postman Requests
ngrok url: https://harry-wealthy-cajolingly.ngrok-free.dev/appointment
Trying putting text like "Book dermatalogist appointment at 6pm today" keeping body settings as "raw" and "text". Send a POST request
Add images with appointment related text using "form-data" option with value type set as "file". Send a POST request with image.






