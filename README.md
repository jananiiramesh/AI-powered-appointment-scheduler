### AI-powered-appointment-scheduler-assistant

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
---
## 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
---
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
Test API using Postman or any HTTP client.
