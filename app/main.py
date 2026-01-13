from flask import Flask, request, jsonify
from agents.ocr_processing import OCRProcessor
from agents.text_cleaner import TextCleaner
from agents.text_extractor_qwen import TextExtractor
from agents.text_normalizer import TextNormalizer
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

ocr_engine = OCRProcessor()
text_cleaner = TextCleaner()
entity_extractor = TextExtractor()
text_normalizer = TextNormalizer()
print("agents have begun their work")

@app.route("/classify", methods=["POST"])
def classify_input():
    # if its text input:
    if request.content_type == "text/plain":
        text = request.get_data(as_text=True)


        # clean the text to remove any short forms, special character usages
        cleaned_text = text_cleaner.clean_to_json(text)
        # extract appointment related entities
        extracted_entities = entity_extractor.extract(cleaned_text['raw_text'])
        # normalize entities to local time and date
        normalized_entity = text_normalizer.normalize(extracted_entities)

        return jsonify(normalized_entity)

        
    elif "image" in request.files:
        image_file = request.files["image"]

        try:
            image_bytes = image_file.read()
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                return jsonify({"error": "Invalid image file, can't use PIL"}), 400

            image_array = np.array(image)
            ocr_text = ocr_engine.extract_text(image_array)
            
            # checking if OCR returned valid text
            if not ocr_text or 'raw_text' not in ocr_text:
                return jsonify({"error": "No text extracted from image"}), 400
            # clean extracted text
            cleaned_text = text_cleaner.clean_to_json(ocr_text['raw_text'])
            # extract appointment related entities
            extracted_entities = entity_extractor.extract(cleaned_text['raw_text'])
            # normalize entities
            normalized_entity = text_normalizer.normalize(extracted_entities)

            return jsonify(normalized_entity)
        
        except Exception as e:
            return jsonify({"error":"Invalid image"}), 400

    else:    
        return jsonify({"error": "Unsupported input type"}), 400

if __name__ == "__main__":
    app.run(debug=True)