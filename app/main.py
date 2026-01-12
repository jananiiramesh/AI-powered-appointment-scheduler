from flask import Flask, request, jsonify
from tools.image_to_text import OCRTextCleaner
from tools.text_extractor_qwen import TextExtractor
from tools.text_normalizer import TextNormalizer
from PIL import Image
import io

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify_input():
    if request.content_type == "text/plain":
        text = request.get_data(as_text=True)
        # testing code
        # return jsonify({
        #     "input_type": "raw text",
        #     "content": text
        # })
        
    elif "image" in request.files:
        image_file = request.files["image"]

        try:
            image = Image.open(io.BytesIO(image_file.read()))
            return jsonify({
                "input_type":"image",
                "image_format": image.format,
                "image_size": image.size
            })
        
        except Exception as e:
            return jsonify({"error":"Invalid image"}), 400
        
    return jsonify({"error": "Unsupported input type"}), 400

if __name__ == "__main__":
    app.run(debug=True)