from agents.ocr_processing import OCRProcessor
from agents.text_cleaner import TextCleaner
from agents.text_extractor_qwen import TextExtractor
from agents.text_normalizer import TextNormalizer
import numpy as np
import io
from PIL import Image

ocr_engine = OCRProcessor()
text_cleaner = TextCleaner()
entity_extractor = TextExtractor()
text_normalizer = TextNormalizer()

# text = "Book an appointment @ 5pm to the dentist"
image_path = "sample_inputs/test3.jpeg"

# 1. Read image bytes properly
with open(image_path, "rb") as f:
    image_bytes = f.read()

# 2. Load image using PIL
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# 3. Convert to NumPy array
try:
    image_array = np.array(image)
except Exception as e:
    print ({"error": f"NumPy conversion failed: {str(e)}"})


text = ocr_engine.extract_text(image_array)
# cleaned text
cleaned_text = text_cleaner.clean_to_json(text['raw_text'])
# extract appointment related entities
extracted_entities = entity_extractor.extract(cleaned_text['raw_text'])
# normalize entities to local time and date
normalized_entity = text_normalizer.normalize(extracted_entities)

print("CLEANED:", cleaned_text)
print("EXTRACTED:", extracted_entities)
print("NORMALIZED:", normalized_entity)

