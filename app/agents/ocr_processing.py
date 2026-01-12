from paddleocr import PaddleOCR
from text_cleaner import TextCleaner
## import os


class OCRProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True
        )

    def extract_text(self, image_path: str) -> dict:
        result = self.ocr.predict(image_path)
        page = result[0]

        boxes = (
            page.get("dt_polys")
            or page.get("det_polygons")
            or page.get("det_boxes")
        )

        if boxes is None:
            raise RuntimeError(f"No detection boxes found. Keys: {page.keys()}")

        structured = [
            {
                "text": t,
                "confidence": float(s),
                "box": b
            }
            for t, s, b in zip(
                page["rec_texts"],
                page["rec_scores"],
                boxes
            )
        ]

        # Sort top-to-bottom, left-to-right
        structured.sort(key=lambda x: (x["box"][0][1], x["box"][0][0]))

        raw_text = "\n".join(item["text"] for item in structured)
        avg_conf = sum(item["confidence"] for item in structured) / len(structured)

        return {
            "raw_text": raw_text,
            "average_confidence": avg_conf,
            "structured": structured
        }


# unit testing code for ocr  
# ocr = OCRProcessor()
# cleaner = TextCleaner()

# image_path = os.path.join(
#     os.path.dirname(__file__), 
#     "..",                      
#     "sample_inputs",
#     "test3.jpeg"
# )

# ocr_output = ocr.extract_text(image_path)
# cleaned_text = cleaner.clean(ocr_output["raw_text"])

# print(cleaned_text)
# works as expected