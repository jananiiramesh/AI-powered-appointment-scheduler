from agents.llms.singleton_ocr import get_ocr
import numpy as np
from flask import jsonify

class OCRProcessor:
    def __init__(self):
        self.ocr = get_ocr()

    def extract_text(self, image_array) -> dict:
        """
        Args:
            image_input: Can be PIL.Image, np.ndarray, or file path (str)
        """
        try:
            result = self.ocr.predict(image_array)
            page = result[0]

            boxes = (
                page.get("dt_polys")
                or page.get("det_polygons")
                or page.get("det_boxes")
            )

            if boxes is None:
                raise RuntimeError(
                    f"No detection boxes found. Available keys: {page.keys()}"
                )

            structured = [
                {
                    "text": text,
                    "confidence": float(score),
                    "box": box
                }
                for text, score, box in zip(
                    page.get("rec_texts", []),
                    page.get("rec_scores", []),
                    boxes
                )
            ]

            if not structured:
                return {
                    "raw_text": "",
                    "average_confidence": 0.0,
                    "structured": []
                }

            # Sort top-to-bottom, left-to-right
            structured.sort(
                key=lambda x: (x["box"][0][1], x["box"][0][0])
            )

            raw_text = "\n".join(item["text"] for item in structured)
            avg_conf = sum(item["confidence"] for item in structured) / len(structured)

            return {
                "raw_text": raw_text,
                "average_confidence": round(avg_conf, 4),
                "structured": structured
            }
        except Exception as e:
            return jsonify({"error":f"OCR Processing issue: {str(e)}"})
