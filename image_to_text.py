from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)

result = ocr.predict("test4.jpeg")

page = result[0]

# which one works in this? I should check :)
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

structured.sort(key=lambda x: (x["box"][0][1], x["box"][0][0]))

final_text = "\n".join(item["text"] for item in structured)

avg_conf = sum(item["confidence"] for item in structured) / len(structured)

print("----- FINAL TEXT -----")
print(final_text)
print(f"\nAverage OCR confidence: {avg_conf:.3f}")



