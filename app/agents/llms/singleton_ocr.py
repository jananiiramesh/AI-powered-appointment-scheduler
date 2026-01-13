from paddleocr import PaddleOCR
import threading

_ocr_instance = None
_lock = threading.Lock()

def get_ocr():
    global _ocr_instance

    if _ocr_instance is None:
        with _lock:
            if _ocr_instance is None: 
                _ocr_instance = PaddleOCR(
                    lang="en",
                    use_textline_orientation=True
                )

    return _ocr_instance
