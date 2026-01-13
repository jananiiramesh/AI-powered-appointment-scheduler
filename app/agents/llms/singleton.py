import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

_model = None
_tokenizer = None
_device = None
_lock = threading.Lock()

def get_qwen(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    global _model, _tokenizer, _device

    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                _device = "cuda" if torch.cuda.is_available() else "cpu"

                _tokenizer = AutoTokenizer.from_pretrained(model_id)

                _model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
                    low_cpu_mem_usage=False
                ).to(_device)

                _model.eval()

    return _model, _tokenizer, _device, _lock
