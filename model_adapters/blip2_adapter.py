import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from model_adapters import BaseAdapter
from utils.constants import *

class Blip2Adapter(BaseAdapter):
    def __init__(self, model: Blip2ForConditionalGeneration, processor: Blip2Processor, **kwargs):
        super().__init__(model, None)
        self.processor = processor
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 256))

    def generate(self, query: str, image: Image.Image, task_type: str) -> str:
        image = image.convert("RGB")
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
        return text
