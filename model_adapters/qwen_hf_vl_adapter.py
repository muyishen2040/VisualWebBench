import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from model_adapters import BaseAdapter
from utils.constants import *

class QwenHFVLAdapter(BaseAdapter):
    """
    Hugging Face-native adapter for Qwen/Qwen2-VL-* models.
    Expects (model, processor) in the constructor.
    """
    def __init__(self, model: AutoModelForCausalLM, processor: AutoProcessor, **kwargs):
        # BaseAdapter wants something tokenizer-like; processor.tokenizer works
        super().__init__(model, processor.tokenizer if hasattr(processor, "tokenizer") else None)
        self.processor = processor
        # You can override with YAML (optional)
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 512))
        self.do_sample = bool(kwargs.get("do_sample", False))

    def _build_inputs(self, query: str, image: Image.Image):
      image = image.convert("RGB")
      if hasattr(self.processor, "apply_chat_template") and hasattr(self.processor, "chat_template"):
          # For Qwen2-VL, LLaVA-HF, etc.
          messages = [
              {
                  "role": "user",
                  "content": [
                      {"type": "image"},
                      {"type": "text", "text": query},
                  ],
              }
          ]
          chat_text = self.processor.apply_chat_template(
              messages, tokenize=False, add_generation_prompt=True
          )
          return self.processor(text=[chat_text], images=[image], return_tensors="pt")
      else:
          # For Phi-3-Vision (needs <|image|> tag in the text)
          if "<|image_1|>" not in query:
            query = "<|image_1|>\n" + query
          return self.processor(text=query, images=[image], return_tensors="pt")



    def generate(self, query: str, image: Image.Image, task_type: str) -> str:
        inputs = self._build_inputs(query, image).to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=False,
            )
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        # Task-specific post-processing (mirrors your LlavaAdapter behavior)
        if task_type == CAPTION_TASK:
            m = re.search(r'<meta name="description" content="(.*?)">', text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else text

        if task_type in [WEBQA_TASK, ELEMENT_OCR_TASK]:
            # Keep only the part after the first colon if present
            if ":" in text:
                text = ":".join(text.split(":")[1:])
            return text.strip().strip('"').strip("'")

        if task_type == ACTION_PREDICTION_TASK:
            # If model returns e.g. "a" or "A:", normalize to one uppercase letter
            return text[:1].upper()

        return text
