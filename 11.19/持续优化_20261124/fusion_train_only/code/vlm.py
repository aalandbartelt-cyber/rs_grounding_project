"""Offline quantized model loaders and deterministic generation."""
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import (AutoProcessor, AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration,
                          Qwen3VLForConditionalGeneration)

from common import ROOT


def quantization(skip):
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=skip)


class Qwen:
    def __init__(self):
        path = ROOT / "weights/Qwen3-VL-4B"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True,
                                                       min_pixels=65536, max_pixels=262144)
        self.model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
            path, local_files_only=True, device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quantization(["visual", "lm_head"]),
            output_loading_info=True, attn_implementation="sdpa")
        if loading["missing_keys"] or loading["unexpected_keys"] or loading["mismatched_keys"]:
            raise ValueError(f"Qwen parameter loading is incomplete: {loading}")
        if self.model.get_output_embeddings().weight is not self.model.get_input_embeddings().weight:
            raise ValueError("Qwen's shared input/output embeddings were not tied")
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

    @torch.inference_mode()
    def answer(self, prompt, image=None, max_tokens=128):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        batch = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to("cuda")
        output = self.model.generate(**batch, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
        return self.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def match_score(self, prompt, image):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": prompt}]}]
        batch = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to("cuda")
        logits = self.model(**batch, logits_to_keep=1).logits[0, -1].float()
        yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        no = self.tokenizer.encode("No", add_special_tokens=False)
        if len(yes) != 1 or len(no) != 1:
            raise ValueError("Expected one-token Yes/No answers")
        return float((logits[yes[0]] - logits[no[0]]).cpu())


class GeoGround:
    def __init__(self):
        path = ROOT / "weights/GeoGround/llava-v1.5-7b-task-lora-geoground"
        original = json.loads((path / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
        text_keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "rms_norm_eps",
                     "max_position_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
                     "rope_theta", "tie_word_embeddings", "attention_bias")
        text = LlamaConfig(**{key: original[key] for key in text_keys if key in original})
        vision = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                                  num_attention_heads=16, image_size=336, patch_size=14,
                                  hidden_act="quick_gelu", layer_norm_eps=1e-5)
        # Token 0 acts as the valid in-vocabulary image placeholder. Its embedding is replaced.
        config = LlavaConfig(text_config=text, vision_config=vision, image_token_index=0,
                             image_seq_length=576, vision_feature_layer=-2,
                             vision_feature_select_strategy="default", projector_hidden_act="gelu")
        mapping = {
            r"^model\.vision_tower\.vision_tower\.": "model.vision_tower.",
            r"^model\.mm_projector\.0\.": "model.multi_modal_projector.linear_1.",
            r"^model\.mm_projector\.2\.": "model.multi_modal_projector.linear_2.",
            r"^model\.(layers|embed_tokens|norm)\.": r"model.language_model.\1.",
        }
        self.model, loading = LlavaForConditionalGeneration.from_pretrained(
            path, config=config, local_files_only=True, key_mapping=mapping, output_loading_info=True,
            device_map={"": 0}, dtype=torch.bfloat16,
            quantization_config=quantization(["vision_tower", "multi_modal_projector"]),
            attn_implementation="sdpa")
        if loading["missing_keys"] or loading["unexpected_keys"]:
            raise ValueError(f"GeoGround parameter conversion is incomplete: {loading}")
        self.model.eval()
        self.processor = CLIPImageProcessor(
            size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336},
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711])

    @torch.inference_mode()
    def answer(self, description, image):
        prefix = ("A chat between a curious human and an artificial intelligence assistant. "
                  "The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ")
        suffix = "\n[refer] output the bounding box of the <ref>" + description + "</ref> in the image. ASSISTANT:"
        before = self.tokenizer(prefix, return_tensors="pt")["input_ids"]
        after = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat((before, torch.zeros(1, 576, dtype=torch.long), after), dim=1).cuda()
        pixels = self.processor(images=image.resize((336, 336)), return_tensors="pt")["pixel_values"]
        output = self.model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            pixel_values=pixels.cuda().to(torch.bfloat16), do_sample=False, max_new_tokens=96,
            use_cache=True, pad_token_id=0)
        return self.tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
