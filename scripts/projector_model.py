"""Frozen acoustic features -> two-layer GELU projector -> Qwen embeddings."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file


class AudioProjector(nn.Module):
    def __init__(self, audio_size=768, text_size=1024, hidden_size=1024):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(audio_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, text_size))

    def forward(self, features):
        return self.layers(features)


class ProjectedQwen(nn.Module):
    def __init__(self, llm, projector, chunk_size=128):
        super().__init__()
        self.llm, self.projector, self.chunk_size = llm, projector, chunk_size

    def base(self):
        return self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm

    def embeddings(self, features, prefix_ids, suffix_ids, target_ids=None):
        embedding = self.llm.get_input_embeddings()
        prefix, suffix = embedding(prefix_ids), embedding(suffix_ids)
        audio = self.projector(features.to(dtype=next(self.projector.parameters()).dtype)).to(prefix.dtype)
        parts = [prefix, audio, suffix]
        prompt_length = sum(part.shape[1] for part in parts)
        if target_ids is not None:
            parts.append(embedding(target_ids))
        combined = torch.cat(parts, dim=1)
        labels = torch.full(combined.shape[:2], -100, device=combined.device, dtype=torch.long)
        if target_ids is not None:
            labels[:, prompt_length:] = target_ids
        return combined, labels, prompt_length

    def forward(self, features, prefix_ids, suffix_ids, target_ids):
        inputs, labels, _ = self.embeddings(features, prefix_ids, suffix_ids, target_ids)
        base = self.base()
        # Deliberately retain autograd THROUGH the frozen decoder in stage 1.
        hidden = base.model(inputs_embeds=inputs, attention_mask=torch.ones(labels.shape, device=inputs.device, dtype=torch.long),
                            use_cache=False, return_dict=True).last_hidden_state
        shifted = labels[:, 1:]
        mask = shifted.ne(-100)
        selected, targets = hidden[:, :-1][mask], shifted[mask]
        if not len(targets):
            raise ValueError("Projector loss has no supervised response tokens")
        losses = []
        for start in range(0, len(targets), self.chunk_size):
            def projection_loss(values, truth):
                return F.cross_entropy(base.lm_head(values).float(), truth, reduction="sum")
            values, truth = selected[start:start + self.chunk_size], targets[start:start + self.chunk_size]
            losses.append(checkpoint(projection_loss, values, truth, use_reentrant=False)
                          if torch.is_grad_enabled() else projection_loss(values, truth))
        return torch.stack(losses).sum() / len(targets)

    @torch.inference_mode()
    def generate_response(self, features, prefix_ids, suffix_ids, *, eos_token_id, pad_token_id, **generation):
        inputs, _, prompt_length = self.embeddings(features, prefix_ids, suffix_ids)
        # Supplying matching dummy IDs makes the generation prefix length explicit.
        dummy = torch.full(inputs.shape[:2], pad_token_id, device=inputs.device, dtype=torch.long)
        output = self.llm.generate(input_ids=dummy, inputs_embeds=inputs,
                                   attention_mask=torch.ones_like(dummy), eos_token_id=eos_token_id,
                                   pad_token_id=pad_token_id, use_cache=True, **generation)
        return output[:, prompt_length:]

    def save_weights(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        save_file({key: value.detach().cpu().contiguous() for key, value in self.projector.state_dict().items()},
                  str(folder / "projector.safetensors"))
        if hasattr(self.llm, "peft_config"):
            self.llm.save_pretrained(folder / "adapter", safe_serialization=True)

    def load_weights(self, folder):
        self.projector.load_state_dict(load_file(str(folder / "projector.safetensors")))
        if hasattr(self.llm, "peft_config"):
            from peft import set_peft_model_state_dict
            result = set_peft_model_state_dict(self.llm, load_file(str(folder / "adapter/adapter_model.safetensors")))
            if result.unexpected_keys:
                raise ValueError("Unexpected adapter keys: " + str(result.unexpected_keys))


def build_model(config, stage, *, training, initial=None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from scripts.l2arctic_plus_data import rooted
    if not torch.cuda.is_available():
        raise RuntimeError("The configured full model training/evaluation requires CUDA")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = {"local_files_only": True, "torch_dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": 0}}
    if training:
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    llm = AutoModelForCausalLM.from_pretrained(str(rooted(config["model_path"])), **options)
    llm.config.use_cache = not training
    if training:
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        llm.requires_grad_(False)
    if stage == "feedback":
        if training:
            lora = config["lora"]
            llm = get_peft_model(llm, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"], target_modules=lora["target_modules"], task_type="CAUSAL_LM", bias="none"))
        else:
            llm = PeftModel.from_pretrained(llm, str(initial / "adapter"), is_trainable=False, local_files_only=True)
    projector = AudioProjector(config["audio_hidden_size"], config["text_hidden_size"], config["projector_hidden_size"]).cuda()
    result = ProjectedQwen(llm, projector, config["loss_chunk_size"])
    if initial:
        if training and stage == "feedback":
            projector.load_state_dict(load_file(str(initial / "projector.safetensors")))
        else:
            result.load_weights(initial)
    return result
