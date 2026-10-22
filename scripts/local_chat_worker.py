"""CPU-only Qwen worker. JSONL IPC; model generation stays on its main thread."""
from __future__ import annotations

from contextlib import nullcontext
import faulthandler
import json
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generation_options(profile=None):
    """Keep structured tasks greedy; sample only ordinary conversation.

    Qwen's non-thinking recommendation:
    https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
    """
    if profile == "conversation":
        return {"do_sample": True, "temperature": .7, "top_p": .8, "top_k": 20, "min_p": 0.}
    if profile is not None:
        raise ValueError("Unsupported generation profile")
    return {"do_sample": False}


def bounded_inputs(tokenizer, messages, context, max_tokens):
    """Keep the system and current utterance while evicting complete old turns.

    A topic can begin with an authored assistant opener, before the first
    user/assistant pair. Evict that opener independently so a context limit
    never strands an old assistant answer without its corresponding user.
    ``dropped`` counts actual messages removed, including the opener.
    """
    history, dropped = list(messages), 0
    while True:
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cpu")
        if inputs["input_ids"].shape[-1] + max_tokens <= context:
            return inputs, dropped
        if len(history) <= 2:
            raise ValueError("chat_context_too_long")
        if history[1].get("role") == "assistant":
            del history[1]
            dropped += 1
        elif (len(history) >= 4 and history[1].get("role") == "user"
                and history[2].get("role") == "assistant"):
            del history[1:3]
            dropped += 2
        else:
            # Do not make room by deleting a current utterance or a fragment
            # whose boundaries are unknown. The service supplies valid turns.
            raise ValueError("chat_context_too_long")


class QwenChatEngine:
    def __init__(self):
        self.model = self.tokenizer = None
        self.adapter_key = None

    def load(self, trained):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is None:
            # Set thread pools once, before executing any tensor computation.
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            path = ROOT / "models/pretrained/Qwen3-0.6B"
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
        if trained:
            from peft import PeftModel
            path = ROOT / "runs/cascade-l2arctic-plus-seed42/Q/adapter"
            weight = path / "adapter_model.safetensors"
            key = (weight.stat().st_mtime_ns, weight.stat().st_size)
            if self.adapter_key != key:
                if isinstance(self.model, PeftModel):
                    self.model.delete_adapter("chat")
                    self.model.load_adapter(str(path), adapter_name="chat", is_trainable=False, local_files_only=True)
                else:
                    self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name="chat", is_trainable=False, local_files_only=True)
                self.adapter_key = key
                self.model.eval().requires_grad_(False)

    def generate(self, request, cancelled, emit):
        profile = request.get("generation_profile")
        sampling = generation_options(profile)
        import torch
        from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
        trained = request["model_id"] == "qwen-trained"
        self.load(trained)
        if cancelled.is_set():
            return {"finish_reason":"cancelled", "usage":{}}
        inputs, dropped = bounded_inputs(self.tokenizer, request["messages"],
            min(4096, self.model.config.max_position_embeddings), request["max_tokens"])
        length = inputs["input_ids"].shape[-1]
        emit({"type":"ready"})

        class Stream(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                if text and not cancelled.is_set():
                    emit({"type":"chunk", "text":text})

        class Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        mode = self.model.disable_adapter() if self.adapter_key is not None and not trained else nullcontext()
        started = time.monotonic()
        with mode, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=request["max_tokens"], max_time=request["timeout"],
                **sampling, pad_token_id=self.tokenizer.eos_token_id,
                streamer=Stream(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                stopping_criteria=StoppingCriteriaList([Cancel()]))
        ids = output[0, length:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        return {"finish_reason":"cancelled" if cancelled.is_set() else "stop" if len(ids) and int(ids[-1]) in eos_ids else "length",
                "usage":{"prompt_tokens":length,"completion_tokens":len(ids)}, "context_messages_dropped":dropped,
                "generation_profile": profile, "sampling": sampling,
                "latency_sec":round(time.monotonic() - started, 3)}


def main():
    # Windows native libraries must initialize before the stdin reader thread.
    # In particular, NumPy/OpenBLAS DLL initialization can block if first
    # imported while another Python thread is waiting on a Windows pipe.
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "2"
    faulthandler.dump_traceback_later(90, repeat=False)
    import numpy
    import torch
    from transformers import Qwen3ForCausalLM, AutoTokenizer, TextStreamer
    from peft import PeftModel
    import soundfile
    from scipy.signal import resample_poly
    from scripts.local_speech_engine import LocalSpeechEngine
    faulthandler.cancel_dump_traceback_later()
    queue, events, lock, output_lock = Queue(), {}, threading.Lock(), threading.Lock()
    def write(value):
        with output_lock:
            print(json.dumps(value, ensure_ascii=False), flush=True)
    def read():
        try:
            for line in sys.stdin:
                value = json.loads(line)
                if value.get("type") in {"generate", "speech", "regenerate_reference"}:
                    identifier = value["id"]
                    with lock:
                        event = events.setdefault(identifier, threading.Event())
                    queue.put((value, event))
                elif value.get("type") == "cancel":
                    with lock:
                        if value["id"] in events: events[value["id"]].set()
                elif value.get("type") == "shutdown":
                    break
        finally:
            with lock:
                for event in events.values(): event.set()
            queue.put(None)
    threading.Thread(target=read, daemon=True, name="chat-ipc-reader").start()
    engine = QwenChatEngine()
    speech_engine = LocalSpeechEngine()
    while True:
        item = queue.get()
        if item is None: break
        request, event = item
        identifier = request["id"]
        try:
            faulthandler.dump_traceback_later(300 if request["type"] in {'speech', 'regenerate_reference'} else 150, repeat=False)
            if request["model_id"] not in {"qwen-base", "qwen-trained"}:
                raise ValueError("invalid_model")
            emit = lambda value: write({"id":identifier, **value})
            if request['type'] == 'speech':
                metadata = speech_engine.run(request, event, emit, engine)
            elif request['type'] == 'regenerate_reference':
                metadata = speech_engine.regenerate(request, event, emit, engine)
            else:
                metadata = engine.generate(request, event, emit)
            write({"id":identifier,"type":"done","metadata":metadata})
        except Exception as error:
            write({"id":identifier,"type":"error","code":"chat_context_too_long" if str(error) == "chat_context_too_long" else "generation_failed"})
        finally:
            faulthandler.cancel_dump_traceback_later()
            with lock: events.pop(identifier, None)


if __name__ == "__main__":
    main()
