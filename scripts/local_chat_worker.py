"""Local Qwen worker. JSONL IPC; model generation stays on its main thread."""
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
sys.path.insert(0, str(ROOT / 'src'))


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


def bounded_inputs(tokenizer, messages, context, max_tokens, device='cpu'):
    """Keep the system and current utterance while evicting complete old turns.

    A topic can begin with an authored assistant opener, before the first
    user/assistant pair. Evict that opener independently so a context limit
    never strands an old assistant answer without its corresponding user.
    ``dropped`` counts actual messages removed, including the opener.
    """
    history, dropped = list(messages), 0
    while True:
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
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
    def __init__(self, device='cpu'):
        self.model = self.tokenizer = None
        self.adapter_key = None
        self.adapter_keys = {}
        self.device = 'cpu'
        self.runtime = {}
        self.requested_device = device
        self._threads_set = False

    def load(self, trained):
        import hashlib
        if trained not in (False, True, 'qwen-base', 'qwen-trained', 'qwen-grammar'):
            raise ValueError('invalid_model')
        name = 'grammar' if trained == 'qwen-grammar' else 'chat' if trained is True or trained == 'qwen-trained' else None
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model is None:
            from mdd.chat_runtime import select_runtime
            # Set thread pools once, before executing any tensor computation.
            if not self._threads_set:
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
                self._threads_set = True
            path = ROOT / "models/pretrained/Qwen3-0.6B"
            self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            self.runtime = select_runtime(torch, self.requested_device)
            self.device = self.runtime['device']
            try:
                self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                    trust_remote_code=False, dtype=getattr(torch, self.runtime['dtype']),
                    attn_implementation=self.runtime['attention'], device_map={'': self.device})
            except torch.cuda.OutOfMemoryError:
                if self.runtime['requested'] != 'auto' or self.device == 'cpu':
                    raise
                import gc
                self.model = None
                gc.collect()
                torch.cuda.empty_cache()
                self.runtime = {**select_runtime(torch, 'cpu'), 'requested': 'auto', 'reason': 'cuda_load_oom'}
                self.device = 'cpu'
                self.model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation='eager', device_map={'': 'cpu'})
            self.model.eval().requires_grad_(False)
        if name:
            from peft import PeftModel
            if name == 'grammar':
                from mdd.demo_model_registry import grammar_selection
                selected = grammar_selection(ROOT)
                if not selected['ready']: raise ValueError('Grammar adapter is not validated')
                path, expected = selected['adapter'], selected['sha256']
            else:
                run = ROOT / 'runs/cascade-l2arctic-plus-seed42/Q'
                path = run / 'adapter'
                expected = json.loads((run / 'training_provenance.json').read_text(encoding='utf-8'))['adapter_sha256']
            weight = path / "adapter_model.safetensors"
            key = (weight.stat().st_mtime_ns, weight.stat().st_size, expected)
            if self.adapter_keys.get(name) != key:
                if hashlib.sha256(weight.read_bytes()).hexdigest() != expected:
                    raise ValueError('Adapter checksum mismatch')
                if isinstance(self.model, PeftModel):
                    if name in self.model.peft_config: self.model.delete_adapter(name)
                    self.model.load_adapter(str(path), adapter_name=name, is_trainable=False, local_files_only=True,
                                            torch_device=self.device, device_map={'': self.device})
                else:
                    self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name, is_trainable=False, local_files_only=True,
                                                            torch_device=self.device, device_map={'': self.device})
                self.model.to(self.device)
                self.adapter_keys[name] = key
                self.adapter_key = key
            self.model.set_adapter(name)
            self.model.requires_grad_(False)
            self.model.eval().requires_grad_(False)

    def generate(self, request, cancelled, emit):
        profile = request.get("generation_profile")
        sampling = generation_options(profile)
        import torch
        from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
        trained = request["model_id"] != "qwen-base"
        loading_started = time.monotonic()
        self.load(request['model_id'])
        loading_sec = round(time.monotonic() - loading_started, 3)
        if cancelled.is_set():
            return {"finish_reason":"cancelled", "usage":{}}
        inputs, dropped = bounded_inputs(self.tokenizer, request["messages"],
            min(4096, self.model.config.max_position_embeddings), request["max_tokens"], device=self.device)
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
                "latency_sec":round(time.monotonic() - started, 3),
                "generation_latency_sec":round(time.monotonic() - started, 3),
                "model_load_sec":loading_sec, "qwen_runtime": dict(self.runtime)}


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
                if value.get("type") in {"generate", "speech", "regenerate_reference", "warmup"}:
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
    from mdd.chat_runtime import requested_device
    engine = QwenChatEngine(device=requested_device())
    speech_engine = LocalSpeechEngine()
    import atexit
    atexit.register(speech_engine.close)
    while True:
        item = queue.get()
        if item is None: break
        request, event = item
        identifier = request["id"]
        try:
            faulthandler.dump_traceback_later(300 if request["type"] in {'speech', 'regenerate_reference', 'warmup'} else 150, repeat=False)
            if request["model_id"] not in {"qwen-base", "qwen-trained", "qwen-grammar"}:
                raise ValueError("invalid_model")
            emit = lambda value: write({"id":identifier, **value})
            if request['type'] == 'warmup':
                metadata = speech_engine.warmup(event, engine)
            elif request['type'] == 'speech':
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
    speech_engine.close()


if __name__ == "__main__":
    main()
