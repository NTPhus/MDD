"""Server-owned Qwen inference placement; acoustic models keep their CPU policy."""
from __future__ import annotations

import os


def requested_device(environment=None):
    environment = os.environ if environment is None else environment
    value = environment.get('MDD_CHAT_DEVICE', 'auto').strip().lower()
    if value not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('MDD_CHAT_DEVICE must be auto, cpu or cuda')
    return value


def select_runtime(torch, requested=None):
    requested = requested_device() if requested is None else requested_device({'MDD_CHAT_DEVICE': requested})

    def cpu(reason):
        return {'requested': requested, 'device': 'cpu', 'dtype': 'float32',
                'attention': 'eager', 'reason': reason}

    if requested == 'cpu':
        return cpu('cpu_requested')
    try:
        if not torch.cuda.is_available():
            if requested == 'cuda':
                raise RuntimeError('CUDA was requested but is unavailable')
            return cpu('cuda_unavailable')
        # Leave room for Windows, KV cache and adapter execution. Do not put
        # the acoustic models on CUDA or start/stop another GPU process.
        if requested == 'auto' and torch.cuda.mem_get_info(0)[0] < 2048 * 1024**2:
            return cpu('cuda_memory_low')
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        return {'requested': requested, 'device': 'cuda:0', 'dtype': dtype,
                'attention': 'sdpa', 'reason': 'cuda_available'}
    except (RuntimeError, OSError):
        if requested == 'cuda':
            raise
        return cpu('cuda_probe_failed')
