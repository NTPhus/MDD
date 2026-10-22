"""Bounded, server-only storage of CTC posteriors for reference reassessment."""
from __future__ import annotations

import base64
import binascii
import math
import re
import zlib

import numpy as np

SCHEMA = 'mdd-private-ctc-evidence-v1'
MAX_FRAMES = 2000  # wav2vec2 produces about 50 frames/s for the 30 s demo limit.
MAX_VOCAB = 256
MAX_BYTES = MAX_FRAMES * MAX_VOCAB * 4


def pack_evidence(evidence):
    matrix = np.asarray(evidence['log_probs'], dtype='<f4')
    if (matrix.ndim != 2 or not 1 <= matrix.shape[0] <= MAX_FRAMES
            or not 2 <= matrix.shape[1] <= MAX_VOCAB or not np.isfinite(matrix).all()):
        raise ValueError('Invalid bounded CTC evidence')
    packed = {'schema': SCHEMA, 'shape': list(matrix.shape),
              'data': base64.b64encode(zlib.compress(matrix.tobytes(), level=1)).decode('ascii'),
              'id_to_phone': {str(key): value for key, value in evidence['id_to_phone'].items()},
              'blank_id': evidence['blank_id'], 'duration_sec': evidence['duration_sec'],
              'checkpoint_sha256': evidence['checkpoint_sha256'],
              'vocabulary_sha256': evidence['vocabulary_sha256']}
    # Validate the same constraints used when a stored recording is reassessed.
    unpack_evidence(packed)
    return packed


def unpack_evidence(packed):
    try:
        if not isinstance(packed, dict) or packed.get('schema') != SCHEMA:
            raise ValueError('Unsupported CTC evidence')
        shape = packed['shape']
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int for size in shape)
                or not 1 <= shape[0] <= MAX_FRAMES or not 2 <= shape[1] <= MAX_VOCAB):
            raise ValueError('Invalid CTC shape')
        size = shape[0] * shape[1] * 4
        data = packed['data']
        if not isinstance(data, str) or len(data) > (MAX_BYTES + 65536) * 2:
            raise ValueError('Oversized CTC evidence')
        compressed = base64.b64decode(data, validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid compressed CTC evidence')
        matrix = np.frombuffer(raw, dtype='<f4').reshape(shape)
        mapping = packed['id_to_phone']
        if not isinstance(mapping, dict) or len(mapping) != shape[1]:
            raise ValueError('Invalid phone vocabulary')
        mapping = {int(key): value for key, value in mapping.items()}
        if (set(mapping) != set(range(shape[1]))
                or any(not isinstance(phone, str) or not 1 <= len(phone) <= 16 for phone in mapping.values())
                or type(packed['blank_id']) is not int or packed['blank_id'] not in mapping
                or not np.isfinite(matrix).all() or matrix.max() > 1e-4
                or not math.isfinite(packed['duration_sec']) or not .25 <= packed['duration_sec'] <= 30):
            raise ValueError('Invalid CTC metadata')
        for key in ('checkpoint_sha256', 'vocabulary_sha256'):
            if not isinstance(packed[key], str) or not re.fullmatch('[a-f0-9]{64}', packed[key]):
                raise ValueError('Missing CTC identity')
        return {'log_probs': matrix, 'id_to_phone': mapping, 'blank_id': packed['blank_id'],
                'duration_sec': packed['duration_sec'], 'checkpoint_sha256': packed['checkpoint_sha256'],
                'vocabulary_sha256': packed['vocabulary_sha256']}
    except (KeyError, TypeError, OverflowError, binascii.Error, zlib.error) as error:
        raise ValueError('Stored acoustic evidence is unavailable') from error
