---
language:
- en
license: apache-2.0
tags:
- automatic-speech-recognition
- pytorch
- transformers
- en
- generated_from_trainer
model-index:
- name: wav2vec2-xls-r-300m-phoneme
  results:
  - task:
      name: Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: DARPA TIMIT
      type: timit
      args: en
    metrics:
    - name: Test CER
      type: cer
      value: 7.996
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

## Model

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the Timit dataset. Check [this notebook](https://www.kaggle.com/code/vitouphy/phoneme-recognition-with-wav2vec2) for training detail.

## Usage 

**Approach 1:** Using HuggingFace's pipeline, this will cover everything end-to-end from raw audio input to text output.

```python
from transformers import pipeline

# Load the model
pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
# Process raw audio
output = pipe("audio_file.wav", chunk_length_s=10, stride_length_s=(4, 2))
```

**Approach 2:** More custom way to predict phonemes.
```python

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 
from datasets import load_dataset
import torch
import soundfile as sf

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

# Read and process the input
audio_input, sample_rate = sf.read("audio_file.wav")
inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# Decode id into string
predicted_ids = torch.argmax(logits, axis=-1)      
predicted_sentences = processor.batch_decode(predicted_ids)
print(predicted_sentences)

```

## Training and evaluation data
We use [DARPA TIMIT dataset](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech) for this model.
- We split into **80/10/10** for training, validation, and testing respectively. 
- That roughly corresponds to about **137/17/17** minutes. 
- The model obtained **7.996%** on this test set.


## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2000
- training_steps: 10000
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.10.2+cu102
- Datasets 1.18.2.dev0
- Tokenizers 0.11.0

### Citation
```
@misc { phy22-phoneme,
  author       = {Phy, Vitou},
  title        = {{Automatic Phoneme Recognition on TIMIT Dataset with Wav2Vec 2.0}},
  year         = 2022,
  note         = {{If you use this model, please cite it using these metadata.}},
  publisher    = {Hugging Face},
  version      = {1.0},
  doi          = {10.57967/hf/0125},
  url          = {https://huggingface.co/vitouphy/wav2vec2-xls-r-300m-timit-phoneme}
}
```