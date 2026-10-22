'use strict';

// Capture on the audio rendering thread; inference and WAV encoding run elsewhere.
class MDDRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.used = 0;
    this.active = true;
    this.port.onmessage = ({ data }) => {
      if (data === 'flush') {
        this.active = false;
        if (this.used) this.port.postMessage({ samples: this.buffer.slice(0, this.used) });
        this.used = 0;
        this.port.postMessage({ stopped: true });
      }
    };
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (this.active && input) {
      for (const sample of input) {
        this.buffer[this.used++] = sample;
        if (this.used === this.buffer.length) {
          this.port.postMessage({ samples: this.buffer.slice() });
          this.used = 0;
        }
      }
    }
    return this.active;
  }
}
registerProcessor('mdd-recorder', MDDRecorder);
