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
    const channels = inputs[0];
    if (this.active && channels?.length) {
      for (let index = 0; index < channels[0].length; index++) {
        let sample = 0;
        for (const channel of channels) sample += channel[index] / channels.length;
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
