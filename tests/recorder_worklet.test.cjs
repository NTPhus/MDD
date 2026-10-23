'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

test('recorder downmixes both microphone channels before sending mono samples', () => {
  let Recorder;
  class FakeAudioWorkletProcessor {
    constructor() {
      this.messages = [];
      this.port = { onmessage: null, postMessage: message => this.messages.push(message) };
    }
  }
  const source = fs.readFileSync(path.join(__dirname, '..', 'demo', 'recorder-worklet.js'), 'utf8');
  vm.runInNewContext(source, {
    AudioWorkletProcessor: FakeAudioWorkletProcessor,
    Float32Array,
    registerProcessor: (_name, constructor) => { Recorder = constructor; },
  });
  const recorder = new Recorder();
  recorder.process([[new Float32Array([.8, .2]), new Float32Array([.2, -.2])]]);
  recorder.port.onmessage({ data: 'flush' });
  assert.deepEqual(Array.from(recorder.messages[0].samples), [.5, 0]);
  assert.equal(recorder.messages[1].stopped, true);
});
