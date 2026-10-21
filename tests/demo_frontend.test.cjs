'use strict';

// Run with: node --test tests/demo_frontend.test.cjs
// Restricted Windows environments can use: node --test --experimental-test-isolation=none tests/demo_frontend.test.cjs
// The real browser script runs in a VM with a small DOM and an in-memory API.
// No dependencies, local server, microphone, LLM, or voice provider are used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const demoDirectory = path.join(__dirname, '..', 'demo');
const appSource = fs.readFileSync(path.join(demoDirectory, 'app.js'), 'utf8');
const pageSource = fs.readFileSync(path.join(demoDirectory, 'index.html'), 'utf8');

class FakeNode {
  constructor(tag, document) {
    this.tagName = tag.toUpperCase();
    this.ownerDocument = document;
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.className = '';
    this.value = '';
    this.hidden = false;
    this.disabled = false;
    this.autoplay = false;
    this._text = '';
    this.pauseCount = 0;
    this.loadCount = 0;
    this.playCount = 0;
    this.focusCount = 0;
    const updateClasses = (values) => { this.className = [...values].join(' '); };
    const classes = () => new Set(this.className.split(/\s+/).filter(Boolean));
    this.classList = {
      add: (...values) => { const current = classes(); values.forEach((value) => current.add(value)); updateClasses(current); },
      remove: (...values) => { const current = classes(); values.forEach((value) => current.delete(value)); updateClasses(current); },
      contains: (value) => classes().has(value),
      toggle: (value, force) => {
        const current = classes();
        const enabled = force === undefined ? !current.has(value) : force;
        if (enabled) current.add(value); else current.delete(value);
        updateClasses(current);
        return enabled;
      },
    };
  }

  get textContent() { return this._text + this.children.map((child) => child.textContent).join(''); }
  set textContent(value) {
    this.children.forEach((child) => { child.parentNode = null; });
    this.children = [];
    this._text = String(value ?? '');
  }
  get innerHTML() { throw new Error('The DOM harness does not permit innerHTML.'); }
  set innerHTML(_value) { throw new Error('Use textContent and DOM nodes for untrusted content.'); }
  get options() { return this.querySelectorAll('option'); }
  get isConnected() { return this === this.ownerDocument.body || Boolean(this.parentNode?.isConnected); }
  get offsetWidth() { return this.getBoundingClientRect().width; }
  get offsetHeight() { return this.getBoundingClientRect().height; }

  append(...nodes) {
    for (let node of nodes) {
      if (!(node instanceof FakeNode)) {
        const text = new FakeNode('#text', this.ownerDocument);
        text.textContent = node;
        node = text;
      }
      if (node.parentNode) node.remove();
      node.parentNode = this;
      this.children.push(node);
    }
  }
  appendChild(node) { this.append(node); return node; }
  replaceChildren(...nodes) { this.textContent = ''; this.append(...nodes); }
  remove() {
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    this.parentNode = null;
  }
  setAttribute(name, value = '') {
    const text = String(value);
    this.attributes.set(name, text);
    if (name === 'id') { this.id = text; this.ownerDocument.ids.set(text, this); }
    else if (name === 'class') this.className = text;
    else if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = text;
    else if (['hidden', 'disabled', 'controls', 'autoplay'].includes(name)) this[name] = true;
    else if (['value', 'src', 'label', 'href'].includes(name)) this[name] = text;
  }
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }
  removeAttribute(name) {
    this.attributes.delete(name);
    if (['hidden', 'disabled'].includes(name)) this[name] = false;
    if (name === 'src') this.src = '';
  }
  matches(selector) {
    if (selector.startsWith('#')) return this.id === selector.slice(1);
    if (selector.startsWith('.')) return this.classList.contains(selector.slice(1));
    const attribute = /^\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/.exec(selector);
    if (attribute) return this.attributes.has(attribute[1]) && (attribute[2] === undefined || this.getAttribute(attribute[1]) === attribute[2]);
    return this.tagName.toLowerCase() === selector.toLowerCase();
  }
  querySelectorAll(selector) {
    const found = [];
    for (const child of this.children) {
      if (child.matches(selector)) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  contains(node) {
    for (let current = node; current; current = current.parentNode) if (current === this) return true;
    return false;
  }
  addEventListener(name, callback, options = {}) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push({ callback, once: options.once });
  }
  dispatchEvent(event) {
    if (typeof event === 'string') event = { type: event };
    event.target ||= this;
    event.currentTarget = this;
    event.preventDefault ||= function () { this.defaultPrevented = true; };
    event.stopPropagation ||= function () { this.propagationStopped = true; };
    for (const listener of [...(this.listeners.get(event.type) || [])]) {
      listener.callback(event);
      if (listener.once) this.listeners.set(event.type, this.listeners.get(event.type).filter((item) => item !== listener));
    }
    return !event.defaultPrevented;
  }
  click() { this.dispatchEvent('click'); }
  focus() {
    if (this.disabled || this.ownerDocument.activeElement === this) return;
    const previous = this.ownerDocument.activeElement;
    this.ownerDocument.activeElement = this;
    this.focusCount += 1;
    previous?.dispatchEvent({ type: 'blur', relatedTarget: this });
    this.dispatchEvent({ type: 'focus', relatedTarget: previous });
  }
  blur() {
    if (this.ownerDocument.activeElement !== this) return;
    this.ownerDocument.activeElement = this.ownerDocument.body;
    this.dispatchEvent({ type: 'blur', relatedTarget: this.ownerDocument.body });
  }
  pause() { this.pauseCount += 1; }
  play() { this.playCount += 1; return Promise.resolve(); }
  load() { this.loadCount += 1; }
  scrollIntoView() {}
  getBoundingClientRect() {
    const width = this.bounds?.width ?? (this.tagName === 'CANVAS' ? 0 : this.id === 'word-error-tooltip' ? 340 : 100);
    const height = this.bounds?.height ?? (this.tagName === 'CANVAS' ? 0 : this.id === 'word-error-tooltip' ? 280 : 32);
    const left = this.bounds?.left ?? 150;
    const top = this.bounds?.top ?? 200;
    return { x: left, y: top, width, height, left, top, right: left + width, bottom: top + height };
  }
}

function makeDocument() {
  const document = {
    ids: new Map(), created: [], hidden: false, listeners: new Map(),
    createElement(tag) { const node = new FakeNode(tag, this); this.created.push(node); return node; },
    getElementById(id) { return this.ids.get(id) || null; },
    querySelectorAll(selector) { return this.body.querySelectorAll(selector); },
    addEventListener(name, callback) {
      if (!this.listeners.has(name)) this.listeners.set(name, []);
      this.listeners.get(name).push(callback);
    },
    dispatchEvent(event) {
      if (typeof event === 'string') event = { type: event };
      event.target ||= this;
      event.currentTarget = this;
      event.preventDefault ||= function () { this.defaultPrevented = true; };
      event.stopPropagation ||= function () { this.propagationStopped = true; };
      for (const callback of this.listeners.get(event.type) || []) callback(event);
      return !event.defaultPrevented;
    },
  };
  document.body = document.createElement('body');
  document.activeElement = document.body;
  document.documentElement = { clientWidth: 1200, clientHeight: 800 };
  // Parse the static page only. Browser application code cannot invoke HTML parsing.
  const stack = [document.body];
  const voidTags = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);
  for (const token of pageSource.matchAll(/<\/?([a-z][\w:-]*)\b([^>]*?)>|([^<]+)/gi)) {
    if (token[3]) { stack.at(-1).append(token[3]); continue; }
    const tag = token[1].toLowerCase();
    if (token[0].startsWith('</')) {
      const index = stack.findLastIndex((node) => node.tagName.toLowerCase() === tag);
      if (index > 0) stack.length = index;
      continue;
    }
    if (tag === 'body') { stack.length = 1; continue; }
    if (tag === 'html' || tag === 'head') continue;
    const node = document.createElement(tag);
    for (const attribute of token[2].matchAll(/([:\w-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
      node.setAttribute(attribute[1], attribute[2] ?? attribute[3] ?? attribute[4] ?? '');
    }
    stack.at(-1).append(node);
    if (!voidTags.has(tag) && !token[0].endsWith('/>')) stack.push(node);
  }
  return document;
}

function deferred() {
  let resolve; let reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

const defaultModels = [
  { id: 'run:demo-A:best', label: 'A · source', stage: 'A', training_phase: 'source', purpose: 'scientific_training', ready: true },
  { id: 'run:demo-B:best', label: 'B · Vietnamese', stage: 'B', training_phase: 'adapted_vi', purpose: 'scientific_training', ready: true },
];

function analysisResult(overrides = {}) {
  return {
    analysis_id: 'analysis-001', feedback_available: true,
    model: defaultModels[0], audio: { duration_sec: 1.5, sample_rate: 16000 },
    phonemes: [{ phone: 'TH', start_sec: 0.1, end_sec: 0.3 }],
    comparison: { words: [{ text: 'Think', status: 'difference' }, { text: 'well', status: 'difference' }], counts: { substitution: 2, deletion: 1 } },
    assessment: {
      words: [{ index: 0, text: 'Think' }, { index: 1, text: 'well' }],
      errors: [
        { id: 'error-1', word_index: 0, type: 'substitution', expected_phone: 'TH', expected_ipa: 'θ', observed_phone: 'T', observed_ipa: 't', start_sec: 0.1, end_sec: 0.3 },
        { id: 'error-2', word_index: 0, type: 'deletion', expected_phone: 'NG', expected_ipa: 'ŋ' },
        { id: 'error-3', word_index: 1, type: 'substitution', expected_phone: 'W', observed_phone: 'V' },
      ],
    },
    ...overrides,
  };
}

function guidanceResult(analysis, overrides = {}) {
  return {
    analysis_id: analysis.analysis_id, status: 'ready', feedback_source: 'llm',
    assessment: analysis.assessment, voice_status: 'not_requested', audio: [], notices: [],
    feedback: { summary: 'Hãy luyện từng âm.', tips: [{ error_id: 'error-1', explanation: 'Đầu lưỡi chạm nhẹ giữa hai hàm răng.', instruction: 'Thổi hơi nhẹ.', practice_words: ['think', 'thin'] }] },
    ...overrides,
  };
}

function voiceResult(analysis, overrides = {}) {
  return guidanceResult(analysis, {
    voice_status: 'ready',
    audio: [
      { status: 'ready', language: 'vi-VN', purpose: 'feedback', url: '/api/voice/vi-feedback', text: 'Hướng dẫn tiếng Việt.' },
      { status: 'ready', language: 'en-US', purpose: 'reference', url: '/api/voice/en-reference', text: 'Think well.' },
      { status: 'ready', language: 'en-US', purpose: 'practice', url: '/api/voice/en-practice', text: 'Think. Thin.' },
    ],
    ...overrides,
  });
}

async function flush() { for (let index = 0; index < 4; index += 1) await new Promise(setImmediate); }

async function harness(options = {}) {
  const document = makeDocument();
  const requests = [];
  const micRequests = [];
  const timers = new Map();
  let nextTimer = 0;
  let feedbackIndex = 0;
  const analysis = options.analysis || analysisResult();
  const status = options.status || { models: defaultModels, default_model_id: defaultModels[0].id, training: [], capabilities: {} };
  const response = (data) => ({ ok: true, status: 200, json: async () => data });
  const fakeFetch = (url, init = {}) => {
    const call = { url, ...init, payload: init.body ? JSON.parse(init.body) : undefined };
    requests.push(call);
    if (url === '/api/status') return Promise.resolve(response(status));
    if (url === '/api/prompts') return Promise.resolve(response({ prompts: [{ id: 'sample-001', text: 'Think well.' }] }));
    if (url === '/api/analyze') return Promise.resolve(response(analysis));
    if (url === '/api/feedback') {
      const index = feedbackIndex++;
      try {
        const value = options.onFeedback ? options.onFeedback(call, index) : guidanceResult(analysis, { feedback_source: 'template', status: 'unavailable', voice_status: 'unavailable' });
        return Promise.resolve(value).then(response);
      } catch (error) { return Promise.reject(error); }
    }
    throw new Error(`Unexpected network request: ${url}`);
  };
  const location = new URL(options.origin || 'http://localhost:8765');
  const window = {
    location: { origin: location.origin, protocol: location.protocol, hostname: location.hostname, host: location.host, search: options.search || '' },
    isSecureContext: options.secureContext ?? true,
    AudioContext: Object.hasOwn(options, 'AudioContext') ? options.AudioContext : class {},
    innerWidth: options.width || 1200, innerHeight: 800, devicePixelRatio: 1,
    addEventListener() {},
  };
  let mediaDevices = Object.hasOwn(options, 'mediaDevices') ? options.mediaDevices : {
    getUserMedia: async () => { throw Object.assign(new Error('Permission denied by the fake microphone.'), { name: 'NotAllowedError' }); },
  };
  if (mediaDevices && Object.hasOwn(options, 'getUserMedia')) mediaDevices = { ...mediaDevices, getUserMedia: options.getUserMedia };
  if (typeof mediaDevices?.getUserMedia === 'function') {
    const getUserMedia = mediaDevices.getUserMedia;
    mediaDevices = {
      ...mediaDevices,
      getUserMedia(constraints) { micRequests.push(constraints); return getUserMedia.call(this, constraints); },
    };
  }
  const navigator = { mediaDevices };
  const schedule = (callback, delay) => { const id = ++nextTimer; timers.set(id, { callback, delay }); return id; };
  const context = vm.createContext({
    document, window, fetch: fakeFetch, URL, URLSearchParams, AbortController, Blob,
    console, navigator, performance: { now: () => 0 },
    setTimeout: schedule, setInterval: schedule,
    clearTimeout: (id) => timers.delete(id), clearInterval: (id) => timers.delete(id),
    requestAnimationFrame: (callback) => callback(),
    btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
  });
  vm.runInContext(`${appSource}\n;globalThis.__app = { state, analyze, startRecording, updateControls, loadGuidance, clearResult, clearAudio, renderModels, renderGuidance, renderGuidanceErrors, renderGuidanceAudio, safeVoiceUrl, refreshStatus };`, context, { filename: 'demo/app.js' });
  await flush();
  const app = context.__app;
  assert.equal(app.state.statusPending, false, 'boot status request finishes');
  assert.equal(document.getElementById('connection-error').hidden, true, 'boot succeeds against the fake API');
  app.state.sampleId = 'sample-001';
  return {
    app, analysis, status, requests, micRequests, document, timers, window, navigator,
    node: (id) => { const node = document.getElementById(id); assert.ok(node, `page contains #${id}`); return node; },
    feedbackCalls: () => requests.filter((call) => call.url === '/api/feedback'),
    runShortTimers: () => {
      for (const [id, timer] of [...timers]) {
        if (timer.delay <= 500) { timers.delete(id); timer.callback(); }
      }
    },
  };
}

function assertControlsAvailable(h) {
  for (const id of ['record-button', 'upload-button', 'sample-button', 'shuffle-prompt', 'prompt-select', 'reference-text', 'model-select', 'analyze-button']) {
    assert.equal(h.node(id).disabled, false, `#${id} remains usable while providers run`);
  }
  assert.equal(h.app.state.busy, false);
}

function stage(h, name) { return h.node(`guidance-step-${name}`).dataset.status; }

function wordDetailAnalysis() {
  return analysisResult({
    comparison: {
      words: [{ text: 'Think', status: 'difference' }, { text: 'well', status: 'match' }, { text: 'there', status: 'difference' }],
      counts: { substitution: 2, deletion: 1, insertion: 1 },
    },
    assessment: {
      status: 'ok', words: [{ index: 0, text: 'Think' }, { index: 1, text: 'well' }, { index: 2, text: 'there' }],
      errors: [
        { id: 'error-1', word_index: 0, type: 'substitution', expected_phone: 'TH', expected_ipa: 'θ', observed_phone: 'T', observed_ipa: 't', start_sec: null, end_sec: null },
        { id: 'error-2', word_index: 0, type: 'deletion', expected_phone: 'NG', expected_ipa: 'ŋ', observed_phone: null, observed_ipa: null },
        { id: 'error-3', word_index: 0, type: 'addition', expected_phone: null, expected_ipa: null, observed_phone: 'R*', observed_ipa: null },
        { id: 'error-4', word_index: 2, type: 'substitution', expected_phone: 'AH', expected_ipa: null, observed_phone: 'err', observed_ipa: null },
      ],
    },
  });
}

function wordTrigger(h, index = 0) {
  const trigger = h.node('comparison-words').querySelectorAll('.word-error-trigger').find((node) => node.dataset.wordIndex === String(index));
  assert.ok(trigger, `word ${index + 1} has an error-detail control`);
  return trigger;
}

test('local assessment renders immediately, then LLM tips, then voice, without blocking controls', async () => {
  const llm = deferred(); const tts = deferred();
  const h = await harness({ onFeedback: (_call, index) => index === 0 ? llm.promise : tts.promise });
  let analyzeFinished = false;
  const analyzing = h.app.analyze().then(() => { analyzeFinished = true; });
  await flush();
  assert.equal(analyzeFinished, true, 'analyze() finishes while LLM is unresolved');
  await analyzing;
  assert.equal(h.node('result-content').hidden, false);
  assert.equal(h.node('result-loading').hidden, true);
  assert.equal(h.node('save-result').hidden, false);
  const groups = h.node('guidance-errors').querySelectorAll('article');
  assert.equal(groups.length, 2, 'local errors are grouped by the reference word');
  assert.match(groups[0].textContent, /Think/);
  assert.equal(groups[0].querySelectorAll('.guidance-error').length, 2);
  assert.match(groups[0].textContent, /θ/);
  assert.match(groups[0].textContent, /t/);
  assert.equal(stage(h, 'mdd'), 'complete');
  assert.equal(stage(h, 'llm'), 'processing');
  assert.equal(stage(h, 'voice'), 'waiting');
  assert.deepEqual(h.feedbackCalls().map((call) => call.payload), [{ analysis_id: 'analysis-001', include_voice: false }]);
  assertControlsAvailable(h);
  await Promise.all([h.app.loadGuidance(), h.app.loadGuidance()]);
  assert.equal(h.feedbackCalls().length, 1, 'duplicate retries do not start another LLM request');

  llm.resolve(guidanceResult(h.analysis));
  await flush();
  assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi chạm nhẹ/);
  assert.match(h.node('guidance-errors').textContent, /think · thin/);
  assert.match(h.node('guidance-summary').textContent, /Hãy luyện/);
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 0);
  assert.equal(stage(h, 'llm'), 'complete');
  assert.equal(stage(h, 'voice'), 'processing');
  assert.deepEqual(h.feedbackCalls().map((call) => call.payload), [
    { analysis_id: 'analysis-001', include_voice: false },
    { analysis_id: 'analysis-001', include_voice: true },
  ]);
  assertControlsAvailable(h);
  await h.app.loadGuidance();
  assert.equal(h.feedbackCalls().length, 2, 'duplicate retry is also ignored during TTS');

  tts.resolve(voiceResult(h.analysis));
  await flush();
  const players = h.node('guidance-audio').querySelectorAll('audio');
  assert.equal(players.length, 3);
  assert.equal(players[0].lang, 'vi');
  assert.equal(players[1].lang, 'en');
  assert.ok(players.every((player) => player.controls && player.autoplay === false));
  assert.equal(stage(h, 'voice'), 'complete');
  assert.equal(h.app.state.guidancePending, false);
  assert.equal(h.node('guidance-retry').disabled, false);
  assert.equal(h.node('guidance-block').getAttribute('aria-busy'), null);
  assertControlsAvailable(h);
});

test('changing model aborts LLM work and ignores a late response without starting voice', async () => {
  const llm = deferred();
  const h = await harness({ onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  const oldCall = h.feedbackCalls()[0];
  assert.equal(oldCall.signal.aborted, false);
  h.node('model-select').value = defaultModels[1].id;
  h.node('model-select').dispatchEvent('change');
  assert.equal(oldCall.signal.aborted, true);
  assert.equal(h.app.state.result, null);
  assert.equal(h.node('result-content').hidden, true);
  assert.equal(h.node('guidance-block').hidden, true);
  assert.equal(h.node('save-result').hidden, true);
  llm.resolve(guidanceResult(h.analysis));
  await flush();
  assert.equal(h.feedbackCalls().length, 1);
  assert.equal(h.node('guidance-errors').textContent, '');
  assert.equal(h.node('guidance-summary').textContent, '');
  assert.equal(h.app.state.guidancePending, false);
  assert.equal(h.app.state.modelId, defaultModels[1].id);
});

test('changing model during TTS prevents late audio from replacing the new selection', async () => {
  const tts = deferred();
  const h = await harness({ onFeedback: (_call, index) => index === 0 ? guidanceResult(h.analysis) : tts.promise });
  await h.app.analyze();
  await flush();
  assert.equal(h.feedbackCalls().length, 2);
  assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi/);
  const oldCall = h.feedbackCalls()[1];
  h.node('model-select').value = defaultModels[1].id;
  h.node('model-select').dispatchEvent('change');
  assert.equal(oldCall.signal.aborted, true);
  tts.resolve(voiceResult(h.analysis));
  await flush();
  assert.equal(h.app.state.result, null);
  assert.equal(h.node('guidance-block').hidden, true);
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 0);
});

test('template and unavailable feedback never trigger an automatic voice request', async (t) => {
  for (const [source, status] of [['template', 'partial'], ['unavailable', 'unavailable']]) {
    await t.test(source, async () => {
      const h = await harness({ onFeedback: () => guidanceResult(h.analysis, { feedback_source: source, status, voice_status: 'unavailable' }) });
      await h.app.analyze();
      await flush();
      assert.equal(h.feedbackCalls().length, 1);
      assert.equal(h.feedbackCalls()[0].payload.include_voice, false);
      assert.equal(h.app.state.guidancePending, false);
      assert.ok(['template', 'unavailable'].includes(stage(h, 'llm')));
      assert.equal(stage(h, 'voice'), 'unavailable');
      assertControlsAvailable(h);
    });
  }
});

test('local-only assessments remain visible when guidance is unavailable', async (t) => {
  for (const overrides of [{ feedback_available: false, feedback_unavailable_reason: 'Cần nhãn tham chiếu.' }, { analysis_id: null }]) {
    await t.test(JSON.stringify(overrides), async () => {
      const h = await harness({ analysis: analysisResult(overrides) });
      await h.app.analyze();
      await flush();
      assert.equal(h.feedbackCalls().length, 0);
      assert.match(h.node('guidance-errors').textContent, /Think/);
      assert.equal(stage(h, 'mdd'), 'complete');
      assert.equal(stage(h, 'llm'), 'unavailable');
      assert.equal(stage(h, 'voice'), 'unavailable');
      assert.equal(h.node('guidance-retry').hidden, true);
      assertControlsAvailable(h);
    });
  }
});

test('provider configuration skips unavailable LLM or TTS stages', async (t) => {
  await t.test('LLM not configured', async () => {
    const h = await harness({ status: { models: defaultModels, default_model_id: defaultModels[0].id, training: [], capabilities: { llm_configured: false, tts_configured: true } } });
    await h.app.analyze();
    await flush();
    assert.equal(h.feedbackCalls().length, 0);
    assert.match(h.node('guidance-errors').textContent, /Think/);
    assert.equal(stage(h, 'llm'), 'unavailable');
    assert.equal(stage(h, 'voice'), 'unavailable');
  });
  await t.test('TTS not configured', async () => {
    const h = await harness({
      status: { models: defaultModels, default_model_id: defaultModels[0].id, training: [], capabilities: { llm_configured: true, tts_configured: false } },
      onFeedback: () => guidanceResult(h.analysis),
    });
    await h.app.analyze();
    await flush();
    assert.equal(h.feedbackCalls().length, 1);
    assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi/);
    assert.equal(stage(h, 'llm'), 'complete');
    assert.equal(stage(h, 'voice'), 'unavailable');
  });
});

test('network failure preserves local assessment and unlocks a single successful retry', async () => {
  const retry = deferred();
  const h = await harness({ onFeedback: (_call, index) => {
    if (index === 0) throw new Error('Provider temporarily offline');
    if (index === 1) return retry.promise;
    return voiceResult(h.analysis);
  } });
  await h.app.analyze();
  await flush();
  assert.equal(h.app.state.guidancePending, false);
  assert.equal(h.node('result-content').hidden, false);
  assert.match(h.node('guidance-errors').textContent, /Think/);
  assert.equal(stage(h, 'llm'), 'error');
  assert.equal(h.node('guidance-retry').hidden, false);
  assert.equal(h.node('guidance-retry').disabled, false);
  assert.match(h.node('guidance-message').textContent, /Provider temporarily offline/);
  h.node('guidance-retry').click();
  await flush();
  assert.equal(h.app.state.guidancePending, true);
  h.node('guidance-retry').click();
  assert.equal(h.feedbackCalls().length, 2);
  retry.resolve(guidanceResult(h.analysis));
  await flush();
  assert.deepEqual(h.feedbackCalls().map((call) => call.payload.include_voice), [false, false, true]);
  assert.equal(h.requests.filter((call) => call.url === '/api/analyze').length, 1);
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 3);
  assert.equal(h.app.state.guidancePending, false);
});

test('mismatched analysis IDs cannot render tips or start voice', async () => {
  const h = await harness({ onFeedback: () => guidanceResult(h.analysis, { analysis_id: 'a-different-analysis' }) });
  await h.app.analyze();
  await flush();
  assert.equal(h.feedbackCalls().length, 1);
  assert.equal(h.node('guidance-summary').textContent, '');
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 0);
  assert.equal(stage(h, 'llm'), 'error');
  assert.equal(h.node('guidance-retry').hidden, false);
  assert.equal(h.app.state.guidancePending, false);
});

test('TTS failure retains LLM tips and allows retry without repeating local inference', async () => {
  const tts = deferred(); const retry = deferred();
  const h = await harness({ onFeedback: (_call, index) => {
    if (index === 0) return guidanceResult(h.analysis);
    if (index === 1) return tts.promise;
    if (index === 2) return retry.promise;
    return voiceResult(h.analysis);
  } });
  await h.app.analyze();
  await flush();
  assert.equal(h.feedbackCalls().length, 2);
  assert.equal(stage(h, 'voice'), 'processing');
  assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi chạm nhẹ/);
  assertControlsAvailable(h);
  tts.reject(new Error('Voice service temporarily offline'));
  await flush();
  assert.equal(stage(h, 'mdd'), 'complete');
  assert.equal(stage(h, 'llm'), 'complete');
  assert.equal(stage(h, 'voice'), 'error');
  assert.equal(h.node('result-content').hidden, false);
  assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi chạm nhẹ/);
  assert.match(h.node('guidance-summary').textContent, /Hãy luyện/);
  assert.match(h.node('guidance-message').textContent, /Voice service temporarily offline/);
  assert.equal(h.node('guidance-retry').hidden, false);
  assert.equal(h.node('guidance-retry').disabled, false);
  assert.equal(h.app.state.guidancePending, false);
  assertControlsAvailable(h);

  const retrying = h.app.loadGuidance();
  await flush();
  assert.match(h.node('guidance-errors').textContent, /Đầu lưỡi chạm nhẹ/, 'known LLM tips remain visible while retry runs');
  await h.app.loadGuidance();
  assert.equal(h.feedbackCalls().length, 3);
  retry.resolve(guidanceResult(h.analysis));
  await retrying;
  await flush();
  assert.deepEqual(h.feedbackCalls().map((call) => call.payload.include_voice), [false, true, false, true]);
  assert.equal(h.requests.filter((call) => call.url === '/api/analyze').length, 1);
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 3);
  assert.equal(stage(h, 'voice'), 'complete');
});

test('untrusted guidance is literal text and audio is restricted to same-origin voice routes', async () => {
  const h = await harness();
  const literal = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
  const analysis = analysisResult();
  h.app.state.result = analysis;
  analysis.assessment.words[0].text = literal;
  analysis.assessment.errors[0].expected_ipa = literal;
  const beforeMarkup = h.document.created.filter((node) => ['IMG', 'SCRIPT'].includes(node.tagName)).length;
  h.app.renderGuidance(guidanceResult(analysis, {
    feedback: { summary: literal, tips: [{ error_id: 'error-1', explanation: literal, instruction: literal, practice_words: [literal] }] },
    notices: [literal], voice_status: 'ready',
    audio: [
      { status: 'ready', language: 'vi-VN', url: '/api/voice/safe-clip', text: literal },
      ...['https://external.example/api/voice/clip', '//external.example/api/voice/clip', 'javascript:alert(1)', 'data:audio/mpeg;base64,AA==', '/other/clip', '/api/voice/', '/api/voice/../status', 'http://user:secret@localhost:8765/api/voice/clip'].map((url) => ({ status: 'ready', language: 'en-US', url, text: literal })),
    ],
  }));
  assert.equal(h.node('guidance-summary').textContent, literal);
  assert.ok(h.node('guidance-errors').textContent.includes(literal));
  assert.ok(h.node('guidance-notices').textContent.includes(literal));
  assert.equal(h.document.created.filter((node) => ['IMG', 'SCRIPT'].includes(node.tagName)).length, beforeMarkup);
  const players = h.node('guidance-audio').querySelectorAll('audio');
  assert.equal(players.length, 1);
  assert.equal(players[0].src, 'http://localhost:8765/api/voice/safe-clip');
  assert.equal(h.app.safeVoiceUrl('/api/voice/valid'), 'http://localhost:8765/api/voice/valid');
  assert.equal(h.app.safeVoiceUrl(null), null);
  h.node('guidance-retry').hidden = true;
  players[0].dispatchEvent('error');
  assert.equal(h.node('guidance-retry').hidden, false);
  h.app.clearResult();
  assert.ok(players[0].pauseCount > 0);
  assert.ok(players[0].loadCount > 0);
  assert.equal(players[0].src, '');
  players[0].dispatchEvent('error');
  assert.equal(h.node('guidance-retry').hidden, true, 'late errors from cleared players do not restore old retry controls');
  assert.equal(h.feedbackCalls().length, 0);
});

test('model phase groups honor metadata and URL selection, and update when metadata changes', async () => {
  const models = [
    { id: 'base', label: 'BASE', kind: 'backbone', training_phase: 'pretrained', ready: true },
    { id: 'opaque-source', label: 'Source', training_phase: 'source', ready: true },
    { id: 'opaque-vietnamese', label: 'Vietnamese', training_phase: 'adapted_vi', ready: true },
    { id: 'opaque-control', label: 'Control', training_phase: 'control', ready: true },
    { id: 'opaque-smoke', label: 'Smoke', training_phase: 'technical', ready: true },
    { id: 'opaque-other', label: 'Other', ready: true },
  ];
  const h = await harness({ search: '?model=opaque-vietnamese', status: { models, default_model_id: 'opaque-source', training: [] } });
  const select = h.node('model-select');
  assert.equal(h.app.state.modelId, 'opaque-vietnamese');
  assert.equal(select.value, 'opaque-vietnamese');
  const option = (id) => select.querySelectorAll('option').find((node) => node.value === id);
  for (const model of models) {
    assert.equal(option(model.id).parentNode.tagName, 'OPTGROUP');
    assert.equal(option(model.id).parentNode.dataset.phase, model.training_phase || 'other');
  }
  const changedModels = models.map((model) => model.id === 'opaque-other' ? { ...model, training_phase: 'source' } : model);
  h.app.renderModels(changedModels, 'opaque-source');
  assert.equal(option('opaque-other').parentNode.dataset.phase, 'source', 'phase metadata changes refresh the option groups');
  assert.equal(select.value, 'opaque-vietnamese', 'a status refresh preserves the selected model');
});

test('model grouping refreshes when an existing checkpoint becomes a technical checkpoint', async () => {
  const models = [{ id: 'run:changing-A:best', label: 'Checkpoint', stage: 'A', purpose: 'scientific_training', ready: true }];
  const h = await harness({ status: { models, default_model_id: models[0].id, training: [] } });
  assert.equal(h.node('model-select').querySelector('option').parentNode.dataset.phase, 'source');
  h.app.renderModels([{ ...models[0], purpose: 'technical_smoke_only' }], models[0].id);
  assert.equal(h.node('model-select').querySelector('option').parentNode.dataset.phase, 'technical');
  assert.equal(h.node('model-select').value, models[0].id);
});

test('model groups fall back to stage and run IDs when phase metadata is absent', async () => {
  const models = [
    { id: 'base', label: 'BASE', kind: 'backbone', ready: true },
    { id: 'opaque-a', label: 'A', stage: 'A', purpose: 'scientific_training', ready: true },
    { id: 'run:trial-B:best', label: 'B', purpose: 'scientific_training', ready: true },
    { id: 'run:trial-C:best', label: 'C', purpose: 'scientific_training', ready: true },
    { id: 'run:technical-A:best', label: 'Technical', purpose: 'technical_smoke_only', ready: true },
  ];
  const h = await harness({ status: { models, default_model_id: 'opaque-a', training: [] } });
  const options = h.node('model-select').querySelectorAll('option');
  const actual = Object.fromEntries(options.map((option) => [option.value, option.parentNode.dataset.phase]));
  assert.deepEqual(actual, { base: 'pretrained', 'opaque-a': 'source', 'run:trial-B:best': 'adapted_vi', 'run:trial-C:best': 'control', 'run:technical-A:best': 'technical' });
});

test('hovering an error word immediately shows every local error and correction before the LLM returns', async () => {
  const llm = deferred();
  const h = await harness({ analysis: wordDetailAnalysis(), onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  const trigger = wordTrigger(h);
  assert.equal(trigger.tagName, 'BUTTON');
  assert.equal(trigger.getAttribute('aria-controls'), 'word-error-tooltip');
  assert.equal(trigger.getAttribute('aria-haspopup'), 'dialog');
  const beforeRequests = h.requests.length;
  trigger.dispatchEvent('mouseenter');
  assert.equal(h.node('word-error-tooltip').hidden, false);
  assert.equal(trigger.getAttribute('aria-expanded'), 'true');
  assert.match(h.node('word-error-tooltip-title').textContent, /Từ 1.*Think/);
  const cards = h.node('word-error-tooltip-content').querySelectorAll('.word-tooltip-error');
  assert.deepEqual(cards.map((card) => card.dataset.errorId), ['error-1', 'error-2', 'error-3']);
  assert.match(cards[0].textContent, /Thay âm/);
  assert.match(cards[0].textContent, /Model nghe: \/t\//);
  assert.match(cards[0].textContent, /Âm mục tiêu: \/θ\//);
  assert.match(cards[0].textContent, /lưỡi.*răng|răng.*lưỡi/);
  assert.match(cards[1].textContent, /Thiếu âm/);
  assert.match(cards[1].textContent, /∅/);
  assert.match(cards[1].textContent, /ŋ/);
  assert.match(cards[2].textContent, /Thêm âm/);
  assert.match(cards[2].textContent, /R\*/);
  assert.match(cards[2].textContent, /∅/);
  assert.doesNotMatch(cards[2].textContent, /\/ɹ\/|\/r\//, 'starred phone is not converted to a definite IPA');
  assert.ok(cards.every((card) => card.querySelector('.word-tooltip-correction')?.textContent.length > 15));
  assert.doesNotMatch(h.node('word-error-tooltip-content').textContent, /AH|error-4/, 'another word is not included');
  assert.equal(stage(h, 'llm'), 'processing');
  assert.equal(h.feedbackCalls().length, 1);
  assert.equal(h.requests.length, beforeRequests, 'opening details does not request guidance again');
  assert.equal(h.node('guidance-audio').querySelectorAll('audio').length, 0);
});

test('matched, unavailable, unknown-index and text-mismatched words never receive another word’s error popup', async () => {
  const analysis = wordDetailAnalysis();
  analysis.assessment.words[2].text = 'a different canonical word';
  analysis.comparison.words.push({ text: 'maybe', status: 'unavailable' });
  analysis.assessment.words.push({ index: 3, text: 'maybe' });
  analysis.assessment.errors.push(
    { id: 'unknown', word_index: null, type: 'deletion', expected_phone: 'UNKNOWN_NULL' },
    { id: 'outside', word_index: 99, type: 'deletion', expected_phone: 'UNKNOWN_OUTSIDE' },
    { id: 'boolean', word_index: true, type: 'deletion', expected_phone: 'UNKNOWN_BOOLEAN' },
    { id: 'string', word_index: '0', type: 'deletion', expected_phone: 'UNKNOWN_STRING' },
    { id: 'unavailable', word_index: 3, type: 'deletion', expected_phone: 'UNKNOWN_UNAVAILABLE' },
  );
  const llm = deferred();
  const h = await harness({ analysis, onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  const words = h.node('comparison-words').children;
  assert.equal(words[0].tagName, 'BUTTON');
  for (const word of [words[1], words[2], words[3]]) {
    assert.equal(word.tagName, 'SPAN');
    assert.equal(word.getAttribute('aria-controls'), null);
    word.dispatchEvent('mouseenter');
    word.click();
    assert.equal(h.node('word-error-tooltip').hidden, true);
  }
  wordTrigger(h).dispatchEvent('mouseenter');
  const content = h.node('word-error-tooltip-content');
  assert.equal(content.querySelectorAll('.word-tooltip-error').length, 3);
  assert.doesNotMatch(content.textContent, /UNKNOWN_|AH|a different canonical word/);
  assert.equal(h.feedbackCalls().length, 1);
});

test('unknown IPA and distortion markers remain literal without inventing target sounds', async () => {
  const llm = deferred();
  const h = await harness({ analysis: wordDetailAnalysis(), onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  wordTrigger(h, 2).dispatchEvent('mouseenter');
  const text = h.node('word-error-tooltip-content').textContent;
  assert.match(text, /AH/);
  assert.match(text, /err/);
  assert.doesNotMatch(text, /\/ə\/|\/ʌ\/|\/ɹ\//);
  assert.match(text, /Đọc chậm|đọc lại/);
  assert.equal(h.node('word-error-tooltip-content').querySelectorAll('.word-tooltip-error').length, 1);
});

test('hover transitions keep the popup usable, then dismiss after leaving without making requests', async () => {
  const llm = deferred();
  const h = await harness({ analysis: wordDetailAnalysis(), onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  const trigger = wordTrigger(h); const popup = h.node('word-error-tooltip');
  const beforeRequests = h.requests.length;
  trigger.dispatchEvent('mouseenter');
  trigger.dispatchEvent({ type: 'mouseleave', relatedTarget: popup });
  popup.dispatchEvent('mouseenter');
  h.runShortTimers();
  assert.equal(popup.hidden, false, 'the pointer can move from highlighted word into details');
  popup.dispatchEvent({ type: 'mouseleave', relatedTarget: h.document.body });
  h.runShortTimers();
  assert.equal(popup.hidden, true);
  assert.equal(trigger.getAttribute('aria-expanded'), 'false');
  assert.equal(trigger.getAttribute('aria-describedby'), null);
  assert.equal(h.requests.length, beforeRequests);
});

test('keyboard navigation and mobile tap can pin, close and dismiss error details accessibly', async () => {
  const llm = deferred();
  const h = await harness({ width: 390, analysis: wordDetailAnalysis(), onFeedback: () => llm.promise });
  await h.app.analyze();
  await flush();
  const trigger = wordTrigger(h); const popup = h.node('word-error-tooltip');
  assert.equal(popup.getAttribute('role'), 'dialog');
  assert.equal(popup.getAttribute('aria-modal'), 'false');
  trigger.focus();
  assert.equal(popup.hidden, false);
  assert.equal(h.document.activeElement, trigger);
  trigger.dispatchEvent({ type: 'keydown', key: 'ArrowDown' });
  assert.equal(h.document.activeElement, h.node('word-error-tooltip-close'));
  trigger.dispatchEvent('mouseleave');
  h.runShortTimers();
  assert.equal(popup.hidden, false);
  h.document.dispatchEvent({ type: 'keydown', key: 'Escape' });
  assert.equal(popup.hidden, true);
  assert.equal(h.document.activeElement, trigger);
  h.runShortTimers();
  assert.equal(popup.hidden, true, 'restoring focus after Escape does not reopen details');
  trigger.click();
  assert.equal(popup.hidden, false, 'tap pins details');
  trigger.click();
  assert.equal(popup.hidden, true, 'tapping the same word toggles pinned details off');
  trigger.click();
  h.document.dispatchEvent({ type: 'pointerdown', target: h.node('word-error-tooltip-content') });
  assert.equal(popup.hidden, false, 'interaction inside the popup is retained');
  h.document.dispatchEvent({ type: 'pointerdown', target: h.document.body });
  assert.equal(popup.hidden, true);
  trigger.click();
  h.node('word-error-tooltip-close').click();
  assert.equal(popup.hidden, true);
  assert.equal(trigger.getAttribute('aria-expanded'), 'false');
  assert.equal(h.feedbackCalls().length, 1);
});

test('an open local-error popup receives LLM enrichment without replacing its trigger or focus', async () => {
  const llm = deferred(); const tts = deferred();
  const h = await harness({ analysis: wordDetailAnalysis(), onFeedback: (_call, index) => index === 0 ? llm.promise : tts.promise });
  await h.app.analyze();
  await flush();
  const trigger = wordTrigger(h); const close = h.node('word-error-tooltip-close');
  trigger.focus();
  trigger.dispatchEvent({ type: 'keydown', key: 'ArrowDown' });
  const instruction = 'Giữ luồng hơi liên tục, rồi nối cả từ theo một nhịp.';
  assert.doesNotMatch(h.node('word-error-tooltip-content').textContent, /theo một nhịp/);
  const feedback = guidanceResult(h.analysis);
  feedback.feedback.tips[0].instruction = instruction;
  feedback.feedback.tips.push(
    { error_id: 'error-4', explanation: 'OTHER_WORD_ONLY', instruction: 'OTHER_WORD_ONLY', practice_words: [] },
    { error_id: 'invented-error', explanation: 'DO_NOT_ATTACH', instruction: 'DO_NOT_ATTACH', practice_words: [] },
  );
  llm.resolve(feedback);
  await flush();
  assert.equal(h.node('word-error-tooltip').hidden, false);
  assert.equal(wordTrigger(h), trigger, 'LLM completion does not rebuild word buttons');
  assert.equal(h.node('word-error-tooltip-close'), close);
  assert.equal(h.document.activeElement, close, 'reading details does not lose keyboard focus');
  assert.ok(h.node('word-error-tooltip-content').textContent.includes(instruction));
  assert.doesNotMatch(h.node('word-error-tooltip-content').textContent, /OTHER_WORD_ONLY|DO_NOT_ATTACH/);
  assert.match(h.node('word-error-tooltip-content').textContent, /think · thin/);
  assert.equal(h.node('word-error-tooltip-content').querySelectorAll('.word-tooltip-error').length, 3, 'errors lacking a tip retain local correction');
  assert.deepEqual(h.feedbackCalls().map((call) => call.payload.include_voice), [false, true]);
  const beforeRequests = h.requests.length;
  trigger.dispatchEvent('mouseenter');
  assert.equal(h.requests.length, beforeRequests);
  tts.resolve(voiceResult(h.analysis));
  await flush();
  assert.equal(h.node('word-error-tooltip').hidden, false);
  assert.equal(h.feedbackCalls().length, 2);
  assert.ok(h.node('guidance-audio').querySelectorAll('audio').every((player) => !player.autoplay && player.playCount === 0));
});

test('clearing a result, changing audio or changing model hides details and rejects late LLM updates', async (t) => {
  for (const action of ['result', 'audio', 'model']) {
    await t.test(action, async () => {
      const llm = deferred();
      const h = await harness({ analysis: wordDetailAnalysis(), onFeedback: () => llm.promise });
      await h.app.analyze();
      await flush();
      const trigger = wordTrigger(h);
      trigger.click();
      assert.equal(h.node('word-error-tooltip').hidden, false);
      if (action === 'result') h.app.clearResult();
      else if (action === 'audio') h.app.clearAudio();
      else {
        h.node('model-select').value = defaultModels[1].id;
        h.node('model-select').dispatchEvent('change');
      }
      assert.equal(h.node('word-error-tooltip').hidden, true);
      assert.equal(trigger.getAttribute('aria-expanded'), 'false');
      llm.resolve(guidanceResult(h.analysis));
      await flush();
      assert.equal(h.node('word-error-tooltip').hidden, true);
      assert.equal(h.app.state.wordTooltip, null);
      assert.equal(h.feedbackCalls().length, 1, 'a stale LLM result cannot launch voice work');
      assert.equal(h.node('guidance-summary').textContent, '');
    });
  }
});

test('word popup treats reference words, phones and LLM tips as literal text', async () => {
  const literal = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
  const analysis = wordDetailAnalysis();
  analysis.comparison.words[0].text = literal;
  analysis.assessment.words[0].text = literal;
  analysis.assessment.errors[0].expected_ipa = literal;
  const llm = deferred(); const tts = deferred();
  const h = await harness({ analysis, onFeedback: (_call, index) => index === 0 ? llm.promise : tts.promise });
  await h.app.analyze();
  await flush();
  const beforeMarkup = h.document.created.filter((node) => ['IMG', 'SCRIPT'].includes(node.tagName)).length;
  wordTrigger(h).dispatchEvent('mouseenter');
  assert.ok(h.node('word-error-tooltip-title').textContent.includes(literal));
  assert.ok(h.node('word-error-tooltip-content').textContent.includes(literal));
  const guidance = guidanceResult(h.analysis);
  guidance.feedback.tips = [{ error_id: 'error-1', explanation: literal, instruction: literal, practice_words: [literal] }];
  llm.resolve(guidance);
  await flush();
  assert.ok(h.node('word-error-tooltip-content').textContent.includes(literal));
  assert.equal(h.document.created.filter((node) => ['IMG', 'SCRIPT'].includes(node.tagName)).length, beforeMarkup);
  assert.equal(h.node('word-error-tooltip-content').querySelectorAll('audio').length, 0);
});

test('insecure LAN disables microphone with an HTTPS hint while sample analysis stays usable', async () => {
  const h = await harness({ origin: 'http://192.168.1.20:8765', secureContext: false });
  assert.equal(h.node('record-button').disabled, true);
  assert.match(h.node('record-hint').textContent, /HTTPS/i);
  assert.equal(h.node('upload-button').disabled, false);
  assert.equal(h.node('sample-button').disabled, false);
  assert.equal(h.node('prompt-select').disabled, false);
  assert.equal(h.node('model-select').disabled, false);

  await h.app.startRecording();
  assert.equal(h.micRequests.length, 0, 'an insecure page cannot request the mic even when getUserMedia is present');
  assert.equal(h.app.state.recordingPending, false);
  assert.equal(h.app.state.recording, false);
  assert.match(h.node('audio-error').textContent, /HTTPS/i);
  assert.match(h.node('record-hint').textContent, /HTTPS/i, 'the explanation remains visible beside the disabled control');

  h.app.state.sampleId = null;
  h.app.updateControls();
  assert.equal(h.node('analyze-button').disabled, true);
  h.app.state.sampleId = 'sample-001';
  h.app.updateControls();
  assert.equal(h.node('analyze-button').disabled, false);
  await h.app.analyze();
  await flush();
  const analysisCalls = h.requests.filter((call) => call.url === '/api/analyze');
  assert.equal(analysisCalls.length, 1);
  assert.equal(analysisCalls[0].payload.sample_id, 'sample-001');
  assert.equal(h.node('result-content').hidden, false);
  assert.equal(h.node('result-loading').hidden, true);
  assert.equal(h.node('record-button').disabled, true);
  assert.match(h.node('record-hint').textContent, /HTTPS/i);
  assert.equal(h.node('upload-button').disabled, false);
  assert.equal(h.node('sample-button').disabled, false);
  assert.equal(h.node('analyze-button').disabled, false);
  assert.equal(h.micRequests.length, 0);
});

test('secure localhost requests microphone permission and recovers cleanly when permission is denied', async () => {
  const h = await harness({ getUserMedia: async () => {
    throw Object.assign(new Error('Microphone permission denied.'), { name: 'NotAllowedError' });
  } });
  assert.equal(h.window.location.origin, 'http://localhost:8765');
  assert.equal(h.window.isSecureContext, true);
  assert.equal(h.node('record-button').disabled, false);
  await h.app.startRecording();
  assert.equal(h.micRequests.length, 1);
  assert.equal(h.micRequests[0].audio.channelCount, 1);
  assert.equal(h.app.state.recordingPending, false);
  assert.equal(h.app.state.recording, false);
  assert.equal(h.node('record-button').disabled, false);
  assert.equal(h.node('upload-button').disabled, false);
  assert.equal(h.node('audio-error').hidden, false);
  assert.match(h.node('audio-error').textContent, /quyền|cho phép/i);
  assert.doesNotMatch(h.node('audio-error').textContent, /TypeError|undefined|not a function/);
});

test('secure pages without a usable microphone API explain the limitation without throwing', async (t) => {
  const cases = [
    { name: 'mediaDevices missing', options: { mediaDevices: undefined } },
    { name: 'getUserMedia missing', options: { mediaDevices: {} } },
    { name: 'getUserMedia unavailable', options: { getUserMedia: null } },
  ];
  for (const item of cases) {
    await t.test(item.name, async () => {
      const h = await harness(item.options);
      assert.equal(h.node('record-button').disabled, true);
      assert.ok(h.node('record-hint').textContent.trim().length > 0);
      assert.equal(h.node('upload-button').disabled, false);
      assert.equal(h.node('sample-button').disabled, false);
      await h.app.startRecording();
      assert.equal(h.micRequests.length, 0);
      assert.equal(h.app.state.recordingPending, false);
      assert.equal(h.app.state.recording, false);
      assert.equal(h.node('audio-error').hidden, false);
      assert.ok(h.node('audio-error').textContent.trim().length > 0);
      assert.doesNotMatch(h.node('audio-error').textContent, /TypeError|undefined|not a function/);
      h.app.updateControls();
      assert.equal(h.node('analyze-button').disabled, false);
    });
  }
});

test('missing Web Audio blocks microphone access before requesting permission', async () => {
  const h = await harness({ AudioContext: undefined });
  assert.equal(h.node('record-button').disabled, true);
  assert.equal(h.node('sample-button').disabled, false);
  await h.app.startRecording();
  assert.equal(h.micRequests.length, 0);
  assert.equal(h.app.state.recordingPending, false);
  assert.equal(h.app.state.recording, false);
  assert.equal(h.node('audio-error').hidden, false);
  assert.doesNotMatch(h.node('audio-error').textContent, /TypeError|undefined|not a function/);
  h.app.updateControls();
  assert.equal(h.node('analyze-button').disabled, false);
});
