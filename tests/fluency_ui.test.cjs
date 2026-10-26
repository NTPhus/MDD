'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { fluencyView, renderFluency } = require('../demo/chat.js');

function speech(overrides = {}, metrics = {}) {
  return { fluency: { status: 'ready', source: 'whisper_word_timestamps', method: 'word_timing_fluency_v1', calibrated: false,
    metrics: { word_count: 24, audio_duration_sec: 12, recording_rate_wpm: 120,
      internal_pause_count: 2, longest_internal_pause_sec: 1.25, ...metrics }, ...overrides } };
}

class Node {
  constructor(tag, ownerDocument) { this.tagName = tag; this.ownerDocument = ownerDocument; this.children = []; this.dataset = {}; this._text = ''; this.open = false; this.scrollTop = 31; }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ''; this.children = children; }
}
function box() { const document = { createElement: tag => new Node(tag, document) }; return new Node('details', document); }

test('ready displays pace and internal timing gaps without inventing an expert score', () => {
  const view = fluencyView(speech());
  assert.equal(view.status, 'ready');
  assert.deepEqual(view.metrics.map(metric => metric.value), ['120', '2', '1,25']);
  assert.equal(view.metrics[0].unit, 'từ/phút');
  assert.match(view.metrics[0].help, /toàn thời lượng/);
  assert.doesNotMatch(JSON.stringify(view), /excellent|good|100\/100|score|calibrated/i);
});

test('pending initial speech and terminal legacy sessions have distinct concise states', () => {
  assert.equal(fluencyView({ initial: true }, true).status, 'pending');
  assert.equal(fluencyView({}, false).status, 'unavailable');
  assert.equal(fluencyView(undefined).metrics.length, 0);
  assert.equal(fluencyView(speech({ status: 'unavailable', reason: '<img src=x onerror=alert(1)>' })).status, 'unavailable');
});

test('partial timing shows measured recording pace and does not fabricate zero pauses', () => {
  const value = speech({ status: 'partial', metrics: { word_count: 3, audio_duration_sec: 4, recording_rate_wpm: 45 } });
  const view = fluencyView(value);
  assert.equal(view.status, 'partial');
  assert.equal(view.metrics.length, 1);
  assert.equal(view.metrics[0].value, '45');
});

test('untrusted numeric fields and incompatible timing sources never reach visible metrics', () => {
  for (const metrics of [
    { recording_rate_wpm: '120' }, { recording_rate_wpm: NaN }, { recording_rate_wpm: Infinity },
    { recording_rate_wpm: -1 }, { recording_rate_wpm: '<script>' }, { word_count: 0 },
    { audio_duration_sec: 0 }, { internal_pause_count: 1.5 }, { internal_pause_count: -1 },
    { internal_pause_count: 24 }, { longest_internal_pause_sec: 15 },
    { internal_pause_count: 0, longest_internal_pause_sec: .5 },
  ]) {
    const view = fluencyView(speech({}, metrics));
    assert.equal(view.status, 'unavailable', JSON.stringify(metrics));
    assert.deepEqual(view.metrics, []);
  }
  assert.equal(fluencyView(speech({ source: 'expert_score' })).status, 'unavailable');
});

test('in-place pending to ready update preserves collapse, scroll and text-only rendering', () => {
  const target = box(); target.open = true;
  renderFluency(target, { initial: true }, true);
  const summary = target.children[0];
  assert.equal(target.dataset.status, 'pending');
  assert.match(target.textContent, /Đang kiểm tra/);
  renderFluency(target, speech());
  assert.equal(target.dataset.status, 'ready');
  assert.equal(target.open, true);
  assert.equal(target.scrollTop, 31);
  assert.equal(target.children[0], summary, 'Keep the focused summary DOM node across the update.');
  assert.match(target.textContent, /120từ\/phút/);
  assert.doesNotMatch(target.textContent, /Đang kiểm tra/);
  assert.equal(target.children.filter(child => child.tagName === 'summary').length, 1);
  target.open = false;
  renderFluency(target, speech({ status: 'unavailable', feedback: [{ text: '<script>bad()</script>' }] }));
  assert.equal(target.open, false);
  assert.doesNotMatch(target.textContent, /script|bad/);
  assert.equal(target.children.filter(child => child.tagName === 'audio').length, 0);
});

let chromium;
try { ({ chromium } = require('../.tools/ui-qa/node_modules/playwright')); } catch {}
const fs = require('node:fs'), path = require('node:path');
const chromePath = process.env.MDD_CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
test('real DOM keeps the focused collapse summary and performs no audio or network work', {
  skip: !chromium || !fs.existsSync(chromePath), timeout: 15000,
}, async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage(), requests = [], errors = [];
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.setContent('<details id="fluency" open></details>');
    await page.addScriptTag({ path: path.join(__dirname, '../demo/chat.js') });
    const result = await page.evaluate(value => {
      const element = document.getElementById('fluency');
      renderFluency(element, { initial: true }, true);
      const summary = element.querySelector('summary'); summary.focus();
      renderFluency(element, value);
      return { open: element.open, focused: document.activeElement === summary,
        text: element.textContent, audioElements: document.querySelectorAll('audio').length };
    }, speech());
    assert.equal(result.open, true); assert.equal(result.focused, true);
    assert.match(result.text, /120từ\/phút/); assert.equal(result.audioElements, 0);
    assert.deepEqual(requests, []); assert.deepEqual(errors, []);
  } finally { await page.close(); await browser.close(); }
});
