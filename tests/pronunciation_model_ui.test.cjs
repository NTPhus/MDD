'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pronunciationModels, preferredPronunciationModel, pronunciationRequest } = require('../demo/chat.js');

const models = [
  { id: 'cascade-B', label: 'wav2vec2 B', ready: true },
  { id: 'crottc-if-l2-arctic', label: 'CROTTC-IF · Thử nghiệm', ready: true },
];
const config = { pronunciation_models: models, default_pronunciation_model_id: 'cascade-B' };

test('B stays default even when installed CROTTC appears first or is server default', () => {
  assert.equal(preferredPronunciationModel(config), 'cascade-B');
  const changed = { pronunciation_models: [...models].reverse(), default_pronunciation_model_id: 'crottc-if-l2-arctic' };
  assert.equal(preferredPronunciationModel(changed), 'cascade-B');
  assert.deepEqual(pronunciationRequest(changed), { pronunciation_model_id: 'cascade-B' });
});

test('only explicit, advertised and ready CROTTC preference reaches a speech request', () => {
  assert.equal(preferredPronunciationModel(config, 'crottc-if-l2-arctic'), 'crottc-if-l2-arctic');
  assert.deepEqual(pronunciationRequest(config, 'crottc-if-l2-arctic'), { pronunciation_model_id: 'crottc-if-l2-arctic' });
  const missing = { pronunciation_models: [models[0], { ...models[1], ready: false }] };
  assert.equal(preferredPronunciationModel(missing, 'crottc-if-l2-arctic'), 'cascade-B');
  assert.deepEqual(pronunciationRequest(missing, 'crottc-if-l2-arctic'), { pronunciation_model_id: 'cascade-B' });
});

test('invalid stored preferences and malformed catalog entries cannot select arbitrary models', () => {
  for (const invalid of [null, {}, 1, 'qwen-base', '../model', '__proto__']) {
    assert.equal(preferredPronunciationModel(config, invalid), 'cascade-B');
  }
  assert.deepEqual(pronunciationModels({ pronunciation_models: [null, { id: 'unknown', ready: true }, ...models, models[0]] }), models);
  assert.equal(preferredPronunciationModel({ pronunciation_models: [{ ...models[1], ready: 'true' }, models[0]] }, models[1].id), 'cascade-B');
});

test('legacy server requests omit the new field and never implicitly select CROTTC', () => {
  for (const legacy of [undefined, {}, { pronunciation_models: null }, { pronunciation_models: [] }]) {
    assert.equal(preferredPronunciationModel(legacy, 'crottc-if-l2-arctic'), 'cascade-B');
    assert.deepEqual(pronunciationRequest(legacy, 'crottc-if-l2-arctic'), {});
  }
});

let chromium;
try { ({ chromium } = require('../.tools/ui-qa/node_modules/playwright')); } catch {}
const chromePath = process.env.MDD_CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

test('Settings selection persists, changes only future speech requests and leaves Qwen coach separate', {
  skip: !chromium || !fs.existsSync(chromePath), timeout: 25000,
}, async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage(), requests = [], errors = [], speechRequests = [];
  const demo = path.join(__dirname, '..', 'demo');
  const html = fs.readFileSync(path.join(demo, 'index.html'), 'utf8')
    .replace(/<script defer src="\/(?:app|pronunciation)\.js"><\/script>/g, '');
  let backendConfig = { ...config, speech_ready: true, default_voice_provider: 'browser', voice_locales: [],
    models: [{ id: 'qwen-trained', ready: true }, { id: 'qwen-base', ready: true }],
    practice_prompts: [], topics: [], english_voices: [], kokoro_voices: [] };
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('pronunciation-qa-seeded')) {
      sessionStorage.setItem('am-chat-selection-v1', JSON.stringify({ autoRead: false, model: 'qwen-trained', voiceProvider: 'browser' }));
      sessionStorage.setItem('pronunciation-qa-seeded', '1');
    }
  });
  await page.route('https://mdd.test/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname.startsWith('/api/')) requests.push(pathname);
    if (pathname === '/') return route.fulfill({ contentType: 'text/html', body: html });
    if (pathname === '/api/chat/config') return route.fulfill({ json: backendConfig });
    if (pathname === '/api/chat/speech') {
      speechRequests.push(route.request().postDataJSON());
      return route.fulfill({ json: { id: String(speechRequests.length).repeat(32), status: 'ready',
        text: 'What kind of tea do you enjoy?', metadata: { reference_revision: 0 },
        speech: { transcript: 'I like tea.', reference_text: 'I like tea.', reference_source: 'asr', reply_ready: true,
          comparison: null, sound_candidates: { status: 'unavailable', words: [] } }, voice: { status: 'not_requested' } } });
    }
    if (/^\/[\w_]+\.(?:js|css)$/.test(pathname)) return route.fulfill({ path: path.join(demo, pathname.slice(1)) });
    return route.fulfill({ status: 404, body: '' });
  });
  const openSettings = async () => {
    await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.evaluate(() => {
      document.getElementById('chat-view').hidden = false;
      window.MDDChat.onViewChange('chat', () => true);
      document.getElementById('chat-settings').open = true;
    });
  };
  try {
    await page.goto('https://mdd.test/');
    await openSettings();
    assert.equal(await page.locator('#chat-pronunciation-model').inputValue(), 'cascade-B');
    assert.equal(await page.locator('#chat-model').inputValue(), 'qwen-trained');
    const initialRequests = requests.length;
    await page.locator('#chat-pronunciation-model').selectOption('crottc-if-l2-arctic');
    await page.waitForTimeout(100);
    assert.equal(requests.length, initialRequests, 'Switching analysis model sends no audio, TTS, opening or regeneration request.');
    assert.equal(await page.locator('#chat-model').inputValue(), 'qwen-trained');
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 0);
    await page.reload();
    await openSettings();
    assert.equal(await page.locator('#chat-pronunciation-model').inputValue(), 'crottc-if-l2-arctic');
    await page.evaluate(async () => { await window.MDDChat.sendAudio(new ArrayBuffer(64), 'test.wav'); });
    assert.equal(speechRequests[0].pronunciation_model_id, 'crottc-if-l2-arctic');
    assert.equal(speechRequests[0].model_id, 'qwen-trained');
    backendConfig = { ...backendConfig, pronunciation_models: [models[0], { ...models[1], ready: false }] };
    await page.reload();
    await openSettings();
    assert.equal(await page.locator('#chat-pronunciation-model').inputValue(), 'cascade-B');
    assert.equal(await page.locator('#chat-pronunciation-model option[value="crottc-if-l2-arctic"]').isDisabled(), true);
    backendConfig = { ...backendConfig };
    delete backendConfig.pronunciation_models;
    await page.reload();
    await openSettings();
    assert.equal(await page.locator('#chat-pronunciation-model').isVisible(), false);
    await page.evaluate(async () => { await window.MDDChat.sendAudio(new ArrayBuffer(64), 'legacy.wav'); });
    assert.equal(Object.hasOwn(speechRequests.at(-1), 'pronunciation_model_id'), false);
    assert.deepEqual(errors, []);
  } finally {
    await page.close(); await browser.close();
  }
});
