'use strict';

// Pure selection tests require only Node. The browser test also runs when the
// existing local UI QA Playwright and Chrome installation are available.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { preferredVoiceProvider, voiceSelection, voiceCatalog, voiceResourceKey } = require('../demo/chat.js');

const config = {
  speech_ready: true, ai33_available: true, kokoro_available: true, voice_locales: ['en-US'],
  default_voice_provider: 'kokoro', default_voice_id: 'Sarah', kokoro_default_voice_id: 'af_heart',
  english_voices: [{ id: 'Sarah', name: 'Sarah', locale: 'en-US', gender: 'female', provider: 'elevenlabs', preview_url: null }],
  kokoro_voices: [{ id: 'af_heart', name: 'Heart', locale: 'en-US', gender: 'female', provider: 'kokoro', preview_url: null },
    { id: 'am_michael', name: 'Michael', locale: 'en-US', gender: 'male', provider: 'kokoro', preview_url: null },
    { id: 'bf_emma', name: 'Emma', locale: 'en-GB', gender: 'female', provider: 'kokoro', preview_url: null },
    { id: 'bm_george', name: 'George', locale: 'en-GB', gender: 'male', provider: 'kokoro', preview_url: null },
    { id: 'af_bella', name: 'Bella', locale: 'en-US', gender: 'female', provider: 'kokoro', preview_url: null }],
  models: [{ id: 'qwen-trained', ready: true }, { id: 'qwen-base', ready: true }],
  practice_prompts: [], topics: [{ id: 'daily-life', title: 'Daily life', level: 'A2', icon: 'sun',
    description: 'Talk about your day.', role: 'A friend', guidance: { keywords: ['morning'], starter: 'Every day I…' } }],
};

test('the simplified English picker migrates previous provider preferences to the local voices', () => {
  assert.equal(preferredVoiceProvider(config), 'kokoro');
  assert.equal(preferredVoiceProvider(config, 'ai33'), 'kokoro');
  assert.equal(preferredVoiceProvider(config, 'ai33', 1), 'kokoro');
  assert.equal(preferredVoiceProvider(config, 'browser'), 'kokoro');
  assert.equal(preferredVoiceProvider(config, 'kokoro', 1), 'kokoro');
});

test('an unavailable local runtime falls back to an available provider', () => {
  assert.equal(preferredVoiceProvider({ ...config, kokoro_available: false }, 'kokoro', 1), 'ai33');
  assert.equal(preferredVoiceProvider({ ...config, kokoro_available: false, ai33_available: false }), 'browser');
});

test('voice IDs are validated against the selected provider and never cross catalogs', () => {
  assert.deepEqual(voiceSelection(config, 'kokoro', 'Sarah', .85), { provider: 'kokoro', voice_id: 'af_heart', speed: .85 });
  assert.deepEqual(voiceSelection(config, 'ai33', 'af_heart', 1), { provider: 'ai33', voice_id: 'Sarah', speed: 1 });
  assert.deepEqual(voiceSelection(config, 'kokoro', 'am_michael', 1.15), { provider: 'kokoro', voice_id: 'am_michael', speed: 1.15 });
  assert.deepEqual(voiceSelection(config, 'kokoro', 'af_bella', 1), { provider: 'kokoro', voice_id: 'af_heart', speed: 1 });
  const curatedIds = ['af_heart', 'am_michael', 'bf_emma', 'bm_george'];
  assert.deepEqual(voiceCatalog(config, 'kokoro').map(voice => voice.id), curatedIds);
  for (const id of curatedIds) assert.equal(voiceSelection(config, 'kokoro', id, 1).voice_id, id);
  assert.deepEqual(voiceCatalog(config, 'browser'), []);
});

test('prepared audio and pending synthesis are isolated by provider even when IDs coincide', () => {
  const key = provider => voiceResourceKey('job', 0, 'sentence', { provider, voice_id: 'same', speed: 1 }, 'Hello.');
  assert.notEqual(key('kokoro'), key('ai33'));
  assert.equal(voiceResourceKey('job', 0, 'sentence', { voice_id: 'same', speed: 1 }, 'Hello.'), key('ai33'));
});

let chromium;
try { ({ chromium } = require('../.tools/ui-qa/node_modules/playwright')); } catch {}
const chromePath = process.env.MDD_CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

test('four English voice choices, saved selection, local synthesis and sentence prefetch', {
  skip: !chromium || !fs.existsSync(chromePath), timeout: 25000,
}, async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage();
  const errors = [], requests = [];
  let pollCount = 0, readyReply = false;
  const id = 'a'.repeat(32), reply = 'Tea sounds great. What kind do you like?';
  const speech = { transcript: 'I like tea.', reference_text: 'I like tea.', reference_source: 'asr', reply_ready: false,
    comparison: { words: [{ text: 'I', status: 'match' }, { text: 'like', status: 'match' }, { text: 'tea.', status: 'match' }], events: [] } };
  const job = { id, status: 'generating', text: 'Partial reply', metadata: { reference_revision: 0 }, speech,
    voice: { status: 'not_requested' }, sentence_voice: { status: 'not_requested' } };
  const demo = path.join(__dirname, '..', 'demo');
  const html = fs.readFileSync(path.join(demo, 'index.html'), 'utf8').replace(/<script defer src="\/(?:app|pronunciation)\.js"><\/script>/g, '');
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    if (!sessionStorage.getItem('kokoro-qa-seeded')) {
      sessionStorage.setItem('am-chat-selection-v1', JSON.stringify({ voiceProvider: 'ai33', voiceId: 'Sarah' }));
      sessionStorage.setItem('kokoro-qa-seeded', '1');
    }
    window.__plays = [];
    window.Audio = class extends EventTarget {
      constructor(src) { super(); Object.assign(this, { src, readyState: 4, paused: true, currentTime: 0, duration: 3 }); }
      load() {}
      play() { this.paused = false; window.__plays.push(this.src); this.onplaying?.(); return Promise.resolve(); }
      pause() { this.paused = true; this.onpause?.(); }
      removeAttribute(name) { if (name === 'src') this.src = ''; }
    };
  });
  await page.route('https://mdd.test/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/') return route.fulfill({ contentType: 'text/html', body: html });
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/speech') return route.fulfill({ json: job });
    if (pathname === '/api/chat/jobs/' + id) {
      pollCount++;
      if (readyReply) { job.status = 'ready'; job.text = reply; job.speech.reply_ready = true; }
      return route.fulfill({ json: job });
    }
    if (pathname === '/api/chat/voice') {
      const request = route.request().postDataJSON();
      requests.push({ ...request, replyReady: job.speech.reply_ready });
      const field = request.target === 'sentence' ? 'sentence_voice' : 'voice';
      job[field] = { status: 'ready', provider: request.provider, url: '/api/chat/audio/' + (field === 'voice' ? 'b' : 'c').repeat(32) };
      return route.fulfill({ json: job });
    }
    if (/^\/[\w_]+\.(?:js|css)$/.test(pathname)) return route.fulfill({ path: path.join(demo, pathname.slice(1)) });
    return route.fulfill({ status: 404, body: '' });
  });
  try {
    await page.goto('https://mdd.test/');
    await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.evaluate(() => { document.getElementById('chat-view').hidden = false; window.MDDChat.onViewChange('chat', () => true); document.getElementById('chat-settings').open = true; });
    assert.equal(await page.locator('#chat-voice-provider').inputValue(), 'kokoro');
    assert.equal(await page.locator('#chat-voice-provider').isVisible(), false);
    assert.equal(await page.locator('#chat-english-voice').inputValue(), 'af_heart');
    assert.equal(await page.locator('#chat-english-voice option').count(), 4);
    assert.equal(await page.locator('#chat-voice-cost').isVisible(), false);
    assert.deepEqual(await page.locator('#chat-english-voice option').allTextContents(),
      ['Mỹ · Nữ · Heart', 'Mỹ · Nam · Michael', 'Anh · Nữ · Emma', 'Anh · Nam · George']);
    await page.locator('#chat-english-voice').selectOption('bm_george');
    await page.reload();
    await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.evaluate(() => { document.getElementById('chat-view').hidden = false; window.MDDChat.onViewChange('chat', () => true); document.getElementById('chat-settings').open = true; });
    assert.equal(await page.locator('#chat-voice-provider').inputValue(), 'kokoro');
    assert.equal(await page.locator('#chat-english-voice').inputValue(), 'bm_george', 'An existing curated voice stays selected.');
    await page.locator('.chat-topic-card').click();
    assert.equal(await page.locator('.topic-voice-choice input[value="bm_george"]').isChecked(), true);
    assert.equal(await page.locator('.topic-voice-row').count(), 4);
    assert.equal(await page.locator('.topic-browser-choice').count(), 0);
    assert.equal(await page.locator('#topic-voice-search').isVisible(), false);
    assert.equal(await page.locator('.topic-voice-filters').isVisible(), false);
    assert.doesNotMatch(await page.locator('#chat-topic-setup').innerText(), /Kokoro|ai33|Edge|ElevenLabs/);
    assert.equal(await page.locator('.topic-voice-preview:enabled').count(), 0, 'Voices without previews have a disabled preview control.');
    assert.equal(await page.locator('#topic-voice-count').textContent(), '4 giọng');
    await page.locator('#topic-setup-close').click();
    await page.evaluate(() => {
      window.MDDChat.onViewChange('chat', () => true);
      window.__sending = window.MDDChat.sendAudio(new ArrayBuffer(64), 'test.wav');
    });
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1)?.sentenceVoice?.status === 'ready');
    assert.equal(requests.filter(request => request.target === 'sentence').length, 1);
    assert.equal(requests.filter(request => !request.target).length, 0, 'Partial replies do not synthesize.');
    assert.deepEqual(await page.evaluate(() => window.__plays), [], 'Sentence preparation must not autoplay.');
    readyReply = true;
    await page.waitForFunction(() => window.__plays.length === 1);
    assert.ok(pollCount >= 1);
    assert.equal(requests.filter(request => !request.target).length, 1);
    for (const request of requests) {
      assert.equal(request.provider, 'kokoro'); assert.equal(request.voice_id, 'bm_george');
      if (!request.target) assert.equal(request.replyReady, true);
    }
    await page.locator('.chat-turn-feedback > summary').click();
    await page.locator('.chat-sentence-listen').click();
    await page.waitForFunction(() => window.__plays.length === 2);
    assert.equal(requests.filter(request => request.target === 'sentence').length, 1, 'Listen reuses prepared local audio.');
    assert.deepEqual(errors, []);
  } finally { await page.close(); await browser.close(); }
});
