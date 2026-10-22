'use strict';
// Real UI, mocked jobs/audio only: no model inference or paid TTS requests.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function run() {
  const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
  const out = path.resolve('data/generated/sentence-sarah-qa'); fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  const errors = [], checks = [], calls = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__audio = [];
    window.Audio = class {
      constructor(url) { this.src = url; this.currentTime = 0; this.duration = 9; this.paused = true; window.__audio.push(this); }
      play() { this.paused = false; this.onplaying?.(); return Promise.resolve(); }
      pause() { this.paused = true; this.onpause?.(); }
      seek(time) { this.currentTime = time; this.ontimeupdate?.(); }
      finish() { this.paused = true; this.onended?.(); }
    };
  });
  const config = await (await page.request.get(base + '/api/chat/config')).json();
  assert.equal(config.default_voice_provider, 'ai33'); assert.match(config.english_voice_name, /^Sarah/);
  const reference = 'Currently, I am a software engineer.';
  const fixture = { id: 'c'.repeat(32), status: 'ready', text: 'What do you enjoy about your work?',
    metadata: { reference_revision: 0 }, voice: { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) },
    sentence_voice: { status: 'not_requested' } };
  let fail = false, pending = false;
  await page.route('**/api/chat/**', route => {
    const url = new URL(route.request().url()).pathname;
    if (url === '/api/chat/config') return route.fulfill({ json: config });
    if (url === '/api/chat/voice') {
      const payload = route.request().postDataJSON(); calls.push(payload);
      assert.deepEqual(payload, { job_id: fixture.id, locale: 'en-US', target: 'sentence' });
      fixture.sentence_voice = fail
        ? { status: 'error', retryable: true, message: 'Bấm Thử lại giọng đọc để tải tiếp.' }
        : { status: 'generating' };
      pending = !fail; fail = false;
      return route.fulfill({ json: fixture });
    }
    if (url === '/api/chat/jobs/' + fixture.id) {
      if (pending) { pending = false; fixture.sentence_voice = { status: 'ready', url: '/api/chat/audio/' + 'b'.repeat(32) }; }
      return route.fulfill({ json: fixture });
    }
    return route.fulfill({ status: 500, json: { error: 'Unexpected API blocked' } });
  });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.locator('#chat-free-talk').click();
    await page.evaluate(({ reference, fixture }) => {
      const chat = window.MDDChat;
      chat.getTurns().push({ id: 'user-fixture', role: 'user', status: 'ready', content: 'Curly am a engineer' },
        { id: 'reply-fixture', role: 'assistant', status: 'ready', content: fixture.text, jobId: fixture.id,
          metadata: fixture.metadata, voice: fixture.voice,
          speech: { transcript: 'Curly am a engineer', reference_text: reference, reference_source: 'user_confirmed',
            comparison: { words: [], events: [] }, target_words: [], missing_words: [],
            rewrite: { status: 'not_needed' }, correction: { status: 'not_needed' } } });
      chat.onViewChange('chat', () => true);
    }, { reference, fixture });
    await page.locator('.chat-turn-feedback > summary').click();
    const button = page.locator('.chat-sentence-listen'), sentence = page.locator('.chat-sentence-text');
    assert.equal(await sentence.textContent(), reference);
    await button.click();
    await page.waitForFunction(() => document.querySelector('.chat-sentence-listen')?.disabled);
    assert.match(await button.textContent(), /Đang tạo giọng Sarah/);
    await page.waitForFunction(() => window.__audio.length === 1 && window.MDDChat.state.speaking === 'sentence-reply-fixture');
    assert.equal(await page.evaluate(() => window.__speech.length), 0);
    assert.equal(await page.evaluate(() => window.__audio[0].src), '/api/chat/audio/' + 'b'.repeat(32));
    assert.equal(await page.locator('.assistant .chat-message-text').textContent(), fixture.text);
    await page.evaluate(() => window.__audio[0].seek(5));
    assert.ok(await sentence.locator('.is-read').count() > 0);
    assert.equal(await page.locator('.assistant .is-speaking').count(), 0);
    checks.push('Nghe cả câu explicitly requests the server reference in English through Sarah; reply audio/text are independent.');
    await page.screenshot({ path: path.join(out, 'sentence-sarah.png'), fullPage: true });
    await button.click();
    assert.equal(await sentence.locator('.is-read').count(), 0);
    await page.evaluate(() => window.__audio[0].seek(9));
    assert.equal(await sentence.locator('.is-read').count(), 0);
    checks.push('Sentence highlighting follows its audio and stops without highlighting the bot reply or accepting stale events.');
    fail = true; await button.click();
    await page.getByRole('button', { name: 'Thử lại giọng Sarah', exact: true }).waitFor();
    await button.click();
    await page.waitForFunction(() => window.__audio.length === 2 && window.MDDChat.state.speaking);
    assert.equal(await page.locator('#chat-error').textContent(), '');
    await page.evaluate(() => window.__audio.at(-1).finish());
    assert.equal(await sentence.locator('.is-read').count(), reference.split(' ').length);
    checks.push('Cached-task errors offer explicit Retry; replay retains the entire sentence and finishes highlighting it.');
    fixture.metadata.reference_revision = 1;
    await page.evaluate(() => {
      const chat = window.MDDChat, turn = chat.getTurns().at(-1);
      turn.metadata = { reference_revision: 1 }; turn.speech.reference_text = 'I work as a software engineer.';
      chat.onViewChange('chat', () => true);
    });
    assert.equal(await sentence.textContent(), 'I work as a software engineer.');
    await button.click();
    await page.waitForFunction(() => window.__audio.length === 3 && window.MDDChat.state.speaking);
    await page.evaluate(() => window.__audio.at(-1).finish());
    assert.equal(await sentence.locator('.is-read').count(), 6);
    checks.push('An edited reference replaces the sentence preview and is played under the new revision.');
    assert.deepEqual(errors, []);
    const report = { status: 'passed', paid_calls: 0, calls, checks, errors };
    fs.writeFileSync(path.join(out, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
