'use strict';
// Cumulative read highlighting, reference layout and cache-retry UI; no paid calls.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function run() {
  const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
  const output = path.resolve('data/generated/progressive-highlight-qa'); fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce' });
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.speechSynthesis.speak = utterance => { window.__utterances.push(utterance); window.__speech.push({ text: utterance.text }); };
    window.__audio = [];
    window.Audio = class {
      constructor(url) { this.src = url; this.currentTime = 0; this.duration = 9; this.paused = true; window.__audio.push(this); }
      play() { this.paused = false; this.onplaying?.(); return Promise.resolve(); }
      pause() { this.paused = true; this.onpause?.(); }
      seek(time) { this.currentTime = time; this.ontimeupdate?.(); }
      finish() { this.paused = true; this.onended?.(); }
    };
  });
  const config = await (await page.request.get(base + '/api/chat/config')).json(); config.default_voice_provider = 'browser';
  const text = 'One two. Three four.';
  let job, voiceCalls = 0;
  const errors = [], checks = []; page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/chat/**', route => {
    const p = new URL(route.request().url()).pathname;
    if (p === '/api/chat/config') return route.fulfill({ json: config });
    if (p === '/api/chat/topic') {
      job = { id: 'b'.repeat(32), status: 'ready', text, metadata: { guidance: config.topics[0].guidance }, voice: { status: 'not_requested' } };
      return route.fulfill({ json: job });
    }
    if (p === '/api/chat/voice') {
      assert.equal(route.request().postDataJSON().job_id, job.id);
      voiceCalls++;
      job.voice = voiceCalls === 1
        ? { status: 'error', code: 'ai33_connection', retryable: true, message: 'Chưa tải được giọng đọc. Thử lại tác vụ đã có.' }
        : { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) };
      return route.fulfill({ json: job });
    }
    return route.fulfill({ status: 500, json: { error: 'Unexpected API blocked' } });
  });
  const words = page.locator('.assistant .chat-message-text').last();
  const read = () => words.locator('.is-read').allTextContents();
  const boundary = (index, finish = false) => page.evaluate(({ index, finish }) => {
    const utterance = window.__utterances.at(-1);
    if (finish) utterance.onend?.(); else {
      if (!utterance.__started) { utterance.__started = true; utterance.onstart?.(); }
      utterance.onboundary?.({ name: 'word', charIndex: index });
    }
  }, { index, finish });
  const noOverflow = async () => assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('.chat-topic-card').first().click();
    await page.waitForFunction(() => window.__utterances.length === 1 && !window.MDDChat.state.starting);
    assert.equal(await page.locator('.sidebar').isVisible(), true);
    assert.equal(await page.locator('#chat-scenario').isVisible(), true);
    assert.equal(await page.locator('#chat-scenario-title').textContent(), config.topics[0].title);
    assert.equal(await words.textContent(), text);
    await boundary(0); assert.deepEqual(await read(), ['One']);
    await boundary(4); assert.deepEqual(await read(), ['One', 'two.']);
    await boundary(0, true); await boundary(0);
    assert.deepEqual(await read(), ['One', 'two.', 'Three']);
    await page.screenshot({ path: path.join(output, 'progress-desktop.png'), fullPage: true });
    await boundary(6); assert.deepEqual(await read(), ['One', 'two.', 'Three', 'four.']);
    await boundary(0, true);
    assert.deepEqual(await read(), ['One', 'two.', 'Three', 'four.']);
    assert.equal(await words.locator('.is-speaking').count(), 0);
    checks.push('Desktop shows the app sidebar and scenario; read words accumulate across sentences and remain bold after completion.');
    await page.locator('.assistant .chat-message-actions button').last().click();
    assert.deepEqual(await read(), []);
    await boundary(0); assert.deepEqual(await read(), ['One']);
    await page.locator('#chat-stop-voice').click(); assert.deepEqual(await read(), []);
    await boundary(6); assert.deepEqual(await read(), []);
    checks.push('Replay resets read progress; Stop clears it and ignores late boundaries.');
    for (const width of [768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 }); await noOverflow();
      await page.screenshot({ path: path.join(output, `layout-${width}.png`), fullPage: true });
    }
    assert.equal(await page.locator('.sidebar').isVisible(), false);
    checks.push('Mobile/tablet hide the desktop sidebar; header, scenario and controls fit without horizontal scrolling.');
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-voice-provider').selectOption('ai33'); await page.keyboard.press('Escape');
    await page.locator('.assistant .chat-message-actions button').last().click();
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1).voice?.status === 'error');
    const retry = page.getByRole('button', { name: 'Thử lại giọng đọc', exact: true });
    assert.equal(await retry.isVisible(), true);
    await retry.click();
    await page.waitForFunction(() => window.__audio.length === 1 && window.MDDChat.state.speaking);
    assert.equal(voiceCalls, 2); assert.equal(await page.locator('#chat-error').textContent(), '');
    await page.evaluate(() => window.__audio.at(-1).seek(9));
    assert.deepEqual(await read(), ['One', 'two.', 'Three', 'four.']);
    await page.evaluate(() => window.__audio.at(-1).seek(0)); assert.deepEqual(await read(), ['One']);
    await page.evaluate(() => window.__audio.at(-1).finish());
    assert.deepEqual(await read(), ['One', 'two.', 'Three', 'four.']);
    assert.equal(await words.textContent(), text);
    checks.push('A failed cached voice offers explicit Retry; the same job resumes and audio seeking updates cumulative read progress.');
    assert.deepEqual(errors, []);
    const result = { status: 'passed', checks, errors, paid_calls: 0, generation: 'mocked', reference: 'User-authorized Brave window screenshot' };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }); console.error(errors); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
