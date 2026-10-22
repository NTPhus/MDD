'use strict';
// Real UI and audio decoding; mock jobs/TTS only. No model or paid calls.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

async function run() {
  const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8767';
  const out = path.resolve('data/generated/purple-sarah-qa');
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const exceptions = [], checks = [];
  page.on('pageerror', e => exceptions.push(e.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__audio = []; window.__visibleAtPlay = [];
    window.Audio = class {
      constructor(url) { this.src = url; this.currentTime = 0; this.duration = 9; this.paused = true; window.__audio.push(this); }
      play() {
        window.__visibleAtPlay.push([...document.querySelectorAll('.assistant .chat-message-text')].at(-1)?.textContent);
        this.paused = false; this.onplaying?.(); return Promise.resolve();
      }
      pause() { this.paused = true; this.onpause?.(); }
      seek(time) { this.currentTime = time; this.ontimeupdate?.(); }
      finish() { this.paused = true; this.onended?.(); }
    };
  });
  const config = await (await page.request.get(base + '/api/chat/config')).json();
  assert.equal(config.english_voice_name, 'Sarah - Mature, Reassuring, Confident');
  assert.equal(config.default_voice_provider, 'ai33');
  const fixture = JSON.parse(fs.readFileSync('data/generated/voice-chat-qa/browser-summary.json', 'utf8')).jobs[0].result;
  const sample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8').trim().split('\n').map(JSON.parse).find(r => r.utterance_id === 'NJS/arctic_a0054');
  const reply = 'That sounds like a busy morning. What do you enjoy most about your day?';
  let job, serial = 0, voiceRequests = 0;
  await page.route('**/api/chat/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/topic') {
      const topic = config.topics.find(t => t.id === route.request().postDataJSON().topic_id);
      job = { id: String(++serial).padStart(32, '0'), status: 'ready', text: 'Hi! What does your morning usually look like?', metadata: { guidance: topic.guidance }, voice: { status: 'not_requested' } };
      return route.fulfill({ json: job });
    }
    if (pathname === '/api/chat/speech') {
      const speech = structuredClone(fixture.speech); speech.coaching = { status: 'unavailable' };
      job = { id: String(++serial).padStart(32, '0'), status: 'ready', text: reply, speech,
        metadata: {}, voice: { status: 'not_requested' } };
      return route.fulfill({ json: job });
    }
    if (pathname === '/api/chat/voice') {
      voiceRequests++;
      return route.fulfill({ json: { ...job, voice: { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) } } });
    }
    if (pathname.startsWith('/api/chat/jobs/')) return route.fulfill({ json: job });
    return route.fulfill({ status: 500, json: { error: 'Unexpected API call blocked by mock QA' } });
  });
  const noOverflow = async () => assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    assert.equal(await page.locator('#chat-voice-provider').inputValue(), 'ai33');
    assert.equal(await page.locator('#chat-voice-provider option:checked').textContent(), config.english_voice_name);
    assert.ok(await page.locator('.chat-topic-card').count());
    for (const width of [1440, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 }); await noOverflow();
      await page.screenshot({ path: path.join(out, `topics-${width}.png`), fullPage: true });
    }
    checks.push('Topic landing fits desktop/tablet/mobile; Sarah selected by server default.');
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('.chat-topic-card').first().click();
    await page.waitForFunction(() => window.__audio.length === 1 && !window.MDDChat.state.starting);
    assert.equal(await page.locator('#chat-settings-label').textContent(), 'Sarah');
    await page.evaluate(() => window.__audio.at(-1).finish());
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
    await page.waitForFunction(() => window.__audio.length === 2 && !window.MDDChat.state.busy);
    const text = page.locator('.assistant .chat-message-text').last();
    assert.equal(await text.textContent(), reply);
    assert.equal(await page.evaluate(() => window.__visibleAtPlay.at(-1)), reply);
    const feedback = page.locator('.chat-turn-feedback').last();
    assert.equal(await feedback.getAttribute('open'), null);
    await page.evaluate(() => window.__audio.at(-1).seek(4));
    assert.equal(await page.locator('.assistant .is-speaking').count(), 1);
    assert.equal(await text.textContent(), reply);
    await page.waitForTimeout(180); // Let the existing highlight color transition finish.
    const colors = await page.evaluate(() => ({
      page: getComputedStyle(document.body).backgroundColor,
      mic: getComputedStyle(document.getElementById('chat-mic')).backgroundColor,
      user: getComputedStyle(document.querySelector('.user .chat-message-text')).backgroundColor,
      word: getComputedStyle(document.querySelector('.assistant .is-speaking')).backgroundColor,
    }));
    assert.equal(colors.page, 'rgb(255, 255, 255)');
    assert.equal(colors.mic, 'rgb(124, 58, 237)');
    assert.equal(colors.user, 'rgb(124, 58, 237)');
    assert.equal(colors.word, 'rgb(233, 213, 255)');
    await page.screenshot({ path: path.join(out, 'conversation-desktop.png'), fullPage: true });
    for (const width of [768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 }); await noOverflow();
      await page.screenshot({ path: path.join(out, `conversation-${width}.png`), fullPage: true });
      await page.locator('#chat-settings>summary').click(); await noOverflow(); await page.keyboard.press('Escape');
    }
    checks.push('White conversation and violet mic/user bubbles; full reply remains visible during estimated Sarah highlights.');
    await feedback.locator(':scope>summary').click(); await noOverflow();
    await page.screenshot({ path: path.join(out, 'feedback-mobile.png'), fullPage: true });
    await page.evaluate(() => window.__audio.at(-1).finish());
    assert.equal(await page.locator('.assistant .is-speaking').count(), 0);
    checks.push('Feedback remains collapsed by default; opening it on mobile fits and playback end clears highlighting.');
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-voice-provider').selectOption('browser'); await page.keyboard.press('Escape');
    await page.locator('.assistant .chat-message-actions button').last().click();
    await page.waitForFunction(() => window.__speech.length === 1);
    assert.equal(await page.evaluate(() => window.__speech[0].text), reply);
    await page.locator('#chat-stop-voice').click();
    checks.push('User can switch back to browser voice and replay the full response.');
    await page.locator('#chat-topics').click(); await page.locator('#chat-open-reading').click();
    assert.equal(await page.locator('#chat-reading-card').isVisible(), true); await noOverflow();
    await page.evaluate(() => window.switchView('practice'));
    assert.equal(await page.locator('#practice-view').isVisible(), true); await noOverflow();
    await page.screenshot({ path: path.join(out, 'practice-mobile.png'), fullPage: true });
    checks.push('Reading mode and the independent pronunciation workspace remain accessible.');
    assert.deepEqual(exceptions, []);
    const report = { status: 'passed', checks, colors, exceptions, mocked_voice_requests: voiceRequests,
      paid_calls_in_browser: 0, inference: 'mock jobs; real audio decoding', physical_speaker_test: false };
    fs.writeFileSync(path.join(out, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await page.screenshot({ path: path.join(out, 'failure.png'), fullPage: true });
    console.error('Browser errors', exceptions); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
