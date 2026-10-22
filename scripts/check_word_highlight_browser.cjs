'use strict';
// Real page and audio-file decoding; controlled speech events, jobs and ai33
// playback. No model inference, API credits or physical speaker assertions.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/word-highlight-qa');
const reply = 'I can help you.';

function controlledPlayback() {
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak = utterance => {
    window.__utterances.push(utterance);
    window.__speech.push({ text: utterance.text, lang: utterance.lang, time: performance.now() });
  };
  window.__audio = [];
  window.__deferNextPlay = false;
  window.Audio = class {
    constructor(url) {
      this.src = url; this.currentTime = 0; this.duration = 9; this.paused = true;
      window.__audio.push(this);
    }
    play() {
      if (window.__deferNextPlay) {
        window.__deferNextPlay = false;
        return new Promise((resolve, reject) => { this.rejectPlay = reject; });
      }
      this.paused = false; this.onplaying?.(); return Promise.resolve();
    }
    pause() { if (!this.paused) { this.paused = true; this.onpause?.(); } }
    seek(time) { this.currentTime = time; this.ontimeupdate?.(); this.onseeked?.(); }
    finish() { this.paused = true; this.onended?.(); }
  };
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const fixture = JSON.parse(fs.readFileSync('data/generated/voice-chat-qa/browser-summary.json', 'utf8')).jobs[0].result;
  const sample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8')
    .trim().split('\n').map(JSON.parse).find(row => row.utterance_id === 'NJS/arctic_a0054');
  const config = { models: [{ id: 'qwen-base', ready: true }, { id: 'qwen-trained', ready: true }],
    topics: [], practice_prompts: [{ id: 'coffee', text: 'I would like a coffee.' }], speech_ready: true,
    ai33_available: true, voice_locales: ['en-US'], max_audio_seconds: 30 };
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const exceptions = [], checks = []; let job, serial = 0, voiceRequests = 0;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(controlledPlayback);
  const fresh = (text = reply, status = 'generating') => {
    const speech = structuredClone(fixture.speech);
    speech.coaching = { status: 'unavailable' }; delete speech.spoken_response;
    return job = { id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status, text,
      speech, metadata: {}, error: null, voice: { status: 'not_requested' } };
  };
  await page.route('**/api/chat/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/speech' || pathname.startsWith('/api/chat/jobs/')) return route.fulfill({ json: job });
    if (pathname === '/api/chat/cancel') { job.status = 'cancelled'; return route.fulfill({ json: job }); }
    if (pathname === '/api/chat/voice') {
      voiceRequests++;
      return route.fulfill({ json: { ...job, voice: { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) } } });
    }
    return route.fulfill({ status: 500, json: { message: 'Unexpected chat API call blocked by word-highlight QA.' } });
  });
  const text = page.locator('.chat-message.assistant .chat-message-text').last();
  const highlight = page.locator('.chat-message.assistant .chat-reply-word.is-speaking');
  const timing = page.locator('.chat-message.assistant .chat-word-timing').last();
  const replay = () => page.locator('.chat-message.assistant .chat-message-actions button').last();
  const speakingWord = () => highlight.allTextContents();
  const trigger = async (kind, index) => page.evaluate(({ kind, index }) => {
    const voice = window.__utterances.at(-1);
    if (kind === 'start') voice.onstart?.();
    else if (kind === 'boundary') voice.onboundary?.({ name: 'word', charIndex: index });
    else voice.onend?.();
  }, { kind, index });
  const selectVoice = async value => {
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-voice-provider').selectOption(value);
    await page.keyboard.press('Escape');
  };
  const upload = async () => page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready && window.MDDSpeechHighlight);
    await page.locator('#chat-free-talk').click();
    fresh(); await upload();
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'generating');
    assert.equal(await text.textContent(), '');
    assert.equal(await page.evaluate(() => window.__utterances.length), 0);
    job.status = 'ready';
    await page.waitForFunction(() => window.__utterances.length === 1 && !window.MDDChat.state.busy);
    assert.equal(await text.textContent(), reply, 'The whole reply must be visible before the audio start event.');
    await trigger('start'); await trigger('boundary', 2);
    assert.deepEqual(await speakingWord(), ['can']); assert.equal(await timing.isVisible(), false);
    assert.equal(await text.textContent(), reply, 'Upcoming words must stay visible while speaking.');
    await page.waitForTimeout(420); // Includes a job poll and render with no next boundary.
    assert.deepEqual(await speakingWord(), ['can']);
    await trigger('boundary', 6); assert.deepEqual(await speakingWord(), ['help']);
    checks.push('Generation is held; the entire reply is visible before autoplay and stays visible as native highlights advance.');
    await trigger('end'); job.status = 'ready';
    await page.waitForFunction(() => !window.MDDChat.state.busy && window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
    assert.equal(await text.textContent(), reply); assert.equal(await highlight.count(), 0);

    await replay().click(); await trigger('start'); await trigger('boundary', 6);
    assert.equal(await text.textContent(), reply); assert.deepEqual(await speakingWord(), ['help']);
    assert.equal(await highlight.getAttribute('aria-current'), 'true');
    await page.screenshot({ path: path.join(output, 'desktop-replay-highlight.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await page.screenshot({ path: path.join(output, 'mobile-replay-highlight.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await trigger('boundary', 11); assert.deepEqual(await speakingWord(), ['you.']);
    await trigger('end'); assert.equal(await highlight.count(), 0); assert.equal(await text.textContent(), reply);
    checks.push('Manual replay keeps the complete sentence visible, highlights each word and clears on end; mobile has no horizontal overflow.');

    await replay().click(); await trigger('start'); await trigger('boundary', 0);
    const oldVoice = await page.evaluate(() => window.__utterances.length - 1);
    await page.locator('#chat-stop-voice').click(); assert.equal(await highlight.count(), 0);
    await replay().click(); await trigger('start'); await trigger('boundary', 6);
    await page.evaluate(index => {
      const old = window.__utterances[index]; old.onboundary?.({ name: 'word', charIndex: 11 }); old.onend?.();
    }, oldVoice);
    assert.deepEqual(await speakingWord(), ['help']);
    assert.ok(await page.evaluate(() => window.MDDChat.state.speaking));
    await trigger('end');
    checks.push('Stop clears the highlight; stale old boundary/end callbacks cannot restart or clear newer replay.');

    await replay().click(); await trigger('start');
    assert.deepEqual(await speakingWord(), ['I']); assert.equal(await timing.isVisible(), true);
    await page.waitForTimeout(850);
    assert.notDeepEqual(await speakingWord(), ['I']); assert.equal(await highlight.count(), 1);
    assert.equal(await timing.textContent(), 'Đánh dấu từ ước lượng');
    await page.screenshot({ path: path.join(output, 'estimated-highlight.png'), fullPage: true, animations: 'disabled' });
    await trigger('end'); assert.equal(await highlight.count(), 0);
    checks.push('A voice with no native boundaries advances a bounded estimate and displays an estimate label.');

    await replay().click(); await trigger('start'); await trigger('boundary', 6);
    const navigationVoice = await page.evaluate(() => window.__utterances.length - 1);
    await page.evaluate(() => window.switchView('practice'));
    assert.equal(await highlight.count(), 0);
    await page.evaluate(index => {
      window.__utterances[index].onboundary?.({ name: 'word', charIndex: 11 }); window.__utterances[index].onend?.();
      window.switchView('chat');
    }, navigationVoice);
    assert.equal(await highlight.count(), 0); assert.equal(await page.evaluate(() => window.MDDChat.state.speaking), null);
    checks.push('Leaving the chat clears highlighting and ignores late speech callbacks after returning.');

    await selectVoice('ai33'); await replay().click();
    await page.waitForFunction(() => window.__audio.length === 1 && window.MDDChat.state.speaking);
    assert.deepEqual(await speakingWord(), ['I']); assert.equal(await timing.isVisible(), true);
    await page.evaluate(() => window.__audio.at(-1).seek(9));
    assert.deepEqual(await speakingWord(), ['you.']);
    await page.evaluate(() => window.__audio.at(-1).pause()); assert.equal(await highlight.count(), 0);
    await page.evaluate(() => window.__audio.at(-1).play()); assert.deepEqual(await speakingWord(), ['you.']);
    await page.evaluate(() => window.__audio.at(-1).seek(0)); assert.deepEqual(await speakingWord(), ['I']);
    await page.evaluate(() => window.__audio.at(-1).finish()); assert.equal(await highlight.count(), 0);
    checks.push('Mock ai33 playhead drives explicitly estimated highlights, including pause, resume, backward seek and end.');

    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-new').click(); await page.keyboard.press('Escape');
    const longReply = Array.from({ length: 1000 }, (_, i) => 'word' + i).join(' ');
    fresh(longReply, 'ready'); await upload();
    await page.waitForFunction(() => window.__audio.length === 2 && window.MDDChat.state.speaking);
    assert.equal(await text.textContent(), longReply);
    await page.evaluate(() => window.__audio.at(-1).seek(9));
    const lastRange = await page.evaluate(() => ({ ...window.MDDChat.state.playback.range }));
    assert.ok(lastRange.end <= 5000 && lastRange.end > 4900);
    assert.equal(await highlight.evaluate(node => +node.dataset.start >= 5000), false);
    await page.evaluate(() => window.__audio.at(-1).finish());
    checks.push('ai33 highlights only the exact first 5000 characters synthesized while the complete longer reply remains visible.');

    await page.evaluate(() => { window.__deferNextPlay = true; });
    await replay().click(); await page.waitForFunction(() => Boolean(window.__audio.at(-1).rejectPlay));
    const rejectedAudio = await page.evaluate(() => window.__audio.length - 1);
    await selectVoice('browser'); await replay().click(); await trigger('start'); await trigger('boundary', 12);
    const beforeReject = await speakingWord();
    await page.evaluate(index => window.__audio[index].rejectPlay(new Error('Old simulated play rejection')), rejectedAudio);
    await page.waitForTimeout(80);
    assert.deepEqual(await speakingWord(), beforeReject); assert.ok(await page.evaluate(() => window.MDDChat.state.speaking));
    assert.equal(await page.locator('#chat-error').textContent(), '');
    await trigger('end');
    checks.push('An old rejected ai33 play promise cannot clear newer browser replay or surface a stale error.');

    assert.deepEqual(exceptions, []);
    const report = { status: 'passed', checks, exceptions, inference: 'mock jobs only',
      speech: 'controlled browser boundaries and audio playhead; physical sound/timing not tested',
      mocked_ai33_voice_requests: voiceRequests, paid_tts_calls: 0 };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    const snapshot = await page.evaluate(() => ({ error: document.getElementById('chat-error')?.textContent,
      speaking: window.MDDChat?.state.speaking, playback: window.MDDChat?.state.playback && {
        turnId: window.MDDChat.state.playback.turnId, range: window.MDDChat.state.playback.range, mode: window.MDDChat.state.playback.mode },
      highlighted: [...document.querySelectorAll('.is-speaking')].map(node => node.textContent),
      speech_calls: window.__speech?.map(call => call.text), audio_instances: window.__audio?.length }));
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify({ status: 'failed', checks, exceptions, message: error.message, snapshot }, null, 2));
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true });
    console.error(JSON.stringify(snapshot, null, 2)); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
