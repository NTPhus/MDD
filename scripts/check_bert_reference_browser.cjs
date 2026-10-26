'use strict';
// Real browser UI; every speech, reference and voice job is mocked. No model or paid TTS calls.
const { chromium } = require(process.env.MDD_PLAYWRIGHT_MODULE || '../.tools/ui-qa/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';

async function run() {
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const exceptions = [], voices = [], references = [], checks = [];
  const raw = 'It was the divicote true getesir', recovered = 'It was difficult to get there.';
  const comparisons = text => ({ words: text.split(/\s+/).map((word, index) => ({ text: word, status: index === 2 ? 'difference' : 'uncertain' })), events: [] });
  let job = { id: 'b'.repeat(32), status: 'correcting', text: '', metadata: { reference_revision: 0 },
    speech: { transcript: raw, reference_text: raw, reference_source: 'asr', pronunciation_reference_ready: false,
      comparison: comparisons(raw), interpretation: { status: 'pending' }, reply_ready: false },
    voice: { status: 'not_requested' }, sentence_voice: { status: 'not_requested' } };
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__voicePlays = [];
    window.Audio = class extends EventTarget {
      constructor(src) { super(); Object.assign(this, { src, paused: true, currentTime: 0, duration: 4, readyState: 4 }); }
      load() {}
      play() { this.paused = false; window.__voicePlays.push(this.src); this.onplaying?.(); return Promise.resolve(); }
      pause() { this.paused = true; this.onpause?.(); }
    };
  });
  await page.route('**/api/chat/config', async route => {
    const response = await route.fetch(), config = await response.json();
    return route.fulfill({ json: { ...config, kokoro_available: false, ai33_available: true,
      default_voice_provider: 'ai33', voice_locales: ['en-US'] } });
  });
  await page.route('**/api/chat/speech', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/voice', route => {
    const request = route.request().postDataJSON(), target = request.target || 'reply';
    voices.push({ target, reference: job.speech.reference_text, ready: job.speech.pronunciation_reference_ready });
    assert.equal(job.speech.pronunciation_reference_ready, true, 'Sentence synthesis must wait for the BERT reference.');
    job[target === 'sentence' ? 'sentence_voice' : 'voice'] = { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) };
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/reference', route => {
    const request = route.request().postDataJSON(); references.push(request);
    const changed = request.reference_text !== job.speech.reference_text;
    job = { ...job, status: 'ready', ...(changed ? { text: 'Sorry for the misunderstanding. What made the trip easier?' } : {}),
      metadata: { reference_revision: changed ? 1 : 0 },
      speech: { ...job.speech, reference_text: request.reference_text, reference_source: 'user_confirmed',
        conversation_text: request.reference_text, conversation_source: 'user_confirmed',
        comparison: comparisons(request.reference_text) }, reference_update: { changed, regenerated_reply: changed } };
    return route.fulfill({ json: job });
  });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click();
    await page.locator('#chat-settings > summary').click(); await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    await page.evaluate(() => { void window.MDDChat.sendAudio(new ArrayBuffer(44), 'mock.wav'); });
    await page.waitForFunction(() => window.MDDChat.getTurns()[0]?.pronunciationReferenceReady === false);
    assert.equal(await page.locator('.chat-message.user .chat-message-text').textContent(), '');
    assert.equal((await page.locator('#chat-messages').textContent()).includes(raw), false);
    assert.equal(voices.length, 0);
    checks.push('Pending recovery hides raw ASR and does not request sentence audio.');

    job = { ...job, status: 'coaching', text: 'What made the journey difficult?', speech: { ...job.speech,
      reference_text: recovered, reference_source: 'bert_inferred', pronunciation_reference_ready: true,
      conversation_source: 'asr_interpreted', conversation_text: recovered, comparison: comparisons(recovered),
      interpretation: { status: 'accepted', text: recovered, requires_confirmation: true }, reply_ready: true } };
    await page.waitForFunction(() => window.MDDChat.getTurns()[0]?.interpretedText === 'It was difficult to get there.');
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1)?.sentenceVoice?.status === 'ready');
    assert.equal(await page.locator('.chat-message.user .chat-message-text').textContent(), recovered);
    assert.equal(await page.locator('.chat-message.user .chat-message-meta').textContent(), '');
    assert.equal(await page.locator('.chat-message.assistant .chat-message-text').textContent(), job.text);
    assert.deepEqual(voices, [{ target: 'sentence', reference: recovered, ready: true }]);
    assert.equal(await page.evaluate(() => window.__voicePlays.length), 0, 'Preparation must not autoplay.');
    checks.push('The BERT sentence is visible immediately and its voice is prepared while coaching continues.');

    job.status = 'ready'; await page.waitForFunction(() => !window.MDDChat.state.busy);
    const feedback = page.locator('.chat-turn-feedback'); await feedback.locator(':scope > summary').click();
    assert.equal(await feedback.locator('.chat-target-text').textContent(), recovered);
    assert.equal(await feedback.locator('.chat-sentence-text').textContent(), recovered);
    assert.deepEqual(await feedback.locator('.chat-sound-word > summary').allTextContents(), recovered.split(/\s+/));
    assert.equal(await feedback.locator('.chat-reference-input').inputValue(), recovered);
    assert.equal((await page.locator('#chat-messages').textContent()).includes(raw), false);
    assert.equal(/ASR gốc|AI hiểu là/.test(await page.locator('#chat-messages').textContent()), false);
    assert.equal(await feedback.locator('.chat-feedback-label').textContent(), '1 từ cần kiểm tra');
    await feedback.locator('.chat-sentence-listen').click();
    await page.waitForFunction(() => window.__voicePlays.length === 1);
    assert.equal(voices.length, 1, 'Listen reuses the reference audio prepared earlier.');
    checks.push('Bubble, feedback, per-word sounds, editor and whole-sentence audio all use BERT; raw ASR is absent.');

    await feedback.locator('.chat-reference-editor > summary').click();
    await feedback.locator('.chat-reference-apply').click();
    await page.waitForFunction(() => window.MDDChat.getTurns()[0]?.confirmedSource === 'user_confirmed');
    assert.equal(references.length, 1); assert.equal(references[0].reference_text, recovered);
    assert.equal(await page.locator('.chat-message.assistant .chat-message-text').textContent(), 'What made the journey difficult?');
    const corrected = 'It was easy to get there.';
    await feedback.locator('.chat-reference-input').fill(corrected); await feedback.locator('.chat-reference-apply').click();
    await page.waitForFunction(() => window.MDDChat.getTurns()[0]?.confirmedText === 'It was easy to get there.' && !window.MDDChat.state.busy);
    assert.equal(await page.locator('.chat-message.user .chat-message-text').textContent(), corrected);
    assert.equal(await feedback.locator('.chat-target-text').textContent(), corrected);
    assert.equal(await feedback.locator('.chat-reference-input').inputValue(), corrected);
    assert.equal((await page.locator('#chat-messages').textContent()).includes(raw), false);
    checks.push('Manual correction remains available; unchanged BERT text does not regenerate the reply and changed text updates the reference.');
    assert.deepEqual(exceptions, []);
    const report = { status: 'passed', checks, browser_exceptions: exceptions, mocked_voice_requests: voices.length,
      paid_tts_calls: 0, inference: 'mocked jobs, real browser UI' };
    const output = path.resolve('data/generated/bert-reference-qa'); fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
