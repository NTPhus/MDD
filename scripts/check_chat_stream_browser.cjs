'use strict';
// Deterministic browser checks: fake incremental jobs and speech boundaries, real UI/audio decoding.
const { chromium } = require(process.env.MDD_PLAYWRIGHT_MODULE || '../.tools/ui-qa/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/chat-stream-qa');
const reply = 'I am learning to speak English. It gets easier with practice.';

// Shared by the real-inference check too. This verifies calls/order, not a physical speaker.
function mockVoice({ preserveHistory = false } = {}) {
  if (!preserveHistory) { sessionStorage.removeItem('am-voice-chat-v2'); sessionStorage.removeItem('am-chat-selection-v1'); }
  window.__speech = []; window.__utterances = [];
  const timers = new Set();
  const later = (fn, ms) => { const id = setTimeout(() => { timers.delete(id); fn(); }, ms); timers.add(id); };
  Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
    cancel() { for (const id of timers) clearTimeout(id); timers.clear(); }, getVoices() { return []; },
    speak(utterance) {
      window.__utterances.push(utterance);
      window.__speech.push({ text: utterance.text, lang: utterance.lang, status: window.MDDChat?.state.activeTurn?.status || 'idle', time: performance.now() });
      later(() => utterance.onstart?.(), 1);
      [...utterance.text.matchAll(/\S+/g)].forEach((word, index) => later(() => utterance.onboundary?.({ name: 'word', charIndex: word.index }), 20 + index * 115));
      later(() => utterance.onend?.(), 60 + utterance.text.split(/\s+/).length * 115);
    },
  }});
}
module.exports = { mockVoice };

async function checkOpeningVoice(browser, checks) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = []; let job, serial = 0, voiceRequests = 0, voiceError = false;
  const opening = 'Hello! What do you enjoy doing after work?';
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__openingAudio = []; window.__openingPlays = [];
    window.Audio = class extends EventTarget {
      constructor(src) { super(); Object.assign(this, { src, readyState: 0, paused: true, currentTime: 0, duration: 3 }); window.__openingAudio.push(this); }
      load() { this.loadedAt = performance.now(); }
      ready() { this.readyState = 4; this.readyAt = performance.now(); this.dispatchEvent(new Event('canplay')); }
      play() {
        this.paused = false;
        window.__openingPlays.push({ text: document.querySelector('.chat-message.assistant .chat-message-text')?.textContent,
          readyDelayMs: performance.now() - this.readyAt });
        this.onplaying?.(); return Promise.resolve();
      }
      pause() { this.paused = true; this.onpause?.(); }
    };
  });
  await page.route('**/api/chat/config', async route => {
    const response = await route.fetch(); return route.fulfill({ json: { ...await response.json(), default_voice_provider: 'ai33' } });
  });
  await page.route('**/api/chat/topic', route => {
    job = { id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status: 'ready', text: opening,
      metadata: { input: 'topic', source: 'topic_catalog' }, voice: { status: 'not_requested' } };
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/voice', route => {
    voiceRequests++;
    if (voiceError) return route.fulfill({ status: 500, json: { message: 'Opening voice test error.' } });
    if (job.voice.status !== 'ready') job.voice = { status: 'generating' };
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  const text = page.locator('.chat-message.assistant .chat-message-text').last();
  const newChat = async ({ autoRead = true, provider = 'ai33' } = {}) => {
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-voice-provider').selectOption(provider);
    await page.locator('#chat-auto-read').setChecked(autoRead);
    await page.locator('#chat-new').click(); await page.keyboard.press('Escape');
    await page.waitForFunction(() => window.MDDChat.getTurns().length === 1);
  };
  const readyVoice = async () => {
    job.voice = { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) };
    await page.waitForFunction(() => window.__openingAudio.length > 0 && window.MDDChat.state.audio === window.__openingAudio.at(-1));
  };
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('.chat-topic-card').first().click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => window.MDDChat.state.openingPending?.voice?.status === 'generating');
    assert.equal(await text.textContent(), ''); assert.equal(voiceRequests, 1);
    assert.match(await page.locator('.chat-message-meta').last().textContent(), /Đang chuẩn bị/);
    await readyVoice(); assert.equal(await text.textContent(), '');
    await page.evaluate(() => window.__openingAudio.at(-1).ready());
    await page.waitForTimeout(100); assert.equal(await text.textContent(), '', 'Opening must wait for its settling pause.');
    await page.waitForFunction(() => window.__openingPlays.length === 1);
    assert.equal(await text.textContent(), opening);
    const firstPlay = await page.evaluate(() => window.__openingPlays[0]);
    assert.equal(firstPlay.text, opening); assert.ok(firstPlay.readyDelayMs >= 340);
    assert.equal(voiceRequests, 1, 'Buffering and presentation must not resubmit voice.');
    await page.evaluate(() => { const audio = window.__openingAudio.at(-1); audio.currentTime = 2; audio.ontimeupdate?.(); });
    assert.ok(await text.locator('.is-read').count() > 1, 'Cumulative word highlighting still advances.');
    checks.push('Opening stays behind voice preparation until audio is playable, then shows the entire sentence and plays after 350 ms with cumulative highlighting.');
    await page.locator('#chat-stop-voice').click();
    await page.locator('.chat-message.assistant .chat-message-actions button').last().click();
    await page.waitForFunction(() => window.__openingPlays.length === 2);
    assert.equal(await page.evaluate(() => window.MDDChat.state.openingPending), null);
    assert.equal(await page.evaluate(() => window.__openingAudio.at(-1).loadedAt), undefined,
      'Manual replay must bypass the opening preload/settling gate.');
    checks.push('Manual replay uses the ready voice and does not repeat the opening presentation delay.');
    const beforeSilent = voiceRequests;
    await newChat({ autoRead: false });
    assert.equal(await text.textContent(), opening); assert.equal(voiceRequests, beforeSilent);
    checks.push('Opening text is immediate when automatic reading is off.');
    voiceError = true; await newChat();
    await page.waitForFunction(() => document.getElementById('chat-error').textContent.includes('Opening voice test error'));
    assert.equal(await text.textContent(), opening); voiceError = false;
    checks.push('Opening voice errors reveal the complete text and leave manual replay available.');
    await newChat(); await readyVoice();
    const beforeStop = await page.evaluate(() => window.__openingPlays.length);
    await page.locator('#chat-stop-voice').click();
    assert.equal(await text.textContent(), opening);
    await page.evaluate(() => window.__openingAudio.at(-1).ready()); await page.waitForTimeout(450);
    assert.equal(await page.evaluate(() => window.__openingPlays.length), beforeStop);
    checks.push('Stopping during opening buffering reveals text and prevents late playback.');
    await newChat(); await readyVoice();
    await page.evaluate(() => window.__openingAudio.at(-1).ready());
    await page.waitForTimeout(50); await page.locator('#chat-end').click();
    await page.waitForTimeout(400);
    assert.equal(await page.evaluate(() => window.__openingPlays.length), beforeStop);
    checks.push('Leaving during the opening settling pause prevents stale playback.');
    // Return to the same topic and explicitly select browser speech for a new opening.
    await page.locator('.chat-topic-card').first().click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    await page.evaluate(() => {
      window.speechSynthesis.speak = utterance => { window.__utterances.push(utterance); };
    });
    await newChat({ provider: 'browser' });
    assert.equal(await text.textContent(), '');
    await page.evaluate(() => window.__utterances.at(-1).onstart?.());
    assert.equal(await text.textContent(), opening);
    checks.push('Browser voice fallback reveals the complete opening at its speech-start event.');
    assert.deepEqual(errors, []);
    return voiceRequests;
  } finally { await page.close(); }
}

async function checkSentencePrefetch(browser, checks, fixture, sample) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [], requests = []; let job;
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__voiceBuffers = []; window.__voicePlays = [];
    window.Audio = class extends EventTarget {
      constructor(src) { super(); Object.assign(this, { src, paused: true, currentTime: 0, duration: 4, readyState: 4 }); window.__voiceBuffers.push(this); }
      load() { this.loaded = true; }
      play() { this.paused = false; window.__voicePlays.push(this.src); this.onplaying?.(); return Promise.resolve(); }
      pause() { this.paused = true; this.onpause?.(); }
    };
  });
  await page.route('**/api/chat/config', async route => {
    const response = await route.fetch();
    return route.fulfill({ json: { ...await response.json(), default_voice_provider: 'ai33' } });
  });
  await page.route('**/api/chat/speech', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/voice', route => {
    const payload = route.request().postDataJSON(), target = payload.target || 'reply';
    requests.push({ target, revision: job.metadata.reference_revision, voice_id: payload.voice_id, speed: payload.speed });
    if (target === 'sentence') job.sentence_voice = { status: 'generating' };
    else job.voice = { status: 'ready', url: '/api/chat/audio/' + 'c'.repeat(32) };
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/reference', route => {
    const reference = route.request().postDataJSON().reference_text;
    job = { ...job, status: 'generating', text: '', metadata: { reference_revision: 1 },
      voice: { status: 'not_requested' }, sentence_voice: { status: 'not_requested' },
      reference_update: { changed: true, regenerated_reply: true }, speech: { ...job.speech,
        reference_text: reference, conversation_text: reference, reference_source: 'user_confirmed', reply_ready: false,
        rewrite: { mode: 'grammar_and_natural', status: 'pending' } } };
    return route.fulfill({ json: job });
  });
  const feedback = page.locator('.chat-turn-feedback').last();
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click();
    job = { id: 'd'.repeat(32), status: 'generating', text: '', metadata: { reference_revision: 0 },
      voice: { status: 'not_requested' }, sentence_voice: { status: 'not_requested' },
      speech: { ...structuredClone(fixture.speech), transcript: 'I usually wake up at seven.', reference_text: 'I usually wake up at seven.',
        reference_source: 'asr', reply_ready: false, rewrite: { mode: 'grammar_and_natural', status: 'pending' } } };
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1)?.sentenceVoice?.status === 'generating');
    assert.deepEqual(requests.map(item => item.target), ['sentence'], 'Sentence preparation starts from the ASR snapshot before the reply is ready.');
    assert.equal(await page.evaluate(() => window.__voicePlays.length), 0);
    assert.equal(await feedback.locator('.chat-sentence-listen').isDisabled(), true);
    assert.equal(await feedback.locator('.chat-sentence-listen').getAttribute('aria-busy'), 'true');
    job.sentence_voice = { status: 'ready', url: '/api/chat/audio/' + 'a'.repeat(32) };
    await page.waitForFunction(() => window.__voiceBuffers.some(audio => audio.loaded));
    assert.equal(await page.evaluate(() => window.__voicePlays.length), 0, 'Prefetch must never autoplay.');
    job.text = reply; job.status = 'correcting'; job.speech.reply_ready = true;
    await page.waitForFunction(() => window.__voicePlays.length === 1);
    assert.equal(await page.evaluate(() => window.__voicePlays[0]), '/api/chat/audio/' + 'c'.repeat(32));
    assert.equal(job.speech.rewrite.status, 'pending');
    assert.equal(await page.evaluate(() => window.MDDChat.state.speaking === window.MDDChat.getTurns().at(-1).id), true);
    job.speech.rewrite = { mode: 'grammar_and_natural', status: 'ready', grammar_status: 'ready', natural_status: 'ready',
      grammar_corrected: job.speech.transcript, natural: job.speech.transcript, changed: false, feedback_state: 'no_change_suggested' };
    job.status = 'ready'; await page.waitForFunction(() => !window.MDDChat.state.busy);
    assert.equal(requests.filter(item => item.target === 'sentence').length, 1);
    assert.equal(requests.filter(item => item.target === 'reply').length, 1);
    assert.equal(await page.evaluate(() => window.__voicePlays.length), 1, 'Late feedback must neither replay nor interrupt the bot.');
    await feedback.locator(':scope > summary').click();
    const listen = feedback.locator('.chat-sentence-listen');
    assert.equal(await listen.textContent(), 'Nghe cả câu');
    await listen.click(); await page.waitForFunction(() => window.__voicePlays.length === 2);
    assert.equal(requests.length, 2, 'Listening to a prefetched sentence must not POST another provider request.');
    assert.equal(await page.evaluate(() => window.__voicePlays[1]), '/api/chat/audio/' + 'a'.repeat(32));
    assert.equal(await page.locator('.chat-word-timing:not([hidden])').count(), 0);
    assert.equal(await page.locator('#chat-messages').textContent().then(value => /Đánh dấu từ ước lượng|Đang tạo giọng/.test(value)), false);
    checks.push('Sentence audio is generated and buffered from the first ASR snapshot while feedback is pending; it never autoplays or interrupts the bot, and Listen reuses it with no additional TTS request.');
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    await feedback.locator('.chat-reference-editor > summary').click();
    await feedback.locator('.chat-reference-input').fill('I usually wake up at eight.');
    await feedback.locator('.chat-reference-apply').click();
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1).referenceRevision === 1
      && window.MDDChat.getTurns().at(-1).sentenceVoice?.status === 'generating');
    assert.equal(requests.filter(item => item.target === 'sentence' && item.revision === 1).length, 1);
    const beforeRevisedReady = await page.evaluate(() => window.__voicePlays.length);
    job.sentence_voice = { status: 'ready', url: '/api/chat/audio/' + 'b'.repeat(32) };
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1).sentenceVoice?.url?.endsWith('b'.repeat(32)));
    job.status = 'ready'; job.text = 'Eight sounds like a good time. Do you usually have breakfast first?'; job.speech.reply_ready = true;
    await page.waitForFunction(() => !window.MDDChat.state.busy);
    assert.equal(await page.evaluate(() => window.__voicePlays.length), beforeRevisedReady, 'Read aloud off still prefetches but never autoplays.');
    await feedback.locator('.chat-sentence-listen').click();
    await page.waitForFunction(count => window.__voicePlays.length > count, beforeRevisedReady);
    assert.equal(await page.evaluate(() => window.__voicePlays.at(-1)), '/api/chat/audio/' + 'b'.repeat(32), 'Confirmed text must use its own prepared revision, never the old sentence.');
    assert.equal(requests.length, 3); assert.deepEqual(errors, []);
    checks.push('A confirmed text change prepares exactly one new sentence voice; Read aloud off prevents autoplay and the Listen action plays the revised buffer, not stale audio.');
    return requests.length;
  } finally { await page.close(); }
}

async function checkInterpretation(browser, checks, fixture) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [], submissions = [], references = []; let job;
  const original = 'It was the divicote true getesir', inferred = 'It was difficult to get there.';
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(mockVoice);
  await page.route('**/api/chat/config', async route => {
    const response = await route.fetch();
    return route.fulfill({ json: { ...await response.json(), default_voice_provider: 'browser' } });
  });
  await page.route('**/api/chat/speech', route => { submissions.push(route.request().postDataJSON()); return route.fulfill({ json: job }); });
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/voice', () => { throw new Error('This check must not call a voice provider.'); });
  let first;
  await page.route('**/api/chat/reference', route => {
    const request = route.request().postDataJSON(); references.push(request);
    const speech = { ...first.speech, reference_text: request.reference_text, reference_source: 'user_confirmed', conversation_source: 'user_confirmed',
      conversation_text: request.reference_text, comparison: { words: request.reference_text.split(/\s+/).map(text => ({ text, status: 'match' })), events: [] } };
    return route.fulfill({ json: { ...first, speech, reference_update: { changed: false, regenerated_reply: false } } });
  });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click();
    await page.locator('#chat-settings > summary').click(); await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    const speech = { ...structuredClone(fixture.speech), transcript: original, reference_text: original, reference_source: 'asr',
      conversation_source: 'asr_interpreted', conversation_text: inferred, reply_ready: true, sound_candidates: null,
      comparison: { words: original.split(/\s+/).map(text => ({ text, status: 'uncertain' })), events: [] },
      interpretation: { status: 'accepted', text: inferred, changed: true, requires_confirmation: true, source: 'qwen_inferred' } };
    job = first = { id: 'e'.repeat(32), status: 'ready', text: 'What made the journey difficult?', metadata: { reference_revision: 0 }, speech };
    await page.evaluate(() => window.MDDChat.sendAudio(new ArrayBuffer(44), 'mock.wav'));
    const user = page.locator('.chat-message.user').first(), feedback = page.locator('.chat-turn-feedback').first();
    assert.equal(await user.locator('.chat-message-text').textContent(), inferred);
    assert.equal(await user.locator('.chat-message-meta').textContent(), 'AI hiểu là');
    assert.equal(await user.locator('[data-evidence],.chat-word-practice,.chat-sound-word').count(), 0,
      'Raw-reference pronunciation marks cannot be attached to inferred words.');
    await feedback.locator(':scope > summary').click();
    assert.equal(await feedback.locator('.chat-turn-review > .chat-review-label').first().textContent(), 'ASR gốc');
    assert.equal(await feedback.locator('.chat-target-text').textContent(), original);
    assert.equal(await feedback.locator('.chat-sentence-text').textContent(), original,
      'The sentence voice remains bound to the raw acoustic reference until explicit confirmation.');
    assert.deepEqual(await feedback.locator('.chat-sound-word > summary').allTextContents(), original.split(/\s+/));
    assert.equal(await feedback.locator('.chat-reference-input').inputValue(), inferred);
    assert.equal(await feedback.locator('.chat-reference-apply').textContent(), 'Xác nhận câu này');
    assert.equal(references.length, 0, 'An interpretation must never auto-confirm itself.');
    checks.push('Accepted Qwen interpretation is shown and labelled separately; raw ASR, sound tokens and sentence audio stay in feedback, and confirmation remains explicit.');
    job = { ...first, id: 'f'.repeat(32), text: 'Could you say that again?', speech: { ...speech, transcript: 'And he dishurt my m',
      reference_text: 'And he dishurt my m', conversation_source: 'asr', conversation_text: 'And he dishurt my m',
      interpretation: { status: 'ambiguous', text: 'And he hurt my arm.', requires_confirmation: true } } };
    await page.evaluate(() => window.MDDChat.sendAudio(new ArrayBuffer(44), 'second.wav'));
    assert.equal(submissions[1].history[0].content, inferred, 'Later chat history must use the same inferred text the bot received.');
    assert.equal(await page.locator('.chat-message.user .chat-message-text').last().textContent(), 'And he dishurt my m');
    assert.equal(await page.locator('.chat-message.user .chat-message-meta').last().textContent(), '');
    await feedback.locator('.chat-reference-editor > summary').click();
    await feedback.locator('.chat-reference-apply').click();
    await page.waitForFunction(() => window.MDDChat.getTurns()[0].confirmedSource === 'user_confirmed');
    assert.equal(references.length, 1); assert.equal(references[0].reference_text, inferred);
    assert.equal(await user.locator('.chat-message-meta').textContent(), '');
    assert.equal(await feedback.locator('.chat-target-text').textContent(), original, 'Confirmation must preserve the original ASR.');
    assert.equal(await feedback.locator('.chat-confirmed-text').textContent(), inferred);
    assert.equal(await feedback.locator('.chat-sentence-text').textContent(), inferred);
    assert.deepEqual(errors, []);
    checks.push('Inferred chat history remains consistent, ambiguous alternatives leave raw text alone, and a no-regeneration confirmation updates only the explicit pronunciation reference.');
  } finally { await page.close(); }
}

async function run() {
  const fixture = JSON.parse(fs.readFileSync('data/generated/voice-chat-qa/browser-summary.json', 'utf8')).jobs[0].result;
  const sample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8').trim().split('\n').map(JSON.parse).find(row => row.utterance_id === 'NJS/arctic_a0054');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const exceptions = [], checks = []; let job, serial = 0, voiceRequests = 0;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  // This suite exercises browser speech; production may default to Sarah/ai33.
  await page.route('**/api/chat/config', async route => {
    const response = await route.fetch();
    return route.fulfill({ json: { ...await response.json(), default_voice_provider: 'browser' } });
  });
  const fresh = (text = 'I am learning to speak ') => {
    const speech = structuredClone(fixture.speech); speech.coaching = { status: 'pending' }; speech.reply_ready = false; delete speech.spoken_response;
    return job = { id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status: 'generating', text, speech, metadata: {}, error: null };
  };
  await page.route('**/api/chat/speech', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/cancel', route => { job.status = 'cancelled'; return route.fulfill({ json: job }); });
  await page.route('**/api/chat/voice', route => { voiceRequests++; return route.fulfill({ status: 500, json: { message: 'Paid TTS is blocked in QA.' } }); });
  const text = page.locator('.chat-message.assistant .chat-message-text').last();
  const feedback = page.locator('.chat-turn-feedback').last();
  const soundDetails = feedback.locator('.chat-feedback-content > .chat-pronunciation:not(.chat-completion-extra)');
  const newConversation = async () => {
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-new').click();
    await page.keyboard.press('Escape');
  };
  const upload = async () => {
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
    await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.state.activeTurn?.status === 'generating');
  };
  const finish = async () => {
    job.text = reply; job.status = 'ready'; job.speech.spoken_response = reply + ' Say English slowly.';
    job.speech.coaching = { status: 'unavailable', message: 'No validated coaching for this fixture.' };
    await page.waitForFunction(() => !window.MDDChat.state.busy);
    await page.waitForFunction(() => !window.MDDChat.state.liveReply || window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
  };
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    assert.equal(await page.locator('#chat-settings').getAttribute('open'), null);
    assert.equal(await page.locator('#chat-view textarea').count(), 0);
    await page.locator('#chat-free-talk').click();
    await page.waitForFunction(() => document.body.classList.contains('chat-room-open'));
    await page.screenshot({ path: path.join(output, 'desktop-empty.png'), fullPage: true, animations: 'disabled' });
    fresh(); await upload();
    assert.equal(await text.textContent(), '', 'An unfinished reply should stay behind the generating status.');
    assert.equal(await page.evaluate(() => window.__speech.length), 0);
    checks.push('Generation stays behind the status; no text animation or early voice.');
    job.text = 'I am learning to speak English. It ';
    await page.waitForFunction(value => window.MDDChat.state.activeTurn?.content === value, job.text);
    assert.equal(await text.textContent(), ''); assert.equal(await page.evaluate(() => window.__speech.length), 0);
    job.text = reply; job.status = 'correcting'; job.speech.reply_ready = true;
    job.speech.rewrite = { mode: 'grammar_and_natural', status: 'pending' };
    await page.waitForFunction(() => window.__speech.length === 1);
    assert.equal(job.speech.rewrite.status, 'pending');
    assert.equal(job.speech.coaching.status, 'pending');
    assert.equal(await text.textContent(), reply, 'All sentences must be visible before speech completes.');
    assert.equal(await page.evaluate(() => window.__speech[0].status), 'correcting');
    await page.waitForFunction(() => document.querySelector('.chat-reply-word.is-speaking'));
    await page.screenshot({ path: path.join(output, 'desktop-speaking.png'), fullPage: true, animations: 'disabled' });
    checks.push('The completed reply is shown in full before voice, then only the spoken word is highlighted.');
    await page.waitForFunction(() => window.__speech.length === 2);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await feedback.getAttribute('open'), null);
    job.speech.rewrite = { mode: 'grammar_and_natural', status: 'ready', natural: fixture.speech.transcript };
    await finish(); await page.waitForTimeout(150);
    assert.deepEqual(await page.evaluate(() => window.__speech.map(s => s.text)), ['I am learning to speak English.', 'It gets easier with practice.']);
    checks.push('Late pronunciation tips are never read automatically; only the bot reply is spoken.');
    assert.equal(await feedback.getAttribute('open'), null);
    assert.equal(await feedback.locator('.chat-feedback-label').textContent(), 'Chưa đủ bằng chứng');
    assert.equal(await soundDetails.isVisible(), false);
    await feedback.locator(':scope > summary').click();
    assert.ok(await soundDetails.isVisible());
    assert.equal(await soundDetails.locator('.chat-sound-word.difference').count(), 0,
      'An unconfirmed ASR transcript must not assign sound errors.');
    await soundDetails.locator('.chat-sound-word.uncertain summary').first().click();
    assert.ok(await soundDetails.locator('.chat-sound-word[open] .chat-sound-detail').first().isVisible());
    assert.equal(await feedback.locator('.chat-completion-extra').getAttribute('open'), null);
    checks.push('Pronunciation stays behind the closed feedback bar; opening it reveals sounds, with optional completion still closed.');
    await page.screenshot({ path: path.join(output, 'desktop-result.png'), fullPage: true, animations: 'disabled' });
    for (const width of [1440, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Overflow at ${width}`);
      assert.ok(await text.evaluate(node => parseFloat(getComputedStyle(node).fontSize)) >= 15, `Small message text at ${width}`);
      await page.locator('#chat-settings>summary').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Settings overflow at ${width}`);
      await page.keyboard.press('Escape');
      if (width === 390) await page.screenshot({ path: path.join(output, 'mobile-result.png'), fullPage: true, animations: 'disabled' });
    }
    checks.push('Readable compact type and no horizontal overflow at 1440/768/390/360, including Settings.');
    await page.setViewportSize({ width: 1440, height: 1080 });
    await newConversation(); fresh('First sentence.'); await upload();
    job.status = 'coaching'; job.speech.reply_ready = true;
    await page.waitForFunction(() => window.MDDChat.state.speaking);
    const beforeStop = await page.evaluate(() => window.__speech.length);
    await page.locator('#chat-stop-voice').click();
    await page.evaluate(() => window.__utterances.at(-1).onend?.()); // Simulate a late device callback.
    await finish(); await page.waitForTimeout(300);
    assert.equal(await page.evaluate(() => window.__speech.length), beforeStop); assert.equal(await text.textContent(), reply);
    checks.push('Stop voice cancels queued and late audio while the remaining text still appears.');
    await newConversation(); fresh('A short reply.'); await upload();
    job.status = 'coaching'; job.speech.reply_ready = true;
    await page.waitForFunction(() => window.MDDChat.state.speaking);
    const beforeLeave = await page.evaluate(() => window.__speech.length);
    // The full-screen room hides the workspace sidebar; exercise its view hook.
    await page.evaluate(() => window.switchView('practice'));
    assert.ok(await page.locator('#practice-view').isVisible());
    await finish(); await page.locator('.navigation [data-view=chat]').click();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeLeave);
    checks.push('Leaving the chat prevents late voice playback, including after returning.');
    await newConversation(); fresh('Another reply.'); await upload();
    job.status = 'coaching'; job.speech.reply_ready = true;
    await page.waitForFunction(() => window.MDDChat.state.speaking);
    const beforeCancel = await page.evaluate(() => window.__speech.length);
    await page.locator('#chat-stop').click(); await page.waitForFunction(() => !window.MDDChat.state.busy);
    assert.equal(await page.evaluate(() => window.__speech.length), beforeCancel);
    assert.equal(await page.evaluate(() => window.MDDChat.state.speaking), null);
    checks.push('Stop AI also stops speech and leaves a cancelled result.');
    await newConversation();
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    const beforeSilent = await page.evaluate(() => window.__speech.length);
    fresh(); await upload(); await finish();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeSilent);
    await page.locator('.chat-message-actions button').last().click();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeSilent + 1);
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), reply);
    checks.push('Read aloud off shows the full completed reply; manual replay reads only the reply, without legacy tips.');
    await newConversation();
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-voice-provider').selectOption('ai33');
    await page.locator('#chat-auto-read').check(); await page.keyboard.press('Escape');
    fresh(); await upload();
    job.text = reply; job.status = 'correcting'; job.speech.reply_ready = true;
    job.speech.rewrite = { mode: 'grammar_and_natural', status: 'pending' };
    await page.waitForFunction(() => document.getElementById('chat-error').textContent.includes('Paid TTS is blocked'));
    assert.equal(voiceRequests, 2, 'Reply voice and sentence prefetch must both be requested before feedback finishes.');
    assert.equal(await text.textContent(), reply);
    assert.equal(job.speech.rewrite.status, 'pending');
    await finish(); await page.waitForTimeout(800);
    assert.equal(voiceRequests, 2, 'Late feedback must not automatically repeat either voice request.');
    checks.push('ai33 reply and sentence preparation start while grammar is pending; feedback completion never submits either voice twice (provider mocked).');
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-voice-provider').selectOption('browser');
    await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-mode').selectOption('reading'); await page.keyboard.press('Escape');
    assert.ok(await page.locator('#chat-reading-card').isVisible());
    assert.ok((await page.locator('#chat-reading-text').textContent()).length > 10);
    await page.locator('#chat-reading-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), await page.locator('#chat-reading-text').textContent());
    await page.setViewportSize({ width: 360, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Reading mode overflows on mobile');
    await page.screenshot({ path: path.join(output, 'mobile-reading.png'), fullPage: true, animations: 'disabled' });
    checks.push('The reading sentence and its Listen button remain visible after closing Settings.');
    assert.equal(voiceRequests, 2); assert.deepEqual(exceptions, []);
    const openingVoiceRequests = await checkOpeningVoice(browser, checks);
    const sentenceVoiceRequests = await checkSentencePrefetch(browser, checks, fixture, sample);
    await checkInterpretation(browser, checks, fixture);
    const report = { status: 'passed', checks, exceptions, inference: 'mock jobs', speech: 'mock English speech boundaries; physical audio not tested', mock_voice_requests: voiceRequests + openingVoiceRequests + sentenceVoiceRequests, opening_mock_voice_requests: openingVoiceRequests, sentence_mock_voice_requests: sentenceVoiceRequests, paid_tts_calls: 0 };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    const snapshot = await page.evaluate(() => ({ errors: document.getElementById('chat-error').textContent,
      busy: window.MDDChat?.state.busy, visible: window.MDDChat?.state.visible,
      status: window.MDDChat?.getTurns().at(-1)?.status, reply: window.MDDChat?.getTurns().at(-1)?.content,
      speechCalls: window.__speech.map(call => call.text) }));
    console.error(JSON.stringify(snapshot, null, 2));
    console.error('Browser exceptions:', exceptions); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
