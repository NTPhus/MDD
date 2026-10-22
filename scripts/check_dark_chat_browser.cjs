'use strict';
// UI-only layout and interaction checks. All chat jobs and voices are fixtures:
// this script never starts training, runs model inference, or uses paid TTS.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { mockVoice } = require('./check_chat_stream_browser.cjs');

const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/light-chat-qa');
const original = 'Japan';
const corrected = 'I would love to visit Japan.';
const reply = 'Japan sounds exciting. What would you like to do there?';
const config = {
  speech_ready: true, default_model_id: 'qwen-trained', device: 'cpu', threads: 2,
  ai33_available: false, voice_locales: ['en-US'], max_audio_seconds: 30,
  models: [{ id: 'qwen-base', ready: true }, { id: 'qwen-trained', ready: true }],
  practice_prompts: [{ id: 'idea', text: 'I think this is a good idea.' }],
  topics: [
    { id: 'travel', title: 'Travel', icon: 'plane', level: 'A2', description: 'Plan a journey.', guidance: { keywords: ['Japan', 'visit', 'holiday'], starter: 'I would love to visit…' } },
    { id: 'daily-life', title: 'Daily life', icon: 'sun', level: 'A1–A2', description: 'Talk about your day.', guidance: { keywords: ['park', 'my friend'], starter: 'Yesterday, I…' } },
  ],
};
function speechFixture() {
  return {
    transcript: original, reference_text: original, reference_source: 'asr', missing_words: [],
    comparison: {
      words: original.split(' ').map((text, index) => ({ text, status: index === 0 ? 'difference' : 'match' })),
      events: [{ kind: 'substitution', expected_phone: 'JH', observed_phone: 'CH', word_index: 0 }],
    },
    target_words: original.split(' ').map(text => ({ text, phones: ['JH', 'AH0', 'P', 'AE1', 'N'] })),
    phone_symbols: { JH: 'dʒ', CH: 'tʃ' },
    coaching: { status: 'ready', feedback: { Japan: [{ issue: 'A possible JH/CH difference was detected.', suggestion: 'Practise the voiced JH sound in Japan.' }] } },
    correction: { status: 'pending', mode: 'completion_only', original },
    guidance: config.topics[0].guidance,
  };
}
function threeWordFixture() {
  const transcript = 'I added this with genuine interest.';
  const words = transcript.split(' '), flagged = new Set([1, 3, 4]);
  const phones = [['AY1'], ['AE1', 'D', 'IH0', 'D'], ['DH', 'IH1', 'S'], ['W', 'IH1', 'DH'],
    ['JH', 'EH1', 'N', 'Y', 'UW0', 'AH0', 'N'], ['IH1', 'N', 'T', 'R', 'AH0', 'S', 'T']];
  return {
    transcript, reference_text: transcript, reference_source: 'user_confirmed', missing_words: [],
    comparison: { words: words.map((text, index) => ({ text, status: flagged.has(index) ? 'difference' : 'match' })),
      events: [{ kind: 'substitution', word_index: 1, expected_phone: 'AE', observed_phone: 'EH' },
        { kind: 'substitution', word_index: 3, expected_phone: 'DH', observed_phone: 'D' },
        { kind: 'deletion', word_index: 4, expected_phone: 'JH' }] },
    target_words: words.map((text, index) => ({ text, phones: phones[index].map(phone => phone.replace(/\d/g, '')), pronunciation_phones: phones[index] })),
    phone_symbols: { AE: 'æ', EH: 'ɛ', IH: 'ɪ', D: 'd', W: 'w', DH: 'ð', JH: 'dʒ', N: 'n', Y: 'j', UW: 'u', AY: 'aɪ', S: 's', T: 't', R: 'ɹ', AH: 'ʌ' },
    coaching: { status: 'pending' }, correction: { status: 'not_needed', mode: 'completion_only' },
    guidance: config.topics[0].guidance,
  };
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.MDD_QA_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const exceptions = [], requests = [], cancellations = [], checks = [], screenshots = [], viewportMeasurements = [];
  let job, serial = 0, paidCalls = 0, delayPost = false, releasePost = null;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  await page.addInitScript(() => {
    window.__mediaCalls = [];
    HTMLMediaElement.prototype.play = function () {
      window.__mediaCalls.push({ action: 'play', src: this.src }); this.dispatchEvent(new Event('play')); return Promise.resolve();
    };
    HTMLMediaElement.prototype.pause = function () { window.__mediaCalls.push({ action: 'pause', src: this.src }); };
  });
  const screenshot = async name => {
    const filename = name + '.png';
    await page.screenshot({ path: path.join(output, filename), fullPage: true, animations: 'disabled' });
    screenshots.push(filename);
  };
  const frameFeedback = () => feedback().evaluate(node => {
    const scroll = node.closest('#chat-scroll');
    scroll.scrollTop += node.getBoundingClientRect().top - scroll.getBoundingClientRect().top - 16;
  });
  const fresh = (status = 'correcting') => job = {
    id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status, text: '',
    speech: speechFixture(), metadata: {}, error: null,
  };
  const complete = () => {
    job.status = 'ready'; job.text = reply;
    job.speech.spoken_response = reply + ' You said Japan. Practise the voiced JH sound in Japan.';
    job.speech.correction = { status: 'ready', mode: 'completion_only', original, corrected, changed: true,
      explanation: 'The prior travel question supplies the visit context. <img src=x onerror=alert(1)>',
      changes: [{ before: '', after: 'I would love to visit' }], source: 'qwen', model_id: 'qwen-base' };
  };
  const roomOpen = () => page.evaluate(() => document.body.classList.contains('chat-room-open'));
  const waitDone = () => page.waitForFunction(() => !window.MDDChat.state.busy);
  const submit = () => page.evaluate(() => { window.__roomSend = window.MDDChat.sendAudio(new ArrayBuffer(48), 'ui-fixture.wav'); });
  const feedback = () => page.locator('.chat-turn-feedback').last();
  const openSettings = async () => {
    if (!(await page.locator('#chat-settings').evaluate(node => node.open))) await page.locator('#chat-settings>summary').click();
  };
  await page.route('**/api/chat/config', route => route.fulfill({ json: config }));
  await page.route('**/api/chat/topic', route => route.fulfill({ json: {
    id: 'topic-opening', status: 'ready', text: 'Hi, I am your speaking partner. Where would you love to visit?', metadata: { guidance: config.topics[0].guidance },
  } }));
  await page.route('**/api/chat/speech', async route => {
    requests.push(route.request().postDataJSON());
    if (delayPost) await new Promise(resolve => { releasePost = resolve; });
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/cancel', route => {
    cancellations.push(route.request().postDataJSON()); job.status = 'cancelled';
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/voice', route => { paidCalls++; return route.fulfill({ status: 500, json: { message: 'Paid TTS is blocked in UI QA.' } }); });
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    assert.equal(await roomOpen(), false);
    assert.ok(await page.locator('#chat-topic-picker').isVisible());
    assert.equal(await page.locator('#chat-topic-grid button').count(), 2);
    assert.equal(await page.locator('#chat-view textarea').count(), 0);
    await screenshot('topics');
    await page.locator('[data-topic=travel]').click();
    await page.waitForFunction(() => window.MDDChat.state.inRoom && !window.MDDChat.state.starting);
    assert.equal(await roomOpen(), true);
    assert.equal(await page.locator('#chat-settings').evaluate(node => Boolean(node.closest('.chat-room-toolbar'))), true);
    assert.ok(await page.locator('#chat-end').isVisible());
    assert.ok(await page.locator('#chat-topics').isVisible());
    assert.ok(await page.locator('#chat-mic-label').isVisible());
    checks.push('Topic selection enters the conversation layout with back/end controls; the topic picker remains separate.');

    // The native details element is keyboard-operable, not a visual-only menu.
    await page.locator('#chat-settings>summary').focus();
    await page.keyboard.press('Enter');
    assert.ok(await page.locator('#chat-model').isVisible());
    assert.ok(await page.locator('#chat-voice-provider').isVisible());
    await page.locator('#chat-auto-read').uncheck();
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#chat-settings').evaluate(node => node.open), false);
    checks.push('Header Settings can be opened by keyboard and exposes model, voice and read-aloud controls.');

    fresh(); await submit();
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'correcting');
    assert.ok(await feedback().isVisible());
    assert.equal(await feedback().evaluate(node => node.open), false);
    assert.doesNotMatch(await feedback().locator(':scope > summary').textContent(), /\d+\/\d+|hoàn thiện câu|giữ nguyên câu/i);
    assert.ok(await feedback().locator('.chat-correction').isHidden());
    complete(); job.status = 'coaching';
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'coaching');
    assert.equal(await feedback().evaluate(node => node.open), false);
    assert.ok(await feedback().locator('.chat-correction-practise').isDisabled());
    // Keep speech byte-for-byte unchanged; only the terminal job status changes.
    job.status = 'ready'; await waitDone();
    assert.equal(await feedback().evaluate(node => node.open), false);
    assert.equal((await feedback().locator(':scope > summary').textContent()).trim(), 'Cần xác nhận câu');
    assert.ok(await feedback().locator('.chat-correction-practise').isEnabled());
    await page.waitForFunction(() => !window.MDDChat.state.liveReply || window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
    const placement = await feedback().evaluate(node => ({
      before: node.previousElementSibling?.className,
      after: node.nextElementSibling?.className,
    }));
    assert.match(placement.before, /chat-message user/);
    assert.match(placement.after, /chat-message assistant/);
    const bubbles = await page.evaluate(() => {
      const user = [...document.querySelectorAll('.chat-message.user .chat-message-text')].at(-1).getBoundingClientRect();
      const bot = [...document.querySelectorAll('.chat-message.assistant .chat-message-text')].at(-1).getBoundingClientRect();
      return { userLeft: user.left, userRight: user.right, botLeft: bot.left, botRight: bot.right };
    });
    assert.ok(bubbles.userLeft > bubbles.botLeft + 20, 'User bubble should align to the right of the bot bubble.');
    assert.ok(bubbles.userRight > bubbles.botRight, 'User bubble should end to the right of the bot bubble.');
    assert.equal(await page.locator('.chat-message.user .chat-message-text').last().textContent(), original);
    await screenshot('desktop-room');
    checks.push('User speech appears on the right, bot speech on the left, and a collapsed feedback strip sits between the corresponding turns.');
    checks.push('Practise becomes enabled after coaching→ready even when the final speech payload has not changed.');

    const beforeOpening = await page.evaluate(() => window.__speech.length);
    await feedback().locator(':scope > summary').focus(); await page.keyboard.press('Enter');
    assert.equal(await feedback().evaluate(node => node.open), true);
    assert.ok(await feedback().locator('.chat-turn-review').isVisible());
    assert.ok(await feedback().locator('.chat-pronunciation:not(.chat-completion-extra)').isVisible());
    assert.equal(await feedback().locator('.chat-completion-extra').evaluate(node => node.open), false);
    assert.ok(await feedback().locator('.chat-correction-sentence').isHidden());
    assert.equal(await page.evaluate(() => window.__speech.length), beforeOpening, 'Opening pronunciation must not start reading it aloud.');
    await feedback().locator('.chat-sound-word.difference>summary').click();
    assert.match(await feedback().locator('.chat-sound-word[open] .chat-sound-detail').textContent(), /Target phones.*JH AH0 P AE1 N/);
    assert.match(await feedback().locator('.chat-sound-word[open] .chat-sound-detail').textContent(), /expected JH.*detected CH/);
    await feedback().locator('.chat-sound-word[open] button').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), original);
    await frameFeedback(); await screenshot('desktop-phoneme-detail');
    await feedback().locator('.chat-completion-extra>summary').click();
    assert.equal(await feedback().locator('.chat-correction-sentence').textContent(), corrected);
    assert.equal(await feedback().locator('.chat-correction-edits').textContent(), 'Add “I would love to visit”');
    assert.equal(await feedback().locator('img').count(), 0);
    assert.ok(await feedback().locator('.chat-correction-practise').isEnabled());
    await feedback().locator('.chat-correction-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), corrected);
    await frameFeedback();
    await screenshot('desktop-feedback-expanded');
    checks.push('Feedback stays closed through coaching and ready; opening shows raw pronunciation first without speaking it. Completion requires another explicit expansion; word/optional-sentence listening still works.');
    await page.locator('.chat-message.assistant .chat-message-actions button').last().click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), reply);
    checks.push('Main replay speaks only the bot reply, ignoring legacy spoken_response transcript echoes and pronunciation tips.');

    for (const width of [1440, 1024, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 980 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Room overflow at ' + width);
      assert.ok(await page.locator('#chat-mic').isVisible());
      await openSettings();
      assert.ok(await page.locator('#chat-model').isVisible());
      assert.ok(await page.locator('#chat-voice-provider').isVisible());
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Settings overflow at ' + width);
      if (width === 1440 || width === 390) await screenshot(width === 1440 ? 'desktop-settings' : 'mobile-settings');
      await page.keyboard.press('Escape');
      if (width === 390) {
        await feedback().locator(':scope > summary').click();
        await page.locator('#chat-scroll').evaluate(node => { node.scrollTop = node.scrollHeight; });
        await screenshot('mobile-room');
        await feedback().locator(':scope > summary').click(); await frameFeedback();
        await screenshot('mobile-feedback-expanded');
      }
    }
    checks.push('Room, feedback, microphone and Settings fit 1440/1024/768/390/360 without horizontal overflow.');
    await feedback().locator(':scope > summary').click();
    for (const viewport of [{ width: 1920, height: 900 }, { width: 1440, height: 900 }, { width: 1366, height: 768 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.locator('#chat-scroll').evaluate(node => { node.scrollTop = node.scrollHeight; });
      assert.equal(await feedback().evaluate(node => node.open), false);
      assert.ok(await page.locator('#chat-inline-hints').isVisible());
      const controls = await page.evaluate(() => ['chat-mic', 'chat-end', 'chat-settings', 'chat-inline-hints'].map(id => {
        const node = document.getElementById(id), rect = (id === 'chat-settings' ? node.querySelector('summary') : node).getBoundingClientRect();
        return { id, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: innerWidth, height: innerHeight };
      }));
      for (const rect of controls) {
        assert.ok(rect.left >= 0 && rect.right <= rect.width + 1 && rect.top >= 0 && rect.bottom <= rect.height + 1,
          `${rect.id} must fit the ${viewport.width}x${viewport.height} viewport: ${JSON.stringify(rect)}`);
      }
      const measurements = await page.evaluate(() => {
        const text = document.querySelector('.chat-message.assistant .chat-message-text');
        const mic = document.getElementById('chat-mic').getBoundingClientRect();
        return { width: innerWidth, height: innerHeight, outer_scroll: Math.max(0, document.documentElement.scrollHeight - innerHeight),
          message_font: parseFloat(getComputedStyle(text).fontSize), mic_size: mic.width,
          conversation_height: document.getElementById('chat-scroll').clientHeight };
      });
      assert.ok(measurements.outer_scroll <= 2, `Conversation controls must fit without page scrolling: ${JSON.stringify(measurements)}`);
      assert.ok(measurements.message_font >= 15 && measurements.message_font <= 17,
        `Conversation text must stay compact and readable: ${JSON.stringify(measurements)}`);
      viewportMeasurements.push(measurements);
      await openSettings(); await page.locator('#chat-voice-provider').scrollIntoViewIfNeeded();
      assert.ok(await page.locator('#chat-voice-provider').isVisible());
      await page.keyboard.press('Escape');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      await screenshot(`compact-room-${viewport.width}x${viewport.height}`);
    }
    checks.push('At 1920x900, 1440x900, 1366x768 and 390x844, compact text, expanded hints, microphone, End and Settings fit without outer scrolling; feedback stays collapsed and voice settings remain reachable.');
    await feedback().locator(':scope > summary').click();
    await page.setViewportSize({ width: 1440, height: 1080 });

    await feedback().locator('.chat-correction-practise').click();
    assert.equal(await page.locator('#chat-reading-text').textContent(), corrected);
    assert.ok(await page.locator('#chat-reading-card').isVisible());
    const sourceId = job.id;
    fresh('ready'); job.text = 'Good practice. What happened next?';
    job.speech.reference_source = 'corrected_sentence'; job.speech.reference_text = corrected;
    job.speech.correction = { status: 'not_needed', mode: 'completion_only' };
    await submit(); await waitDone();
    assert.equal(requests.at(-1).correction_job_id, sourceId);
    assert.equal(Object.hasOwn(requests.at(-1), 'reference_text'), false);
    assert.equal(Object.hasOwn(requests.at(-1), 'practice_id'), false);
    checks.push('Practise still sends only a server-owned correction job ID; clients cannot supply a replacement reference.');

    await page.locator('.chat-message.user .chat-message-actions button').last().click();
    assert.equal(await page.evaluate(() => Boolean(window.MDDChat.state.recordingAudio)), true);
    const pausedBefore = await page.evaluate(() => window.__mediaCalls.filter(call => call.action === 'pause').length);
    await page.locator('#chat-end').click();
    assert.equal(await page.evaluate(() => window.MDDChat.state.recordingAudio), null);
    assert.ok(await page.evaluate(() => window.__mediaCalls.filter(call => call.action === 'pause').length) > pausedBefore);
    await page.locator('[data-topic=travel]').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    checks.push('The recording replay button remains usable; End pauses the user recording and clears its playback state.');

    for (const view of ['practice', 'training']) {
      await page.evaluate(name => window.switchView(name), view);
      assert.equal(await roomOpen(), false, view + ' must keep its original layout.');
      assert.ok(await page.locator('#' + view + '-view').isVisible());
      await page.evaluate(() => window.switchView('chat'));
      assert.equal(await roomOpen(), true);
    }
    checks.push('Pronunciation and training views retain their existing layout and remain separate from the conversation room.');

    await page.locator('#chat-topics').click();
    assert.equal(await roomOpen(), false);
    assert.ok(await page.locator('#chat-topic-picker').isVisible());
    await page.setViewportSize({ width: 1440, height: 1080 });
    await page.locator('[data-topic=travel]').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    await openSettings(); await page.locator('#chat-auto-read').check(); await page.keyboard.press('Escape');

    const beforeAutomatic = await page.evaluate(() => window.__speech.length);
    fresh('generating'); job.text = reply;
    job.speech.correction = { status: 'ready', mode: 'completion_only', original, corrected, changed: true, changes: [] };
    job.speech.coaching = { status: 'pending' };
    await submit();
    await page.waitForFunction(count => window.MDDChat.state.activeTurn?.status === 'generating' && window.__speech.length > count, beforeAutomatic);
    assert.equal(await feedback().evaluate(node => node.open), false);
    await feedback().locator(':scope > summary').click();
    await feedback().locator('.chat-sound-word.difference>summary').click();
    job.speech.coaching = { status: 'ready', feedback: { Japan: [{ issue: 'Possible late sound difference.', suggestion: 'This late pronunciation tip must stay silent.' }] } };
    complete(); await waitDone();
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await feedback().evaluate(node => node.open), true);
    assert.equal(await feedback().locator('.chat-sound-word.difference').evaluate(node => node.open), true);
    assert.equal(await feedback().locator('.chat-completion-extra').evaluate(node => node.open), false);
    assert.equal(await page.evaluate(count => window.__speech.slice(count).map(part => part.text).join(' '), beforeAutomatic), reply);
    checks.push('Automatic voice reads only reply clauses; late coaching neither reads tips nor changes the learner-opened pronunciation/word state or closed optional completion.');

    fresh('generating'); job.text = 'You are making good progress. Tell me more about your day.';
    job.speech.correction = { status: 'not_needed', mode: 'completion_only' };
    await submit();
    await page.waitForFunction(() => window.MDDChat.state.speaking && window.MDDChat.state.busy);
    const beforeEnd = await page.evaluate(() => window.__speech.length);
    assert.ok(await page.locator('#chat-end').isEnabled());
    await page.locator('#chat-end').click();
    await page.waitForFunction(() => !window.MDDChat.state.inRoom && !window.MDDChat.state.busy);
    assert.equal(await roomOpen(), false);
    assert.equal(await page.evaluate(() => window.MDDChat.state.speaking), null);
    assert.ok(cancellations.length > 0);
    assert.equal(cancellations.at(-1).job_id, job.id);
    await page.evaluate(() => window.__utterances.at(-1)?.onend?.());
    await page.waitForTimeout(350);
    assert.equal(await page.evaluate(() => window.__speech.length), beforeEnd);
    assert.ok(await page.locator('#chat-topic-picker').isVisible());
    await screenshot('topics-after-end');
    checks.push('Back returns to topics; End cancels the active job and queued voice immediately, without late playback.');

    // Cancel while the initial request is still in flight, before a server job ID
    // is known. The eventual response must be cancelled rather than restarting.
    await page.locator('[data-topic=travel]').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    fresh('generating'); job.text = 'This response must never be spoken.';
    delayPost = true;
    const countBeforeRace = requests.length, voiceBeforeRace = await page.evaluate(() => window.__speech.length);
    await submit();
    await page.waitForFunction(() => window.MDDChat.state.busy && !window.MDDChat.state.job);
    for (let attempt = 0; attempt < 100 && requests.length === countBeforeRace; attempt++) await page.waitForTimeout(20);
    assert.ok(requests.length > countBeforeRace && releasePost, 'The delayed initial POST must be in flight.');
    await page.locator('#chat-end').click();
    assert.ok(await page.locator('#chat-topic-picker').isVisible());
    assert.equal(await roomOpen(), false);
    assert.equal(await page.evaluate(() => window.MDDChat.state.inRoom), false);
    releasePost(); releasePost = null; delayPost = false;
    await waitDone();
    assert.equal(cancellations.at(-1).job_id, job.id);
    assert.equal(await page.evaluate(() => window.__speech.length), voiceBeforeRace);
    assert.equal(await page.evaluate(() => window.MDDChat.state.speaking), null);
    checks.push('End also handles an in-flight initial POST: the picker opens immediately and the eventual job ID is cancelled without speech.');

    // Simulate a worklet whose flush response is delayed while the learner
    // leaves the room. A late Stop & send must never upload into a new view.
    await page.locator('[data-topic=travel]').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    const beforeFlushRace = requests.length;
    await page.evaluate(() => {
      window.__micCleanup = { tracks: 0, context: 0, processor: 0, source: 0, sink: 0, flush: null };
      const state = window.MDDChat.state;
      state.recording = true; state.sampleCount = 8000; state.chunks = [new Float32Array(8000)];
      state.context = { sampleRate: 16000, state: 'running', close() { window.__micCleanup.context++; return Promise.resolve(); } };
      state.processor = { port: { postMessage(value) { window.__micCleanup.flush = value; }, onmessage: null }, disconnect() { window.__micCleanup.processor++; } };
      state.source = { disconnect() { window.__micCleanup.source++; } };
      state.sink = { disconnect() { window.__micCleanup.sink++; } };
      state.stream = { getTracks() { return [{ stop() { window.__micCleanup.tracks++; } }]; } };
      window.__heldMicStop = window.MDDChat.stopMic(true);
    });
    await page.waitForFunction(() => window.__micCleanup.flush === 'flush' && window.MDDChat.state.stopping);
    await page.evaluate(() => { window.switchView('practice'); window.switchView('chat'); });
    assert.deepEqual(await page.evaluate(() => ({ tracks: window.__micCleanup.tracks, context: window.__micCleanup.context,
      processor: window.__micCleanup.processor, source: window.__micCleanup.source, sink: window.__micCleanup.sink })),
    { tracks: 1, context: 1, processor: 1, source: 1, sink: 1 });
    assert.deepEqual(await page.evaluate(() => ({ recording: window.MDDChat.state.recording, stopping: window.MDDChat.state.stopping,
      context: window.MDDChat.state.context, chunks: window.MDDChat.state.chunks.length })),
    { recording: false, stopping: false, context: null, chunks: 0 });
    await page.evaluate(() => window.__heldMicStop);
    assert.equal(requests.length, beforeFlushRace);
    checks.push('Leaving during a delayed worklet flush releases every track/audio node immediately; returning to chat cannot submit the stale recording.');

    // Practice is opt-in: opening feedback shows a phonetic reference and
    // separate listening controls without replaying either sentence or tutor.
    await openSettings(); await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    const voicesBeforePractice = await page.evaluate(() => window.__speech.length);
    fresh('coaching'); job.text = 'That sounds interesting. What happened next?';
    job.speech = threeWordFixture();
    await submit();
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'coaching');
    assert.equal(await feedback().evaluate(node => node.open), false);
    assert.equal(await page.evaluate(() => window.__speech.length), voicesBeforePractice);
    await feedback().locator(':scope>summary').click();
    const wordRows = feedback().locator('.chat-word-practice');
    assert.equal(await wordRows.count(), 3);
    assert.deepEqual(await wordRows.locator('.chat-word-details').allTextContents(), ['added', 'with', 'genuine']);
    const practiceIPA = await wordRows.locator('.chat-word-ipa').allTextContents();
    assert.ok(practiceIPA.every(text => /^\/.+\/$/.test(text)), 'Word references must display IPA in slashes.');
    assert.deepEqual(practiceIPA.map(text => text.replace(/[ˈˌ]/g, '')), ['/ædɪd/', '/wɪð/', '/dʒɛnjuən/']);
    assert.equal(await wordRows.locator('.chat-word-listen').count(), 3);
    assert.equal(await page.evaluate(() => window.__speech.length), voicesBeforePractice, 'Opening word practice must stay silent.');
    await wordRows.nth(0).locator('.chat-word-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), 'added');
    await wordRows.nth(1).locator('.chat-word-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), 'with');
    assert.deepEqual(await page.evaluate(count => window.__speech.slice(count).map(call => call.text), voicesBeforePractice), ['added', 'with']);
    await wordRows.nth(1).locator('.chat-word-details').click();
    assert.equal(await feedback().locator('.chat-sound-word').nth(3).evaluate(node => node.open), true);
    const voiceBeforePoll = await page.evaluate(() => window.__speech.length);
    job.speech.coaching = { status: 'ready', feedback: { added: [{ issue: 'Possible AE/EH difference.', suggestion: 'Listen to the vowel in added.' }] } };
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.speech?.coaching?.status === 'ready');
    assert.equal(await feedback().evaluate(node => node.open), true);
    assert.equal(await feedback().locator('.chat-sound-word').nth(3).evaluate(node => node.open), true);
    assert.equal(await page.evaluate(() => window.__speech.length), voiceBeforePoll);
    job.status = 'ready'; await waitDone();
    const soundPanel = feedback().locator('.chat-pronunciation:not(.chat-completion-extra)');
    assert.equal(await feedback().locator('.chat-turn-review .chat-sentence-listen').count(), 0);
    assert.equal(await soundPanel.locator('.chat-sentence-listen').count(), 1);
    if (await soundPanel.evaluate(node => node.open)) await soundPanel.locator(':scope>summary').click();
    assert.ok(await soundPanel.locator('.chat-sentence-listen').isHidden());
    const voicesBeforeSoundExpansion = await page.evaluate(() => window.__speech.length);
    await soundPanel.locator(':scope>summary').click();
    assert.equal(await page.evaluate(() => window.__speech.length), voicesBeforeSoundExpansion);
    await soundPanel.locator('.chat-sentence-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), job.speech.reference_text);
    assert.equal(await page.evaluate(() => window.__speech.at(-1).lang), 'en-US');
    assert.notEqual(await page.evaluate(() => window.__speech.at(-1).text), job.text);
    checks.push('Each flagged word has IPA and independent listening; whole-sentence playback lives inside Sound details and says the pronunciation reference only on click. Expanding details and late polling never start speech.');
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport); await frameFeedback();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      const bounds = await wordRows.evaluateAll(rows => rows.map(row => {
        const rect = row.getBoundingClientRect();
        return { left: rect.left, right: rect.right, width: innerWidth };
      }));
      assert.ok(bounds.every(rect => rect.left >= 0 && rect.right <= rect.width), 'Every word practice row must fit the viewport.');
      await screenshot(`word-practice-${viewport.width}x${viewport.height}`);
    }
    await feedback().locator(':scope>summary').click();
    assert.equal(await feedback().evaluate(node => node.open), false);
    assert.ok(await wordRows.first().isHidden());
    checks.push('Word IPA/listening rows and whole-sentence playback fit desktop/mobile, and all remain behind the learner-controlled pronunciation collapse.');

    assert.deepEqual(exceptions, []); assert.equal(paidCalls, 0);
    const report = { status: 'passed', base_url: base, checks, screenshots, exceptions,
      scope: 'Mock chat/ASR jobs and simulated English browser speech. No model inference, physical microphone/speaker test, training, or paid TTS.',
      paid_tts_calls: paidCalls, speech_request_count: requests.length, cancelled_jobs: cancellations.length, viewport_measurements: viewportMeasurements,
      repeat_payload_verified: true };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await screenshot('failure').catch(() => {});
    console.error('Browser exceptions:', exceptions);
    throw error;
  } finally { releasePost?.(); await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
