'use strict';
// Repeatable UI-only QA. Chat jobs, ASR text and speech playback are fixtures;
// this never performs inference, starts training or calls a paid voice provider.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/sentence-correction-qa');
const corrected = 'I would love to visit Japan.';
const original = 'Japan';
const wholeUtterance = 'Yesterday I go to the park with my friend.';
const config = {
  speech_ready: true, default_model_id: 'qwen-trained', device: 'cpu', threads: 2,
  ai33_available: false, voice_locales: ['en-US'], max_audio_seconds: 30,
  models: [{ id: 'qwen-base', ready: true }, { id: 'qwen-trained', ready: true }],
  practice_prompts: [{ id: 'idea', text: 'I think this is a good idea.' }],
  topics: [{ id: 'travel', title: 'Travel', icon: 'plane', level: 'A2',
    description: 'Talk about a trip.', guidance: { keywords: ['Japan', 'visit'], starter: 'I would love to visit…' } }],
};
const speechFixture = () => ({
  transcript: original, reference_text: original, reference_source: 'asr', missing_words: [],
  comparison: {
    words: original.split(' ').map((text, index) => ({ text, status: index === 0 ? 'difference' : 'match' })),
    events: [{ kind: 'substitution', expected_phone: 'JH', observed_phone: 'CH', word_index: 0 }],
  },
  target_words: original.split(' ').map(text => ({ text, phones: ['JH', 'AH0', 'P', 'AE1', 'N'] })),
  phone_symbols: { JH: 'dʒ', CH: 'tʃ' },
  coaching: { status: 'ready', feedback: { Japan: [{ issue: 'The sound model detected a possible JH/CH difference.', suggestion: 'Practise the voiced JH sound in Japan.' }] } },
  correction: { status: 'pending', mode: 'completion_only', original },
});

function mockVoice() {
  window.__spoken = [];
  Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
    cancel() {}, getVoices() { return []; },
    speak(utterance) {
      window.__spoken.push({ text: utterance.text, lang: utterance.lang });
      setTimeout(() => utterance.onend?.(), 5);
    },
  } });
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.MDD_QA_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const exceptions = [], requests = [], checks = [], screenshots = [];
  let job, serial = 0, paidVoiceCalls = 0;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  const screenshot = async name => {
    const filename = name + '.png';
    await page.screenshot({ path: path.join(output, filename), fullPage: true, animations: 'disabled' });
    screenshots.push(filename);
  };
  const fresh = () => job = {
    id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status: 'correcting', text: '',
    speech: speechFixture(), metadata: {}, error: null,
  };
  const ready = (options = {}) => {
    job.status = 'ready'; job.text = 'Japan sounds exciting. What would you like to do there?';
    job.speech.spoken_response = job.text + ' You said Japan. Practise the voiced JH sound in Japan.';
    job.speech.correction = { status: 'ready', mode: 'completion_only', original, corrected, changed: true,
      explanation: 'The prior question asks where you would love to visit.',
      changes: [{ before: '', after: 'I would love to visit' }], source: 'qwen', model_id: 'qwen-base', ...options };
  };
  await page.route('**/api/chat/config', route => route.fulfill({ json: config }));
  await page.route('**/api/chat/speech', route => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ json: job });
  });
  await page.route('**/api/chat/jobs/*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/voice', route => {
    paidVoiceCalls++;
    return route.fulfill({ status: 500, json: { message: 'Paid TTS is blocked in UI QA.' } });
  });
  await page.route('**/api/chat/topic', route => route.fulfill({ json: {
    id: 'topic-opening', status: 'ready', text: 'Where would you love to visit?', metadata: { guidance: config.topics[0].guidance },
  } }));
  const submit = async () => {
    // The tiny buffer is intentionally intercepted by the mock route above.
    // Audio decoding and real recognition are outside this UI-only check.
    await page.evaluate(() => { window.__correctionQaSend = window.MDDChat.sendAudio(new ArrayBuffer(48), 'ui-fixture.wav'); });
  };
  const waitDone = () => page.waitForFunction(() => !window.MDDChat.state.busy);
  const openFeedback = async (details = page.locator('.chat-turn-feedback').last()) => {
    if (!(await details.evaluate(node => node.open))) await details.locator(':scope > summary').click();
  };
  const openCompletion = async (details = page.locator('.chat-turn-feedback').last()) => {
    await openFeedback(details);
    const completion = details.locator('.chat-completion-extra');
    if (!(await completion.evaluate(node => node.open))) await completion.locator(':scope > summary').click();
  };
  const practiseFirst = async () => {
    await openCompletion(page.locator('.chat-turn-feedback').filter({ has: page.locator('.chat-correction-practise') }).first());
    await page.locator('.chat-correction-practise').first().click();
  };
  try {
    await page.goto(base + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('[data-topic=travel]').click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-auto-read').uncheck();
    await page.keyboard.press('Escape');

    fresh(); await submit();
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'correcting');
    assert.equal(await page.locator('.chat-turn-feedback').last().evaluate(node => node.open), false);
    await openFeedback();
    assert.ok(await page.locator('.chat-correction').last().isHidden());
    assert.equal(await page.locator('.chat-correction-original').count(), 0);
    assert.ok(await page.locator('.chat-turn-review').last().isVisible());
    assert.ok(await page.locator('.chat-pronunciation:not(.chat-completion-extra)').last().isVisible());
    assert.ok(await page.locator('.chat-completion-extra').last().isHidden());
    assert.equal(await page.locator('.chat-correction-sentence').count(), 0);
    await screenshot('desktop-pending');
    checks.push('The feedback strip defaults closed; opening it during processing reveals available original-recording sound estimates, with no duplicate transcript or invented completion.');

    ready({ explanation: 'The prior travel question supplies the visit context. <img src=x onerror=alert(1)>' }); await waitDone();
    assert.equal(await page.locator('.chat-turn-feedback').last().evaluate(node => node.open), true);
    assert.equal(await page.locator('.chat-completion-extra').last().evaluate(node => node.open), false);
    assert.equal((await page.locator('.chat-feedback-label').last().textContent()).trim(), 'Cần xác nhận câu');
    assert.ok(await page.locator('.chat-correction-sentence').last().isHidden());
    await openCompletion();
    assert.equal(await page.locator('.chat-correction-sentence').last().textContent(), corrected);
    assert.equal(await page.locator('.chat-message.user .chat-message-text').last().textContent(), original);
    assert.equal(await page.locator('.chat-correction img').count(), 0);
    assert.equal(await page.locator('.chat-correction-edits').last().textContent(), 'Add “I would love to visit”');
    assert.doesNotMatch(await page.locator('.chat-correction-edits').last().textContent(), /past tense|grammar|go.*went/i);
    assert.equal(await page.locator('.chat-correction-why').last().getAttribute('open'), null);
    assert.ok(await page.locator('.chat-correction-why .chat-correction-note').last().isHidden());
    await page.locator('.chat-correction-why>summary').last().click();
    assert.match(await page.locator('.chat-correction-why .chat-correction-note').last().textContent(), /<img src=x/);
    assert.ok(await page.locator('.chat-correction-why .chat-correction-note').last().isVisible());
    await page.locator('.chat-correction-why>summary').last().click();
    const order = await page.locator('.chat-feedback-content').last().evaluate(element => Array.from(element.children).map(child => child.className));
    assert.ok(order.indexOf('chat-turn-review') < order.indexOf('chat-pronunciation'));
    assert.ok(order.indexOf('chat-pronunciation') < order.indexOf('chat-pronunciation chat-completion-extra'));
    assert.match(await page.locator('.chat-turn-feedback').last().evaluate(element => element.nextElementSibling.className), /chat-message assistant/);
    assert.equal(await page.locator('.chat-target-text').last().textContent(), original);
    assert.equal(await page.locator('.chat-pronunciation:not(.chat-completion-extra)>.chat-pronunciation-title').last().textContent(), 'Sound details');
    await page.waitForFunction(() => !window.MDDChat.state.liveReply || window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
    await page.locator('.chat-turn-feedback').last().evaluate(node => {
      const scroll = node.closest('#chat-scroll');
      scroll.scrollTop += node.getBoundingClientRect().top - scroll.getBoundingClientRect().top - 12;
    });
    await screenshot('desktop-ready');
    checks.push('Feedback puts original-recording pronunciation before a separately collapsed optional completion and before the bot reply; untrusted text is literal and Japan stays the user transcript.');
    checks.push('The primary explanation shows only added context; Qwen explanation is collapsed, and a single target sentence replaces the old repeated pronunciation disclaimers.');

    await page.locator('.chat-correction-listen').last().click();
    assert.deepEqual(await page.evaluate(() => window.__spoken.at(-1)), { text: corrected, lang: 'en-US' });
    checks.push('Listen speaks the completed sentence in English.');
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    await page.locator('.chat-message.assistant .chat-message-actions button').last().click();
    assert.equal(await page.evaluate(() => window.__spoken.at(-1).text), job.text);
    checks.push('Bot replay omits legacy spoken_response echoes and pronunciation instructions; those remain opt-in.');

    await practiseFirst();
    assert.ok(await page.locator('#chat-reading-card').isVisible());
    assert.equal(await page.locator('#chat-reading-text').textContent(), corrected);
    assert.equal(await page.locator('#chat-mode').inputValue(), 'conversation');
    await page.locator('#chat-reading-listen').click();
    assert.equal(await page.evaluate(() => window.__spoken.at(-1).text), corrected);
    for (const width of [1440, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Overflow at ' + width);
    }
    await screenshot('mobile-repeat');
    checks.push('Practise keeps the conversation mode, shows the repeat target with Listen, and fits widths 1440/768/390/360.');

    const sourceId = job.id;
    fresh(); job.status = 'ready'; job.text = 'Good practice. How did it feel?';
    job.speech.reference_source = 'corrected_sentence'; job.speech.reference_text = corrected;
    job.speech.correction = { status: 'not_needed', mode: 'completion_only' };
    await submit(); await waitDone();
    await openFeedback();
    assert.equal(requests.at(-1).correction_job_id, sourceId);
    assert.ok(!Object.hasOwn(requests.at(-1), 'practice_id'));
    assert.ok(!Object.hasOwn(requests.at(-1), 'reference_text'));
    assert.deepEqual(requests.at(-1).history.slice(-2), [
      { role: 'user', content: original },
      { role: 'assistant', content: 'Japan sounds exciting. What would you like to do there?' },
    ]);
    assert.ok(!requests.at(-1).history.some(turn => turn.role === 'user' && turn.content === corrected));
    assert.ok(await page.locator('.chat-correction').last().isHidden());
    assert.equal(await page.locator('.chat-target-text').last().textContent(), corrected);
    await page.locator('#chat-repeat-clear').click();
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    assert.ok(await page.locator('#chat-reading-card').isHidden());
    checks.push('Repeating sends only the server-owned correction job ID, hides not_needed correction, and Clear restores conversation recording.');
    checks.push('The second speech request carries the original user fragment and actual bot reply in history, never the completed output as user speech.');

    await practiseFirst();
    // The application navigation is intentionally hidden in the focused room.
    // Exercise its public view switch to retain the navigation cleanup check.
    await page.evaluate(() => window.switchView('practice'));
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    await page.locator('.navigation [data-view=chat]').click();
    await practiseFirst();
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-model').selectOption('qwen-base');
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-model').selectOption('qwen-trained');
    await page.keyboard.press('Escape');
    await practiseFirst(); await page.locator('#chat-topics').click();
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    await page.locator('[data-topic=travel]').click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting);
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    await practiseFirst(); await page.locator('#chat-settings>summary').click(); await page.locator('#chat-new').click();
    await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    checks.push('Repeat targets are cleared on navigation, model change, topic selection and New.');

    await page.setViewportSize({ width: 1440, height: 1080 });
    fresh(); await submit();
    await page.waitForFunction(() => window.MDDChat.state.activeTurn?.status === 'correcting');
    job.status = 'ready'; job.text = 'What would you like to tell me next?';
    job.speech.correction = { status: 'unavailable', mode: 'completion_only', original, corrected: null, changed: false,
      explanation: null, message: 'Qwen could not reliably finish this thought.' };
    await waitDone();
    await openFeedback();
    assert.ok(await page.locator('.chat-correction[data-status=unavailable]').last().isHidden());
    assert.ok(await page.locator('.chat-completion-extra').last().isHidden());
    assert.equal(await page.locator('.chat-correction-sentence').count(), 0);
    assert.equal(await page.locator('.chat-correction-practise').count(), 0);
    assert.ok(await page.locator('.chat-turn-review').last().isVisible());
    assert.ok(await page.locator('.chat-pronunciation:not(.chat-completion-extra)').last().isVisible());
    assert.equal(await page.locator('.chat-target-text').last().textContent(), original);
    await screenshot('desktop-unavailable');
    checks.push('Pending to unavailable keeps the original acoustic review and never invents a completed sentence.');

    fresh();
    job.speech.transcript = wholeUtterance; job.speech.reference_text = wholeUtterance;
    job.speech.comparison = { words: wholeUtterance.split(' ').map(text => ({ text, status: 'match' })), events: [] };
    job.speech.target_words = wholeUtterance.split(' ').map(text => ({ text, phones: [] }));
    ready({ original: wholeUtterance, corrected: wholeUtterance, changed: false, explanation: 'Your thought is already complete; your wording is kept.', changes: [] });
    await submit(); await waitDone();
    assert.equal(await page.locator('.chat-turn-feedback').last().evaluate(node => node.open), false);
    await openFeedback();
    assert.ok(await page.locator('.chat-completion-extra').last().isHidden());
    assert.ok(await page.locator('.chat-correction-sentence').last().isHidden());
    assert.equal(await page.locator('.chat-message.user .chat-message-text').last().textContent(), wholeUtterance);
    checks.push('A complete utterance with nonstandard grammar keeps go unchanged in its user bubble and is not repeated in an unnecessary completion panel.');

    fresh(); ready({ mode: undefined, original: wholeUtterance, corrected: 'Yesterday I went to the park with my friend.',
      changes: [{ before: 'go', after: 'went' }], explanation: 'Legacy grammar correction.' });
    await submit(); await waitDone(); await openFeedback();
    assert.ok(await page.locator('.chat-correction').last().isHidden());
    assert.equal(await page.locator('.chat-turn-feedback').last().locator('.chat-correction-practise').count(), 0);
    assert.doesNotMatch(await page.locator('.chat-feedback-label').last().textContent(), /hoàn thiện câu|sửa câu/i);
    checks.push('Legacy grammar output without completion_only mode is hidden and has no repeat button; it is never relabelled as a completion.');

    fresh(); ready(); await submit(); await waitDone();
    assert.equal(await page.locator('.chat-turn-feedback').last().evaluate(node => node.open), false);
    await page.reload();
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    const restored = page.locator('.chat-turn-feedback').filter({ has: page.locator('.chat-correction-sentence') }).last();
    assert.equal(await restored.evaluate(node => node.open), false);
    assert.equal(await restored.locator('.chat-completion-extra').evaluate(node => node.open), false);
    await openCompletion(restored);
    assert.equal(await page.locator('.chat-correction-sentence').last().textContent(), corrected);
    assert.ok(await page.locator('.chat-correction-practise').last().isDisabled());
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().at(-1).jobId), undefined);
    assert.equal(await page.evaluate(() => window.MDDChat.state.repeatTarget), null);
    await page.locator('.chat-correction-listen').last().click();
    assert.equal(await page.evaluate(() => window.__spoken.at(-1).text), corrected);
    await page.setViewportSize({ width: 390, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await screenshot('mobile-restored');
    checks.push('Reload restores pronunciation and optional completion closed; explicit Listen remains usable and missing current-session job ID disables Practise.');

    assert.deepEqual(exceptions, []); assert.equal(paidVoiceCalls, 0);
    const report = { status: 'passed', base_url: base, checks, screenshots, exceptions,
      scope: 'UI-only mock ASR/jobs and simulated browser speech. No real inference, microphone or physical audio validation.',
      paid_tts_calls: paidVoiceCalls, request_count: requests.length,
      repeat_payload_verified: true, original_reference_preserved: true };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    await screenshot('failure').catch(() => {});
    console.error('Browser exceptions:', exceptions);
    throw error;
  } finally { await browser.close(); }
}

if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
