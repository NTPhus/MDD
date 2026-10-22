'use strict';
// Real UI, deterministic job/reference/voice events. No model or paid API calls.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/chat-ui-cleanup-qa');
const original = 'I wend to London last wick';
const confirmed = 'I went to London last week';
const oldReply = 'I did not understand. Could you say that again?';
const newReply = 'Sorry for the misunderstanding. What did you enjoy about London?';
const lexical = text => (text.normalize('NFKC').replace(/’/g, "'").toLowerCase()
  .match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) || []).join('\u0000');

function controlledVoice() {
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak = utterance => { window.__utterances.push(utterance); window.__speech.push({ text: utterance.text }); };
  window.__audio = [];
  window.Audio = class {
    constructor(url) { this.src = url; this.currentTime = 0; this.duration = 6; this.paused = true; window.__audio.push(this); }
    play() { this.paused = false; this.onplaying?.(); return Promise.resolve(); }
    pause() { this.paused = true; this.onpause?.(); }
  };
}
function speechFixture(text = original, source = 'asr') {
  const tokens = text.split(/\s+/);
  return { transcript: original, reference_text: text, reference_source: source,
    ...(source === 'user_confirmed' ? { conversation_text: text } : {}),
    reference_resolution: { status: source === 'asr' ? 'needs_confirmation' : 'confirmed', suggested_text: text, missing_words: [] },
    transcript_clarity: { needs_clarification: source === 'asr' },
    comparison: { words: tokens.map(text => ({ text, status: 'uncertain' })), events: [] },
    assessment: { calibration: { applied: false, reason: source === 'asr' ? 'reference_unconfirmed' : 'no_valid_calibration' } },
    sound_candidates: { status: 'available', diagnostic_only: true,
      reference_confirmed: source === 'user_confirmed', provisional: source === 'asr', source: source === 'asr' ? 'asr' : 'confirmed_reference', words: [], count: 0 },
    target_words: tokens.map(text => ({ text, phones: ['W', 'EH', 'N', 'T'] })),
    coaching: { status: 'unavailable' }, correction: { mode: 'completion_only', status: 'not_needed' } };
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const errors = [], checks = [], requests = [], speechRequests = []; let job, serial = 0, failReference = false;
  let stalePolls = [], cancelCount = 0, holdReference = false, releaseReference = null, holdVoice = false, releaseVoice = null, voiceCount = 0;
  let readingFixture = false, forceNoOp = false;
  const config = { models: [{ id: 'qwen-base', ready: true }, { id: 'qwen-trained', ready: true }], speech_ready: true,
    topics: [], practice_prompts: [{ id: 'test', text: confirmed }], ai33_available: true, voice_locales: ['en-US'], reference_regenerates_reply: true };
  const fresh = () => job = { id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status: 'ready', text: oldReply,
    speech: speechFixture(), metadata: { reference_revision: 0, reply_source: 'clarification_fallback' }, voice: { status: 'not_requested' }, error: null };
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(mockVoice); await page.addInitScript(controlledVoice);
  await page.route('**/api/chat/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/speech') { speechRequests.push(route.request().postDataJSON()); return route.fulfill({ json: job }); }
    if (pathname.startsWith('/api/chat/jobs/')) return route.fulfill({ json: stalePolls.shift() || job });
    if (pathname === '/api/chat/cancel') { cancelCount++; job.status = 'cancelled'; return route.fulfill({ json: job }); }
    if (pathname === '/api/chat/reference') {
      const payload = route.request().postDataJSON(); requests.push(payload);
      if (failReference) return route.fulfill({ status: 409, json: { error: 'chat_busy', message: 'The local model is busy. Please retry.' } });
      const previous = structuredClone(job), revision = job.metadata.reference_revision + 1;
      if (holdReference) await new Promise(resolve => { releaseReference = resolve; });
      if (forceNoOp || lexical(payload.reference_text) === lexical(job.speech.conversation_text || job.speech.reference_text || job.speech.transcript)) {
        job.speech = { ...job.speech, reference_text: payload.reference_text, reference_source: 'user_confirmed',
          conversation_text: job.speech.conversation_text || job.speech.reference_text || job.speech.transcript,
          reference_resolution: { status: 'confirmed' } };
        return route.fulfill({ json: { ...structuredClone(job), reference_update: { changed: false, regenerated_reply: false } } });
      }
      job = { ...job, status: 'generating', text: '', speech: speechFixture(payload.reference_text, 'user_confirmed'),
        metadata: { reference_revision: revision, pronunciation_reference_source: 'user_confirmed' }, voice: { status: 'not_requested' }, error: null };
      if (payload.reference_text.includes('Quendor')) {
        job.speech.missing_words = ['Quendor']; job.speech.reference_resolution.missing_words = ['Quendor'];
        job.speech.comparison.words = job.speech.comparison.words.map(word => ({ ...word, status: 'unavailable' }));
        job.speech.sound_candidates = { status: 'unavailable', diagnostic_only: true, reference_confirmed: true,
          reason: 'reference_contains_unknown_words', missing_words: ['Quendor'], words: [], count: 0 };
      }
      if (readingFixture) job.speech.coaching = { status: 'ready', feedback: { week: Array.from({ length: 12 }, (_, i) => ({
        issue: `Optional sound detail ${i + 1}.`, suggestion: 'Listen carefully to this word and compare it with your original recording.' })) } };
      stalePolls.push(previous); // Simulate a cached late GET from before the POST.
      return route.fulfill({ json: { ...structuredClone(job), reference_update: { changed: true, regenerated_reply: true } } });
    }
    if (pathname === '/api/chat/voice') {
      voiceCount++;
      const snapshot = { ...structuredClone(job), voice: { status: 'ready', url: '/api/chat/audio/' + 'b'.repeat(32) } };
      if (holdVoice) { holdVoice = false; await new Promise(resolve => { releaseVoice = resolve; }); }
      return route.fulfill({ json: snapshot });
    }
    return route.fulfill({ status: 500, json: { message: 'Unexpected request blocked in UI QA.' } });
  });
  const feedback = () => page.locator('.chat-turn-feedback').last();
  const input = () => feedback().locator('.chat-reference-input');
  const apply = () => feedback().locator('.chat-reference-apply');
  const assistantText = () => page.locator('.chat-message.assistant .chat-message-text').last();
  const userText = () => page.locator('.chat-message.user .chat-message-text').last();
  const done = () => page.waitForFunction(() => !window.MDDChat.state.busy);
  const submit = async () => {
    const count = await page.locator('.chat-message.assistant').count();
    await page.evaluate(() => { window.__send = window.MDDChat.sendAudio(new ArrayBuffer(48), 'fixture.wav'); });
    await page.waitForFunction(count => document.querySelectorAll('.chat-message.assistant').length > count && !window.MDDChat.state.busy, count);
  };
  const drainVoice = async () => {
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => { const voice = window.__utterances.at(-1); voice?.onstart?.(); voice?.onend?.(); });
      if (await page.evaluate(() => !window.MDDChat.state.liveReply?.active && !window.MDDChat.state.liveReply?.queue.length)) break;
    }
    await page.waitForFunction(() => !window.MDDChat.state.liveReply || window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
  };
  const openEditor = async () => {
    if (!await feedback().evaluate(node => node.open)) await feedback().locator(':scope > summary').click();
    if (!await feedback().locator('.chat-reference-editor').evaluate(node => node.open)) await feedback().locator('.chat-reference-editor > summary').click();
  };
  const newConversation = async () => {
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-new').click(); await page.keyboard.press('Escape');
  };
  const voiceSetting = async (provider, auto = true) => {
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-voice-provider').selectOption(provider);
    await page.locator('#chat-auto-read').setChecked(auto); await page.keyboard.press('Escape');
  };
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click(); fresh(); await submit();
    await page.locator('.chat-clarify-action').click();
    assert.equal(await feedback().evaluate(node => node.open), true); assert.equal(await input().evaluate(node => node === document.activeElement), true);
    const sameInputState = await page.evaluate(() => ({ epoch: window.MDDChat.state.speechEpoch,
      speech: window.__speech.length, revision: window.MDDChat.getTurns()[1].metadata.reference_revision,
      raw: window.MDDChat.getTurns()[0].content, content: window.MDDChat.getTurns()[1].content }));
    await input().fill('  I WEND to London, last wick!  '); await apply().click();
    await page.waitForFunction(() => !window.MDDChat.state.referenceOperation);
    assert.equal(await page.evaluate(() => window.MDDChat.state.busy), false);
    assert.deepEqual(await page.evaluate(() => ({ epoch: window.MDDChat.state.speechEpoch,
      speech: window.__speech.length, revision: window.MDDChat.getTurns()[1].metadata.reference_revision,
      raw: window.MDDChat.getTurns()[0].content, content: window.MDDChat.getTurns()[1].content })), sameInputState);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[0].confirmedText), undefined);
    assert.equal(await feedback().locator('.chat-reference-error').count(), 0);
    checks.push('Case, punctuation and spacing changes confirm pronunciation without blanking/replaying the reply, changing revision/history or stopping ongoing voice.');
    await input().fill(confirmed);
    const identity = await page.evaluate(() => ({ user: window.MDDChat.getTurns()[0].id, bot: window.MDDChat.getTurns()[1].id,
      raw: window.MDDChat.getTurns()[0].content, audio: document.querySelector('.chat-message.user audio').src }));
    const staleVoice = await page.evaluate(() => window.__utterances.length - 1);
    readingFixture = true; await apply().click();
    await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.getTurns()[1].referenceRevision === 1);
    assert.equal(await assistantText().textContent(), ''); assert.equal(await userText().textContent(), confirmed);
    assert.equal(await page.locator('#chat-upload').isDisabled(), true); assert.equal(await apply().isDisabled(), true);
    assert.equal(await feedback().locator('.chat-sentence-listen').isDisabled(), true);
    assert.equal(await feedback().evaluate(node => node.open), true); assert.equal(await feedback().locator('.chat-reference-editor').evaluate(node => node.open), true);
    checks.push('Confirm immediately clears the old bot reply, updates the displayed user sentence, keeps the editor/collapse open and locks other audio/reference input.');
    const readPosition = await page.evaluate(() => { const scroll = document.getElementById('chat-scroll');
      scroll.scrollTop = Math.min(170, scroll.scrollHeight - scroll.clientHeight); return { top: scroll.scrollTop, maximum: scroll.scrollHeight - scroll.clientHeight }; });
    assert.ok(readPosition.maximum > 200 && readPosition.top > 0, 'The open feedback fixture must really overflow for the scroll check.');
    job.text = 'Sorry for the misunderstanding.';
    await page.waitForFunction(() => window.__speech.some(call => call.text === 'Sorry for the misunderstanding.'));
    await page.evaluate(index => {
      const next = window.__utterances.at(-1); next.onstart?.(); next.onboundary?.({ name: 'word', charIndex: 0 });
      const old = window.__utterances[index]; old.onboundary?.({ name: 'word', charIndex: 2 }); old.onend?.();
    }, staleVoice);
    assert.deepEqual(await page.locator('.chat-message.assistant .is-speaking').allTextContents(), ['Sorry']);
    assert.equal((await assistantText().textContent()).includes('understand'), false);
    assert.ok(Math.abs(await page.locator('#chat-scroll').evaluate(node => node.scrollTop) - readPosition.top) < 2);
    assert.equal(await feedback().evaluate(node => node.open), true);
    checks.push('Streaming text, native word highlighting and polling do not move the viewport while an open feedback collapse is being read.');
    job.text = newReply; job.status = 'ready'; job.metadata.reply_source = 'qwen';
    await done(); await drainVoice();
    readingFixture = false;
    assert.equal(await assistantText().textContent(), newReply);
    assert.deepEqual(await page.evaluate(() => ({ user: window.MDDChat.getTurns()[0].id, bot: window.MDDChat.getTurns()[1].id,
      raw: window.MDDChat.getTurns()[0].content, audio: document.querySelector('.chat-message.user audio').src })), identity);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[1].speech.transcript), original);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[0].confirmedText), confirmed);
    assert.equal(await page.locator('.chat-message.assistant').count(), 1);
    checks.push('Same-job streamed regeneration ignores stale revision polls/old voice callbacks, replaces the same bot turn and preserves raw ASR plus the original audio.');
    await page.screenshot({ path: path.join(output, 'desktop-regenerated-reference.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 390, height: 844 }); assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await page.screenshot({ path: path.join(output, 'mobile-regenerated-reference.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 1440, height: 1000 });

    const repeatedState = await page.evaluate(() => ({ revision: window.MDDChat.getTurns()[1].referenceRevision,
      epoch: window.MDDChat.state.speechEpoch, voice: window.__speech.length, user: window.MDDChat.getTurns()[0].confirmedText }));
    await input().fill('I went to London last week!'); await apply().click();
    await page.waitForFunction(() => !window.MDDChat.state.referenceOperation);
    assert.equal(await assistantText().textContent(), newReply);
    assert.deepEqual(await page.evaluate(() => ({ revision: window.MDDChat.getTurns()[1].referenceRevision,
      epoch: window.MDDChat.state.speechEpoch, voice: window.__speech.length, user: window.MDDChat.getTurns()[0].confirmedText })), repeatedState);
    checks.push('Repeating an already confirmed sentence is a successful no-op and does not generate an apology or a second reply.');

    fresh(); job.text = 'Thanks for sharing.'; job.metadata.reply_source = 'qwen'; await submit(); await drainVoice();
    assert.equal(speechRequests.at(-1).history[0].content, confirmed);
    assert.equal(speechRequests.at(-1).history[1].content, newReply);
    checks.push('The next conversation request uses the confirmed user sentence and regenerated bot reply in history.');

    await newConversation(); fresh(); await submit(); await drainVoice(); await openEditor(); await input().fill(confirmed);
    failReference = true; await apply().click(); await done(); failReference = false;
    assert.equal(await assistantText().textContent(), oldReply); assert.equal(await userText().textContent(), original);
    assert.ok((await feedback().locator('.chat-reference-error').textContent()).includes('busy'));
    assert.equal(await input().inputValue(), confirmed); assert.equal(await apply().isDisabled(), false);
    checks.push('A rejected POST restores the previous reply, preserves the user draft and allows retry without falsely confirming the transcript.');

    forceNoOp = true; await apply().click(); await page.waitForFunction(() => !window.MDDChat.state.referenceOperation); forceNoOp = false;
    assert.equal(await assistantText().textContent(), oldReply);
    assert.equal(await feedback().locator('.chat-reference-error').count(), 0);
    checks.push('A backend-declared no-op is accepted independently of the client prediction, restoring the existing reply without a regeneration error.');

    await apply().click(); await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.getTurns()[1].referenceRevision === 1);
    await page.locator('#chat-stop').click(); await done();
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[1].status), 'cancelled');
    assert.equal(await assistantText().textContent(), '');
    checks.push('Cancel during regeneration stops that same job and never restores the obsolete bot answer.');

    await newConversation(); fresh(); job.speech.reference_source = 'reading_prompt';
    job.speech.reference_text = confirmed; job.speech.reference_resolution = { status: 'provided' };
    await submit(); await openEditor(); await input().fill(confirmed);
    const readingNoOp = await page.evaluate(() => ({ epoch: window.MDDChat.state.speechEpoch, speech: window.__speech.length }));
    await apply().click(); await page.waitForFunction(() => !window.MDDChat.state.referenceOperation);
    assert.deepEqual(await page.evaluate(() => ({ epoch: window.MDDChat.state.speechEpoch, speech: window.__speech.length })), readingNoOp);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[1].metadata.reference_revision), 0);
    checks.push('An unchanged reading prompt takes precedence over a different ASR transcript, preserving its ongoing reply and voice.');

    await newConversation(); fresh(); await submit(); await drainVoice(); await openEditor(); await input().fill('I work at Quendor.');
    await apply().click(); await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.getTurns()[1].referenceRevision === 1);
    assert.ok((await feedback().locator('.chat-reference-note').first().textContent()).includes('Quendor'));
    assert.equal(await userText().textContent(), 'I work at Quendor.');
    job.text = 'What do you do there?'; job.status = 'ready'; job.metadata.reply_source = 'qwen';
    await done(); await drainVoice(); assert.equal(await assistantText().textContent(), job.text);
    checks.push('A confirmed proper name still regenerates the conversation while pronunciation clearly reports the missing dictionary reference.');

    await newConversation(); await voiceSetting('ai33', false); fresh(); await submit();
    holdVoice = true; await page.locator('.chat-message.assistant .chat-message-actions button').first().click();
    await page.waitForTimeout(40); assert.ok(releaseVoice);
    await openEditor(); await input().fill(confirmed); await apply().click();
    await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.getTurns()[1].referenceRevision === 1);
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-auto-read').check(); await page.keyboard.press('Escape');
    job.text = newReply; job.status = 'ready'; job.metadata.reply_source = 'qwen';
    await done(); await page.waitForFunction(() => window.__audio.length === 1);
    const beforeRelease = voiceCount; releaseVoice(); releaseVoice = null;
    await page.waitForTimeout(80);
    assert.equal(voiceCount, beforeRelease); assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[1].voice.status), 'ready');
    assert.equal(await page.evaluate(() => window.__audio.length), 1);
    checks.push('New-revision ai33 playback is not blocked by an older pending voice task; the old voice response cannot replace the new one.');

    await voiceSetting('browser', false); await newConversation(); fresh();
    const longText = Array.from({ length: 700 }, (_, i) => 'sentence' + i).join(' ');
    job.status = 'generating'; job.text = longText;
    await page.evaluate(() => { window.__send = window.MDDChat.sendAudio(new ArrayBuffer(48), 'scroll-fixture.wav'); });
    await page.waitForFunction(() => window.MDDChat.state.busy && window.MDDChat.state.activeTurn?.status === 'generating');
    const manualPosition = await page.evaluate(() => {
      window.MDDChat.state.liveReply.revealAll();
      const scroll = document.getElementById('chat-scroll'); scroll.scrollTop = scroll.scrollHeight;
      scroll.scrollTop = Math.max(0, scroll.scrollTop - 180); scroll.dispatchEvent(new WheelEvent('wheel', { deltaY: -180 }));
      return { top: scroll.scrollTop, maximum: scroll.scrollHeight - scroll.clientHeight };
    });
    assert.ok(manualPosition.maximum > 180);
    job.text = longText + ' additional streamed words';
    await page.waitForTimeout(850);
    assert.ok(Math.abs(await page.locator('#chat-scroll').evaluate(node => node.scrollTop) - manualPosition.top) < 2);
    job.status = 'ready'; await done();
    await page.evaluate(() => window.MDDChat.state.liveReply.revealAll());
    checks.push('Manual upward scrolling is preserved during later polling and text updates, even with every feedback collapse closed.');

    await voiceSetting('browser', true); await newConversation(); fresh(); await submit(); await drainVoice(); await openEditor(); await input().fill(confirmed);
    holdReference = true; await apply().click(); await page.waitForTimeout(40); assert.ok(releaseReference);
    await page.evaluate(() => window.switchView('practice'));
    releaseReference(); releaseReference = null; holdReference = false;
    await done();
    assert.equal(await page.evaluate(() => window.MDDChat.state.speaking), null);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[1].status), 'cancelled');
    checks.push('Leaving the chat during an in-flight confirmation prevents late voice and cancels the accepted background generation.');
    assert.deepEqual(errors, []);
    const report = { status: 'passed', checks, exceptions: errors, reference_requests: requests.length, cancel_requests: cancelCount,
      mocked_voice_requests: voiceCount, paid_tts_calls: 0, inference: 'mock jobs only',
      speech: 'simulated speech boundaries and audio; physical audio not tested' };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    const snapshot = await page.evaluate(() => ({ busy: window.MDDChat?.state.busy, speaking: window.MDDChat?.state.speaking,
      turns: window.MDDChat?.getTurns().map(turn => ({ id: turn.id, role: turn.role, status: turn.status, content: turn.content.slice(0, 500),
        confirmedText: turn.confirmedText, revision: turn.referenceRevision })), error: document.getElementById('chat-error')?.textContent,
      referenceError: document.querySelector('.chat-reference-error')?.textContent, spoken: window.__speech?.map(call => call.text) }));
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify({ status: 'failed', checks, exceptions: errors, error: error.message, snapshot }, null, 2));
    await page.screenshot({ path: path.join(output, 'failure.png'), fullPage: true }); console.error(JSON.stringify(snapshot, null, 2)); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
