'use strict';
// Exercise the real UI with saved local inference fixtures. Network jobs and
// speech playback are mocked; this script never starts inference or paid TTS.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const output = path.resolve('data/generated/chat-ui-cleanup-qa/pronunciation-browser');
const fixturePath = process.env.MDD_QA_FIXTURES || path.resolve('data/generated/pronunciation-error-qa/diagnostic-cases.json');
const speechOf = value => value?.speech || value?.result?.speech || value;

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const fixtures = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const optionalFixtures = JSON.parse(fs.readFileSync('data/generated/optional-reference-qa/diagnostics.json', 'utf8'));
  const provisionalCase = optionalFixtures.examples.find(value => value.initial?.sound_candidates?.status === 'available'
    && value.initial.sound_candidates.count > 0 && value.initial.reference_text !== value.confirmed.reference_text);
  assert.ok(provisionalCase, 'A usable provisional ASR reference with immediate diagnostic suggestions is required.');
  const errorCase = { ...provisionalCase, speech: provisionalCase.confirmed,
    provenance: 'saved cached real-audio ASR+B evidence reassessed for optional reference; no new inference' }, correctCase = fixtures.correct;
  const errorSpeech = speechOf(errorCase), correctSpeech = speechOf(correctCase);
  const noCandidatesCase = fixtures.no_candidates || correctCase;
  const noCandidatesSpeech = structuredClone(speechOf(noCandidatesCase));
  const noCandidatesTranscriptPlaceholder = !noCandidatesSpeech.transcript;
  // Cached posterior evidence has no ASR transcript. This UI-only placeholder
  // makes the reference controls testable without changing the evidence file or
  // claiming an additional ASR inference.
  if (noCandidatesTranscriptPlaceholder) noCandidatesSpeech.transcript = noCandidatesSpeech.reference_text;
  assert.ok(errorSpeech.sound_candidates?.words?.length, 'A confirmed diagnostic-error fixture is required.');
  assert.equal(noCandidatesSpeech.sound_candidates?.words?.length, 0, 'Include a separate no_candidates control if the manually correct recording has diagnostic false positives.');
  const fallbackSample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8')
    .trim().split('\n').map(JSON.parse).find(row => row.utterance_id === 'NJS/arctic_a0054');
  const config = { models: [{ id: 'qwen-base', ready: true }, { id: 'qwen-trained', ready: true }], topics: [],
    practice_prompts: [{ id: 'practice', text: errorSpeech.reference_text }], speech_ready: true,
    ai33_available: false, voice_locales: [], max_audio_seconds: 30, reference_regenerates_reply: true };
  const reply = 'Hi! I can help you practise that sound. What would you like to talk about next?';
  const regeneratedReply = 'Sorry for the misunderstanding. Which topic would you like to practise next?';
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const exceptions = [], checks = [], actionSpacing = {}; let job, serial = 0, referenceRequests = 0, referenceReplyPending = false;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  const fresh = speech => job = { id: String(++serial).padStart(32, '0'), model_id: 'qwen-trained', status: 'ready',
    text: reply, speech: structuredClone(speech), metadata: { reference_revision: 0 }, error: null, voice: { status: 'not_requested' } };
  await page.route('**/api/chat/**', route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/speech') return route.fulfill({ json: job });
    if (pathname.startsWith('/api/chat/jobs/')) {
      if (referenceReplyPending) { referenceReplyPending = false; job.status = 'ready'; job.text = regeneratedReply; job.metadata.reply_source = 'qwen'; }
      return route.fulfill({ json: job });
    }
    if (pathname === '/api/chat/reference') {
      referenceRequests++;
      const request = route.request().postDataJSON();
      assert.equal(request.reference_text, errorSpeech.reference_text);
      job.speech = structuredClone(errorSpeech); job.speech.reference_source = 'user_confirmed';
      job.speech.conversation_text = request.reference_text;
      job.speech.reference_resolution = { status: 'confirmed', source: 'user_confirmed' };
      job.metadata = { reference_revision: 1 };
      job.status = 'generating'; job.text = ''; job.voice = { status: 'not_requested' };
      referenceReplyPending = true;
      return route.fulfill({ json: { ...structuredClone(job), reference_update: { changed: true, regenerated_reply: true } } });
    }
    return route.fulfill({ status: 500, json: { message: 'Unexpected API request blocked by pronunciation UI QA.' } });
  });
  const feedback = page.locator('.chat-turn-feedback').last();
  const summary = feedback.locator(':scope > summary');
  const label = feedback.locator('.chat-feedback-label');
  const fullReply = page.locator('.chat-message.assistant .chat-message-text').last();
  const upload = async (sample, expectedReply = reply) => {
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample?.audio_path || fallbackSample.audio_path));
    await page.waitForFunction(expected => !window.MDDChat.state.busy && window.MDDChat.getTurns().at(-1)?.content === expected, expectedReply);
    await page.waitForFunction(expected => document.querySelector('.chat-message.assistant:last-child .chat-message-text')?.textContent === expected, expectedReply);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
  };
  const newConversation = async () => {
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-new').click(); await page.keyboard.press('Escape');
  };
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click();
    const provisional = structuredClone(provisionalCase.initial);
    fresh(provisional); await upload(errorCase);
    assert.equal(await label.textContent(), `${provisional.sound_candidates.count} từ cần kiểm tra`);
    assert.equal(await feedback.getAttribute('open'), null); assert.equal(await feedback.getAttribute('data-assessment'), 'candidates');
    const measureActions = () => page.evaluate(() => [...document.querySelectorAll('.chat-message-actions button')]
      .filter(button => ['Phát lại', 'Nghe bản thu'].includes(button.textContent.trim())).map(button => {
        const style = getComputedStyle(button), svg = button.querySelector('svg');
        const text = [...button.childNodes].find(child => child.nodeType === Node.TEXT_NODE && child.textContent.trim());
        let measuredGap = null;
        if (svg && text) { const range = document.createRange(); range.selectNodeContents(text); measuredGap = range.getBoundingClientRect().left - svg.getBoundingClientRect().right; }
        return { text: button.textContent.trim(), display: style.display, align_items: style.alignItems, css_gap: style.columnGap,
          svg_flex_shrink: svg ? getComputedStyle(svg).flexShrink : null, measured_gap_px: measuredGap };
      }));
    actionSpacing.desktop = await measureActions();
    await page.screenshot({ path: path.join(output, 'desktop-message-actions.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 390, height: 844 }); actionSpacing.mobile = await measureActions();
    await page.screenshot({ path: path.join(output, 'mobile-message-actions.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await summary.click(); assert.equal(await feedback.locator('.chat-reference-editor').getAttribute('open'), null);
    assert.equal(await feedback.locator('.chat-word-practice').count(), provisional.sound_candidates.count);
    assert.equal(await feedback.locator('.chat-sound-word.possible_difference').count(), provisional.sound_candidates.count);
    assert.equal(await feedback.locator('.chat-reference-editor > summary').textContent(), 'Sửa câu');
    assert.equal((await feedback.textContent()).includes('Cần xác nhận câu'), false);
    checks.push('Usable provisional ASR immediately shows amber word suggestions and listening controls; the transcript editor stays closed and optional.');
    const original = await page.evaluate(() => ({ text: window.MDDChat.getTurns().find(turn => turn.role === 'user').content,
      audio: document.querySelector('.chat-message.user audio')?.src }));
    await feedback.locator('.chat-reference-editor > summary').click();
    await feedback.locator('.chat-reference-input').fill(errorSpeech.reference_text);
    await feedback.locator('.chat-reference-apply').click();
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1).speech.reference_source === 'user_confirmed');
    await page.waitForFunction(expected => !window.MDDChat.state.busy && document.querySelector('.chat-message.assistant:last-child .chat-message-text')?.textContent === expected, regeneratedReply);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await page.locator('.chat-message.user .chat-message-text').last().textContent(), errorSpeech.reference_text);
    const candidates = errorSpeech.sound_candidates.words.length;
    assert.equal(await label.textContent(), `${candidates} từ cần kiểm tra`);
    assert.equal(await feedback.getAttribute('data-assessment'), 'candidates');
    assert.equal(await feedback.locator('.chat-review-result').count(), 0);
    assert.equal(await feedback.locator('.chat-word-practice[data-evidence=diagnostic]').count(), candidates);
    assert.equal(await feedback.locator('.chat-sound-word.possible_difference').count(), candidates);
    assert.equal(await feedback.locator('.chat-word-practice').first().locator('.chat-word-ipa').textContent() === 'Chưa có phiên âm', false);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().at(-1).speech.comparison.words.some(word => word.status === 'difference')), false);
    assert.deepEqual(await page.evaluate(() => ({ text: window.MDDChat.getTurns().find(turn => turn.role === 'user').content,
      audio: document.querySelector('.chat-message.user audio')?.src })), original);
    checks.push('Confirming the reference regenerates the bot reply, displays the confirmed sentence and reveals amber candidates with IPA, while calibrated comparison and raw ASR/audio remain unchanged.');
    const first = feedback.locator('.chat-word-practice').first();
    const firstText = await first.locator('.chat-word-details').textContent();
    await first.locator('.chat-word-details').click();
    assert.equal(await feedback.locator('.chat-sound-word.possible_difference[open]').count(), 1);
    assert.ok((await feedback.locator('.chat-sound-word.possible_difference[open]').textContent()).includes('Target phones'));
    const beforeListen = await page.evaluate(() => window.__speech.length);
    await first.locator('.chat-word-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeListen + 1);
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), firstText);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await feedback.locator('.chat-pronunciation:not(.chat-completion-extra) .chat-sentence-listen').count(), 1);
    await page.screenshot({ path: path.join(output, 'desktop-diagnostic-candidates.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    await page.screenshot({ path: path.join(output, 'mobile-diagnostic-candidates.png'), fullPage: true, animations: 'disabled' });
    await page.setViewportSize({ width: 1440, height: 1000 });
    checks.push('Each suggested word opens tentative sound details and can be heard independently; full-sentence listening stays in Sound details and mobile does not overflow.');
    assert.equal(await fullReply.textContent(), regeneratedReply);
    assert.ok((await page.evaluate(() => window.__speech.map(call => call.text).join(' '))).includes('Which topic would you like'));
    checks.push('The regenerated conversation reply and simulated voice use the new response, beyond the old initial greeting.');

    if (correctSpeech.sound_candidates.words.length) {
      await newConversation(); fresh(correctSpeech); await upload(correctCase);
      assert.equal(await label.textContent(), `${correctSpeech.sound_candidates.words.length} từ cần kiểm tra`);
      await summary.click();
      assert.equal(await feedback.locator('.chat-word-practice[data-evidence=diagnostic]').count(), correctSpeech.sound_candidates.words.length);
      assert.equal(await feedback.locator('.chat-review-result').count(), 0);
      assert.equal(await page.evaluate(() => window.MDDChat.getTurns().at(-1).speech.comparison.words.some(word => word.status === 'difference')), false);
      checks.push('Diagnostic false positives on a manually correct recording remain explicitly tentative; the UI does not promote them to pronunciation errors.');
    }
    await newConversation(); fresh(noCandidatesSpeech); await upload(noCandidatesCase);
    assert.equal(await label.textContent(), 'Chưa đủ bằng chứng'); assert.equal(await feedback.getAttribute('open'), null);
    await summary.click();
    assert.equal(await feedback.locator('.chat-word-practice').count(), 0);
    assert.equal(await feedback.locator('.chat-sound-word.possible_difference').count(), 0);
    assert.equal(/all correct|phát âm đúng hết|không có lỗi/i.test(await feedback.textContent()), false);
    assert.equal(await feedback.locator('.chat-review-result').count(), 0);
    await page.screenshot({ path: path.join(output, 'control-no-candidates.png'), fullPage: true, animations: 'disabled' });
    checks.push('A control with no candidates displays uncertainty, without treating abstention as correct pronunciation.');
    assert.equal(await page.locator('.chat-message.assistant .chat-message-label').last().textContent(), 'Qwen');
    await newConversation(); fresh(noCandidatesSpeech);
    job.metadata.reply_source = 'clarification_fallback'; job.text = 'Could you say that last part again?';
    await upload(noCandidatesCase, job.text);
    assert.equal(await page.locator('.chat-message.assistant .chat-message-label').last().textContent(), 'Cần làm rõ');
    checks.push('An explicit authored clarification fallback has its own label; normal model responses retain Qwen.');
    await newConversation(); fresh(noCandidatesSpeech);
    job.metadata.reply_source = 'conversation_unavailable'; job.metadata.reference_revision = 1;
    job.speech.reference_source = 'user_confirmed'; job.speech.conversation_text = noCandidatesSpeech.reference_text;
    job.speech.reference_resolution = { status: 'confirmed' };
    job.text = "I'm having trouble continuing this conversation right now.";
    await upload(noCandidatesCase, job.text);
    assert.equal(await page.locator('.chat-message.assistant .chat-message-label').last().textContent(), 'Chưa có phản hồi');
    const retryAction = page.locator('.chat-message.assistant .chat-clarify-action').last();
    assert.equal(await retryAction.textContent(), 'Sửa câu hoặc nói lại');
    await retryAction.click();
    assert.equal(await feedback.evaluate(node => node.open), true);
    assert.equal(await feedback.locator('.chat-reference-input').evaluate(node => node === document.activeElement), true);
    checks.push('A conversation-unavailable response after confirmed text has a distinct label and a visible edit/repeat recovery action.');
    assert.equal(referenceRequests, 1); assert.deepEqual(exceptions, []);
    const report = { status: 'passed', checks, exceptions, action_spacing: actionSpacing, fixture_file: path.relative(process.cwd(), fixturePath),
      speech_output: 'mock browser speech events; physical audio not tested', inference: 'saved local fixture outputs replayed; no new inference',
      reference_requests: referenceRequests, candidate_words: errorSpeech.sound_candidates.words.map(word => word.text),
      provisional_candidate_words: provisional.sound_candidates.words.map(word => word.text),
      manually_correct_control_candidate_count: correctSpeech.sound_candidates.words.length,
      no_candidates_fixture: noCandidatesCase.utterance_id || null,
      fixture_provenance: { error: errorCase.provenance, correct: correctCase.provenance, no_candidates: noCandidatesCase.provenance },
      no_candidates_transcript: noCandidatesTranscriptPlaceholder ? 'UI-only mock copied from confirmed reference; no ASR inference' : 'saved ASR transcript',
      reference_contract: 'same job, generating blank POST snapshot, incremented revision, confirmed conversation text and polled regenerated reply',
      paid_tts_calls: 0 };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    const snapshot = await page.evaluate(() => ({ error: document.getElementById('chat-error')?.textContent,
      feedback: document.querySelector('.chat-turn-feedback')?.textContent,
      speaking: window.MDDChat?.state.speaking, reply: window.MDDChat?.getTurns().at(-1)?.content,
      spoken: window.__speech?.map(call => call.text) }));
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify({ status: 'failed', checks, exceptions, error: error.message, snapshot }, null, 2));
    await page.screenshot({ path: path.join(output, 'ui-failure.png'), fullPage: true }); console.error(JSON.stringify(snapshot, null, 2)); throw error;
  } finally { await browser.close(); }
}
if (require.main === module) run().catch(error => { console.error(error); process.exitCode = 1; });
