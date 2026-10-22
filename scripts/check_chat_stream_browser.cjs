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
    const speech = structuredClone(fixture.speech); speech.coaching = { status: 'pending' }; delete speech.spoken_response;
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
    job.text = reply; job.status = 'coaching';
    await page.waitForFunction(() => window.__speech.length === 1);
    assert.equal(await text.textContent(), reply, 'All sentences must be visible before speech completes.');
    assert.equal(await page.evaluate(() => window.__speech[0].status), 'coaching');
    await page.waitForFunction(() => document.querySelector('.chat-reply-word.is-speaking'));
    await page.screenshot({ path: path.join(output, 'desktop-speaking.png'), fullPage: true, animations: 'disabled' });
    checks.push('The completed reply is shown in full before voice, then only the spoken word is highlighted.');
    await page.waitForFunction(() => window.__speech.length === 2);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await feedback.getAttribute('open'), null);
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
    job.status = 'coaching';
    await page.waitForFunction(() => window.MDDChat.state.speaking);
    const beforeStop = await page.evaluate(() => window.__speech.length);
    await page.locator('#chat-stop-voice').click();
    await page.evaluate(() => window.__utterances.at(-1).onend?.()); // Simulate a late device callback.
    await finish(); await page.waitForTimeout(300);
    assert.equal(await page.evaluate(() => window.__speech.length), beforeStop); assert.equal(await text.textContent(), reply);
    checks.push('Stop voice cancels queued and late audio while the remaining text still appears.');
    await newConversation(); fresh('A short reply.'); await upload();
    job.status = 'coaching';
    await page.waitForFunction(() => window.MDDChat.state.speaking);
    const beforeLeave = await page.evaluate(() => window.__speech.length);
    // The full-screen room hides the workspace sidebar; exercise its view hook.
    await page.evaluate(() => window.switchView('practice'));
    assert.ok(await page.locator('#practice-view').isVisible());
    await finish(); await page.locator('.navigation [data-view=chat]').click();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeLeave);
    checks.push('Leaving the chat prevents late voice playback, including after returning.');
    await newConversation(); fresh('Another reply.'); await upload();
    job.status = 'coaching';
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
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-mode').selectOption('reading'); await page.keyboard.press('Escape');
    assert.ok(await page.locator('#chat-reading-card').isVisible());
    assert.ok((await page.locator('#chat-reading-text').textContent()).length > 10);
    await page.locator('#chat-reading-listen').click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), await page.locator('#chat-reading-text').textContent());
    await page.setViewportSize({ width: 360, height: 900 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'Reading mode overflows on mobile');
    await page.screenshot({ path: path.join(output, 'mobile-reading.png'), fullPage: true, animations: 'disabled' });
    checks.push('The reading sentence and its Listen button remain visible after closing Settings.');
    assert.equal(voiceRequests, 0); assert.deepEqual(exceptions, []);
    const report = { status: 'passed', checks, exceptions, inference: 'mock jobs', speech: 'mock English speech boundaries; physical audio not tested', paid_tts_calls: voiceRequests };
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
