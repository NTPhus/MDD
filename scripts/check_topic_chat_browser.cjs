'use strict';
// Real topic API + audio decoding; incremental acoustic/model jobs and voices are fixtures.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
const out = path.resolve('data/generated/topic-chat-qa');
const sample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8').trim().split('\n').map(JSON.parse).find(r => r.utterance_id === 'NJS/arctic_a0054');
const acoustic = JSON.parse(fs.readFileSync('data/generated/voice-chat-qa/browser-summary.json', 'utf8')).jobs[0].result.speech;

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [], payloads = [], checks = []; let job, serial = 0, paidCalls = 0;
  page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(mockVoice, { preserveHistory: true });
  await page.route('**/api/chat/speech', async route => { payloads.push(route.request().postDataJSON()); await route.fulfill({ json: job }); });
  await page.route('**/api/chat/jobs/ffffffff*', route => route.fulfill({ json: job }));
  await page.route('**/api/chat/voice', route => { paidCalls++; return route.fulfill({ status: 500, json: { message: 'Paid calls blocked in QA' } }); });
  const newJob = () => job = { id: 'ffffffff' + String(++serial).padStart(24, '0'), status: 'transcribing', text: '', speech: null, metadata: {}, error: null };
  const upload = async () => { await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path)); await page.waitForFunction(() => window.MDDChat.state.busy); };
  const complete = async text => {
    job.text = text; job.status = 'ready'; job.speech.spoken_response = text;
    await page.waitForFunction(() => !window.MDDChat.state.busy);
    await page.waitForFunction(() => !window.MDDChat.state.speaking && window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text);
  };
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.topics?.length === 8);
    assert.equal(await page.locator('.chat-topic-card').count(), 8); assert.equal(await page.evaluate(() => window.__speech.length), 0);
    assert.equal(await page.locator('#chat-view textarea').count(), 0);
    await page.screenshot({ path: path.join(out, 'topics-desktop.png'), fullPage: true, animations: 'disabled' });
    await page.locator('[data-topic=travel]').click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting && window.MDDChat.getTurns().at(-1)?.opening);
    await page.waitForFunction(() => !window.MDDChat.state.speaking);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 1);
    assert.match(await page.locator('.assistant .chat-message-text').textContent(), /plan a trip/);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns()[0].metadata.source), 'topic_catalog');
    assert.equal(payloads.length, 0); checks.push('Selecting a topic opens an authored English greeting before any recording, with voice.');
    await page.screenshot({ path: path.join(out, 'travel-opening.png'), fullPage: true, animations: 'disabled' });
    const words = await page.locator('#chat-hint-keywords button').allTextContents(); assert.ok(words.includes('destination'));
    await page.locator('#chat-hint-keywords button').first().click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), 'destination');
    const beforeRecording = await page.evaluate(() => { window.MDDChat.state.recording = true; return window.__speech.length; });
    await page.locator('#chat-hint-keywords button').first().click();
    assert.equal(await page.evaluate(() => window.__speech.length), beforeRecording);
    await page.evaluate(() => { window.MDDChat.state.recording = false; });
    checks.push('Keyword pronunciation and sentence starters are available without submitting a message.');
    newJob(); await upload();
    job.status = 'analyzing'; job.speech = { ...structuredClone(acoustic), coaching: { status: 'pending' }, guidance: { topic_id: 'travel', keywords: ['explore', 'sightseeing', 'try local food', 'take photos'], starter: 'When I get there, I would like to ...' } };
    delete job.speech.spoken_response;
    await page.waitForFunction(() => document.querySelector('.chat-turn-review button'));
    assert.equal(await page.locator('.assistant .chat-message-text').last().textContent(), '');
    assert.match(await page.locator('.chat-turn-review').last().textContent(), /MODEL ESTIMATE/);
    await page.locator('.chat-review-words button').first().click();
    assert.ok(await page.locator('.chat-sound-word[open] .chat-sound-detail').isVisible());
    checks.push('Acoustic word estimates appear before Qwen output; tapping a flagged word opens exact phoneme details.');
    assert.equal(payloads[0].topic_id, 'travel'); assert.deepEqual(payloads[0].history, []);
    const firstReply = 'That sounds like a quiet place. What would you like to do when you get there?';
    job.status = 'generating'; job.text = firstReply;
    await page.waitForFunction(() => window.__speech.some(s => s.text === 'That sounds like a quiet place.'));
    await complete(firstReply);
    assert.ok(await page.locator('.chat-sound-word[open] .chat-sound-detail').isVisible());
    assert.ok((await page.locator('#chat-hint-keywords button').allTextContents()).includes('explore'));
    await page.screenshot({ path: path.join(out, 'travel-feedback.png'), fullPage: true, animations: 'disabled' });
    newJob(); await upload(); job.speech = { ...structuredClone(acoustic), coaching: { status: 'unavailable' } };
    await complete('I see. How would you plan your trip?');
    assert.equal(payloads[1].history.length, 2);
    assert.equal(payloads[1].history[0].content, acoustic.transcript); assert.equal(payloads[1].history[1].content, firstReply);
    assert.equal(await page.locator('.chat-turn-review:visible').count(), 2);
    checks.push('Each audio turn has a separate assessment and history excludes the authored opener.');
    await page.locator('#chat-topics').click(); await page.locator('[data-topic=food]').click(); await page.locator('#topic-setup-start').click();
    await page.waitForFunction(() => !window.MDDChat.state.starting && window.MDDChat.getTurns()[0]?.metadata?.topic_id === 'food');
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 1);
    await page.locator('#chat-topics').click(); await page.locator('[data-topic=travel]').click(); await page.locator('#topic-setup-start').click();
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 5);
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-model').selectOption('qwen-base'); await page.keyboard.press('Escape');
    await page.waitForFunction(() => !window.MDDChat.state.starting && window.MDDChat.getTurns().length === 1);
    await page.locator('#chat-settings>summary').click(); await page.locator('#chat-model').selectOption('qwen-trained'); await page.keyboard.press('Escape');
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 5);
    checks.push('Topic and coach histories are isolated and can be resumed independently.');
    for (const width of [1440, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Room overflow at ${width}`);
      if (width === 390) {
        assert.ok(await page.locator('#chat-inline-hints').isVisible());
        await page.screenshot({ path: path.join(out, 'room-mobile.png'), fullPage: true, animations: 'disabled' });
      }
      await page.locator('#chat-topics').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Topic overflow at ${width}`);
      if (width === 390) await page.screenshot({ path: path.join(out, 'topics-mobile.png'), fullPage: true, animations: 'disabled' });
      await page.locator('[data-topic=travel]').click(); await page.locator('#topic-setup-start').click();
    }
    checks.push('Topic cards, chat and inline vocabulary fit 1440/768/390/360 px.');
    await page.reload(); await page.waitForFunction(() => window.MDDChat?.state.config?.topics?.length === 8);
    assert.equal(await page.evaluate(() => window.MDDChat.getTurns().length), 5); assert.equal(await page.evaluate(() => window.__speech.length), 0);
    checks.push('Reload restores the current topic and turns without replaying old voice.');
    await page.locator('#chat-topics').click(); await page.locator('#chat-open-reading').click();
    assert.ok(await page.locator('#chat-reading-card').isVisible());
    await page.locator('.navigation [data-view=practice]').click(); assert.ok(await page.locator('#practice-view').isVisible());
    await page.locator('.navigation [data-view=training]').click(); assert.ok(await page.locator('#training-view').isVisible());
    checks.push('Reading mode, pronunciation workspace and training remain available.');
    assert.deepEqual(errors, []); assert.equal(paidCalls, 0);
    const report = { status: 'passed', checks, errors, paid_tts_calls: paidCalls, inference: 'real topic API, simulated speech jobs and voice boundaries' };
    fs.writeFileSync(path.join(out, 'browser-summary.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    console.error('Browser errors', errors);
    console.error(await page.evaluate(() => ({ error: document.getElementById('chat-error').textContent, model: window.MDDChat?.state.model, topic: window.MDDChat?.state.topic, starting: window.MDDChat?.state.starting })));
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
