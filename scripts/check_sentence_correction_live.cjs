'use strict';
// Full local audio -> completion-only Qwen -> original pronunciation -> sentence retry.
// English input fixtures use Windows SAPI; browser playback events are simulated.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const output = path.resolve('data/generated/sentence-correction-qa');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';

(async () => {
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [], payloads = []; let paid = 0;
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().endsWith('/api/chat/speech')) { const data = request.postDataJSON(); delete data.audio_base64; payloads.push(data); } });
  await page.addInitScript(mockVoice);
  await page.route('**/api/chat/voice', route => { paid++; return route.fulfill({ status: 500, json: { message: 'Paid TTS blocked in QA' } }); });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    await page.locator('#chat-free-talk').click();
    await page.locator('#chat-audio-file').setInputFiles(path.join(output, 'grammar-fragment.wav'));
    await page.waitForFunction(() => window.MDDChat.state.busy);
    await page.waitForFunction(() => ['ready', 'unavailable'].includes(window.MDDChat.getTurns().at(-1)?.speech?.correction?.status), null, { timeout: 180000 });
    const firstCorrection = await page.evaluate(() => window.MDDChat.getTurns().at(-1).speech.correction);
    console.log(JSON.stringify({ stage: 'correction', ...firstCorrection }));
    assert.equal(firstCorrection.status, 'ready'); assert.equal(firstCorrection.mode, 'completion_only');
    const words = text => text.toLowerCase().match(/[a-z]+/g);
    assert.deepEqual(words(firstCorrection.corrected), words(firstCorrection.original));
    await page.screenshot({ path: path.join(output, 'live-correction.png'), fullPage: true, animations: 'disabled' });
    await page.waitForFunction(() => !window.MDDChat.state.busy, null, { timeout: 240000 });
    await page.waitForFunction(() => !window.MDDChat.state.speaking, null, { timeout: 30000 });
    const original = await page.evaluate(() => window.MDDChat.getTurns().at(-1));
    assert.equal(original.status, 'ready'); assert.equal(original.speech.reference_text, original.speech.transcript);
    assert.equal(original.speech.reference_source, 'asr'); assert.ok(original.speech.comparison);
    assert.equal(original.speech.spoken_response, original.content);
    assert.equal(await page.locator('.chat-turn-feedback').last().evaluate(node => node.open), false);
    await page.locator('.chat-turn-feedback').last().locator(':scope > summary').click();
    const extra = page.locator('.chat-completion-extra').last();
    if (!(await extra.isVisible())) {
      assert.equal(original.speech.correction.changed, false);
      fs.writeFileSync(path.join(output, 'live-summary.json'), JSON.stringify({ status: 'passed', original,
        repeat_completion: 'not needed: no completion was added', paid_tts_calls: paid, errors }, null, 2));
      return;
    }
    await extra.locator(':scope > summary').click();
    assert.equal(await page.locator('.chat-correction-sentence').last().textContent(), original.speech.correction.corrected);
    await page.locator('.chat-correction-listen').last().click();
    assert.equal(await page.evaluate(() => window.__speech.at(-1).text), original.speech.correction.corrected);
    await page.locator('.chat-correction-practise').last().click();
    assert.equal(await page.locator('#chat-reading-text').textContent(), original.speech.correction.corrected);
    await page.locator('#chat-audio-file').setInputFiles(path.join(output, 'grammar-fragment.wav'));
    await page.waitForFunction(() => window.MDDChat.state.busy);
    await page.waitForFunction(() => window.MDDChat.getTurns().at(-1)?.speech?.reference_source === 'corrected_sentence', null, { timeout: 120000 });
    const repeatedEvidence = await page.evaluate(() => window.MDDChat.getTurns().at(-1).speech);
    assert.equal(repeatedEvidence.reference_text, original.speech.correction.corrected);
    assert.equal(repeatedEvidence.correction.status, 'not_needed'); assert.ok(repeatedEvidence.comparison);
    assert.equal(payloads[1].correction_job_id, original.jobId); assert.equal('practice_id' in payloads[1], false);
    // All acoustic evidence is available. Stop this own QA job instead of waiting for another full coach pass.
    await page.locator('#chat-stop').click(); await page.waitForFunction(() => !window.MDDChat.state.busy, null, { timeout: 30000 });
    await page.screenshot({ path: path.join(output, 'live-repeat.png'), fullPage: true, animations: 'disabled' });
    assert.deepEqual(errors, []); assert.equal(paid, 0);
    const report = { status: 'passed', input: 'local Windows SAPI English fixture, including original grammar', inference: 'real local ASR/B/Qwen completion-only + selected pronunciation coach',
      playback: 'simulated browser speech events; physical speaker not tested', paid_tts_calls: paid, errors,
      original, repeated_evidence: repeatedEvidence, requests_without_audio: payloads, repeat_completion: 'cancelled after real acoustic comparison' };
    fs.writeFileSync(path.join(output, 'live-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ status: 'passed', original: firstCorrection.original, corrected: firstCorrection.corrected, repeat_reference: repeatedEvidence.reference_text }));
  } catch (error) {
    console.error(await page.evaluate(() => ({ error: document.getElementById('chat-error').textContent, turn: window.MDDChat?.getTurns().at(-1) })));
    throw error;
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
