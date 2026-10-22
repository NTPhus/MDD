'use strict';
// Real local ASR/MDD/Qwen. Chrome microphone receives a public test WAV, not a physical device.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve('data/generated/voice-chat-qa');
const baseUrl = process.env.MDD_QA_URL || 'http://127.0.0.1:8767';
const rows = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8').trim().split('\n').map(line => JSON.parse(line));
const sample = rows.find(row => row.utterance_id === 'NJS/arctic_a0054');

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
    args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--use-file-for-fake-audio-capture=' + path.resolve(sample.audio_path)] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1080 }, permissions: ['microphone'] });
  const page = await context.newPage(), exceptions = [], jobs = [];
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(() => {
    sessionStorage.removeItem('am-voice-chat-v2'); window.__speech = [];
    Object.defineProperty(window, 'speechSynthesis', { configurable: true, value: {
      cancel() {}, getVoices() { return []; }, speak(utterance) { window.__speech.push({ text: utterance.text, lang: utterance.lang }); setTimeout(() => utterance.onend?.(), 10); }
    }});
  });
  try {
    await page.goto(baseUrl + '/#chat');
    await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    assert.equal(await page.locator('#chat-view textarea').count(), 0);
    assert.equal(await page.locator('#chat-input').count(), 0);
    assert.equal(await page.locator('#chat-language').count(), 0);
    await page.screenshot({ path: path.join(output, 'voice-desktop.png'), fullPage: true, animations: 'disabled' });
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
    await page.waitForFunction(() => window.MDDChat.state.busy);
    await page.waitForFunction(() => !window.MDDChat.state.busy, null, { timeout: 240000 });
    let turn = await page.evaluate(() => window.MDDChat.state.conversations[window.MDDChat.state.model].at(-1));
    assert.equal(turn.status, 'ready', JSON.stringify(turn));
    assert.ok(turn.content.trim()); assert.ok(turn.speech.transcript);
    assert.equal(turn.metadata.language, 'en-US'); assert.equal(turn.speech.reference_source, 'asr');
    assert.ok(await page.locator('#chat-messages audio').count());
    assert.ok(await page.locator('.chat-sound-word.difference').count());
    await page.locator('.chat-pronunciation-title').last().click();
    await page.locator('.chat-sound-word.difference summary').first().click();
    assert.ok(await page.locator('.chat-sound-word[open] .chat-sound-detail').isVisible());
    await page.waitForFunction(() => window.__speech.some(value => value.lang === 'en-US'));
    jobs.push({ input: 'uploaded WAV, browser decoding/resampling', result: turn });
    await page.screenshot({ path: path.join(output, 'voice-upload-result.png'), fullPage: true, animations: 'disabled' });
    // Verify physical PCM capture through Web Audio/AudioWorklet using Chrome's fixture microphone.
    await page.locator('#chat-settings>summary').click();
    await page.locator('#chat-model').selectOption('qwen-base');
    await page.locator('#chat-mode').selectOption('reading');
    assert.ok(await page.locator('#chat-practice-panel').isVisible());
    await page.locator('#chat-read-prompt').click();
    await page.keyboard.press('Escape');
    await page.locator('#chat-mic').click();
    await page.waitForFunction(() => window.MDDChat.state.recording);
    await page.waitForTimeout(2200);
    await page.locator('#chat-mic').click();
    await page.waitForFunction(() => window.MDDChat.state.busy);
    await page.waitForFunction(() => !window.MDDChat.state.busy, null, { timeout: 240000 });
    turn = await page.evaluate(() => window.MDDChat.state.conversations[window.MDDChat.state.model].at(-1));
    assert.equal(turn.status, 'ready', JSON.stringify(turn)); assert.equal(turn.speech.reference_source, 'reading_prompt');
    assert.equal(turn.speech.reference_text, 'I think this is a good idea.');
    jobs.push({ input: 'Web Audio microphone, Chrome fake-device WAV', result: turn });
    await page.locator('.navigation [data-view=practice]').click(); assert.ok(await page.locator('#practice-view').isVisible());
    await page.locator('.navigation [data-view=training]').click(); assert.ok(await page.locator('#training-view').isVisible());
    await page.locator('.navigation [data-view=chat]').click();
    for (const width of [1440, 768, 390, 360]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Overflow at ${width}`);
      if (width === 390) await page.screenshot({ path: path.join(output, 'voice-mobile.png'), fullPage: true, animations: 'disabled' });
    }
    assert.deepEqual(exceptions, []);
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify({ status: 'passed',
      inference: 'real CPU ASR, phoneme B, Qwen original dialogue + selected pronunciation coach',
      microphone: 'Chrome fixture microphone; physical device not tested', tts: 'simulated English browser speech; no paid ai33 calls', jobs }, null, 2));
    console.log('Voice chat browser QA passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
