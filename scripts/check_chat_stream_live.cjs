'use strict';
// One actual local inference. Voice events are simulated: this does not test a physical speaker.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const { mockVoice } = require('./check_chat_stream_browser.cjs');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const topic = process.env.MDD_QA_TOPIC || null;
const output = path.resolve(topic ? 'data/generated/topic-chat-qa' : 'data/generated/chat-stream-qa');
const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';

(async () => {
  fs.mkdirSync(output, { recursive: true });
  const sample = fs.readFileSync('data/processed/feedback-projector-en-v1/test.jsonl', 'utf8').trim().split('\n').map(JSON.parse).find(row => row.utterance_id === 'NJS/arctic_a0054');
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const exceptions = []; let paidCalls = 0;
  page.on('pageerror', error => exceptions.push(error.message));
  await page.addInitScript(mockVoice);
  await page.route('**/api/chat/voice', route => { paidCalls++; return route.fulfill({ status: 500, json: { message: 'Paid TTS disabled in QA.' } }); });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config?.speech_ready);
    if (topic) {
      await page.locator(`[data-topic="${topic}"]`).click();
      await page.waitForFunction(() => !window.MDDChat.state.starting && window.MDDChat.getTurns()[0]?.opening);
      await page.waitForFunction(() => !window.MDDChat.state.speaking);
      await page.evaluate(() => { window.__speech = []; });
    }
    const start = await page.evaluate(() => performance.now());
    await page.locator('#chat-audio-file').setInputFiles(path.resolve(sample.audio_path));
    await page.waitForFunction(() => window.MDDChat.state.busy);
    await page.waitForFunction(() => window.__speech.length || !window.MDDChat.state.busy, null, { timeout: 240000 });
    const first = await page.evaluate(() => ({ voice: window.__speech[0], busy: window.MDDChat.state.busy,
      status: window.MDDChat.state.activeTurn?.status, visibleText: [...document.querySelectorAll('.assistant .chat-message-text')].at(-1).textContent }));
    assert.ok(first.voice, JSON.stringify(await page.evaluate(() => window.MDDChat.getTurns().at(-1))));
    assert.ok(['generating', 'coaching'].includes(first.voice.status), JSON.stringify(first));
    console.log(JSON.stringify({ first_voice_sec: (first.voice.time - start) / 1000, status: first.voice.status, text: first.voice.text }));
    await page.screenshot({ path: path.join(output, 'live-speaking.png'), fullPage: true, animations: 'disabled' });
    await page.waitForFunction(() => !window.MDDChat.state.busy, null, { timeout: 240000 });
    const ready = await page.evaluate(() => performance.now());
    await page.waitForFunction(() => !window.MDDChat.state.speaking && window.MDDChat.state.liveReply.visibleText === window.MDDChat.state.liveReply.text, null, { timeout: 30000 });
    const result = await page.evaluate(() => ({ turn: window.MDDChat.getTurns().at(-1), speech: window.__speech }));
    assert.equal(result.turn.status, 'ready'); assert.ok(result.turn.speech.transcript); assert.ok(result.speech.every(s => s.lang === 'en-US'));
    assert.equal(result.speech.filter(s => s.text === first.voice.text).length, 1);
    assert.equal(await page.locator('.chat-message.assistant .chat-message-text').last().textContent(), result.turn.content);
    if (topic) {
      assert.equal(result.turn.metadata.topic_id, topic); assert.ok(result.turn.speech.follow_up.question.endsWith('?'));
      assert.ok(['qwen', 'topic_catalog_fallback'].includes(result.turn.speech.follow_up.source));
      assert.equal(result.turn.speech.guidance.topic_id, topic);
    }
    await page.locator('.chat-pronunciation-title').last().click();
    await page.screenshot({ path: path.join(output, 'live-result.png'), fullPage: true, animations: 'disabled' });
    assert.deepEqual(exceptions, []); assert.equal(paidCalls, 0);
    const report = { status: 'passed', topic_id: topic, sample: sample.utterance_id, inference: 'real local CPU ASR + B + Qwen',
      speech_playback: 'simulated browser word boundaries; physical audio not tested', paid_tts_calls: paidCalls,
      first_voice_sec: +((first.voice.time - start) / 1000).toFixed(3), completed_sec: +((ready - start) / 1000).toFixed(3), first_voice_status: first.voice.status,
      exceptions, ...result };
    fs.writeFileSync(path.join(output, 'live-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ status: report.status, first_voice_sec: report.first_voice_sec, completed_sec: report.completed_sec, first_voice_status: report.first_voice_status }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
