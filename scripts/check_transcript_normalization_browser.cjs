'use strict';
// UI/history binding only. Mocked ASR/Qwen; no external requests or paid voice.
const { chromium } = require('../.tools/ui-qa/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const base = process.env.MDD_QA_URL || 'http://127.0.0.1:8765';
  const output = path.resolve('data/generated/transcript-normalization-v1'); fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
  const errors = [], requests = [], checks = [];
  page.on('pageerror', e => errors.push(e.message));
  const config = await (await page.request.get(base + '/api/chat/config')).json();
  const raw = 'i dont work as a soft ware engineer', normalized = "I don't work as a software engineer.";
  let job, serial = 0, paidCalls = 0;
  await page.route('**/api/chat/**', route => {
    const url = new URL(route.request().url()).pathname;
    if (url === '/api/chat/config') return route.fulfill({ json: config });
    if (url === '/api/chat/speech') {
      const request = route.request().postDataJSON(); delete request.audio_base64; requests.push(request);
      job = { id: String(++serial).padStart(32, '0'), status: 'ready', text: 'What kind of work do you do instead?',
        metadata: { reference_revision: 0, conversation_source: 'asr_normalized' },
        voice: { status: 'not_requested' }, speech: { transcript: raw, reference_text: raw, reference_source: 'asr',
          conversation_text: normalized, conversation_source: 'asr_normalized',
          normalization: { status: 'accepted', text: normalized, lexical_preservation: true, reference_confirmed: false },
          comparison: { words: [], events: [] }, target_words: [], missing_words: [],
          rewrite: { status: 'not_needed' }, correction: { status: 'not_needed' } } };
      return route.fulfill({ json: job });
    }
    if (url === '/api/chat/voice') paidCalls++;
    return route.fulfill({ status: 500, json: { error: 'Unexpected API blocked in QA' } });
  });
  try {
    await page.goto(base + '/#chat'); await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.locator('#chat-free-talk').click();
    await page.locator('#chat-settings > summary').click();
    await page.locator('#chat-auto-read').uncheck(); await page.keyboard.press('Escape');
    await page.evaluate(() => window.MDDChat.sendAudio(new ArrayBuffer(32000), 'Mocked audio'));
    await page.waitForFunction(() => !window.MDDChat.state.busy);
    assert.equal(await page.locator('.user .chat-message-text').last().textContent(), normalized);
    const turn = await page.evaluate(() => window.MDDChat.getTurns().at(-1));
    assert.equal(turn.speech.transcript, raw); assert.equal(turn.speech.reference_text, raw);
    assert.equal(turn.speech.reference_source, 'asr');
    checks.push('The user bubble shows normalized spelling while raw ASR and pronunciation reference remain unchanged.');
    await page.locator('.chat-turn-feedback > summary').first().click();
    assert.equal(await page.locator('.chat-sentence-text').first().textContent(), raw);
    await page.screenshot({ path: path.join(output, 'browser-normalization.png'), fullPage: true });
    await page.evaluate(() => window.MDDChat.sendAudio(new ArrayBuffer(32000), 'Second mocked audio'));
    assert.equal(requests[1].history[0].content, normalized);
    checks.push('The next conversation request receives the normalized previous user turn, retaining negation and meaning.');
    await page.evaluate(() => {
      const chat = window.MDDChat, user = chat.getTurns()[0];
      user.confirmedText = 'I work as a doctor.'; user.confirmedSource = 'user_confirmed';
      chat.onViewChange('chat', () => true);
    });
    assert.equal(await page.locator('.user .chat-message-text').first().textContent(), 'I work as a doctor.');
    checks.push('A user-confirmed transcript overrides the automatic formatting view.');
    assert.equal(paidCalls, 0); assert.deepEqual(errors, []);
    const report = { status: 'passed', checks, errors, paid_calls: paidCalls, inference: 'mocked UI regression' };
    fs.writeFileSync(path.join(output, 'browser-summary.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { await browser.close(); }
}
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
