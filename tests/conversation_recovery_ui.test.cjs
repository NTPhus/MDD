'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { conversationInterpretation, conversationUserText, conversationUserNote, updateUserSpeech,
  pronunciationReferenceText } = require('../demo/chat.js');

const raw = 'Placoll tellar askd her to bring the things with her fromer stor';
const soundTarget = 'Pla coll tel lar asked her to bring the things with her fromer torr';
const recovered = 'Please call Stella. Ask her to bring these things with her from the store.';
const speechResult = (overrides = {}) => ({
  transcript: raw, reference_text: soundTarget, reference_source: 'bert_inferred',
  pronunciation_reference_ready: true,
  interpretation: { status: 'accepted', text: soundTarget, requires_confirmation: true },
  conversation_source: 'asr_interpreted', conversation_text: recovered,
  conversation_recovery: { status: 'accepted', original: raw, text: recovered,
    reference_confirmed: false, requires_confirmation: true }, ...overrides,
});

test('the recovered dialogue drives the displayed and history text while acoustic evidence stays intact', () => {
  const speech = speechResult(), user = { content: 'Audio recording' };
  updateUserSpeech(user, speech);
  assert.equal(conversationInterpretation(speech), speech.conversation_recovery);
  assert.equal(conversationUserText(user), recovered);
  assert.equal(conversationUserNote(user), 'Bản chép lời được khôi phục');
  assert.equal(user.content, raw, 'The original recognition remains available for inspection.');
  assert.equal(pronunciationReferenceText(speech), soundTarget, 'Word checks and sentence audio keep their own target.');
  assert.equal(user.confirmedSource, undefined, 'Recovery does not become user confirmation.');
  assert.equal(user.interpretedRequiresConfirmation, true);
});

test('an unchanged recovered sentence can be selected without a misleading recovery label', () => {
  const speech = speechResult({ conversation_text: raw,
    conversation_recovery: { status: 'unchanged', original: raw, text: raw, reference_confirmed: false } });
  const user = {};
  updateUserSpeech(user, speech);
  assert.equal(conversationUserText(user), raw);
  assert.equal(conversationInterpretation(speech), speech.conversation_recovery);
  assert.equal(conversationUserNote(user), '');
});

test('a recovery from another recording or an unselected candidate never replaces the actual dialogue text', () => {
  const base = speechResult();
  for (const invalid of [
    { ...base.conversation_recovery, original: 'An earlier recording.' },
    { ...base.conversation_recovery, text: 'Unselected different words.' },
    { ...base.conversation_recovery, reference_confirmed: true },
    { ...base.conversation_recovery, text: 'x'.repeat(1001) },
    { ...base.conversation_recovery, status: 'ambiguous' },
    { ...base.conversation_recovery, status: 'unavailable' },
  ]) {
    const speech = speechResult({ conversation_recovery: invalid }), user = {};
    updateUserSpeech(user, speech);
    assert.equal(conversationInterpretation(speech), null);
    assert.equal(conversationUserText(user), raw);
    assert.equal(conversationUserNote(user), '');
  }
});

test('pending recovery clears stale text until its final backend selection arrives', () => {
  const user = {};
  updateUserSpeech(user, speechResult());
  assert.equal(conversationUserText(user), recovered);
  const pending = speechResult({ pronunciation_reference_ready: false,
    conversation_recovery: { status: 'pending', original: raw } });
  updateUserSpeech(user, pending);
  assert.equal(conversationUserText(user), '');
  assert.equal(user.interpretedText, undefined);
  assert.equal(conversationUserNote(user), 'Đang nhận giọng nói…');
  assert.equal(pronunciationReferenceText(pending), '');
  updateUserSpeech(user, speechResult());
  assert.equal(conversationUserText(user), recovered);
});

test('a pending dialogue recovery cannot display an interim pronunciation inference', () => {
  const user = {};
  updateUserSpeech(user, speechResult({ conversation_recovery: { status: 'pending', original: raw } }));
  assert.equal(conversationUserText(user), '');
  assert.equal(conversationUserNote(user), 'Đang nhận giọng nói…');
});

test('a rejected sentence recovery can use the BERT fallback only when the backend selects it', () => {
  const speech = speechResult({ conversation_text: soundTarget,
    conversation_recovery: { status: 'ambiguous', original: raw, text: recovered, reference_confirmed: false } });
  const user = {};
  updateUserSpeech(user, speech);
  assert.equal(conversationInterpretation(speech), speech.interpretation);
  assert.equal(conversationUserText(user), soundTarget);
  const unselected = speechResult({ conversation_source: 'asr', conversation_text: raw,
    conversation_recovery: speech.conversation_recovery });
  updateUserSpeech(user, unselected);
  assert.equal(conversationInterpretation(unselected), null);
  assert.equal(conversationUserText(user), raw);
});

test('explicit confirmation stays ahead of a pending or accepted automatic recovery', () => {
  const confirmed = 'Please call my sister.';
  const user = { confirmedSource: 'user_confirmed', confirmedText: confirmed };
  for (const speech of [speechResult(), speechResult({ pronunciation_reference_ready: false,
    conversation_recovery: { status: 'pending', original: raw } })]) {
    updateUserSpeech(user, speech);
    assert.equal(conversationUserText(user), confirmed);
    assert.equal(conversationUserNote(user), '');
  }
});

test('reading and confirmed conversation sources do not inherit an unrelated recovery draft', () => {
  for (const source of ['reading_prompt', 'user_confirmed']) {
    const speech = speechResult({ reference_source: source, conversation_source: source, conversation_text: raw });
    assert.equal(conversationInterpretation(speech), null);
  }
});

let chromium;
try { ({ chromium } = require('../.tools/ui-qa/node_modules/playwright')); } catch {}
const fs = require('node:fs'), path = require('node:path');
const chromePath = process.env.MDD_CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

test('the rendered conversation, editable sound target, raw transcript and next-turn history stay distinct', {
  skip: !chromium || !fs.existsSync(chromePath), timeout: 20000,
}, async () => {
  const browser = await chromium.launch({ executablePath: chromePath, headless: true });
  const page = await browser.newPage(), requests = [], errors = [];
  const demo = path.join(__dirname, '..', 'demo');
  const html = fs.readFileSync(path.join(demo, 'index.html'), 'utf8')
    .replace(/<script defer src="\/(?:app|pronunciation)\.js"><\/script>/g, '');
  const config = { speech_ready: true, default_voice_provider: 'browser', models: [{ id: 'qwen-trained', ready: true }],
    practice_prompts: [], topics: [], english_voices: [], kokoro_voices: [] };
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    sessionStorage.setItem('am-chat-selection-v1', JSON.stringify({ autoRead: false, voiceProvider: 'browser' }));
  });
  await page.route('https://mdd.test/**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/') return route.fulfill({ contentType: 'text/html', body: html });
    if (pathname === '/api/chat/config') return route.fulfill({ json: config });
    if (pathname === '/api/chat/speech') {
      requests.push(route.request().postDataJSON());
      return route.fulfill({ json: { id: String(requests.length).repeat(32), status: 'ready',
        text: 'What else should she bring?', metadata: { reference_revision: 0 },
        speech: { ...speechResult(), reply_ready: true }, voice: { status: 'not_requested' } } });
    }
    if (/^\/[\w_]+\.(?:js|css)$/.test(pathname)) return route.fulfill({ path: path.join(demo, pathname.slice(1)) });
    return route.fulfill({ status: 404, body: '' });
  });
  try {
    await page.goto('https://mdd.test/');
    await page.waitForFunction(() => window.MDDChat?.state.config);
    await page.evaluate(async () => {
      document.getElementById('chat-view').hidden = false;
      window.MDDChat.onViewChange('chat', () => true);
      await window.MDDChat.sendAudio(new ArrayBuffer(64), 'test.wav');
    });
    assert.equal(await page.locator('.chat-message.user .chat-message-text').innerText(), recovered);
    assert.equal(await page.locator('.chat-message.user .chat-message-meta').innerText(), 'Bản chép lời được khôi phục');
    await page.locator('.chat-turn-feedback > summary').click();
    assert.equal(await page.locator('.chat-reference-input').inputValue(), soundTarget);
    assert.match(await page.locator('.chat-turn-review .chat-review-label').innerText(), /đối chiếu phát âm/);
    assert.equal(await page.locator('.chat-original-transcript').evaluate(element => element.open), false);
    await page.locator('.chat-original-transcript > summary').click();
    assert.equal(await page.locator('.chat-original-transcript .chat-target-text').innerText(), raw);
    await page.evaluate(async () => { await window.MDDChat.sendAudio(new ArrayBuffer(64), 'next.wav'); });
    assert.deepEqual(requests[1].history, [
      { role: 'user', content: recovered }, { role: 'assistant', content: 'What else should she bring?' },
    ]);
    assert.deepEqual(errors, []);
  } finally { await page.close(); await browser.close(); }
});
