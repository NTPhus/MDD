'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { LiveReply } = require('../demo/chat.js');

function harness(options = {}) {
  const tasks = new Map(), spoken = [], frames = [], errors = []; let next = 0, cancelled = 0;
  const reply = new LiveReply({
    schedule(fn) { tasks.set(++next, fn); return next; }, unschedule(id) { tasks.delete(id); },
    onUpdate(text, range) { frames.push({ text, range }); },
    speak(text, callbacks) { spoken.push({ text, callbacks }); }, cancelSpeech() { cancelled++; },
    onError(error) { errors.push(error); }, ...options,
  });
  const tick = () => { const first = tasks.entries().next().value; if (first) { tasks.delete(first[0]); first[1](); } };
  const flush = () => { for (let i = 0; tasks.size && i < 200; i++) tick(); assert.equal(tasks.size, 0); };
  return { reply, spoken, frames, errors, tasks, tick, flush, get cancelled() { return cancelled; } };
}

test('holds generation, then shows the entire reply immediately without text animation', () => {
  const h = harness({ voice: false });
  h.reply.update('Hello there par'); assert.equal(h.reply.visibleText, '');
  h.flush(); assert.equal(h.reply.visibleText, '');
  h.reply.update('Hello there partner'); h.flush(); assert.equal(h.reply.visibleText, '');
  h.reply.update('Hello there partner', { final: true });
  assert.equal(h.reply.visibleText, 'Hello there partner'); assert.equal(h.spoken.length, 0);
  assert.equal(h.tasks.size, 0);
});
test('the full reply is painted before autoplay starts, including words in later sentences', () => {
  const h = harness(); h.reply.update('Hello there. How ');
  assert.equal(h.reply.final, false); assert.equal(h.spoken.length, 0);
  h.reply.update('Hello there. How are you?', { final: true });
  assert.equal(h.spoken[0].text, 'Hello there.');
  assert.equal(h.reply.visibleText, 'Hello there. How are you?'); h.spoken[0].callbacks.start();
  assert.equal(h.reply.visibleText, 'Hello there. How are you?'); assert.deepEqual(h.reply.range, { start: 0, end: 5 });
});
test('onUpdate publishes complete text before the speak callback can run', () => {
  let painted = '';
  const text = 'One sentence. Another sentence.';
  const h = harness({ onUpdate: value => { painted = value; }, speak: () => assert.equal(painted, text) });
  h.reply.update(text, { final: true });
  assert.equal(painted, text);
});
test('sentences play in order without overlaps or duplicate playback from repeated polls', () => {
  const h = harness(); const text = 'First sentence. Second sentence!';
  h.reply.update(text); h.reply.update(text); h.reply.update(text, { final: true });
  assert.equal(h.spoken.length, 1); h.spoken[0].callbacks.end();
  assert.equal(h.spoken[1].text, 'Second sentence!'); h.spoken[1].callbacks.end();
  h.reply.update(text, { final: true }); assert.equal(h.spoken.length, 2); h.flush();
  assert.equal(h.reply.visibleText, text);
});
test('long replies are entirely visible before shorter speech clauses are queued', () => {
  const h = harness(), words = Array.from({ length: 15 }, (_, i) => `word${i}`);
  h.reply.update(words.join(' ') + ' ');
  assert.equal(h.spoken.length, 0);
  h.reply.update(words.join(' ') + ' ', { final: true });
  assert.equal(h.spoken[0].text, words.slice(0, 12).join(' '));
  assert.equal(h.reply.visibleText, words.join(' ') + ' ');
  h.reply.update(words.join(' ') + ' ', { final: true }); h.spoken[0].callbacks.end();
  assert.equal(h.spoken[1].text, words.slice(12).join(' '));
});
test('word boundaries track offsets across sentences while the entire reply stays visible', () => {
  const h = harness(); h.reply.update('One two. Three four.', { final: true });
  h.spoken[0].callbacks.boundary(4); assert.deepEqual(h.reply.range, { start: 4, end: 8 });
  assert.equal(h.reply.visibleText, 'One two. Three four.'); h.spoken[0].callbacks.end();
  h.spoken[1].callbacks.boundary(6); assert.deepEqual(h.reply.range, { start: 15, end: 20 });
  assert.equal(h.reply.visibleText, 'One two. Three four.');
});
test('device word boundaries only move the highlight and stopping voice retains all text', () => {
  const h = harness(); h.reply.update('One two three.', { final: true });
  h.spoken[0].callbacks.boundary(0); h.flush(); assert.equal(h.reply.visibleText, 'One two three.');
  h.spoken[0].callbacks.boundary(4); h.flush(); assert.equal(h.reply.visibleText, 'One two three.');
  h.reply.mute(); h.flush(); assert.equal(h.reply.visibleText, 'One two three.');
});

test('estimated highlighting never hides words and yields immediately to a real boundary', () => {
  const h = harness(); h.reply.update('One two three.', { final: true });
  h.spoken[0].callbacks.boundary(4, 'estimated'); h.flush();
  assert.equal(h.reply.visibleText, 'One two three.');
  assert.equal(h.reply.rangeMode, 'estimated');
  h.spoken[0].callbacks.boundary(8); h.flush();
  assert.equal(h.reply.visibleText, 'One two three.');
  assert.deepEqual(h.reply.range, { start: 8, end: 14 });
  assert.equal(h.reply.rangeMode, 'boundary');
  h.spoken[0].callbacks.end(); assert.equal(h.reply.rangeMode, null);
});

test('invalid speech offsets cannot highlight text and replay can reveal all actual output without another generation', () => {
  const h = harness(); h.reply.update('One two three.', { final: true });
  for (const value of [-1, NaN, null, .5, 1000]) {
    h.spoken[0].callbacks.boundary(value);
    assert.equal(h.reply.range, null);
  }
  h.reply.mute(); h.reply.revealAll(); h.flush();
  assert.equal(h.reply.visibleText, 'One two three.');
  assert.equal(h.spoken.length, 1);
  assert.equal(h.reply.range, null);
});
test('late coaching never adds optional tips or repeats the learner in the conversation voice', () => {
  const h = harness(); h.reply.update('A reply.', { final: true }); h.spoken[0].callbacks.end();
  h.reply.update('A reply.', { final: true, spokenText: 'A reply. Practise slowly.' });
  assert.equal(h.spoken.length, 1);
  h.reply.update('A reply.', { final: true, spokenText: 'A reply. Practise slowly.' });
  assert.equal(h.spoken.length, 1);
});
test('an unrelated spoken response is never appended as duplicate reply audio', () => {
  const h = harness(); h.reply.update('A reply.', { final: true, spokenText: 'A different reply.' });
  h.spoken[0].callbacks.end(); assert.equal(h.spoken.length, 1);
});
test('stop voice keeps full text but ignores old callbacks and future speech', () => {
  const h = harness(); h.reply.update('First sentence.', { final: true }); const old = h.spoken[0].callbacks;
  h.reply.mute(); old.start(); old.end(); old.error('late');
  h.reply.update('First sentence. Later words.', { final: true, spokenText: 'First sentence. Later words. A tip.' }); h.flush();
  assert.equal(h.spoken.length, 1); assert.equal(h.reply.range, null);
  assert.equal(h.reply.visibleText, 'First sentence. Later words.'); assert.equal(h.errors.length, 0);
});
test('disposing on navigation ignores late polls and never resumes speech', () => {
  const h = harness(); h.reply.update('First sentence. More words.', { final: true }); const old = h.spoken[0].callbacks;
  h.reply.dispose(); const visible = h.reply.visibleText;
  old.start(); old.end(); h.reply.update('First sentence. More words.', { final: true }); h.flush();
  assert.equal(h.reply.visibleText, visible); assert.equal(h.spoken.length, 1);
});
test('speech failure stops automatic audio while text still completes', () => {
  const h = harness(); h.reply.update('A sentence.', { final: true }); h.spoken[0].callbacks.error('not-allowed');
  h.reply.update('A sentence. More words.', { final: true }); h.flush();
  assert.equal(h.errors.length, 1); assert.equal(h.spoken.length, 1); assert.equal(h.reply.visibleText, h.reply.text);
});
test('revised model text cancels stale audio and resets its display', () => {
  const h = harness(); h.reply.update('Wrong reply.', { final: true }); h.flush(); h.reply.update('Correct reply.', { final: true }); h.flush();
  assert.equal(h.reply.visibleText, 'Correct reply.'); assert.equal(h.spoken.length, 1); assert.ok(h.cancelled);
});
test('missing browser TTS never blocks the complete text response', () => {
  const h = harness({ speak: null }); h.reply.update('A complete response', { final: true }); h.flush();
  assert.equal(h.reply.visibleText, 'A complete response'); assert.equal(h.spoken.length, 0);
});

test('read progress accumulates across clauses, completes the whole reply and resets on stop', () => {
  const h = harness(), text = 'One two. Three four.';
  h.reply.update(text, { final: true });
  h.spoken[0].callbacks.boundary(0); assert.equal(h.reply.readUntil, 3);
  h.spoken[0].callbacks.boundary(4); assert.equal(h.reply.readUntil, 8);
  h.reply.update(text, { final: true }); assert.equal(h.reply.readUntil, 8);
  h.spoken[0].callbacks.end(); assert.equal(h.reply.readUntil, 8);
  h.spoken[1].callbacks.boundary(0); assert.equal(h.reply.readUntil, 14);
  h.spoken[1].callbacks.end(); assert.equal(h.reply.readUntil, text.length);
  assert.equal(h.reply.range, null); assert.equal(h.reply.visibleText, text);
  h.reply.mute(); assert.equal(h.reply.readUntil, 0);
  h.spoken[1].callbacks.boundary(6); assert.equal(h.reply.readUntil, 0);
});
