'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { SpeechWordTracker, wordRange } = require('../demo/speech_highlight.js');

function harness(text, options = {}) {
  let time = 0, nextId = 0;
  const tasks = new Map(), frames = [], cancelled = [];
  const tracker = new SpeechWordTracker(text, {
    onUpdate(range, mode) { frames.push({ range, mode }); }, now: () => time,
    schedule(fn, delay) { tasks.set(++nextId, { fn, at: time + delay }); return nextId; },
    unschedule(id) { cancelled.push(id); tasks.delete(id); }, ...options,
  });
  function advance(amount) {
    const end = time + amount;
    for (let count = 0; count < 1000; count++) {
      const entry = [...tasks.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (!entry || entry[1].at > end) { time = end; return; }
      time = entry[1].at; tasks.delete(entry[0]); entry[1].fn();
    }
    assert.fail('Unbounded timer queue');
  }
  return { tracker, frames, tasks, cancelled, advance, last: () => frames.at(-1) };
}

test('word ranges retain full contractions, punctuation and UTF16 offsets', () => {
  const text = '  I\'d order café ☕️ today.';
  assert.deepEqual(wordRange(text, 3), { start: 2, end: 5 });
  assert.deepEqual(wordRange(text, 4), { start: 2, end: 5 });
  const emoji = text.indexOf('☕');
  assert.deepEqual(wordRange(text, emoji + 1), { start: emoji, end: emoji + 2 });
  const astral = 'hello 😀 there';
  assert.deepEqual(wordRange(astral, 7), { start: 6, end: 8 });
  assert.deepEqual(wordRange(text, text.length - 1), { start: text.indexOf('today'), end: text.length });
});

test('invalid and whitespace offsets never manufacture a word', () => {
  for (const index of [null, undefined, NaN, Infinity, -1, .5, '0', 3, 7]) assert.equal(wordRange('one two', index), null);
  assert.equal(wordRange(null, 0), null);
  const h = harness('one two');
  assert.equal(h.tracker.boundary(0), false);
  h.tracker.start();
  for (const index of [null, NaN, -1, .5, 3, 7]) assert.equal(h.tracker.boundary(index), false);
  assert.equal(h.last().mode, 'estimated'); assert.equal(h.tasks.size, 1);
  h.tracker.dispose();
});

test('the estimated clock starts only with actual playback and terminates after the last word', () => {
  const h = harness('one two six');
  h.advance(10000); assert.equal(h.frames.length, 0); assert.equal(h.tasks.size, 0);
  assert.equal(h.tracker.start(), true);
  assert.deepEqual(h.last(), { range: { start: 0, end: 3 }, mode: 'estimated' });
  h.advance(400);
  assert.deepEqual(h.last(), { range: { start: 4, end: 7 }, mode: 'estimated' });
  h.advance(10000);
  assert.deepEqual(h.last(), { range: { start: 8, end: 11 }, mode: 'estimated' });
  assert.equal(h.tasks.size, 0);
  h.tracker.end(); assert.deepEqual(h.last(), { range: null, mode: null });
});

test('a real native boundary immediately replaces estimates and permanently cancels their timer', () => {
  const h = harness('one two six'); h.tracker.start();
  const stale = [...h.tasks.values()][0].fn;
  h.advance(400); assert.deepEqual(h.last().range, { start: 4, end: 7 });
  assert.equal(h.tracker.boundary(1), true);
  assert.deepEqual(h.last(), { range: { start: 0, end: 3 }, mode: 'boundary' });
  assert.equal(h.tasks.size, 0);
  stale(); h.advance(10000); assert.deepEqual(h.last().range, { start: 0, end: 3 });
  assert.equal(h.tracker.audioTime(9, 10), false);
  h.tracker.boundary(9); assert.deepEqual(h.last().range, { start: 8, end: 11 });
  h.tracker.end(); assert.deepEqual(h.last().range, null);
});

test('audio progress remains estimated, handles seeking and replaces the wall clock', () => {
  const h = harness('one two six'); h.tracker.start();
  assert.equal(h.tracker.audioTime(5, 9), true);
  assert.deepEqual(h.last(), { range: { start: 4, end: 7 }, mode: 'estimated' });
  assert.equal(h.tasks.size, 0);
  h.advance(10000); assert.deepEqual(h.last().range, { start: 4, end: 7 });
  h.tracker.audioTime(8, 9); assert.deepEqual(h.last().range, { start: 8, end: 11 });
  h.tracker.audioTime(0, 9); assert.deepEqual(h.last().range, { start: 0, end: 3 });
  h.tracker.audioTime(900, 9); assert.deepEqual(h.last().range, { start: 8, end: 11 });
  h.tracker.end(); assert.equal(h.tracker.audioTime(1, 9), false);
});

test('audio timing rejects missing, nonfinite, negative or zero-duration values', () => {
  const h = harness('one two six'); h.tracker.start();
  for (const [time, duration] of [[null, 9], [NaN, 9], [Infinity, 9], [-1, 9], [1, null], [1, 0], [1, -1], [1, Infinity], ['1', 9]]) {
    assert.equal(h.tracker.audioTime(time, duration), false);
  }
  assert.equal(h.frames.length, 1); assert.equal(h.tasks.size, 1);
  h.tracker.dispose();
});

test('stop clears highlight, ignores late callbacks and permits explicit new playback', () => {
  const h = harness('one two'); h.tracker.start();
  const stale = [...h.tasks.values()][0].fn;
  h.tracker.end(); const count = h.frames.length;
  stale(); h.tracker.boundary(4); h.tracker.audioTime(8, 9); h.advance(10000);
  assert.equal(h.frames.length, count); assert.equal(h.tasks.size, 0);
  assert.equal(h.tracker.start(), true); assert.deepEqual(h.last().range, { start: 0, end: 3 });
  h.tracker.dispose();
});

test('dispose permanently ignores queued timers, native events and attempts to replay', () => {
  const h = harness('one two'); h.tracker.start();
  const stale = [...h.tasks.values()][0].fn;
  h.tracker.dispose(); const count = h.frames.length;
  stale(); h.tracker.start(); h.tracker.boundary(4); h.tracker.audioTime(8, 9); h.tracker.end(); h.tracker.dispose();
  h.advance(10000);
  assert.equal(h.frames.length, count); assert.equal(h.tasks.size, 0);
  assert.deepEqual(h.last(), { range: null, mode: null });
});

test('a cancelled callback from prior playback cannot lose the new playback timer', () => {
  const h = harness('one two'); h.tracker.start();
  const stale = [...h.tasks.values()][0].fn;
  h.tracker.end(); h.tracker.start(); const current = h.tracker.timer;
  stale();
  assert.equal(h.tracker.timer, current); assert.equal(h.tasks.size, 1);
  h.tracker.dispose(); assert.equal(h.tasks.size, 0);
});

test('repeated start and identical boundaries do not duplicate highlights or timers', () => {
  const h = harness('one two'); h.tracker.start();
  assert.equal(h.tracker.start(), false); assert.equal(h.tasks.size, 1);
  h.tracker.boundary(0); h.tracker.boundary(1);
  assert.equal(h.frames.length, 2); assert.equal(h.tasks.size, 0);
  h.tracker.dispose();
});

test('empty text never starts and timing estimates respect the selected speech rate', () => {
  const empty = harness('  \n'); assert.equal(empty.tracker.start(), false); assert.equal(empty.tasks.size, 0);
  const slow = harness('one two', { rate: .5 }), fast = harness('one two', { rate: 2 });
  slow.tracker.start(); fast.tracker.start(); slow.advance(300); fast.advance(300);
  assert.deepEqual(slow.last().range, { start: 0, end: 3 });
  assert.deepEqual(fast.last().range, { start: 4, end: 7 });
  slow.tracker.dispose(); fast.tracker.dispose();
});
