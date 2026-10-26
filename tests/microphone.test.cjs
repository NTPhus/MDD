'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { MicrophoneManager, normalizeDevices, rankDevices, STORAGE_KEY, AUDIO } = require('../demo/microphone.js');

function events(extra = {}) {
  const listeners = new Map();
  return { ...extra, addEventListener(name, fn) { if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name).add(fn); },
    removeEventListener(name, fn) { listeners.get(name)?.delete(fn); },
    emit(name, event = {}) { for (const fn of listeners.get(name) || []) fn(event); }, listeners };
}
function row(deviceId, label, groupId = deviceId) { return { kind: 'audioinput', deviceId, label, groupId }; }
function stream(deviceId, label = '', groupId = deviceId, count = 1) {
  const tracks = Array.from({ length: count }, () => ({ label, stops: 0, stop() { this.stops++; }, getSettings() { return { deviceId, groupId }; } }));
  return { tracks, getTracks: () => tracks, getAudioTracks: () => tracks };
}
function named(name) { const error = new Error('test-only browser failure'); error.name = name; return error; }
function harness({ rows = [row('internal', 'Built-in Microphone'), row('usb', 'USB Microphone')], getUserMedia, storage, document } = {}) {
  let currentRows = rows;
  const calls = [], enumeration = [], target = events();
  const values = new Map();
  const store = storage || { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const media = events({ async enumerateDevices() { enumeration.push(true); return typeof currentRows === 'function' ? currentRows() : currentRows; },
    async getUserMedia(options) { calls.push(options); return getUserMedia ? getUserMedia(options, calls.length) : stream(options.audio.deviceId?.exact || 'internal'); } });
  const manager = new MicrophoneManager({ mediaDevices: media, storage: store, eventTarget: target, document });
  return { manager, media, target, store, values, calls, enumeration, setRows(value) { currentRows = value; } };
}

test('construction is private and starts neither enumeration nor capture', () => {
  const h = harness();
  assert.equal(h.manager.getState().preference, 'auto');
  assert.equal(h.enumeration.length, 0); assert.equal(h.calls.length, 0);
  h.manager.destroy();
});

test('deduplicate system aliases only when a real nonempty physical group matches', () => {
  const devices = normalizeDevices([
    row('default', 'Default - USB', 'group'), row('communications', 'Communications - USB', 'group'),
    row('one', 'USB', 'group'), row('two', 'USB', 'group'), row('one', 'Duplicate', 'group'),
    row('blank-a', 'Same name', ''), row('blank-b', 'Same name', ''),
    { kind: 'audiooutput', deviceId: 'speaker', label: 'Speaker' },
  ]);
  assert.deepEqual(devices.map(item => item.deviceId), ['one', 'two', 'blank-a', 'blank-b']);
  assert.equal(devices[0].isDefault, true); assert.equal(devices[1].isDefault, true);
  const redacted = normalizeDevices([row('default', '', ''), row('communications', '', ''), row('real', '', '')]);
  assert.equal(redacted.length, 3);
});

test('Auto ranks USB or wired inputs above built-in and lowers Bluetooth and virtual inputs', () => {
  const ordered = rankDevices(normalizeDevices([
    row('bt', 'Bluetooth Headset'), row('virtual', 'Virtual USB loopback'),
    row('internal', 'Built-in Microphone'), row('usb', 'USB Microphone'), row('wired', 'Wired Headset'),
  ])).map(item => item.deviceId);
  assert.deepEqual(ordered, ['usb', 'wired', 'internal', 'bt', 'virtual']);
});

test('remaining Auto inputs prefer physical OS default before an unclassified built-in mic', () => {
  const ordered = rankDevices(normalizeDevices([row('internal', 'Built-in Microphone'), row('default', 'Default - Interface', 'interface'),
    row('interface', 'Audio Interface', 'interface')])).map(item => item.deviceId);
  assert.deepEqual(ordered, ['interface', 'internal']);
});

test('explicit permission probe refreshes labels while live and stops every probe track', async () => {
  const probe = stream('internal', '', 'internal', 2);
  const h = harness({ rows: () => { assert.equal(probe.tracks[0].stops, 0); return [row('internal', 'Built-in Microphone')]; }, getUserMedia: () => probe });
  await h.manager.requestPermission();
  assert.equal(h.calls.length, 1); assert.equal(h.enumeration.length, 1);
  assert.equal(h.manager.getState().permission, 'granted');
  assert.ok(probe.tracks.every(track => track.stops === 1)); h.manager.destroy();
});

test('failed probe enumeration still releases all microphone tracks', async () => {
  const probe = stream('internal', '', '', 2);
  const h = harness({ rows: () => { throw new Error('enumeration failed'); }, getUserMedia: () => probe });
  await assert.rejects(h.manager.requestPermission(), /enumeration failed/);
  assert.ok(probe.tracks.every(track => track.stops === 1)); h.manager.destroy();
});

test('first permission in Auto can select external mic before recorder starts, stopping initial stream', async () => {
  let allowed = false;
  const initial = stream('internal', 'Built-in Microphone'), chosen = stream('usb', 'USB Microphone');
  const h = harness({ rows: () => allowed ? [row('internal', 'Built-in Microphone'), row('usb', 'USB Microphone')] : [row('', '', '')],
    getUserMedia(options, count) { allowed = true; if (count === 1) return initial; assert.equal(initial.tracks[0].stops, 1); return chosen; } });
  const result = await h.manager.getStream({ audio: { channelCount: 2, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  assert.equal(result, chosen); assert.equal(h.calls.length, 2);
  assert.equal(h.calls[0].audio.deviceId, undefined); assert.deepEqual(h.calls[1].audio.deviceId, { exact: 'usb' });
  for (const call of h.calls) { for (const [key, value] of Object.entries(AUDIO)) assert.equal(call.audio[key], value); assert.equal(call.video, false); }
  assert.equal(chosen.tracks[0].stops, 0); chosen.tracks[0].stop(); h.manager.destroy();
});

test('Auto reuses the first stream when its actual device already matches the chosen mic', async () => {
  let allowed = false;
  const selected = stream('usb', 'USB Microphone');
  const h = harness({ rows: () => allowed ? [row('usb', 'USB Microphone')] : [], getUserMedia() { allowed = true; return selected; } });
  assert.equal(await h.manager.getStream(), selected);
  assert.equal(h.calls.length, 1); assert.equal(selected.tracks[0].stops, 0);
  selected.tracks[0].stop(); h.manager.destroy();
});

test('distinct physical device IDs sharing one interface group still trigger the correct Auto swap', async () => {
  let allowed = false;
  const initial = stream('internal', 'Built-in Microphone', 'shared'), usb = stream('usb', 'USB Microphone', 'shared');
  const h = harness({ rows: () => allowed ? [row('internal', 'Built-in Microphone', 'shared'), row('usb', 'USB Microphone', 'shared')] : [],
    getUserMedia(options, count) { allowed = true; return count === 1 ? initial : usb; } });
  assert.equal(await h.manager.getStream(), usb); assert.equal(h.calls.length, 2);
  assert.equal(initial.tracks[0].stops, 1); assert.deepEqual(h.calls[1].audio.deviceId, { exact: 'usb' });
  usb.tracks[0].stop(); h.manager.destroy();
});

test('concurrent explicit permission clicks share one pending probe and stop it once', async () => {
  let finish;
  const probe = stream('internal', 'Built-in Microphone', 'internal', 2);
  const h = harness({ getUserMedia: () => new Promise(resolve => { finish = resolve; }) });
  const first = h.manager.requestPermission(), second = h.manager.requestPermission();
  assert.equal(first, second); assert.equal(h.calls.length, 1); assert.equal(h.manager.getState().permissionPending, true);
  finish(probe); await Promise.all([first, second]);
  assert.equal(h.manager.getState().permissionPending, false); assert.ok(probe.tracks.every(track => track.stops === 1));
  h.manager.destroy();
});

test('an exact selected input is not reopened on browsers without track settings', async () => {
  const selected = stream('usb'); delete selected.tracks[0].getSettings;
  const h = harness({ getUserMedia: () => selected });
  assert.equal(await h.manager.getStream(), selected); assert.equal(h.calls.length, 1);
  selected.tracks[0].stop(); h.manager.destroy();
});

test('manual preference persists for both views and is honored above Auto ranking', async () => {
  const h = harness(); await h.manager.refresh(); h.manager.select('internal');
  assert.equal(h.values.get(STORAGE_KEY), 'internal');
  const next = harness({ storage: h.store });
  const selected = await next.manager.getStream();
  assert.deepEqual(next.calls[0].audio.deviceId, { exact: 'internal' });
  assert.equal(next.calls.length, 1); assert.equal(next.manager.getState().preference, 'internal');
  selected.tracks[0].stop(); h.manager.destroy(); next.manager.destroy();
});

test('unplugged manual choice falls back to Auto without forgetting it and recovers on reconnect', async () => {
  const h = harness(); await h.manager.refresh(); h.manager.select('internal');
  h.setRows([row('usb', 'USB Microphone')]); h.media.emit('devicechange'); await h.manager.refresh();
  let state = h.manager.getState();
  assert.equal(state.preference, 'internal'); assert.equal(state.missingPreferredDevice, true); assert.match(state.message, /giữ lại/);
  const selected = await h.manager.getStream(); assert.deepEqual(h.calls[0].audio.deviceId, { exact: 'usb' });
  h.setRows([row('internal', 'Built-in Microphone'), row('usb', 'USB Microphone')]); h.media.emit('devicechange'); await h.manager.refresh();
  state = h.manager.getState(); assert.equal(state.missingPreferredDevice, false); assert.equal(state.resolvedDeviceId, 'internal');
  assert.equal(h.values.get(STORAGE_KEY), 'internal'); selected.tracks[0].stop(); h.manager.destroy();
});

for (const name of ['NotFoundError', 'OverconstrainedError']) test(`stale manual ${name} falls back once and keeps its saved preference`, async () => {
  const h = harness({ getUserMedia(options, count) { if (count === 1) throw named(name); return stream('usb'); } });
  await h.manager.refresh(); h.manager.select('internal');
  const selected = await h.manager.getStream();
  assert.equal(h.calls.length, 2); assert.deepEqual(h.calls[0].audio.deviceId, { exact: 'internal' });
  assert.deepEqual(h.calls[1].audio.deviceId, { exact: 'usb' });
  assert.equal(h.manager.getState().preference, 'internal'); assert.match(h.manager.getState().message, /giữ lại/);
  selected.tracks[0].stop(); h.manager.destroy();
});

test('a redacted stale manual choice unlocks labels then uses the same external Auto preference', async () => {
  let allowed = false;
  const initial = stream('internal', 'Built-in Microphone'), usb = stream('usb', 'USB Microphone');
  const h = harness({ rows: () => allowed ? [row('internal', 'Built-in Microphone'), row('usb', 'USB Microphone')] : [],
    getUserMedia(options, count) {
      if (count === 1) throw named('NotFoundError');
      allowed = true;
      if (count === 2) return initial;
      assert.equal(initial.tracks[0].stops, 1); return usb;
    } });
  h.manager.select('old-device');
  assert.equal(await h.manager.getStream(), usb);
  assert.equal(h.calls.length, 3); assert.equal(h.calls[0].audio.deviceId.exact, 'old-device');
  assert.equal(h.calls[2].audio.deviceId.exact, 'usb'); assert.equal(h.manager.getState().preference, 'old-device');
  usb.tracks[0].stop(); h.manager.destroy();
});

for (const name of ['NotAllowedError', 'NotReadableError']) test(`${name} never causes a second permission or capture attempt`, async () => {
  const h = harness({ getUserMedia() { throw named(name); } });
  await h.manager.refresh(); h.manager.select('internal');
  await assert.rejects(h.manager.getStream(), error => error.name === name);
  assert.equal(h.calls.length, 1);
  if (name === 'NotAllowedError') assert.equal(h.manager.getState().permission, 'denied');
  h.manager.destroy();
});

test('fallback failure is not followed by another device or permission loop', async () => {
  const h = harness({ getUserMedia() { throw named('NotFoundError'); } });
  await h.manager.refresh(); h.manager.select('internal');
  await assert.rejects(h.manager.getStream(), error => error.name === 'NotFoundError');
  assert.equal(h.calls.length, 2); h.manager.destroy();
});

test('devicechange updates the list without opening or switching any active mic', async () => {
  const h = harness(); const selected = await h.manager.getStream(); const before = h.calls.length;
  h.setRows([row('internal', 'Built-in Microphone')]); h.media.emit('devicechange'); await h.manager.refresh();
  assert.equal(h.calls.length, before); assert.equal(selected.tracks[0].stops, 0);
  selected.tracks[0].stop(); h.manager.destroy();
  assert.equal(h.media.listeners.get('devicechange').size, 0);
});

test('a manager closed during pending capture immediately stops the late stream', async () => {
  let finish, started;
  const capturing = new Promise(resolve => { started = resolve; });
  const late = stream('usb');
  const h = harness({ getUserMedia() { started(); return new Promise(resolve => { finish = resolve; }); } });
  const promise = h.manager.getStream(); await capturing; h.manager.destroy(); finish(late);
  await assert.rejects(promise, error => error.name === 'AbortError');
  assert.equal(late.tracks[0].stops, 1);
});

test('returned recording streams remain owned by their caller', async () => {
  const h = harness(); const selected = await h.manager.getStream(); h.manager.destroy();
  assert.equal(selected.tracks[0].stops, 0); selected.tracks[0].stop();
});

test('hidden documents enumerate only after visibility returns and never request permission', async () => {
  const document = events({ visibilityState: 'hidden' });
  const h = harness({ document });
  await h.manager.refresh(); assert.equal(h.enumeration.length, 0);
  document.visibilityState = 'visible'; document.emit('visibilitychange'); await h.manager.refresh();
  assert.equal(h.enumeration.length, 1); assert.equal(h.calls.length, 0); h.manager.destroy();
});

test('shared subscriptions and another tab update preference without capture', async () => {
  const h = harness(), first = [], second = [];
  const off = h.manager.subscribe(value => first.push(value.preference)); h.manager.subscribe(value => second.push(value.preference));
  h.manager.select('internal'); off(); h.target.emit('storage', { key: STORAGE_KEY, newValue: 'usb' });
  assert.deepEqual(first, ['auto', 'internal']); assert.deepEqual(second, ['auto', 'internal', 'usb']);
  assert.equal(h.calls.length, 0); h.manager.destroy();
});

test('storage access denial keeps Auto and recording usable', async () => {
  const h = harness({ storage: { getItem() { throw named('SecurityError'); }, setItem() { throw named('SecurityError'); } } });
  assert.equal(h.manager.getState().preference, 'auto'); await h.manager.refresh(); h.manager.select('usb');
  const selected = await h.manager.getStream(); assert.equal(h.calls.length, 1);
  selected.tracks[0].stop(); h.manager.destroy();
});
