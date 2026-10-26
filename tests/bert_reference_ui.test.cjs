'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { conversationInterpretation, conversationUserText, updateUserSpeech,
  pronunciationReferenceReady, pronunciationReferenceText } = require('../demo/chat.js');

const raw = 'It was the divicote true getesir';
const recovered = 'It was difficult to get there.';
const bertSpeech = (overrides = {}) => ({
  transcript: raw, reference_text: recovered, reference_source: 'bert_inferred',
  pronunciation_reference_ready: true, conversation_source: 'asr_interpreted', conversation_text: recovered,
  interpretation: { status: 'accepted', text: recovered, requires_confirmation: true }, ...overrides,
});

test('BERT reference drives the user bubble and sound target without confirming itself', () => {
  const speech = bertSpeech(), user = { content: 'Audio recording' };
  updateUserSpeech(user, speech);
  assert.equal(conversationUserText(user), recovered);
  assert.equal(pronunciationReferenceText(speech), recovered);
  assert.equal(user.content, raw, 'Raw ASR remains available to internal provenance.');
  assert.equal(user.confirmedSource, undefined);
});

test('pending BERT recovery exposes neither raw text nor a sentence voice target', () => {
  const speech = bertSpeech({ pronunciation_reference_ready: false }), user = { content: 'Audio recording' };
  updateUserSpeech(user, speech);
  assert.equal(pronunciationReferenceReady(speech), false);
  assert.equal(conversationUserText(user), '');
  assert.equal(pronunciationReferenceText(speech), '');
  assert.equal(conversationInterpretation(speech), null);
});

test('reassessed BERT reference can be displayed before dialogue metadata is attached', () => {
  const speech = bertSpeech({ conversation_source: undefined, conversation_text: undefined });
  const user = { content: 'Audio recording' };
  updateUserSpeech(user, speech);
  assert.equal(conversationUserText(user), recovered);
  assert.equal(conversationInterpretation(speech), speech.interpretation);
});

test('a mismatched or rejected inference cannot replace the selected reference in the bubble', () => {
  for (const interpretation of [
    { status: 'accepted', text: 'Unexpected different words.' },
    { status: 'ambiguous', text: recovered },
    { status: 'unavailable', text: recovered },
  ]) {
    const speech = bertSpeech({ reference_source: 'asr', reference_text: raw, conversation_source: 'asr', interpretation });
    const user = { content: 'Audio recording' };
    updateUserSpeech(user, speech);
    assert.equal(conversationInterpretation(speech), null);
    assert.equal(conversationUserText(user), raw);
    assert.equal(pronunciationReferenceText(speech), raw);
  }
});

test('explicit user correction stays ahead of BERT and selects its own sound and voice target', () => {
  const confirmed = 'It was easy to get there.';
  const speech = bertSpeech({ reference_source: 'user_confirmed', reference_text: confirmed,
    conversation_source: 'user_confirmed', conversation_text: confirmed });
  const user = { content: raw, confirmedSource: 'user_confirmed', confirmedText: confirmed };
  updateUserSpeech(user, speech);
  assert.equal(conversationUserText(user), confirmed);
  assert.equal(pronunciationReferenceText(speech), confirmed);
});

test('reading mode retains the actual spoken bubble and the supplied reading reference', () => {
  const speech = { transcript: 'I went home', reference_text: 'I walked home.', reference_source: 'reading_prompt',
    pronunciation_reference_ready: true };
  const user = { content: 'Audio recording' };
  updateUserSpeech(user, speech);
  assert.equal(conversationUserText(user), speech.transcript);
  assert.equal(pronunciationReferenceText(speech), speech.reference_text);
});

test('saved legacy sessions keep a usable reference while absent speech never schedules a voice', () => {
  assert.equal(pronunciationReferenceReady(undefined), false);
  assert.equal(pronunciationReferenceText(undefined), '');
  assert.equal(pronunciationReferenceText({ transcript: 'Legacy spoken sentence.' }), 'Legacy spoken sentence.');
});
