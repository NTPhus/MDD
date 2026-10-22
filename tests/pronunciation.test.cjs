'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const { toIPA } = require('../demo/pronunciation.js');

test('canonical pronunciations distinguish word stress, schwa, and affricates', () => {
  assert.equal(toIPA(['AE1', 'D', 'AH0', 'D']), 'ˈædəd');
  assert.equal(toIPA(['AE1', 'D', 'IH0', 'D']), 'ˈædɪd');
  assert.equal(toIPA(['JH', 'EH1', 'N', 'Y', 'UW0', 'AH0', 'N']), 'ˈdʒɛnjuən');
  assert.equal(toIPA(['W', 'IH1', 'DH']), 'wɪð');
});

test('stress appears before the syllable onset, including consonant clusters', () => {
  assert.equal(toIPA(['AH0', 'B', 'AW1', 'T']), 'əˈbaʊt');
  assert.equal(toIPA(['B', 'AH0', 'N', 'AE1', 'N', 'AH0']), 'bəˈnænə');
  assert.equal(toIPA(['AH2', 'N', 'D', 'ER0', 'S', 'T', 'AE1', 'N', 'D']), 'ˌʌndɚˈstænd');
  assert.equal(toIPA(['IH0', 'K', 'S', 'P', 'R', 'EH1', 'S']), 'ɪkˈspɹɛs');
});

test('r-colored vowels and diphthongs use broad US-English symbols', () => {
  assert.equal(toIPA(['B', 'ER1', 'D']), 'bɝd');
  assert.equal(toIPA(['T', 'IY1', 'CH', 'ER0']), 'ˈtitʃɚ');
  assert.equal(toIPA(['K', 'OW1', 'B', 'OY2']), 'ˈkoʊˌbɔɪ');
  assert.equal(toIPA(['SH', 'AY1']), 'ʃaɪ');
  assert.equal(toIPA(['TH', 'IH1', 'NG']), 'θɪŋ');
});

test('legacy unstressed phone arrays do not acquire invented stress or reductions', () => {
  const phones = ['JH', 'EH', 'N', 'Y', 'UW', 'AH', 'N'];
  assert.equal(toIPA(phones), 'dʒɛnjuʌn');
  assert.deepEqual(phones, ['JH', 'EH', 'N', 'Y', 'UW', 'AH', 'N']);
  assert.equal(toIPA(['AH0']), 'ə');
  assert.equal(toIPA(['ER0']), 'ɚ');
});

test('unknown, malformed, and non-lexical sequences produce no guessed pronunciation', () => {
  for (const value of [null, undefined, '', 'W IH1 DH', [], {}, ['UNKNOWN'], ['AE3'],
    ['AE10'], ['D1'], ['IH-1'], ['ih1'], [' IH1'], ['IH1 '], [1], [null], ['S'],
    ['IH1', 'constructor'], ['IH1', '__proto__'], Array(257).fill('IH1')]) {
    assert.equal(toIPA(value), null, JSON.stringify(value));
  }
});

test('plain browser script exposes the same small API without CommonJS', () => {
  const context = vm.createContext({});
  vm.runInContext(fs.readFileSync(require.resolve('../demo/pronunciation.js'), 'utf8'), context);
  assert.equal(context.MDDPronunciation.toIPA(['W', 'IH1', 'DH']), 'wɪð');
  assert.deepEqual(Object.keys(context.MDDPronunciation), ['toIPA']);
});
