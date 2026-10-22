'use strict';

(function (root) {
  const VOWELS = Object.freeze({
    AA: 'ɑ', AE: 'æ', AH: 'ʌ', AO: 'ɔ', AW: 'aʊ', AY: 'aɪ', EH: 'ɛ',
    ER: 'ɝ', EY: 'eɪ', IH: 'ɪ', IY: 'i', OW: 'oʊ', OY: 'ɔɪ', UH: 'ʊ', UW: 'u',
  });
  const CONSONANTS = Object.freeze({
    B: 'b', CH: 'tʃ', D: 'd', DH: 'ð', F: 'f', G: 'ɡ', HH: 'h', JH: 'dʒ',
    K: 'k', L: 'l', M: 'm', N: 'n', NG: 'ŋ', P: 'p', R: 'ɹ', S: 's',
    SH: 'ʃ', T: 't', TH: 'θ', V: 'v', W: 'w', Y: 'j', Z: 'z', ZH: 'ʒ',
  });
  // This is a display-only, broad US-English transcription, not a G2P model.
  // CMUdict gives stress but no syllable boundaries. Put stress before the longest
  // conventional English onset preceding that vowel; do not infer missing stress.
  // Ambisyllabic consonants and accent-specific/allophonic details are not modeled.
  const ONSETS = new Set([
    'P R', 'P L', 'P Y', 'B R', 'B L', 'B Y', 'T R', 'T W', 'T Y',
    'D R', 'D W', 'D Y', 'K R', 'K L', 'K W', 'K Y', 'G R', 'G L', 'G W', 'G Y',
    'F R', 'F L', 'F Y', 'V Y', 'TH R', 'TH W', 'SH R', 'HH Y', 'M Y', 'N Y',
    'S P', 'S T', 'S K', 'S F', 'S M', 'S N', 'S L', 'S W', 'S Y',
    'S P R', 'S P L', 'S P Y', 'S T R', 'S T Y', 'S K R', 'S K L', 'S K W', 'S K Y',
  ]);
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

  function toIPA(phones) {
    if (!Array.isArray(phones) || !phones.length || phones.length > 256) return null;
    const parsed = [];
    for (const phone of phones) {
      if (typeof phone !== 'string') return null;
      const match = /^([A-Z]+)([012]?)$/.exec(phone);
      if (!match) return null;
      const [, name, stress] = match;
      if (own(VOWELS, name)) {
        const ipa = name === 'AH' && stress === '0' ? 'ə'
          : name === 'ER' && stress === '0' ? 'ɚ' : VOWELS[name];
        parsed.push({ name, stress, ipa, vowel: true });
      } else if (own(CONSONANTS, name) && !stress) {
        parsed.push({ name, stress: '', ipa: CONSONANTS[name], vowel: false });
      } else return null;
    }
    const vowels = parsed.map((phone, index) => phone.vowel ? index : -1).filter(index => index >= 0);
    if (!vowels.length) return null;
    const stressAt = new Map();
    if (vowels.length > 1) {
      for (let syllable = 0; syllable < vowels.length; syllable++) {
        const index = vowels[syllable], stress = parsed[index].stress;
        if (stress !== '1' && stress !== '2') continue;
        let start = syllable === 0 ? 0 : index;
        if (syllable > 0) {
          const previousVowel = vowels[syllable - 1];
          for (let candidate = previousVowel + 1; candidate < index; candidate++) {
            const onset = parsed.slice(candidate, index).map(phone => phone.name);
            if ((onset.length === 1 && onset[0] !== 'NG') || ONSETS.has(onset.join(' '))) {
              start = candidate;
              break;
            }
          }
        }
        stressAt.set(start, stress === '1' ? 'ˈ' : 'ˌ');
      }
    }
    return parsed.map((phone, index) => (stressAt.get(index) || '') + phone.ipa).join('');
  }

  const api = Object.freeze({ toIPA });
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MDDPronunciation = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
