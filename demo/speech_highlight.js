'use strict';

// Native word boundaries are preferred. The clock is only a visual estimate for
// voices which provide no boundaries; it never supplies pronunciation evidence.
(() => {
  const clock = () => typeof performance !== 'undefined' ? performance.now() : Date.now();
  const tokensOf = text => [...text.matchAll(/\S+/gu)].map(match => {
    const length = [...match[0]].length;
    const pause = /[.!?]["'\)\]]*$/.test(match[0]) ? .5 : /[,;:]["'\)\]]*$/.test(match[0]) ? .25 : 0;
    return { start: match.index, end: match.index + match[0].length,
      weight: .65 + Math.min(length, 12) * .07 + pause };
  });
  function containing(tokens, index) {
    let low = 0, high = tokens.length - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1, token = tokens[middle];
      if (index < token.start) high = middle - 1;
      else if (index >= token.end) low = middle + 1;
      else return token;
    }
    return null;
  }
  function wordRange(text, index) {
    if (typeof text !== 'string' || !Number.isInteger(index) || index < 0 || index >= text.length) return null;
    const token = containing(tokensOf(text), index);
    return token ? { start: token.start, end: token.end } : null;
  }

  class SpeechWordTracker {
    constructor(text, { onUpdate = () => {}, schedule = (fn, ms) => setTimeout(fn, ms),
      unschedule = id => clearTimeout(id), now = clock, rate = .98 } = {}) {
      this.text = typeof text === 'string' ? text : '';
      Object.assign(this, { onUpdate, schedule, unschedule, now });
      this.rate = Number.isFinite(rate) && rate > 0 ? Math.min(10, Math.max(.1, rate)) : .98;
      this.tokens = tokensOf(this.text);
      this.totalWeight = this.tokens.reduce((sum, token) => sum + token.weight, 0);
      let weight = 0;
      this.starts = this.tokens.map(token => { const start = weight; weight += token.weight; return start; });
      this.range = null; this.mode = null; this.timer = null;
      this.active = false; this.closed = false; this.native = false; this.audio = false; this.version = 0;
    }
    _clearTimer() {
      if (this.timer !== null) this.unschedule(this.timer);
      this.timer = null;
    }
    _emit(token, mode) {
      const range = token ? { start: token.start, end: token.end } : null;
      if (this.mode === mode && this.range?.start === range?.start && this.range?.end === range?.end) return;
      this.range = range; this.mode = mode;
      this.onUpdate(range ? { ...range } : null, mode);
    }
    _indexAt(weight) {
      let low = 0, high = this.starts.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (this.starts[middle] <= weight) low = middle + 1;
        else high = middle;
      }
      return Math.max(0, Math.min(this.tokens.length - 1, low - 1));
    }
    _scheduleNext(index) {
      if (!this.active || this.closed || this.native || this.audio || index + 1 >= this.tokens.length) return;
      const version = this.version;
      const elapsed = Math.max(0, this.now() - this.startedAt);
      const nextAt = this.starts[index + 1] * this.millisecondsPerWeight;
      this.timer = this.schedule(() => {
        if (!this.active || this.closed || this.native || this.audio || version !== this.version) return;
        this.timer = null;
        const progress = Math.max(0, this.now() - this.startedAt) / this.millisecondsPerWeight;
        const current = this._indexAt(progress);
        this._emit(this.tokens[current], 'estimated');
        this._scheduleNext(current);
      }, Math.max(1, nextAt - elapsed));
    }
    start() {
      if (this.closed || this.active || !this.tokens.length) return false;
      this.active = true; this.native = false; this.audio = false; this.version++;
      this.startedAt = this.now(); this.millisecondsPerWeight = 340 / this.rate;
      this._emit(this.tokens[0], 'estimated');
      this._scheduleNext(0);
      return true;
    }
    boundary(index) {
      if (!this.active || this.closed || !Number.isInteger(index) || index < 0 || index >= this.text.length) return false;
      const token = containing(this.tokens, index);
      if (!token) return false;
      this.native = true; this.version++; this._clearTimer();
      this._emit(token, 'boundary');
      return true;
    }
    audioTime(currentTime, duration) {
      if (!this.active || this.closed || this.native || !Number.isFinite(currentTime) || currentTime < 0
        || !Number.isFinite(duration) || duration <= 0) return false;
      this.audio = true; this.version++; this._clearTimer();
      const progress = Math.min(1, currentTime / duration) * this.totalWeight;
      this._emit(this.tokens[this._indexAt(progress)], 'estimated');
      return true;
    }
    end() {
      this.active = false; this.version++; this._clearTimer(); this._emit(null, null);
    }
    dispose() {
      if (this.closed) return;
      this.end(); this.closed = true;
    }
  }

  if (typeof module === 'object' && module.exports) module.exports = { SpeechWordTracker, wordRange };
  if (typeof window !== 'undefined') window.MDDSpeechHighlight = { SpeechWordTracker, wordRange };
})();
