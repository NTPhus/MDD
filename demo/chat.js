'use strict';

// Show the completed reply before autoplay. Speech boundaries only move the
// highlight; they never hide the words which have not been read yet.
function replyReady(job) {
  return job?.status === 'ready' || (job?.speech?.reply_ready === true
    && !['error', 'cancelled', 'cancelling'].includes(job?.status));
}
function chatPollDelay(job) {
  // Any speech stage can finish the reply between polls. Observe that gate
  // promptly; later pronunciation-only updates need less frequent polling.
  return replyReady(job) ? 700 : 150;
}

class LiveReply {
  constructor({ onUpdate = () => {}, onVoice = () => {}, onError = () => {}, speak = null,
    cancelSpeech = () => {}, voice = true } = {}) {
    Object.assign(this, { onUpdate, onVoice, onError, speak, cancelSpeech });
    this.text = ''; this.visible = 0; this.queuedUntil = 0; this.queue = []; this.active = null;
    this.final = false; this.muted = !voice || !speak; this.closed = false;
    this.epoch = 0; this.range = null; this.rangeMode = null; this.readUntil = 0;
  }
  get visibleText() { return this.text.slice(0, this.visible); }
  revealAll() {
    if (this.closed) return;
    this.visible = this.text.length;
    this.onUpdate(this.visibleText, this.range, this.rangeMode, this.readUntil);
  }
  update(text, { final = false } = {}) {
    if (this.closed) return;
    if (!text.startsWith(this.text)) {
      // A revised response must not leave obsolete speech in the device queue.
      this.mute(); this.visible = 0; this.queuedUntil = 0;
      this.final = false;
    }
    this.text = text; this.final ||= final;
    this.visible = this.final ? this.text.length : 0;
    // Paint all text synchronously before onVoice/speak can start playback.
    this.onUpdate(this.visibleText, this.range, this.rangeMode, this.readUntil);
    if (this.final && !this.muted) {
      this._queueClauses();
      this._play();
    }
  }
  _queueClauses() {
    while (this.queuedUntil < this.text.length) {
      const tail = this.text.slice(this.queuedUntil);
      const sentence = /[.!?]+["'\)\]]*(?=\s|$)/.exec(tail);
      const words = [...tail.matchAll(/\S+\s+/g)];
      const shortEnd = words.length >= 12 ? words[11].index + words[11][0].length : 0;
      let end = sentence ? sentence.index + sentence[0].length : 0;
      if (shortEnd && (!end || shortEnd < end)) end = shortEnd;
      if (!end && this.final) end = tail.length;
      if (!end) break;
      const chunk = tail.slice(0, end), leading = chunk.length - chunk.trimStart().length;
      if (chunk.trim()) this.queue.push({ text: chunk.trim(), offset: this.queuedUntil + leading });
      this.queuedUntil += end;
    }
  }
  _play() {
    if (this.closed || this.muted || this.active || !this.queue.length) return;
    const part = this.queue.shift(), epoch = this.epoch;
    this.active = part; this.onVoice(true);
    const valid = () => !this.closed && !this.muted && epoch === this.epoch && this.active === part;
    const boundary = (index = 0, mode = 'boundary') => {
      if (!valid() || part.offset === null || !Number.isInteger(index) || index < 0 || index >= part.text.length) return;
      const rest = part.text.slice(index), word = /^\s*\S+/.exec(rest);
      if (!word) return;
      const start = part.offset + index + (word[0].length - word[0].trimStart().length);
      this.range = { start, end: part.offset + index + word[0].length };
      this.rangeMode = mode;
      this.readUntil = this.range.end;
      this.onUpdate(this.visibleText, this.range, this.rangeMode, this.readUntil);
    };
    try {
      part.utterance = this.speak(part.text, {
        start: () => boundary(0, 'estimated'), boundary: (index, mode = 'boundary') => boundary(index, mode),
        end: () => { if (!valid()) return; this.readUntil = part.offset + part.text.length; this.active = null; this.range = null; this.rangeMode = null; this.onUpdate(this.visibleText, null, null, this.readUntil); this.onVoice(false); this._play(); },
        error: error => { if (!valid()) return; this.mute(); this.onError(error); },
      });
    } catch (error) { if (valid()) { this.mute(); this.onError(error); } }
  }
  mute() {
    this.muted = true; this.epoch++; this.queue = []; this.active = null; this.range = null; this.rangeMode = null; this.readUntil = 0;
    this.cancelSpeech(); this.onVoice(false); this.onUpdate(this.visibleText, null, null, 0);
  }
  dispose() {
    this.closed = true; this.mute();
  }
}
// Opening audio is requested immediately; only its presentation waits for a
// playable buffer and a short settling pause. Ordinary replies bypass this.
async function prepareOpeningAudio(audio, { isCurrent = () => true, delayMs = 350, timeoutMs = 25000 } = {}) {
  const playable = await new Promise((resolve, reject) => {
    let poll, deadline;
    const finish = (value, error) => {
      clearInterval(poll); clearTimeout(deadline);
      audio.removeEventListener('canplay', check); audio.removeEventListener('error', failed);
      if (error) reject(error); else resolve(value);
    };
    const check = () => { if (!isCurrent()) finish(false); else if (audio.readyState >= 3) finish(true); };
    const failed = () => finish(false, new Error('Không tải được giọng đọc. Bạn thử Phát lại.'));
    audio.addEventListener('canplay', check); audio.addEventListener('error', failed);
    poll = setInterval(check, 50);
    deadline = setTimeout(() => finish(false, new Error('Giọng đọc tải quá lâu. Bạn thử Phát lại.')), timeoutMs);
    audio.preload = 'auto';
    try { audio.load(); check(); } catch (error) { finish(false, error); }
  });
  if (!playable) return false;
  await new Promise(resolve => setTimeout(resolve, delayMs));
  return isCurrent();
}
// Prefetch and an explicit Listen action share the same provider request. A
// rejected request is removed so a later, explicit retry can try again.
function sharedVoiceRequest(requests, key, create) {
  if (requests.has(key)) return requests.get(key);
  const pending = Promise.resolve().then(create).finally(() => {
    if (requests.get(key) === pending) requests.delete(key);
  });
  requests.set(key, pending); return pending;
}
function voiceResourceKey(jobId, revision, target, selection, text) {
  return JSON.stringify([jobId, revision, target, selection.provider || 'ai33', selection.voice_id || '', Number(selection.speed), text]);
}
function voiceCatalog(config, provider) {
  if (provider === 'kokoro') return ['af_heart', 'am_michael', 'bf_emma', 'bm_george']
    .map(id => config?.kokoro_voices?.find(voice => voice.id === id)).filter(Boolean);
  return provider === 'ai33' ? config?.english_voices || [] : [];
}
function providerAvailable(config, provider) {
  if (provider === 'kokoro') return config?.kokoro_available === true;
  if (provider === 'ai33') return config?.ai33_available === true && config?.voice_locales?.includes('en-US');
  return provider === 'browser';
}
function preferredVoiceProvider(config, preference, preferenceVersion = 0) {
  // The English voice picker has four local choices when they are available.
  // Old browser/API preferences belong to the previous provider picker.
  if (providerAvailable(config, 'kokoro')) return 'kokoro';
  if (providerAvailable(config, preference)) return preference;
  if (providerAvailable(config, config?.default_voice_provider)) return config.default_voice_provider;
  return ['kokoro', 'ai33', 'browser'].find(provider => providerAvailable(config, provider));
}
function voiceSelection(config, provider, voiceId, speed) {
  const voices = voiceCatalog(config, provider);
  const defaultId = provider === 'kokoro' ? config?.kokoro_default_voice_id : config?.default_voice_id;
  const selected = voices.find(voice => voice.id === voiceId) || voices.find(voice => voice.id === defaultId) || voices[0];
  return { provider, ...(selected ? { voice_id: selected.id } : {}), speed };
}
function voiceDescription(voice) {
  return `${voice?.locale === 'en-GB' ? 'Anh' : 'Mỹ'} · ${voice?.gender === 'male' ? 'Nam' : 'Nữ'}`;
}
function pronunciationModels(config) {
  if (!Array.isArray(config?.pronunciation_models)) return [];
  const seen = new Set();
  return config.pronunciation_models.filter(model => model && ['cascade-B', 'crottc-if-l2-arctic'].includes(model.id)
    && !seen.has(model.id) && seen.add(model.id)).map(model => ({ id: model.id,
    label: typeof model.label === 'string' && model.label.trim() ? model.label.slice(0, 160)
      : model.id === 'cascade-B' ? 'wav2vec2 B' : 'CROTTC-IF · Thử nghiệm', ready: model.ready === true }));
}
function preferredPronunciationModel(config, preference) {
  const models = pronunciationModels(config);
  // Installing an experimental model must not silently change existing speech
  // checks. It is used only after an explicit, still-available user selection.
  return models.some(model => model.id === preference && model.ready) ? preference : 'cascade-B';
}
function pronunciationRequest(config, preference) {
  const selected = preferredPronunciationModel(config, preference);
  return pronunciationModels(config).some(model => model.id === selected && model.ready)
    ? { pronunciation_model_id: selected } : {};
}
function conversationInterpretation(speech) {
  if (!pronunciationReferenceReady(speech) || speech.conversation_recovery?.status === 'pending') return null;
  const selected = speech.conversation_source === 'asr_interpreted' ? speech.conversation_text : null;
  const recovery = speech.conversation_recovery;
  if (['accepted', 'unchanged'].includes(recovery?.status) && recovery.original === speech.transcript
    && recovery.reference_confirmed === false && typeof recovery.text === 'string'
    && recovery.text.trim() && recovery.text.length <= 1000
    && typeof selected === 'string' && selected.trim() === recovery.text.trim()) return recovery;
  const interpretation = speech?.interpretation;
  if (!['accepted', 'unchanged'].includes(interpretation?.status)
    || typeof interpretation.text !== 'string' || !interpretation.text.trim() || interpretation.text.length > 1000) return null;
  // The dialogue text can differ from the pronunciation target. Older saved
  // turns without dialogue metadata still use their selected BERT reference.
  const fallback = selected ?? (!recovery && !speech.conversation_source && speech.reference_source === 'bert_inferred'
    ? pronunciationReferenceText(speech) : null);
  if (typeof fallback !== 'string' || fallback.trim() !== interpretation.text.trim()) return null;
  return interpretation;
}
function pronunciationReferenceReady(speech) {
  return Boolean(speech) && speech.pronunciation_reference_ready !== false;
}
function pronunciationReferenceText(speech) {
  return pronunciationReferenceReady(speech) ? String(speech.reference_text || speech.transcript || '').trim() : '';
}
function conversationUserText(turn) {
  if (turn.confirmedSource === 'user_confirmed' && typeof turn.confirmedText === 'string'
    && turn.confirmedText.trim() && turn.confirmedText.length <= 1000) return turn.confirmedText;
  if (turn.pronunciationReferenceReady === false || turn.conversationTextPending === true) return '';
  for (const [source, expected, text] of [[turn.confirmedSource, 'user_confirmed', turn.confirmedText],
    [turn.interpretedSource, 'asr_interpreted', turn.interpretedText], [turn.normalizedSource, 'asr_normalized', turn.normalizedText]]) {
    if (source === expected && typeof text === 'string' && text.trim() && text.length <= 1000) return text;
  }
  return turn.content;
}
function conversationUserNote(turn) {
  if (turn.confirmedSource === 'user_confirmed') return '';
  if (turn.pronunciationReferenceReady === false || turn.conversationTextPending === true) return 'Đang nhận giọng nói…';
  return turn.interpretedSource === 'asr_interpreted' && conversationUserText(turn)?.trim() !== turn.content?.trim()
    ? 'Bản chép lời được khôi phục' : '';
}
function updateUserSpeech(turn, speech) {
  if (speech?.transcript) turn.content = speech.transcript;
  turn.pronunciationReferenceReady = speech?.pronunciation_reference_ready !== false;
  turn.conversationTextPending = speech?.conversation_recovery?.status === 'pending';
  const interpretation = conversationInterpretation(speech);
  turn.interpretedText = interpretation?.text;
  turn.interpretedSource = interpretation ? 'asr_interpreted' : undefined;
  turn.interpretedRequiresConfirmation = interpretation?.requires_confirmation === true;
  turn.normalizedText = speech?.conversation_source === 'asr_normalized' && speech.normalization?.lexical_preservation === true
    ? speech.conversation_text : undefined;
  turn.normalizedSource = turn.normalizedText ? 'asr_normalized' : undefined;
}
function microphoneConstraints(supported = {}, deviceId = '', mode = 'clear') {
  const audio = {};
  if (deviceId) audio.deviceId = { exact: deviceId };
  if (supported.channelCount) audio.channelCount = { ideal: 1 };
  for (const key of ['echoCancellation', 'noiseSuppression', 'autoGainControl']) {
    if (supported[key]) audio[key] = mode === 'clear';
  }
  return { audio };
}
function fluencyView(speech, pending = false) {
  const unavailable = { status: 'unavailable', metrics: [], message: 'Chưa có dữ liệu nhịp nói.' };
  const fluency = speech?.fluency;
  if (!fluency) return pending ? { status: 'pending', metrics: [], message: 'Đang kiểm tra nhịp nói…' } : unavailable;
  if (!['ready', 'partial'].includes(fluency.status) || fluency.source !== 'whisper_word_timestamps'
    || fluency.method !== 'word_timing_fluency_v1') return unavailable;
  const value = fluency.metrics;
  const finite = (number, max) => typeof number === 'number' && Number.isFinite(number) && number >= 0 && number <= max;
  if (!value || !Number.isSafeInteger(value.word_count) || value.word_count < 1 || value.word_count > 10000
    || !finite(value.audio_duration_sec, 3600) || value.audio_duration_sec === 0
    || !finite(value.recording_rate_wpm, 1000)) return unavailable;
  const number = (amount, digits = 0) => amount.toLocaleString('vi-VN', { maximumFractionDigits: digits });
  const metrics = [{ label: 'Nhịp nói', value: number(value.recording_rate_wpm), unit: 'từ/phút',
    help: 'Số từ trên toàn thời lượng bản thu, gồm cả khoảng chờ đầu và cuối.' }];
  if (fluency.status === 'partial') return { status: 'partial', metrics, message: 'Chưa có thời gian từng từ.' };
  if (!Number.isSafeInteger(value.internal_pause_count) || value.internal_pause_count < 0
    || value.internal_pause_count > value.word_count - 1
    || !finite(value.longest_internal_pause_sec, value.audio_duration_sec)
    || (value.internal_pause_count === 0 && value.longest_internal_pause_sec !== 0)) return unavailable;
  metrics.push({ label: 'Khoảng nghỉ', value: number(value.internal_pause_count), unit: '',
    help: 'Khoảng cách từ 0,25 giây giữa các từ theo thời gian Whisper; không tính đầu và cuối bản thu.' },
  { label: 'Nghỉ dài nhất', value: number(value.longest_internal_pause_sec, 2), unit: 'giây',
    help: 'Khoảng cách dài nhất giữa các từ; đây không phải điểm đánh giá chuyên gia.' });
  return { status: 'ready', metrics, message: '' };
}
function renderFluency(box, speech, pending = false) {
  const view = fluencyView(speech, pending), owner = box.ownerDocument;
  const element = (tag, className, text) => {
    const item = owner.createElement(tag); item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  // Keep the summary node and details.open state, including keyboard focus.
  // Updating metrics never scrolls, plays audio or issues requests.
  let body = [...box.children].find(item => item.className === 'chat-fluency-body');
  if (!body) {
    box.replaceChildren(); body = element('div', 'chat-fluency-body');
    box.append(element('summary', 'chat-pronunciation-title', 'Fluency'), body);
  }
  body.replaceChildren(); box.dataset.status = view.status;
  if (view.metrics.length) {
    const metrics = element('dl', 'chat-fluency-metrics');
    for (const metric of view.metrics) {
      const item = element('div', 'chat-fluency-metric'); item.title = metric.help;
      const result = element('dd', 'chat-fluency-value', metric.value);
      if (metric.unit) result.append(element('span', 'chat-fluency-unit', metric.unit));
      item.append(element('dt', 'chat-fluency-label', metric.label), result); metrics.append(item);
    }
    body.append(metrics);
  }
  if (view.message) body.append(element('p', 'chat-fluency-note', view.message));
}
if (typeof module === 'object' && module.exports) module.exports = { LiveReply, replyReady, chatPollDelay, prepareOpeningAudio, sharedVoiceRequest,
  voiceResourceKey, voiceCatalog, providerAvailable, preferredVoiceProvider, voiceSelection,
  pronunciationModels, preferredPronunciationModel, pronunciationRequest,
  conversationInterpretation, conversationUserText, conversationUserNote, updateUserSpeech, microphoneConstraints, pronunciationReferenceReady, pronunciationReferenceText,
  fluencyView, renderFluency };

(() => {
  if (typeof document === 'undefined') return;
  const node = (id) => document.getElementById(id);
  if (!node('chat-view')) return;
  const make = (tag, className, text) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const icon = name => {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use'); use.setAttribute('href', '#' + name); svg.append(use); return svg;
  };
  const settingsAnchor = document.createComment('chat-settings-home');
  node('chat-settings').before(settingsAnchor);
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const state = { config: null, model: 'qwen-trained', pronunciationModel: 'cascade-B', conversations: {}, rows: new Map(), busy: false,
    job: null, activeTurn: null, visible: false, recording: false, pending: false, decoding: false,
    stopping: false, context: null, stream: null, source: null, processor: null, sink: null,
    chunks: [], sampleCount: 0, timer: null, flush: null, epoch: 0, micAllowed: () => true,
    inputDeviceId: '', inputNotice: '', deviceHint: '', inputRms: 0, clippedBlocks: 0, signalHint: '',
    speaking: null, speechEpoch: 0, audio: null, playback: null, speechTrackers: new Set(), audioFrame: null,
    voiceTasks: new Set(), voiceRequests: new Map(), voiceBuffers: new Map(), urls: new Set(), lastAudio: null, destroyed: false,
    liveReply: null, liveTurnId: null, voiceStopped: false, openingPending: null, topic: null, inRoom: false, starting: false, hintKey: '', repeatTarget: null,
    stopRequested: false, recordingAudio: null, referenceRequests: new Map(), referenceOperation: null,
    scrollHeld: false, lastScrollTop: 0, voiceId: '', voiceSpeed: 1, voicePreference: null, voicePreferenceVersion: 0,
    pendingTopic: null, setupVoiceId: '', setupVoiceProvider: 'browser', setupSpeed: 1, voiceFilter: '', previewAudio: null };
  const sessionKey = () => state.model + (state.topic ? ':' + state.topic : '');
  const conversations = () => state.conversations[sessionKey()] ||= [];
  const turnRevision = turn => Math.max(Number.isSafeInteger(turn.referenceRevision) ? turn.referenceRevision : 0,
    Number.isSafeInteger(turn.metadata?.reference_revision) ? turn.metadata.reference_revision : 0);
  const userConversationText = conversationUserText;
  const lexicalText = text => (String(text || '').normalize('NFKC').replace(/’/g, "'").toLowerCase()
    .match(/[\p{L}\p{N}]+(?:'[\p{L}\p{N}]+)*/gu) || []).join('\u0000');
  function diagnosticView(speech) {
    const diagnostic = speech?.sound_candidates;
    const provisional = diagnostic?.provisional === true && (['asr', 'bert_inferred'].includes(diagnostic.source)
      || ['asr', 'bert_inferred'].includes(speech.reference_source));
    const confirmed = diagnostic?.reference_confirmed === true && ['reading_prompt', 'corrected_sentence', 'user_confirmed'].includes(speech?.reference_source);
    const words = diagnostic?.status === 'available' && diagnostic.diagnostic_only === true && (confirmed || provisional)
      ? [...new Map((diagnostic.words || []).filter(word => Number.isSafeInteger(word.index) && word.index >= 0
        && word.index < (speech.comparison?.words?.length || 0)).map(word => [word.index, word])).values()] : [];
    return { diagnostic, provisional, confirmed, words };
  }
  function chatScrollSnapshot() {
    const scroll = node('chat-scroll');
    return { top: scroll.scrollTop, follow: !state.scrollHeld && scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <= 32
      && !node('chat-messages').querySelector('.chat-turn-feedback[open]') };
  }
  function restoreChatScroll(snapshot) {
    const scroll = node('chat-scroll');
    scroll.scrollTop = snapshot.follow ? scroll.scrollHeight : snapshot.top;
    state.lastScrollTop = scroll.scrollTop;
  }
  const uuid = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), v => v.toString(16).padStart(2, '0')).join('');
  const pause = (ms) => new Promise(resolve => setTimeout(resolve, ms));
  const terminal = (status) => ['ready', 'error', 'cancelled'].includes(status);
  const showError = (text = '') => { node('chat-error').textContent = text; node('chat-error').hidden = !text; };

  async function api(path, body) {
    const options = { cache: 'no-store', signal: AbortSignal.timeout(25000) };
    if (body !== undefined) { options.method = 'POST'; options.headers = { 'Content-Type': 'application/json' }; options.body = JSON.stringify(body); }
    const response = await fetch(path, options);
    let value;
    try { value = await response.json(); } catch { throw new Error('The server did not return a valid response.'); }
    if (!response.ok) {
      const error = new Error(value.message || 'The recording could not be processed.');
      error.status = response.status; error.code = value.error;
      throw error;
    }
    return value;
  }
  function save() {
    try {
      sessionStorage.setItem('am-voice-chat-v2', JSON.stringify(state.conversations, (key, value) => ['audioUrl', 'voice'].includes(key) ? undefined : value));
      sessionStorage.setItem('am-chat-selection-v1', JSON.stringify({ model: state.model, topic: state.topic, inRoom: state.inRoom, mode: node('chat-mode').value,
        pronunciationModel: state.pronunciationModel,
        voiceId: state.voiceId, voiceSpeed: state.voiceSpeed, voiceProvider: node('chat-voice-provider').value,
        voiceProviderVersion: state.voicePreferenceVersion, autoRead: node('chat-auto-read').checked }));
    } catch {}
  }
  function restore() {
    try {
      const data = JSON.parse(sessionStorage.getItem('am-voice-chat-v2') || '{}');
      for (const key of Object.keys(data)) if (/^qwen-(?:base|trained)(?::[a-z-]+)?$/.test(key) && Array.isArray(data[key])) {
        state.conversations[key] = data[key].filter(v => v && ['user', 'assistant'].includes(v.role) && typeof v.content === 'string' && typeof v.id === 'string')
          .slice(-40).map(v => ({ ...v, status: terminal(v.status) ? v.status : 'cancelled', jobId: undefined }));
      }
      const selection = JSON.parse(sessionStorage.getItem('am-chat-selection-v1') || '{}');
      if (['qwen-base', 'qwen-trained'].includes(selection.model)) state.model = selection.model;
      if (['cascade-B', 'crottc-if-l2-arctic'].includes(selection.pronunciationModel)) state.pronunciationModel = selection.pronunciationModel;
      if (typeof selection.topic === 'string' && /^[a-z-]+$/.test(selection.topic)) state.topic = selection.topic;
      state.inRoom = selection.inRoom === true;
      if (['conversation', 'reading'].includes(selection.mode)) node('chat-mode').value = selection.mode;
      if (typeof selection.voiceId === 'string') state.voiceId = selection.voiceId;
      if ([.85, 1, 1.15].includes(selection.voiceSpeed)) state.voiceSpeed = selection.voiceSpeed;
      if (['kokoro', 'ai33', 'browser'].includes(selection.voiceProvider)) state.voicePreference = selection.voiceProvider;
      if (selection.voiceProviderVersion === 1) state.voicePreferenceVersion = 1;
      if (typeof selection.autoRead === 'boolean') node('chat-auto-read').checked = selection.autoRead;
    } catch {}
  }
  function controls() {
    const phonemeModels = pronunciationModels(state.config);
    const phonemeReady = !phonemeModels.length || phonemeModels.some(model => model.id === state.pronunciationModel && model.ready);
    const ready = state.config?.speech_ready && phonemeReady && state.config.models.some(m => m.id === state.model && m.ready);
    const locked = state.busy || state.pending || state.decoding || state.stopping || state.starting;
    const audioLocked = state.recording || state.pending || state.stopping || Boolean(state.referenceOperation?.regenerating);
    node('chat-mic').disabled = locked || !ready || !AudioContextClass || !navigator.mediaDevices?.getUserMedia || !window.isSecureContext;
    node('chat-mic').setAttribute('aria-pressed', String(state.recording));
    node('chat-mic').querySelector('span').textContent = state.recording ? 'Send recording' : 'Record';
    node('chat-mic').setAttribute('aria-label', state.recording ? 'Gửi bản ghi âm' : 'Nhấn để nói tiếng Anh');
    node('chat-mic-label').textContent = state.recording ? 'Nhấn để gửi' : locked ? 'Đang xử lý…' : 'Nhấn để nói';
    node('chat-input-device').disabled = locked || state.recording;
    node('chat-input-processing').disabled = locked || state.recording;
    node('chat-input-meter').hidden = !state.recording;
    node('chat-upload').disabled = locked || state.recording || !ready || !AudioContextClass;
    node('chat-stop').hidden = !state.busy;
    node('chat-stop').disabled = !state.job || state.activeTurn?.status === 'cancelling';
    for (const id of ['chat-model', 'chat-mode', 'chat-practice', 'chat-new', 'chat-start-new', 'chat-topics']) node(id).disabled = locked || state.recording || !state.config;
    if (node('chat-pronunciation-model')) node('chat-pronunciation-model').disabled = locked || state.recording || !phonemeModels.length;
    for (const id of ['chat-free-talk', 'chat-open-reading']) node(id).disabled = locked || state.recording || !ready;
    node('chat-availability').hidden = !state.config || ready;
    if (state.config && !ready) node('chat-availability').textContent = !state.config.speech_ready
      ? 'Máy này chưa có checkpoint nhận dạng tiếng nói. Cài model để bắt đầu luyện nói.'
      : 'Máy này chưa có model hội thoại. Cài model để bắt đầu luyện nói.';
    for (const button of node('chat-topic-grid').querySelectorAll('button')) {
      button.disabled = locked || state.recording || !ready;
      const cta = button.querySelector('.chat-topic-cta');
      cta.toggleAttribute('hidden', !ready);
    }
    node('chat-retry-audio').disabled = locked || state.recording;
    for (const id of ['chat-read-prompt', 'chat-reading-listen']) node(id).disabled = audioLocked;
    for (const button of document.querySelectorAll('.chat-hint-keywords button')) button.disabled = audioLocked;
    node('chat-stop-voice').hidden = !state.speaking && !state.openingPending;
    for (const button of document.querySelectorAll('.chat-correction-listen,.chat-rewrite-listen,.chat-word-listen,.chat-sentence-listen,.chat-sound-listen')) button.disabled = audioLocked;
    for (const turn of conversations()) {
      const button = state.rows.get(turn.id)?.details.querySelector('.chat-sentence-listen');
      if (!button) continue;
      const voice = turn.sentenceVoice;
      const waiting = Boolean(turn.sentenceVoiceSelection && state.voiceTasks.has(voiceRequestInfo(turn, true).key));
      button.disabled = audioLocked || turn.status !== 'ready' || waiting;
      button.setAttribute('aria-busy', String(waiting));
      button.lastChild.textContent = state.speaking === 'sentence-' + turn.id ? 'Dừng đọc câu'
          : voice?.status === 'error' && voice.retryable ? 'Thử lại giọng đọc' : 'Nghe cả câu';
    }
    for (const button of document.querySelectorAll('.chat-correction-practise')) button.disabled = locked || state.recording || !button.dataset.jobId;
    if (node('chat-repeat-clear')) node('chat-repeat-clear').disabled = locked || state.recording;
    const statuses = { transcribing: 'Listening…', correcting: 'Preparing suggestions…', analyzing: 'Checking sounds…', generating: 'Replying…', coaching: 'Checking pronunciation…', cancelling: 'Stopping…' };
    node('chat-room-status').textContent = state.recording ? 'Recording…' : state.decoding ? 'Opening audio…' : state.pending ? 'Opening mic…'
      : state.starting ? 'Starting…' : state.openingPending ? 'Preparing voice…' : state.speaking ? 'Speaking…' : state.busy ? statuses[state.activeTurn?.status] || 'Thinking…' : 'Your turn';
    node('chat-current-model').textContent = state.repeatTarget ? 'Practise your sentence' : node('chat-mode').value === 'reading' ? 'Read & practise' : state.config?.topics?.find(t => t.id === state.topic)?.title || 'English conversation';
    node('chat-topic-level').textContent = state.config?.topics?.find(t => t.id === state.topic)?.level || 'EN';
    const scenario = state.config?.topics?.find(t => t.id === state.topic);
    node('chat-scenario').hidden = !scenario || node('chat-mode').value !== 'conversation';
    node('chat-scenario-title').textContent = scenario?.title || '';
    node('chat-scenario-description').textContent = scenario?.description || '';
    node('chat-settings-label').textContent = state.visible && state.inRoom
      ? node('chat-voice-provider').value !== 'browser' ? (selectedVoice()?.name.split(' - ')[0] || 'Giọng tiếng Anh') : 'Giọng đọc' : 'Giọng đọc & cài đặt';
    node('chat-english-voice').disabled = node('chat-voice-provider').value === 'browser';
    const voiceProvider = node('chat-voice-provider').value;
    const simpleVoices = state.config?.kokoro_available === true;
    node('chat-voice-provider').hidden = simpleVoices;
    document.querySelector('label[for="chat-voice-provider"]').hidden = simpleVoices;
    document.querySelector('label[for="chat-english-voice"]').textContent = simpleVoices ? 'Giọng tiếng Anh'
      : `Giọng tiếng Anh${voiceProvider === 'ai33' ? ' · ai33' : ''}`;
    node('chat-english-voice').setAttribute('aria-label', 'Giọng tiếng Anh');
    node('chat-voice-cost').hidden = simpleVoices;
    node('chat-voice-cost').textContent = voiceProvider === 'kokoro' ? ''
      : voiceProvider === 'ai33' ? 'ai33 reads the finished reply; API credits may apply.' : 'Giọng đọc sử dụng tính năng có sẵn của trình duyệt.';
    node('chat-settings').querySelector('summary').setAttribute('aria-label', 'Giọng đọc và cài đặt hội thoại');
    node('chat-model-description').textContent = state.model === 'qwen-trained' ? 'Fine-tuned Qwen for pronunciation practice tips.' : 'Original Qwen for comparison; it may not follow the coaching format.';
    node('chat-model-note').hidden = false;
    node('chat-model-note').textContent = 'Conversation replies use original Qwen. This selection controls the pronunciation coach.';
    node('chat-practice-panel').hidden = node('chat-mode').value !== 'reading';
    node('chat-mic-status').textContent = !window.isSecureContext ? 'Micro cần HTTPS hoặc localhost. Bạn vẫn có thể tải audio lên.'
      : state.recording ? state.signalHint || 'Đang thu âm · nhấn nút micro để gửi'
      : state.pending ? 'Đang mở micro và chờ quyền truy cập…' : state.decoding ? 'Đang chuẩn bị âm thanh…'
      : state.inputNotice || state.deviceHint || 'Nói tiếng Anh rõ ràng · tối đa 30 giây';
    const prompt = state.config?.practice_prompts.find(p => p.id === node('chat-practice').value);
    node('chat-practice-text').textContent = prompt?.text || '';
    node('chat-reading-card').hidden = !state.repeatTarget && node('chat-mode').value !== 'reading';
    node('chat-reading-card').querySelector('span').textContent = state.repeatTarget ? 'Read your completed sentence aloud' : 'Read aloud';
    node('chat-reading-text').textContent = state.repeatTarget?.text || prompt?.text || '';
    if (node('chat-repeat-clear')) node('chat-repeat-clear').hidden = !state.repeatTarget;
  }
  function soundText(event, symbols) {
    const phone = value => value ? `${value}${symbols?.[value] ? ' /' + symbols[value] + '/' : ''}` : 'no sound';
    if (event.kind === 'deletion') return `Possible omission: expected ${phone(event.expected_phone)}.`;
    if (event.kind === 'insertion') return `Possible extra sound: ${phone(event.observed_phone)}.`;
    return `Possible substitution: expected ${phone(event.expected_phone)}; detected ${phone(event.observed_phone)}.`;
  }
  function wordPronunciation(speech, index) {
    const target = speech.target_words?.[index];
    const ipa = window.MDDPronunciation?.toIPA(target?.pronunciation_phones || target?.phones);
    return ipa ? '/' + ipa + '/' : null;
  }
  function pronunciation(box, speech, turnId) {
    const previousWords = [...box.querySelectorAll('.chat-sound-word')];
    const opened = new Set(previousWords.map((word, index) => word.open ? index : -1));
    const focused = previousWords.findIndex(word => word.contains(document.activeElement));
    box.replaceChildren();
    if (!pronunciationReferenceReady(speech) || !speech?.transcript || !speech.comparison) return;
    const { diagnostic, words: candidateWords } = diagnosticView(speech);
    const unconfirmed = speech.reference_source === 'asr';
    const candidates = new Map(candidateWords.map(word => [word.index, word]));
    box.append(make('summary', 'chat-pronunciation-title', 'Sound details'));
    if (diagnostic?.status === 'unavailable' && diagnostic.reason === 'reference_contains_unknown_words') {
      const missing = diagnostic.missing_words || speech.missing_words || speech.reference_resolution?.missing_words || [];
      box.append(make('p', 'chat-reference-note', `Chưa có phiên âm: ${missing.join(', ') || 'một số từ'}.`));
    }
    if (speech.comparison) {
      const words = make('div', 'chat-sound-words');
      speech.comparison.words.forEach((word, index) => {
        const difference = !unconfirmed && word.status === 'difference';
        const candidate = !difference ? candidates.get(index) : null;
        const events = difference ? (speech.comparison.events || []).filter(e => e.word_index === index)
          : candidate ? candidate.events || [] : [];
        const descriptions = events.map(e => soundText(e, speech.phone_symbols));
        const displayedStatus = candidate ? 'possible_difference' : unconfirmed && word.status !== 'unavailable' ? 'uncertain'
          : word.status === 'match' && speech.assessment?.calibration?.applied === false ? 'uncertain' : word.status;
        const detail = make('details', 'chat-sound-word ' + displayedStatus);
        detail.dataset.evidence = candidate ? 'diagnostic' : difference ? 'model_difference' : 'undecided';
        detail.open = opened.has(index);
        const title = make('summary', '', word.text);
        title.title = descriptions.join('\n') || ({
          match: 'Chưa thấy khác biệt.',
          uncertain: 'Chưa đủ dữ liệu.',
          unavailable: 'Chưa có phiên âm.',
        }[displayedStatus] || 'Chưa đủ dữ liệu.');
        detail.append(title);
        const body = make('div', 'chat-sound-detail');
        const target = speech.target_words?.[index];
        const ipa = wordPronunciation(speech, index);
        if (ipa) body.append(make('p', 'chat-sound-ipa', ipa));
        if (target?.phones?.length) body.append(make('p', '', 'Target phones (ARPAbet): ' + target.phones.join(' ')));
        descriptions.forEach(text => body.append(make('p', '', text)));
        if (!descriptions.length && !ipa && !target?.phones?.length) body.append(make('p', '', title.title));
        const listen = make('button', 'text-button chat-sound-listen', 'Hear this word'); listen.type = 'button';
        listen.addEventListener('click', () => speakText(word.text, 'word-' + index)); body.append(listen); detail.append(body); words.append(detail);
      });
      box.append(words);
      if (focused >= 0) words.children[focused]?.querySelector('summary')?.focus({ preventScroll: true });
    }
    const coaching = speech.coaching;
    if (coaching?.status === 'ready') {
      const pairs = Object.entries(coaching.feedback || {});
      for (const [word, tips] of pairs) {
        const item = make('div', 'chat-coaching-tip'); item.append(make('strong', '', word));
        for (const tip of tips) { item.append(make('p', '', tip.issue), make('p', 'chat-suggestion', tip.suggestion)); }
        box.append(item);
      }
    }
    const target = pronunciationReferenceText(speech);
    if (target) {
      const text = make('p', 'chat-sentence-text'), timing = make('span', 'chat-word-timing');
      const sentenceRow = { text, timing }; timing.hidden = true;
      paintText(sentenceRow, target, null);
      const row = state.rows.get(turnId); if (row) row.sentenceRow = sentenceRow;
      const listenSentence = make('button', 'text-button chat-sentence-listen', 'Nghe cả câu'); listenSentence.type = 'button';
      listenSentence.prepend(icon('i-speaker'));
      listenSentence.addEventListener('click', () => {
        const turn = conversations().find(t => t.id === turnId);
        if (state.speaking === 'sentence-' + turnId) stopSpeech();
        else if (turn) speak(turn, true);
      });
      box.append(text, timing, listenSentence);
    }
  }
  function render() {
    const roomOpen = state.visible && state.inRoom;
    document.body.classList.toggle('chat-room-open', roomOpen);
    const settings = node('chat-settings'), toolbar = document.querySelector('.chat-toolbar-actions');
    if (roomOpen && settings.parentElement !== toolbar) { settings.open = false; toolbar.insertBefore(settings, node('chat-stop-voice')); }
    else if (!roomOpen && settings.parentElement === toolbar) { settings.open = false; settingsAnchor.after(settings); }
    node('chat-topic-picker').hidden = state.inRoom;
    node('chat-layout').hidden = !state.inRoom;
    const scrollSnapshot = chatScrollSnapshot();
    node('chat-welcome').hidden = conversations().length > 0;
    const turns = conversations();
    for (const [index, turn] of turns.entries()) {
      const followingTurn = turn.role === 'user' ? turns[index + 1] : null;
      const audioLoading = Boolean(turn.audioUrl && followingTurn?.role === 'assistant'
        && !terminal(followingTurn.status));
      let row = state.rows.get(turn.id);
      if (!row) {
        const item = make('article', 'chat-message ' + turn.role), body = make('div', 'chat-message-body');
        item.append(make('div', 'chat-message-avatar', turn.role === 'user' ? 'You' : 'M'), body);
        const label = make('span', 'chat-message-label', turn.role === 'user' ? 'You' : 'Mezon Coach');
        const text = make('div', 'chat-message-text'), timing = make('span', 'chat-word-timing'), meta = make('div', 'chat-message-meta'), details = make('details', 'chat-pronunciation'), review = make('div', 'chat-turn-review'), correction = make('section', 'chat-correction');
        timing.hidden = true;
        // The outer feedback is opt-in; opening it reveals the sound check first.
        details.open = true;
        const completionExtra = make('details', 'chat-pronunciation chat-completion-extra');
        completionExtra.append(make('summary', 'chat-pronunciation-title', 'Hoàn thiện câu (tùy chọn)'), correction);
        const fluency = make('details', 'chat-pronunciation chat-fluency');
        const rewriteExtra = make('details', 'chat-rewrite-extra'), rewrite = make('section', 'chat-rewrite');
        rewriteExtra.append(make('summary', 'chat-pronunciation-title', 'Ngữ pháp & cách nói tự nhiên'), rewrite);
        const actions = make('div', 'chat-message-actions'), feedback = make('details', 'chat-turn-feedback');
        const summary = make('summary', 'chat-feedback-summary'), feedbackLabel = make('span', 'chat-feedback-label');
        summary.append(icon('i-chat'), feedbackLabel);
        summary.title = 'Xem phát âm';
        const content = make('div', 'chat-feedback-content'), notice = make('p', 'chat-feedback-notice', 'Đang nhận giọng nói…');
        feedback.append(summary, content); content.append(notice, review, details, fluency, completionExtra, rewriteExtra);
        feedback.dataset.turnId = turn.id;
        if (turn.role === 'assistant' && index > 0 && turns[index - 1].role === 'user') node('chat-messages').append(feedback);
        body.append(label, text, timing, meta, actions); node('chat-messages').append(item);
        row = { item, body, text, timing, meta, details, fluency, review, correction, completionExtra, rewrite, rewriteExtra, feedback, feedbackLabel, notice, actions, audio: null, speechKey: '' }; state.rows.set(turn.id, row);
      }
      row.item.dataset.status = turn.status;
      row.item.classList.toggle('error', turn.status === 'error');
      row.item.querySelector('.chat-message-label').textContent = turn.role === 'user' ? 'You'
        : turn.metadata?.reply_source === 'clarification_fallback' ? 'Cần làm rõ'
          : turn.metadata?.reply_source === 'conversation_unavailable' ? 'Chưa có phản hồi' : 'Mezon Coach';
      const live = state.liveTurnId === turn.id ? state.liveReply : null;
      const playback = state.playback?.turnId === turn.id ? state.playback : null;
      const openingPending = state.openingPending === turn;
      const displayText = openingPending ? '' : turn.role === 'user'
        ? (audioLoading ? 'Audio loading…' : userConversationText(turn))
        : playback ? turn.content : live ? live.visibleText : turn.content;
      paintText(row, displayText,
        playback ? playback.range : live?.range, playback ? playback.mode : live?.rangeMode);
      row.meta.textContent = openingPending ? 'Đang chuẩn bị giọng đọc…' : turn.role === 'user'
        ? conversationUserNote(turn)
        : replyReady(turn) ? '' : turn.error || ({ transcribing: 'Listening…', correcting: 'Preparing suggestions…', analyzing: 'Checking sounds…', generating: 'Replying…', coaching: 'Checking pronunciation…', cancelling: 'Stopping…', cancelled: 'Stopped' }[turn.status] || '');
      row.meta.hidden = !row.meta.textContent;
      const speechKey = JSON.stringify(turn.speech);
      if (speechKey !== row.speechKey || row.correctionJobId !== turn.jobId) {
        row.speechKey = speechKey; row.correctionJobId = turn.jobId;
        sentenceCorrection(row, turn); sentenceRewrite(row, turn); pronunciation(row.details, turn.speech, turn.id); turnReview(row, turn);
      }
      const fluencyKey = JSON.stringify([turn.speech?.fluency, terminal(turn.status)]);
      if (fluencyKey !== row.fluencyKey) {
        row.fluencyKey = fluencyKey; renderFluency(row.fluency, turn.speech, !terminal(turn.status));
      }
      row.fluency.hidden = !turn.speech;
      const practise = row.correction.querySelector('.chat-correction-practise');
      if (practise) practise.dataset.jobId = turn.speech?.correction?.mode === 'completion_only' && turn.status === 'ready' ? turn.jobId || '' : '';
      const referenceApply = row.review.querySelector('.chat-reference-apply');
      if (referenceApply) referenceApply.disabled = Boolean(state.busy || state.referenceOperation || state.referenceRequests.has(turn.jobId) || row.referenceExpired || turn.status !== 'ready' || !turn.jobId);
      const completion = turn.speech?.correction?.mode === 'completion_only' ? turn.speech.correction : null;
      row.correction.hidden = !completion || completion.status === 'not_needed';
      row.completionExtra.hidden = completion?.status !== 'ready' || !completion.changed;
      row.rewriteExtra.hidden = turn.speech?.rewrite?.mode !== 'grammar_and_natural' || turn.speech?.rewrite?.status === 'not_needed';
      row.details.hidden = !pronunciationReferenceReady(turn.speech) || !turn.speech?.comparison;
      row.review.hidden = !pronunciationReferenceReady(turn.speech) || !turn.speech?.transcript;
      row.notice.hidden = pronunciationReferenceReady(turn.speech) && Boolean(turn.speech?.transcript);
      row.notice.textContent = terminal(turn.status) ? 'Chưa có dữ liệu phát âm cho lượt nói này.' : 'Đang kiểm tra phát âm…';
      row.feedbackLabel.textContent = feedbackSummary(turn);
      row.feedback.classList.toggle('is-pending', !terminal(turn.status));
      if (turn.audioUrl && state.urls.has(turn.audioUrl) && !row.audio) {
        row.audio = make('audio', 'chat-message-audio'); row.audio.controls = true; row.audio.hidden = true; row.audio.preload = 'none'; row.audio.src = turn.audioUrl;
        row.audio.addEventListener('play', () => { if (state.recordingAudio !== row.audio) stopSpeech(); state.recordingAudio = row.audio; render(); });
        row.audio.addEventListener('ended', () => { if (state.recordingAudio === row.audio) state.recordingAudio = null; render(); });
        row.body.insertBefore(row.audio, row.meta);
      }
      row.actions.replaceChildren();
      if (turn.role === 'user' && row.audio) {
        const replay = make('button', '', state.recordingAudio === row.audio ? 'Dừng bản thu' : 'Nghe bản thu'); replay.type = 'button'; replay.prepend(icon('i-speaker'));
        replay.disabled = state.recording || state.pending || state.stopping || Boolean(state.referenceOperation?.regenerating);
        replay.addEventListener('click', () => {
          if (state.recordingAudio === row.audio) stopSpeech();
          else { stopSpeech(); row.audio.play().catch(() => showError('Không phát được bản ghi âm.')); }
        }); row.actions.append(replay);
      }
      if (turn.role === 'assistant' && turn.content && !openingPending && (replyReady(turn) || terminal(turn.status))) {
        const read = make('button', '', state.speaking === turn.id ? 'Dừng giọng đọc' : turn.voice?.status === 'error' && turn.voice.retryable ? 'Thử lại giọng đọc' : 'Phát lại');
        read.prepend(icon('i-speaker'));
        read.type = 'button'; read.disabled = turn.voice?.status === 'generating' || Boolean(state.referenceOperation?.regenerating);
        read.setAttribute('aria-busy', String(turn.voice?.status === 'generating'));
        read.addEventListener('click', () => state.speaking === turn.id ? stopSpeech() : speak(turn)); row.actions.append(read);
      }
      if (turn.role === 'assistant' && terminal(turn.status)
        && (['clarification_fallback', 'conversation_unavailable'].includes(turn.metadata?.reply_source) || turn.speech?.transcript_clarity?.needs_clarification)) {
        const clarify = make('button', 'chat-clarify-action', 'Sửa câu hoặc nói lại'); clarify.type = 'button';
        clarify.disabled = state.busy || state.recording || state.pending || state.stopping;
        clarify.title = 'Xác nhận câu đã nói, hoặc nhấn mic để thu lại.';
        clarify.addEventListener('click', () => {
          row.feedback.open = true;
          const editor = row.review.querySelector('.chat-reference-editor'); if (editor) editor.open = true;
          const input = row.review.querySelector('.chat-reference-input');
          if (input) { input.focus({ preventScroll: true }); input.scrollIntoView({ block: 'nearest' }); }
          else node('chat-mic').focus({ preventScroll: true });
        });
        row.actions.append(clarify);
      }
    }
    renderHints(); controls(); restoreChatScroll(scrollSnapshot);
  }
  function feedbackSummary(turn) {
    const speech = turn.speech;
    if (!pronunciationReferenceReady(speech) || !speech?.transcript) return terminal(turn.status) ? 'Chưa đủ bằng chứng' : 'Đang kiểm tra phát âm…';
    const flagged = speech.reference_source === 'asr' ? 0 : (speech.comparison?.words || []).filter(word => word.status === 'difference').length;
    if (flagged) return `${flagged} từ cần kiểm tra`;
    const count = diagnosticView(speech).words.length;
    if (count) return `${count} từ cần kiểm tra`;
    return 'Chưa đủ bằng chứng';
  }
  function sentenceCorrection(row, turn) {
    const correction = turn.speech?.correction;
    const whyOpen = row.correction.querySelector('.chat-correction-why')?.open || false;
    row.correction.replaceChildren();
    // Older saved turns may contain grammar rewrites. They are not completions.
    if (!correction || correction.mode !== 'completion_only' || correction.status === 'not_needed') return;
    row.correction.dataset.status = correction.status;
    row.correction.append(make('span', 'chat-correction-label', 'OPTIONAL SENTENCE COMPLETION'));
    if (correction.status === 'pending') {
      row.correction.append(make('p', 'chat-correction-note', terminal(turn.status)
        ? 'Sentence completion was stopped. Your original recording is kept below.'
        : 'Qwen is finishing your thought while keeping your words…'));
      return;
    }
    if (correction.status !== 'ready' || typeof correction.corrected !== 'string' || !correction.corrected.trim()) {
      row.correction.append(make('p', 'chat-correction-note', correction.message || 'Qwen could not reliably finish this thought. Your original wording is kept.'));
      return;
    }
    row.correction.append(make('p', 'chat-correction-caption', correction.changed ? 'Your completed thought' : 'Your sentence'));
    row.correction.append(make('p', 'chat-correction-sentence', correction.corrected));
    const changes = Array.isArray(correction.changes) ? correction.changes.filter(change => change
      && typeof change.before === 'string' && typeof change.after === 'string' && change.before !== change.after) : [];
    const edits = changes.map(change => change.before && change.after ? `“${change.before}” → “${change.after}”`
      : change.after ? `Add “${change.after}”` : `Remove “${change.before}”`);
    row.correction.append(make('p', 'chat-correction-edits', edits.length ? edits.join(' · ')
      : correction.changed ? 'Your words are kept; the ending is added.' : 'No completion needed. Your wording is kept.'));
    if (correction.explanation) {
      const why = make('details', 'chat-correction-why'); why.open = whyOpen;
      why.append(make('summary', '', 'About this completion'), make('p', 'chat-correction-note', correction.explanation));
      row.correction.append(why);
    }
    const actions = make('div', 'chat-correction-actions');
    const listen = make('button', 'text-button chat-correction-listen', 'Listen'); listen.type = 'button';
    listen.addEventListener('click', () => speakText(correction.corrected, 'correction-' + turn.id));
    const practise = make('button', 'chat-correction-practise', 'Practise this sentence'); practise.type = 'button';
    practise.dataset.jobId = turn.status === 'ready' ? turn.jobId || '' : '';
    practise.title = turn.jobId ? 'Use this sentence as the target for your next recording.' : 'Record a new turn to practise a completed sentence in this page session.';
    practise.addEventListener('click', () => {
      if (correction.mode !== 'completion_only' || !turn.jobId || turn.status !== 'ready' || state.busy || state.recording || state.pending || state.decoding || state.stopping || state.starting) return;
      stopSpeech(); state.repeatTarget = { jobId: turn.jobId, text: correction.corrected };
      render(); node('chat-reading-card').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); node('chat-mic').focus({ preventScroll: true });
    });
    actions.append(listen, practise); row.correction.append(actions);
    if (!turn.jobId) row.correction.append(make('p', 'chat-correction-session-note', 'Saved completion · Make a new recording to practise in this session.'));
  }
  function sentenceRewrite(row, turn) {
    const rewrite = turn.speech?.rewrite;
    row.rewrite.replaceChildren();
    if (!rewrite || rewrite.mode !== 'grammar_and_natural') return;
    row.rewrite.dataset.status = rewrite.status;
    if (rewrite.status === 'pending') {
      row.rewrite.append(make('p', 'chat-rewrite-note', terminal(turn.status)
        ? 'Chưa hoàn tất gợi ý cho lượt nói này.' : 'Qwen đang gợi ý cách diễn đạt…'));
      return;
    }
    const fields = [['grammar_corrected', 'grammar_status', 'Ngữ pháp'], ['natural', 'natural_status', 'Cách nói tự nhiên hơn']]
      .filter(([field, status]) => (rewrite[status] === undefined || rewrite[status] === 'ready')
        && typeof rewrite[field] === 'string' && rewrite[field].trim());
    if (rewrite.status !== 'ready' || !fields.length) {
      row.rewrite.append(make('p', 'chat-rewrite-note', 'Bạn thử nói lại hoặc sửa câu ở trên nhé.'));
      return;
    }
    if (!rewrite.changed && fields.length === 2 && !rewrite.partial
      && !['partial', 'unavailable'].includes(rewrite.feedback_state)) {
      row.rewrite.append(make('p', 'chat-rewrite-note', 'Qwen giữ nguyên cách diễn đạt của bạn.'));
    }
    for (const [field, , caption] of fields) {
      const line = make('div', 'chat-rewrite-line');
      line.append(make('span', 'chat-rewrite-caption', caption), make('p', 'chat-rewrite-sentence', rewrite[field]));
      const listen = make('button', 'text-button chat-rewrite-listen', 'Nghe'); listen.type = 'button';
      listen.prepend(icon('i-speaker')); listen.setAttribute('aria-label', 'Nghe ' + caption.toLowerCase());
      listen.addEventListener('click', () => speakText(rewrite[field], `rewrite-${turn.id}-${field}`));
      line.append(listen); row.rewrite.append(line);
    }
    if (rewrite.explanation) row.rewrite.append(make('p', 'chat-rewrite-note', rewrite.explanation));
  }
  function turnReview(row, turn) {
    const speech = turn.speech;
    const pendingReference = state.referenceRequests.get(turn.jobId);
    const editorOpen = row.review.querySelector('.chat-reference-editor')?.open;
    const soundsOpen = row.review.querySelector('.chat-phonetic-choices')?.open;
    const originalOpen = row.review.querySelector('.chat-original-transcript')?.open;
    row.review.replaceChildren(); if (!pronunciationReferenceReady(speech) || !speech?.transcript) return;
    const resolution = speech.reference_resolution || {};
    const { diagnostic, words: candidates } = diagnosticView(speech);
    const unconfirmed = speech.reference_source === 'asr';
    row.feedback.dataset.assessment = candidates.length ? 'candidates' : 'uncertain';
    row.review.dataset.assessment = row.feedback.dataset.assessment;
    const missing = speech.missing_words?.length ? speech.missing_words : resolution.missing_words || [];
    const target = pronunciationReferenceText(speech);
    const targetLabel = speech.reference_source === 'user_confirmed' ? 'Câu đã xác nhận · đối chiếu phát âm'
      : speech.reference_source === 'reading_prompt' ? 'Câu đọc mẫu · đối chiếu phát âm'
        : speech.reference_source === 'bert_inferred' ? 'Câu gợi ý để đối chiếu phát âm' : 'Câu đối chiếu phát âm';
    row.review.append(make('span', 'chat-review-label', targetLabel), make('p', 'chat-target-text', target));
    const original = make('details', 'chat-original-transcript'); original.open = originalOpen ?? false;
    original.append(make('summary', 'chat-reference-note', 'Xem bản nhận dạng gốc'),
      make('p', 'chat-target-text', speech.transcript));
    row.review.append(original);
    let note = '';
    if (diagnostic?.status === 'unavailable' && diagnostic.reason === 'reference_contains_unknown_words') {
      const unknown = diagnostic.missing_words || missing;
      note = `Chưa có phiên âm: ${unknown.join(', ') || 'một số từ'}.`;
    } else if (diagnostic?.status === 'unavailable') note = 'Chưa có kết quả phát âm.';
    if (!speech.comparison) {
      note = missing.length ? 'Chưa có phiên âm: ' + missing.join(', ') + '.' : 'Chưa có kết quả phát âm.';
    }
    if (note) row.review.append(make('p', 'chat-reference-note', note));
    if (speech.comparison) {
      const flagged = speech.comparison.words.map((word, index) => ({ ...word, index })).filter(word => !unconfirmed && word.status === 'difference');
      const suggestions = candidates.filter(word => !flagged.some(flag => flag.index === word.index))
        .map(word => ({ ...word, text: speech.comparison.words[word.index].text, tentative: true }));
      const practiceWords = [...flagged, ...suggestions].sort((a, b) => a.index - b.index);
      if (practiceWords.length) {
        const words = make('div', 'chat-review-words');
        for (const word of practiceWords) {
          const practice = make('div', 'chat-word-practice');
          practice.dataset.evidence = word.tentative ? 'diagnostic' : 'model_difference';
          const button = make('button', 'chat-word-details', word.text); button.type = 'button';
          button.setAttribute('aria-label', 'Xem chi tiết âm của từ ' + word.text);
          button.title = (word.tentative ? word.events || [] : (speech.comparison.events || []).filter(e => e.word_index === word.index))
              .map(e => soundText(e, speech.phone_symbols)).join('\n');
          button.addEventListener('click', () => {
            row.details.open = true; const detail = row.details.querySelectorAll('.chat-sound-word')[word.index];
            if (detail) { detail.open = true; detail.querySelector('summary').focus(); detail.scrollIntoView({ block: 'nearest' }); }
          });
          const ipa = wordPronunciation(speech, word.index);
          const phonetic = make('span', 'chat-word-ipa', ipa || 'Chưa có phiên âm');
          phonetic.lang = 'en'; phonetic.setAttribute('aria-label', ipa ? 'Phiên âm: ' + ipa : 'Chưa có phiên âm');
          const listen = make('button', 'text-button chat-word-listen', 'Nghe'); listen.type = 'button';
          listen.prepend(icon('i-speaker')); listen.setAttribute('aria-label', 'Nghe từ ' + word.text);
          listen.addEventListener('click', () => speakText(word.text, `word-${turn.id}-${word.index}`));
          practice.append(button, phonetic, listen); words.append(practice);
        }
        row.review.append(words);
      }
    }
    const editor = make('details', 'chat-reference-editor');
    editor.open = editorOpen ?? false;
    editor.append(make('summary', '', 'Sửa câu'));
    const input = make('textarea', 'chat-reference-input'); input.rows = 2; input.maxLength = 1000;
    input.setAttribute('aria-label', 'Câu tiếng Anh bạn đã nói trong bản thu');
    input.spellcheck = false;
    // The editor, word choices and sentence voice share the pronunciation
    // target; the conversation bubble may use a separate recovered transcript.
    // Keep any manual edit intact while later coaching snapshots arrive.
    if (!row.referenceDirty) row.referenceDraft = String(pendingReference?.referenceText || target);
    input.value = row.referenceDraft ?? target;
    input.disabled = Boolean(pendingReference);
    input.addEventListener('input', () => { row.referenceDraft = input.value; row.referenceDirty = true; });
    editor.append(input);
    const selectWord = (index, original, candidate) => {
      const matches = [...input.value.matchAll(/\S+/g)], token = matches[index];
      if (!token || lexicalText(token[0]) !== lexicalText(original)) return;
      const replacement = token[0].replace(/[A-Za-z]+(?:['’][A-Za-z]+)*/, candidate);
      input.value = input.value.slice(0, token.index) + replacement + input.value.slice(token.index + token[0].length);
      row.referenceDraft = input.value; row.referenceDirty = true; input.focus({ preventScroll: true });
    };
    const alternatives = Array.isArray(resolution.suggestions) ? resolution.suggestions : [];
    if (alternatives.length) {
      const choices = make('div', 'chat-spelling-choices');
      choices.append(make('p', 'chat-reference-note', 'Chưa có trong từ điển · gợi ý theo cách viết'));
      for (const suggestion of alternatives) {
        if (typeof suggestion?.word !== 'string' || !Array.isArray(suggestion.candidates)) continue;
        const line = make('div', 'chat-spelling-row'); line.append(make('strong', '', suggestion.word));
        for (const candidate of suggestion.candidates) {
          if (typeof candidate !== 'string' || !/^[a-z]{3,24}$/.test(candidate)) continue;
          const choose = make('button', 'chat-spelling-choice', candidate); choose.type = 'button';
          choose.disabled = Boolean(pendingReference); choose.title = 'Điền gợi ý; chưa đối chiếu audio';
          choose.addEventListener('click', () => {
            // Only replace this still-matching raw token. A prior manual edit
            // must not be silently overwritten by an outdated suggestion.
            selectWord(suggestion.index, suggestion.word, candidate);
          });
          line.append(choose);
        }
        choices.append(line);
      }
      editor.append(choices);
    }
    const sounds = speech.phonetic_candidates;
    if (speech.reference_source === 'asr' && sounds?.status === 'available'
      && sounds.diagnostic_only === true && sounds.requires_confirmation === true) {
      const choices = make('details', 'chat-phonetic-choices'); choices.open = soundsOpen ?? false;
      choices.append(make('summary', '', 'Từ gần âm đã nghe'));
      for (const word of sounds.words || []) {
        const line = make('div', 'chat-spelling-row');
        line.append(make('strong', '', word.asr_word));
        const observed = (word.heard_phones || []).join(' ');
        const heard = make('span', 'chat-heard-phones', observed ? `Âm model (ARPAbet): ${observed}` : 'Chưa tách được âm');
        line.append(heard);
        for (const candidate of (word.candidates || []).slice(0, 3)) {
          if (typeof candidate.word !== 'string' || !/^[a-z]+(?:'[a-z]+)?$/.test(candidate.word) || candidate.word.length > 24) continue;
          const choose = make('button', 'chat-spelling-choice chat-phonetic-choice', candidate.word); choose.type = 'button';
          choose.disabled = Boolean(pendingReference); choose.title = 'Gợi ý theo âm; bấm để điền, rồi xác nhận câu bạn đã nói';
          choose.addEventListener('click', () => selectWord(word.index, word.asr_word, candidate.word));
          line.append(choose);
        }
        choices.append(line);
      }
      editor.append(choices);
    }
    const apply = make('button', 'chat-reference-apply', pendingReference ? 'Đang đối chiếu…' : 'Đối chiếu audio này');
    apply.type = 'button'; apply.disabled = Boolean(state.busy || state.referenceOperation || pendingReference || row.referenceExpired || turn.status !== 'ready' || !turn.jobId);
    apply.addEventListener('click', async () => {
      if (state.busy || state.referenceOperation || state.recording || state.pending || state.stopping || state.decoding || state.starting
        || state.referenceRequests.has(turn.jobId) || row.referenceExpired || turn.status !== 'ready' || !turn.jobId) return;
      const reference = input.value.trim();
      if (!reference) { row.referenceError = 'Nhập câu tiếng Anh bạn đã nói.'; turnReview(row, turn); return; }
      await regenerateReference(row, turn, reference);
    });
    editor.append(apply);
    if (!turn.jobId || row.referenceExpired) editor.append(make('p', 'chat-reference-note', 'Bản thu của phiên cũ không còn để đối chiếu lại.'));
    else if (turn.status !== 'ready' && !terminal(turn.status)) editor.append(make('p', 'chat-reference-note', 'Có thể đối chiếu lại khi AI trả lời xong.'));
    if (row.referenceError) { const error = make('p', 'chat-reference-error', row.referenceError); error.setAttribute('role', 'alert'); editor.append(error); }
    row.review.append(editor);
  }
  async function regenerateReference(row, turn, reference) {
    const requestedJobId = turn.jobId, epoch = state.epoch;
    const turns = conversations(), index = turns.indexOf(turn), user = index > 0 && turns[index - 1].role === 'user' ? turns[index - 1] : null;
    const currentText = turn.speech?.conversation_text || turn.speech?.reference_text || turn.speech?.transcript || (user && userConversationText(user));
    const request = { referenceText: reference, revision: turnRevision(turn), epoch, regenerating: lexicalText(reference) !== lexicalText(currentText) };
    const previous = { content: turn.content, status: turn.status, metadata: turn.metadata, speech: turn.speech,
      error: turn.error, voice: turn.voice, referenceRevision: turn.referenceRevision };
    const owns = () => state.referenceOperation === request && state.referenceRequests.get(requestedJobId) === request;
    let accepted = false, started = false, failures = 0, busyOwned = false, regenerated = false;
    state.referenceRequests.set(requestedJobId, request); state.referenceOperation = request;
    row.referenceDraft = reference; row.referenceDirty = true; row.referenceError = '';
    const prepareGeneration = () => {
      if (busyOwned) return;
      request.regenerating = true; busyOwned = true;
      stopSpeech(); clearReply();
      turn.content = ''; turn.status = 'generating'; turn.error = undefined; turn.voice = { status: 'not_requested' };
      turn.metadata = { reference_revision: request.revision };
      state.stopRequested = false; state.busy = true; state.job = requestedJobId; state.activeTurn = turn;
    };
    if (request.regenerating) prepareGeneration();
    row.speechKey = ''; render();
    const update = job => {
      const revision = Number.isSafeInteger(job.metadata?.reference_revision) ? job.metadata.reference_revision : -1;
      if (job.id !== requestedJobId || revision < request.revision || !owns()) return false;
      if (revision > request.revision && started) { stopSpeech(); clearReply(); started = false; }
      request.revision = revision;
      turn.referenceRevision = revision; turn.metadata = job.metadata || {};
      turn.speech = job.speech; turn.voice = job.voice || { status: 'not_requested' };
      if (user && job.speech?.reference_source === 'user_confirmed' && typeof job.speech.conversation_text === 'string'
        && job.speech.conversation_text.trim() && job.speech.conversation_text.length <= 1000) {
        user.confirmedText = job.speech.conversation_text; user.confirmedSource = 'user_confirmed'; user.confirmedRevision = revision;
      }
      turn.content = job.text || ''; turn.status = job.status; turn.error = job.error?.message;
      if (!started && epoch === state.epoch && state.visible) { beginReply(turn); started = true; }
      if (['error', 'cancelled'].includes(job.status)) stopSpeech();
      if (started) state.liveReply?.update(turn.content, { final: replyReady(job) || terminal(job.status) });
      autoplayReply(turn);
      prefetchSentenceVoice(turn);
      row.referenceDirty = false; row.referenceDraft = job.speech?.conversation_text || reference;
      row.speechKey = ''; render(); return true;
    };
    try {
      let job = await api('/api/chat/reference', { job_id: requestedJobId, reference_text: reference });
      if (!owns()) return;
      const revision = Number.isSafeInteger(job.metadata?.reference_revision) ? job.metadata.reference_revision : -1;
      if (job.id !== requestedJobId || revision < request.revision || job.speech?.reference_source !== 'user_confirmed'
        || typeof job.speech?.conversation_text !== 'string' || !job.speech.conversation_text.trim()) {
        throw new Error('Máy chủ chưa xác nhận câu đối chiếu cho bản thu này.');
      }
      if (job.reference_update?.changed === false && job.reference_update.regenerated_reply === false) {
        if (revision !== request.revision || job.status !== 'ready') throw new Error('Kết quả đối chiếu không khớp phiên hiện tại.');
        // A lexical no-op confirms/reassesses the recording only. Do not touch
        // the live reply, its speech queue, the displayed user or the history.
        if (busyOwned) Object.assign(turn, previous);
        turn.speech = job.speech; turn.referenceRevision = revision; turn.metadata = job.metadata || previous.metadata;
        if (user) {
          user.confirmedText = job.speech.conversation_text; user.confirmedSource = 'user_confirmed'; user.confirmedRevision = revision;
        }
        row.referenceDirty = false; row.referenceDraft = job.speech.reference_text || reference;
        accepted = true; return;
      }
      if (job.reference_update?.changed !== true || job.reference_update.regenerated_reply !== true || revision <= request.revision) {
        throw new Error('Máy chủ chưa xác nhận lượt trả lời mới cho bản thu này.');
      }
      prepareGeneration(); regenerated = true;
      accepted = true; request.revision = revision;
      if (state.stopRequested) job = await api('/api/chat/cancel', { job_id: requestedJobId });
      while (!state.destroyed && owns()) {
        if (epoch !== state.epoch || !state.visible) {
          if (!terminal(job.status)) {
            try { job = await api('/api/chat/cancel', { job_id: requestedJobId }); } catch {}
          }
          // Keep the accepted reference and raw recording, without starting a
          // reply or applying stale audio after the learner leaves the room.
          update({ ...job, status: 'cancelled' }); break;
        }
        if (update(job) && terminal(job.status)) break;
        await pause(chatPollDelay(job));
        try { job = await api('/api/chat/jobs/' + requestedJobId); failures = 0; }
        catch (error) { if (++failures >= 3) throw error; await pause(1000); }
      }
    } catch (error) {
      if (!owns()) return;
      if (busyOwned) { stopSpeech(); clearReply(); }
      if (!accepted && busyOwned) {
        Object.assign(turn, previous);
        if (turn.voice?.status === 'generating') turn.voice = { status: 'not_requested' };
      }
      else if (accepted && regenerated) { turn.status = 'error'; turn.error = error.message || 'Chưa tạo được câu trả lời mới.'; }
      row.referenceExpired ||= error.status === 404 && error.code === 'chat_not_found';
      row.referenceError = error.message || 'Chưa đối chiếu được. Bạn thử lại nhé.';
    } finally {
      const owner = owns();
      if (state.referenceRequests.get(requestedJobId) === request) state.referenceRequests.delete(requestedJobId);
      if (owner) {
        state.referenceOperation = null;
        if (busyOwned) { state.busy = false; state.job = null; state.activeTurn = null; }
      }
      const currentRow = state.rows.get(turn.id);
      if (currentRow && currentRow !== row) {
        currentRow.referenceError = row.referenceError; currentRow.referenceExpired = row.referenceExpired;
        if (!currentRow.referenceDirty) { currentRow.referenceDraft = row.referenceDraft; currentRow.referenceDirty = row.referenceDirty; }
        currentRow.speechKey = '';
      }
      row.speechKey = ''; render(); save();
      if (owner && accepted && regenerated && epoch === state.epoch) autoplayReply(turn);
      if (owner && accepted && epoch === state.epoch) prefetchSentenceVoice(turn);
    }
  }
  function renderHints() {
    const topic = state.config?.topics?.find(t => t.id === state.topic);
    const turns = conversations(), latest = [...turns].reverse().find(t => t.role === 'assistant' && (t.speech?.guidance || t.metadata?.guidance));
    const hint = latest?.speech?.guidance || latest?.metadata?.guidance || topic?.guidance;
    const visible = Boolean(topic && state.inRoom && Array.isArray(hint?.keywords) && hint.keywords.length);
    node('chat-hints').hidden = !visible;
    node('chat-inline-hints').hidden = !visible;
    node('chat-layout').classList.toggle('with-hints', visible);
    if (!visible) return;
    const key = JSON.stringify(hint);
    if (key !== state.hintKey) {
      state.hintKey = key;
      for (const id of ['chat-hint-keywords', 'chat-inline-keywords']) {
        node(id).replaceChildren();
        for (const keyword of hint.keywords) {
          const button = make('button', '', keyword); button.type = 'button'; button.title = 'Listen: ' + keyword;
          button.addEventListener('click', () => speakText(keyword, 'keyword')); node(id).append(button);
        }
      }
      node('chat-hint-starter').textContent = hint.starter;
      node('chat-inline-starter').textContent = hint.starter;
    }
    const count = turns.filter(t => t.role === 'user').length;
    node('chat-turn-count').textContent = count ? `${count} speaking turn${count > 1 ? 's' : ''} · Keep going` : 'Take your time. Your first turn starts here.';
  }
  function renderTopics() {
    const icons = { sun: 'i-sun', plane: 'i-plane', cup: 'i-cup', briefcase: 'i-briefcase', spark: 'i-spark', laptop: 'i-laptop', book: 'i-book', city: 'i-city' };
    const artwork = {
      'daily-life': 'daily-life', travel: 'travel', food: 'food', work: 'work',
      hobbies: 'hobbies', technology: 'technology', learning: 'learning', city: 'city'
    };
    const topics = state.config?.topics || [];
    node('chat-topic-count').textContent = `${topics.length} chủ đề`;
    node('chat-topic-grid').replaceChildren();
    for (const topic of topics) {
      const button = make('button', 'chat-topic-card'); button.type = 'button'; button.dataset.topic = topic.id;
      const art = make('div', 'chat-topic-art'); art.setAttribute('aria-hidden', 'true');
      if (artwork[topic.id]) {
        const image = make('img', 'chat-topic-image');
        image.src = `/assets/topics/${artwork[topic.id]}.webp`;
        image.alt = '';
        image.decoding = 'async';
        art.append(image);
      }
      const head = make('div', 'chat-topic-card-top');
      const topicIcon = make('span', 'chat-topic-icon'); topicIcon.append(icon(icons[topic.icon] || 'i-spark'));
      head.append(topicIcon, make('span', 'chat-topic-level', topic.level));
      art.append(head, make('span', 'chat-topic-art-orbit'));
      button.append(art, make('h3', '', topic.title), make('p', '', topic.description));
      const keywords = make('div', 'chat-topic-preview'); topic.guidance.keywords.slice(0, 3).forEach(word => keywords.append(make('span', '', word)));
      const cta = make('span', 'chat-topic-cta'); cta.append(make('span', '', 'Bắt đầu'), icon('i-arrow'));
      button.append(keywords, cta); button.addEventListener('click', () => openTopicSetup(topic.id)); node('chat-topic-grid').append(button);
    }
  }
  const currentVoiceProvider = () => node('chat-voice-provider').value;
  const serverVoiceSelected = () => ['kokoro', 'ai33'].includes(currentVoiceProvider());
  function selectedVoice() { return voiceCatalog(state.config, currentVoiceProvider()).find(v => v.id === state.voiceId); }
  function renderVoiceOptions() {
    const provider = currentVoiceProvider();
    const selection = voiceSelection(state.config, provider, state.voiceId, state.voiceSpeed);
    state.voiceId = selection.voice_id || '';
    node('chat-english-voice').replaceChildren(...voiceCatalog(state.config, provider).map(voice => {
      const option = make('option', '', provider === 'kokoro' ? `${voiceDescription(voice)} · ${voice.name}`
        : `${voice.name} · ${voice.locale}`); option.value = voice.id; return option;
    }));
    node('chat-english-voice').value = state.voiceId;
  }
  function stopVoicePreview() {
    if (state.previewAudio) { state.previewAudio.pause(); state.previewAudio.removeAttribute('src'); state.previewAudio.load(); state.previewAudio = null; }
  }
  function setupSelection() {
    const voice = voiceCatalog(state.config, state.setupVoiceProvider).find(v => v.id === state.setupVoiceId);
    node('topic-setup-selection').textContent = `${voice ? state.setupVoiceProvider === 'kokoro'
      ? voiceDescription(voice) : voice.name.split(' - ')[0] + ' · ' + voice.locale : 'Giọng trình duyệt'} · ${state.setupSpeed}×`;
  }
  function renderSetupVoices() {
    const container = node('topic-voice-options'); container.replaceChildren();
    const simpleVoices = state.config.kokoro_available === true;
    node('chat-topic-setup').classList.toggle('curated-voices', simpleVoices);
    node('topic-voice-search').hidden = simpleVoices;
    document.querySelector('label[for="topic-voice-search"]').hidden = simpleVoices;
    document.querySelector('.topic-voice-filters').hidden = simpleVoices;
    const query = node('topic-voice-search').value.trim().toLocaleLowerCase();
    const voices = (simpleVoices ? ['kokoro'] : ['ai33']).flatMap(provider => providerAvailable(state.config, provider)
      ? voiceCatalog(state.config, provider).map(voice => ({ ...voice, voiceProvider: provider })) : []).filter(v => simpleVoices
        || (!state.voiceFilter || v.locale === state.voiceFilter) && `${v.name} ${v.provider} ${v.locale}`.toLocaleLowerCase().includes(query));
    node('topic-voice-count').textContent = `${voices.length} giọng`;
    for (const voice of voices) {
      const row = make('div', 'topic-voice-row'), label = make('label', 'topic-voice-choice');
      const radio = make('input'); radio.type = 'radio'; radio.name = 'topic-voice'; radio.value = voice.id;
      radio.checked = state.setupVoiceId === voice.id && state.setupVoiceProvider === voice.voiceProvider;
      radio.disabled = !providerAvailable(state.config, voice.voiceProvider);
      const accent = voice.locale === 'en-GB' ? 'UK' : 'US';
      label.append(radio);
      if (!simpleVoices) label.append(make('span', 'topic-voice-accent', accent));
      const copy = make('span', 'topic-voice-copy'); copy.append(make('strong', '', simpleVoices ? voiceDescription(voice) : voice.name.split(' - ')[0]));
      copy.append(make('small', '', simpleVoices ? voice.name : `${voice.gender === 'female' ? 'Nữ' : voice.gender === 'male' ? 'Nam' : 'English'} · ${voice.provider === 'edge' ? 'Edge' : 'ElevenLabs'}`));
      label.append(copy);
      if (!simpleVoices && voice.voiceProvider === preferredVoiceProvider(state.config) && voice.id === (voice.voiceProvider === 'kokoro'
        ? state.config.kokoro_default_voice_id : state.config.default_voice_id)) label.append(make('span', 'topic-voice-badge', 'Mặc định'));
      radio.addEventListener('change', () => { state.setupVoiceId = voice.id; state.setupVoiceProvider = voice.voiceProvider; setupSelection(); });
      const preview = make('button', 'topic-voice-preview'); preview.type = 'button'; preview.append(icon('i-play'));
      preview.setAttribute('aria-label', `Nghe thử ${simpleVoices ? voiceDescription(voice) : voice.name}`); preview.title = 'Nghe thử'; preview.disabled = !voice.preview_url;
      preview.addEventListener('click', async () => {
        stopVoicePreview();
        if (!/^\/api\/chat\/voice-preview\/[\w-]+$/.test(voice.preview_url || '')) return;
        const audio = new Audio(voice.preview_url); state.previewAudio = audio; audio.playbackRate = state.setupSpeed;
        node('topic-setup-status').textContent = `Đang nghe thử ${voice.name.split(' - ')[0]}…`;
        audio.onended = () => { if (state.previewAudio === audio) { state.previewAudio = null; node('topic-setup-status').textContent = ''; } };
        try { await audio.play(); }
        catch { if (state.previewAudio === audio) node('topic-setup-status').textContent = 'Chưa phát được bản nghe thử. Bạn có thể chọn giọng khác hoặc thử lại.'; }
      });
      row.append(label, preview); container.append(row);
    }
    if (!voices.length) container.append(make('p', 'topic-no-voices', query ? 'Không tìm thấy giọng phù hợp.' : 'Chưa có giọng đọc cho bộ lọc này.'));
    if (!simpleVoices) {
      const fallback = make('label', 'topic-browser-choice'), radio = make('input');
      radio.type = 'radio'; radio.name = 'topic-voice'; radio.value = 'browser'; radio.checked = state.setupVoiceId === 'browser';
      radio.addEventListener('change', () => { state.setupVoiceId = 'browser'; state.setupVoiceProvider = 'browser'; setupSelection(); });
      fallback.append(radio, make('span', '', 'Dùng giọng tiếng Anh của trình duyệt')); container.append(fallback);
    }
    setupSelection();
  }
  function openTopicSetup(topicId) {
    if (state.busy || state.recording || state.starting || !state.config) return;
    const topic = state.config.topics?.find(t => t.id === topicId); if (!topic) return;
    stopSpeech(); stopVoicePreview(); state.pendingTopic = topicId; state.voiceFilter = '';
    state.setupVoiceProvider = serverVoiceSelected() && selectedVoice() ? currentVoiceProvider() : 'browser';
    state.setupVoiceId = state.setupVoiceProvider !== 'browser' ? state.voiceId : 'browser';
    state.setupSpeed = state.voiceSpeed;
    node('topic-setup-title').textContent = topic.title; node('topic-setup-level').textContent = topic.level;
    node('topic-setup-description').textContent = topic.description;
    node('topic-setup-role').textContent = topic.role || 'Your English conversation partner';
    node('topic-setup-auto-read').checked = node('chat-auto-read').checked;
    node('topic-voice-search').value = '';
    node('topic-setup-status').textContent = state.config.kokoro_available ? 'Chọn giọng Mỹ hoặc Anh, nam hoặc nữ. Nhấn nút nghe thử để tìm giọng bạn thích.'
      : state.config.ai33_available ? 'Nghe thử mẫu có sẵn. Giọng ai33 khi trò chuyện sử dụng credit của tài khoản.' : 'Bạn có thể dùng giọng tiếng Anh của trình duyệt.';
    for (const button of document.querySelectorAll('[data-voice-locale]')) button.setAttribute('aria-pressed', String(!button.dataset.voiceLocale));
    for (const radio of document.querySelectorAll('[name=topic-speed]')) radio.checked = Number(radio.value) === state.setupSpeed;
    renderSetupVoices(); node('chat-topic-setup').showModal();
  }
  function resetRows() { node('chat-messages').replaceChildren(); state.rows.clear(); state.hintKey = ''; }
  async function startTopic(topicId, reset = false) {
    if (state.busy || state.recording || state.starting || !state.config) return;
    stopSpeech(); clearReply(); state.repeatTarget = null; state.topic = topicId; state.inRoom = true; node('chat-mode').value = 'conversation';
    if (reset) state.conversations[sessionKey()] = [];
    resetRows(); showError(); state.lastAudio = null; node('chat-audio-retry').hidden = true;
    if (conversations().length) { render(); save(); return; }
    const epoch = state.epoch; state.starting = true; render();
    try {
      const job = await api('/api/chat/topic', { model_id: state.model, topic_id: topicId, request_id: uuid() });
      if (epoch !== state.epoch || !state.visible) return;
      const turn = { id: uuid(), role: 'assistant', opening: true, content: job.text, status: 'ready', jobId: job.id, metadata: job.metadata };
      if (node('chat-auto-read').checked) state.openingPending = turn;
      conversations().push(turn); beginReply(turn); render(); state.liveReply.update(turn.content, { final: true });
      if (state.openingPending === turn) speak(turn, false, { opening: true });
    } catch (error) { showError(error.message); }
    finally { state.starting = false; render(); save(); }
  }
  function openFree(mode = 'conversation') {
    if (state.busy || state.recording || state.starting) return;
    stopSpeech(); clearReply(); state.repeatTarget = null; state.topic = null; state.inRoom = true; node('chat-mode').value = mode;
    resetRows(); showError(); node('chat-audio-retry').hidden = true; state.lastAudio = null; render(); save();
  }
  function paintText(row, text, range, mode = null, readUntil = null) {
    if (!text.startsWith(row.shownText || '')) { row.text.replaceChildren(); row.shownText = ''; row.readUntil = 0; }
    if (Number.isInteger(readUntil) && readUntil >= 0) row.readUntil = Math.min(readUntil, text.length);
    else if (range) row.readUntil = range.end;
    const from = (row.shownText || '').length;
    // Append words as safe DOM text, keeping earlier spans stable while speaking.
    for (const match of text.slice(from).matchAll(/\s+|\S+/g)) {
      if (/^\s/.test(match[0])) row.text.append(document.createTextNode(match[0]));
      else {
        const span = make('span', 'chat-reply-word', match[0]); span.dataset.start = from + match.index;
        span.dataset.end = from + match.index + match[0].length; row.text.append(span);
      }
    }
    row.shownText = text;
    for (const span of row.text.querySelectorAll('.chat-reply-word')) {
      const active = Boolean(range && +span.dataset.start < range.end && +span.dataset.end > range.start);
      span.classList.toggle('is-read', +span.dataset.end <= (row.readUntil || 0));
      span.classList.toggle('is-speaking', active);
      if (active) span.setAttribute('aria-current', 'true'); else span.removeAttribute('aria-current');
    }
    row.timing.hidden = true;
    row.timing.textContent = '';
  }
  function cancelBrowserSpeech() {
    for (const tracker of state.speechTrackers) tracker.dispose();
    state.speechTrackers.clear(); window.speechSynthesis?.cancel();
  }
  function browserVoice(text, callbacks, rate = .98) {
    rate *= state.voiceSpeed;
    const utterance = new SpeechSynthesisUtterance(text); utterance.lang = 'en-US'; utterance.rate = rate;
    const voices = window.speechSynthesis.getVoices();
    utterance.voice = voices.find(v => /^en[-_]/i.test(v.lang) && v.localService) || voices.find(v => /^en[-_]/i.test(v.lang)) || null;
    const tracker = new window.MDDSpeechHighlight.SpeechWordTracker(text, {
      rate, onUpdate: (range, mode) => { if (range) callbacks.boundary?.(range.start, mode); },
    });
    state.speechTrackers.add(tracker);
    const finish = callback => value => {
      tracker.dispose(); state.speechTrackers.delete(tracker); callback?.(value);
    };
    utterance.onstart = () => { callbacks.start?.(); tracker.start(); };
    utterance.onboundary = event => { if (!event.name || event.name === 'word') tracker.boundary(event.charIndex); };
    utterance.onend = finish(callbacks.end); utterance.onerror = finish(callbacks.error);
    try { window.speechSynthesis.speak(utterance); }
    catch (error) { tracker.dispose(); state.speechTrackers.delete(tracker); throw error; }
    return utterance;
  }
  function beginReply(turn) {
    state.liveReply?.dispose(); state.liveTurnId = turn.id; state.voiceStopped = false;
    state.liveReply = new LiveReply({
      voice: state.openingPending !== turn && state.visible && node('chat-auto-read').checked && node('chat-voice-provider').value === 'browser',
      speak: window.speechSynthesis && window.SpeechSynthesisUtterance ? browserVoice : null,
      cancelSpeech: cancelBrowserSpeech,
      onUpdate: (text, range, mode, readUntil) => {
        const row = state.rows.get(turn.id); if (!row) return;
        if (state.openingPending === turn) return;
        if (state.playback?.turnId === turn.id) return;
        const scrollSnapshot = chatScrollSnapshot();
        paintText(row, text, range, mode, readUntil); restoreChatScroll(scrollSnapshot);
      },
      onVoice: active => { if (active) state.speaking = turn.id; else if (!state.playback && state.speaking === turn.id) state.speaking = null; render(); },
      onError: () => showError('Voice could not start. Tap Listen when the reply is ready.'),
    });
  }
  function clearReply() {
    state.liveReply?.dispose(); state.liveReply = null; state.liveTurnId = null;
  }
  function history() {
    const turns = conversations().filter(t => !t.opening), result = [];
    for (let i = 0; i + 1 < turns.length; i += 2) if (turns[i].role === 'user' && turns[i + 1].role === 'assistant' && turns[i + 1].status === 'ready') {
      result.push({ role: 'user', content: userConversationText(turns[i]) }, { role: 'assistant', content: turns[i + 1].content });
    }
    while (result.length > 20 || result.reduce((sum, x) => sum + x.content.length, 0) > 12000) result.splice(0, 2);
    return result;
  }
  function base64(buffer) {
    const bytes = new Uint8Array(buffer); let text = '';
    for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode(...bytes.subarray(i, i + 32768));
    return btoa(text);
  }
  function wav(samples) {
    const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
    const text = (at, value) => { for (let i = 0; i < value.length; i++) view.setUint8(at + i, value.charCodeAt(i)); };
    text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE'); text(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) { const value = Math.max(-1, Math.min(1, samples[i])); view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true); }
    return buffer;
  }
  async function resample(samples, rate) {
    if (rate === 16000) return samples;
    const context = new OfflineAudioContext(1, Math.ceil(samples.length * 16000 / rate), 16000);
    const buffer = context.createBuffer(1, samples.length, rate); buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource(); source.buffer = buffer; source.connect(context.destination); source.start();
    return (await context.startRendering()).getChannelData(0);
  }
  async function sendAudio(buffer, label) {
    if (state.busy || !state.config) return;
    stopSpeech(); showError(); state.inRoom = true; state.stopRequested = false;
    const previous = history(), user = { id: uuid(), role: 'user', content: 'Audio recording', label, status: 'ready' };
    const turn = { id: uuid(), role: 'assistant', content: '', status: 'transcribing' };
    user.audioUrl = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' })); state.urls.add(user.audioUrl);
    state.lastAudio = { buffer, label }; conversations().push(user, turn); state.busy = true; state.activeTurn = turn; state.job = null;
    node('chat-audio-retry').hidden = true; state.scrollHeld = false; render();
    node('chat-scroll').scrollTop = node('chat-scroll').scrollHeight; state.lastScrollTop = node('chat-scroll').scrollTop;
    beginReply(turn); save();
    try {
      const request = { model_id: state.model, ...pronunciationRequest(state.config, state.pronunciationModel),
        request_id: uuid(), audio_base64: base64(buffer), history: previous, topic_id: state.topic };
      if (state.repeatTarget) request.correction_job_id = state.repeatTarget.jobId;
      else request.practice_id = node('chat-mode').value === 'reading' ? node('chat-practice').value : null;
      let job = await api('/api/chat/speech', request);
      state.job = job.id;
      // End can be pressed while the initial request is still in flight.
      // Cancel that same job once its server-owned ID becomes available.
      if (state.stopRequested) job = await api('/api/chat/cancel', { job_id: job.id });
      let pollFailures = 0;
      while (!state.destroyed) {
        turn.content = job.text || ''; turn.status = job.status; turn.metadata = job.metadata; turn.error = job.error?.message; turn.jobId = job.id;
        turn.speech = job.speech; updateUserSpeech(user, job.speech);
        if (['error', 'cancelled'].includes(job.status)) stopSpeech();
        state.liveReply?.update(turn.content, { final: replyReady(job) || terminal(job.status) });
        autoplayReply(turn);
        prefetchSentenceVoice(turn);
        render(); if (terminal(job.status)) break;
        await pause(chatPollDelay(job));
        try { job = await api('/api/chat/jobs/' + job.id); pollFailures = 0; }
        catch (error) { if (++pollFailures >= 3) throw error; await pause(1000); }
      }
    } catch (error) { turn.status = 'error'; turn.error = error.message; stopSpeech(); state.liveReply?.update(turn.content, { final: true }); showError(error.message); }
    finally {
      state.busy = false; state.job = null; state.activeTurn = null;
      if (turn.status === 'error') { node('chat-audio-preview').src = user.audioUrl; node('chat-audio-retry').hidden = false; }
      render(); save();
      autoplayReply(turn);
    }
  }
  async function upload(file) {
    if (!file || state.busy || state.recording || state.decoding) return;
    if (file.size > 25 * 1024 * 1024) { showError('Choose an audio file smaller than 25 MB and no longer than 30 seconds.'); return; }
    if (!AudioContextClass) { showError('Audio decoding is not supported in this browser.'); return; }
    const epoch = state.epoch; state.decoding = true; stopSpeech(); showError(); render();
    let context;
    try {
      context = new AudioContextClass();
      const decoded = await context.decodeAudioData(await file.arrayBuffer());
      if (decoded.duration < .25 || decoded.duration > 30 || decoded.numberOfChannels > 2) throw new Error('Use mono or stereo audio between 0.25 and 30 seconds.');
      const samples = new Float32Array(decoded.length);
      for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
        const values = decoded.getChannelData(channel); for (let i = 0; i < samples.length; i++) samples[i] += values[i] / decoded.numberOfChannels;
      }
      const buffer = wav(await resample(samples, decoded.sampleRate));
      if (epoch === state.epoch && state.visible) { state.decoding = false; await sendAudio(buffer, file.name.slice(0, 120)); }
    } catch (error) { showError(error.message || 'This audio file could not be decoded. Try WAV, MP3, M4A, OGG or FLAC.'); }
    finally { if (context) await context.close().catch(() => {}); state.decoding = false; node('chat-audio-file').value = ''; render(); }
  }
  async function refreshInputDevices() {
    if (!navigator.mediaDevices?.enumerateDevices || !window.isSecureContext) return;
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter(device => device.kind === 'audioinput' && device.deviceId && device.deviceId !== 'default' && device.label);
      const select = node('chat-input-device');
      const seen = new Set();
      const options = [new Option('Micro mặc định', '')];
      for (const device of devices) {
        if (seen.has(device.deviceId)) continue;
        seen.add(device.deviceId);
        options.push(new Option(device.label, device.deviceId));
      }
      if (state.inputDeviceId && seen.size && !seen.has(state.inputDeviceId)) {
        state.inputDeviceId = '';
        state.inputNotice = 'Micro đã chọn không còn khả dụng. Đã chuyển sang micro mặc định.';
      }
      if (state.inputDeviceId && !seen.size) options.push(new Option('Micro đã chọn', state.inputDeviceId));
      state.deviceHint = seen.size ? '' : 'Nhấn nút micro để cấp quyền, sau đó chọn thiết bị.';
      select.replaceChildren(...options);
      select.value = state.inputDeviceId;
      if (!state.recording && !state.pending) render();
    } catch { /* Device labels may be unavailable before permission is granted. */ }
  }
  function inputConstraints(deviceId = state.inputDeviceId) {
    const supported = navigator.mediaDevices.getSupportedConstraints?.() || {};
    return microphoneConstraints(supported, deviceId, node('chat-input-processing').value);
  }
  function releaseMic() {
    clearInterval(state.timer); state.timer = null;
    if (state.processor) { if (state.processor.port) state.processor.port.onmessage = null; else state.processor.onaudioprocess = null; state.processor.disconnect(); }
    state.source?.disconnect(); state.sink?.disconnect(); state.stream?.getTracks().forEach(t => t.stop());
    if (state.context && state.context.state !== 'closed') state.context.close().catch(() => {});
    state.processor = state.source = state.sink = state.stream = state.context = null; state.recording = false; state.stopping = false; state.flush = null;
    state.inputRms = 0; state.clippedBlocks = 0; state.signalHint = '';
    node('chat-input-meter-fill').style.width = '0%';
  }
  async function stopMic(send = false) {
    if (!state.recording || state.stopping) return;
    const epoch = state.epoch;
    state.stopping = true; render();
    if (send && state.processor?.port) await new Promise(resolve => {
      const timer = setTimeout(resolve, 350); state.flush = () => { clearTimeout(timer); resolve(); }; state.processor.port.postMessage('flush');
    });
    if (!state.context) return;
    const rate = state.context.sampleRate, samples = new Float32Array(state.sampleCount); let offset = 0;
    for (const chunk of state.chunks) { samples.set(chunk, offset); offset += chunk.length; }
    releaseMic(); state.chunks = []; render();
    if (send && epoch === state.epoch && state.visible) {
      if (samples.length / rate < .25) { showError('Record at least a short English word or sentence.'); return; }
      state.decoding = true; render();
      try { const buffer = wav(await resample(samples, rate)); state.decoding = false; if (epoch === state.epoch && state.visible) await sendAudio(buffer, 'Microphone recording'); }
      catch (error) { showError(error.message || 'The recording could not be prepared.'); }
      finally { state.decoding = false; render(); }
    }
  }
  async function microphone() {
    if (state.recording) { await stopMic(true); return; }
    if (state.busy || state.pending || state.decoding) return;
    if (!state.micAllowed()) { showError('Stop the recording in the pronunciation workspace before using this microphone.'); return; }
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { showError('Microphone access needs HTTPS or localhost. You can upload an audio file instead.'); return; }
    const epoch = state.epoch; state.pending = true; stopSpeech(); showError(); render();
    try {
      let stream;
      try { stream = await navigator.mediaDevices.getUserMedia(inputConstraints()); }
      catch (error) {
        if (!state.inputDeviceId || !['OverconstrainedError', 'NotFoundError'].includes(error.name)) throw error;
        state.inputDeviceId = '';
        state.inputNotice = 'Micro đã chọn không còn khả dụng. Đang dùng micro mặc định.';
        stream = await navigator.mediaDevices.getUserMedia(inputConstraints(''));
      }
      if (epoch !== state.epoch || !state.visible) { stream.getTracks().forEach(t => t.stop()); return; }
      state.stream = stream;
      stream.getAudioTracks()[0].onended = () => {
        if (state.recording && !state.stopping) {
          stopMic(false);
          showError('Micro bị ngắt kết nối. Hãy chọn lại micro và thu âm lại.');
          refreshInputDevices();
        }
      };
      refreshInputDevices();
      state.context = new AudioContextClass(); await state.context.resume();
      state.source = state.context.createMediaStreamSource(stream); state.sink = state.context.createGain(); state.sink.gain.value = 0;
      state.chunks = []; state.sampleCount = 0; state.inputRms = 0; state.clippedBlocks = 0; state.signalHint = '';
      const collect = input => {
        if (!state.recording || !state.context) return;
        const chunk = new Float32Array(input.subarray(0, Math.max(0, 30 * state.context.sampleRate - state.sampleCount)));
        state.chunks.push(chunk); state.sampleCount += chunk.length;
        if (chunk.length) {
          let power = 0, peak = 0;
          for (const value of chunk) { power += value * value; peak = Math.max(peak, Math.abs(value)); }
          state.inputRms = state.inputRms * .65 + Math.sqrt(power / chunk.length) * .35;
          state.clippedBlocks = state.clippedBlocks * .9 + (peak >= .985 ? 1 : 0);
        }
      };
      if (state.context.audioWorklet && window.AudioWorkletNode) {
        try {
          await state.context.audioWorklet.addModule('/recorder-worklet.js'); state.processor = new AudioWorkletNode(state.context, 'mdd-recorder');
          state.processor.port.onmessage = ({ data }) => { if (data.samples) collect(data.samples); if (data.stopped) state.flush?.(); };
        } catch { state.processor = null; }
      }
      if (!state.processor) {
        state.processor = state.context.createScriptProcessor(4096, 1, 1);
        state.processor.onaudioprocess = event => collect(event.inputBuffer.getChannelData(0));
      }
      if (epoch !== state.epoch || !state.visible) { releaseMic(); return; }
      state.recording = true; state.source.connect(state.processor); state.processor.connect(state.sink); state.sink.connect(state.context.destination);
      const start = performance.now(); state.timer = setInterval(() => {
        const seconds = Math.min(30, Math.floor((performance.now() - start) / 1000)); node('chat-recording-time').textContent = `00:${String(seconds).padStart(2, '0')}`;
        const db = 20 * Math.log10(Math.max(state.inputRms, .00001));
        node('chat-input-meter-fill').style.width = `${Math.max(2, Math.min(100, (db + 55) * 2.5))}%`;
        state.signalHint = performance.now() - start < 1000 ? 'Đang kiểm tra tín hiệu micro…'
          : state.clippedBlocks >= 3 ? 'Âm quá lớn · giảm âm lượng micro'
          : state.inputRms < .009 ? 'Âm hơi nhỏ · đưa micro lại gần'
          : 'Tín hiệu tốt · nhấn nút micro để gửi';
        if (node('chat-mic-status').textContent !== state.signalHint) node('chat-mic-status').textContent = state.signalHint;
        if (seconds >= 30) stopMic(true);
      }, 200);
    } catch (error) { releaseMic(); showError(error.name === 'NotAllowedError' ? 'Hãy cấp quyền micro trong trình duyệt hoặc tải audio lên.' : 'Không mở được micro. Hãy kiểm tra thiết bị hoặc tải audio lên.'); }
    finally { state.pending = false; render(); }
  }
  function revealOpening(turn) {
    if (state.openingPending !== turn) return;
    state.openingPending = null;
    if (state.liveTurnId === turn.id) state.liveReply?.revealAll();
    render();
  }
  function stopSpeech(preserveOpening = null) {
    state.voiceStopped = true; state.speechEpoch++;
    const playback = state.playback; state.playback = null; playback?.tracker?.dispose();
    const activeRow = playback?.row || state.rows.get(playback?.turnId || state.speaking);
    if (activeRow) { activeRow.readUntil = 0; paintText(activeRow, activeRow.shownText || '', null, null, 0); }
    if (state.audioFrame !== null) cancelAnimationFrame(state.audioFrame); state.audioFrame = null;
    state.liveReply?.mute(); cancelBrowserSpeech();
    if (state.recordingAudio) { state.recordingAudio.pause(); state.recordingAudio = null; }
    if (state.audio) { state.audio.pause(); state.audio = null; } state.speaking = null; render();
    if (state.openingPending && state.openingPending !== preserveOpening) revealOpening(state.openingPending);
  }
  function speakText(text, id, turn = null, { opening = false } = {}) {
    if (state.recording || state.pending || state.stopping || state.referenceOperation?.regenerating) return;
    showError();
    stopSpeech(opening ? turn : null);
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) { if (opening) revealOpening(turn); showError('Browser speech is unavailable. Select Kokoro or ai33 to hear a completed reply.'); return; }
    const epoch = state.speechEpoch; let openingTimer;
    if (turn && state.liveTurnId === turn.id) state.liveReply?.revealAll();
    const playback = turn ? { turnId: turn.id, range: null, mode: null } : null;
    state.playback = playback;
    const valid = () => epoch === state.speechEpoch && !state.destroyed && (!playback || state.playback === playback);
    const finish = (completed = false) => {
      if (!valid()) return;
      clearTimeout(openingTimer);
      const row = turn && state.rows.get(turn.id);
      if (row) paintText(row, turn.content, null, null, completed ? text.length : 0);
      state.playback = null; state.speaking = null; render();
    };
    try {
      browserVoice(text, {
        start: () => { if (valid()) { clearTimeout(openingTimer); if (opening) revealOpening(turn); state.speaking = id; render(); } },
        boundary: (index, mode) => {
          if (!valid() || !playback) return;
          const range = window.MDDSpeechHighlight.wordRange(text, index); if (!range) return;
          playback.range = range; playback.mode = mode;
          const row = state.rows.get(turn.id); if (row) paintText(row, turn.content, range, mode);
        },
        end: () => { if (valid() && opening) revealOpening(turn); finish(true); },
        error: () => { if (valid()) { if (opening) revealOpening(turn); finish(); showError('Không phát được giọng đọc. Bạn thử bấm Phát lại.'); } },
      }, turn ? .98 : .92);
      if (opening && state.openingPending === turn) openingTimer = setTimeout(() => {
        if (valid() && state.openingPending === turn) { stopSpeech(); showError('Giọng đọc chưa bắt đầu. Bạn thử Phát lại.'); }
      }, 25000);
    } catch (error) { if (valid()) { if (opening) revealOpening(turn); finish(); showError('Không phát được giọng đọc. Bạn thử bấm Phát lại.'); } }
  }
  function autoplayReply(turn) {
    const revision = turnRevision(turn);
    if (!replyReady(turn) || !turn.content || !state.visible || state.voiceStopped
      || !node('chat-auto-read').checked || !serverVoiceSelected()
      || turn.autoVoiceRevision === revision) return;
    turn.autoVoiceRevision = revision;
    speak(turn);
  }
  function voiceRequestInfo(turn, sentence) {
    const selectionField = sentence ? 'sentenceVoiceSelection' : 'voiceSelection';
    const otherSelection = turn[sentence ? 'voiceSelection' : 'sentenceVoiceSelection'];
    // A turn keeps the voice and speed chosen when its first audio was queued.
    // Changing Settings applies to the next turn, matching the server contract.
    turn[selectionField] ||= otherSelection ? { ...otherSelection }
      : voiceSelection(state.config, currentVoiceProvider(), state.voiceId, state.voiceSpeed);
    // Legacy sessions omitted provider and used AI33 voice IDs. Keep their
    // provider explicit rather than sending an AI33 ID to the new local default.
    turn[selectionField].provider ||= 'ai33';
    const text = sentence ? pronunciationReferenceText(turn.speech) : turn.content;
    const revision = turnRevision(turn), target = sentence ? 'sentence' : 'reply';
    return { text, revision, selection: turn[selectionField],
      key: voiceResourceKey(turn.jobId, revision, target, turn[selectionField], text),
      field: sentence ? 'sentence_voice' : 'voice', turnField: sentence ? 'sentenceVoice' : 'voice',
      keyField: sentence ? 'sentenceVoiceKey' : 'voiceKey' };
  }
  function bufferedSentenceAudio(key, voice) {
    let audio = state.voiceBuffers.get(key);
    if (!audio || audio.src !== new URL(voice.url, location.href).href && audio.src !== voice.url) {
      audio = new Audio(voice.url); audio.preload = 'auto';
      state.voiceBuffers.set(key, audio);
      // Loading a buffer is independent of the currently playing bot reply.
      // It must never call play(), stopSpeech(), or claim the playback slot.
      try { audio.load(); } catch {}
    }
    // Keep a small recent set without ever evicting the audio being played.
    state.voiceBuffers.delete(key); state.voiceBuffers.set(key, audio);
    while (state.voiceBuffers.size > 24) {
      const oldest = [...state.voiceBuffers].find(([otherKey, buffer]) => otherKey !== key && buffer !== state.audio);
      if (!oldest) break;
      state.voiceBuffers.delete(oldest[0]);
    }
    return audio;
  }
  async function requestVoiceResource(turn, sentence = false) {
    if (sentence && !pronunciationReferenceText(turn.speech)) return null;
    const info = voiceRequestInfo(turn, sentence), epoch = state.epoch;
    const current = () => epoch === state.epoch && info.revision === turnRevision(turn) && !state.destroyed
      && !['error', 'cancelled', 'cancelling'].includes(turn.status)
      && voiceRequestInfo(turn, sentence).key === info.key;
    if (turn[info.keyField] === info.key && turn[info.turnField]?.status === 'ready') return turn[info.turnField];
    return sharedVoiceRequest(state.voiceRequests, info.key, async () => {
      if (!current()) return null;
      state.voiceTasks.add(info.key); controls();
      try {
        const deadline = performance.now() + 60000;
        let job;
        while (current()) {
          try {
            job = await api('/api/chat/voice', { job_id: turn.jobId, locale: 'en-US', ...info.selection,
              ...(sentence ? { target: 'sentence' } : {}) });
            break;
          } catch (error) {
            // Another sentence may still hold the bounded background slot.
            // A busy response has not submitted a provider task, so retrying
            // that response cannot create duplicate synthesis jobs.
            if (error.code !== 'chat_voice_busy' || performance.now() >= deadline) throw error;
            await pause(400);
          }
        }
        if (!job || !current() || (job.metadata?.reference_revision || 0) !== info.revision) return null;
        turn[info.keyField] = info.key; turn[info.turnField] = job[info.field]; render();
        while (job[info.field]?.status === 'generating') {
          await pause(info.selection.provider === 'kokoro' ? 150 : 700); if (!current()) return null;
          job = await api('/api/chat/jobs/' + turn.jobId);
          if (!current() || (job.metadata?.reference_revision || 0) !== info.revision) return null;
          turn[info.turnField] = job[info.field]; render();
        }
        const voice = job[info.field];
        if (voice?.status === 'error') throw new Error(voice.message);
        if (voice?.status === 'ready' && /^\/api\/chat\/audio\/[0-9a-f]{32}$/.test(voice.url)) {
          if (sentence) bufferedSentenceAudio(info.key, voice);
          return voice;
        }
        return null;
      } finally { state.voiceTasks.delete(info.key); render(); }
    });
  }
  function prefetchSentenceVoice(turn) {
    if (!serverVoiceSelected() || !state.visible || !state.inRoom || !turn.jobId
      || ['error', 'cancelled', 'cancelling'].includes(turn.status)
      || !pronunciationReferenceText(turn.speech)) return;
    const info = voiceRequestInfo(turn, true);
    if (turn.sentencePrefetchKey === info.key) return;
    turn.sentencePrefetchKey = info.key;
    // Read aloud controls autoplay, not preparation of this opt-in sound-detail
    // button. Browser voice does not request any provider audio.
    requestVoiceResource(turn, true).catch(() => {});
  }
  async function speak(turn, sentence = false, { opening = false } = {}) {
    if (state.recording || state.pending || state.stopping || (state.referenceOperation?.regenerating && !replyReady(turn))) return;
    const text = sentence ? pronunciationReferenceText(turn.speech) : turn.content;
    const playId = sentence ? 'sentence-' + turn.id : turn.id;
    if (!text) return;
    if (!serverVoiceSelected()) { speakText(text, playId, sentence ? null : turn, { opening }); return; }
    if (!turn.jobId || (sentence ? turn.status !== 'ready' : !replyReady(turn))) { showError('Giọng đọc cần câu trả lời hoàn tất trong phiên hiện tại.'); return; }
    const info = voiceRequestInfo(turn, sentence), revision = info.revision;
    showError();
    stopSpeech(opening ? turn : null); const model = state.model, epoch = state.speechEpoch;
    const current = () => epoch === state.speechEpoch && revision === turnRevision(turn) && !state.destroyed
      && (!opening || state.visible && state.inRoom && state.model === model && conversations().includes(turn));
    try {
      const voice = await requestVoiceResource(turn, sentence);
      if (!current()) return;
      if (voice?.status === 'ready' && /^\/api\/chat\/audio\/[0-9a-f]{32}$/.test(voice.url) && state.visible && state.model === model && epoch === state.speechEpoch) {
        // Provider audio has no word timestamps. Follow the real
        // playhead using an explicitly estimated word map, for the exact slice
        // synthesized by ChatService._synthesize (not the rest of a long reply).
        if (!sentence && state.liveTurnId === turn.id) state.liveReply?.revealAll();
        const row = sentence ? state.rows.get(turn.id)?.sentenceRow : state.rows.get(turn.id);
        if (row && !opening) paintText(row, text, null, null, 0);
        const audio = sentence ? bufferedSentenceAudio(info.key, voice) : new Audio(voice.url);
        if (sentence) audio.currentTime = 0;
        const playback = { turnId: playId, row, range: null, mode: null };
        state.audio = audio; state.playback = playback;
        const valid = () => current() && state.audio === audio && state.playback === playback;
        const spokenText = [...text].slice(0, 5000).join('');
        const tracker = new window.MDDSpeechHighlight.SpeechWordTracker(spokenText, {
          onUpdate: (range, mode) => {
            if (!valid()) return;
            playback.range = range; playback.mode = mode;
            if (row) paintText(row, text, range, mode);
          },
        });
        playback.tracker = tracker;
        const clearFrame = () => { if (state.audioFrame !== null) cancelAnimationFrame(state.audioFrame); state.audioFrame = null; };
        const updateTime = () => { if (valid() && !audio.paused) tracker.audioTime(audio.currentTime, audio.duration); };
        const tick = () => { if (!valid() || audio.paused) return; updateTime(); state.audioFrame = requestAnimationFrame(tick); };
        const finish = (completed = false) => {
          if (!valid()) return;
          clearFrame(); tracker.dispose(); state.playback = null; state.audio = null; state.speaking = null; render();
          if (row) paintText(row, text, null, null, completed ? spokenText.length : 0);
        };
        audio.onplaying = () => {
          if (!valid()) return;
          clearFrame(); tracker.start(); state.speaking = playId; updateTime(); render(); tick();
        };
        audio.ontimeupdate = updateTime; audio.onseeked = updateTime;
        audio.onpause = () => { if (valid()) { clearFrame(); tracker.end(); state.speaking = null; render(); } };
        audio.onended = () => finish(true);
        audio.onerror = () => { if (valid()) { finish(); showError('Không phát được giọng đọc. Bạn thử Phát lại.'); } };
        if (opening) {
          if (!await prepareOpeningAudio(audio, { isCurrent: valid })) return;
          revealOpening(turn);
        }
        await audio.play();
      }
    } catch (error) { if (current()) { stopSpeech(); showError(error.message || 'The English voice could not be played.'); } }
    finally { render(); }
  }
  async function cancel() {
    if (!state.activeTurn) return;
    state.stopRequested = true;
    stopSpeech(); state.activeTurn.status = 'cancelling'; render();
    if (!state.job) return;
    try { await api('/api/chat/cancel', { job_id: state.job }); } catch (error) { showError(error.message); }
  }
  function endRoom() {
    state.epoch++; state.inRoom = false; state.repeatTarget = null;
    node('chat-settings').open = false;
    releaseMic(); state.chunks = [];
    stopSpeech(); clearReply(); showError();
    if (state.busy) cancel();
    render(); save();
  }
  async function initialize() {
    try {
      state.config = await api('/api/chat/config');
      const previousSession = sessionKey();
      if (!state.config.topics?.some(t => t.id === state.topic)) state.topic = null;
      if (!state.config.models.some(m => m.id === state.model && m.ready)) state.model = 'qwen-base';
      if (previousSession !== sessionKey()) resetRows();
      node('chat-model').replaceChildren(...state.config.models.map(model => { const option = make('option', '', (model.id === 'qwen-trained' ? 'Qwen · Fine-tuned' : 'Qwen · Original') + (model.ready ? '' : ' · Unavailable')); option.value = model.id; option.disabled = !model.ready; return option; }));
      node('chat-model').value = state.model;
      const phonemeModels = pronunciationModels(state.config), phonemeSelect = node('chat-pronunciation-model');
      state.pronunciationModel = preferredPronunciationModel(state.config, state.pronunciationModel);
      if (phonemeSelect) {
        phonemeSelect.hidden = !phonemeModels.length;
        node('chat-pronunciation-model-label').hidden = !phonemeModels.length;
        phonemeSelect.replaceChildren(...phonemeModels.map(model => {
          const option = make('option', '', model.label + (model.ready ? '' : ' · Chưa sẵn sàng'));
          option.value = model.id; option.disabled = !model.ready; return option;
        }));
        phonemeSelect.value = state.pronunciationModel;
      }
      node('chat-practice').replaceChildren(...state.config.practice_prompts.map(prompt => { const option = make('option', '', prompt.text); option.value = prompt.id; return option; }));
      const voice = node('chat-voice-provider').querySelector('option[value="ai33"]'); voice.disabled = !state.config.ai33_available || !state.config.voice_locales.includes('en-US');
      voice.textContent = voice.disabled ? 'ai33 · Unavailable' : state.config.english_voice_name || 'ai33 · Full reply';
      const kokoro = node('chat-voice-provider').querySelector('option[value="kokoro"]'); kokoro.disabled = !state.config.kokoro_available;
      kokoro.textContent = kokoro.disabled ? 'Kokoro local · Unavailable' : 'Kokoro local · English';
      if (state.config.kokoro_available && state.voicePreference !== 'kokoro') state.voiceId = '';
      node('chat-voice-provider').value = preferredVoiceProvider(state.config, state.voicePreference, state.voicePreferenceVersion);
      if (state.config.kokoro_available) state.voicePreferenceVersion = 1;
      renderVoiceOptions(); node('chat-voice-speed').value = state.voiceSpeed;
      if (!state.config.speech_ready) showError('Local speech recognition is not ready on the server.');
      if (!window.isSecureContext) node('chat-mic-status').textContent = 'Microphone access needs HTTPS. Audio uploads are available here.';
      renderTopics(); render(); save();
    } catch (error) { showError(error.message); }
  }
  const readingActions = make('div', 'chat-reading-actions'), clearRepeat = make('button', 'text-button chat-repeat-clear', 'Back to conversation');
  clearRepeat.id = 'chat-repeat-clear'; clearRepeat.type = 'button'; clearRepeat.hidden = true;
  clearRepeat.addEventListener('click', () => {
    if (state.busy || state.recording || state.pending || state.decoding || state.stopping || state.starting) return;
    stopSpeech(); state.repeatTarget = null; render();
  });
  node('chat-reading-card').append(readingActions);
  readingActions.append(node('chat-reading-listen'), clearRepeat);
  node('chat-mic').addEventListener('click', microphone);
  window.addEventListener('focus', refreshInputDevices);
  node('chat-input-device').addEventListener('change', event => {
    state.inputDeviceId = event.target.value;
    state.inputNotice = state.inputDeviceId ? 'Đã chọn micro. Nhấn nút để bắt đầu thu âm.' : '';
    render();
  });
  node('chat-input-processing').addEventListener('change', () => {
    state.inputNotice = node('chat-input-processing').value === 'raw'
      ? 'Ít xử lý phù hợp khi dùng tai nghe và ở nơi yên tĩnh.' : 'Lọc ồn và vọng loa sẽ áp dụng nếu trình duyệt hỗ trợ.';
    render();
  });
  navigator.mediaDevices?.addEventListener?.('devicechange', refreshInputDevices);
  node('chat-scroll').addEventListener('wheel', event => { if (event.deltaY < 0) state.scrollHeld = true; }, { passive: true });
  node('chat-scroll').addEventListener('scroll', () => {
    const scroll = node('chat-scroll');
    if (scroll.scrollTop < state.lastScrollTop - 1) state.scrollHeld = true;
    if (scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <= 2) state.scrollHeld = false;
    state.lastScrollTop = scroll.scrollTop;
  }, { passive: true });
  node('chat-upload').addEventListener('click', () => node('chat-audio-file').click());
  node('chat-audio-file').addEventListener('change', event => upload(event.target.files[0]));
  node('chat-retry-audio').addEventListener('click', () => { if (state.lastAudio) sendAudio(state.lastAudio.buffer, state.lastAudio.label); });
  node('chat-stop').addEventListener('click', cancel);
  node('chat-stop-voice').addEventListener('click', stopSpeech);
  node('chat-end').addEventListener('click', endRoom);
  node('chat-model').addEventListener('change', () => {
    if (state.busy || state.recording) return; stopSpeech(); clearReply(); state.repeatTarget = null; state.model = node('chat-model').value; resetRows(); render(); save();
    if (state.inRoom && state.topic && !conversations().length) startTopic(state.topic);
  });
  node('chat-pronunciation-model')?.addEventListener('change', () => {
    if (state.busy || state.recording || state.pending || state.decoding || state.stopping || state.starting) return;
    state.pronunciationModel = preferredPronunciationModel(state.config, node('chat-pronunciation-model').value);
    node('chat-pronunciation-model').value = state.pronunciationModel;
    controls(); save();
  });
  node('chat-mode').addEventListener('change', () => openFree(node('chat-mode').value));
  node('chat-practice').addEventListener('change', () => { stopSpeech(); state.repeatTarget = null; render(); });
  node('chat-read-prompt').addEventListener('click', () => speakText(node('chat-practice-text').textContent, 'prompt'));
  node('chat-reading-listen').addEventListener('click', () => speakText(node('chat-reading-text').textContent, 'prompt'));
  node('chat-voice-provider').addEventListener('change', () => {
    stopSpeech(); state.voicePreference = currentVoiceProvider();
    if (state.config?.kokoro_available) state.voicePreferenceVersion = 1;
    renderVoiceOptions(); controls(); save();
  });
  node('chat-english-voice').addEventListener('change', () => { stopSpeech(); state.voiceId = node('chat-english-voice').value; controls(); save(); });
  node('chat-voice-speed').addEventListener('change', () => { stopSpeech(); state.voiceSpeed = Number(node('chat-voice-speed').value); save(); });
  node('topic-voice-search').addEventListener('input', renderSetupVoices);
  for (const button of document.querySelectorAll('[data-voice-locale]')) button.addEventListener('click', () => {
    state.voiceFilter = button.dataset.voiceLocale;
    for (const other of document.querySelectorAll('[data-voice-locale]')) other.setAttribute('aria-pressed', String(other === button));
    renderSetupVoices();
  });
  for (const radio of document.querySelectorAll('[name=topic-speed]')) radio.addEventListener('change', () => {
    state.setupSpeed = Number(radio.value); if (state.previewAudio) state.previewAudio.playbackRate = state.setupSpeed; setupSelection();
  });
  node('topic-setup-close').addEventListener('click', () => node('chat-topic-setup').close());
  node('chat-topic-setup').addEventListener('click', event => { if (event.target === node('chat-topic-setup')) node('chat-topic-setup').close(); });
  node('chat-topic-setup').addEventListener('close', () => { stopVoicePreview(); state.pendingTopic = null; });
  node('topic-setup-start').addEventListener('click', () => {
    const topic = state.pendingTopic; if (!topic) return;
    node('chat-voice-provider').value = state.setupVoiceProvider;
    state.voicePreference = state.setupVoiceProvider;
    if (state.config?.kokoro_available) state.voicePreferenceVersion = 1;
    if (state.setupVoiceId !== 'browser') state.voiceId = state.setupVoiceId;
    state.voiceSpeed = state.setupSpeed; renderVoiceOptions(); node('chat-voice-speed').value = state.voiceSpeed;
    node('chat-auto-read').checked = node('topic-setup-auto-read').checked;
    stopVoicePreview(); node('chat-topic-setup').close(); save(); startTopic(topic);
  });
  node('chat-auto-read').addEventListener('change', () => { if (!node('chat-auto-read').checked) stopSpeech(); });
  function newConversation() {
    if (state.busy || state.recording || state.pending || state.decoding || state.stopping || state.starting) return;
    node('chat-settings').open = false; state.scrollHeld = false; state.lastScrollTop = 0;
    if (state.topic) { startTopic(state.topic, true); return; }
    stopSpeech(); clearReply(); state.repeatTarget = null; state.conversations[sessionKey()] = []; resetRows();
    node('chat-audio-retry').hidden = true; state.lastAudio = null; showError(); render(); save();
  }
  node('chat-new').addEventListener('click', newConversation);
  node('chat-start-new').addEventListener('click', newConversation);
  node('chat-export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify({ model: state.model, topic_id: state.topic, language: 'en-US', messages: conversations() }, (key, value) => key === 'audioUrl' ? undefined : value, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob), link = make('a'); link.href = url; link.download = 'english-voice-practice.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.querySelectorAll('[data-speech-mode]').forEach(button => button.addEventListener('click', () => {
    openFree(button.dataset.speechMode);
    node('chat-settings').open = button.dataset.speechMode === 'reading'; render(); node('chat-mic').focus();
  }));
  node('chat-free-talk').addEventListener('click', () => openFree());
  node('chat-open-reading').addEventListener('click', () => openFree('reading'));
  node('chat-topics').addEventListener('click', () => {
    if (state.busy || state.recording || state.starting) return;
    stopSpeech(); clearReply(); state.repeatTarget = null; state.inRoom = false; render(); save();
  });
  document.addEventListener('pointerdown', event => { if (!node('chat-settings').contains(event.target)) node('chat-settings').open = false; });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') node('chat-settings').open = false; });
  window.MDDChat = { state, upload, sendAudio, stopMic, getTurns: conversations, onViewChange(name, micAllowed) {
    state.visible = name === 'chat'; state.micAllowed = micAllowed;
    if (!state.visible) {
      if (node('chat-topic-setup').open) node('chat-topic-setup').close(); stopVoicePreview();
      state.epoch++; state.repeatTarget = null; releaseMic(); state.chunks = []; stopSpeech(); clearReply();
    }
    render();
  } };
  window.addEventListener('beforeunload', () => {
    state.destroyed = true; state.epoch++; stopSpeech(); state.liveReply?.dispose(); releaseMic(); for (const url of state.urls) URL.revokeObjectURL(url);
    if (state.job) fetch('/api/chat/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_id: state.job }), keepalive: true }).catch(() => {});
  });
  restore(); render(); initialize(); refreshInputDevices();
  if (window.location.hash === '#chat' && typeof window.switchView === 'function') window.switchView('chat');
})();
