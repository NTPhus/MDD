'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  models: [], prompts: [], training: [], capabilities: {}, modelId: new URLSearchParams(window.location.search).get('model') || '', promptId: '', sampleId: null,
  audioBase64: null, audioSamples: null, audioRate: 0, audioUrl: null,
  recording: false, recordingPending: false, busy: false, loadingAudio: false,
  context: null, stream: null, source: null, processor: null, silentGain: null,
  chunks: [], sampleCount: 0, timer: null, startedAt: 0, result: null,
  statusPending: false, toastTimer: null, audioRevision: 0,
  resultRevision: 0, guidanceRevision: 0, guidancePending: false,
  guidanceController: null, guidanceTimer: null, guidanceTimeout: null,
  wordTooltip: null, wordTooltipTimer: null, wordTooltipSuppressFocus: null,
  practiceMode: 'sentence', wordIndex: 0, wordResults: {}, liveEnabled: true,
  liveActive: false, liveGeneration: 0, livePromise: null, liveController: null,
  liveLastSample: 0, liveTimeout: null, finalizing: false, stopping: false,
  recordingLimit: 30, lastVoiceSample: 0, voicedSamples: 0, flushRecording: null,
  referenceInfo: null, referenceRevision: 0, referenceTimer: null, referenceController: null,
};
const AudioContextClass = window.AudioContext || window.webkitAudioContext;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function finite(value) { return value !== null && value !== undefined && Number.isFinite(Number(value)); }
function displayNumber(value, digits = 3) { return finite(value) ? Number(value).toLocaleString('vi-VN', { maximumFractionDigits: digits }) : '—'; }
function timeLabel(seconds) {
  const duration = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(duration / 60)).padStart(2, '0')}:${String(duration % 60).padStart(2, '0')}`;
}

async function request(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  let result;
  try { result = await response.json(); }
  catch { throw new Error('Máy chủ không trả về dữ liệu hợp lệ. Hãy kiểm tra cửa sổ chạy demo.'); }
  if (!response.ok) {
    const error = new Error(result.message || result.error || `Không thể hoàn thành yêu cầu (${response.status}).`);
    error.code = result.error; error.status = response.status;
    throw error;
  }
  return result;
}

function showAudioError(message = '') {
  $('audio-error').textContent = message;
  $('audio-error').hidden = !message;
}

function toast(message) {
  clearTimeout(state.toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  state.toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
}

function switchView(name) {
  if (!['practice', 'training', 'chat'].includes(name)) return;
  if (name !== 'practice') hideWordTooltip();
  $('practice-view').hidden = name !== 'practice';
  $('training-view').hidden = name !== 'training';
  $('chat-view').hidden = name !== 'chat';
  $('breadcrumb-current').textContent = { practice: 'Luyện phát âm', training: 'Huấn luyện', chat: 'Đối thoại AI' }[name];
  window.MDDChat?.onViewChange(name, () => !state.recording && !state.recordingPending);
  document.querySelectorAll('.nav-item').forEach((button) => {
    const active = button.dataset.view === name;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page');
    else button.removeAttribute('aria-current');
  });
  if (name === 'training') {
    renderTraining();
    refreshStatus();
  } else if (name === 'practice') drawWaveform(state.audioSamples);
}

function microphoneUnavailableReason() {
  if (window.isSecureContext === false) return 'Thu microphone trên thiết bị khác cần HTTPS. Bạn vẫn có thể tải âm thanh lên hoặc dùng bản thu mẫu.';
  if (!AudioContextClass) return 'Trình duyệt chưa hỗ trợ xử lý âm thanh. Hãy dùng Chrome, Edge hoặc Firefox.';
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') return 'Microphone chưa khả dụng trong trình duyệt này. Hãy dùng HTTPS hoặc localhost; bạn vẫn có thể tải âm thanh lên hoặc dùng bản thu mẫu.';
  return '';
}

function practicePrompt() {
  const key = (value) => String(value).normalize('NFKC').toLowerCase().trim().replace(/\s+/g, ' ');
  return state.prompts.find((prompt) => key(prompt.text) === key($('reference-text').value));
}

function practiceAvailable() {
  const model = state.models.find((item) => item.id === state.modelId);
  const prompt = practicePrompt();
  const custom = state.referenceInfo?.reference_text === $('reference-text').value.trim() && state.referenceInfo?.available;
  return state.capabilities.realtime === true && model && model.ready !== false
    && model.kind !== 'backbone' && !model.experimental && ((prompt && prompt.live_available !== false) || custom);
}

function resolveCustomReference() {
  const revision = ++state.referenceRevision;
  clearTimeout(state.referenceTimer);
  if (state.referenceController) state.referenceController.abort();
  state.referenceInfo = null;
  const text = $('reference-text').value.trim();
  if (practicePrompt() || !text || state.capabilities.realtime !== true) return;
  state.referenceTimer = setTimeout(async () => {
    const controller = new AbortController();
    state.referenceController = controller;
    try {
      const info = await request('/api/reference', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference_text: text }), signal: controller.signal });
      if (revision !== state.referenceRevision) return;
      state.referenceInfo = info;
    } catch (error) {
      if (revision !== state.referenceRevision || error.name === 'AbortError') return;
      state.referenceInfo = { reference_text: text, available: false, missing_words: [] };
    } finally { if (revision === state.referenceRevision) updateControls(); }
  }, 350);
}

function renderPractice() {
  const blocked = state.busy || state.loadingAudio || state.recordingPending || state.recording || state.finalizing;
  const available = practiceAvailable();
  for (const mode of ['sentence', 'word']) {
    const button = $(`mode-${mode}`);
    button.disabled = blocked || (mode === 'word' && !available);
    button.classList.toggle('selected', state.practiceMode === mode);
    button.setAttribute('aria-pressed', String(state.practiceMode === mode));
  }
  $('live-enabled').disabled = blocked || !available;
  $('word-practice').hidden = state.practiceMode !== 'word';
  $('practice-mode-hint').textContent = !available
    ? state.referenceInfo?.missing_words?.length
      ? `Chưa có âm chuẩn cho: ${state.referenceInfo.missing_words.join(', ')}. Sửa từ hoặc chọn một câu mẫu để dùng realtime.`
      : 'Realtime và từng từ cần model A/B/C cùng câu có âm chuẩn. Câu tự nhập sẽ được tra từ điển tiếng Anh Mỹ.'
    : !state.liveEnabled ? 'Phân tích sau khi dừng thu. Bật cập nhật trực tiếp để xem kết quả trong lúc đọc.'
      : 'Cập nhật khoảng mỗi giây cộng thời gian xử lý. Kết quả tạm thời có thể đổi; phần chưa đọc ở trạng thái chờ.';
  const words = $('reference-text').value.trim().split(/\s+/).filter(Boolean);
  state.wordIndex = Math.max(0, Math.min(state.wordIndex, words.length - 1));
  const signature = JSON.stringify([words, state.wordIndex, state.wordResults, blocked]);
  if ($('practice-words').dataset.signature !== signature) {
    $('practice-words').replaceChildren();
    words.forEach((word, index) => {
      const button = element('button', `practice-word ${state.wordResults[index] || ''}${index === state.wordIndex ? ' selected' : ''}`, word);
      button.type = 'button'; button.disabled = blocked;
      button.setAttribute('aria-pressed', String(index === state.wordIndex));
      button.setAttribute('aria-label', `Luyện từ ${index + 1}: ${word}`);
      button.addEventListener('click', () => choosePracticeWord(index));
      $('practice-words').append(button);
    });
    $('practice-words').dataset.signature = signature;
  }
  $('word-target-label').textContent = words.length ? `Từ ${state.wordIndex + 1}/${words.length}: ${words[state.wordIndex]}` : 'Chọn một câu để luyện';
  $('word-next').disabled = blocked || state.wordIndex >= words.length - 1;
}

function choosePracticeWord(index) {
  if (state.recording || state.busy || state.finalizing || state.recordingPending) return;
  state.wordIndex = index;
  clearAudio();
}

function setPracticeMode(mode) {
  if (state.recording || state.busy || state.finalizing || state.recordingPending || !['sentence', 'word'].includes(mode)) return;
  if (mode === 'word' && !practiceAvailable()) return;
  state.practiceMode = mode; state.wordIndex = 0; state.wordResults = {};
  clearAudio();
}

function cancelLive() {
  state.liveGeneration += 1;
  state.liveActive = false;
  if (state.liveController) state.liveController.abort();
  state.liveController = null;
  clearTimeout(state.liveTimeout);
  state.livePromise = null;
  $('live-status').hidden = true;
}

function recordingSamples() {
  const samples = new Float32Array(state.sampleCount);
  let offset = 0;
  for (const chunk of state.chunks) { samples.set(chunk, offset); offset += chunk.length; }
  return samples;
}

function livePayload(audio, final) {
  return { model_id: state.modelId, reference_text: $('reference-text').value.trim(),
    mode: state.practiceMode, word_index: state.wordIndex, final, audio_base64: audio };
}

function pumpLive() {
  if (!state.recording || state.stopping || !state.liveActive || state.livePromise) return;
  const rate = state.context.sampleRate;
  if (state.sampleCount < rate * 0.8 || (state.liveLastSample && state.sampleCount - state.liveLastSample < rate * 1.2)) return;
  state.liveLastSample = state.sampleCount;
  const generation = state.liveGeneration;
  const controller = new AbortController();
  state.liveController = controller;
  state.liveTimeout = setTimeout(() => controller.abort(), 30000);
  $('live-status').hidden = false;
  $('live-status').textContent = 'Đang cập nhật phần vừa đọc…';
  const payload = livePayload(base64Buffer(encodeWav(recordingSamples(), rate)), false);
  state.livePromise = request('/api/analyze-live', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload), signal: controller.signal }).then((result) => {
    if (generation !== state.liveGeneration || !state.recording || state.stopping) return;
    state.result = result;
    renderResult(result);
    $('live-status').textContent = 'Đang đọc · Kết quả tạm thời; phần xám chưa được đọc.';
  }).catch((error) => {
    if (generation !== state.liveGeneration || !state.recording || state.stopping) return;
    if (error.code === 'reference_unavailable' || error.code === 'live_model_unavailable' || error.code === 'invalid_word_reference') {
      state.liveActive = false;
      $('live-status').textContent = error.message;
    } else $('live-status').textContent = error.code === 'busy'
      ? 'Model đang bận · sẽ lấy đoạn mới nhất, không xếp hàng các đoạn cũ.'
      : error.code === 'inference_failed' ? 'Đang chờ tiếng nói rõ hơn…'
        : 'Chưa cập nhật được · vẫn đang thu, sẽ thử lại với đoạn mới nhất.';
  }).finally(() => {
    if (generation !== state.liveGeneration) return;
    clearTimeout(state.liveTimeout);
    state.liveController = null; state.livePromise = null;
  });
}

function updateControls() {
  const model = state.models.find((item) => item.id === state.modelId);
  const blocked = state.busy || state.loadingAudio || state.recordingPending || state.finalizing;
  const microphoneReason = microphoneUnavailableReason();
  $('record-button').disabled = blocked || Boolean(microphoneReason) || (state.practiceMode === 'word' && !practiceAvailable());
  if (microphoneReason) {
    $('record-hint').textContent = microphoneReason;
    $('record-label').textContent = window.isSecureContext === false ? 'Thu microphone cần HTTPS' : 'Microphone chưa khả dụng';
    $('record-button').title = microphoneReason;
  } else $('record-button').removeAttribute('title');
  $('upload-button').disabled = blocked || state.recording || !AudioContextClass;
  $('sample-button').disabled = blocked || state.recording || !state.prompts.length || state.practiceMode === 'word';
  $('shuffle-prompt').disabled = blocked || state.recording || !state.prompts.length;
  $('prompt-select').disabled = blocked || state.recording;
  $('reference-text').disabled = blocked || state.recording;
  $('model-select').disabled = blocked || state.recording || !state.models.length;
  $('analyze-button').disabled = blocked || state.recording || !model || model.ready === false || !(state.audioBase64 || state.sampleId)
    || (state.practiceMode === 'word' && !practiceAvailable());
  $('analyze-button').querySelector('span').textContent = state.busy ? 'Đang phân tích…' : 'Phân tích bản thu';
  renderPractice();
}

function modelPhase(model) {
  if (['pretrained', 'source', 'adapted_vi', 'control', 'technical'].includes(model.training_phase)) return model.training_phase;
  if (model.kind === 'backbone') return 'pretrained';
  if (model.experimental || model.purpose === 'technical_smoke_only') return 'technical';
  const stage = model.stage || model.id?.match(/-([ABC])(?:-fold\d+)?:(?:best|latest)$/)?.[1];
  return ({ A: 'source', B: 'adapted_vi', C: 'control' })[stage] || 'other';
}

const MODEL_GROUPS = {
  pretrained: 'Trước fine-tune MDD · BASE (chỉ đặc trưng)',
  source: 'A · Trước thích nghi tiếng Việt (đã train MDD)',
  adapted_vi: 'B · Sau thích nghi tiếng Việt',
  control: 'C · Đối chứng source-only',
  technical: 'Kiểm tra kỹ thuật · smoke',
  other: 'Checkpoint khác',
};

function modelDescription(model) {
  if (!model) return 'Chưa tìm thấy mô hình sẵn sàng. Hãy kiểm tra thư mục model và checkpoint.';
  if (model.ready === false) return model.warning || 'Mô hình chưa sẵn sàng để phân tích.';
  if (model.kind === 'backbone') return 'BASE trước fine-tune MDD · Chỉ trích xuất đặc trưng âm thanh, chưa có đầu phone-CTC để chỉ lỗi phát âm.';
  if (model.experimental) return 'Checkpoint thử nghiệm · Có thể xem âm vị, chưa dùng để đánh giá phát âm.';
  if (modelPhase(model) === 'source') return 'A đã huấn luyện MDD trên source · Mốc trước thích nghi tiếng Việt để so với B.';
  if (modelPhase(model) === 'adapted_vi') return 'B đã thích nghi tiếng Việt · Chọn cùng bản thu và câu tham chiếu để so với A trước thích nghi.';
  if (modelPhase(model) === 'control') return 'C là đối chứng source-only · Dùng để đối chiếu với B sau thích nghi tiếng Việt.';
  if (model.warning) return model.warning;
  return 'Mô hình phone-CTC · Đối chiếu âm vị khi có câu tham chiếu trong bộ dữ liệu.';
}

function renderModels(models, defaultId) {
  const previousId = state.modelId;
  state.models = Array.isArray(models) ? models : [];
  const select = $('model-select');
  const signature = JSON.stringify(state.models.map(({ id, label, ready, step, training_phase, phase_label, stage, kind, experimental, purpose }) => ({ id, label, ready, step, training_phase, phase_label, stage, kind, experimental, purpose })));
  if (select.dataset.signature !== signature) {
    select.replaceChildren();
    for (const [phase, label] of Object.entries(MODEL_GROUPS)) {
      const modelsInGroup = state.models.filter((model) => modelPhase(model) === phase);
      if (!modelsInGroup.length) continue;
      const group = element('optgroup');
      group.label = label;
      group.dataset.phase = phase;
      for (const model of modelsInGroup) {
        const name = model.label || model.id;
        const option = element('option', '', `${name}${model.ready === false ? ' · chưa sẵn sàng' : ''}`);
        option.value = model.id;
        option.disabled = model.ready === false;
        group.append(option);
      }
      select.append(group);
    }
    if (!state.models.length) {
      const option = element('option', '', 'Chưa có mô hình local');
      option.value = '';
      select.append(option);
    }
    select.dataset.signature = signature;
  }
  const selected = state.models.find((model) => model.id === previousId && model.ready !== false)
    || state.models.find((model) => model.id === defaultId && model.ready !== false)
    || state.models.find((model) => model.ready !== false);
  state.modelId = selected ? selected.id : '';
  if (previousId && previousId !== state.modelId) clearResult();
  select.value = state.modelId;
  $('model-note').textContent = modelDescription(selected);
  updateControls();
}

async function refreshStatus() {
  if (state.statusPending) return;
  state.statusPending = true;
  try {
    const result = await request('/api/status');
    const previouslyLive = state.capabilities.realtime;
    state.capabilities = result.capabilities || {};
    if (!previouslyLive && state.capabilities.realtime) resolveCustomReference();
    renderModels(result.models, result.default_model_id);
    state.training = Array.isArray(result.training) ? result.training : [];
    $('connection-error').hidden = true;
    $('connection-dot').classList.add('online');
    $('sidebar-system').textContent = `${state.models.filter((model) => model.ready !== false).length} mô hình · CPU inference`;
    const system = result.system || {};
    const gpu = system.gpu || system.gpu_name || system.cuda_device;
    $('training-device').textContent = typeof gpu === 'string' ? gpu : 'Huấn luyện trên máy';
    $('last-updated').textContent = `Lần đọc gần nhất ${new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
    $('training-count').textContent = state.training.length;
    $('training-count').hidden = !state.training.length;
    const latest = state.training[0];
    $('training-peek-text').textContent = latest
      ? `${latest.run_id} · ${displayNumber(latest.step, 0)} / ${displayNumber(latest.max_steps, 0)} bước · ${statusLabel(latest.status)}`
      : 'Chưa có phiên chạy. Các checkpoint sẽ tự xuất hiện khi được tạo.';
    if (!$('training-view').hidden) renderTraining();
  } catch (error) {
    $('connection-error').textContent = `Chưa kết nối được demo local. ${error.message}`;
    $('connection-error').hidden = false;
    $('connection-dot').classList.remove('online');
    $('sidebar-system').textContent = 'Đang chờ máy chủ local';
    $('training-peek-text').textContent = 'Chưa đọc được trạng thái. Kiểm tra cửa sổ máy chủ demo.';
    $('last-updated').textContent = 'Mất kết nối · đang thử lại';
  } finally { state.statusPending = false; }
}

async function loadPrompts() {
  try {
    const result = await request('/api/prompts');
    state.prompts = Array.isArray(result.prompts) ? result.prompts : [];
    const select = $('prompt-select');
    for (const prompt of state.prompts) {
      const label = prompt.text.length > 85 ? `${prompt.text.slice(0, 82)}…` : prompt.text;
      const option = element('option', '', label);
      option.value = prompt.id;
      select.append(option);
    }
    if (state.prompts.length) choosePrompt(state.prompts[0].id);
  } catch (error) {
    $('prompt-select').querySelector('option').textContent = 'Chưa tải được câu mẫu · bạn vẫn có thể tự nhập';
    showAudioError(`Chưa tải được bản thu mẫu. ${error.message}`);
  }
  updateControls();
}

function choosePrompt(id) {
  const prompt = state.prompts.find((item) => item.id === id);
  state.promptId = prompt ? prompt.id : '';
  state.wordIndex = 0; state.wordResults = {};
  $('prompt-select').value = state.promptId;
  if (prompt) $('reference-text').value = prompt.text;
  resolveCustomReference();
  updateReferenceCount();
  if (state.sampleId && state.sampleId !== state.promptId) clearAudio();
  if (state.practiceMode === 'word' && state.audioSamples) clearAudio();
}

function updateReferenceCount() {
  $('character-count').textContent = `${$('reference-text').value.length} ký tự`;
}

function clearResult() {
  cancelLive();
  hideWordTooltip();
  state.resultRevision += 1;
  clearGuidance();
  state.result = null;
  $('result-content').hidden = true;
  $('result-loading').hidden = true;
  $('result-empty').hidden = false;
  $('save-result').hidden = true;
}

function clearAudio() {
  state.audioRevision += 1;
  state.audioBase64 = null;
  state.sampleId = null;
  state.audioSamples = null;
  state.audioRate = 0;
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = null;
  $('audio-preview').pause();
  $('audio-preview').removeAttribute('src');
  $('audio-preview').hidden = true;
  $('audio-state').textContent = 'Sẵn sàng lắng nghe';
  $('recording-time').textContent = '00:00';
  drawWaveform(null);
  clearResult();
  updateControls();
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => { for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i)); };
  writeText(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, 'WAVE'); writeText(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeText(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return buffer;
}

function base64Buffer(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let start = 0; start < bytes.length; start += 32768) binary += String.fromCharCode(...bytes.subarray(start, start + 32768));
  return btoa(binary);
}

function setAudio(samples, rate, label, sampleId = null) {
  if (!samples.length) throw new Error('Bản thu chưa có âm thanh. Hãy thử thu lại.');
  if (samples.length / rate > 30.05) throw new Error('Bản thu dài hơn 30 giây. Hãy chọn một đoạn ngắn hơn.');
  clearAudio();
  const wav = encodeWav(samples, rate);
  state.audioSamples = samples;
  state.audioRate = rate;
  state.audioBase64 = sampleId ? null : base64Buffer(wav);
  state.sampleId = sampleId;
  state.audioUrl = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
  $('audio-preview').src = state.audioUrl;
  $('audio-preview').hidden = false;
  $('audio-state').textContent = label;
  $('recording-time').textContent = timeLabel(samples.length / rate);
  $('record-label').textContent = 'Thu một lần nữa';
  $('record-hint').textContent = 'Nghe lại bản thu, hoặc thử một nhịp mới';
  drawWaveform(samples);
  showAudioError();
  updateControls();
}

async function decodeAudio(arrayBuffer) {
  if (!AudioContextClass) throw new Error('Trình duyệt chưa hỗ trợ xử lý âm thanh. Hãy mở bằng Chrome, Edge hoặc Firefox.');
  const context = new AudioContextClass();
  try {
    const decoded = await context.decodeAudioData(arrayBuffer);
    if (decoded.duration > 30.05) throw new Error('Âm thanh dài hơn 30 giây. Hãy chọn một bản thu ngắn hơn.');
    const samples = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < samples.length; i += 1) samples[i] += data[i] / decoded.numberOfChannels;
    }
    return { samples, rate: decoded.sampleRate };
  } finally { await context.close(); }
}

async function startRecording() {
  if (state.busy || state.recordingPending || state.recording || state.finalizing) return;
  if (state.practiceMode === 'word' && !practiceAvailable()) { showAudioError('Chọn model A/B/C và câu có âm chuẩn để luyện từng từ.'); return; }
  const microphoneReason = microphoneUnavailableReason();
  if (microphoneReason) {
    showAudioError(microphoneReason);
    updateControls();
    return;
  }
  state.recordingPending = true;
  updateControls();
  showAudioError();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    state.context = new AudioContextClass();
    await state.context.resume();
    clearAudio();
    state.source = state.context.createMediaStreamSource(state.stream);
    state.silentGain = state.context.createGain();
    state.silentGain.gain.value = 0;
    state.chunks = [];
    state.sampleCount = 0;
    state.liveLastSample = 0; state.lastVoiceSample = 0; state.voicedSamples = 0;
    state.recordingLimit = state.practiceMode === 'word' ? 10 : 30;
    delete state.wordResults[state.wordIndex];
    if (state.context.audioWorklet && window.AudioWorkletNode) {
      await state.context.audioWorklet.addModule('/recorder-worklet.js');
      state.processor = new window.AudioWorkletNode(state.context, 'mdd-recorder');
      state.processor.port.onmessage = ({ data }) => {
        if (data.samples) collectRecordingChunk(data.samples);
        if (data.stopped && state.flushRecording) state.flushRecording();
      };
    } else {
      state.processor = state.context.createScriptProcessor(4096, 1, 1);
      state.processor.onaudioprocess = (event) => collectRecordingChunk(event.inputBuffer.getChannelData(0));
    }
    state.recording = true;
    state.liveActive = Boolean(state.liveEnabled && practiceAvailable());
    state.startedAt = performance.now();
    state.source.connect(state.processor);
    state.processor.connect(state.silentGain);
    state.silentGain.connect(state.context.destination);
    $('recording-area').classList.add('recording');
    $('record-button').setAttribute('aria-label', 'Dừng thu âm');
    $('record-label').textContent = 'Đang thu âm…';
    $('record-hint').textContent = state.practiceMode === 'word' ? 'Đọc từ đã chọn rồi ngừng một nhịp · Tối đa 10 giây' : 'Nhấn nút để dừng · Tối đa 30 giây';
    $('audio-state').textContent = 'Đang lắng nghe bạn';
    state.timer = setInterval(() => {
      const duration = (performance.now() - state.startedAt) / 1000;
      $('recording-time').textContent = timeLabel(Math.min(state.recordingLimit, duration));
      pumpLive();
      if (duration >= state.recordingLimit + 0.2) stopRecording();
    }, 160);
  } catch (error) {
    releaseMicrophone();
    const permission = error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError';
    showAudioError(permission ? 'Chưa có quyền dùng microphone. Hãy cho phép trong trình duyệt, hoặc tải âm thanh lên.' : `Không thể mở microphone. ${error.message}`);
  } finally {
    state.recordingPending = false;
    updateControls();
  }
}

function collectRecordingChunk(input) {
  if (!state.recording || !state.context) return;
  const rate = state.context.sampleRate;
  const remaining = Math.max(0, Math.floor(state.recordingLimit * rate) - state.sampleCount);
  const chunk = new Float32Array(input.subarray(0, remaining));
  if (chunk.length) {
    state.chunks.push(chunk); state.sampleCount += chunk.length;
    let power = 0;
    for (const value of chunk) power += value * value;
    if (Math.sqrt(power / chunk.length) >= 0.008) {
      state.lastVoiceSample = state.sampleCount; state.voicedSamples += chunk.length;
    }
    drawWaveform(chunk, true);
  }
  const wordEnded = state.practiceMode === 'word' && state.voicedSamples >= 0.12 * rate
    && state.sampleCount >= rate && state.sampleCount - state.lastVoiceSample >= 0.8 * rate;
  if (!state.stopping && (remaining <= input.length || wordEnded)) stopRecording();
}

function releaseMicrophone() {
  clearInterval(state.timer);
  state.timer = null;
  if (state.processor) {
    if (state.processor.port) state.processor.port.onmessage = null;
    else state.processor.onaudioprocess = null;
    state.processor.disconnect();
  }
  if (state.source) state.source.disconnect();
  if (state.silentGain) state.silentGain.disconnect();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  if (state.context && state.context.state !== 'closed') state.context.close().catch(() => {});
  state.processor = state.source = state.silentGain = state.stream = state.context = null;
  state.recording = false;
  state.stopping = false; state.flushRecording = null;
  $('recording-area').classList.remove('recording');
  $('record-button').setAttribute('aria-label', 'Bắt đầu thu âm');
}

async function stopRecording() {
  if (!state.recording || state.stopping) return;
  state.stopping = true;
  state.finalizing = true;
  updateControls();
  if (state.processor.port) {
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 300);
      state.flushRecording = () => { clearTimeout(timeout); resolve(); };
      state.processor.port.postMessage('flush');
    });
  }
  if (!state.context) { state.finalizing = false; return; }
  const rate = state.context.sampleRate;
  const samples = recordingSamples();
  const autoAnalyze = state.liveActive || state.practiceMode === 'word';
  const pending = state.livePromise;
  releaseMicrophone();
  $('live-status').hidden = false;
  $('live-status').textContent = 'Đã dừng thu · đang chốt kết quả…';
  if (pending) await pending;
  state.chunks = [];
  try { setAudio(samples, rate, 'Bản thu của bạn'); }
  catch (error) { showAudioError(error.message); }
  state.finalizing = false;
  updateControls();
  if (autoAnalyze && state.audioBase64) await analyze();
}

async function uploadAudio(file) {
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) { showAudioError('Tệp quá lớn. Hãy chọn tệp âm thanh dưới 25 MB, tối đa 30 giây.'); return; }
  state.loadingAudio = true;
  updateControls();
  showAudioError();
  try {
    const { samples, rate } = await decodeAudio(await file.arrayBuffer());
    setAudio(samples, rate, file.name);
    toast('Đã tải bản thu. Bạn có thể nghe lại trước khi phân tích.');
  } catch (error) { showAudioError(`Chưa đọc được tệp âm thanh. ${error.message}`); }
  finally { state.loadingAudio = false; $('audio-file').value = ''; updateControls(); }
}

async function loadSample() {
  let prompt = state.prompts.find((item) => item.id === state.promptId);
  if (!prompt) { prompt = state.prompts[0]; if (prompt) choosePrompt(prompt.id); }
  if (!prompt) { showAudioError('Chưa có câu mẫu trong bộ dữ liệu local.'); return; }
  state.loadingAudio = true;
  updateControls();
  showAudioError();
  $('audio-state').textContent = 'Đang tải bản thu mẫu…';
  try {
    const url = prompt.audio_url || `/api/samples/${encodeURIComponent(prompt.id)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Không tìm thấy âm thanh của câu mẫu trên máy.');
    const { samples, rate } = await decodeAudio(await response.arrayBuffer());
    $('reference-text').value = prompt.text;
    updateReferenceCount();
    setAudio(samples, rate, `Bản thu mẫu · ${prompt.speaker || 'source train'}`, prompt.id);
    toast('Đã nạp bản thu mẫu từ tập source train.');
  } catch (error) { showAudioError(error.message); $('audio-state').textContent = 'Chưa tải được bản thu mẫu'; }
  finally { state.loadingAudio = false; updateControls(); }
}

function drawWaveform(samples, live = false) {
  const canvas = $('waveform');
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const width = bounds.width; const height = bounds.height;
  const center = height / 2;
  ctx.clearRect(0, 0, width, height);
  if (!samples || !samples.length) {
    ctx.strokeStyle = '#ddd0ef'; ctx.setLineDash([2, 5]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(5, center); ctx.lineTo(width - 5, center); ctx.stroke();
    return;
  }
  const count = Math.max(1, Math.floor(width / 5));
  ctx.fillStyle = live ? '#7c3aed' : '#c8b6e7';
  for (let bar = 0; bar < count; bar += 1) {
    const start = Math.floor(bar * samples.length / count);
    const end = Math.max(start + 1, Math.floor((bar + 1) * samples.length / count));
    let peak = 0;
    for (let i = start; i < end && i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
    const amplitude = Math.max(1.5, Math.min(height * 0.85, peak * height * 1.8));
    const x = bar * 5 + 1;
    ctx.fillRect(x, center - amplitude / 2, 2.5, amplitude);
  }
}

async function analyze() {
  if (state.busy || state.recording || state.finalizing || !(state.audioBase64 || state.sampleId)) return;
  clearResult();
  const revision = state.resultRevision;
  state.busy = true;
  updateControls();
  showAudioError();
  $('result-empty').hidden = true;
  $('result-content').hidden = true;
  $('result-loading').hidden = false;
  $('save-result').hidden = true;
  $('result-card')?.setAttribute('aria-busy', 'true');
  $('guidance-progress').hidden = false;
  setGuidanceStage('mdd', 'processing', 'Đang phân tích');
  setGuidanceStage('llm', 'waiting', 'Chờ MDD');
  setGuidanceStage('voice', 'waiting', 'Chờ hướng dẫn');
  const payload = { model_id: state.modelId, reference_text: $('reference-text').value.trim() };
  if (state.sampleId) payload.sample_id = state.sampleId;
  else payload.audio_base64 = state.audioBase64;
  const wordMode = state.practiceMode === 'word';
  const endpoint = wordMode ? '/api/analyze-live' : '/api/analyze';
  if (wordMode) Object.assign(payload, { mode: 'word', word_index: state.wordIndex, final: true });
  try {
    const result = await request(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (revision !== state.resultRevision) return;
    state.result = result;
    if (wordMode) {
      const status = result.comparison?.words?.[0]?.status;
      if (['match', 'difference'].includes(status)) state.wordResults[state.wordIndex] = status;
      renderPractice();
    }
    renderResult(state.result);
    loadGuidance();
    if (window.innerWidth <= 930) $('result-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    if (revision !== state.resultRevision) return;
    $('result-empty').hidden = false;
    setGuidanceStage('mdd', 'error', 'Chưa phân tích được');
    setGuidanceStage('llm', 'unavailable', 'Chưa chạy');
    setGuidanceStage('voice', 'unavailable', 'Chưa chạy');
    showAudioError(error.message);
  } finally {
    state.busy = false;
    $('result-loading').hidden = true;
    $('result-card')?.removeAttribute('aria-busy');
    updateControls();
  }
}

function clearGuidance() {
  state.guidanceRevision += 1;
  if (state.guidanceController) state.guidanceController.abort();
  state.guidanceController = null;
  state.guidancePending = false;
  clearTimeout(state.guidanceTimer);
  clearTimeout(state.guidanceTimeout);
  for (const player of $('guidance-audio').querySelectorAll('audio')) {
    player.pause();
    player.removeAttribute('src');
    player.load();
  }
  $('guidance-block').hidden = true;
  $('guidance-block').removeAttribute('aria-busy');
  $('guidance-retry').hidden = true;
  $('guidance-retry').disabled = false;
  $('guidance-status').textContent = '';
  $('guidance-source').textContent = '';
  $('guidance-message').textContent = '';
  $('guidance-summary').textContent = '';
  $('guidance-errors').replaceChildren();
  $('guidance-audio').replaceChildren();
  $('guidance-notices').replaceChildren();
  $('guidance-audio-block').hidden = true;
  $('guidance-progress').hidden = true;
  for (const stage of ['mdd', 'llm', 'voice']) setGuidanceStage(stage, 'waiting', 'Chưa chạy');
}

function setGuidanceStage(stage, status, label) {
  const node = $(`guidance-step-${stage}`);
  node.dataset.status = status;
  $(`guidance-step-${stage}-label`).textContent = label;
  if (status === 'processing') node.setAttribute('aria-current', 'step');
  else node.removeAttribute('aria-current');
}

function phoneLabel(ipa, phone) {
  if (typeof ipa === 'string' && ipa.trim()) return `/${ipa.replace(/^\/+|\/+$/g, '')}/`;
  if (typeof phone === 'string' && phone.trim()) return phone;
  return '∅';
}

// Word-popover contract: one #word-error-tooltip, mapped button[data-word-index]
// triggers, and .word-tooltip-error[data-error-id] cards. Rendering never calls APIs.
function mappedWordErrors(result, index) {
  if (!Number.isInteger(index) || index < 0) return [];
  const assessment = result?.assessment;
  if (!assessment || (assessment.status && assessment.status !== 'ok')) return [];
  const comparedWord = result.comparison?.words?.[index];
  const word = Array.isArray(assessment.words) ? assessment.words.find((item) => item.index === index) : null;
  if (!word || !comparedWord || comparedWord.status !== 'difference' || word.text !== comparedWord.text) return [];
  return (Array.isArray(assessment.errors) ? assessment.errors : []).filter((error) => error.word_index === index);
}

function basicWordCorrection(error, wordText) {
  const target = phoneLabel(error.expected_ipa, error.expected_phone);
  const heard = phoneLabel(error.observed_ipa, error.observed_phone);
  const articulation = {
    TH: 'Đặt nhẹ đầu lưỡi giữa hai hàm răng rồi thổi hơi đều, không rung dây thanh.',
    DH: 'Đặt nhẹ đầu lưỡi giữa hai hàm răng rồi đẩy hơi, đồng thời rung dây thanh.',
    W: 'Tròn môi nhẹ rồi chuyển nhanh sang âm tiếp theo.',
    V: 'Chạm răng trên vào môi dưới, đẩy hơi và rung dây thanh.',
    F: 'Chạm răng trên vào môi dưới và thổi hơi nhẹ, không rung dây thanh.',
    S: 'Đưa hai hàm răng gần nhau và thổi luồng hơi hẹp, không rung dây thanh.',
    Z: 'Đưa hai hàm răng gần nhau, đẩy luồng hơi hẹp và rung dây thanh.',
    NG: 'Nâng phần sau lưỡi về phía ngạc mềm, cho hơi thoát qua mũi.',
    M: 'Khép hai môi, rung dây thanh và cho hơi thoát qua mũi.',
    N: 'Chạm đầu lưỡi vào lợi sau răng trên và cho hơi thoát qua mũi.',
  }[error.expected_phone] || '';
  if (['addition', 'insertion'].includes(error.type)) return `Thử đọc lại từ “${wordText}” mà không thêm âm ${heard} model đã đánh dấu. Luyện chậm rồi đọc lại trong cả câu.`;
  const action = error.type === 'deletion' ? `Bổ sung âm mục tiêu ${target} khi đọc lại từ “${wordText}”.` : `Đọc lại từ “${wordText}” với âm mục tiêu ${target}${error.observed_phone ? ` thay cho âm model nhận diện ${heard}` : ''}.`;
  return `${action}${articulation ? ` ${articulation}` : ''} Đọc chậm từng âm rồi nối lại cả từ.`;
}

function hideWordTooltip({ restoreFocus = false } = {}) {
  clearTimeout(state.wordTooltipTimer);
  const current = state.wordTooltip;
  state.wordTooltip = null;
  $('word-error-tooltip').hidden = true;
  $('word-error-tooltip').removeAttribute('data-word-index');
  if (current) {
    current.anchor.setAttribute('aria-expanded', 'false');
    current.anchor.removeAttribute('aria-describedby');
    if (restoreFocus && current.anchor.isConnected && document.activeElement !== current.anchor) {
      state.wordTooltipSuppressFocus = current.anchor;
      current.anchor.focus({ preventScroll: true });
      state.wordTooltipSuppressFocus = null;
    }
  }
}

function positionWordTooltip() {
  const current = state.wordTooltip;
  const popup = $('word-error-tooltip');
  if (!current || popup.hidden) return;
  if (!current.anchor.isConnected) { hideWordTooltip(); return; }
  const bounds = current.anchor.getBoundingClientRect();
  const width = window.innerWidth || 1024;
  const height = window.innerHeight || 768;
  if (bounds.bottom < 0 || bounds.top > height || bounds.right < 0 || bounds.left > width) { hideWordTooltip(); return; }
  const margin = 12;
  const gap = 10;
  popup.style.maxWidth = `${Math.max(1, width - margin * 2)}px`;
  const popupBounds = popup.getBoundingClientRect();
  const popupWidth = popupBounds.width || Math.min(360, width - margin * 2);
  const naturalHeight = Math.min(460, Math.max(popupBounds.height, popup.scrollHeight || 0) || 300, height - margin * 2);
  const availableBelow = Math.max(0, height - margin - bounds.bottom - gap);
  const availableAbove = Math.max(0, bounds.top - gap - margin);
  const useAbove = naturalHeight > availableBelow && (naturalHeight <= availableAbove || availableAbove > availableBelow);
  const availableHeight = useAbove ? availableAbove : availableBelow;
  popup.style.maxHeight = `${Math.max(1, Math.min(460, availableHeight))}px`;
  const popupHeight = popup.getBoundingClientRect().height || Math.min(naturalHeight, availableHeight);
  const top = useAbove ? bounds.top - gap - popupHeight : bounds.bottom + gap;
  const left = Math.max(margin, Math.min(bounds.left + bounds.width / 2 - popupWidth / 2, width - margin - popupWidth));
  popup.dataset.placement = useAbove ? 'top' : 'bottom';
  popup.style.top = `${Math.round(top)}px`;
  popup.style.left = `${Math.round(left)}px`;
}

function refreshWordTooltip() {
  const current = state.wordTooltip;
  if (!current) return;
  const errors = mappedWordErrors(state.result, current.wordIndex);
  if (!errors.length || !current.anchor.isConnected) { hideWordTooltip(); return; }
  const word = state.result.assessment.words.find((item) => item.index === current.wordIndex);
  const tips = state.result.guidance?.feedback?.tips;
  $('word-error-tooltip-title').textContent = `Từ ${current.wordIndex + 1} · ${word.text}`;
  const content = $('word-error-tooltip-content');
  content.replaceChildren(element('p', 'word-tooltip-estimate', 'Sai khác do model ước lượng. Âm nhận diện có thể chưa chính xác.'));
  for (const error of errors) {
    const card = element('section', 'word-tooltip-error');
    card.setAttribute('data-error-id', error.id || '');
    const type = ({ substitution: 'Thay âm', deletion: 'Thiếu âm', addition: 'Thêm âm', insertion: 'Thêm âm' })[error.type] || 'Khác biệt âm vị';
    card.append(element('h5', '', type));
    const observed = phoneLabel(error.observed_ipa, error.observed_phone);
    const expected = phoneLabel(error.expected_ipa, error.expected_phone);
    card.append(element('p', 'word-tooltip-phones', `Model nghe: ${observed} · Âm mục tiêu: ${expected}`));
    const linked = (Array.isArray(tips) ? tips : []).filter((tip) => tip.error_id === error.id);
    if (linked.length) {
      for (const tip of linked) {
        if (tip.explanation) card.append(element('p', '', tip.explanation));
        card.append(element('p', 'word-tooltip-correction', tip.instruction || basicWordCorrection(error, word.text)));
        if (Array.isArray(tip.practice_words) && tip.practice_words.length) card.append(element('p', 'word-tooltip-practice', `Từ luyện: ${tip.practice_words.join(' · ')}`));
      }
    } else card.append(element('p', 'word-tooltip-correction', basicWordCorrection(error, word.text)));
    content.append(card);
  }
  positionWordTooltip();
}

function showWordTooltip(anchor, wordIndex, { pin = false } = {}) {
  if (!mappedWordErrors(state.result, wordIndex).length || !anchor.isConnected) return;
  clearTimeout(state.wordTooltipTimer);
  const previous = state.wordTooltip;
  if (previous && previous.anchor !== anchor) {
    previous.anchor.setAttribute('aria-expanded', 'false');
    previous.anchor.removeAttribute('aria-describedby');
  }
  state.wordTooltip = { anchor, wordIndex, pinned: pin || (previous?.anchor === anchor && previous.pinned), hovering: previous?.anchor === anchor && previous.hovering };
  anchor.setAttribute('aria-expanded', 'true');
  anchor.setAttribute('aria-describedby', 'word-error-tooltip-content');
  $('word-error-tooltip').dataset.wordIndex = String(wordIndex);
  $('word-error-tooltip').hidden = false;
  refreshWordTooltip();
}

function scheduleWordTooltipHide() {
  clearTimeout(state.wordTooltipTimer);
  state.wordTooltipTimer = setTimeout(() => {
    const current = state.wordTooltip;
    if (!current || current.pinned || current.hovering) return;
    const active = document.activeElement;
    if (active && (current.anchor.contains(active) || $('word-error-tooltip').contains(active))) return;
    hideWordTooltip();
  }, 160);
}

function renderComparisonWords(result) {
  hideWordTooltip();
  $('comparison-words').replaceChildren();
  const words = result.comparison?.words;
  if (!Array.isArray(words)) return;
  words.forEach((word, index) => {
    const status = ['match', 'difference', 'unavailable', 'pending', 'listening'].includes(word.status) ? word.status : 'unavailable';
    const errors = mappedWordErrors(result, index);
    const node = element(errors.length ? 'button' : 'span', `word ${status}${errors.length ? ' word-error-trigger' : ''}`, word.text);
    if (errors.length) {
      node.type = 'button';
      node.setAttribute('data-word-index', index);
      node.setAttribute('aria-haspopup', 'dialog');
      node.setAttribute('aria-controls', 'word-error-tooltip');
      node.setAttribute('aria-expanded', 'false');
      node.setAttribute('aria-label', `Từ ${index + 1}, ${word.text}, ${errors.length} sai khác âm vị model ước lượng. Xem âm mục tiêu và cách luyện.`);
      node.addEventListener('mouseenter', () => {
        showWordTooltip(node, index);
        if (state.wordTooltip?.anchor === node) state.wordTooltip.hovering = true;
      });
      node.addEventListener('mouseleave', () => {
        if (state.wordTooltip?.anchor === node) state.wordTooltip.hovering = false;
        scheduleWordTooltipHide();
      });
      node.addEventListener('focus', () => { if (state.wordTooltipSuppressFocus !== node) showWordTooltip(node, index); });
      node.addEventListener('blur', scheduleWordTooltipHide);
      node.addEventListener('click', () => {
        if (state.wordTooltip?.anchor === node && state.wordTooltip.pinned) hideWordTooltip();
        else showWordTooltip(node, index, { pin: true });
      });
      node.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowDown') {
          event.preventDefault();
          showWordTooltip(node, index, { pin: true });
          $('word-error-tooltip-close').focus();
        }
      });
    }
    $('comparison-words').append(node);
  });
}

function renderGuidanceErrors(assessment = {}, feedback = {}) {
  assessment = assessment || {};
  feedback = feedback || {};
  const root = $('guidance-errors');
  root.replaceChildren();
  const words = new Map((Array.isArray(assessment.words) ? assessment.words : []).map((word) => [word.index, word.text]));
  const errors = Array.isArray(assessment.errors) ? assessment.errors : [];
  const tips = Array.isArray(feedback.tips) ? feedback.tips : [];
  const groups = new Map();
  for (const error of errors) {
    const index = Number.isInteger(error.word_index) && error.word_index >= 0 ? error.word_index : null;
    if (!groups.has(index)) groups.set(index, []);
    groups.get(index).push(error);
  }
  for (const [index, items] of groups) {
    const group = element('article', 'guidance-word');
    const label = index === null ? 'Vị trí từ chưa xác định' : `Từ ${index + 1}${words.has(index) ? ` · ${words.get(index)}` : ''}`;
    group.append(element('h4', '', label));
    for (const error of items) {
      const detail = element('div', 'guidance-error');
      const type = ({ substitution: 'Thay âm', deletion: 'Thiếu âm', addition: 'Thêm âm', insertion: 'Thêm âm' })[error.type] || 'Khác biệt âm vị';
      detail.append(element('p', 'guidance-error-type', type));
      const phones = element('div', 'guidance-phone-pair');
      const expected = element('div');
      expected.append(element('span', '', 'Âm mục tiêu'), element('strong', '', phoneLabel(error.expected_ipa, error.expected_phone)));
      const observed = element('div');
      observed.append(element('span', '', 'Âm nhận diện'), element('strong', '', phoneLabel(error.observed_ipa, error.observed_phone)));
      phones.append(expected, observed);
      detail.append(phones);
      if (finite(error.start_sec) && finite(error.end_sec)) {
        detail.append(element('p', 'small-description', `Mốc mô hình xấp xỉ ${displayNumber(error.start_sec, 2)}–${displayNumber(error.end_sec, 2)} giây.`));
      }
      const linked = tips.filter((tip) => tip.error_id === error.id);
      for (const tip of linked) {
        if (tip.explanation) detail.append(element('p', 'guidance-explanation', tip.explanation));
        if (tip.instruction) {
          const instruction = element('p', 'guidance-instruction');
          instruction.append(element('strong', '', 'Cách luyện: '), element('span', '', tip.instruction));
          detail.append(instruction);
        }
        if (Array.isArray(tip.practice_words) && tip.practice_words.length) {
          detail.append(element('p', 'guidance-practice', `Từ luyện tập: ${tip.practice_words.join(' · ')}`));
        }
      }
      if (!linked.length) detail.append(element('p', 'small-description', 'Chưa có hướng dẫn riêng cho khác biệt này.'));
      group.append(detail);
    }
    root.append(group);
  }
  if (errors.length) root.append(element('p', 'small-description', 'Vị trí theo thứ tự từ trong câu tham chiếu. Chỉ hiển thị mốc thời gian khi mô hình có dữ liệu.'));
  refreshWordTooltip();
}

function safeVoiceUrl(value) {
  if (typeof value !== 'string' || !value) return null;
  try {
    const url = new URL(value, window.location.origin);
    if (url.origin !== window.location.origin || !url.pathname.startsWith('/api/voice/') || url.pathname === '/api/voice/' || url.username || url.password) return null;
    return url.href;
  } catch { return null; }
}

function renderGuidanceAudio(audio, voiceStatus) {
  const root = $('guidance-audio');
  const revision = state.guidanceRevision;
  for (const player of root.querySelectorAll('audio')) { player.pause(); player.removeAttribute('src'); player.load(); }
  root.replaceChildren();
  const clips = Array.isArray(audio) ? audio : [];
  $('guidance-audio-block').hidden = false;
  for (const clip of clips) {
    const card = element('div', 'guidance-voice');
    const vietnamese = typeof clip.language === 'string' && clip.language.toLowerCase().startsWith('vi');
    const label = vietnamese ? 'Hướng dẫn bằng tiếng Việt' : clip.purpose === 'practice' ? 'Luyện từ tiếng Anh' : clip.purpose === 'reference' ? 'Câu mẫu tiếng Anh' : 'Hướng dẫn đọc mẫu';
    card.append(element('h5', '', label));
    if (clip.text) card.append(element('p', '', clip.text));
    const url = safeVoiceUrl(clip.url);
    if (clip.status === 'ready' && url) {
      const player = element('audio');
      player.controls = true;
      player.preload = 'none';
      player.src = url;
      player.lang = vietnamese ? 'vi' : 'en';
      player.setAttribute('aria-label', label);
      player.addEventListener('error', () => {
        if (revision !== state.guidanceRevision || !state.result || !card.isConnected) return;
        const message = element('p', 'guidance-voice-error', 'Chưa tải được giọng đọc này. Bạn có thể thử lại phần hướng dẫn.');
        card.append(message);
        $('guidance-retry').hidden = false;
      }, { once: true });
      card.append(player);
    } else card.append(element('p', 'guidance-voice-error', clip.message || 'Giọng đọc này chưa sẵn sàng.'));
    root.append(card);
  }
  if (!clips.length) root.append(element('p', 'small-description', voiceStatus === 'processing' ? 'ai33 đang tạo giọng đọc tiếng Việt và mẫu tiếng Anh. Hướng dẫn bằng chữ đã có ở trên.' : voiceStatus === 'not_requested' ? 'Chưa tạo giọng đọc cho kết quả này.' : 'Giọng đọc chưa sẵn sàng. Bạn vẫn có thể xem hướng dẫn bằng chữ.'));
}

function renderGuidance(result, { voicePending = false } = {}) {
  $('guidance-block').hidden = false;
  const status = result.status || 'unavailable';
  const feedback = result.feedback || {};
  const fromLLM = result.feedback_source === 'llm';
  $('guidance-status').textContent = voicePending ? 'Đang tạo giọng đọc' : status === 'ready' ? 'Đã sẵn sàng' : status === 'partial' ? 'Sẵn sàng một phần' : 'Chưa sẵn sàng';
  $('guidance-source').textContent = result.feedback_source === 'llm' ? 'Hướng dẫn từ LLM' : 'Hướng dẫn mẫu';
  $('guidance-message').textContent = voicePending ? 'LLM đã tạo hướng dẫn. ai33 đang tạo giọng đọc; bạn có thể đọc cách luyện ngay bên dưới.' : !fromLLM ? 'LLM chưa trả về hướng dẫn hợp lệ; đang hiển thị nội dung mẫu. Chưa gọi ai33 cho lượt này.' : status === 'partial' ? 'Hướng dẫn LLM đã có. Một phần giọng đọc chưa hoàn tất; kết quả MDD và cách luyện vẫn được giữ.' : status === 'unavailable' ? 'Kết quả MDD local vẫn hiển thị phía trên. Hướng dẫn hoặc giọng đọc chưa khả dụng cho lần phân tích này.' : 'Đọc hướng dẫn rồi nhấn phát để nghe tiếng Việt hoặc câu mẫu tiếng Anh.';
  $('guidance-summary').textContent = feedback.summary || '';
  renderGuidanceErrors(result.assessment, feedback);
  renderGuidanceAudio(voicePending ? [] : result.audio, voicePending ? 'processing' : result.voice_status);
  setGuidanceStage('llm', fromLLM ? 'complete' : status === 'unavailable' ? 'unavailable' : 'template', fromLLM ? 'Đã có hướng dẫn' : status === 'unavailable' ? 'Chưa kết nối' : 'Đang dùng mẫu');
  const voiceStatus = voicePending ? 'processing' : result.voice_status === 'ready' ? 'complete' : result.voice_status === 'partial' ? 'error' : 'unavailable';
  setGuidanceStage('voice', voiceStatus, voicePending ? 'Đang tạo giọng đọc' : voiceStatus === 'complete' ? 'Đã có giọng đọc' : voiceStatus === 'error' ? 'Có một phần âm thanh' : 'Chưa tạo giọng đọc');
  $('guidance-notices').replaceChildren();
  for (const notice of Array.isArray(result.notices) ? result.notices : []) {
    $('guidance-notices').append(element('p', 'small-description', notice));
  }
  const audioFailed = Array.isArray(result.audio) && result.audio.some((clip) => clip.status !== 'ready' || !safeVoiceUrl(clip.url));
  $('guidance-retry').hidden = voicePending || (status === 'ready' && !audioFailed && !['partial', 'unavailable'].includes(result.voice_status));
}

async function loadGuidance() {
  if (state.guidancePending || !state.result) return;
  const analysis = state.result;
  const analysisId = analysis.analysis_id;
  $('guidance-block').hidden = false;
  $('guidance-progress').hidden = false;
  setGuidanceStage('mdd', 'complete', 'Đã có kết quả');
  if (analysis.assessment) renderGuidanceErrors(analysis.assessment, analysis.guidance?.feedback);
  if (analysis.feedback_available === false || !analysisId) {
    $('guidance-status').textContent = 'Chưa có hướng dẫn';
    $('guidance-source').textContent = '';
    $('guidance-message').textContent = analysis.feedback_unavailable_reason || analysis.feedback_reason || analysis.notice || 'Cần kết quả MDD có đối chiếu đáng tin cậy và câu tham chiếu để tạo hướng dẫn.';
    $('guidance-retry').hidden = true;
    setGuidanceStage('llm', 'unavailable', 'Không có đối chiếu');
    setGuidanceStage('voice', 'unavailable', 'Chưa chạy');
    return;
  }
  if (state.capabilities.llm_configured === false) {
    $('guidance-status').textContent = 'LLM chưa kết nối';
    $('guidance-message').textContent = 'Trợ giảng chưa được kết nối. Các vị trí khác biệt âm vị từ MDD vẫn có sẵn bên dưới.';
    setGuidanceStage('llm', 'unavailable', 'Chưa kết nối');
    setGuidanceStage('voice', 'unavailable', 'Chờ hướng dẫn LLM');
    $('guidance-retry').hidden = false;
    return;
  }
  const revision = state.guidanceRevision;
  const controller = new AbortController();
  state.guidanceController = controller;
  state.guidancePending = true;
  $('guidance-block').setAttribute('aria-busy', 'true');
  $('guidance-status').textContent = 'Đang tạo hướng dẫn';
  $('guidance-message').textContent = 'MDD đã chỉ ra các vị trí khác biệt. Đang đợi LLM tạo cách luyện; ai33 sẽ tạo giọng đọc sau khi có hướng dẫn.';
  $('guidance-retry').hidden = true;
  $('guidance-retry').disabled = true;
  setGuidanceStage('llm', 'processing', 'Đang tạo cách luyện');
  setGuidanceStage('voice', 'waiting', 'Chờ hướng dẫn LLM');
  let activeStage = 'llm';
  state.guidanceTimer = setTimeout(() => {
    if (revision === state.guidanceRevision) $('guidance-message').textContent = activeStage === 'llm' ? 'LLM vẫn đang tạo hướng dẫn. Bạn có thể xem lỗi MDD ngay và đổi bản thu hoặc mô hình khi cần.' : 'ai33 vẫn đang tạo giọng đọc. Hướng dẫn LLM đã hiển thị; bạn có thể luyện theo chữ trước.';
  }, 15000);
  state.guidanceTimeout = setTimeout(() => controller.abort(), 900000);
  try {
    const requestStage = (includeVoice) => request('/api/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis_id: analysisId, include_voice: includeVoice }), signal: controller.signal,
    });
    const result = await requestStage(false);
    if (revision !== state.guidanceRevision || state.result !== analysis) return;
    if (result.analysis_id !== analysisId) throw new Error('Hướng dẫn trả về chưa khớp với lần phân tích này.');
    state.result.guidance = result;
    if (result.feedback_source !== 'llm' || result.status === 'unavailable') {
      renderGuidance(result);
      return;
    }
    renderGuidance(result, { voicePending: true });
    if (state.capabilities.tts_configured === false) {
      setGuidanceStage('voice', 'unavailable', 'ai33 chưa kết nối');
      $('guidance-status').textContent = 'Đã có hướng dẫn LLM';
      $('guidance-message').textContent = 'Cách luyện đã sẵn sàng. ai33 chưa được kết nối để tạo giọng đọc.';
      $('guidance-audio').replaceChildren(element('p', 'small-description', 'Giọng đọc ai33 chưa sẵn sàng; bạn có thể luyện theo hướng dẫn bằng chữ.'));
      $('guidance-retry').hidden = false;
      return;
    }
    activeStage = 'voice';
    const voiceResult = await requestStage(true);
    if (revision !== state.guidanceRevision || state.result !== analysis) return;
    if (voiceResult.analysis_id !== analysisId) throw new Error('Giọng đọc trả về chưa khớp với lần phân tích này.');
    state.result.guidance = voiceResult;
    renderGuidance(voiceResult);
  } catch (error) {
    if (revision !== state.guidanceRevision || state.result !== analysis) return;
    setGuidanceStage(activeStage === 'llm' ? 'llm' : 'voice', 'error', activeStage === 'llm' ? 'Chưa tải được hướng dẫn' : 'Chưa tải được giọng đọc');
    if (activeStage === 'llm') setGuidanceStage('voice', 'unavailable', 'Chờ hướng dẫn LLM');
    $('guidance-status').textContent = activeStage === 'llm' ? 'Chưa tải được hướng dẫn' : 'Hướng dẫn đã có · giọng đọc gặp lỗi';
    $('guidance-message').textContent = error.name === 'AbortError' ? 'Dịch vụ chưa hoàn tất trong thời gian chờ. Kết quả đã có được giữ lại; bạn có thể tải lại phần còn thiếu.' : `Kết quả MDD${activeStage === 'voice' ? ' và hướng dẫn LLM' : ''} được giữ lại. ${error.message}`;
    $('guidance-retry').hidden = false;
  } finally {
    if (revision === state.guidanceRevision && state.result === analysis) {
      clearTimeout(state.guidanceTimer);
      clearTimeout(state.guidanceTimeout);
      state.guidancePending = false;
      state.guidanceController = null;
      $('guidance-retry').disabled = false;
      $('guidance-block').removeAttribute('aria-busy');
    }
  }
}

function resultStat(value, label) {
  const box = element('div', 'result-stat');
  box.append(element('strong', '', value), element('span', '', label));
  return box;
}

function renderResult(result) {
  const preview = result.live?.is_final === false;
  $('result-empty').hidden = true;
  $('result-content').hidden = false;
  $('save-result').hidden = preview;
  $('guidance-progress').hidden = false;
  setGuidanceStage('mdd', 'complete', 'Đã có kết quả');
  $('guidance-block').hidden = false;
  renderGuidanceErrors(result.assessment);
  const model = result.model || {};
  const audio = result.audio || {};
  const features = result.features;
  const phones = Array.isArray(result.phonemes) ? result.phonemes : [];
  const comparison = result.comparison;
  const isBase = result.analysis_kind === 'backbone_features' || model.kind === 'backbone';
  const experimental = model.experimental || result.analysis_kind === 'experimental_phonemes';
  $('result-kind').textContent = isBase ? 'BASE · ĐẶC TRƯNG ÂM THANH' : experimental ? 'CHECKPOINT THỬ NGHIỆM' : comparison ? 'ĐỐI CHIẾU ÂM VỊ' : 'NHẬN DIỆN ÂM VỊ';
  $('result-headline').textContent = isBase ? 'Âm thanh, dưới một góc nhìn khác.' : experimental ? 'Những âm đầu tiên mô hình nghe được.' : comparison ? 'Cùng nhìn lại từng âm.' : 'Mô hình đã lắng nghe.';
  if (preview) {
    $('result-kind').textContent = 'ĐANG ĐỌC · KẾT QUẢ TẠM THỜI';
    $('result-headline').textContent = result.live.mode === 'word' ? 'Đang nghe từ bạn chọn.' : 'Kết quả theo nhịp bạn đọc.';
    $('guidance-block').hidden = true;
    setGuidanceStage('mdd', 'processing', 'Đang cập nhật');
    setGuidanceStage('llm', 'waiting', 'Chờ dừng thu');
    setGuidanceStage('voice', 'waiting', 'Chờ hướng dẫn');
  }
  $('result-notice').textContent = result.notice || (isBase ? 'Mô hình BASE trích xuất biểu diễn âm thanh, chưa nhận diện âm vị hay chấm chất lượng phát âm.' : experimental ? 'Đây là kết quả thử nghiệm, chưa đủ cơ sở để kết luận về phát âm của bạn.' : 'Kết quả dưới đây là đầu ra thực tế của mô hình local.');
  $('result-stats').replaceChildren(
    resultStat(`${displayNumber(audio.duration_sec, 1)} s`, 'Độ dài bản thu'),
    resultStat(finite(audio.sample_rate) ? `${displayNumber(Number(audio.sample_rate) / 1000, 1)} kHz` : '—', 'Tần số xử lý'),
    resultStat(isBase ? displayNumber(features?.frames, 0) : displayNumber(phones.length, 0), isBase ? 'Khung đặc trưng' : 'Âm vị nhận diện'),
  );
  $('comparison-block').hidden = !comparison;
  renderComparisonWords(result);
  if (comparison) {
    const counts = comparison.counts || {};
    $('comparison-explanation').textContent = `Đối chiếu âm vị: ${displayNumber(counts.match, 0)} khớp · ${displayNumber(counts.substitution, 0)} thay thế · ${displayNumber(counts.deletion, 0)} bị thiếu · ${displayNumber(counts.insertion, 0)} thêm vào. Di chuột, dùng phím Tab hoặc chạm từ màu vàng để xem sai khác và cách luyện; nhấn lại để đóng.`;
    if (preview) $('comparison-explanation').textContent = 'Xám: chưa đọc · Xanh lam: đang đọc · Xanh lá: khớp tạm thời · Vàng: khác biệt tạm thời. Di chuột hoặc chạm để xem âm mục tiêu và âm model nhận. Kết quả có thể đổi khi đọc tiếp.';
  }
  $('phoneme-block').hidden = isBase;
  $('phoneme-count').textContent = `${phones.length} âm vị`;
  $('phoneme-list').replaceChildren();
  for (const phone of phones) {
    const chip = element('span', 'phone', phone.phone);
    const timing = finite(phone.start_sec) && finite(phone.end_sec) ? `xấp xỉ ${displayNumber(phone.start_sec, 2)}–${displayNumber(phone.end_sec, 2)} s` : 'Không có mốc thời gian';
    const confidence = finite(phone.confidence) ? ` · xác suất mô hình ${displayNumber(Number(phone.confidence) * 100, 1)}%` : '';
    chip.title = `${timing}${confidence}`;
    chip.tabIndex = 0;
    chip.setAttribute('aria-label', `${phone.phone}, ${timing}${confidence}`);
    $('phoneme-list').append(chip);
  }
  if (!isBase && !phones.length) $('phoneme-list').append(element('p', 'small-description', 'Mô hình chưa giải mã được âm vị trong bản thu này.'));
  $('feature-block').hidden = !features;
  $('feature-values').replaceChildren();
  if (features) {
    for (const [label, value] of [['Số khung', features.frames], ['Chiều đặc trưng', features.dimensions], ['Norm trung bình', features.norm_mean]]) {
      const row = element('div', 'feature-row');
      row.append(element('span', '', label), element('strong', '', displayNumber(value)));
      $('feature-values').append(row);
    }
  }
  const warning = model.warning || (experimental ? 'Checkpoint smoke chỉ kiểm tra luồng xử lý. Âm vị ở đây không phải kết luận chẩn đoán hay điểm phát âm.' : isBase ? 'Chưa có điểm phát âm: BASE không có đầu phone-CTC đã huấn luyện.' : !comparison ? 'Chưa có đối chiếu lỗi: cần câu tham chiếu khớp với câu mẫu có nhãn âm vị.' : 'Đối chiếu tự động cần được xem xét trong bối cảnh và chất lượng mô hình.');
  $('result-warning').textContent = warning;
  $('result-warning').hidden = !warning;
  $('result-model').textContent = model.label || model.id || state.modelId;
  $('result-time').textContent = finite(result.inference_seconds) ? `${displayNumber(result.inference_seconds, 2)} s · CPU` : 'Local inference';
}

function statusLabel(status) {
  return ({ completed: 'Đã hoàn thành', recent_activity: 'Vừa cập nhật', checkpointed: 'Đã lưu checkpoint', initialized: 'Đã khởi tạo', running: 'Đang chạy', failed: 'Cần kiểm tra', error: 'Có lỗi', stopped: 'Đã dừng', interrupted: 'Đã gián đoạn' })[status] || 'Đã ghi nhận';
}

function renderTraining() {
  const root = $('training-runs');
  root.replaceChildren();
  if (!state.training.length) {
    root.append(element('div', 'no-runs', 'Chưa có phiên huấn luyện được ghi nhận. Khi một run được tạo, tiến trình và loss thực tế sẽ tự xuất hiện tại đây.'));
    return;
  }
  for (const run of state.training) {
    const card = element('article', 'training-run');
    const heading = element('div', 'run-heading');
    const title = element('div');
    title.append(element('h2', '', run.run_id));
    const meta = element('div', 'run-meta');
    meta.append(element('span', '', `Giai đoạn ${run.stage || '—'}`), element('span', '', run.purpose === 'technical_smoke_only' ? 'Kiểm tra kỹ thuật' : run.purpose === 'scientific_training' ? 'Huấn luyện nghiên cứu' : 'Phiên local'));
    title.append(meta);
    const knownStatus = ['completed', 'recent_activity', 'checkpointed', 'initialized', 'running', 'failed', 'error', 'stopped', 'interrupted'].includes(run.status) ? run.status : '';
    heading.append(title, element('span', `run-state ${knownStatus}`, statusLabel(run.status)));
    card.append(heading);
    const measures = element('div', 'run-measures');
    const step = element('div');
    step.append(element('strong', '', `${displayNumber(run.step, 0)} / ${displayNumber(run.max_steps, 0)}`), element('span', '', 'BƯỚC OPTIMIZER'));
    const loss = element('div');
    loss.append(element('strong', '', displayNumber(run.loss, 4)), element('span', '', 'TRAINING LOSS'));
    const f1 = element('div');
    f1.append(element('strong', '', displayNumber(run.best_f1, 4)), element('span', '', run.purpose === 'technical_smoke_only' ? 'F1 · KIỂM TRA KỸ THUẬT' : 'VALIDATION F1 TỐT NHẤT'));
    measures.append(step, loss, f1);
    card.append(measures);
    const progress = element('div', 'run-progress');
    const bar = element('span');
    bar.style.width = `${finite(run.step) && finite(run.max_steps) && Number(run.max_steps) > 0 ? Math.max(0, Math.min(100, Number(run.step) / Number(run.max_steps) * 100)) : 0}%`;
    progress.append(bar);
    progress.setAttribute('role', 'progressbar');
    progress.setAttribute('aria-label', `Tiến trình ${run.run_id}`);
    if (finite(run.step)) progress.setAttribute('aria-valuenow', Math.max(0, Math.min(Number(run.step), Number(run.max_steps) || Number(run.step))));
    progress.setAttribute('aria-valuemin', '0');
    if (finite(run.max_steps)) progress.setAttribute('aria-valuemax', run.max_steps);
    card.append(progress);
    const history = (Array.isArray(run.history) ? run.history : []).filter((point) => finite(point.step) && finite(point.loss));
    if (history.length) {
      const chart = element('canvas', 'loss-chart');
      chart.setAttribute('role', 'img');
      chart.setAttribute('aria-label', `Training loss, ${history.length} điểm, từ bước ${history[0].step} đến ${history[history.length - 1].step}`);
      card.append(chart);
      requestAnimationFrame(() => drawLossChart(chart, history));
      const caption = element('div', 'chart-caption');
      caption.append(element('span', '', 'LOSS THEO BƯỚC · DỮ LIỆU THỰC TẾ'), element('span', '', `${history.length} điểm`));
      card.append(caption);
    } else card.append(element('p', 'small-description', 'Chưa có loss trong nhật ký huấn luyện.'));
    if (run.updated_at) {
      const date = new Date(run.updated_at);
      if (!Number.isNaN(date.getTime())) card.append(element('p', 'small-description', `Cập nhật file gần nhất: ${date.toLocaleString('vi-VN')}`));
    }
    root.append(card);
  }
}

function drawLossChart(canvas, history) {
  if (!canvas.isConnected) return;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = bounds.width * ratio;
  canvas.height = bounds.height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const padding = { top: 12, right: 14, bottom: 24, left: 37 };
  const width = bounds.width - padding.left - padding.right;
  const height = bounds.height - padding.top - padding.bottom;
  let low = Math.min(...history.map((point) => Number(point.loss)));
  let high = Math.max(...history.map((point) => Number(point.loss)));
  const range = high - low || Math.max(Math.abs(high) * 0.2, 1);
  low = Math.max(0, low - range * 0.12); high += range * 0.12;
  const minStep = Number(history[0].step);
  const maxStep = Number(history[history.length - 1].step);
  const x = (point) => padding.left + (maxStep === minStep ? width / 2 : (Number(point.step) - minStep) / (maxStep - minStep) * width);
  const y = (point) => padding.top + height - (Number(point.loss) - low) / (high - low) * height;
  ctx.font = '11px Segoe UI, sans-serif'; ctx.fillStyle = '#776b86'; ctx.strokeStyle = '#eee6f7'; ctx.lineWidth = 1;
  for (let line = 0; line <= 2; line += 1) {
    const yPos = padding.top + height * line / 2;
    ctx.beginPath(); ctx.moveTo(padding.left, yPos); ctx.lineTo(bounds.width - padding.right, yPos); ctx.stroke();
    ctx.fillText(displayNumber(high - (high - low) * line / 2, 1), 0, yPos + 3);
  }
  ctx.fillText(displayNumber(minStep, 0), padding.left, bounds.height - 4);
  if (maxStep !== minStep) ctx.fillText(displayNumber(maxStep, 0), bounds.width - padding.right - 20, bounds.height - 4);
  ctx.beginPath();
  history.forEach((point, index) => { if (!index) ctx.moveTo(x(point), y(point)); else ctx.lineTo(x(point), y(point)); });
  ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 1.7; ctx.stroke();
  if (history.length === 1) { ctx.beginPath(); ctx.arc(x(history[0]), y(history[0]), 3, 0, Math.PI * 2); ctx.fillStyle = '#7c3aed'; ctx.fill(); }
}

function saveResult() {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = element('a');
  anchor.href = url;
  anchor.download = `am-analysis-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast('Đã lưu kết quả phân tích dưới dạng JSON.');
}

document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => switchView(button.dataset.view)));
$('record-button').addEventListener('click', () => { if (state.recording) stopRecording(); else startRecording(); });
$('mode-sentence').addEventListener('click', () => setPracticeMode('sentence'));
$('mode-word').addEventListener('click', () => setPracticeMode('word'));
$('word-next').addEventListener('click', () => choosePracticeWord(state.wordIndex + 1));
$('live-enabled').addEventListener('change', (event) => { state.liveEnabled = event.target.checked; clearResult(); updateControls(); });
$('upload-button').addEventListener('click', () => $('audio-file').click());
$('audio-file').addEventListener('change', (event) => uploadAudio(event.target.files[0]));
$('sample-button').addEventListener('click', loadSample);
$('prompt-select').addEventListener('change', (event) => { choosePrompt(event.target.value); clearResult(); });
$('shuffle-prompt').addEventListener('click', () => {
  const available = state.prompts.filter((prompt) => prompt.id !== state.promptId);
  const prompt = available[Math.floor(Math.random() * available.length)] || state.prompts[0];
  if (prompt) { choosePrompt(prompt.id); clearResult(); }
});
$('reference-text').addEventListener('input', () => {
  state.wordIndex = 0; state.wordResults = {};
  resolveCustomReference();
  if (state.practiceMode === 'word' && state.audioSamples) clearAudio();
  updateReferenceCount();
  const prompt = state.prompts.find((item) => item.id === state.promptId);
  if (prompt && $('reference-text').value !== prompt.text) {
    state.promptId = '';
    $('prompt-select').value = '';
    if (state.sampleId) {
      state.audioBase64 = base64Buffer(encodeWav(state.audioSamples, state.audioRate));
      state.sampleId = null;
    }
  }
  clearResult(); updateControls();
});
$('model-select').addEventListener('change', (event) => {
  state.wordResults = {};
  state.modelId = event.target.value;
  $('model-note').textContent = modelDescription(state.models.find((item) => item.id === state.modelId));
  clearResult(); updateControls();
});
$('analyze-button').addEventListener('click', analyze);
$('guidance-retry').addEventListener('click', loadGuidance);
$('save-result').addEventListener('click', saveResult);
$('refresh-training').addEventListener('click', refreshStatus);
$('word-error-tooltip-close').addEventListener('click', () => hideWordTooltip({ restoreFocus: true }));
$('word-error-tooltip').addEventListener('mouseenter', () => {
  clearTimeout(state.wordTooltipTimer);
  if (state.wordTooltip) state.wordTooltip.hovering = true;
});
$('word-error-tooltip').addEventListener('mouseleave', () => {
  if (state.wordTooltip) state.wordTooltip.hovering = false;
  scheduleWordTooltipHide();
});
$('word-error-tooltip').addEventListener('focusin', () => clearTimeout(state.wordTooltipTimer));
$('word-error-tooltip').addEventListener('focusout', scheduleWordTooltipHide);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && state.wordTooltip) {
    event.preventDefault();
    hideWordTooltip({ restoreFocus: true });
  }
});
document.addEventListener('pointerdown', (event) => {
  const current = state.wordTooltip;
  if (current && !current.anchor.contains(event.target) && !$('word-error-tooltip').contains(event.target)) hideWordTooltip();
});
window.addEventListener('scroll', positionWordTooltip, true);
window.addEventListener('resize', () => { positionWordTooltip(); drawWaveform(state.audioSamples); if (!$('training-view').hidden) renderTraining(); });
window.addEventListener('beforeunload', () => { cancelLive(); if (state.recording) releaseMicrophone(); });
if (!AudioContextClass) showAudioError('Trình duyệt chưa hỗ trợ Web Audio. Hãy mở demo bằng Chrome, Edge hoặc Firefox.');
drawWaveform(null);
updateControls();
Promise.allSettled([refreshStatus(), loadPrompts()]);
setInterval(() => { if (!document.hidden) refreshStatus(); }, 4000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshStatus(); });
