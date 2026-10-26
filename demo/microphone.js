(function (root, factory) {
  'use strict';
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root && root.document) root.MDDMicrophone = api;
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  const STORAGE_KEY = 'mdd.microphone.device.v1';
  const AUTO = 'auto';
  const ALIASES = new Set(['default', 'communications']);
  const AUDIO = Object.freeze({ channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false });
  const renders = new Set();
  const locks = new Set();
  let singleton = null, nextPicker = 0;

  function stopStream(stream) {
    for (const track of stream?.getTracks?.() || []) {
      try { track.stop(); } catch (_) { /* A removed device may already be stopped. */ }
    }
  }

  function errorNamed(name, message) {
    const error = new Error(message); error.name = name; return error;
  }

  function normalizeDevices(rows) {
    const seen = new Set(), devices = [];
    for (const row of rows || []) {
      if (row?.kind !== 'audioinput' || typeof row.deviceId !== 'string' || !row.deviceId || seen.has(row.deviceId)) continue;
      seen.add(row.deviceId);
      devices.push({ deviceId: row.deviceId, groupId: typeof row.groupId === 'string' ? row.groupId : '',
        label: typeof row.label === 'string' ? row.label : '', alias: ALIASES.has(row.deviceId), isDefault: row.deviceId === 'default' });
    }
    const defaults = devices.filter(row => row.deviceId === 'default' && row.groupId);
    const physical = devices.filter(row => !row.alias);
    for (const row of physical) row.isDefault = defaults.some(alias => alias.groupId === row.groupId);
    return devices.filter(row => !row.alias || !row.groupId ||
      (!physical.some(device => device.groupId && device.groupId === row.groupId) &&
        !(row.deviceId === 'communications' && defaults.some(alias => alias.groupId === row.groupId))));
  }

  function deviceScore(device) {
    const label = device.label.toLowerCase();
    let score = device.isDefault ? 40 : 0;
    if (/\busb\b|external|wired|headset|headphone|earphone|\bjack\b|3\.5\s?mm/.test(label)) score += 500;
    if (/bluetooth|\bbt\b|hands[ -]?free|\ba2dp\b|wireless/.test(label)) score -= 700;
    if (/virtual|loopback|stereo mix|what u hear|monitor of|vb.audio|voicemeeter|cable output|screen capture|soundflower|blackhole/.test(label)) score -= 1500;
    if (device.alias) score -= 50;
    return score;
  }

  function rankDevices(devices) {
    return devices.map((row, index) => ({ row, index, score: deviceScore(row) }))
      .sort((left, right) => right.score - left.score || left.index - right.index).map(item => item.row);
  }

  function trackOf(stream) { return stream?.getAudioTracks?.()[0] || stream?.getTracks?.()[0] || null; }
  function streamMatches(stream, device) {
    if (!device) return true;
    const track = trackOf(stream), settings = track?.getSettings?.() || {};
    if (settings.deviceId && !ALIASES.has(settings.deviceId) && !ALIASES.has(device.deviceId)) return settings.deviceId === device.deviceId;
    return settings.deviceId === device.deviceId || Boolean(settings.groupId && device.groupId && settings.groupId === device.groupId) ||
      Boolean(track?.label && device.label && track.label === device.label);
  }

  class MicrophoneManager {
    constructor(options = {}) {
      this.mediaDevices = options.mediaDevices || root.navigator?.mediaDevices;
      this.document = options.document || root.document;
      this.eventTarget = options.eventTarget || root;
      try { this.storage = options.storage === undefined ? root.localStorage : options.storage; } catch (_) { this.storage = null; }
      this.preference = AUTO;
      try {
        const value = this.storage?.getItem(STORAGE_KEY);
        if (typeof value === 'string' && value.length && value.length <= 512 && !/[\x00-\x1f]/.test(value)) this.preference = value;
      } catch (_) { /* Recording also works when browser storage is disabled. */ }
      this.devices = []; this.permission = 'unknown'; this.message = '';
      this._listeners = new Set(); this._pendingStreams = new Set(); this._refresh = null; this._permissionPromise = null; this._permissionPending = false;
      this._destroyed = false; this._failedPreferred = ''; this._lastMissing = false;
      this._onDeviceChange = () => { this._failedPreferred = ''; this.refresh().catch(() => {}); };
      this._onVisible = () => { if (this._isVisible()) this.refresh().catch(() => {}); };
      this._onStorage = event => {
        if (event.key !== STORAGE_KEY) return;
        const value = event.newValue;
        this.preference = typeof value === 'string' && value.length && value.length <= 512 && !/[\x00-\x1f]/.test(value) ? value : AUTO;
        this._failedPreferred = ''; this._emit();
      };
      this.mediaDevices?.addEventListener?.('devicechange', this._onDeviceChange);
      this.document?.addEventListener?.('visibilitychange', this._onVisible);
      this.eventTarget?.addEventListener?.('storage', this._onStorage);
    }

    _isVisible() { return this.document?.visibilityState !== 'hidden' && !this.document?.prerendering; }
    _choice(excluded = new Set(), forceAuto = false) {
      const available = this.devices.filter(row => !excluded.has(row.deviceId));
      const manual = this.preference !== AUTO && !forceAuto;
      const preferred = available.find(row => row.deviceId === this.preference);
      const known = this.permission === 'granted' || this.devices.some(row => row.label);
      if (manual && this._failedPreferred !== this.preference && preferred) return preferred;
      if (manual && !known && !excluded.has(this.preference)) return { deviceId: this.preference, groupId: '', label: '', alias: false };
      return rankDevices(available)[0] || null;
    }
    getState() {
      const chosen = this._choice();
      const known = this.permission === 'granted' || this.devices.some(row => row.label);
      const missing = this.preference !== AUTO && (this._failedPreferred === this.preference ||
        (known && !this.devices.some(row => row.deviceId === this.preference)));
      return { preference: this.preference, mode: this.preference === AUTO ? AUTO : 'manual', devices: this.devices.map(row => ({ ...row })),
        resolvedDeviceId: chosen?.deviceId || '', resolvedLabel: chosen?.label || '', permission: this.permission,
        missingPreferredDevice: missing, message: this.message,
        permissionPending: this._permissionPending,
        needsPermission: !this.devices.some(row => row.label) && this.permission !== 'granted' };
    }
    _emit() {
      if (this._destroyed) return;
      const state = this.getState();
      for (const listener of this._listeners) { try { listener(state); } catch (_) { /* One view must not break the other recorder. */ } }
    }
    subscribe(listener) { this._listeners.add(listener); listener(this.getState()); return () => this._listeners.delete(listener); }
    select(deviceId) {
      if (typeof deviceId !== 'string' || !deviceId || deviceId.length > 512 || /[\x00-\x1f]/.test(deviceId)) throw new TypeError('Invalid microphone selection');
      this.preference = deviceId; this._failedPreferred = ''; this._lastMissing = false;
      try { this.storage?.setItem(STORAGE_KEY, deviceId); } catch (_) { /* Keep the choice for this session. */ }
      const selected = this.devices.find(row => row.deviceId === deviceId);
      this.message = deviceId === AUTO ? 'Đang dùng Tự động · Ưu tiên mic ngoài.' : `Đã chọn ${selected?.label || 'microphone'} cho lần thu tiếp theo.`;
      this._emit(); return this.getState();
    }
    async refresh() {
      if (this._destroyed || !this._isVisible() || !this.mediaDevices?.enumerateDevices) return this.getState();
      if (this._refresh) return this._refresh;
      this._refresh = (async () => {
        try {
          const rows = await this.mediaDevices.enumerateDevices();
          if (this._destroyed) return this.getState();
          this.devices = normalizeDevices(rows);
          const missing = this.getState().missingPreferredDevice;
          if (missing && !this._lastMissing) this.message = 'Microphone đã chọn không còn kết nối. Đang dùng Tự động; lựa chọn của bạn được giữ lại.';
          else if (!missing && this._lastMissing) this.message = 'Đã nhận lại microphone đã chọn.';
          this._lastMissing = missing; this._emit(); return this.getState();
        } finally { this._refresh = null; }
      })();
      return this._refresh;
    }
    async _capture(deviceId, audio = AUDIO) {
      if (this._destroyed) throw errorNamed('AbortError', 'Microphone manager was closed');
      if (!this.mediaDevices?.getUserMedia) throw errorNamed('NotSupportedError', 'Microphone is unavailable');
      const constraints = { ...audio, ...AUDIO }; delete constraints.deviceId;
      if (deviceId) constraints.deviceId = { exact: deviceId };
      const stream = await this.mediaDevices.getUserMedia({ audio: constraints, video: false });
      if (this._destroyed) { stopStream(stream); throw errorNamed('AbortError', 'Microphone manager was closed'); }
      this._pendingStreams.add(stream); this.permission = 'granted'; return stream;
    }
    _release(stream) { this._pendingStreams.delete(stream); stopStream(stream); }
    _permissionError(error) {
      if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
        this.permission = 'denied'; this.message = 'Chưa có quyền dùng microphone. Hãy cho phép trong trình duyệt.'; this._emit();
      }
    }
    requestPermission() {
      if (this._permissionPromise) return this._permissionPromise;
      this._permissionPending = true;
      this._permissionPromise = (async () => {
        let stream;
        try {
          stream = await this._capture('');
          await this.refresh(); // Read labels while the explicit probe is still active.
          return this.getState();
        } catch (error) { this._permissionError(error); throw error; }
        finally { if (stream) this._release(stream); this._permissionPending = false; this._permissionPromise = null; this._emit(); }
      })();
      this._emit(); return this._permissionPromise;
    }
    async getStream(options = {}) {
      if (this._permissionPromise) await this._permissionPromise;
      const audio = options.audio && typeof options.audio === 'object' ? options.audio : AUDIO;
      let stream = null, retried = false, requestedDeviceId = '';
      const excluded = new Set();
      const capture = async choice => {
        requestedDeviceId = choice?.deviceId || '';
        try { return await this._capture(requestedDeviceId, audio); }
        catch (error) {
          if (!retried && choice?.deviceId && ['NotFoundError', 'DevicesNotFoundError', 'OverconstrainedError', 'ConstraintNotSatisfiedError'].includes(error?.name)) {
            retried = true; excluded.add(choice.deviceId);
            if (this.preference !== AUTO) this._failedPreferred = this.preference;
            await this.refresh().catch(() => {});
            this.message = 'Microphone đã chọn chưa khả dụng. Đang dùng Tự động; lựa chọn của bạn được giữ lại.'; this._emit();
            requestedDeviceId = this._choice(excluded, true)?.deviceId || '';
            return this._capture(requestedDeviceId, audio);
          }
          throw error;
        }
      };
      try {
        await this.refresh().catch(() => {});
        stream = await capture(this._choice(excluded));
        await this.refresh().catch(() => {});
        const better = this._choice(excluded, true);
        const autoMode = this.preference === AUTO || this.getState().missingPreferredDevice;
        if (autoMode && better && requestedDeviceId !== better.deviceId && !streamMatches(stream, better)) {
          this._release(stream); stream = null;
          stream = await capture(better);
          await this.refresh().catch(() => {});
        }
        if (this._destroyed) throw errorNamed('AbortError', 'Microphone manager was closed');
        this._pendingStreams.delete(stream); this._emit(); return stream;
      } catch (error) {
        if (stream) this._release(stream);
        this._permissionError(error); throw error;
      }
    }
    destroy() {
      this._destroyed = true;
      for (const stream of this._pendingStreams) stopStream(stream);
      this._pendingStreams.clear(); this._listeners.clear();
      this.mediaDevices?.removeEventListener?.('devicechange', this._onDeviceChange);
      this.document?.removeEventListener?.('visibilitychange', this._onVisible);
      this.eventTarget?.removeEventListener?.('storage', this._onStorage);
    }
  }

  function manager() { if (!singleton) singleton = new MicrophoneManager(); return singleton; }
  function refreshPickers() { for (const render of renders) render(); }
  function pickerLocked() { for (const check of locks) { try { if (check()) return true; } catch (_) { return true; } } return false; }
  function pickerError(error) {
    if (['NotAllowedError', 'PermissionDeniedError'].includes(error?.name)) return 'Chưa có quyền dùng microphone. Hãy cho phép trong trình duyệt.';
    if (['NotFoundError', 'DevicesNotFoundError'].includes(error?.name)) return 'Chưa tìm thấy microphone. Hãy kiểm tra kết nối và làm mới danh sách.';
    if (error?.name === 'NotReadableError') return 'Microphone đang bận hoặc không mở được. Hãy đóng ứng dụng khác đang dùng mic.';
    return 'Chưa đọc được danh sách microphone. Hãy kiểm tra quyền và thử lại.';
  }

  function mountPicker({ hostId, isLocked = () => false, onError = () => {}, onStatus = () => {}, microphoneManager } = {}) {
    const document = root.document, host = typeof hostId === 'string' ? document?.getElementById(hostId) : hostId;
    if (!host || !document) return { refresh: refreshPickers, destroy() {} };
    const controller = microphoneManager || manager(), menu = document.createElement('div'), button = document.createElement('button');
    const menuId = `mdd-microphone-menu-${++nextPicker}`;
    const lock = () => Boolean(isLocked()); locks.add(lock);
    host.classList.add('mic-picker'); button.type = 'button'; button.className = 'mic-picker__toggle';
    button.setAttribute('aria-label', 'Chọn microphone'); button.setAttribute('aria-haspopup', 'menu');
    button.setAttribute('aria-controls', menuId); button.setAttribute('aria-expanded', 'false');
    button.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 10v2a6 6 0 0 0 12 0v-2M12 18v3M8 21h8"/></svg>';
    menu.id = menuId; menu.className = 'mic-picker__menu'; menu.setAttribute('role', 'menu'); menu.hidden = true;
    host.appendChild(button); document.body.appendChild(menu);
    let state = controller.getState(), lastMessage = '', destroyed = false;
    function close(focus = false) { menu.hidden = true; button.setAttribute('aria-expanded', 'false'); if (focus) button.focus(); }
    function position() {
      if (menu.hidden) return;
      const rect = button.getBoundingClientRect(), width = Math.min(320, Math.max(180, root.innerWidth - 24));
      menu.style.position = 'fixed'; menu.style.width = `${width}px`;
      menu.style.left = `${Math.max(12, Math.min(rect.left, root.innerWidth - width - 12))}px`;
      const height = Math.min(menu.scrollHeight || 300, root.innerHeight - 24);
      menu.style.maxHeight = `${root.innerHeight - 24}px`;
      menu.style.top = `${Math.max(12, Math.min(rect.top >= height + 20 ? rect.top - height - 8 : rect.bottom + 8, root.innerHeight - height - 12))}px`;
    }
    function option(id, title, detail, container) {
      const item = document.createElement('button'); item.type = 'button'; item.className = 'mic-picker__option';
      item.dataset.deviceId = id;
      const checked = id === AUTO ? state.preference === AUTO || state.missingPreferredDevice : !state.missingPreferredDevice && state.preference === id;
      item.setAttribute('role', 'menuitemradio'); item.setAttribute('aria-checked', String(checked));
      const name = document.createElement('strong'), subtitle = document.createElement('small'); name.textContent = title; subtitle.textContent = detail;
      item.append(name, subtitle); item.addEventListener('click', () => { if (pickerLocked()) return refreshPickers(); controller.select(id); close(true); });
      container.appendChild(item);
    }
    function render() {
      if (destroyed) return;
      const locked = pickerLocked(); button.disabled = locked;
      button.title = `Chọn microphone · ${state.resolvedLabel || (state.mode === AUTO ? 'Tự động' : 'Thiết bị đã chọn')}`;
      if (locked) close();
      const focusedId = document.activeElement?.getAttribute?.('data-device-id');
      menu.replaceChildren();
      const title = document.createElement('div'); title.className = 'mic-picker__title'; title.textContent = 'Microphone'; menu.appendChild(title);
      const options = document.createElement('div'); options.className = 'mic-picker__options';
      menu.appendChild(options);
      option(AUTO, 'Tự động · Ưu tiên mic ngoài', state.resolvedLabel || 'Ưu tiên USB / tai nghe có dây; dùng mic mặc định khi cần.', options);
      for (const [index, device] of state.devices.entries()) option(device.deviceId, device.label || `Microphone ${index + 1}`,
        device.isDefault ? 'Microphone mặc định của hệ thống' : device.alias ? 'Lựa chọn của hệ thống' : 'Thiết bị thu âm', options);
      if (state.needsPermission) {
        const permission = document.createElement('button'); permission.type = 'button'; permission.className = 'mic-picker__permission';
        permission.disabled = state.permissionPending;
        permission.textContent = state.permissionPending ? 'Đang xin quyền microphone…' : 'Cho phép micro để xem tên thiết bị';
        permission.addEventListener('click', async () => {
          if (pickerLocked()) return refreshPickers();
          permission.disabled = true;
          try { await controller.requestPermission(); } catch (error) { onError(pickerError(error)); }
          render(); position();
        });
        menu.appendChild(permission);
      }
      const status = document.createElement('div'); status.className = 'mic-picker__status'; status.setAttribute('role', 'status');
      status.textContent = state.message || 'Tự động chọn theo loại thiết bị; không đo chất lượng âm thanh.'; menu.appendChild(status);
      const refresh = document.createElement('button'); refresh.type = 'button'; refresh.className = 'mic-picker__refresh'; refresh.textContent = 'Làm mới danh sách';
      refresh.addEventListener('click', async () => { try { await controller.refresh(); } catch (error) { onError(pickerError(error)); } }); menu.appendChild(refresh);
      if (!menu.hidden && focusedId) [...menu.querySelectorAll('[data-device-id]')].find(item => item.dataset.deviceId === focusedId)?.focus();
      position();
    }
    const unsubscribe = controller.subscribe(value => {
      state = value; render();
      if (value.message && value.message !== lastMessage) { lastMessage = value.message; onStatus(value.message); }
    });
    renders.add(render);
    button.addEventListener('click', async () => {
      if (pickerLocked()) return refreshPickers();
      if (!menu.hidden) return close();
      menu.hidden = false; button.setAttribute('aria-expanded', 'true'); render(); position();
      try { await controller.refresh(); } catch (error) { onError(pickerError(error)); }
      if (!menu.hidden) menu.querySelector('[role="menuitemradio"]')?.focus();
    });
    function keydown(event) {
      if (menu.hidden) return;
      if (event.key === 'Escape') { event.preventDefault(); close(true); return; }
      if (event.key === 'Tab') { close(); return; }
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); const items = [...menu.querySelectorAll('button:not(:disabled)')];
      const current = items.indexOf(document.activeElement);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 :
        (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[index]?.focus();
    }
    function outside(event) { if (!host.contains(event.target) && !menu.contains(event.target)) close(); }
    document.addEventListener('pointerdown', outside); document.addEventListener('keydown', keydown);
    root.addEventListener?.('resize', position); root.addEventListener?.('scroll', position, true);
    return { refresh: refreshPickers, destroy() {
      destroyed = true; renders.delete(render); locks.delete(lock); unsubscribe(); close(); refreshPickers();
      document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', keydown);
      root.removeEventListener?.('resize', position); root.removeEventListener?.('scroll', position, true); button.remove(); menu.remove();
    } };
  }

  return { MicrophoneManager, normalizeDevices, rankDevices, STORAGE_KEY, AUTO, AUDIO, mountPicker, refreshPickers,
    getStream: options => manager().getStream(options), getState: () => manager().getState(),
    subscribe: listener => manager().subscribe(listener), refresh: () => manager().refresh(), select: value => manager().select(value) };
});
