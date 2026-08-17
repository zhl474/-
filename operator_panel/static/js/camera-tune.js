// 「07 相机调参」视图：运行时曝光/增益调节 + YOLO 识别画面预览。
// 曝光写入走 /api/camera/exposure（ROS /camera/set_exposure_param），流不重启；
// YOLO 预览由面板后端 1Hz 拉取 /perception/yolo_preview，SSE image 事件推送更新。

const SENSORS = ['rgb', 'depth'];
const NUMERIC_KEYS = ['exposure', 'gain'];
const KEY_LABELS = { exposure: '曝光', gain: '增益' };
const CONFIG_SECTIONS = { rgb: 'rgb_camera', depth: 'depth_camera' };

export class CameraTuneCard {
  constructor(hooks) {
    this.hooks = hooks; // { api, toast, confirmAction, formatError }
    this.state = null; // GET /api/camera/exposure 的结果
    this.previewEnabled = false;
    this.pending = new Set(); // 正在提交的 sensor.key，防止重复发送
    this.refreshButton = document.querySelector('#ct-refresh');
    this.yoloToggleButton = document.querySelector('#ct-yolo-toggle');
    this.persistButton = document.querySelector('#ct-persist');
    this.refreshButton?.addEventListener('click', () => this.refresh());
    this.yoloToggleButton?.addEventListener('click', () => this.setYoloPreview(!this.previewEnabled));
    this.persistButton?.addEventListener('click', () => this.persistToConfig());
    for (const card of document.querySelectorAll('.ct-sensor-card')) {
      const sensor = card.dataset.sensor;
      card.querySelector('input[data-key="auto_exposure"]')?.addEventListener('change', (event) => {
        this.setValue(sensor, 'auto_exposure', event.target.checked ? 1 : 0);
      });
    }
    this.renderYoloToggle();
  }

  async enter() {
    // 进入视图只同步状态，不自动开 YOLO 预览：识别刷新必须手动点击开启，
    // 避免占用 GPU 或在感知未启动时反复重试。
    try {
      const state = await this.hooks.api('/api/camera/yolo-preview');
      this.previewEnabled = Boolean(state.enabled);
      this.renderYoloToggle();
    } catch (_error) { /* 状态同步失败不影响曝光调参 */ }
    await this.refresh();
  }

  leave() {
    this.setYoloPreview(false);
  }

  pauseForTask() {
    // 正式识别/任务执行期间自动让路（只关不自动重开，结束后手动再开）。
    if (this.previewEnabled) this.setYoloPreview(false);
  }

  onYoloFrame(data) {
    const summary = document.querySelector('#ct-yolo-summary');
    const time = document.querySelector('#ct-yolo-time');
    if (data.summary && summary) summary.textContent = data.summary;
    if (data.updated_at && time) time.textContent = data.updated_at;
  }

  async refresh() {
    let state;
    try {
      state = await this.hooks.api('/api/camera/exposure');
    } catch (error) {
      this.renderUnavailable(this.hooks.formatError(error));
      return;
    }
    this.state = state;
    this.render();
  }

  renderUnavailable(message) {
    this.state = null;
    for (const box of document.querySelectorAll('.ct-fields')) {
      box.innerHTML = '';
      const note = document.createElement('p');
      note.className = 'muted';
      note.textContent = `读取相机曝光状态失败：${message}`;
      box.append(note);
    }
    for (const input of document.querySelectorAll('.ct-sensor-card input[data-key="auto_exposure"]')) {
      input.checked = false;
      input.disabled = true;
    }
  }

  render() {
    for (const sensor of SENSORS) {
      const sensorState = this.state?.[sensor];
      const box = document.querySelector(`.ct-fields[data-sensor="${sensor}"]`);
      if (!box || !sensorState) continue;
      const aeInput = document.querySelector(
        `.ct-sensor-card[data-sensor="${sensor}"] input[data-key="auto_exposure"]`,
      );
      if (aeInput) {
        aeInput.checked = Boolean(sensorState.auto_exposure);
        aeInput.disabled = false;
      }
      box.innerHTML = '';
      for (const key of NUMERIC_KEYS) box.append(this.buildSliderRow(sensor, key, sensorState));
    }
  }

  buildSliderRow(sensor, key, sensorState) {
    const minimum = Number(sensorState[`${key}_min`]);
    const maximum = Number(sensorState[`${key}_max`]);
    const row = document.createElement('div');
    row.className = 'ct-field';
    row.dataset.sensor = sensor;
    row.dataset.key = key;

    const label = document.createElement('span');
    label.className = 'ct-label';
    const name = document.createElement('b');
    name.textContent = KEY_LABELS[key] || key;
    const range = document.createElement('small');
    range.className = 'ct-range';
    range.textContent = `${minimum} ~ ${maximum}`;
    label.append(name, range);
    row.append(label);

    const control = document.createElement('div');
    control.className = 'ct-control';
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = minimum;
    slider.max = maximum;
    slider.step = 1;
    slider.value = sensorState[key];
    const number = document.createElement('input');
    number.type = 'number';
    number.min = minimum;
    number.max = maximum;
    number.step = 1;
    number.value = sensorState[key];
    control.append(slider, number);
    row.append(control);

    const submit = (raw) => {
      const value = Math.round(Number(raw));
      if (!Number.isFinite(value)) return;
      this.setValue(sensor, key, value, { slider, number });
    };
    // 用 change（滑杆松手/输入框回车或失焦）而不是 input，避免拖动过程反复写硬件。
    slider.addEventListener('change', () => submit(slider.value));
    number.addEventListener('change', () => submit(number.value));
    return row;
  }

  async setValue(sensor, key, value, elements = null) {
    if (!this.state?.[sensor]) return;
    const token = `${sensor}.${key}`;
    if (this.pending.has(token)) return;
    this.pending.add(token);
    try {
      // 手动曝光/增益会被自动曝光覆盖；后端也会兜底关闭，这里先发一次让界面状态同步。
      if (key !== 'auto_exposure' && this.state[sensor].auto_exposure) {
        try {
          await this.hooks.api('/api/camera/exposure', {
            method: 'POST',
            json: { sensor, key: 'auto_exposure', value: 0 },
          });
          this.state[sensor].auto_exposure = false;
          const aeInput = document.querySelector(
            `.ct-sensor-card[data-sensor="${sensor}"] input[data-key="auto_exposure"]`,
          );
          if (aeInput) aeInput.checked = false;
        } catch (error) {
          this.hooks.toast('关闭自动曝光失败', this.hooks.formatError(error), 'error');
          return;
        }
      }
      const result = await this.hooks.api('/api/camera/exposure', {
        method: 'POST',
        json: { sensor, key, value },
      });
      if (key === 'auto_exposure') {
        this.state[sensor].auto_exposure = Boolean(result.applied);
      } else {
        this.state[sensor][key] = result.applied;
        if (elements?.slider) elements.slider.value = result.applied;
        if (elements?.number) elements.number.value = result.applied;
      }
      if (result.message && result.message !== '设置成功') {
        this.hooks.toast('相机参数已设置', result.message);
      }
    } catch (error) {
      this.hooks.toast('设置相机参数失败', this.hooks.formatError(error), 'error', 7000);
    } finally {
      this.pending.delete(token);
    }
  }

  async setYoloPreview(enabled) {
    const target = Boolean(enabled);
    if (this.previewEnabled === target) return;
    try {
      await this.hooks.api('/api/camera/yolo-preview', {
        method: 'POST',
        json: { enabled: target },
      });
    } catch (error) {
      this.hooks.toast('YOLO 预览开关失败', this.hooks.formatError(error), 'error');
      return;
    }
    this.previewEnabled = target;
    this.renderYoloToggle();
  }

  renderYoloToggle() {
    if (!this.yoloToggleButton) return;
    this.yoloToggleButton.textContent = this.previewEnabled ? '暂停识别刷新' : '开始识别刷新';
  }

  async persistToConfig() {
    if (!this.state) {
      this.hooks.toast('尚未读取相机状态', '请先点「读取当前值」', 'warning');
      return;
    }
    let configDocument;
    try {
      configDocument = await this.hooks.api('/api/config/camera');
    } catch (error) {
      this.hooks.toast('读取相机配置失败', this.hooks.formatError(error), 'error', 7000);
      return;
    }
    const data = structuredClone(configDocument.data);
    const lines = [];
    for (const sensor of SENSORS) {
      const section = data[CONFIG_SECTIONS[sensor]];
      if (!section) continue;
      section.auto_exposure = Boolean(this.state[sensor].auto_exposure);
      section.exposure = Number(this.state[sensor].exposure);
      section.gain = Number(this.state[sensor].gain);
      lines.push(
        `${sensor}: 自动曝光=${section.auto_exposure ? '开' : '关'}`
        + ` 曝光=${section.exposure} 增益=${section.gain}`,
      );
    }
    const confirmed = await this.hooks.confirmAction(
      '写入相机配置？',
      '把当前曝光/增益值保存到 camera 配置文件，下次重启硬件后按此值启动。',
      lines.join('\n'),
    );
    if (!confirmed) return;
    try {
      await this.hooks.api('/api/config/camera', {
        method: 'PUT',
        json: {
          data,
          revision: configDocument.revision,
          confirm_dangerous: false,
          apply: false,
        },
      });
      this.hooks.toast('相机配置已写入', '下次重启硬件后按保存值启动');
    } catch (error) {
      if (error.code === 'revision_conflict') {
        this.hooks.toast('文件版本冲突', 'camera 配置已被其它程序修改，请重试。', 'error', 7000);
      } else {
        this.hooks.toast('相机配置未保存', this.hooks.formatError(error), 'error', 7000);
      }
    }
  }
}
