// 运行控制台的“运动调参”卡片：高频 motion/servo 参数步进调节。
// 读写完全复用 /api/config/execution，保存后下一轮识别自动生效，无需重启节点。

const GROUPS = [
  {
    title: '速度',
    fields: [
      { path: 'motion.arm_speed', step: 5 },
      { path: 'motion.pick_speed', step: 5 },
      { path: 'motion.servo_speed', step: 5 },
      { path: 'motion.pick_approach_speed', step: 5 },
    ],
  },
  {
    title: '抓取',
    fields: [
      { path: 'motion.pick_surface_offset_mm', step: 0.5 },
      { path: 'motion.pick_approach_clearance_mm', step: 0.5 },
      { path: 'motion.pick_retreat_blend_radius_mm', step: 0.5 },
      { path: 'motion.pick_rotate_safe_lift_mm', step: 0.5 },
    ],
  },
  {
    title: '摆放',
    fields: [
      { path: 'motion.place_descent_offset_mm', step: 0.5 },
      { path: 'motion.place_descent_blend_radius_mm', step: 0.5 },
      { path: 'motion.place_lift_blend_radius_mm', step: 0.5 },
    ],
  },
  {
    title: '视觉伺服',
    fields: [
      { path: 'servo.block_error_threshold_px', step: 0.1 },
      { path: 'servo.tray_error_threshold_px', step: 0.1 },
      { path: 'servo.min_step_mm', step: 0.05 },
      { path: 'servo.max_step_mm', step: 0.5 },
      { path: 'servo.settle_sec', step: 0.1 },
    ],
  },
  {
    title: '安全',
    fields: [
      { path: 'motion.minimum_tcp_z_mm', step: 0.5 },
    ],
  },
];

function valueAt(root, path) {
  return path.split('.').reduce((current, key) => current?.[key], root);
}

function setValueAt(root, path, value) {
  const keys = path.split('.');
  let current = root;
  for (const key of keys.slice(0, -1)) current = current[key];
  current[keys.at(-1)] = value;
}

function decimalsOf(step) {
  const text = String(step);
  const dot = text.indexOf('.');
  return dot === -1 ? 0 : text.length - dot - 1;
}

export class TuneCard {
  constructor(hooks) {
    this.hooks = hooks; // { toast, confirmAction, formatError, lockReason }
    this.document = null;
    this.values = new Map();
    this.dirty = new Set();
    this.saving = false;
    this.body = document.querySelector('#tune-body');
    this.dirtyBadge = document.querySelector('#tune-dirty');
    this.reloadButton = document.querySelector('#tune-reload');
    this.resetButton = document.querySelector('#tune-reset');
    this.saveButton = document.querySelector('#tune-save');
    this.reloadButton?.addEventListener('click', () => this.refresh());
    this.resetButton?.addEventListener('click', () => this.reset());
    this.saveButton?.addEventListener('click', () => this.save());
  }

  original(path) {
    return this.document ? valueAt(this.document.data, path) : undefined;
  }

  metadata(path) {
    return this.document?.schema?.[path] || {};
  }

  clamp(path, value) {
    const metadata = this.metadata(path);
    if (Number.isFinite(value)) {
      if (metadata.min !== undefined) value = Math.max(Number(metadata.min), value);
      if (metadata.max !== undefined) value = Math.min(Number(metadata.max), value);
    }
    return value;
  }

  applyStep(path, step, direction, decimals) {
    const current = Number(this.values.get(path));
    if (!Number.isFinite(current)) return;
    const next = this.clamp(path, Number((current + direction * step).toFixed(decimals + 1)));
    this.setLocalValue(path, next);
  }

  setLocalValue(path, value) {
    this.values.set(path, value);
    const unchanged = JSON.stringify(this.original(path)) === JSON.stringify(value);
    if (unchanged) this.dirty.delete(path);
    else this.dirty.add(path);
    this.renderRow(path);
    this.renderFooter();
  }

  async refresh() {
    try {
      this.document = await this.hooks.api('/api/config/execution');
    } catch (error) {
      this.body.innerHTML = '';
      const note = document.createElement('p');
      note.className = 'muted';
      note.textContent = `读取 execution.yaml 失败：${this.hooks.formatError(error)}`;
      this.body.append(note);
      this.dirtyBadge.textContent = '读取失败';
      return;
    }
    this.values = new Map();
    this.dirty = new Set();
    for (const group of GROUPS) {
      for (const field of group.fields) {
        this.values.set(field.path, valueAt(this.document.data, field.path));
      }
    }
    this.render();
  }

  reset() {
    if (!this.document) return;
    for (const [path] of this.values) this.values.set(path, this.original(path));
    this.dirty.clear();
    this.render();
  }

  changes() {
    if (!this.document) return [];
    const result = [];
    for (const [path] of this.values) {
      const before = this.original(path);
      const after = this.values.get(path);
      if (JSON.stringify(before) !== JSON.stringify(after)) {
        result.push({ path, before, after, metadata: this.metadata(path) });
      }
    }
    return result;
  }

  differenceText(changes) {
    return changes.map(({ path, before, after, metadata }) => {
      const risk = metadata.risk === 'danger' ? '〔危险参数〕 ' : '';
      const label = metadata.label ? `（${metadata.label}）` : '';
      return `${risk}${path}${label}\n  修改前：${before}\n  修改后：${after}`;
    }).join('\n');
  }

  async save() {
    if (!this.document || this.saving) return;
    const changes = this.changes();
    if (!changes.length) {
      this.hooks.toast('当前没有修改', '', 'warning');
      return;
    }
    const lockReason = this.hooks.lockReason();
    if (lockReason) {
      this.hooks.toast('暂时不能保存参数', lockReason, 'warning');
      return;
    }
    let confirmDangerous = false;
    if (changes.some((item) => item.metadata.risk === 'danger')) {
      confirmDangerous = await this.hooks.confirmAction(
        '保存危险参数修改？',
        '修改包含速度、高度或安全范围参数，保存前请再次核对现场安全。',
        this.differenceText(changes),
      );
      if (!confirmDangerous) return;
    }
    const data = structuredClone(this.document.data);
    for (const { path, after } of changes) setValueAt(data, path, after);
    this.saving = true;
    this.renderFooter();
    try {
      await this.hooks.api('/api/config/execution', {
        method: 'PUT',
        json: {
          data,
          revision: this.document.revision,
          confirm_dangerous: confirmDangerous,
          apply: false,
        },
      });
      this.hooks.toast('已保存，下一轮识别生效', `${changes.length} 项参数已写入 execution.yaml`);
      await this.refresh();
    } catch (error) {
      if (error.code === 'revision_conflict') {
        this.hooks.toast('文件版本冲突', 'execution.yaml 已被其它程序修改，已重新读取，请重新调整后保存。', 'error', 7000);
        await this.refresh();
      } else if (error.code === 'dangerous_confirmation_required') {
        const retry = await this.hooks.confirmAction('保存危险参数修改？', error.message, this.differenceText(changes));
        if (!retry) return;
        try {
          await this.hooks.api('/api/config/execution', {
            method: 'PUT',
            json: { data, revision: this.document.revision, confirm_dangerous: true, apply: false },
          });
          this.hooks.toast('已保存，下一轮识别生效', `${changes.length} 项参数已写入 execution.yaml`);
          await this.refresh();
        } catch (retryError) {
          this.hooks.toast('配置未保存', this.hooks.formatError(retryError), 'error', 7000);
        }
      } else {
        this.hooks.toast('配置未保存', this.hooks.formatError(error), 'error', 7000);
      }
    } finally {
      this.saving = false;
      this.renderFooter();
    }
  }

  updateLock() {
    if (!this.document) return;
    const reason = this.hooks.lockReason();
    this.saveButton.disabled = Boolean(reason) || this.saving;
    this.saveButton.title = reason;
  }

  render() {
    this.body.innerHTML = '';
    if (!this.document) return;
    for (const group of GROUPS) {
      const box = document.createElement('div');
      box.className = 'tune-group';
      const title = document.createElement('strong');
      title.textContent = group.title;
      box.append(title);
      const fields = document.createElement('div');
      fields.className = 'tune-fields';
      for (const field of group.fields) fields.append(this.buildRow(field));
      box.append(fields);
      this.body.append(box);
    }
    this.renderFooter();
    this.updateLock();
  }

  renderRow(path) {
    const row = this.body.querySelector(`[data-path="${CSS.escape(path)}"]`);
    if (!row) return;
    const input = row.querySelector('input');
    const value = this.values.get(path);
    input.value = value ?? '';
    row.classList.toggle('modified', this.dirty.has(path));
  }

  buildRow(field) {
    const path = field.path;
    const metadata = this.metadata(path);
    const value = this.values.get(path);
    const row = document.createElement('div');
    row.className = `tune-field risk-${metadata.risk || 'normal'}`;
    row.dataset.path = path;

    const label = document.createElement('span');
    label.className = 'tune-label';
    const name = document.createElement('b');
    name.textContent = metadata.label || path.split('.').pop();
    label.append(name);
    if (metadata.unit) {
      const unit = document.createElement('small');
      unit.textContent = metadata.unit;
      label.append(unit);
    }
    row.append(label);

    const stepper = document.createElement('div');
    stepper.className = 'tune-stepper';
    const decimals = decimalsOf(field.step);
    const integer = metadata.type === 'integer' || Number.isInteger(field.step) && Number.isInteger(value);
    const input = document.createElement('input');
    input.type = 'number';
    input.step = field.step;
    if (metadata.min !== undefined) input.min = metadata.min;
    if (metadata.max !== undefined) input.max = metadata.max;
    input.value = value ?? '';
    input.disabled = value === undefined;
    const makeStepButton = (direction, symbol) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tune-step';
      button.textContent = symbol;
      button.disabled = value === undefined;
      button.addEventListener('click', () => this.applyStep(path, field.step, direction, decimals));
      return button;
    };
    stepper.append(makeStepButton(-1, '−'), input, makeStepButton(1, '＋'));
    input.addEventListener('input', () => {
      if (input.value === '') return;
      const parsed = integer ? Number.parseInt(input.value, 10) : Number(input.value);
      if (!Number.isFinite(parsed)) return;
      this.setLocalValue(path, this.clamp(path, parsed));
    });
    row.append(stepper);
    return row;
  }

  renderFooter() {
    if (!this.document) return;
    const count = this.dirty.size;
    this.dirtyBadge.textContent = count ? `${count} 项未保存` : '与文件一致';
    this.dirtyBadge.classList.toggle('dirty', count > 0);
    this.saveButton.disabled = this.saving || Boolean(this.hooks.lockReason());
    this.saveButton.textContent = this.saving ? '保存中…' : '保存（下一轮生效）';
    this.resetButton.disabled = this.saving || count === 0;
  }
}
