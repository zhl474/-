function clone(value) {
  return structuredClone(value);
}

function valueAt(root, path) {
  const tokens = [...path.matchAll(/(?:^|\.)([^.\[]+)|\[(\d+)\]/g)]
    .map((match) => (match[2] !== undefined ? Number(match[2]) : match[1]));
  return tokens.reduce((current, token) => current[token], root);
}

function setValue(root, path, value) {
  const tokens = [...path.matchAll(/(?:^|\.)([^.\[]+)|\[(\d+)\]/g)]
    .map((match) => (match[2] !== undefined ? Number(match[2]) : match[1]));
  let current = root;
  for (const token of tokens.slice(0, -1)) current = current[token];
  current[tokens.at(-1)] = value;
}

function stable(value) {
  return JSON.stringify(value);
}

function formatValue(value) {
  if (typeof value === 'string') return `“${value}”`;
  if (value === null) return 'null';
  return JSON.stringify(value);
}

function createBadge(text, className = '') {
  const badge = document.createElement('span');
  badge.className = `field-badge ${className}`.trim();
  badge.textContent = text;
  return badge;
}

export class ConfigEditor {
  constructor(container, onChange) {
    this.container = container;
    this.onChange = onChange;
    this.document = null;
    this.original = null;
    this.data = null;
  }

  load(document) {
    this.document = document;
    this.original = clone(document.data);
    this.data = clone(document.data);
    this.render();
  }

  changedFields() {
    if (!this.document) return [];
    const result = [];
    for (const [path, metadata] of Object.entries(this.document.schema)) {
      if (['object', 'array'].includes(metadata.type)) continue;
      const before = valueAt(this.original, path);
      const after = valueAt(this.data, path);
      if (stable(before) !== stable(after)) result.push({ path, before, after, metadata });
    }
    return result;
  }

  differenceText() {
    const changes = this.changedFields();
    if (!changes.length) return '当前没有未保存修改。';
    return changes.map(({ path, before, after, metadata }) => {
      const risk = metadata.risk === 'danger' ? '〔危险参数〕 ' : '';
      return `${risk}${path}\n  修改前：${formatValue(before)}\n  修改后：${formatValue(after)}`;
    }).join('\n\n');
  }

  hasDangerousChanges() {
    return this.changedFields().some((item) => item.metadata.risk === 'danger');
  }

  reset() {
    if (!this.document) return;
    this.data = clone(this.original);
    this.render();
    this.onChange?.(this.changedFields());
  }

  render() {
    this.container.className = 'config-editor';
    this.container.replaceChildren();
    if (!this.document) return;
    for (const [key, value] of Object.entries(this.data)) {
      this.container.append(this.renderNode(value, key, key, 0));
    }
  }

  renderNode(value, path, label, depth) {
    const metadata = this.document.schema[path] || {};
    if (value !== null && typeof value === 'object') {
      const details = document.createElement('details');
      details.className = `config-group depth-${Math.min(depth, 3)}`;
      details.open = depth < 2;
      details.dataset.path = path;
      const summary = document.createElement('summary');
      const title = document.createElement('span');
      const titleText = document.createElement('strong');
      titleText.textContent = metadata.label || label;
      const pathText = document.createElement('code');
      pathText.textContent = path;
      title.append(titleText, pathText);
      summary.append(title);
      const count = Array.isArray(value) ? `${value.length} 项固定数组` : `${Object.keys(value).length} 个字段`;
      summary.append(createBadge(count));
      if (metadata.risk === 'danger') summary.append(createBadge('危险', 'danger'));
      if (metadata.description) {
        const note = document.createElement('small');
        note.textContent = metadata.description;
        summary.append(note);
      }
      details.append(summary);
      const body = document.createElement('div');
      body.className = 'config-group-body';
      if (Array.isArray(value)) {
        value.forEach((item, index) => body.append(this.renderNode(item, `${path}[${index}]`, `第 ${index + 1} 项`, depth + 1)));
      } else {
        Object.entries(value).forEach(([key, item]) => body.append(this.renderNode(item, `${path}.${key}`, key, depth + 1)));
      }
      details.append(body);
      return details;
    }

    const row = document.createElement('label');
    row.className = `config-field risk-${metadata.risk || 'normal'}`;
    row.dataset.path = path;
    const info = document.createElement('span');
    info.className = 'field-info';
    const titleLine = document.createElement('span');
    titleLine.className = 'field-title';
    const title = document.createElement('strong');
    title.textContent = metadata.label || label;
    titleLine.append(title);
    if (metadata.unit) titleLine.append(createBadge(metadata.unit));
    if (metadata.risk === 'danger') titleLine.append(createBadge('危险参数', 'danger'));
    else if (metadata.risk === 'warning') titleLine.append(createBadge('需留意', 'warning'));
    if (metadata.expert) titleLine.append(createBadge('专家参数', 'expert'));
    info.append(titleLine);
    const code = document.createElement('code');
    code.textContent = path;
    info.append(code);
    if (metadata.description) {
      const description = document.createElement('small');
      description.textContent = metadata.description;
      info.append(description);
    }
    row.append(info);
    row.append(this.createControl(value, path, metadata));
    return row;
  }

  createControl(value, path, metadata) {
    let control;
    if (metadata.options?.length) {
      control = document.createElement('select');
      metadata.options.forEach((option) => {
        const item = document.createElement('option');
        item.value = option;
        item.textContent = option;
        item.selected = value === option;
        control.append(item);
      });
    } else if (metadata.type === 'boolean') {
      control = document.createElement('select');
      control.innerHTML = '<option value="true">开启 / true</option><option value="false">关闭 / false</option>';
      control.value = String(value);
    } else if (['integer', 'number'].includes(metadata.type)) {
      control = document.createElement('input');
      control.type = 'number';
      control.step = metadata.type === 'integer' ? '1' : 'any';
      if (metadata.min !== undefined) control.min = metadata.min;
      if (metadata.max !== undefined) control.max = metadata.max;
      control.value = value;
    } else {
      control = document.createElement('input');
      control.type = 'text';
      control.value = value ?? '';
      if (metadata.type === 'path') control.classList.add('path-input');
    }
    control.classList.add('field-control');
    control.dataset.path = path;
    control.addEventListener('input', () => {
      let next;
      if (metadata.type === 'boolean') next = control.value === 'true';
      else if (metadata.type === 'integer') next = control.value === '' ? 0 : Number.parseInt(control.value, 10);
      else if (metadata.type === 'number') next = control.value === '' ? 0 : Number(control.value);
      else next = control.value;
      setValue(this.data, path, next);
      const original = valueAt(this.original, path);
      control.closest('.config-field')?.classList.toggle('modified', stable(original) !== stable(next));
      this.onChange?.(this.changedFields());
    });
    return control;
  }

  focusPath(path) {
    const escaped = CSS.escape(path);
    const target = this.container.querySelector(`[data-path="${escaped}"]`);
    if (!target) return false;
    let parent = target.parentElement;
    while (parent) {
      if (parent.tagName === 'DETAILS') parent.open = true;
      parent = parent.parentElement;
    }
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.add('highlight');
    setTimeout(() => target.classList.remove('highlight'), 1800);
    return true;
  }
}

export function flattenSearch(document, query) {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  return Object.entries(document.schema)
    .filter(([, metadata]) => !['object', 'array'].includes(metadata.type))
    .map(([path, metadata]) => ({
      fileId: document.file_id,
      fileLabel: document.label,
      path,
      metadata,
      value: valueAt(document.data, path),
    }))
    .filter((item) => [
      item.fileId, item.fileLabel, item.path, item.metadata.label,
      item.metadata.description, formatValue(item.value),
    ].join(' ').toLowerCase().includes(needle));
}
