import { api, ApiError, cacheBustedImage, eventStream } from './api.js';
import { ConfigEditor, flattenSearch } from './config-editor.js';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const app = {
  state: null,
  selectedView: 'console',
  selectedDebugImage: 'block_mask',
  order: [0, 1, 2, 3, 4, 5, 6],
  categories: ['L_blue', 'L_yellow', 'z_blue', 'z_green', 'square', 'T', 'line'],
  categoryLabels: {
    L_blue: '蓝 L', L_yellow: '黄 L', z_blue: '蓝 Z', z_green: '绿 Z',
    square: '方块', T: 'T 形', line: '长条',
  },
  configFiles: [],
  configDocuments: new Map(),
  currentConfigId: null,
  executionConfig: null,
  pose: { tcp: null, camera: null },
  logs: [],
  logRenderScheduled: false,
  selectedLogChannel: 'all',
  launchLogKind: 'hardware',
  seenOperations: new Set(),
  interactionDialogKey: '',
  interactionDialogKind: '',
  interactionSubmitting: false,
  interactionSuppressed: new Set(),
  promptDeadline: 0,
  promptDeadlineId: '',
  rosSystemLoading: false,
  rosSystemLastLoaded: 0,
  usbOccupancy: null,
  usbOccupancyLoading: false,
  usbOccupancyPromise: null,
  usbOccupancyLastLoaded: 0,
};

const imageViewer = {
  dialog: null,
  stage: null,
  canvas: null,
  ctx: null,
  sourceElement: null,
  image: null,
  offscreen: document.createElement('canvas'),
  offCtx: null,
  sourceUrl: '',
  loadToken: 0,
  fitScale: 1,
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  mouseX: null,
  mouseY: null,
  mouseInside: false,
  dragging: false,
  moved: false,
  dragStartX: 0,
  dragStartY: 0,
  dragOffsetX: 0,
  dragOffsetY: 0,
  locked: null,
  watchTimer: null,
  showGrid: true,
};

const configEditor = new ConfigEditor($('#config-editor'), (changes) => {
  const badge = $('#dirty-count');
  badge.textContent = changes.length ? `${changes.length} 项未保存` : '无修改';
  badge.classList.toggle('dirty', changes.length > 0);
});

function toast(title, message = '', level = 'success', duration = 4500) {
  const region = $('#toast-region');
  const item = document.createElement('div');
  item.className = `toast ${level}`;
  const strong = document.createElement('strong');
  strong.textContent = title;
  item.append(strong);
  if (message) {
    const small = document.createElement('small');
    small.textContent = message;
    item.append(small);
  }
  region.append(item);
  setTimeout(() => item.remove(), duration);
}

function formatError(error) {
  if (error instanceof ApiError) return error.message;
  return error?.message || String(error);
}

async function confirmAction(title, message, detail = '') {
  const dialog = $('#confirm-dialog');
  $('#confirm-title').textContent = title;
  $('#confirm-message').textContent = message;
  const detailElement = $('#confirm-detail');
  detailElement.textContent = detail;
  detailElement.hidden = !detail;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true });
  });
}

function showDiff(title, content) {
  $('#diff-title').textContent = title;
  $('#diff-content').textContent = content || '没有差异。';
  $('#diff-dialog').showModal();
}

async function command(path, json = {}, acceptedMessage = '操作已提交') {
  try {
    const result = await api(path, { method: 'POST', json });
    if (result.operation_id) toast(acceptedMessage, `操作编号 ${result.operation_id.slice(0, 8)}`);
    return result;
  } catch (error) {
    toast('操作未执行', formatError(error), 'error', 6500);
    throw error;
  }
}

function setStatus(name, online, text, warning = false) {
  const chip = $(`[data-status="${name}"]`);
  if (!chip) return;
  chip.classList.toggle('online', Boolean(online));
  chip.classList.toggle('warning', Boolean(warning));
  $('b', chip).textContent = text;
}

function stateClass(state) {
  if (['失败', '已中止'].includes(state)) return 'error';
  if (['准备识别', '等待确认', '可以执行', '执行中'].includes(state)) return 'active';
  return '';
}

function currentPreparePayload() {
  const advanced = $('input[name="task-level"]:checked').value === 'advanced';
  return { advanced, place_order: advanced ? app.order : [] };
}

async function submitPrepare({ confirmRetry = false } = {}) {
  if (confirmRetry && ['等待确认', '可以执行'].includes(app.state?.task?.state)) {
    const confirmed = await confirmAction(
      '重新识别？',
      '当前识别结果将失效，机械臂会重新回到高位拍摄位。',
    );
    if (!confirmed) return null;
  }
  return command('/api/task/prepare', currentPreparePayload(), '识别任务已提交');
}

function rememberInteractionChoice(key) {
  if (!key) return;
  app.interactionSuppressed.add(key);
  if (app.interactionSuppressed.size > 100) {
    app.interactionSuppressed.delete(app.interactionSuppressed.values().next().value);
  }
}

function closeTaskInteraction({ suppress = false } = {}) {
  const dialog = $('#task-interaction-dialog');
  if (suppress) rememberInteractionChoice(app.interactionDialogKey);
  if (dialog.open) dialog.close();
  app.interactionDialogKey = '';
  app.interactionDialogKind = '';
  app.interactionSubmitting = false;
}

function setInteractionButtonsDisabled(disabled) {
  $$('#task-interaction-actions button').forEach((button) => { button.disabled = disabled; });
}

function appendInteractionButton(label, style, handler) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `button ${style}`;
  button.textContent = label;
  button.addEventListener('click', handler);
  $('#task-interaction-actions').append(button);
  return button;
}

function imageViewerFitScale() {
  if (!imageViewer.image || !imageViewer.stage) return 1;
  const width = imageViewer.stage.clientWidth;
  const height = imageViewer.stage.clientHeight;
  if (width < 2 || height < 2) return 1;
  return Math.min(
    width / imageViewer.image.width,
    height / imageViewer.image.height,
  );
}

function imageViewerRefit() {
  if (!imageViewer.image || !imageViewer.stage) return;
  const scale = imageViewerFitScale();
  imageViewer.fitScale = scale;
  imageViewer.scale = scale;
  imageViewer.offsetX = (
    imageViewer.stage.clientWidth - imageViewer.image.width * scale
  ) / 2;
  imageViewer.offsetY = (
    imageViewer.stage.clientHeight - imageViewer.image.height * scale
  ) / 2;
}

function imageViewerResizeCanvas() {
  if (!imageViewer.canvas || !imageViewer.stage || !imageViewer.ctx) return;
  const rect = imageViewer.stage.getBoundingClientRect();
  const cssWidth = Math.max(1, rect.width);
  const cssHeight = Math.max(1, rect.height);
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
  const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));
  if (
    imageViewer.canvas.width !== pixelWidth
    || imageViewer.canvas.height !== pixelHeight
  ) {
    imageViewer.canvas.width = pixelWidth;
    imageViewer.canvas.height = pixelHeight;
  }
  imageViewer.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!imageViewer.image) {
    imageViewer.ctx.clearRect(0, 0, cssWidth, cssHeight);
    return;
  }
  const previousFit = imageViewer.fitScale;
  const wasFit = previousFit > 0
    && Math.abs(imageViewer.scale - previousFit) / previousFit < 0.01;
  imageViewer.fitScale = imageViewerFitScale();
  if (wasFit || imageViewer.scale < imageViewer.fitScale) {
    imageViewerRefit();
  }
  imageViewerDraw();
}

function imageViewerPointFromEvent(event) {
  if (!imageViewer.canvas) return null;
  const rect = imageViewer.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function imageViewerImagePoint(point) {
  if (!imageViewer.image || !point) return null;
  const x = Math.floor((point.x - imageViewer.offsetX) / imageViewer.scale);
  const y = Math.floor((point.y - imageViewer.offsetY) / imageViewer.scale);
  if (x < 0 || y < 0 || x >= imageViewer.image.width || y >= imageViewer.image.height) {
    return null;
  }
  return { x, y };
}

function imageViewerPixelAt(imageX, imageY) {
  if (!imageViewer.offCtx || !imageViewer.image) return null;
  if (
    imageX < 0 || imageY < 0
    || imageX >= imageViewer.image.width
    || imageY >= imageViewer.image.height
  ) {
    return null;
  }
  const data = imageViewer.offCtx.getImageData(imageX, imageY, 1, 1).data;
  return { r: data[0], g: data[1], b: data[2] };
}

function imageViewerUpdateStatus() {
  const coords = $('#image-zoom-coords');
  const locked = $('#image-zoom-locked');
  if (!coords) return;
  const point = imageViewer.mouseInside
    ? { x: imageViewer.mouseX, y: imageViewer.mouseY }
    : null;
  const imagePoint = imageViewerImagePoint(point);
  if (!imagePoint) {
    coords.textContent = 'x=—　y=—　RGB=—';
  } else {
    const rgb = imageViewerPixelAt(imagePoint.x, imagePoint.y);
    coords.textContent = rgb
      ? `x=${imagePoint.x}　y=${imagePoint.y}　RGB=(${rgb.r}, ${rgb.g}, ${rgb.b})`
      : `x=${imagePoint.x}　y=${imagePoint.y}　RGB=—`;
  }
  if (locked) {
    const lock = imageViewer.locked;
    if (!lock) {
      locked.textContent = '未锁定';
      locked.className = '';
      return;
    }
    const rgb = imageViewerPixelAt(lock.x, lock.y);
    locked.textContent = rgb
      ? `锁定 x=${lock.x}　y=${lock.y}　RGB=(${rgb.r}, ${rgb.g}, ${rgb.b})`
      : `锁定 x=${lock.x}　y=${lock.y}`;
    locked.className = 'image-zoom-locked';
  }
}

function imageViewerDraw() {
  if (!imageViewer.ctx || !imageViewer.stage) return;
  const cssWidth = imageViewer.stage.clientWidth;
  const cssHeight = imageViewer.stage.clientHeight;
  if (cssWidth < 2 || cssHeight < 2) return;
  imageViewer.ctx.clearRect(0, 0, cssWidth, cssHeight);
  if (!imageViewer.image) {
    imageViewerUpdateStatus();
    return;
  }

  const { ctx, image } = imageViewer;
  const dpr = window.devicePixelRatio || 1;
  const fitScale = imageViewer.fitScale || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = imageViewer.scale < 1
    || Math.abs(imageViewer.scale - fitScale) / fitScale < 0.01;
  ctx.drawImage(
    image,
    imageViewer.offsetX,
    imageViewer.offsetY,
    image.width * imageViewer.scale,
    image.height * imageViewer.scale,
  );

  if (imageViewer.showGrid && imageViewer.scale >= 8) {
    const left = Math.max(0, Math.floor((0 - imageViewer.offsetX) / imageViewer.scale));
    const right = Math.min(
      image.width - 1,
      Math.ceil((cssWidth - imageViewer.offsetX) / imageViewer.scale),
    );
    const top = Math.max(0, Math.floor((0 - imageViewer.offsetY) / imageViewer.scale));
    const bottom = Math.min(
      image.height - 1,
      Math.ceil((cssHeight - imageViewer.offsetY) / imageViewer.scale),
    );
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = left; x <= right; x += 1) {
      const screenX = imageViewer.offsetX + x * imageViewer.scale;
      ctx.moveTo(screenX, 0);
      ctx.lineTo(screenX, cssHeight);
    }
    for (let y = top; y <= bottom; y += 1) {
      const screenY = imageViewer.offsetY + y * imageViewer.scale;
      ctx.moveTo(0, screenY);
      ctx.lineTo(cssWidth, screenY);
    }
    ctx.stroke();
    ctx.restore();
  }

  const hover = imageViewer.mouseInside
    ? imageViewerImagePoint({ x: imageViewer.mouseX, y: imageViewer.mouseY })
    : null;
  if (hover) {
    const centerX = imageViewer.offsetX + (hover.x + 0.5) * imageViewer.scale;
    const centerY = imageViewer.offsetY + (hover.y + 0.5) * imageViewer.scale;
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.95)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, cssHeight);
    ctx.moveTo(0, centerY);
    ctx.lineTo(cssWidth, centerY);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.9)';
    ctx.strokeRect(
      imageViewer.offsetX + hover.x * imageViewer.scale,
      imageViewer.offsetY + hover.y * imageViewer.scale,
      Math.max(1, imageViewer.scale),
      Math.max(1, imageViewer.scale),
    );
    ctx.restore();
  }

  const lock = imageViewer.locked;
  if (
    lock
    && lock.x >= 0 && lock.y >= 0
    && lock.x < image.width && lock.y < image.height
  ) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 209, 102, 0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      imageViewer.offsetX + lock.x * imageViewer.scale,
      imageViewer.offsetY + lock.y * imageViewer.scale,
      Math.max(1, imageViewer.scale),
      Math.max(1, imageViewer.scale),
    );
    ctx.restore();
  }
}

function imageViewerZoomAt(point, factor) {
  if (!imageViewer.image || !point) return;
  const previous = imageViewer.scale;
  const next = Math.min(
    80,
    Math.max(imageViewer.fitScale * 0.2, previous * factor),
  );
  if (Math.abs(next - previous) < 1e-6) return;
  const imageX = (point.x - imageViewer.offsetX) / previous;
  const imageY = (point.y - imageViewer.offsetY) / previous;
  imageViewer.scale = next;
  imageViewer.offsetX = point.x - imageX * next;
  imageViewer.offsetY = point.y - imageY * next;
  imageViewerDraw();
  imageViewerUpdateStatus();
}

function imageViewerSetScale(nextScale, point) {
  if (!imageViewer.image || !point) return;
  const previous = imageViewer.scale;
  const next = Math.min(
    80,
    Math.max(imageViewer.fitScale * 0.2, nextScale),
  );
  if (Math.abs(next - previous) < 1e-6) return;
  const imageX = (point.x - imageViewer.offsetX) / previous;
  const imageY = (point.y - imageViewer.offsetY) / previous;
  imageViewer.scale = next;
  imageViewer.offsetX = point.x - imageX * next;
  imageViewer.offsetY = point.y - imageY * next;
  imageViewerDraw();
  imageViewerUpdateStatus();
}

function imageViewerLoadImage(source, preserveTransform = false) {
  if (!source) return;
  const token = ++imageViewer.loadToken;
  imageViewer.sourceUrl = source;
  const nextImage = new Image();
  nextImage.onload = () => {
    if (token !== imageViewer.loadToken) return;
    const previous = imageViewer.image;
    const sameSize = Boolean(
      previous
      && previous.width === nextImage.width
      && previous.height === nextImage.height,
    );
    imageViewer.image = nextImage;
    imageViewer.offscreen.width = nextImage.width;
    imageViewer.offscreen.height = nextImage.height;
    if (!imageViewer.offCtx) {
      imageViewer.offCtx = imageViewer.offscreen.getContext('2d', {
        willReadFrequently: true,
      });
    }
    imageViewer.offCtx.clearRect(0, 0, nextImage.width, nextImage.height);
    imageViewer.offCtx.drawImage(nextImage, 0, 0);
    imageViewer.fitScale = imageViewerFitScale();
    if (!preserveTransform || !sameSize) imageViewerRefit();
    imageViewerDraw();
    imageViewerUpdateStatus();
  };
  nextImage.onerror = () => {
    if (token !== imageViewer.loadToken) return;
    toast(
      '像素查看器加载失败',
      '该图片可能尚未生成，或浏览器无法解码当前 JPEG',
      'warning',
    );
  };
  nextImage.src = source;
}

function imageViewerStartWatching() {
  if (imageViewer.watchTimer) clearInterval(imageViewer.watchTimer);
  imageViewer.watchTimer = setInterval(() => {
    const element = imageViewer.sourceElement;
    const source = element?.currentSrc || element?.src;
    if (source && source !== imageViewer.sourceUrl) {
      imageViewerLoadImage(source, true);
    }
  }, 400);
}

function imageViewerClose() {
  if (imageViewer.watchTimer) {
    clearInterval(imageViewer.watchTimer);
    imageViewer.watchTimer = null;
  }
  imageViewer.loadToken += 1;
  imageViewer.sourceElement = null;
  imageViewer.sourceUrl = '';
  imageViewer.image = null;
  imageViewer.locked = null;
  imageViewer.mouseInside = false;
  imageViewer.dragging = false;
  imageViewer.moved = false;
  imageViewer.mouseX = null;
  imageViewer.mouseY = null;
  imageViewer.fitScale = 1;
  imageViewer.scale = 1;
  imageViewer.offsetX = 0;
  imageViewer.offsetY = 0;
  imageViewer.offscreen.width = 1;
  imageViewer.offscreen.height = 1;
  if (imageViewer.offCtx) {
    imageViewer.offCtx.clearRect(0, 0, 1, 1);
  }
  if (imageViewer.ctx && imageViewer.stage) {
    imageViewer.ctx.clearRect(
      0, 0,
      imageViewer.stage.clientWidth,
      imageViewer.stage.clientHeight,
    );
  }
  const coords = $('#image-zoom-coords');
  const locked = $('#image-zoom-locked');
  if (coords) coords.textContent = 'x=—　y=—　RGB=—';
  if (locked) {
    locked.textContent = '未锁定';
    locked.className = '';
  }
}

function openImageZoom(image, titleText) {
  const source = image?.currentSrc || image?.src;
  if (!source || !image?.hasAttribute('src')) {
    toast('调试图尚未生成', '可以稍后点击页面中的“刷新”重试', 'warning');
    return false;
  }
  if (!imageViewer.dialog) {
    toast('像素查看器未初始化', '请刷新页面后重试', 'error');
    return false;
  }
  imageViewer.sourceElement = image;
  imageViewer.locked = null;
  imageViewer.mouseInside = false;
  imageViewer.mouseX = null;
  imageViewer.mouseY = null;
  $('#image-zoom-title').textContent = titleText;
  if (!imageViewer.dialog.open) imageViewer.dialog.showModal();
  imageViewerLoadImage(source, false);
  imageViewerStartWatching();
  requestAnimationFrame(() => imageViewerResizeCanvas());
  return true;
}

function openTaskInteraction(key, kind, title, message) {
  const dialog = $('#task-interaction-dialog');
  if (app.interactionSuppressed.has(key)) return false;
  if (dialog.open && app.interactionDialogKey === key) return true;
  if ($('#image-zoom-dialog').open) $('#image-zoom-dialog').close();
  if (dialog.open) dialog.close();
  app.interactionDialogKey = key;
  app.interactionDialogKind = kind;
  app.interactionSubmitting = false;
  $('#task-interaction-title').textContent = title;
  $('#task-interaction-message').textContent = message;
  $('#task-interaction-actions').replaceChildren();
  $('#task-interaction-warning').hidden = true;
  $('#task-interaction-warning').textContent = '';
  $('#task-interaction-summary').replaceChildren();
  $('#task-interaction-summary').hidden = true;
  $('#task-interaction-countdown').hidden = true;
  $('#task-interaction-stop-motion').disabled = false;
  dialog.showModal();
  return true;
}

async function answerDynamicPrompt(choice, button) {
  if (app.interactionSubmitting) return;
  let confirmed = false;
  if (choice === 'continue_dynamic') {
    if (button.dataset.confirmed !== 'true') {
      const warning = $('#task-interaction-warning');
      warning.hidden = false;
      warning.textContent = '实际机械臂仍按当前配置速度运行，但盘面预测时间不再代表真实秒数。请再次点击按钮确认继续。';
      button.dataset.confirmed = 'true';
      button.textContent = '再次确认：忽略差异并继续';
      return;
    }
    confirmed = true;
  }
  app.interactionSubmitting = true;
  setInteractionButtonsDisabled(true);
  try {
    await api('/api/task/interaction/respond', {
      method: 'POST',
      json: {
        prompt_id: app.state?.interaction?.prompt_id,
        choice,
        confirm_speed_mismatch: confirmed,
      },
    });
    toast('选择已提交', choice === 'stop' ? '正在安全结束本轮' : '正在继续处理识别结果');
    closeTaskInteraction({ suppress: true });
  } catch (error) {
    app.interactionSubmitting = false;
    setInteractionButtonsDisabled(false);
    toast('选择未提交', formatError(error), 'error', 6500);
    refreshState();
  }
}

function renderDynamicPrompt(interaction) {
  const key = `prompt:${interaction.prompt_id}`;
  const alreadyOpen = $('#task-interaction-dialog').open
    && app.interactionDialogKey === key;
  if (!openTaskInteraction(
    key,
    'dynamic',
    interaction.title || 'V5 动态盘面需要选择',
    interaction.message || '动态盘面选择失败。',
  )) return;
  if (app.promptDeadlineId !== interaction.prompt_id) {
    app.promptDeadlineId = interaction.prompt_id;
    app.promptDeadline = Date.now() + Number(interaction.remaining_seconds || 0) * 1000;
  } else {
    app.promptDeadline = Math.min(
      app.promptDeadline,
      Date.now() + Number(interaction.remaining_seconds || 0) * 1000,
    );
  }
  const countdown = $('#task-interaction-countdown');
  countdown.hidden = false;
  if (alreadyOpen) {
    updateInteractionCountdown();
    return;
  }
  interaction.choices.forEach((choice) => {
    const styles = { danger: 'danger', secondary: 'secondary', warning: 'warning' };
    const button = appendInteractionButton(
      choice.label,
      styles[choice.tone] || 'ghost',
      () => answerDynamicPrompt(choice.id, button),
    );
  });
  appendInteractionButton(
    '查看识别调试图',
    'ghost',
    () => openImageZoom($('#debug-image'), '识别调试图'),
  );
  updateInteractionCountdown();
}

function addRecognitionSummary(task) {
  const summary = $('#task-interaction-summary');
  const rows = [
    ['任务总数', task.task_count || 0],
    ['方块', task.block_count || 0],
    ['托盘点', task.tray_count || 0],
  ];
  rows.forEach(([label, value]) => {
    const item = document.createElement('div');
    const small = document.createElement('small'); small.textContent = label;
    const strong = document.createElement('strong'); strong.textContent = value;
    item.append(small, strong); summary.append(item);
  });
  summary.hidden = false;
}

async function handleRecognitionChoice(choice) {
  if (app.interactionSubmitting) return;
  app.interactionSubmitting = true;
  setInteractionButtonsDisabled(true);
  try {
    if (choice === 'retry') {
      await submitPrepare();
    } else if (choice === 'discard') {
      await api('/api/task/discard', { method: 'POST', json: {} });
      toast('本轮已结束', '硬件和感知保持运行，可以再次识别');
    } else if (choice === 'confirm') {
      await api('/api/task/confirm', { method: 'POST', json: {} });
      toast('识别结果已确认', '请检查现场后点击“开始执行”');
    }
    closeTaskInteraction({ suppress: true });
    refreshState();
  } catch (error) {
    app.interactionSubmitting = false;
    setInteractionButtonsDisabled(false);
    toast('操作未执行', formatError(error), 'error', 6500);
    refreshState();
  }
}

function renderRecognitionChoice(state) {
  const task = state.task || {};
  const latest = state.operation?.latest || {};
  if (state.operation?.active) return false;
  const success = task.state === '等待确认' && task.recognition_valid;
  const failed = task.state === '失败' && latest.kind === '识别与规划';
  if (!success && !failed) return false;
  const key = `recognition:${latest.operation_id || task.message}:${task.state}`;
  const modeLabel = task.mode === 'calibration' ? '标定采集' : '抓放任务';
  const title = success ? `${modeLabel}识别完成` : `${modeLabel}识别未完成`;
  const message = task.error || task.message || (success ? '请确认识别结果。' : '请调整现场后重试。');
  if ($('#task-interaction-dialog').open && app.interactionDialogKey === key) return true;
  if (!openTaskInteraction(key, 'recognition', title, message)) return true;
  addRecognitionSummary(task);
  appendInteractionButton(
    '查看识别调试图',
    'ghost',
    () => openImageZoom($('#debug-image'), '识别调试图'),
  );
  appendInteractionButton(
    success ? '重新识别' : '调整后重新识别',
    'secondary',
    () => handleRecognitionChoice('retry'),
  );
  appendInteractionButton('结束本轮', 'ghost', () => handleRecognitionChoice('discard'));
  if (success) {
    appendInteractionButton('确认结果', 'primary', () => handleRecognitionChoice('confirm'));
  }
  return true;
}

function renderTaskInteraction(state) {
  const interaction = state.interaction || {};
  if (interaction.pending) {
    renderDynamicPrompt(interaction);
    return;
  }
  if (renderRecognitionChoice(state)) return;
  if ($('#task-interaction-dialog').open && !app.interactionSubmitting) {
    closeTaskInteraction();
  }
}

function updateInteractionCountdown() {
  if (app.interactionDialogKind !== 'dynamic' || !$('#task-interaction-dialog').open) return;
  const remaining = Math.max(0, Math.ceil((app.promptDeadline - Date.now()) / 1000));
  const countdown = $('#task-interaction-countdown');
  countdown.textContent = remaining > 0
    ? `${remaining} 秒后自动停止本轮`
    : '正在自动停止本轮…';
  if (remaining <= 0) setInteractionButtonsDisabled(true);
}

function formatUsbOccupant(occupant) {
  const pid = occupant?.pid ?? '?';
  const command = occupant?.cmdline || `PID ${pid}`;
  const shortCommand = command.length > 110 ? `${command.slice(0, 109)}…` : command;
  return `PID ${pid}（${shortCommand}）`;
}

function usbBlockedMessage(data) {
  if (!data) return '';
  const labels = { camera: '相机', servo: '舵机' };
  const parts = [];
  for (const key of ['camera', 'servo']) {
    const device = data[key];
    if (!device) continue;
    const status = device.status;
    if (status === 'missing' || status === 'unknown') {
      parts.push(`${labels[key]}：${device.message || '状态未知'}`);
      continue;
    }
    if (status === 'occupied') {
      const external = (device.occupants || [])
        .filter((occupant) => !['self', 'panel'].includes(occupant.owner));
      if (external.length) {
        parts.push(`${labels[key]}被 ${external.map(formatUsbOccupant).join('；')} 占用`);
      }
    }
  }
  return parts.join('；');
}

function renderUsbOccupancy(data, errorMessage = '') {
  const labels = { camera: '相机', servo: '舵机' };
  for (const [key, label] of Object.entries(labels)) {
    const element = $(`#usb-${key}-status`);
    if (!element) continue;
    if (!data) {
      element.textContent = `${label}：${errorMessage || '读取失败'}`;
      element.className = 'usb-status warn';
      continue;
    }
    const device = data[key] || {};
    const occupants = device.occupants || [];
    const external = occupants.some(
      (occupant) => !['self', 'panel'].includes(occupant.owner),
    );
    let className = '';
    if (device.status === 'free') {
      className = device.inspection_limited ? 'warn' : 'free';
    } else if (device.status === 'occupied') {
      className = external ? 'blocked' : 'free';
    } else {
      className = 'warn';
    }

    let text;
    if (device.status === 'occupied' && occupants.length && !external) {
      text = `${label}：控制台内部占用（正常）`;
    } else {
      text = `${label}：${device.message || '状态未知'}`;
    }
    element.textContent = text;
    element.className = `usb-status ${className}`;
    element.title = occupants.length
      ? occupants.map(formatUsbOccupant).join('\n')
      : (device.message || '');
  }
  const checkedAt = $('#usb-checked-at');
  if (checkedAt) {
    checkedAt.textContent = data?.checked_at
      ? `检查时间：${data.checked_at.replace('T', ' ').slice(0, 19)}`
      : '';
  }
}

async function loadUsbOccupancy(force = false) {
  if (app.usbOccupancyLoading) return app.usbOccupancyPromise;
  if (!force && Date.now() - app.usbOccupancyLastLoaded < 3000) {
    return app.usbOccupancy;
  }
  app.usbOccupancyLoading = true;
  app.usbOccupancyPromise = (async () => {
    try {
      app.usbOccupancy = await api('/api/hardware/usb-occupancy');
      app.usbOccupancyLastLoaded = Date.now();
      renderUsbOccupancy(app.usbOccupancy);
      return app.usbOccupancy;
    } catch (error) {
      renderUsbOccupancy(null, formatError(error));
      return null;
    } finally {
      app.usbOccupancyLoading = false;
      app.usbOccupancyPromise = null;
    }
  })();
  return app.usbOccupancyPromise;
}

function renderState(state) {
  app.state = state;
  const health = state.health || {};
  const hardware = state.hardware || {};
  const process = state.process || {};
  const task = state.task || {};
  const stopOperation = state.operation?.stop;
  const active = state.operation?.active || (stopOperation?.status === 'running' ? stopOperation : null);

  setStatus('ros', health.ros_master, health.ros_master ? '在线' : '离线');
  setStatus('camera', health.camera_frame_fresh, health.camera_frame_fresh ? '画面正常' : (health.camera_node ? '等待画面' : '未连接'), health.camera_node && !health.camera_frame_fresh);
  setStatus('arm', health.control_services_ready, health.control_services_ready ? '服务就绪' : (health.control_node ? '启动中' : '未连接'), health.control_node && !health.control_services_ready);
  const runtimeMode = process.runtime?.mode;
  const runtimeText = process.runtime?.running ? (runtimeMode === 'calibration' ? '标定模式' : (runtimeMode === 'formal' ? '正式模式' : '外部节点')) : '未启动';
  setStatus('runtime', health.perception_services_ready, runtimeText, process.runtime?.running && !health.perception_services_ready);
  setStatus('holding', hardware.holding_block, hardware.holding_block ? '是' : '否', hardware.holding_block);
  $('#camera-topic-hz').textContent = `${Number(health.camera_topic_hz || 0).toFixed(1)} Hz`;

  const stopChip = $('[data-status="stop"]');
  stopChip.classList.toggle('locked', Boolean(hardware.stop_latched));
  $('b', stopChip).textContent = hardware.stop_latched ? '已锁定' : '未锁定';

  const indicator = $('#operation-indicator');
  indicator.className = `operation-indicator ${active ? 'running' : 'idle'}`;
  $('span', indicator).textContent = active ? `正在${active.kind}` : '当前无操作';

  const pendingScopes = (state.config?.pending_restart || []).filter((scope) => scopeRunning(scope));
  $('#pending-restart').textContent = pendingScopes.length
    ? pendingScopes.map((scope) => scope === 'hardware' ? '硬件' : '感知').join('、')
    : '无';

  const taskBadge = $('#task-state-badge');
  taskBadge.textContent = task.state || '空闲';
  taskBadge.className = `state-badge ${stateClass(task.state)}`;
  const message = $('#task-message');
  message.textContent = task.error || task.message || '请先启动硬件和感知模式。';
  message.classList.toggle('error', Boolean(task.error));
  const current = Number(task.current || 0);
  const total = Number(task.total || task.task_count || 0);
  $('#progress-count').textContent = `${current} / ${total}`;
  $('#progress-bar').style.width = `${total > 0 ? Math.min(100, current / total * 100) : 0}%`;
  $('#progress-phase').textContent = task.phase || '未开始';
  $('#progress-category').textContent = app.categoryLabels[task.category] || task.category || '—';
  $('#progress-target').textContent = ({ pick_place: '抓放', block: '方块标定', tray: '托盘标定' })[task.target_type] || task.target_type || '—';

  $$('[data-ready]').forEach((element) => {
    const ready = element.dataset.ready === 'control'
      ? health.control_services_ready
      : element.dataset.ready === 'frame'
        ? health.camera_frame_fresh
        : health.perception_services_ready;
    element.classList.toggle('ready', Boolean(ready));
  });

  $$('#runtime-mode button').forEach((button) => button.classList.toggle('active', process.runtime?.running && button.dataset.mode === runtimeMode));
  const busy = Boolean(active);
  const stopped = Boolean(hardware.stop_latched);
  if (!process.hardware?.running && Date.now() - app.usbOccupancyLastLoaded > 5000) {
    loadUsbOccupancy();
  }
  // 停止锁期间仍允许只重建硬件服务，以便同步 StopMotion 并解除锁；感知和动作继续禁用。
  $('#hardware-start').disabled = busy || Boolean(process.hardware?.running);
  $('#hardware-stop').disabled = busy || !process.hardware?.owned || ['准备识别', '执行中'].includes(task.state);
  $$('#runtime-mode button').forEach((button) => { button.disabled = busy || stopped || !health.control_services_ready || (process.runtime?.running && button.dataset.mode === runtimeMode); });
  $('#runtime-stop').disabled = busy || !process.runtime?.owned || ['准备识别', '执行中'].includes(task.state);
  const running = task.state === '执行中' || task.state === '已暂停';
  $('#task-prepare').disabled = busy || stopped || !health.control_services_ready || !health.perception_services_ready || running;
  // 停止始终按任务状态开关；暂停要绕过 busy（执行任务占着队列）；继续要等手动操作结束（busy）。
  $('#task-stop').disabled = !running;
  if (task.state === '执行中') {
    $('#task-start').disabled = false;
    $('#task-start').textContent = '暂停';
  } else if (task.state === '已暂停') {
    $('#task-start').disabled = busy;
    $('#task-start').textContent = '继续';
  } else {
    $('#task-start').disabled = busy || stopped || task.state !== '可以执行' || !task.confirmed;
    $('#task-start').textContent = '开始执行';
  }
  $('#task-prepare').textContent = ['等待确认', '可以执行', '失败'].includes(task.state) ? '重新识别' : '开始识别';
  $('#task-start').classList.toggle('attention', !$('#task-start').disabled && task.state === '可以执行');
  $('#task-start').classList.toggle('warning', task.state === '执行中');
  $('#clear-stop').disabled = busy || !stopped;

  const sweeping = Boolean(hardware.servo_sweep_running);
  const manualEnabled = !busy && !stopped && !sweeping && !['准备识别', '执行中'].includes(task.state);
  $$('[data-suction], #servo-send, #servo-quick button, #arm-reset, #relative-move').forEach((button) => { button.disabled = !manualEnabled; });
  $('#pose-refresh').disabled = !manualEnabled;
  $('#servo-sweep-start').disabled = !manualEnabled;
  $('#servo-sweep-stop').disabled = !sweeping;

  const arucoAlignRunning = active?.kind === 'ArUco 单次对准';
  const arucoAlignReady = manualEnabled
    && health.control_services_ready
    && health.camera_frame_fresh;
  const arucoStatus = $('#aruco-align-status');
  arucoStatus.textContent = arucoAlignRunning ? '运行中' : '未运行';
  arucoStatus.className = `state-badge ${arucoAlignRunning ? 'active' : ''}`;
  $('#aruco-align-start').disabled = !arucoAlignReady;
  $('#aruco-low-z').disabled = arucoAlignRunning;
  const manualRoute = $('#manual-route');
  if (manualRoute) manualRoute.textContent = health.control_node ? '控制链路：控制节点（ROS）' : '控制链路：直连硬件';

  const suctionNames = { '-1': '状态未知', 0: '吸气', 1: '喷气', 2: '关闭' };
  $('#suction-status').textContent = suctionNames[hardware.suction_state] ?? '状态未知';
  $('#suction-status').className = `state-badge ${hardware.suction_state === 0 ? 'active' : ''}`;
  $('#servo-status').textContent = hardware.servo_target_known ? `目标 ${Number(hardware.servo_target_angle_deg).toFixed(1)}°` : '角度未知';
  const sweepStatus = $('#servo-sweep-status');
  if (sweeping) {
    sweepStatus.textContent = `往复中 · 第 ${Number(hardware.servo_sweep_cycle || 0)} 次 · ${Number(hardware.servo_sweep_target_angle_deg || 0).toFixed(1)}°`;
    sweepStatus.className = 'state-badge active';
  } else {
    sweepStatus.textContent = '未运行';
    sweepStatus.className = 'state-badge';
  }

  const warning = hardware.emergency_warning || '';
  const overlay = $('#emergency-overlay');
  const stopFailed = warning.includes('无法确认');
  overlay.hidden = !stopFailed;
  document.body.classList.toggle('has-emergency', stopFailed);
  if (stopFailed) $('#emergency-message').textContent = warning;

  if (health.last_frame_at) $('#camera-time').textContent = health.last_frame_at.replace('T', ' ').slice(0, 23);
  renderConfigFileStates();
  renderConfigActions();
  renderTaskInteraction(state);
}

function renderOperation(operation) {
  if (!operation?.operation_id) return;
  const indicator = $('#operation-indicator');
  if (operation.status === 'running') {
    indicator.className = 'operation-indicator running';
    $('span', indicator).textContent = `正在${operation.kind}`;
    return;
  }
  const key = `${operation.operation_id}:${operation.status}`;
  if (app.seenOperations.has(key)) return;
  app.seenOperations.add(key);
  if (operation.kind === '识别与规划') refreshDebugImages();
  indicator.className = `operation-indicator ${operation.status === 'success' ? 'idle' : 'error'}`;
  $('span', indicator).textContent = operation.status === 'success' ? `${operation.kind}完成` : `${operation.kind}未完成`;
  if (operation.status === 'success') {
    if (operation.result?.apply_error) {
      toast(`${operation.kind}已保存，但未能自动重启`, operation.result.apply_error, 'warning', 7500);
    } else {
      toast(`${operation.kind}完成`, operation.result?.message || '状态已更新');
    }
    if (['保存配置', '恢复配置历史', '恢复配置预设'].includes(operation.kind)) {
      refreshConfigAfterWrite();
    }
    if (['保存配置预设', '删除配置预设'].includes(operation.kind)) loadPresets();
  } else {
    toast(`${operation.kind}失败`, operation.error || '请查看运行日志', operation.status === 'aborted' ? 'warning' : 'error', 7000);
  }
  refreshState();
}

async function refreshState() {
  try { renderState(await api('/api/state')); }
  catch (error) { toast('状态读取失败', formatError(error), 'error'); }
}

function startEvents() {
  const source = eventStream();
  source.addEventListener('state', (event) => renderState(JSON.parse(event.data)));
  source.addEventListener('health', () => {
    refreshState();
    if (app.selectedView === 'ros') loadRosSystem();
  });
  source.addEventListener('process', () => refreshState());
  source.addEventListener('progress', (event) => {
    if (!app.state) return;
    app.state.task = { ...app.state.task, ...JSON.parse(event.data) };
    renderState(app.state);
  });
  source.addEventListener('operation', (event) => renderOperation(JSON.parse(event.data)));
  source.addEventListener('interaction', (event) => {
    if (!app.state) return;
    app.state.interaction = JSON.parse(event.data);
    renderTaskInteraction(app.state);
  });
  source.addEventListener('log', (event) => appendLog(JSON.parse(event.data)));
  source.addEventListener('image', (event) => {
    const data = JSON.parse(event.data);
    if (data.image_id === 'camera') updateImage($('#camera-image'), 'camera', data.updated_at);
  });
  source.onerror = () => {
    const chip = $('[data-status="ros"]');
    chip?.classList.add('warning');
  };
}

function updateImage(element, imageId, marker = Date.now()) {
  const next = cacheBustedImage(imageId, marker);
  element.onerror = () => element.removeAttribute('src');
  element.src = next;
}

function refreshDebugImages() {
  updateImage($('#debug-image'), app.selectedDebugImage);
}

function bindImageZoom() {
  const dialog = $('#image-zoom-dialog');
  const stage = $('#image-zoom-stage');
  const canvas = $('#image-zoom-canvas');
  const gridButton = $('#image-zoom-grid-toggle');

  imageViewer.dialog = dialog;
  imageViewer.stage = stage;
  imageViewer.canvas = canvas;
  imageViewer.ctx = canvas.getContext('2d');
  imageViewer.offscreen.width = 1;
  imageViewer.offscreen.height = 1;
  imageViewer.offCtx = imageViewer.offscreen.getContext('2d', {
    willReadFrequently: true,
  });

  $$('#camera-image, #debug-image').forEach((image) => {
    image.title = '双击放大';
    image.tabIndex = 0;
    image.addEventListener('dblclick', () => openImageZoom(
      image,
      image.id === 'debug-image'
        ? `识别调试图 · ${$('.debug-tabs button.active')?.textContent || ''}`
        : '相机预览',
    ));
    image.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') openImageZoom(
        image,
        image.id === 'debug-image' ? '识别调试图' : '相机预览',
      );
    });
  });

  $('#image-zoom-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener('close', () => imageViewerClose());

  gridButton.addEventListener('click', () => {
    imageViewer.showGrid = !imageViewer.showGrid;
    gridButton.textContent = imageViewer.showGrid ? '像素网格：开' : '像素网格：关';
    gridButton.classList.toggle('active', imageViewer.showGrid);
    imageViewerDraw();
  });

  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !imageViewer.image) return;
    const point = imageViewerPointFromEvent(event);
    if (!point) return;
    imageViewer.dragging = true;
    imageViewer.moved = false;
    imageViewer.dragStartX = point.x;
    imageViewer.dragStartY = point.y;
    imageViewer.dragOffsetX = imageViewer.offsetX;
    imageViewer.dragOffsetY = imageViewer.offsetY;
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (_) {}
  });

  canvas.addEventListener('pointermove', (event) => {
    const point = imageViewerPointFromEvent(event);
    if (point) {
      imageViewer.mouseX = point.x;
      imageViewer.mouseY = point.y;
      imageViewer.mouseInside = point.x >= 0 && point.y >= 0
        && point.x <= imageViewer.stage.clientWidth
        && point.y <= imageViewer.stage.clientHeight;
      if (imageViewer.dragging) {
        const dx = point.x - imageViewer.dragStartX;
        const dy = point.y - imageViewer.dragStartY;
        if (Math.hypot(dx, dy) > 3) imageViewer.moved = true;
        if (imageViewer.moved) {
          imageViewer.offsetX = imageViewer.dragOffsetX + dx;
          imageViewer.offsetY = imageViewer.dragOffsetY + dy;
        }
      }
    }
    imageViewerDraw();
    imageViewerUpdateStatus();
  });

  canvas.addEventListener('pointerup', (event) => {
    if (event.button !== 0) return;
    const point = imageViewerPointFromEvent(event);
    if (point) {
      imageViewer.mouseX = point.x;
      imageViewer.mouseY = point.y;
      imageViewer.mouseInside = point.x >= 0 && point.y >= 0
        && point.x <= imageViewer.stage.clientWidth
        && point.y <= imageViewer.stage.clientHeight;
      if (imageViewer.dragging && !imageViewer.moved) {
        const imagePoint = imageViewerImagePoint(point);
        if (imagePoint) {
          imageViewer.locked = { x: imagePoint.x, y: imagePoint.y };
        }
      }
    }
    imageViewer.dragging = false;
    imageViewer.moved = false;
    imageViewerDraw();
    imageViewerUpdateStatus();
  });

  canvas.addEventListener('pointercancel', () => {
    imageViewer.dragging = false;
    imageViewer.moved = false;
  });
  canvas.addEventListener('pointerleave', () => {
    imageViewer.mouseInside = false;
    imageViewer.mouseX = null;
    imageViewer.mouseY = null;
    imageViewerDraw();
    imageViewerUpdateStatus();
  });

  canvas.addEventListener('wheel', (event) => {
    if (!event.ctrlKey || !imageViewer.image) return;
    event.preventDefault();
    const point = imageViewerPointFromEvent(event);
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    imageViewerZoomAt(point, factor);
  }, { passive: false });

  canvas.addEventListener('dblclick', (event) => {
    if (!imageViewer.image) return;
    event.preventDefault();
    const point = imageViewerPointFromEvent(event);
    const fitScale = imageViewer.fitScale || 1;
    const isFit = Math.abs(imageViewer.scale - fitScale)
      <= Math.max(0.001, fitScale * 0.01);
    if (isFit) imageViewerSetScale(1, point);
    else imageViewerRefit();
  });

  dialog.addEventListener('keydown', (event) => {
    if (!imageViewer.image) return;
    if (event.key === '0') {
      event.preventDefault();
      imageViewerRefit();
      imageViewerDraw();
      imageViewerUpdateStatus();
    } else if (event.key === '1') {
      event.preventDefault();
      const point = imageViewer.mouseInside
        ? { x: imageViewer.mouseX, y: imageViewer.mouseY }
        : {
          x: imageViewer.stage.clientWidth / 2,
          y: imageViewer.stage.clientHeight / 2,
        };
      imageViewerSetScale(1, point);
    } else if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      const point = {
        x: imageViewer.stage.clientWidth / 2,
        y: imageViewer.stage.clientHeight / 2,
      };
      imageViewerZoomAt(point, 1.18);
    } else if (event.key === '-') {
      event.preventDefault();
      const point = {
        x: imageViewer.stage.clientWidth / 2,
        y: imageViewer.stage.clientHeight / 2,
      };
      imageViewerZoomAt(point, 1 / 1.18);
    }
  });

  const resizeObserver = new ResizeObserver(() => imageViewerResizeCanvas());
  resizeObserver.observe(stage);
  window.addEventListener('resize', () => {
    if (dialog.open) imageViewerResizeCanvas();
  });
}

function renderOrder() {
  const list = $('#block-order');
  list.replaceChildren();
  app.order.forEach((index) => {
    const item = document.createElement('li');
    item.draggable = true;
    item.dataset.index = index;
    const label = document.createElement('b');
    label.textContent = app.categoryLabels[app.categories[index]];
    item.append(label);
    item.addEventListener('dragstart', () => item.classList.add('dragging'));
    item.addEventListener('dragend', () => { item.classList.remove('dragging'); $$('.drag-over', list).forEach((entry) => entry.classList.remove('drag-over')); });
    item.addEventListener('dragover', (event) => { event.preventDefault(); item.classList.add('drag-over'); });
    item.addEventListener('dragleave', () => item.classList.remove('drag-over'));
    item.addEventListener('drop', (event) => {
      event.preventDefault();
      const dragged = $('.dragging', list);
      if (!dragged || dragged === item) return;
      const from = app.order.indexOf(Number(dragged.dataset.index));
      const to = app.order.indexOf(Number(item.dataset.index));
      const [moved] = app.order.splice(from, 1);
      app.order.splice(to, 0, moved);
      renderOrder();
    });
    list.append(item);
  });
}

function rosEmpty(container, message) {
  const paragraph = document.createElement('p');
  paragraph.className = 'ros-empty';
  paragraph.textContent = message;
  container.replaceChildren(paragraph);
}

function renderRosTopology(data) {
  const nodes = new Set(data.nodes || []);
  const topics = new Set((data.topics || []).map((topic) => topic.name));
  const services = (data.services || []).map((service) => service.name);
  $$('[data-ros-node]').forEach((element) => {
    element.classList.toggle('online', nodes.has(element.dataset.rosNode));
  });
  $$('[data-ros-topic]').forEach((element) => {
    element.classList.toggle('online', topics.has(element.dataset.rosTopic));
  });
  $$('[data-ros-service-prefix]').forEach((element) => {
    element.classList.toggle(
      'online',
      services.some((name) => name.startsWith(element.dataset.rosServicePrefix)),
    );
  });
  $('.ros-stage.browser')?.classList.toggle('online', true);
  const cameraTopic = (data.topics || []).find((topic) => topic.name === '/camera/image_rect');
  $('#ros-flow-camera-hz').textContent = `${Number(cameraTopic?.hz || 0).toFixed(1)} Hz`;
}

function renderRosNodes(data) {
  const container = $('#ros-node-list');
  const nodes = data.nodes || [];
  if (!nodes.length) return rosEmpty(container, 'ROS Master 当前没有节点');
  const keyNodes = new Set(['/camera_node', '/control_node', '/image_process_node', '/operator_panel']);
  const labels = {
    '/camera_node': '相机图像发布',
    '/control_node': '机械臂与工具控制',
    '/image_process_node': '识别与任务规划',
    '/operator_panel': '网页 ROS 网关',
  };
  const ordered = [...nodes].sort((a, b) => Number(keyNodes.has(b)) - Number(keyNodes.has(a)) || a.localeCompare(b));
  const fragment = document.createDocumentFragment();
  ordered.forEach((name) => {
    const item = document.createElement('div');
    item.className = `ros-item ${keyNodes.has(name) ? 'key' : ''}`;
    const code = document.createElement('code'); code.textContent = name;
    const detail = document.createElement('small'); detail.textContent = labels[name] || 'ROS 节点';
    item.append(code, detail); fragment.append(item);
  });
  container.replaceChildren(fragment);
}

function renderRosTopics(data) {
  const container = $('#ros-topic-list');
  const topics = data.topics || [];
  if (!topics.length) return rosEmpty(container, 'ROS Master 当前没有话题');
  const keyTopics = new Set(['/camera/image_rect', '/rosout', '/rosout_agg']);
  const ordered = [...topics].sort((a, b) => Number(keyTopics.has(b.name)) - Number(keyTopics.has(a.name)) || a.name.localeCompare(b.name));
  const fragment = document.createDocumentFragment();
  ordered.forEach((topic) => {
    const row = document.createElement('div');
    row.className = `ros-topic-row ${keyTopics.has(topic.name) ? 'key' : ''}`;
    const identity = document.createElement('div');
    const name = document.createElement('strong'); name.textContent = topic.name;
    const hz = document.createElement('small');
    hz.textContent = topic.hz === null || topic.hz === undefined ? '未测频率' : `实测 ${Number(topic.hz).toFixed(1)} Hz`;
    identity.append(name, hz);
    const type = document.createElement('div');
    const typeCode = document.createElement('code'); typeCode.textContent = topic.type || '类型未知';
    const typeLabel = document.createElement('small'); typeLabel.textContent = '消息类型';
    type.append(typeCode, typeLabel);
    const endpoints = document.createElement('div');
    const publishers = document.createElement('small');
    publishers.textContent = `发布：${(topic.publishers || []).join('、') || '无'}`;
    const subscribers = document.createElement('small');
    subscribers.textContent = `订阅：${(topic.subscribers || []).join('、') || '无'}`;
    endpoints.append(publishers, subscribers);
    row.append(identity, type, endpoints); fragment.append(row);
  });
  container.replaceChildren(fragment);
}

function renderRosServices(data) {
  const container = $('#ros-service-list');
  const services = data.services || [];
  if (!services.length) return rosEmpty(container, 'ROS Master 当前没有服务');
  const isKey = (name) => ['/camera/', '/perception/', '/control/'].some((prefix) => name.startsWith(prefix));
  const ordered = [...services].sort((a, b) => Number(isKey(b.name)) - Number(isKey(a.name)) || a.name.localeCompare(b.name));
  const fragment = document.createDocumentFragment();
  ordered.forEach((service) => {
    const item = document.createElement('div');
    item.className = `ros-item ${isKey(service.name) ? 'key' : ''}`;
    const code = document.createElement('code'); code.textContent = service.name;
    const detail = document.createElement('small');
    detail.textContent = `提供：${(service.providers || []).join('、') || '未知节点'}`;
    item.append(code, detail); fragment.append(item);
  });
  container.replaceChildren(fragment);
}

function renderRosSystem(data) {
  const online = Boolean(data.master_online);
  $('#ros-proof-master').textContent = online ? '在线' : '离线';
  $('#ros-proof-master-uri').textContent = data.master_uri || '未设置';
  $('#ros-proof-distro').textContent = data.distro || '未知';
  $('#ros-proof-node-count').textContent = Number(data.node_count || 0);
  $('#ros-proof-topic-count').textContent = Number(data.topic_count || 0);
  $('#ros-proof-service-count').textContent = Number(data.service_count || 0);
  const updated = data.updated_at ? data.updated_at.replace('T', ' ').slice(0, 23) : '';
  $('#ros-proof-updated').textContent = updated ? `ROS 图更新时间 ${updated}` : '等待 ROS Master';
  const badge = $('#ros-proof-state');
  badge.textContent = online ? '实时连接' : 'ROS 离线';
  badge.className = `state-badge ${online ? 'active' : 'error'}`;
  renderRosTopology(data);
  renderRosNodes(data);
  renderRosTopics(data);
  renderRosServices(data);
}

async function loadRosSystem(force = false) {
  const now = Date.now();
  if (app.rosSystemLoading || (!force && now - app.rosSystemLastLoaded < 900)) return;
  app.rosSystemLoading = true;
  try {
    const data = await api('/api/ros-system');
    app.rosSystemLastLoaded = Date.now();
    renderRosSystem(data);
  } catch (error) {
    toast('ROS 系统状态读取失败', formatError(error), 'error');
  } finally {
    app.rosSystemLoading = false;
  }
}

function bindNavigation() {
  $$('.nav-item').forEach((button) => button.addEventListener('click', () => {
    app.selectedView = button.dataset.view;
    $$('.nav-item').forEach((item) => item.classList.toggle('active', item === button));
    $$('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${app.selectedView}`));
    if (app.selectedView === 'config') initializeConfigCenter();
    if (app.selectedView === 'readonly') loadReadOnly();
    if (app.selectedView === 'manual') loadExecutionConfig();
    if (app.selectedView === 'ros') loadRosSystem(true);
  }));
}

function bindRosSystem() {
  $('#ros-proof-refresh').addEventListener('click', () => loadRosSystem(true));
}

function bindConsole() {
  $('#hardware-start').addEventListener('click', async () => {
    const occupancy = await loadUsbOccupancy(true);
    if (occupancy) {
      const blocked = usbBlockedMessage(occupancy);
      if (blocked) {
        toast('硬件启动已取消', blocked, 'error', 9000);
        return;
      }
    }
    command('/api/process/hardware/start', {}, '正在启动硬件');
  });
  $('#usb-refresh').addEventListener('click', () => loadUsbOccupancy(true));
  $('#hardware-stop').addEventListener('click', async () => {
    if (await confirmAction('停止硬件？', '将先停止感知，再结束控制台自己启动的相机与控制节点。外部节点不会被结束。')) {
      command('/api/process/hardware/stop', {}, '正在停止硬件');
    }
  });
  $$('#runtime-mode button').forEach((button) => button.addEventListener('click', async () => {
    const mode = button.dataset.mode;
    const label = mode === 'calibration' ? '标定' : '正式';
    if (app.state?.process?.runtime?.running && !await confirmAction(`切换到${label}模式？`, '模式切换会停止当前感知节点，已有识别结果立即失效。')) return;
    command('/api/process/runtime/start', { mode }, `正在启动${label}感知`);
  }));
  $('#runtime-stop').addEventListener('click', () => command('/api/process/runtime/stop', {}, '正在停止感知'));

  $$('input[name="task-level"]').forEach((input) => input.addEventListener('change', () => {
    $('#advanced-order-panel').hidden = input.value !== 'advanced' || !input.checked;
  }));
  $('#reset-order').addEventListener('click', () => { app.order = [0, 1, 2, 3, 4, 5, 6]; renderOrder(); });
  $('#task-prepare').addEventListener('click', () => submitPrepare({ confirmRetry: true }));
  let taskStopArmTimer = null;
  $('#task-stop').addEventListener('click', async () => {
    const state = app.state?.task?.state;
    if (!['执行中', '已暂停'].includes(state)) return;
    const button = $('#task-stop');
    if (button.dataset.armed === '1') {
      clearTimeout(taskStopArmTimer);
      button.dataset.armed = '0';
      button.textContent = '停止';
      try {
        await command('/api/task/stop', {});
        toast('停止执行已提交', '任务将中止，感知将停止');
      } catch (_) {}
    } else {
      button.dataset.armed = '1';
      button.textContent = '再点一次确认停止';
      taskStopArmTimer = setTimeout(() => {
        button.dataset.armed = '0';
        button.textContent = '停止';
      }, 1500);
    }
  });
  $('#task-start').addEventListener('click', async () => {
    const state = app.state?.task?.state;
    if (state === '可以执行') {
      const count = app.state?.task?.task_count || 0;
      if (await confirmAction('开始执行全部任务？', `即将执行 ${count} 个目标。请确认工作区无人员、障碍物和松动物品。`)) {
        command('/api/task/start', {}, '执行任务已提交');
      }
    } else if (state === '执行中') {
      try {
        await command('/api/task/pause', {});
        toast('已暂停', '当前运动完成后停止，点击“继续”恢复');
      } catch (_) {}
    } else if (state === '已暂停') {
      try {
        await command('/api/task/resume', {});
        toast('已继续', '任务恢复执行');
      } catch (_) {}
    }
  });
  $('#refresh-images').addEventListener('click', refreshDebugImages);
  $$('.debug-tabs button').forEach((button) => button.addEventListener('click', () => {
    app.selectedDebugImage = button.dataset.image;
    $$('.debug-tabs button').forEach((item) => item.classList.toggle('active', item === button));
    refreshDebugImages();
  }));
}

async function triggerStop() {
  try {
    await command('/api/control/stop', {}, '停止请求已通过高优先级通道提交');
    return true;
  } catch (_) {
    return false; // 错误已经显示。
  }
}

function bindManual() {
  $('#global-stop').addEventListener('click', triggerStop);
  $$('[data-suction]').forEach((button) => button.addEventListener('click', async () => {
    const action = button.dataset.suction;
    if (action === 'continuous') {
      if (!await confirmAction('持续喷气？', '持续喷气不会自动关闭。你必须稍后手动点击“关闭”，请确认这是现场需要的动作。')) return;
      command('/api/control/suction', { action: 'blow', continuous: true, confirm_continuous: true }, '持续喷气命令已提交');
    } else {
      command('/api/control/suction', { action }, '吸盘命令已提交');
    }
  }));

  async function sendServo(angle) {
    const value = Number(angle);
    const motor = app.executionConfig?.data?.tool_motor;
    const outside = motor && (value < motor.lower_margin_deg || value > motor.upper_margin_deg);
    let confirmed = false;
    if (outside) {
      confirmed = await confirmAction('角度超出正式安全边界', `目标 ${value}° 超出 ${motor.lower_margin_deg}°～${motor.upper_margin_deg}°。只在确认机构不会顶到限位时继续。`);
      if (!confirmed) return;
    }
    command('/api/control/servo', { angle_deg: value, confirm_outside_safe: confirmed }, '舵机命令已提交');
  }
  $('#servo-send').addEventListener('click', () => sendServo($('#servo-angle').value));
  $$('#servo-quick button').forEach((button) => button.addEventListener('click', async () => {
    if (button.dataset.angle === 'initial') {
      await loadExecutionConfig();
      const angle = app.executionConfig?.data?.tool_motor?.initial_angle_deg;
      if (angle === undefined) return toast('无法读取初始角', '请检查 execution.yaml', 'error');
      $('#servo-angle').value = angle;
      sendServo(angle);
    } else {
      $('#servo-angle').value = button.dataset.angle;
      sendServo(button.dataset.angle);
    }
  }));

  $('#servo-sweep-start').addEventListener('click', async () => {
    const min = Number($('#sweep-min').value);
    const max = Number($('#sweep-max').value);
    const wait = Number($('#sweep-wait').value);
    const repeat = Number($('#sweep-repeat').value);
    if (![min, max, wait, repeat].every(Number.isFinite)) {
      return toast('往复参数无效', '起始角、终止角、间隔、次数必须是数值', 'error');
    }
    if (min < 0 || max > 360 || min >= max) {
      return toast('往复参数无效', '角度必须在 0～360° 且起始角小于终止角', 'error');
    }
    if (wait <= 0) return toast('往复参数无效', '间隔必须大于 0', 'error');
    if (repeat < 0 || !Number.isInteger(repeat)) {
      return toast('往复参数无效', '次数必须是非负整数（0 = 无限）', 'error');
    }
    let confirmed = false;
    const motor = app.executionConfig?.data?.tool_motor;
    const outside = motor && (min < motor.lower_margin_deg || max > motor.upper_margin_deg);
    if (outside) {
      confirmed = await confirmAction(
        '往复范围超出正式安全边界',
        `往复范围 ${min}°～${max}° 超出 ${motor.lower_margin_deg}°～${motor.upper_margin_deg}°。只在确认机构不会顶到限位时继续。`,
      );
      if (!confirmed) return;
    }
    command('/api/control/servo-sweep/start', {
      min_deg: min, max_deg: max, wait_seconds: wait, repeat_count: repeat,
      confirm_outside_safe: confirmed,
    }, '往复测试已开始');
  });
  $('#servo-sweep-stop').addEventListener('click', () => {
    command('/api/control/servo-sweep/stop', {}, '停止往复已提交');
  });

  $('#pose-refresh').addEventListener('click', async () => {
    try {
      const result = await api('/api/control/pose');
      app.pose.tcp = result.tcp_pose;
      app.pose.camera = result.camera_pose;
      $('#tcp-pose').textContent = formatPose(result.tcp_pose);
      $('#camera-pose').textContent = formatPose(result.camera_pose);
      toast('位姿已更新');
    } catch (error) { toast('位姿读取失败', formatError(error), 'error'); }
  });
  $$('[data-copy]').forEach((button) => button.addEventListener('click', async () => {
    const value = app.pose[button.dataset.copy];
    if (!value) return toast('没有可复制的位姿', '请先点击读取位姿', 'warning');
    await navigator.clipboard.writeText(JSON.stringify(value));
    toast('位姿已复制');
  }));
  $('#arm-reset').addEventListener('click', async () => {
    await loadExecutionConfig();
    const pose = app.executionConfig?.data?.shooting_pose;
    if (!pose) return;
    if (await confirmAction('复位到高位拍摄位？', '机械臂将以速度 50 运动。请核对完整目标位姿并确认运动空间安全。', formatPose(pose))) {
      command('/api/control/reset', { confirmed_pose: true }, '机械臂复位已提交');
    }
  });
  $('#relative-move').addEventListener('click', async () => {
    const dx = Number($('#move-dx').value);
    const dy = Number($('#move-dy').value);
    const dz = Number($('#move-dz').value);
    const speed = Number($('#move-speed').value);
    if (![dx, dy, dz].every(Number.isFinite)) return toast('增量无效', 'dx/dy/dz 必须是数值', 'error');
    if (dx === 0 && dy === 0 && dz === 0) return toast('增量无效', 'dx/dy/dz 不能同时为 0', 'warning');
    if (!Number.isFinite(speed) || speed <= 0) return toast('速度无效', '速度必须大于 0', 'error');
    let tcp;
    try {
      const result = await api('/api/control/pose');
      tcp = result.tcp_pose;
    } catch (error) {
      return toast('无法读取当前位姿', formatError(error), 'error');
    }
    const target = tcp.slice();
    target[0] += dx; target[1] += dy; target[2] += dz;
    const detail = `当前 TCP：${formatPose(tcp)}\n增量：ΔX=${dx} ΔY=${dy} ΔZ=${dz} mm\n目标 TCP：${formatPose(target)}\n速度：${speed} mm/s`;
    if (await confirmAction('增量移动机械臂？', '将以当前 TCP 为基准平移并保持姿态不变。请确认运动空间安全。', detail)) {
      command('/api/control/move-relative', { dx, dy, dz, speed }, '增量移动已提交');
    }
  });
  $('#aruco-align-start').addEventListener('click', async () => {
    await loadExecutionConfig();
    const value = Number($('#aruco-low-z').value);
    const motion = app.executionConfig?.data?.motion || {};
    const minimumZ = Number(motion.minimum_tcp_z_mm);
    if (!Number.isFinite(value)) {
      return toast('参数错误', '低位 TCP Z 必须是数值', 'error');
    }
    if (Number.isFinite(minimumZ) && value < minimumZ) {
      return toast('低位 TCP Z 过低', `不能低于安全下限 ${minimumZ} mm`, 'error');
    }
    const pose = app.executionConfig?.data?.shooting_pose;
    const detail = `完整拍摄位姿：${pose ? formatPose(pose) : '未读取到 execution.yaml'}\n低位固定 TCP Z：${value} mm\n结束后保持最终位置，不返回高位。`;
    if (await confirmAction(
      '开始 ArUco 单次对准？',
      '机械臂将先运动到高位拍摄位，再下降到低位做 ArUco 闭环对准。请确认工作区安全、吸盘已关闭且未持块，ArUco 板在高位和低位都能被相机看到。',
      detail,
    )) {
      command('/api/tools/aruco-align', { low_tcp_z_mm: value, confirmed: true }, 'ArUco 单次对准已提交');
    }
  });
  $('#clear-stop').addEventListener('click', async () => {
    if (await confirmAction('解除停止锁？', '请先在现场确认机械臂已经停稳、原因已经排除，且工作区重新安全。解除后也必须重新启动感知并识别。')) {
      command('/api/control/clear-stop', {}, '解除停止锁已提交');
    }
  });
}

function formatPose(pose) {
  return `[${(pose || []).map((value) => Number(value).toFixed(3)).join(', ')}]`;
}

async function loadExecutionConfig(force = false) {
  if (app.executionConfig && !force) return app.executionConfig;
  try {
    app.executionConfig = await api('/api/config/execution');
    const pose = app.executionConfig.data.shooting_pose;
    $('#reset-pose').textContent = formatPose(pose);
    $('#servo-angle').value = app.executionConfig.data.tool_motor.initial_angle_deg;
    const minimumZ = Number(app.executionConfig.data?.motion?.minimum_tcp_z_mm);
    if (Number.isFinite(minimumZ)) {
      const lowZ = $('#aruco-low-z');
      lowZ.min = minimumZ;
      if (Number(lowZ.value) < minimumZ) {
        lowZ.value = Math.max(220, minimumZ);
      }
    }
    return app.executionConfig;
  } catch (error) {
    $('#reset-pose').textContent = `加载失败：${formatError(error)}`;
    return null;
  }
}

async function initializeConfigCenter() {
  if (!app.configFiles.length) await loadConfigFiles();
  if (!app.configDocuments.size) await loadAllConfigDocuments();
  await loadPresets();
  if (!app.currentConfigId && app.configFiles.length) await selectConfig(app.configFiles[0].file_id);
}

async function loadConfigFiles() {
  try {
    const result = await api('/api/config');
    app.configFiles = result.files;
    renderConfigFiles();
  } catch (error) { toast('配置列表读取失败', formatError(error), 'error'); }
}

function renderConfigFiles() {
  const container = $('#config-files');
  container.replaceChildren();
  app.configFiles.forEach((file) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'config-file-card';
    button.dataset.fileId = file.file_id;
    const title = document.createElement('strong');
    title.textContent = file.label;
    const description = document.createElement('small');
    description.textContent = file.description;
    button.append(title, description);
    button.addEventListener('click', () => selectConfig(file.file_id));
    container.append(button);
  });
  renderConfigFileStates();
  renderConfigActions();
}

function scopeRunning(scope) {
  const process = app.state?.process || {};
  if (scope === 'hardware') return Boolean(process.hardware?.running);
  if (scope === 'perception') return Boolean(process.runtime?.running);
  return false;
}

function scopeExternal(scope) {
  const process = app.state?.process || {};
  if (scope === 'hardware') return Boolean(process.hardware?.external || process.runtime?.external);
  if (scope === 'perception') return Boolean(process.runtime?.external);
  return false;
}

function renderConfigFileStates() {
  const pending = new Set(app.state?.config?.pending_restart || []);
  $$('.config-file-card').forEach((button) => {
    button.classList.toggle('active', button.dataset.fileId === app.currentConfigId);
    const file = app.configFiles.find((item) => item.file_id === button.dataset.fileId);
    const scope = file?.restart_scope;
    // 只有“正在运行且持有旧参数”的作用域才显示待重启；外部节点需手动重启。
    const pendingScope = Boolean(scope && pending.has(scope) && scopeRunning(scope));
    button.classList.toggle('pending', pendingScope);
    button.classList.toggle('pending-external', pendingScope && scopeExternal(scope));
  });
}

function renderConfigActions() {
  const applyButton = $('#config-apply');
  const process = app.state?.process || {};
  const stopOperation = app.state?.operation?.stop;
  const active = app.state?.operation?.active || (stopOperation?.status === 'running' ? stopOperation : null);
  const task = app.state?.task || {};
  const stopped = Boolean(app.state?.hardware?.stop_latched);
  const file = app.currentConfigId
    ? app.configFiles.find((item) => item.file_id === app.currentConfigId)
    : null;
  const scope = file?.restart_scope;

  let reason = '';
  let enabled = true;
  if (!file) {
    enabled = false;
    reason = '尚未选择配置';
  } else if (active) {
    enabled = false;
    reason = `正在${active.kind}，请等待结束`;
  } else if (stopped) {
    enabled = false;
    reason = '停止锁已锁定，先解除停止锁';
  } else if (['准备识别', '执行中'].includes(task.state)) {
    enabled = false;
    reason = '识别或任务执行期间不能保存参数';
  } else if (scope === 'hardware') {
    if (!process.hardware?.running) { enabled = false; reason = '硬件未启动，没有节点可重启'; }
    else if (scopeExternal('hardware')) { enabled = false; reason = '节点由外部进程启动，控制台不能重启它'; }
  } else if (scope === 'perception') {
    if (!process.runtime?.running) { enabled = false; reason = '感知未启动，没有节点可重启'; }
    else if (scopeExternal('perception')) { enabled = false; reason = '感知节点由外部进程启动，控制台不能重启它'; }
  }
  applyButton.disabled = !enabled;
  applyButton.title = reason;
}

async function loadAllConfigDocuments(force = false) {
  if (force) app.configDocuments.clear();
  const files = app.configFiles.length ? app.configFiles : (await api('/api/config')).files;
  const results = await Promise.allSettled(files.map((file) => api(`/api/config/${file.file_id}`)));
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') app.configDocuments.set(files[index].file_id, result.value);
  });
}

async function selectConfig(fileId, force = false) {
  if (!force && app.currentConfigId && app.currentConfigId !== fileId && configEditor.changedFields().length) {
    if (!await confirmAction('放弃未保存修改？', '切换配置文件会丢弃当前页面里的未保存修改。')) return;
  }
  try {
    const document = force || !app.configDocuments.has(fileId)
      ? await api(`/api/config/${fileId}`)
      : app.configDocuments.get(fileId);
    app.configDocuments.set(fileId, document);
    app.currentConfigId = fileId;
    configEditor.load(document);
    $('#config-file-id').textContent = `${document.file_id} · 重启范围 ${document.restart_scope === 'hardware' ? '硬件' : '感知'}`;
    $('#config-title').textContent = document.label;
    $('#config-description').textContent = document.description;
    renderConfigFileStates();
    renderConfigActions();
    await loadHistory();
  } catch (error) { toast('配置读取失败', formatError(error), 'error'); }
}

async function saveCurrentConfig(apply) {
  if (!app.currentConfigId || !configEditor.document) return;
  const changes = configEditor.changedFields();
  const scope = app.configFiles.find((item) => item.file_id === app.currentConfigId)?.restart_scope;
  // “保存并重启”在没有未保存修改、但该作用域仍有待重启时，也允许提交一次重启。
  const applyPendingRestart = apply && (app.state?.config?.pending_restart || []).includes(scope);
  if (!changes.length && !applyPendingRestart) return toast('当前没有修改', '', 'warning');
  let confirmDangerous = false;
  if (configEditor.hasDangerousChanges()) {
    confirmDangerous = await confirmAction(
      '保存危险参数修改？',
      '修改包含位姿、高度、安全范围、映射矩阵、控制停止参数或模型路径。保存前请再次核对。',
      configEditor.differenceText(),
    );
    if (!confirmDangerous) return;
  }
  try {
    const result = await api(`/api/config/${app.currentConfigId}`, {
      method: 'PUT',
      json: {
        data: configEditor.data,
        revision: configEditor.document.revision,
        confirm_dangerous: confirmDangerous,
        apply,
      },
    });
    toast(apply ? '保存并重启已提交' : '仅保存已提交', `操作编号 ${result.operation_id.slice(0, 8)}`);
  } catch (error) {
    if (error.code === 'revision_conflict') {
      toast('文件版本冲突', '配置已被 VS Code 或其它程序修改，请重新读取后再编辑。', 'error', 7000);
    } else {
      toast('配置未保存', formatError(error), 'error', 7000);
    }
  }
}

async function refreshConfigAfterWrite() {
  const selected = app.currentConfigId;
  await loadConfigFiles();
  await loadAllConfigDocuments(true);
  app.executionConfig = null;
  if (selected) await selectConfig(selected, true);
  await loadPresets();
}

async function loadHistory() {
  if (!app.currentConfigId) return;
  const container = $('#history-list');
  container.textContent = '加载中…';
  try {
    const result = await api(`/api/config/history?file_id=${encodeURIComponent(app.currentConfigId)}&limit=80`);
    container.replaceChildren();
    if (!result.history.length) {
      const empty = document.createElement('p'); empty.className = 'muted'; empty.textContent = '这份配置还没有网页修改历史。'; container.append(empty); return;
    }
    result.history.forEach((entry) => {
      const row = document.createElement('div'); row.className = 'history-item';
      const time = document.createElement('small'); time.textContent = entry.created_at.replace('T', ' ');
      const reason = document.createElement('strong'); reason.textContent = entry.reason;
      const diff = document.createElement('button'); diff.type = 'button'; diff.textContent = '查看差异'; diff.addEventListener('click', () => showDiff(`${entry.reason} · #${entry.id}`, entry.diff_text));
      const restore = document.createElement('button'); restore.type = 'button'; restore.textContent = '恢复修改前'; restore.addEventListener('click', () => restoreHistory(entry));
      row.append(time, reason, diff, restore); container.append(row);
    });
  } catch (error) { container.textContent = `历史读取失败：${formatError(error)}`; }
}

async function restoreHistory(entry) {
  const confirmed = await confirmAction('恢复到该次修改之前？', '恢复也会生成新的历史记录，并将当前识别结果失效。危险参数按已确认处理。', entry.diff_text);
  if (!confirmed) return;
  try {
    const result = await api('/api/config/history', {
      method: 'POST',
      json: { history_id: entry.id, revision: configEditor.document.revision, confirm_dangerous: true, apply: false },
    });
    toast('历史恢复已提交', `操作编号 ${result.operation_id.slice(0, 8)}`);
  } catch (error) { toast('历史恢复未执行', formatError(error), 'error'); }
}

async function loadPresets() {
  try {
    const result = await api('/api/presets');
    const container = $('#preset-list');
    container.replaceChildren();
    result.presets.forEach((preset) => {
      const row = document.createElement('div'); row.className = 'preset-item';
      const info = document.createElement('div');
      const title = document.createElement('strong'); title.textContent = preset.name;
      const time = document.createElement('small'); time.textContent = preset.updated_at.replace('T', ' ');
      info.append(title, time);
      const actions = document.createElement('div');
      const diff = document.createElement('button'); diff.type = 'button'; diff.textContent = '差异'; diff.addEventListener('click', () => showPresetDiff(preset));
      const restore = document.createElement('button'); restore.type = 'button'; restore.textContent = '恢复'; restore.addEventListener('click', () => restorePreset(preset));
      actions.append(diff, restore);
      if (!preset.protected) {
        const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '删除'; remove.addEventListener('click', () => deletePreset(preset)); actions.append(remove);
      }
      row.append(info, actions); container.append(row);
    });
  } catch (error) { toast('预设读取失败', formatError(error), 'error'); }
}

async function showPresetDiff(preset) {
  try {
    const result = await api(`/api/presets?preset_id=${preset.id}&include_diff=1`);
    const text = Object.entries(result.diffs || {}).map(([file, diff]) => `===== ${file} =====\n${diff}`).join('\n');
    showDiff(`当前配置 ↔ ${preset.name}`, text || '当前配置与该预设完全一致。');
  } catch (error) { toast('预设差异读取失败', formatError(error), 'error'); }
}

async function restorePreset(preset) {
  const result = await api(`/api/presets?preset_id=${preset.id}&include_diff=1`);
  const text = Object.entries(result.diffs || {}).map(([file, diff]) => `===== ${file} =====\n${diff}`).join('\n');
  if (!text) return toast('无需恢复', '当前配置与该预设一致');
  if (!await confirmAction(`恢复预设“${preset.name}”？`, '将以事务方式恢复六份配置。若任一文件版本冲突或写入失败，全部回滚。', text)) return;
  const revisions = Object.fromEntries(app.configFiles.map((file) => [file.file_id, file.revision]));
  try {
    const response = await api('/api/presets', { method: 'POST', json: { action: 'restore', preset_id: preset.id, revisions, confirm_dangerous: true, apply: false } });
    toast('预设恢复已提交', `操作编号 ${response.operation_id.slice(0, 8)}`);
  } catch (error) { toast('预设恢复未执行', formatError(error), 'error'); }
}

async function deletePreset(preset) {
  if (!await confirmAction(`删除预设“${preset.name}”？`, '删除命名预设不会修改当前 YAML，但删除后无法从预设列表恢复。')) return;
  try {
    const response = await api('/api/presets', { method: 'DELETE', json: { preset_id: preset.id } });
    toast('删除预设已提交', `操作编号 ${response.operation_id.slice(0, 8)}`);
  } catch (error) { toast('预设删除失败', formatError(error), 'error'); }
}

function bindConfig() {
  $('#config-reload').addEventListener('click', async () => {
    if (configEditor.changedFields().length && !await confirmAction('重新读取文件？', '页面上的未保存修改将丢失。')) return;
    if (app.currentConfigId) await selectConfig(app.currentConfigId, true);
  });
  $('#config-diff').addEventListener('click', () => showDiff('当前未保存修改', configEditor.differenceText()));
  $('#config-save').addEventListener('click', () => saveCurrentConfig(false));
  $('#config-apply').addEventListener('click', () => saveCurrentConfig(true));
  $('#history-refresh').addEventListener('click', loadHistory);
  $('#preset-new').addEventListener('click', async () => {
    const name = window.prompt('输入预设名称（将保存六份完整配置快照）：');
    if (!name?.trim()) return;
    try {
      const response = await api('/api/presets', { method: 'POST', json: { action: 'save', name: name.trim() } });
      toast('保存预设已提交', `操作编号 ${response.operation_id.slice(0, 8)}`);
    } catch (error) { toast('预设保存失败', formatError(error), 'error'); }
  });

  let searchTimer;
  $('#config-search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(renderConfigSearch, 120);
  });
}

async function renderConfigSearch() {
  const query = $('#config-search').value.trim();
  const container = $('#config-search-results');
  if (!query) { container.hidden = true; container.replaceChildren(); return; }
  if (!app.configDocuments.size) await loadAllConfigDocuments();
  const results = [...app.configDocuments.values()].flatMap((document) => flattenSearch(document, query)).slice(0, 150);
  container.replaceChildren(); container.hidden = false;
  if (!results.length) { container.textContent = '没有匹配字段。'; return; }
  results.forEach((result) => {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'search-result';
    const file = document.createElement('span'); file.textContent = result.fileLabel;
    const path = document.createElement('code'); path.textContent = result.path;
    const value = document.createElement('span'); value.textContent = `${result.metadata.description || ''} · ${JSON.stringify(result.value)}`;
    button.append(file, path, value);
    button.addEventListener('click', async () => {
      $('#config-search').value = '';
      container.hidden = true;
      await selectConfig(result.fileId);
      configEditor.focusPath(result.path);
    });
    container.append(button);
  });
}

async function loadReadOnly() {
  const container = $('#readonly-content');
  container.replaceChildren();
  try {
    const [result, chainResult] = await Promise.all([
      api('/api/read-only-config'),
      api('/api/localization-z-chain').then((chain) => ({ chain }), (error) => ({ chainError: error })),
    ]);
    if (chainResult.chain) {
      container.append(renderZChain(chainResult.chain));
    } else {
      const chainCard = document.createElement('article'); chainCard.className = 'card empty-state';
      chainCard.textContent = `定位 Z 链路读取失败：${formatError(chainResult.chainError)}`;
      container.append(chainCard);
    }
    const liveBoard = app.state?.task?.board;
    let renderedLive = false;
    if (liveBoard?.targets?.length) {
      container.append(renderLayout({
        label: liveBoard.mode_label || '实际盘面',
        subtitle: `实际盘面 · ${liveBoard.targets.length} 项 · 识别结果`,
        revision: '',
        target_count: liveBoard.targets.length,
        data: { targets: liveBoard.targets },
      }));
      renderedLive = true;
    }
    const layout = result.items.find((item) => item.kind === 'layout');
    if (layout?.exists && !renderedLive) {
      container.append(renderLayout({ ...layout, subtitle: `10×14 盘面 · ${layout.target_count} 项 · 固定参考（尚未识别）` }));
    }
    const grid = document.createElement('div'); grid.className = 'calibration-grid';
    result.items.filter((item) => item.kind !== 'layout').forEach((item) => grid.append(renderCalibration(item)));
    container.append(grid);
  } catch (error) {
    const card = document.createElement('article'); card.className = 'card empty-state'; card.textContent = `读取失败：${formatError(error)}`; container.append(card);
  }
}

function formatZChainValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return `${Number(value.min).toFixed(2)} ~ ${Number(value.max).toFixed(2)}`;
  return Number(value).toFixed(2);
}

function renderZChain(data) {
  const card = document.createElement('article'); card.className = 'card calibration-card z-chain-card';
  const top = document.createElement('div'); top.className = 'card-heading';
  top.append(heading('定位 Z 计算链路', data.mode_label || ''));
  const badge = document.createElement('span');
  badge.className = 'state-badge';
  badge.textContent = data.mode === 'fixed_constant' ? '固定 Z' : 'z_plane 斜面';
  top.append(badge); card.append(top);

  const dl = document.createElement('dl');
  data.constants.forEach((item) => {
    const dt = document.createElement('dt'); dt.textContent = `${item.symbol} ${item.label}`;
    const dd = document.createElement('dd');
    dd.textContent = item.value === null ? '—（未启用）' : String(item.value);
    dd.title = item.source;
    dl.append(dt, dd);
  });
  card.append(dl);

  const wrap = document.createElement('div'); wrap.className = 'table-wrap';
  const table = document.createElement('table'); table.className = 'layout-table';
  const header = document.createElement('thead'); const headerRow = document.createElement('tr');
  ['阶段', '公式', '代入', '结果(mm)'].forEach((label) => { const th = document.createElement('th'); th.textContent = label; headerRow.append(th); });
  header.append(headerRow); table.append(header);
  const body = document.createElement('tbody');
  data.rows.forEach((row) => {
    const tr = document.createElement('tr');
    const stageCell = document.createElement('td'); stageCell.textContent = row.stage; tr.append(stageCell);
    [row.formula, row.substitution, formatZChainValue(row.value)].forEach((value) => {
      const td = document.createElement('td'); td.className = 'mono'; td.textContent = value; tr.append(td);
    });
    body.append(tr);
  });
  table.append(body); wrap.append(table); card.append(wrap);

  data.checks.forEach((check) => {
    const line = document.createElement('p'); line.className = 'hint';
    const stateBadge = document.createElement('span');
    stateBadge.className = `state-badge ${check.ok ? '' : 'error'}`.trim();
    stateBadge.textContent = check.ok ? '通过' : '不通过';
    line.append(stateBadge, `${check.name}：${check.detail}`);
    card.append(line);
  });

  const planeLabels = { block: '方块', tray: '托盘' };
  Object.keys(planeLabels).forEach((subject) => {
    const plane = data.z_plane?.[subject];
    if (!plane) return;
    const role = data.mode === 'fixed_constant' ? '对照（已被固定常数绕过）' : '当前生效';
    const [a, b, c] = plane.coefficients;
    const line = document.createElement('p'); line.className = 'hint mono';
    line.textContent = `${role}${planeLabels[subject]}平面：z = ${a}*x + ${b}*y + ${c}，安全区范围 ${formatZChainValue(plane.safe_box_range)}，批次 ${plane.generation_id || '—'}`;
    card.append(line);
  });

  data.notes.forEach((note) => {
    const line = document.createElement('p'); line.className = 'hint'; line.textContent = note;
    card.append(line);
  });
  return card;
}

function heading(title, subtitle = '') {
  const wrapper = document.createElement('div');
  const h2 = document.createElement('h2'); h2.textContent = title; wrapper.append(h2);
  if (subtitle) { const small = document.createElement('small'); small.textContent = subtitle; wrapper.append(small); }
  return wrapper;
}

function renderLayout(item) {
  const card = document.createElement('article'); card.className = 'card layout-card';
  const top = document.createElement('div'); top.className = 'card-heading';
  top.append(heading(item.label, item.subtitle || `10×14 盘面 · ${item.target_count} 项 · 只读`));
  const revision = document.createElement('span'); revision.className = 'mono'; revision.textContent = (item.revision || '').slice(0, 10); top.append(revision); card.append(top);
  const content = document.createElement('div'); content.className = 'layout-content';
  const board = document.createElement('div'); board.className = 'board-preview';
  const occupied = new Map();
  item.data.targets.forEach((target) => target.cells.forEach(([col, row]) => occupied.set(`${col}:${row}`, target.category)));
  // 盘面按机械臂前方的观察方向显示：底部行在上，顶部行在下。
  for (let row = 14; row >= 1; row -= 1) for (let col = 1; col <= 10; col += 1) {
    const cell = document.createElement('span'); cell.className = 'board-cell';
    const category = occupied.get(`${col}:${row}`);
    if (category) { cell.classList.add('filled'); cell.dataset.category = category; cell.title = `${col}, ${row} · ${category}`; }
    board.append(cell);
  }
  const wrap = document.createElement('div'); wrap.className = 'table-wrap';
  const table = document.createElement('table'); table.className = 'layout-table';
  const header = document.createElement('thead'); const headerRow = document.createElement('tr');
  ['序号', '类别', '列', '行', '角度', '占用格'].forEach((label) => { const th = document.createElement('th'); th.textContent = label; headerRow.append(th); }); header.append(headerRow); table.append(header);
  const body = document.createElement('tbody');
  item.data.targets.forEach((target, index) => {
    const row = document.createElement('tr');
    [index + 1, target.category, target.col, target.row, `${target.angle_deg}°`, target.cells.map((cell) => cell.join(',')).join(' / ')].forEach((value) => { const td = document.createElement('td'); td.textContent = value; row.append(td); }); body.append(row);
  });
  table.append(body); wrap.append(table); content.append(board, wrap); card.append(content); return card;
}

function renderCalibration(item) {
  const card = document.createElement('article'); card.className = 'card calibration-card';
  const top = document.createElement('div'); top.className = 'card-heading'; top.append(heading(item.label, item.exists ? item.modified_at : '文件不存在'));
  const badge = document.createElement('span'); badge.className = `state-badge ${item.valid === false ? 'error' : ''}`; badge.textContent = item.exists ? (item.valid === false ? '校验失败' : '只读') : '缺失'; top.append(badge); card.append(top);
  if (!item.exists) return card;
  if (item.kind === 'matrix') {
    const table = document.createElement('table'); table.className = 'matrix-table';
    (item.matrix || []).forEach((matrixRow) => { const tr = document.createElement('tr'); matrixRow.forEach((value) => { const td = document.createElement('td'); td.textContent = Number(value).toFixed(6); tr.append(td); }); table.append(tr); }); card.append(table); return card;
  }
  const summary = item.summary || {};
  const dl = document.createElement('dl');
  const rows = [
    ['生成批次', summary.generation_id || '—'],
    ['标定主体', summary.subject || '—'],
    ['样本数量', summary.metrics?.sample_count ?? summary.coverage?.sample_count ?? '—'],
    ['训练 RMSE', summary.metrics?.xy_full_training_rmse_mm ?? '—'],
    ['覆盖范围', JSON.stringify(summary.coverage?.pixel_range || {})],
    ['文件路径', item.path],
  ];
  rows.forEach(([key, value]) => { const dt = document.createElement('dt'); dt.textContent = key; const dd = document.createElement('dd'); dd.textContent = value; dl.append(dt, dd); }); card.append(dl); return card;
}

function bindReadOnly() {
  $('#readonly-refresh').addEventListener('click', loadReadOnly);
  $('#deploy-calibration').addEventListener('click', async () => {
    const blockFile = $('#block-calibration-file').files[0];
    const trayFile = $('#tray-calibration-file').files[0];
    if (!blockFile || !trayFile) return toast('请选择两份 YAML', '方块与托盘标定必须成对部署', 'warning');
    if (!await confirmAction('成对部署像素标定？', '系统会校验 schema、主体和 generation_id，先备份当前文件，再一起替换。替换后必须重启感知。')) return;
    try {
      const [blockYaml, trayYaml] = await Promise.all([blockFile.text(), trayFile.text()]);
      await command('/api/calibration/deploy-pair', { block_yaml: blockYaml, tray_yaml: trayYaml, confirmed: true }, '像素标定部署已提交');
    } catch (error) { if (!(error instanceof ApiError)) toast('文件读取失败', formatError(error), 'error'); }
  });
  $('#deploy-hand-eye').addEventListener('click', async () => {
    const file = $('#hand-eye-file').files[0];
    if (!file) return toast('请选择 4×4 JSON 文件', '', 'warning');
    let matrix;
    try { matrix = JSON.parse(await file.text()); }
    catch (error) { return toast('JSON 解析失败', formatError(error), 'error'); }
    if (!await confirmAction('部署手眼矩阵？', '系统会校验有限 4×4 数组并备份当前 NPY。替换后必须安全重启硬件。', JSON.stringify(matrix, null, 2))) return;
    try { await command('/api/calibration/deploy-hand-eye', { matrix, confirmed: true }, '手眼矩阵部署已提交'); }
    catch (_) { /* 错误已经显示。 */ }
  });
}

function appendLog(data) {
  app.logs.push({ ...data, localTime: new Date() });
  if (app.logs.length > 1200) app.logs.splice(0, app.logs.length - 1000);
  scheduleLogRender();
}

function scheduleLogRender() {
  if (app.logRenderScheduled) return;
  app.logRenderScheduled = true;
  requestAnimationFrame(() => { app.logRenderScheduled = false; renderLogs(); });
}

function sourceChannel(source) {
  const value = source || '';
  if (value.startsWith('launch:hardware') || value === '/camera_node' || value === '/control_node') return 'hardware';
  if (value.startsWith('launch:runtime') || value === '/image_process_node' || value === '/competition_node') return 'perception';
  if (value === '手动控制' || value === 'ArUco对准') return 'manual';
  return 'system';
}

function renderLogs() {
  const level = $('#log-level').value;
  const channel = app.selectedLogChannel;
  const query = $('#log-search').value.trim().toLowerCase();
  const container = $('#log-list');
  const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
  const filtered = app.logs.filter((entry) => {
    const normalized = ['fatal', 'error'].includes(entry.level) ? 'error' : entry.level;
    const levelMatch = level === 'all' || normalized === level;
    const channelMatch = channel === 'all' || sourceChannel(entry.source) === channel;
    const queryMatch = !query || `${entry.source} ${entry.message}`.toLowerCase().includes(query);
    return levelMatch && channelMatch && queryMatch;
  }).slice(-700);
  const fragment = document.createDocumentFragment();
  filtered.forEach((entry) => {
    const row = document.createElement('div'); row.className = `log-row ${entry.level || 'info'}`;
    const time = document.createElement('span'); time.className = 'time'; time.textContent = entry.localTime.toLocaleTimeString('zh-CN', { hour12: false }) + `.${String(entry.localTime.getMilliseconds()).padStart(3, '0')}`;
    const levelText = document.createElement('span'); levelText.className = 'level'; levelText.textContent = ({ debug: '调试', info: '信息', warning: '警告', error: '错误', fatal: '致命' })[entry.level] || entry.level;
    const source = document.createElement('span'); source.className = 'source'; source.textContent = entry.source || '未知';
    const message = document.createElement('span'); message.className = 'message'; message.textContent = entry.message || '';
    row.append(time, levelText, source, message); fragment.append(row);
  });
  container.replaceChildren(fragment);
  if ($('#log-autoscroll').checked && (atBottom || filtered.length)) container.scrollTop = container.scrollHeight;
}

function bindLogs() {
  $('#log-level').addEventListener('change', renderLogs);
  $('#log-search').addEventListener('input', renderLogs);
  $('#log-clear').addEventListener('click', () => { app.logs = []; renderLogs(); });
  $$('.log-tabs button').forEach((button) => button.addEventListener('click', () => {
    app.selectedLogChannel = button.dataset.logChannel;
    $$('.log-tabs button').forEach((item) => item.classList.toggle('active', item === button));
    renderLogs();
  }));
}

const LAUNCH_LOG_TAIL_LINES = 5000;
const LAUNCH_LOG_ALL_LINES = 1000000;
const LAUNCH_LOG_REFRESH_MS = 2000;

const launchLogViewer = {
  timer: 0,
  allMode: false,
};

function launchLogUrl(kind, lines) {
  return `/api/launch-log/${encodeURIComponent(kind)}?lines=${lines}`;
}

function formatBytes(size) {
  if (!Number.isFinite(size)) return '—';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(2)} MB`;
}

async function loadLaunchLogMeta(kind) {
  const download = $('#launch-log-download');
  download.href = `/api/launch-log/${encodeURIComponent(kind)}?download=1`;
  download.hidden = true;
  try {
    const summary = await api('/api/launch-log');
    const meta = (summary.files || []).find((item) => item.kind === kind);
    if (!meta) {
      $('#launch-log-meta').textContent = '控制台未配置 launch 日志目录。';
      return;
    }
    $('#launch-log-meta').textContent = meta.exists
      ? `${meta.path} · ${formatBytes(meta.size)} · 更新 ${meta.modified_at || '—'}${launchLogViewer.allMode ? ' · 已显示全部行（自动刷新暂停）' : ' · 每 2 秒自动刷新'}`
      : `${meta.path} · 尚未生成（还没有产生过该类输出）`;
    download.hidden = !meta.exists;
  } catch (error) {
    $('#launch-log-meta').textContent = formatError(error);
  }
}

async function loadLaunchLog(options = {}) {
  const kind = app.launchLogKind;
  const content = $('#launch-log-content');
  const follow = $('#launch-log-follow').checked;
  const hadContent = content.textContent !== '' && content.textContent !== '读取中…';
  // 替换内容前记录滚动状态：在底部且开启跟随才贴底，否则还原原位置，绝不拽走阅读位置。
  const atBottom = content.scrollHeight - content.scrollTop - content.clientHeight < 40;
  const scrollFromTop = content.scrollTop;
  loadLaunchLogMeta(kind);
  if (options.showLoading || !hadContent) content.textContent = '读取中…';
  try {
    const text = await api(launchLogUrl(kind, launchLogViewer.allMode ? LAUNCH_LOG_ALL_LINES : LAUNCH_LOG_TAIL_LINES));
    content.textContent = text || '（暂无输出）';
  } catch (error) {
    if (!hadContent) content.textContent = '';
    toast('读取 launch 日志失败', formatError(error), 'error');
    return;
  }
  if (follow && atBottom) content.scrollTop = content.scrollHeight;
  else if (hadContent) content.scrollTop = scrollFromTop;
}

function bindLaunchLogs() {
  const dialog = $('#launch-log-dialog');
  $('#launch-log-open').addEventListener('click', () => {
    launchLogViewer.allMode = false;
    $('#launch-log-content').textContent = '';
    launchLogViewer.timer = window.setInterval(() => {
      // 「显示全部行」时数据量太大，暂停自动刷新，避免每 2 秒重传整份日志。
      if (!launchLogViewer.allMode) loadLaunchLog();
    }, LAUNCH_LOG_REFRESH_MS);
    dialog.showModal();
    loadLaunchLog();
  });
  dialog.addEventListener('close', () => window.clearInterval(launchLogViewer.timer));
  $('#launch-log-close').addEventListener('click', () => dialog.close());
  $('#launch-log-refresh').addEventListener('click', () => {
    launchLogViewer.allMode = false;
    loadLaunchLog({ showLoading: true });
  });
  $('#launch-log-all').addEventListener('click', async () => {
    const content = $('#launch-log-content');
    launchLogViewer.allMode = true;
    content.textContent = '读取中…';
    try {
      content.textContent = (await api(launchLogUrl(app.launchLogKind, LAUNCH_LOG_ALL_LINES))) || '（暂无输出）';
    } catch (error) {
      content.textContent = '';
      toast('读取完整日志失败', formatError(error), 'error');
    }
    loadLaunchLogMeta(app.launchLogKind);
    content.scrollTop = content.scrollHeight;
  });
  $$('.launch-log-tabs button').forEach((button) => button.addEventListener('click', () => {
    app.launchLogKind = button.dataset.launchLogKind;
    launchLogViewer.allMode = false;
    $('#launch-log-content').textContent = '';
    $$('.launch-log-tabs button').forEach((item) => item.classList.toggle('active', item === button));
    loadLaunchLog();
  }));
}

function bindExit() {
  $('#exit-panel').addEventListener('click', async () => {
    if (!await confirmAction('显式退出控制台？', '将停止控制台自己启动的感知、硬件和 ROS Master。外部 ROS 进程不会被结束。关闭浏览器本身不会执行这个动作。')) return;
    try {
      await api('/api/system/exit', { method: 'POST', json: {} });
      toast('控制台正在退出', '此页面稍后将无法连接');
    } catch (error) { toast('退出请求失败', formatError(error), 'error'); }
  });
}

function bindTaskInteraction() {
  const dialog = $('#task-interaction-dialog');
  dialog.addEventListener('cancel', (event) => event.preventDefault());
  window.addEventListener('pagehide', () => {
    const interaction = app.state?.interaction;
    if (!interaction?.pending || !interaction.prompt_id) return;
    // 页面被关闭或刷新时只结束本轮识别，不发 StopMotion，也不改变吸盘状态。
    // keepalive 让请求可在页面卸载后继续发送；断网或浏览器崩溃仍由 60 秒超时兜底。
    api('/api/task/interaction/respond', {
      method: 'POST',
      keepalive: true,
      json: { prompt_id: interaction.prompt_id, choice: 'stop' },
    }).catch(() => {});
  });
  $('#task-interaction-stop-motion').addEventListener('click', async () => {
    if (app.interactionSubmitting) return;
    app.interactionSubmitting = true;
    setInteractionButtonsDisabled(true);
    $('#task-interaction-stop-motion').disabled = true;
    if (await triggerStop()) {
      closeTaskInteraction({ suppress: true });
    } else {
      app.interactionSubmitting = false;
      setInteractionButtonsDisabled(false);
      $('#task-interaction-stop-motion').disabled = false;
    }
  });
  window.setInterval(updateInteractionCountdown, 250);
}

async function initialize() {
  bindNavigation();
  bindRosSystem();
  bindConsole();
  bindManual();
  bindConfig();
  bindReadOnly();
  bindLogs();
  bindLaunchLogs();
  bindExit();
  bindImageZoom();
  bindTaskInteraction();
  renderOrder();
  await Promise.all([
    refreshState(),
    loadExecutionConfig(),
    loadUsbOccupancy(true),
  ]);
  startEvents();
}

// 顶层等待确保首屏状态在 load 事件前完成，避免页面先显示错误的“未连接”占位状态。
await initialize();
