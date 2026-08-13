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
  seenOperations: new Set(),
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

  const stopChip = $('[data-status="stop"]');
  stopChip.classList.toggle('locked', Boolean(hardware.stop_latched));
  $('b', stopChip).textContent = hardware.stop_latched ? '已锁定' : '未锁定';

  const indicator = $('#operation-indicator');
  indicator.className = `operation-indicator ${active ? 'running' : 'idle'}`;
  $('span', indicator).textContent = active ? `正在${active.kind}` : '当前无操作';

  $('#pending-restart').textContent = state.config?.pending_restart?.length
    ? state.config.pending_restart.map((scope) => scope === 'hardware' ? '硬件' : '感知').join('、')
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
  // 停止锁期间仍允许只重建硬件服务，以便同步 StopMotion 并解除锁；感知和动作继续禁用。
  $('#hardware-start').disabled = busy || Boolean(process.hardware?.running);
  $('#hardware-stop').disabled = busy || !process.hardware?.owned || ['准备识别', '执行中'].includes(task.state);
  $$('#runtime-mode button').forEach((button) => { button.disabled = busy || stopped || !health.control_services_ready || (process.runtime?.running && button.dataset.mode === runtimeMode); });
  $('#runtime-stop').disabled = busy || !process.runtime?.owned || ['准备识别', '执行中'].includes(task.state);
  $('#task-prepare').disabled = busy || stopped || !health.control_services_ready || !health.perception_services_ready || task.state === '执行中';
  $('#task-confirm').disabled = busy || task.state !== '等待确认' || !task.recognition_valid;
  $('#task-start').disabled = busy || stopped || task.state !== '可以执行' || !task.confirmed;
  $('#clear-stop').disabled = busy || !stopped;

  const manualEnabled = !busy && !stopped && health.control_services_ready && health.camera_frame_fresh && !['准备识别', '执行中'].includes(task.state);
  $$('[data-suction], #servo-send, #servo-quick button, #arm-reset').forEach((button) => { button.disabled = !manualEnabled; });
  $('#pose-refresh').disabled = !health.control_services_ready || !health.camera_frame_fresh;

  const suctionNames = { '-1': '状态未知', 0: '吸气', 1: '喷气', 2: '关闭' };
  $('#suction-status').textContent = suctionNames[hardware.suction_state] ?? '状态未知';
  $('#suction-status').className = `state-badge ${hardware.suction_state === 0 ? 'active' : ''}`;
  $('#servo-status').textContent = hardware.servo_target_known ? `目标 ${Number(hardware.servo_target_angle_deg).toFixed(1)}°` : '角度未知';

  const warning = hardware.emergency_warning || '';
  const overlay = $('#emergency-overlay');
  const stopFailed = warning.includes('无法确认');
  overlay.hidden = !stopFailed;
  if (stopFailed) $('#emergency-message').textContent = warning;

  if (health.last_frame_at) $('#camera-time').textContent = health.last_frame_at.replace('T', ' ').slice(0, 23);
  renderConfigFileStates();
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
  indicator.className = `operation-indicator ${operation.status === 'success' ? 'idle' : 'error'}`;
  $('span', indicator).textContent = operation.status === 'success' ? `${operation.kind}完成` : `${operation.kind}未完成`;
  if (operation.status === 'success') {
    if (operation.result?.apply_error) {
      toast(`${operation.kind}已落盘，但未能自动应用`, operation.result.apply_error, 'warning', 7500);
    } else {
      toast(`${operation.kind}完成`, operation.result?.message || '状态已更新');
    }
    if (['识别与规划'].includes(operation.kind)) refreshDebugImages();
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
  source.addEventListener('health', () => refreshState());
  source.addEventListener('process', () => refreshState());
  source.addEventListener('progress', (event) => {
    if (!app.state) return;
    app.state.task = { ...app.state.task, ...JSON.parse(event.data) };
    renderState(app.state);
  });
  source.addEventListener('operation', (event) => renderOperation(JSON.parse(event.data)));
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

function bindNavigation() {
  $$('.nav-item').forEach((button) => button.addEventListener('click', () => {
    app.selectedView = button.dataset.view;
    $$('.nav-item').forEach((item) => item.classList.toggle('active', item === button));
    $$('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${app.selectedView}`));
    if (app.selectedView === 'config') initializeConfigCenter();
    if (app.selectedView === 'readonly') loadReadOnly();
    if (app.selectedView === 'manual') loadExecutionConfig();
  }));
}

function bindConsole() {
  $('#hardware-start').addEventListener('click', () => command('/api/process/hardware/start', {}, '正在启动硬件'));
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
  $('#task-prepare').addEventListener('click', async () => {
    const advanced = $('input[name="task-level"]:checked').value === 'advanced';
    if (['等待确认', '可以执行'].includes(app.state?.task?.state)) {
      if (!await confirmAction('重新识别？', '当前识别结果将失效，机械臂会重新回到高位拍摄位。')) return;
    }
    command('/api/task/prepare', { advanced, place_order: advanced ? app.order : [] }, '识别任务已提交');
  });
  $('#task-confirm').addEventListener('click', () => command('/api/task/confirm', {}, '识别结果已确认'));
  $('#task-start').addEventListener('click', async () => {
    const count = app.state?.task?.task_count || 0;
    if (await confirmAction('开始执行全部任务？', `即将执行 ${count} 个目标。请确认工作区无人员、障碍物和松动物品。`)) {
      command('/api/task/start', {}, '执行任务已提交');
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
  } catch (_) { /* 错误已经显示。 */ }
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
}

function renderConfigFileStates() {
  const pending = new Set(app.state?.config?.pending_restart || []);
  $$('.config-file-card').forEach((button) => {
    button.classList.toggle('active', button.dataset.fileId === app.currentConfigId);
    const file = app.configFiles.find((item) => item.file_id === button.dataset.fileId);
    button.classList.toggle('pending', Boolean(file && pending.has(file.restart_scope)));
  });
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
    await loadHistory();
  } catch (error) { toast('配置读取失败', formatError(error), 'error'); }
}

async function saveCurrentConfig(apply) {
  if (!app.currentConfigId || !configEditor.document) return;
  const changes = configEditor.changedFields();
  if (!changes.length) return toast('当前没有修改', '', 'warning');
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
    toast(apply ? '保存并应用已提交' : '保存已提交', `操作编号 ${result.operation_id.slice(0, 8)}`);
  } catch (error) {
    if (error.code === 'revision_conflict') {
      toast('文件版本冲突', '配置已被 VS Code 或其它程序修改，请重新加载后再编辑。', 'error', 7000);
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
    if (configEditor.changedFields().length && !await confirmAction('重新加载文件？', '页面上的未保存修改将丢失。')) return;
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
    const result = await api('/api/read-only-config');
    const layout = result.items.find((item) => item.kind === 'layout');
    if (layout?.exists) container.append(renderLayout(layout));
    const grid = document.createElement('div'); grid.className = 'calibration-grid';
    result.items.filter((item) => item.kind !== 'layout').forEach((item) => grid.append(renderCalibration(item)));
    container.append(grid);
  } catch (error) {
    const card = document.createElement('article'); card.className = 'card empty-state'; card.textContent = `读取失败：${formatError(error)}`; container.append(card);
  }
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
  top.append(heading(item.label, `10×14 盘面 · ${item.target_count} 项 · 只读`));
  const revision = document.createElement('span'); revision.className = 'mono'; revision.textContent = item.revision.slice(0, 10); top.append(revision); card.append(top);
  const content = document.createElement('div'); content.className = 'layout-content';
  const board = document.createElement('div'); board.className = 'board-preview';
  const occupied = new Map();
  item.data.targets.forEach((target) => target.cells.forEach(([col, row]) => occupied.set(`${col}:${row}`, target.category)));
  for (let row = 1; row <= 14; row += 1) for (let col = 1; col <= 10; col += 1) {
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

function renderLogs() {
  const level = $('#log-level').value;
  const query = $('#log-search').value.trim().toLowerCase();
  const container = $('#log-list');
  const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
  const filtered = app.logs.filter((entry) => {
    const normalized = ['fatal', 'error'].includes(entry.level) ? 'error' : entry.level;
    const levelMatch = level === 'all' || normalized === level;
    const queryMatch = !query || `${entry.source} ${entry.message}`.toLowerCase().includes(query);
    return levelMatch && queryMatch;
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

async function initialize() {
  bindNavigation();
  bindConsole();
  bindManual();
  bindConfig();
  bindReadOnly();
  bindLogs();
  bindExit();
  renderOrder();
  await Promise.all([refreshState(), loadExecutionConfig()]);
  startEvents();
}

// 顶层等待确保首屏状态在 load 事件前完成，避免页面先显示错误的“未连接”占位状态。
await initialize();
