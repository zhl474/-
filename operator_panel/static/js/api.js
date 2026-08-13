const tokenElement = document.querySelector('meta[name="operator-token"]');
const pageToken = tokenElement?.content || '';

export class ApiError extends Error {
  constructor(message, status, code, payload = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.payload = payload;
  }
}

export async function api(path, options = {}) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = new Headers(options.headers || {});
  const requestOptions = { ...options, method, headers };

  if (!['GET', 'HEAD'].includes(method)) {
    headers.set('X-Operator-Token', pageToken);
  }
  if (options.json !== undefined) {
    headers.set('Content-Type', 'application/json');
    requestOptions.body = JSON.stringify(options.json);
    delete requestOptions.json;
  }
  const response = await fetch(path, requestOptions);
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof payload === 'object' ? payload.error : payload;
    throw new ApiError(message || `请求失败（${response.status}）`, response.status, payload.code, payload);
  }
  return payload;
}

export function eventStream() {
  return new EventSource('/api/events');
}

export function cacheBustedImage(imageId, marker = Date.now()) {
  return `/api/images/${encodeURIComponent(imageId)}?v=${encodeURIComponent(marker)}`;
}
