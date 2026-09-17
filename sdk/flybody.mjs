/** Portable browser / Node client for FLYLAB Body API v1. No dependencies. */
export class BodyAPIError extends Error {
  constructor(status, response) {
    super(`Body API ${status}: ${JSON.stringify(response)}`);
    this.status = status;
    this.response = response;
  }
}
export class FlyBody {
  constructor(baseUrl = 'http://127.0.0.1:8787') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.sessionId = null;
    this.episodeId = null;
    this.stepIndex = 0;
    this.inFlight = false;
  }
  async post(path, payload) {
    if (this.inFlight) throw new Error('Await the previous body request first');
    this.inFlight = true;
    try {
      const headers = {'Content-Type': 'application/json', 'X-Flybrain-Local': '1'};
      // Browsers supply their own Origin; Node needs this local-origin marker.
      if (typeof window === 'undefined') headers.Origin = this.baseUrl;
      const body = payload === undefined ? undefined : JSON.stringify(payload);
      let response;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          response = await fetch(`${this.baseUrl}/api/v1/body${path}`, {
            method: 'POST', headers, body, signal: AbortSignal.timeout(30000),
          });
          break;
        } catch (error) { if (attempt) throw error; }
      }
      const result = await response.json();
      if (!response.ok) throw new BodyAPIError(response.status, result);
      return result;
    } finally { this.inFlight = false; }
  }
  path(command) {
    if (!this.sessionId) throw new Error('Call connect() first');
    return `/sessions/${this.sessionId}/${command}`;
  }
  async connect({controller_name = 'web-game', preset = 'descending', ...config} = {}) {
    const result = await this.post('/sessions', {
      request_id: crypto.randomUUID(), controller_name, preset, ...config,
    });
    this.sessionId = result.session_id;
    this.episodeId = result.episode_id;
    this.stepIndex = result.step_index;
    return result;
  }
  async reset({seed = 0, reset_learning = true} = {}) {
    const result = await this.post(this.path('reset'), {
      request_id: crypto.randomUUID(), seed, reset_learning,
    });
    this.episodeId = result.episode_id;
    this.stepIndex = 0;
    return result;
  }
  async step(observation, {dt_ms = 20} = {}) {
    if (!this.episodeId) throw new Error('Call reset() before step()');
    const result = await this.post(this.path('step'), {
      episode_id: this.episodeId, step_index: this.stepIndex + 1, dt_ms, observation,
    });
    this.stepIndex = result.step_index;
    return result;
  }
  async reward(value, {components = {}, terminated = false, truncated = false} = {}) {
    return this.post(this.path('reward'), {
      episode_id: this.episodeId, step_index: this.stepIndex, value, components,
      terminated, truncated,
    });
  }
  async release() {
    if (!this.sessionId) return;
    const result = await this.post(this.path('release'));
    this.sessionId = this.episodeId = null;
    return result;
  }
}

/** Convert a small RGB(A) render target into the API's normalized luminance frame.
 * Use flipY:true for raw WebGL readPixels (whose origin is bottom-left).
 * Alpha is ignored; sRGB is linearized before luminance conversion.
 */
export function visionFromRGBA(rgba, width, height, {flipY = false} = {}) {
  if (width < 1 || height < 1 || width > 128 || height > 128 ||
      !Number.isInteger(width) || !Number.isInteger(height) || rgba.length !== width * height * 4)
    throw new Error('Expected a 1–128 px RGBA frame on each axis');
  const pixels = new Array(width * height);
  const linear = value => {const v = value / 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4};
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = ((flipY ? height - 1 - y : y) * width + x) * 4;
    pixels[y * width + x] = linear(rgba[i]) * .2126 + linear(rgba[i + 1]) * .7152 + linear(rgba[i + 2]) * .0722;
  }
  return {width, height, pixels};
}
