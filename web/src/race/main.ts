import './style.css';
import './brain-monitor.css';
import {BodyBridge} from './body.ts';
import {BrainMonitor} from './brain-monitor.ts';
import {Policy, type Checkpoint, validateCheckpoint} from './policy.ts';
import {RaceRenderer, type CameraMode} from './renderer.ts';
import {type Action, type Car, type DriverKind, IDLE, mechanical, Race, REWARD_LABELS} from './simulation.ts';
import {clamp, trackAt, TRACK, TRACK_LENGTH} from './track.ts';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
<canvas id="scene-canvas" aria-label="三维赛车赛道与驾驶视野"></canvas>
<header class="header">
  <a class="brand" href="/race/" aria-label="果蝇赛车实验场"><span class="brand-mark">ƒ</span><div><div class="brand-name">FLY <i>CIRCUIT</i></div><small>NEURAL MOTORSPORT LAB</small></div></a>
  <nav class="header-links" aria-label="主导航"><a class="active" href="/race/">赛车实验场</a><a href="/" target="_blank" rel="noopener">神经观察室 ↗</a></nav>
  <div class="system"><span class="local"><i class="dot"></i>LOCAL EXPERIMENT</span><a href="/" target="_blank" rel="noopener">大脑实验室 ↗</a></div>
</header>
<main class="workspace">
  <section class="main-column">
    <div class="page-title"><div><div class="eyebrow">EXPERIMENT 02 / EMBODIED INTELLIGENCE</div><h1>让每一次过弯，成为经验。<span>果蝇赛车实验场</span></h1></div><span class="session-tag">SEASON / 001</span></div>
    <div class="world-frame">
      <div id="world" class="world" role="img" aria-label="3D 赛道。拖动旋转总览，滚轮缩放，点选赛车跟踪视野。"></div>
      <div class="scene-top"><div class="track-title">VERDANT RING<small>翠野环线 · ${(TRACK_LENGTH / 1000).toFixed(2)} KM · 3 LAPS</small></div><span class="live-chip"><i class="dot"></i><span id="race-status">GRID READY</span></span></div>
      <div class="camera-controls" role="group" aria-label="主镜头"><button data-camera="overview" class="active">◈ 赛道总览</button><button data-camera="chase">↗ 跟车镜头</button><button data-camera="cockpit">⊙ 驾驶视角</button></div>
      <div id="reward-pulse" class="reward-pulse"></div>
      <div id="keyboard-hint" class="keyboard-hint hidden"><kbd>W A S D</kbd> 驾驶 &nbsp; <kbd>Shift</kbd> 氮气 &nbsp; <kbd>Space</kbd> 漂移</div>
      <div id="touch-controls" class="touch-controls hidden" aria-label="触屏驾驶"><button data-key="KeyA" aria-label="左转">←</button><button data-key="KeyD" aria-label="右转">→</button><button data-key="KeyW" aria-label="油门">油门</button><button data-key="KeyS" aria-label="刹车">刹车</button><button data-key="ShiftLeft">N₂O</button><button data-key="Space">漂移</button></div>
      <div class="scene-bottom"><div><div class="hero-number"><strong id="speed">000</strong><small>KM/H</small></div><div class="speed-caption" id="speed-caption">果蝇 01 / NEURAL DRIVER</div></div><canvas id="minimap" class="minimap" width="286" height="200" aria-label="赛道位置图"></canvas></div>
      <div id="results" class="results hidden"><div class="result-card"><div class="eyebrow">SESSION COMPLETE</div><h2>冲线之后，继续进化。</h2><p id="result-summary"></p><div id="result-rows"></div><button id="again" class="primary">再来一场 ↗</button></div></div>
    </div>
    <div class="race-bar"><div class="race-actions"><button id="start" class="primary">开始比赛 ↗</button><button id="reset" class="icon-btn" title="重新排列发车" aria-label="重新排列发车">↺</button></div><div class="bar-stats"><div class="stat"><span>RACE TIME</span><strong id="clock">00:00.00</strong></div><div class="stat"><span>LAP</span><strong id="lap">01 / 03</strong></div><div class="stat"><span>NITRO</span><strong id="nitro">100%</strong></div></div><span id="performance" class="fps">WEBGL2 / 就绪</span></div>
    <div class="bottom-grid">
      <section class="panel"><div class="panel-heading"><h2>配置发车阵容</h2><small>RACE SETUP</small></div><div class="setup"><div class="field-row"><span>果蝇赛车手</span><div class="segments" role="group" aria-label="果蝇数量"><button data-count="2">2 只</button><button data-count="3">3 只</button><button data-count="4" class="active">4 只</button></div></div><div class="field-row"><label class="toggle"><input id="mechanical" type="checkbox" checked />加入传统机械赛车手</label><small>规则控制</small></div><div class="field-row"><span>比赛圈数</span><select id="laps" aria-label="比赛圈数"><option value="3">3 圈 · 标准赛</option><option value="1">1 圈 · 短程赛</option><option value="5">5 圈 · 耐力赛</option></select></div><div class="setup-foot"><span>选中右侧赛车手，可改为手动驾驶。</span><span class="mono">01 — 05</span></div></div></section>
      <section class="panel"><div class="panel-heading"><h2>驾驶记忆</h2><small id="training-state">NEURAL POLICY</small></div><div class="training"><div class="training-top"><strong id="generation">—</strong><span>代训练</span><small id="training-score">加载控制网络</small></div><canvas id="chart" class="chart" width="600" height="100" aria-label="真实训练评估奖励曲线"></canvas><div class="training-actions"><button id="train">开始奖励训练 ↗</button><select id="training-length" aria-label="训练轮数"><option value="24">24 代</option><option value="120">120 代</option><option value="0">持续训练</option></select><button id="apply-policy" class="mini-link" disabled>应用到比赛</button></div><p class="training-detail" id="training-detail">独立驾驶网络 · 11 → 18 → 5<br>奖励更新控制网络；MaleCNS 脑内权重保持冻结。</p><div class="memory-actions"><button id="export-policy" class="mini-link">导出记忆</button><button id="import-policy" class="mini-link">导入记忆</button><button id="export-race" class="mini-link">导出比赛记录</button></div><input id="import-file" class="hidden" type="file" accept="application/json,.json" /></div></section>
    </div>
    <footer class="footer"><span>FLY CIRCUIT / 本机运行 · 固定步长物理</span><span>BODY API v1 · <a href="/openapi.json" target="_blank" rel="noopener">接口契约 ↗</a></span></footer>
  </section>
  <aside class="right-column">
    <section id="brain-monitor" class="panel neural-panel" aria-label="实时果蝇大脑"></section>
    <div class="pilot-heading"><h2>从他的眼睛看世界</h2><small>DRIVER'S EYE</small></div>
    <section class="monitor"><div class="monitor-bar"><span id="eye-driver">果蝇 01</span><span><i class="dot"></i>LIVE POV</span></div><div id="eye-view" class="eye-view"><div class="eye-corner"></div><div class="reticle"></div><span class="eye-label">90° FOV / CAR-MOUNTED CAMERA</span></div><div class="sensor-row"><canvas id="retina" class="retina" width="64" height="32" aria-label="摄像机的 64×32 线性亮度输入"></canvas><div class="sensor-copy">64 × 32 · 视觉输入<small id="vision-caption">与驾驶镜头相同的画面采样</small></div></div></section>
    <section class="panel leaderboard"><div class="panel-heading"><h2>场上赛车手</h2><small>SELECT TO FOLLOW</small></div><div id="drivers"></div><div class="driver-controls"><small>选中赛车手的控制方式</small><select id="driver-kind" aria-label="选中赛车手的控制方式"><option value="neural">神经控制网络</option><option value="mechanical">传统机械车手</option><option value="human">我来驾驶</option></select></div></section>
    <section class="panel body-panel"><div class="body-top"><b><i class="dot"></i>MaleCNS · 原生大脑</b><button id="body-connect">原生接管 ↗</button></div><div class="body-observer"><button id="body-observe">开始随车观察</button><small>一个共享大脑 · 完整连接组</small></div><p id="body-detail" class="body-detail">开始比赛后，大脑接收选中赛车的真实画面、氮气、躯体与奖励反馈。可切换为原生大脑直接驾驶。</p><div class="body-metrics"><span id="body-clock">INPUT / VISION + BODY</span><span id="body-learning">WEIGHTS / FROZEN</span></div><a id="body-journal" class="mini-link hidden" target="_blank" rel="noopener">下载原生大脑记录 ↗</a></section>
    <section class="reward-feed"><div class="reward-heading"><span>行为奖励</span><small id="reward-total">Σ +0.00</small></div><ul id="reward-list" class="reward-list"></ul><p id="empty-reward" class="empty-feed">完成超车、有效氮气、漂移和干净过弯，<br>这里会记录每一次进步。</p></section>
  </aside>
</main><div id="toast" class="toast hidden" role="status" aria-live="polite"></div><div id="loading" class="loading"><strong>FLY / CIRCUIT</strong><span>正在准备赛道与驾驶记忆…</span></div>`;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id)! as T;
const dom = {start: $<HTMLButtonElement>('start'), clock: $('clock'), speed: $('speed'), lap: $('lap'), nitro: $('nitro'), drivers: $('drivers'), trainingDetail: $('training-detail')};
let race = new Race(4, true), selectedId = 0, flies = 4, configuredFlies = 4, running = false, ended = false, transitioning = false;
let renderer: RaceRenderer;
let brainMonitor: BrainMonitor | null = null, observationEnabled = true, observerChanging = false;
let checkpoint: Checkpoint | null = null, appliedCheckpoint: Checkpoint | null = null, appliedGeneration = -1;
let policies = new Map<number, Policy>();
const body = new BodyBridge();
let worker: Worker | null = null, trainingTarget = 0, trainedGenerations = 0, workerBase = 0;
let dirty = true, previous = performance.now(), accumulator = 0, lastHud = 0, lastSensor = 0, frameId = 0;
let neuralTickInFlight = false, previewInFlight = false;
let fps = 0, fpsFrames = 0, fpsAt = performance.now();
const keys = new Set<string>();
const replay: unknown[] = [];
const raceExports: unknown[] = [];
let nextRecord = 0;
let toastTimeout = 0;
const clockText = (seconds: number) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;
const hex = (c: Car) => `#${c.color.toString(16).padStart(6, '0')}`;
const selected = () => race.cars.find(c => c.id === selectedId) ?? race.cars[0];
function toast(message: string) {$('toast').textContent = message; $('toast').classList.remove('hidden'); clearTimeout(toastTimeout); toastTimeout = window.setTimeout(() => $('toast').classList.add('hidden'), 6500);}
function escape(text: string) {return text.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]!));}
function driverLabel(c: Car) {return body.mode === 'drive' && body.boundCar === c.id ? 'MaleCNS / 原生冻结' : c.kind === 'mechanical' ? '传统机械 / RULE-BASED' : c.kind === 'human' ? '手动驾驶 / YOU' : `神经策略 / GEN ${appliedGeneration}`;}
function actions(): Action[] {
  return race.cars.map(c => {
    if ((body.mode === 'drive' && body.boundCar === c.id) || c.finished) return {...IDLE};
    if (c.kind === 'mechanical') return mechanical(c, race.cars);
    if (c.kind === 'human') return {steer: (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0) * .7 - (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0) * .7,
      throttle: keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0, brake: keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0,
      nitro: keys.has('ShiftLeft') || keys.has('ShiftRight'), drift: keys.has('Space')};
    return policies.get(c.id)?.act(c, race.cars) ?? {...IDLE};
  });
}
function applyCheckpoint() {
  if (!checkpoint) return;
  policies = new Map(race.cars.map(c => [c.id, new Policy(checkpoint!.weights)]));
  appliedGeneration = checkpoint.generation;
  appliedCheckpoint = structuredClone(checkpoint);
  $<HTMLButtonElement>('apply-policy').disabled = true;
  dirty = true;
}
function saveMemory() {
  if (!checkpoint) return;
  try {localStorage.setItem('fly-circuit-policy-v1', JSON.stringify(checkpoint));}
  catch {toast('浏览器存储不可用，请用「导出记忆」保存训练结果。');}
}
function archiveRace() {
  if (race.time > 0) raceExports.push({time: race.time, laps: race.laps, roster: race.cars.map(c => ({id: c.id, name: c.name, kind: c.kind})), cars: race.cars, events: race.events, frames: [...replay], appliedPolicy: appliedCheckpoint, brainSamples: body.samples, brainJournal: body.journal});
  if (raceExports.length > 12) raceExports.shift();
}
async function resetRace() {
  if (transitioning) return;
  transitioning = true; running = false; keys.clear(); dirty = true; updateHud();
  try {
    while (body.busy || neuralTickInFlight || observerChanging) await new Promise(r => setTimeout(r, 25));
    archiveRace();
    const oldKinds = new Map(race.cars.map(c => [c.id, c.kind]));
    race = new Race(flies, $<HTMLInputElement>('mechanical').checked);
    race.laps = Number($<HTMLSelectElement>('laps').value);
    race.cars.forEach(c => {if (c.id < Math.min(flies, configuredFlies) && oldKinds.has(c.id)) c.kind = oldKinds.get(c.id)!;});
    configuredFlies = flies;
    selectedId = Math.min(selectedId, race.cars.length - 1);
    renderer.syncCars(race.cars); ended = false; accumulator = 0; replay.length = 0; nextRecord = 0;
    if (body.connected) {
      if (body.mode === 'observe' && body.boundCar !== selectedId) await body.follow(selectedId);
      else await body.reset();
    }
    applyCheckpoint(); $('results').classList.add('hidden'); dirty = true;
  } catch (e) {toast(`重新发车失败：${String(e)}`);}
  finally {transitioning = false; updateHud();}
}
function finish() {
  if (ended) return;
  ended = true; running = false; keys.clear();
  const ranked = race.ranked();
  $('result-summary').textContent = `${race.finished ? '比赛完成' : '本场模拟时间已到'} · 用时 ${clockText(race.time)}。本场驾驶策略为第 ${appliedGeneration} 代。`;
  $('result-rows').innerHTML = ranked.map((c, i) => `<div class="result-row"><span>${i + 1}. ${escape(c.name)}</span><span class="mono">${c.finished ? clockText(c.finishTime!) : `${Math.round(c.s / TRACK_LENGTH * 100)}%`} · ${c.reward.toFixed(1)}</span></div>`).join('');
  $('results').classList.remove('hidden'); dirty = true; updateHud();
}
function record() {
  if (race.time < nextRecord) return;
  nextRecord = race.time + .2;
  replay.push({time: race.time, cars: race.cars.map(c => ({id: c.id, s: c.s, offset: c.offset, heading: c.heading, speed: c.speed, action: {...c.action}, reward: c.reward}))});
  if (replay.length > 1000) replay.shift();
}
async function brainTick() {
  if (neuralTickInFlight) return;
  neuralTickInFlight = true;
  try {
    const currentRace = race, owner = body.client.sessionId;
    const car = currentRace.cars.find(c => c.id === body.boundCar)!;
    const frame = await renderer.capture(car, currentRace.cars);
    if (!running || currentRace !== race || owner !== body.client.sessionId || body.closing || body.mode !== 'drive') return;
    await body.step(currentRace, frame, actions());
    record(); dirty = true;
    if (body.receipt?.terminated || body.receipt?.truncated) finish();
  } catch (e) {running = false; toast(`原生大脑连接停止：${String(e)}`); dirty = true;}
  finally {neuralTickInFlight = false;}
}
async function observeTick() {
  try {const currentRace = race; await body.observe(currentRace, car => renderer.capture(car, currentRace.cars));}
  catch (e) {toast(`脑观察暂时停止，赛车继续运行：${String(e)}`);}
  finally {if (!running) updateHud();}
}
function renderMiniMap() {
  const canvas = $<HTMLCanvasElement>('minimap'), ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, 286, 200); ctx.strokeStyle = '#c6d6b57a'; ctx.lineWidth = 11; ctx.lineJoin = 'round'; ctx.beginPath();
  TRACK.forEach((p, i) => {const x = p.x * 1.17 + 143, y = p.z * 1.1 + 100; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);}); ctx.closePath(); ctx.stroke();
  ctx.lineWidth = 1; ctx.strokeStyle = '#142d27'; ctx.stroke();
  race.cars.forEach(c => {const p = trackAt(c.s, c.offset); ctx.beginPath(); ctx.arc(p.x * 1.17 + 143, p.z * 1.1 + 100, c.id === selectedId ? 6 : 4, 0, Math.PI * 2); ctx.fillStyle = hex(c); ctx.fill(); if (c.id === selectedId) {ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();}});
}
async function renderRetina(now: number) {
  if (!renderer.eyeVisible || previewInFlight) return;
  if (lastSensor && now - lastSensor < 200) return;
  lastSensor = now; previewInFlight = true;
  const currentRace = race, id = selectedId;
  try {
    const rgba = await renderer.capture(selected(), currentRace.cars, true);
    if (currentRace !== race || id !== selectedId || document.hidden) return;
    const ctx = $<HTMLCanvasElement>('retina').getContext('2d')!, frame = ctx.createImageData(64, 32);
    const linear = (v: number) => {v /= 255; return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;};
    for (let y = 0; y < 32; y++) for (let x = 0; x < 64; x++) {
      const from = ((31 - y) * 64 + x) * 4, to = (y * 64 + x) * 4;
      const light = (linear(rgba[from]) * .2126 + linear(rgba[from + 1]) * .7152 + linear(rgba[from + 2]) * .0722) * 255;
      frame.data[to] = frame.data[to + 1] = frame.data[to + 2] = light; frame.data[to + 3] = 255;
    }
    ctx.putImageData(frame, 0, 0);
  } catch (e) {console.error('Camera preview readback failed', e);}
  finally {previewInFlight = false;}
}
function drawChart() {
  if (!checkpoint) return;
  const canvas = $<HTMLCanvasElement>('chart'), ctx = canvas.getContext('2d')!;
  const history = checkpoint.history.slice(-80); ctx.clearRect(0, 0, 600, 100);
  ctx.strokeStyle = '#37472e'; ctx.lineWidth = 1;
  for (let y = 20; y <= 90; y += 30) {ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(600, y); ctx.stroke();}
  if (!history.length) return;
  const min = Math.min(...history.map(h => h.reward)) - 8, max = Math.max(...history.map(h => h.reward)) + 8;
  const points = history.map((h, i) => [i / Math.max(1, history.length - 1) * 598 + 1, 92 - (h.reward - min) / Math.max(1, max - min) * 84]);
  ctx.beginPath(); points.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
  ctx.strokeStyle = '#e6fc88'; ctx.lineWidth = 2.5; ctx.stroke();
  const last = points[points.length - 1]; ctx.beginPath(); ctx.arc(last[0] - 2, last[1], 4, 0, Math.PI * 2); ctx.fillStyle = '#e6fc88'; ctx.fill();
}
function updateTraining() {
  $('generation').textContent = checkpoint ? String(checkpoint.generation).padStart(3, '0') : '—';
  $('training-score').textContent = checkpoint?.evaluation ? `评估奖励 ${checkpoint.evaluation.reward.toFixed(1)}` : '尚无评估';
  $('training-state').textContent = worker ? `TRAINING / ${trainedGenerations}${trainingTarget ? ` OF ${trainingTarget}` : ''}` : 'MEMORY SAVED';
  $<HTMLButtonElement>('apply-policy').disabled = !checkpoint || appliedGeneration === checkpoint.generation;
  drawChart();
}
function updateHud() {
  const c = selected();
  dom.speed.textContent = String(Math.round(c.speed * 3.6)).padStart(3, '0');
  dom.clock.textContent = clockText(race.time);
  dom.lap.textContent = `${String(Math.min(race.laps, c.lap + 1)).padStart(2, '0')} / ${String(race.laps).padStart(2, '0')}`;
  dom.nitro.textContent = `${Math.round(c.nitro)}%`;
  dom.start.textContent = transitioning ? '正在准备…' : ended ? '再来一场 ↗' : running ? 'Ⅱ 暂停比赛' : race.time > 0 ? '继续比赛 ↗' : '开始比赛 ↗';
  dom.start.disabled = transitioning || !checkpoint || !!(body.connected && body.mode === 'drive' && body.error);
  $('race-status').textContent = running ? body.connected ? body.mode === 'observe' ? 'LIVE + BRAIN' : 'BRAIN LOCKSTEP' : 'RACE LIVE' : ended ? 'RACE COMPLETE' : race.time > 0 ? 'PAUSED' : 'GRID READY';
  $('speed-caption').textContent = `${c.name} / ${driverLabel(c)}`;
  $('eye-driver').textContent = c.name;
  $('vision-caption').textContent = body.boundCar === c.id ? body.mode === 'observe' ? '从此镜头异步采样，赛车无需等待' : '此镜头正输入原生果蝇大脑' : body.connected ? `当前观察镜头 · 接口绑定果蝇 ${body.boundCar! + 1}` : '与驾驶镜头相同的画面采样';
  // Keep actual button nodes stable so focus and pointer actions survive live ranking updates.
  const ranked = race.ranked();
  for (let i = 0; i < ranked.length; i++) {
    const car = ranked[i]; let row = dom.drivers.querySelector<HTMLButtonElement>(`[data-driver="${car.id}"]`);
    if (!row) {row = document.createElement('button'); row.className = 'driver-row'; row.dataset.driver = String(car.id); row.innerHTML = '<span class="driver-rank"></span><span class="car-swatch"></span><span class="driver-info"><b></b><small></small></span><span class="driver-score"><span></span><small></small></span>'; dom.drivers.append(row);}
    row.classList.toggle('selected', car.id === selectedId); row.setAttribute('aria-pressed', String(car.id === selectedId)); row.setAttribute('aria-label', `跟踪${car.name}`);
    row.querySelector('.driver-rank')!.textContent = String(i + 1).padStart(2, '0');
    (row.querySelector('.car-swatch')! as HTMLElement).style.backgroundColor = hex(car);
    row.querySelector('.driver-info b')!.textContent = car.name;
    row.querySelector('.driver-info small')!.textContent = driverLabel(car);
    row.querySelector('.driver-score span')!.textContent = `${car.reward >= 0 ? '+' : ''}${car.reward.toFixed(1)}`;
    row.querySelector('.driver-score small')!.textContent = car.finished ? 'FINISHED' : `${Math.round(car.speed * 3.6)} KM/H`;
    if (dom.drivers.children[i] !== row) dom.drivers.insertBefore(row, dom.drivers.children[i] ?? null);
  }
  dom.drivers.querySelectorAll<HTMLButtonElement>('[data-driver]').forEach(row => {if (!race.cars.some(c => String(c.id) === row.dataset.driver)) row.remove();});
  const nativeDriver = body.connected && body.mode === 'drive';
  const kind = $<HTMLSelectElement>('driver-kind'); kind.value = c.kind; kind.disabled = (nativeDriver && body.boundCar === c.id) || transitioning;
  $('keyboard-hint').classList.toggle('hidden', c.kind !== 'human' || (nativeDriver && body.boundCar === c.id));
  $('touch-controls').classList.toggle('hidden', c.kind !== 'human' || (nativeDriver && body.boundCar === c.id));
  $('performance').textContent = running ? `${fps.toFixed(0)} FPS / ${renderer?.calls ?? 0} DRAWS` : 'WEBGL2 / 按需绘制';
  $('reward-total').textContent = `Σ ${c.reward >= 0 ? '+' : ''}${c.reward.toFixed(2)}`;
  const events = race.events.filter(e => e.carId === c.id).slice(-5).reverse();
  $('empty-reward').classList.toggle('hidden', events.length > 0);
  $('reward-list').innerHTML = events.map(e => `<li class="${e.value < 0 ? 'negative' : ''}"><small class="mono">${clockText(e.time).slice(0, 5)}</small><span>${REWARD_LABELS[e.type]}</span><b>${e.value >= 0 ? '+' : ''}${e.value.toFixed(1)}</b></li>`).join('');
  const recent = events[0];
  $('reward-pulse').innerHTML = recent && race.time - recent.time < 1.8 && recent.value > 0 ? `+${recent.value.toFixed(1)}<small>${REWARD_LABELS[recent.type]}</small>` : '';
  const connect = $<HTMLButtonElement>('body-connect');
  connect.textContent = body.connecting ? '正在接入…' : body.closing ? '正在释放…' : nativeDriver ? '释放原生大脑' : '原生驾驶实验 ↗';
  connect.title = '原生驾驶实验严格等待真实大脑计算，比赛速度由神经计算决定';
  connect.disabled = body.connecting || body.closing || transitioning || observerChanging;
  if (body.connected) {
    $('body-detail').textContent = body.error ? `观察连接已停止：${body.error}` : body.mode === 'observe' ? `正在异步观察 ${race.cars.find(c => c.id === body.boundCar)?.name}。赛车实时运行，大脑按自身速度接收最新画面；期间的动作和奖励会累计保留。` : `原生驾驶实验：${race.cars.find(c => c.id === body.boundCar)?.name} 严格等待大脑读出，以真实神经时间推进比赛。`;
    $('body-clock').textContent = `STEP ${body.last?.step_index ?? 0} / ${body.last?.real_time_factor.toFixed(2) ?? '—'}×`;
    $('body-learning').textContent = `WEIGHT UPDATES / ${body.receipt?.learning.weight_updates ?? 0}`;
  } else {
    $('body-detail').textContent = body.error ? `连接失败：${body.error}` : observationEnabled ? '开始比赛后自动随车观察。原生大脑接收选中赛车的感觉输入，驾驶方式保持不变。' : '随车观察已停止，赛车可独立运行；点击「开始随车观察」重新连接。';
    $('body-clock').textContent = 'INPUT / VISION + BODY'; $('body-learning').textContent = 'WEIGHTS / FROZEN';
  }
  const observer = $<HTMLButtonElement>('body-observe');
  observer.textContent = body.connected && body.mode === 'observe' ? '停止随车观察' : '开始随车观察';
  observer.disabled = transitioning || body.connecting || body.closing || observerChanging;
  if (body.journal) {const a = $<HTMLAnchorElement>('body-journal'); a.href = body.journal; a.classList.remove('hidden');}
  document.querySelectorAll<HTMLButtonElement>('[data-count]').forEach(b => {b.disabled = transitioning || nativeDriver; b.classList.toggle('active', Number(b.dataset.count) === flies);});
  $<HTMLInputElement>('mechanical').disabled = transitioning || nativeDriver;
  $<HTMLSelectElement>('laps').disabled = transitioning || nativeDriver;
  $<HTMLButtonElement>('reset').disabled = transitioning || body.connecting || body.closing;
  renderMiniMap();
  brainMonitor?.sync();
}
function frame(now: number) {
  frameId = requestAnimationFrame(frame);
  if (document.hidden || !renderer) return;
  const dt = Math.min((now - previous) / 1000, .05); previous = now;
  if (running && !transitioning) {
    if (body.connected && body.mode === 'drive') {if (!body.busy && !body.closing) void brainTick();}
    else {
      accumulator += dt;
      while (accumulator >= 1 / 60) {race.step(1 / 60, actions()); accumulator -= 1 / 60;}
      record(); if (race.finished || race.time >= 180) finish();
    }
    dirty = true;
  }
  // Sampling and native requests must never gate ordinary game physics or drawing.
  if (body.connected && body.mode === 'observe') {
    body.trackWorld(race);
    if (!transitioning && !observerChanging && (running || ended) && body.observationDue) void observeTick();
  }
  const draw = dirty || running;
  if (draw) {
    renderer.render(race, selectedId, dt); void renderRetina(now); dirty = false;
    fpsFrames++;
    if (now - fpsAt > 1000) {fps = fpsFrames * 1000 / (now - fpsAt); fpsFrames = 0; fpsAt = now;}
  }
  if (now - lastHud > 120 && (draw || transitioning || observerChanging || body.connecting || body.closing)) {updateHud(); lastHud = now;}
}

function download(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function stopTraining(message?: string) {
  worker?.terminate(); worker = null;
  $('train').textContent = '开始奖励训练 ↗'; $<HTMLSelectElement>('training-length').disabled = false;
  saveMemory(); updateTraining(); if (message) toast(message);
}
function startTraining() {
  if (worker) {stopTraining('训练已暂停，已完成的代数与驾驶记忆已保存。'); return;}
  if (!checkpoint) return;
  trainingTarget = Number($<HTMLSelectElement>('training-length').value); trainedGenerations = 0; workerBase = checkpoint.generation;
  worker = new Worker(new URL('./training.worker.ts', import.meta.url), {type: 'module'});
  $('train').textContent = 'Ⅱ 暂停训练'; $<HTMLSelectElement>('training-length').disabled = true;
  worker.onmessage = event => {
    if (event.data.type === 'generation') {
      checkpoint = validateCheckpoint(event.data.checkpoint); trainedGenerations = checkpoint.generation - workerBase;
      saveMemory(); updateTraining();
      if (trainingTarget > 0 && trainedGenerations >= trainingTarget) stopTraining(`已完成 ${trainedGenerations} 代奖励训练。点击「应用到比赛」使用新记忆。`);
    } else if (event.data.type === 'complete' && worker) worker.postMessage({checkpoint});
    else if (event.data.type === 'error') stopTraining(`训练停止：${event.data.message}`);
  };
  worker.onerror = event => stopTraining(`训练线程停止：${event.message}`);
  worker.postMessage({checkpoint}); updateTraining();
}

async function changeObservation(enabled: boolean) {
  if (transitioning || observerChanging) return;
  observerChanging = true; observationEnabled = enabled; updateHud();
  try {
    if (body.connected && (body.mode === 'observe' || enabled)) await body.release();
    if (enabled) {
      await body.recoverOwnSession(); await body.connect(selectedId, 'observe');
      while (body.connected && body.boundCar !== selectedId && !document.hidden && observationEnabled) await body.follow(selectedId);
    }
  } catch (error) {toast(`神经观察连接失败：${String(error)}`);}
  finally {
    observerChanging = false;
    if ((document.hidden || !observationEnabled) && body.connected && body.mode === 'observe') await body.release();
    updateHud();
  }
}
dom.start.addEventListener('click', async () => {
  if (transitioning) return;
  if (ended) await resetRace();
  if (transitioning || !checkpoint) return;
  running = !running;
  if (running && !body.connected && observationEnabled && brainMonitor?.shown) void changeObservation(true);
  accumulator = 0; previous = performance.now(); fpsAt = previous; fpsFrames = 0; dirty = true; updateHud();
});
$('reset').addEventListener('click', () => void resetRace());
$('again').addEventListener('click', () => dom.start.click());
document.querySelectorAll<HTMLButtonElement>('[data-count]').forEach(b => b.addEventListener('click', () => {flies = Number(b.dataset.count); void resetRace();}));
$('mechanical').addEventListener('change', () => void resetRace()); $('laps').addEventListener('change', () => void resetRace());
document.querySelectorAll<HTMLButtonElement>('[data-camera]').forEach(b => b.addEventListener('click', () => {
  renderer.setMode(b.dataset.camera as CameraMode); document.querySelectorAll('[data-camera]').forEach(o => o.classList.toggle('active', o === b)); dirty = true;
}));
async function selectDriver(id: number) {
  if (transitioning) return;
  selectedId = id; keys.clear(); dirty = true;
  if (body.connected && body.mode === 'observe' && body.boundCar !== id && !observerChanging) {
    observerChanging = true; updateHud();
    try {
      while (body.connected && body.boundCar !== selectedId && !document.hidden) await body.follow(selectedId);
    }
    catch (error) {toast(`切换脑观察失败：${String(error)}`);}
    finally {observerChanging = false; if (document.hidden && body.connected) await body.release();}
  }
  updateHud();
}
dom.drivers.addEventListener('click', event => {const b = (event.target as HTMLElement).closest<HTMLButtonElement>('[data-driver]'); if (b) selectDriver(Number(b.dataset.driver));});
$('driver-kind').addEventListener('change', event => {
  const kind = (event.target as HTMLSelectElement).value as DriverKind;
  const c = selected(); c.kind = kind;
  if (kind === 'human') {race.cars.filter(b => b.id !== c.id && b.kind === 'human').forEach(b => b.kind = 'neural'); renderer.setMode('chase'); document.querySelectorAll<HTMLButtonElement>('[data-camera]').forEach(b => b.classList.toggle('active', b.dataset.camera === 'chase'));}
  keys.clear(); dirty = true; updateHud();
});
$('body-connect').addEventListener('click', async () => {
  if (transitioning || observerChanging) return;
  running = false; keys.clear(); dirty = true;
  try {
    if (body.connected && body.mode === 'drive') {await body.release(); observationEnabled = false; toast('原生大脑已释放；赛车恢复所选控制方式。');}
    else {if (body.client.sessionId) await body.release(); await resetRace(); await body.recoverOwnSession(); await body.connect(selectedId, 'drive'); toast('原生大脑已接管，右侧显示其真实神经活动。');}
  } catch (e) {toast(`Body API：${String(e)}`);}
  dirty = true; updateHud();
});
$('body-observe').addEventListener('click', () => {
  const enable = !(body.connected && body.mode === 'observe');
  if (enable && brainMonitor && !brainMonitor.shown) brainMonitor.setShown(true);
  if (!observerChanging) void changeObservation(enable);
});
$('train').addEventListener('click', startTraining);
$('apply-policy').addEventListener('click', async () => {
  if (body.connected && body.mode === 'drive') {toast('请先释放原生驾驶控制，再应用控制网络的新记忆。'); return;}
  await resetRace(); toast(`已装载第 ${appliedGeneration} 代记忆，准备新一场比赛。`);
});
$('export-policy').addEventListener('click', () => {if (checkpoint) download(`fly-circuit-memory-gen-${checkpoint.generation}.json`, checkpoint);});
$('export-race').addEventListener('click', () => download('fly-circuit-race.json', {version: 1, contract: 'docs/body-api.md', clock: 'simulation-seconds', track: 'verdant-ring-v1', seed: 71, appliedPolicy: appliedCheckpoint, latestPolicy: checkpoint, appliedGeneration, archives: raceExports, current: {time: race.time, laps: race.laps, cars: race.cars, events: race.events, frames: replay}, body: {mode: body.mode, journal: body.journal, samples: body.samples, lastStep: body.last, lastReceipt: body.receipt}}));
$('import-policy').addEventListener('click', () => $<HTMLInputElement>('import-file').click());
$('import-file').addEventListener('change', async () => {
  const input = $<HTMLInputElement>('import-file'), file = input.files?.[0]; if (!file) return;
  try {if (file.size > 2_000_000) throw new Error('训练文件过大'); const imported = validateCheckpoint(JSON.parse(await file.text())); if (worker) stopTraining(); checkpoint = imported; saveMemory(); updateTraining(); $<HTMLButtonElement>('apply-policy').disabled = false; toast('驾驶记忆已导入，点击「应用到比赛」载入。');}
  catch (e) {toast(`导入失败：${String(e)}`);} finally {input.value = '';}
});
const controlKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowLeft', 'ArrowDown', 'ArrowRight', 'ShiftLeft', 'ShiftRight', 'Space']);
document.addEventListener('keydown', e => {if ((e.target as HTMLElement).matches('input,select,textarea')) return; if (controlKeys.has(e.code) && race.cars.some(c => c.kind === 'human')) {e.preventDefault(); keys.add(e.code);}});
document.addEventListener('keyup', e => keys.delete(e.code)); window.addEventListener('blur', () => keys.clear());
document.querySelectorAll<HTMLButtonElement>('[data-key]').forEach(b => {
  b.addEventListener('pointerdown', e => {e.preventDefault(); b.setPointerCapture(e.pointerId); keys.add(b.dataset.key!);});
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(type, () => keys.delete(b.dataset.key!));
});
let pointer: {x: number; y: number; downX: number; downY: number} | null = null;
$('world').addEventListener('pointerdown', e => {const p = e as PointerEvent; pointer = {x: p.clientX, y: p.clientY, downX: p.clientX, downY: p.clientY}; $('world').setPointerCapture(p.pointerId);});
$('world').addEventListener('pointermove', e => {if (!pointer || renderer.mode !== 'overview') return; const p = e as PointerEvent; renderer.azimuth -= (p.clientX - pointer.x) * .005; renderer.elevation = clamp(renderer.elevation + (p.clientY - pointer.y) * .004, .35, 1.3); pointer.x = p.clientX; pointer.y = p.clientY; dirty = true;});
$('world').addEventListener('pointerup', e => {const p = e as PointerEvent; if (pointer && Math.hypot(p.clientX - pointer.downX, p.clientY - pointer.downY) < 5) {const id = renderer.pick(p.clientX, p.clientY); if (id !== null) selectDriver(id);} pointer = null;});
$('world').addEventListener('pointercancel', () => pointer = null);
$('world').addEventListener('wheel', e => {if (renderer.mode !== 'overview') return; const w = e as WheelEvent; w.preventDefault(); renderer.distance = clamp(renderer.distance + w.deltaY * .15, 90, 340); dirty = true;}, {passive: false});
window.addEventListener('resize', () => dirty = true); window.addEventListener('scroll', () => dirty = true, {passive: true, capture: true});
document.addEventListener('visibilitychange', () => {
  keys.clear(); previous = performance.now(); dirty = true;
  if (document.hidden) {
    running = false; cancelAnimationFrame(frameId);
    if (worker) stopTraining();
    if (body.connected) void body.release().catch(e => body.error = String(e)).finally(() => {dirty = true;});
  } else {frameId = requestAnimationFrame(frame); updateHud();}
});
window.addEventListener('pagehide', event => {body.releaseOnExit(); if (!event.persisted) brainMonitor?.dispose();});
$('scene-canvas').addEventListener('webglcontextlost', e => {e.preventDefault(); running = false; toast('图形上下文已暂停。恢复后请继续比赛。'); dirty = true;});
$('scene-canvas').addEventListener('webglcontextrestored', () => {renderer.resize(); dirty = true;});

async function init() {
  try {
    renderer = new RaceRenderer($<HTMLCanvasElement>('scene-canvas'), $('world'), $('eye-view'), () => {dirty = true;}); renderer.syncCars(race.cars);
    try {const saved = localStorage.getItem('fly-circuit-policy-v1'); if (saved) checkpoint = validateCheckpoint(JSON.parse(saved));} catch {toast('已保存记忆不可用，将载入随项目提供的基础策略。');}
    if (!checkpoint) {const response = await fetch('/race/baseline.json'); if (!response.ok) throw new Error('基础驾驶记忆加载失败，请重新构建网页'); checkpoint = validateCheckpoint(await response.json());}
    applyCheckpoint(); updateTraining(); updateHud();
    brainMonitor = new BrainMonitor($('brain-monitor'), () => ({body, name: race.cars.find(c => c.id === body.boundCar)?.name ?? selected().name, running, enabled: observationEnabled}), shown => {
      observationEnabled = shown;
      if ((!shown && body.connected && body.mode === 'observe') || (shown && running && !body.connected)) void changeObservation(shown);
    });
    $('loading').remove(); frameId = requestAnimationFrame(frame);
    // Read-only inspection surface used by local acceptance checks.
    Object.defineProperty(window, '__FLY_CIRCUIT__', {value: {sampleLog: () => structuredClone(body.samples), snapshot: () => ({time: race.time, running, selectedId, camera: renderer.mode, flies, cars: race.cars.map(c => ({id: c.id, kind: c.kind, s: c.s, speed: c.speed, offset: c.offset, reward: c.reward, lap: c.lap, components: {...c.components}})), generation: checkpoint?.generation, appliedGeneration, training: !!worker, rendering: {calls: renderer.calls, triangles: renderer.triangles, fps, frames: renderer.frames, readbacks: renderer.readbacks, lastReadbackMs: renderer.lastReadbackMs}, brain: brainMonitor?.snapshot(), body: {connected: body.connected, busy: body.busy, boundCar: body.boundCar, mode: body.mode, input: body.lastInput, step: body.last, receipt: body.receipt, error: body.error, journal: body.journal, sampledWorldTime: body.sampledWorldTime, sampleLatencyMs: body.sampleLatencyMs, observationAgeMs: body.observationAgeMs, samples: body.samples.length, feedbackThroughWorldTime: body.feedbackThroughWorldTime}, world: $('world').getBoundingClientRect().toJSON(), eye: $('eye-view').getBoundingClientRect().toJSON()})}});
  } catch (e) {$('loading').remove(); $('world').innerHTML = `<div class="error-screen"><strong>暂时无法进入赛道</strong><p>${escape(String(e))}</p><p>请使用支持 WebGL2 的 Chrome，并保持本地服务运行。</p></div>`; dom.start.disabled = true;}
}
void init();
window.setInterval(() => {if (!document.hidden) brainMonitor?.sync();}, 1000);
