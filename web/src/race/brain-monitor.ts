import {BrainView} from '../brain.ts';
import {decodeNeuralPacket} from '../neural-packet.ts';
import type {Meta, Neuron, State} from '../types.ts';
import type {BodyBridge} from './body.ts';

const number = (v: number) => v.toLocaleString('en-US');
export interface BrainSource {body: BodyBridge; name: string; running: boolean; enabled: boolean}

/** A standalone observer: owns its canvas/subscription, never advances the brain. */
export class BrainMonitor {
  view: BrainView | null = null;
  shown = true;
  inViewport = true;
  streamSequence = -1;
  windowMs = 0;
  simSeconds = 0;
  active = 0;
  spikes = 0;
  packets = 0;
  packetBytes = 0;
  error: string | null = null;
  private meta: Meta | null = null;
  private state: State | null = null;
  private socket: WebSocket | null = null;
  private owner: string | null = null;
  private selected: Neuron | null = null;
  private selectionSerial = 0;
  private reconnectTimer = 0;
  private metricsAt = 0;
  private disposed = false;
  private intersection: IntersectionObserver;

  constructor(private root: HTMLElement, private source: () => BrainSource, private onVisibility: (shown: boolean) => void) {
    root.innerHTML = `
      <div class="neural-panel-title"><div><span class="eyebrow">LIVE NEURAL ACTIVITY</span><h2>果蝇神经观察</h2></div><div class="neural-title-actions"><button data-n="expand" title="放大神经观察" aria-label="放大神经观察">⛶</button><button data-n="toggle" aria-expanded="true">收起</button></div></div>
      <div data-n="content">
        <div class="neural-source"><i class="dot"></i><span data-n="source">等待比赛开始 · 单个 MaleCNS</span><b data-n="connection">待连接</b></div>
        <div class="neural-viewport" data-n="viewport"><div class="neural-loading" data-n="loading">正在读取完整神经元坐标…</div><span class="neural-canvas-caption">MALECNS v1.0 / 166,700 NEURONS</span></div>
        <div class="neural-toolbar" role="group" aria-label="果蝇脑镜头"><button data-view="all" class="active">全脑</button><button data-view="brain">头部</button><button data-view="vnc">躯干部</button><button data-n="active" aria-pressed="false">只看激活</button><button data-n="anatomy" aria-pressed="true">脑区外形</button></div>
        <div class="neural-stats"><div><strong data-n="active-count">0</strong><small>本窗口激活神经元</small></div><div><strong data-n="spikes">0</strong><small>真实脉冲数</small></div><div><strong data-n="clock">0.0 s</strong><small data-n="window">等待感觉输入</small></div></div>
        <div class="neural-inputs" aria-label="实际神经输入"><span data-input="vision">视觉</span><span data-input="nitro">氮气</span><span data-input="body">躯体</span><span data-input="reward">奖励反馈</span><small data-n="step">STEP 0</small></div>
        <div class="neural-sample"><span data-n="sample-time">等待首个真实采样</span><span data-n="sample-age"></span></div>
        <div class="neural-regions" data-n="regions" aria-label="脑区颜色与激活数量"></div>
        <div class="neural-cell" data-n="cell"><span data-n="cell-detail">拖动旋转 · 滚轮缩放 · 点选神经元</span><div class="neural-cell-actions"><button data-n="skeleton" disabled>查看完整骨架</button><button data-n="focus" disabled>聚焦</button><button data-n="clear" disabled>取消选择</button></div></div>
        <div class="neural-foot"><span data-n="anatomy-status">保留全部神经元与官方脑区</span><span data-n="metrics">按需绘制</span></div>
        <div class="neural-note" data-n="note">比赛会自动连接随车观察；驾驶方式保持不变。</div>
      </div>`;
    this.el('toggle').addEventListener('click', () => this.setShown(!this.shown));
    this.el('expand').addEventListener('click', () => {
      if (document.fullscreenElement === root) void document.exitFullscreen();
      else void root.requestFullscreen().catch(() => {this.el('note').textContent = '当前窗口不支持全屏，可通过滚轮放大脑视图。';});
    });
    root.querySelectorAll<HTMLButtonElement>('[data-view]').forEach(button => button.addEventListener('click', () => {
      this.view?.setView(button.dataset.view!);
      root.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b === button));
    }));
    this.el('active').addEventListener('click', () => {
      if (!this.view) return; this.view.setOnlyActive(!this.view.onlyActive);
      this.el('active').classList.toggle('active', this.view.onlyActive); this.el('active').setAttribute('aria-pressed', String(this.view.onlyActive));
    });
    this.el('anatomy').addEventListener('click', () => {
      if (!this.view) return; this.view.setAnatomy(!this.view.anatomyVisible);
      this.el('anatomy').setAttribute('aria-pressed', String(this.view.anatomyVisible)); this.el('anatomy').classList.toggle('off', !this.view.anatomyVisible);
    });
    this.el('skeleton').addEventListener('click', () => void this.loadSkeleton());
    this.el('focus').addEventListener('click', () => this.view?.focusSelection());
    this.el('clear').addEventListener('click', () => this.clearSelection());
    this.intersection = new IntersectionObserver(entries => {
      this.inViewport = entries[0].isIntersecting;
      this.view?.setVisible(this.shown && this.inViewport && !document.hidden);
      this.sync();
    });
    this.intersection.observe(this.el('viewport'));
    document.addEventListener('visibilitychange', this.onDocumentVisibility);
    void this.load();
  }
  private el(name: string) {return this.root.querySelector<HTMLElement>(`[data-n="${name}"]`)!;}
  private onDocumentVisibility = () => this.sync();
  setShown(value: boolean) {
    this.shown = value;
    this.el('content').hidden = !value;
    this.el('toggle').textContent = value ? '收起' : '显示大脑';
    this.el('toggle').setAttribute('aria-expanded', String(value));
    this.view?.setVisible(value && this.inViewport && !document.hidden);
    this.sync(); this.onVisibility(value);
  }
  private async load() {
    try {
      const response = await fetch('/data/meta.json');
      if (!response.ok) throw new Error('神经数据未就绪');
      this.meta = await response.json() as Meta;
      if (this.meta.neurons !== 166700 || this.meta.regions.length !== 5) throw new Error('神经数据与固定模型不一致');
      this.view = new BrainView(this.el('viewport'), this.meta, index => void this.selectCell(index), {pixelRatio: 1.25, scaleBar: null});
      this.view.setVisible(this.shown && this.inViewport && !document.hidden);
      await this.view.loadNodes();
      if (this.disposed) {this.view.dispose(); return;}
      this.el('loading').hidden = true;
      for (const region of this.meta.regions) {
        const button = document.createElement('button'); button.className = 'neural-region'; button.dataset.region = String(region.id);
        button.setAttribute('aria-label', `切换${region.name}显示`); button.setAttribute('aria-pressed', 'true');
        const dot = document.createElement('i'); dot.style.backgroundColor = region.color;
        const label = document.createElement('span'); label.textContent = region.name;
        const count = document.createElement('b'); count.textContent = '0';
        button.append(dot, label, count); button.addEventListener('click', () => {
          const on = button.getAttribute('aria-pressed') !== 'true'; button.setAttribute('aria-pressed', String(on)); button.classList.toggle('off', !on); this.view?.setRegion(region.id, on);
        }); this.el('regions').append(button);
      }
      this.sync();
      await this.view.loadAnatomy((done, total) => {this.el('anatomy-status').textContent = `官方脑区 ${done} / ${total}`;});
      this.el('anatomy-status').textContent = '114 脑区 · 3 批网格';
    } catch (error) {this.error = String(error); this.el('loading').textContent = this.error; this.el('note').textContent = this.error;}
  }
  sync() {
    if (this.disposed) return;
    const {body, name, running, enabled} = this.source();
    const visible = this.shown && this.inViewport && !document.hidden;
    this.view?.setVisible(visible);
    const owner = body.connected ? body.client.sessionId : null;
    if (owner !== this.owner) {
      this.closeSocket(); this.owner = owner; this.state = null; this.streamSequence = -1;
      this.active = this.spikes = this.simSeconds = this.windowMs = 0;
      this.view?.updateSparse(new Uint32Array(), new Uint16Array()); this.clearSelection();
      this.updateNumbers();
      this.root.querySelectorAll('.neural-region b').forEach(n => n.textContent = '0');
    }
    if (visible && owner && this.view?.points.visible) this.connect(); else this.closeSocket();
    this.el('source').textContent = owner ? `${name} · ${body.mode === 'observe' ? '随车观察' : '原生驾驶'}` : enabled ? '单个 MaleCNS · 等待比赛开始' : '单个 MaleCNS · 随车观察已停止';
    this.el('connection').textContent = this.error || body.error ? '连接异常' : !owner ? '未连接' : !visible ? '已挂起' : this.socket?.readyState === WebSocket.OPEN ? body.finished ? '已结束' : running ? body.mode === 'observe' ? '异步观察' : 'LIVE' : '暂停' : '连接中';
    const matched = !!owner && body.last?.stream_sequence === this.streamSequence;
    const input = matched ? body.lastInput : null;
    const flags: Record<string, boolean> = {vision: !!input?.vision, nitro: (input?.sensors.nitro ?? 0) > 0,
      body: (input?.sensors.body_left ?? 0) > 0 || (input?.sensors.body_right ?? 0) > 0 || (input?.sensors.touch ?? 0) > 0,
      reward: !!input?.sugar};
    this.root.querySelectorAll<HTMLElement>('[data-input]').forEach(el => el.classList.toggle('stimulated', flags[el.dataset.input!]));
    this.el('step').textContent = `STEP ${body.last?.step_index ?? 0}`;
    const sampled = matched && body.mode === 'observe' && body.sampledWorldTime !== null;
    this.el('sample-time').textContent = sampled ? `采样赛时 ${body.sampledWorldTime!.toFixed(2)} s` : owner && body.mode === 'drive' ? '原生驾驶 · 神经与世界锁步' : '等待首个真实采样';
    this.el('sample-age').textContent = sampled ? running ? `数据年龄 ${Math.round(body.observationAgeMs ?? 0)} ms` : `计算 ${Math.round(body.sampleLatencyMs)} ms` : '';
    this.el('note').textContent = this.error ?? body.error ?? (owner ? body.mode === 'observe' ? '赛车实时运行；此处显示最近一次真实神经结果，两个时钟独立。' : '原生驾驶按神经计算速度推进；亮点均为真实放电。' : enabled ? '开始比赛后异步观察，赛车不会等待神经计算。' : '随车观察已停止，可在下方重新开启。');
    const now = performance.now();
    if (this.view && now - this.metricsAt > 1000) {
      this.metricsAt = now; const metrics = this.view.readMetrics();
      this.el('metrics').textContent = `${metrics.draw_calls} DRAW / ${metrics.fps < .5 ? '按需' : `${Math.round(metrics.fps)} FPS`}`;
      if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({type: 'view_metrics', ...metrics}));
    }
  }
  private connect() {
    if (this.socket || this.reconnectTimer || !this.owner) return;
    const owner = this.owner;
    const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/stream`);
    this.socket = socket; socket.binaryType = 'arraybuffer';
    socket.onopen = () => {if (socket === this.socket) {this.error = null; this.sync();}};
    socket.onmessage = event => {
      if (socket !== this.socket || owner !== this.owner) return;
      try {
        if (typeof event.data === 'string') {
          const state = JSON.parse(event.data) as State;
          if (state.body?.session_id !== owner) {this.state = null; return;}
          this.state = state;
        } else this.receive(event.data as ArrayBuffer);
      } catch (error) {this.error = `神经数据读取失败：${String(error)}`; this.closeSocket(); this.el('note').textContent = this.error;}
    };
    socket.onclose = () => {
      if (socket !== this.socket) return;
      this.socket = null;
      if (!this.disposed && this.owner && this.shown && this.inViewport && !document.hidden) {
        this.reconnectTimer = window.setTimeout(() => {this.reconnectTimer = 0; this.sync();}, 1500);
      }
    };
  }
  private closeSocket() {
    clearTimeout(this.reconnectTimer); this.reconnectTimer = 0;
    const socket = this.socket; this.socket = null; socket?.close();
  }
  private receive(buffer: ArrayBuffer) {
    if (!this.view || !this.meta || !this.state) return;
    const packet = decodeNeuralPacket(buffer, this.meta.neurons);
    if (packet.sequence !== this.state.sequence || packet.sequence === this.streamSequence) return;
    if (packet.dense) this.view.updateCounts(packet.dense); else this.view.updateSparse(packet.indices!, packet.values!);
    this.streamSequence = packet.sequence; this.windowMs = packet.windowMs; this.simSeconds = packet.simSeconds;
    this.active = this.state.active_neurons; this.spikes = this.state.spikes;
    this.packets++; this.packetBytes += buffer.byteLength;
    this.updateNumbers();
    for (const group of this.state.regions) {
      const el = this.root.querySelector(`[data-region="${group.id}"] b`); if (el) el.textContent = number(group.active);
    }
    this.updateSelection();
  }
  private updateNumbers() {
    this.el('active-count').textContent = number(this.active);
    this.el('spikes').textContent = number(this.spikes);
    this.el('clock').textContent = `${this.simSeconds.toFixed(1)} s`;
    this.el('window').textContent = this.windowMs ? `${this.windowMs} ms 观察窗口` : '等待感觉输入';
  }
  private async selectCell(index: number) {
    const serial = ++this.selectionSerial;
    try {
      const response = await fetch(`/api/neuron/${index}`); if (!response.ok) throw new Error('神经元详情不可用');
      const neuron = await response.json() as Neuron;
      if (serial !== this.selectionSerial) return;
      this.selected = neuron; this.view?.showSelection(index, neuron.position); this.updateSelection();
      for (const name of ['focus', 'skeleton', 'clear']) (this.el(name) as HTMLButtonElement).disabled = false;
    } catch (error) {this.el('cell-detail').textContent = String(error);}
  }
  private updateSelection() {
    if (!this.selected || !this.view || !this.meta) return;
    const c = this.selected, count = this.view.counts[c.index] ?? 0;
    this.el('cell-detail').textContent = `${c.type} · ID ${c.id} · ${this.meta.regions[c.region].name} · ${count} 脉冲${this.windowMs ? ` / ${(count * 1000 / this.windowMs).toFixed(0)} Hz` : ''}`;
  }
  private async loadSkeleton() {
    if (!this.selected || !this.view) return;
    if (this.view.skeleton) {this.view.focusSelection(); return;}
    const index = this.selected.index;
    (this.el('skeleton') as HTMLButtonElement).disabled = true; this.el('skeleton').textContent = '读取中…';
    try {await this.view.showSkeleton(index, this.selected.region); if (this.selected?.index === index) {this.view.focusSelection(); this.el('skeleton').textContent = '完整骨架已显示';}}
    catch {this.el('skeleton').textContent = '暂不可用，重试';}
    finally {if (this.selected?.index === index) (this.el('skeleton') as HTMLButtonElement).disabled = false;}
  }
  private clearSelection() {
    this.selectionSerial++; this.selected = null; this.view?.clearSelection();
    this.el('cell-detail').textContent = '拖动旋转 · 滚轮缩放 · 点选神经元';
    this.el('skeleton').textContent = '查看完整骨架';
    for (const name of ['focus', 'skeleton', 'clear']) (this.el(name) as HTMLButtonElement).disabled = true;
  }
  snapshot() {
    let displayedActive = 0, displayedSpikes = 0;
    for (const count of this.view?.counts ?? []) {if (count) displayedActive++; displayedSpikes += count;}
    return {shown: this.shown, inViewport: this.inViewport, connected: this.socket?.readyState === WebSocket.OPEN, owner: this.owner,
      streamSequence: this.streamSequence, simSeconds: this.simSeconds, active: this.active, spikes: this.spikes, windowMs: this.windowMs,
      neurons: this.view?.indices.length ?? 0, surfaceBatches: this.view?.surfaces.length ?? 0, packets: this.packets, packetBytes: this.packetBytes,
      displayedActive, displayedSpikes,
      selected: this.selected?.index ?? null, onlyActive: this.view?.onlyActive, metrics: this.view?.readMetrics(),
      camera: this.view?.camera.position.toArray(), regionVisibility: this.view?.visibility.slice(), error: this.error};
  }
  dispose() {
    this.disposed = true; this.closeSocket(); this.intersection.disconnect();
    document.removeEventListener('visibilitychange', this.onDocumentVisibility); this.view?.dispose();
  }
}
