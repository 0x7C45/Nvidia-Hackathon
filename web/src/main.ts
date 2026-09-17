import "./style.css";
import { BrainView } from "./brain";
import { decodeNeuralPacket } from "./neural-packet";
import type { Meta, State, Neuron } from "./types";

const el = <T extends Element = HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error("Missing element " + id);
  return node as unknown as T;
};
const format = (value: number) => Math.round(value).toLocaleString("en-US");
const names: Record<string, string> = {
  cycle: "自动轮播",
  none: "无外部驱动",
  dark: "暗场",
  bright: "全视野光",
  left: "左侧光",
  right: "右侧光",
  pulse: "脉冲光",
};
let view: BrainView,
  meta: Meta,
  state: State | null = null,
  socket: WebSocket | null = null;
let selected: number | null = null,
  selectionSerial = 0,
  lastNeuronFetch = 0,
  lastTopUpdate = 0;
let lastSequence = -1,
  resetCount = -1,
  panel = "stimulus",
  connectionReady = false,
  unloading = false;
let reconnectTimer: number | undefined, toastTimer: number | undefined;
let history: { time: number; active: number }[] = [];

function toast(message: string): void {
  const box = el("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    box.hidden = true;
  }, 4200);
}
function selectPanel(name: string): void {
  panel = name;
  document
    .querySelectorAll<HTMLButtonElement>("[data-panel]")
    .forEach((button) => {
      const on = button.dataset.panel === name;
      button.classList.toggle("selected", on);
      button.setAttribute("aria-selected", String(on));
    });
  for (const id of ["stimulus", "regions", "cells"])
    el("panel-" + id).hidden = id !== name;
  document.querySelector<HTMLElement>(".sidebar-content")!.scrollTop = 0;
}
async function control(
  command: string,
  extra: Record<string, unknown> = {},
): Promise<void> {
  try {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Flybrain-Local": "1" },
      body: JSON.stringify({ command, ...extra }),
    });
    if (!response.ok) throw new Error("控制请求未被接受");
    updateState((await response.json()) as State);
  } catch (error) {
    toast(error instanceof Error ? error.message : "本地模型连接失败");
  }
}
function updateChart(): void {
  const end = history.at(-1)?.time ?? 0,
    start = Math.max(0, end - 12);
  history = history.filter((row) => row.time >= start);
  el("chart-start").textContent = start.toFixed(1) + " s";
  el("chart-end").textContent = end.toFixed(1) + " s";
  if (history.length < 2) {
    el("chart-line").removeAttribute("d");
    el("chart-area").removeAttribute("d");
    return;
  }
  const maximum = Math.max(1, ...history.map((row) => row.active)) * 1.08;
  const points = history.map(
    (row) =>
      `${(((row.time - start) / Math.max(0.1, end - start)) * 320).toFixed(2)},${(76 - (row.active / maximum) * 65).toFixed(2)}`,
  );
  el("chart-line").setAttribute("d", "M" + points.join(" L"));
  el("chart-area").setAttribute(
    "d",
    "M0,82 L" +
      points.join(" L") +
      " L" +
      points.at(-1)!.split(",")[0] +
      ",82 Z",
  );
  el("chart-start").textContent = start.toFixed(1) + " s";
  el("chart-end").textContent = end.toFixed(1) + " s";
}
function updateInput(s: State): void {
  if (s.controller === "body") {
    el("phase-note").textContent = `${s.body?.controller_name ?? "外部场景"} 提供感觉输入；此处仅观察神经活动`;
    document.querySelector(".eye-left")!.classList.remove("left-lit");
    document.querySelector(".eye-right")!.classList.remove("right-lit");
    document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach(button => {
      button.classList.remove("selected");button.setAttribute("aria-pressed", "false");
    });
    return;
  }
  const on =
    s.phase === "bright" || (s.phase === "pulse" && s.sim_seconds % 1 < 0.2);
  document
    .querySelector(".eye-left")!
    .classList.toggle(
      "left-lit",
      s.intensity > 0 && (on || s.phase === "left"),
    );
  document
    .querySelector(".eye-right")!
    .classList.toggle(
      "right-lit",
      s.intensity > 0 && (on || s.phase === "right"),
    );
  el("retina-preview").style.setProperty(
    "--light-intensity",
    String(s.intensity * 0.72),
  );
  el("phase-note").textContent =
    s.mode === "cycle"
      ? `当前 ${names[s.phase] ?? s.phase}，每 2 个模拟秒切换`
      : `${names[s.mode] ?? s.mode}，输入到感觉神经元`;
  const slider = el<HTMLInputElement>("intensity");
  if (document.activeElement !== slider)
    slider.value = String(Math.round(s.intensity * 100));
  slider.style.setProperty("--range-fill", slider.value + "%");
  el("intensity-value").textContent = Math.round(s.intensity * 100) + "%";
  document
    .querySelectorAll<HTMLButtonElement>("[data-mode]")
    .forEach((button) => {
      const on = button.dataset.mode === s.mode;
      button.classList.toggle("selected", on);
      button.setAttribute("aria-pressed", String(on));
    });
}
function updateTop(s: State): void {
  const list = el("active-list");
  if (performance.now() - lastTopUpdate < 1200 || list.matches(":hover"))
    return;
  lastTopUpdate = performance.now();
  list.replaceChildren();
  if (!s.top.length) {
    const note = document.createElement("p");
    note.className = "subtle-note";
    note.textContent = "这个观察窗口内没有细胞放电。";
    list.append(note);
    return;
  }
  for (const neuron of s.top.slice(0, 5)) {
    const button = document.createElement("button");
    button.className = "active-neuron";
    button.setAttribute("aria-label", `检查神经元 ${neuron.type} ${neuron.id}`);
    const dot = document.createElement("i"),
      label = document.createElement("span"),
      id = document.createElement("small"),
      rate = document.createElement("em");
    dot.style.background = meta.regions[neuron.region].color;
    label.textContent = neuron.type;
    id.textContent = neuron.id;
    label.append(id);
    rate.textContent = neuron.rate_hz.toFixed(0) + " Hz";
    button.append(dot, label, rate);
    button.addEventListener("click", () => {
      void selectNeuron(neuron.index);
    });
    list.append(button);
  }
}
function updateState(s: State): void {
  state = s;
  const external = s.controller === "body";
  document.querySelectorAll<HTMLButtonElement>("[data-mode], #reset").forEach(button => {button.disabled = external});
  el<HTMLInputElement>("intensity").disabled = external;
  el("window-label").textContent = `活跃神经元 · ${s.window_ms || 0} ms`;
  if (
    resetCount !== s.reset_count ||
    (history.length > 0 && s.sim_seconds < history.at(-1)!.time)
  ) {
    history = [];
    lastSequence = -1;
    resetCount = s.reset_count;
    view?.clearSelection();
    selected = null;
    el("neuron-card").hidden = true;
    el("cell-empty").hidden = false;
  }
  el("active-count").textContent = format(s.active_neurons);
  el("population-share").textContent =
    ((100 * s.active_neurons) / meta.neurons).toFixed(1) + "%";
  const minutes = Math.floor(s.sim_seconds / 60);
  el("sim-time").textContent =
    String(minutes).padStart(2, "0") +
    ":" +
    (s.sim_seconds % 60).toFixed(1).padStart(4, "0");
  el("run-label").textContent = external ? "场景控制中" : s.running ? "暂停模拟" : "开始模拟";
  el<SVGUseElement>("run-symbol").setAttribute(
    "href",
    s.running ? "#i-pause" : "#i-play",
  );
  el<HTMLButtonElement>("run-toggle").disabled = external || !connectionReady || !!s.error;
  el("live-dot").classList.toggle("live", s.running && connectionReady);
  el("connection-label").textContent = s.error
    ? "模拟停止"
    : external ? (s.body?.busy ? "场景驱动中" : "等待场景输入") : s.running
      ? "运行中"
      : "已暂停";
  el("speed").textContent = s.running
    ? s.real_time_factor.toFixed(2) + "×"
    : "—";
  for (const group of s.regions) {
    const value = document.getElementById("region-active-" + group.id);
    if (value) value.textContent = format(group.active);
  }
  updateInput(s);
  updateTop(s);
  if (s.sequence !== lastSequence) {
    history.push({ time: s.sim_seconds, active: s.active_neurons });
    updateChart();
    lastSequence = s.sequence;
  }
  if (
    selected !== null &&
    panel === "cells" &&
    performance.now() - lastNeuronFetch > 500
  ) {
    lastNeuronFetch = performance.now();
    void refreshNeuron(selected);
  }
  if (s.error) toast(s.error);
}
function receivePacket(buffer: ArrayBuffer): void {
  const packet = decodeNeuralPacket(buffer, meta.neurons);
  if (packet.dense) view.updateCounts(packet.dense);
  else view.updateSparse(packet.indices!, packet.values!);
}
function connect(): void {
  if (
    document.hidden ||
    unloading ||
    (socket &&
      (socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING))
  )
    return;
  const current = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/stream`,
  );
  socket = current;
  current.binaryType = "arraybuffer";
  current.onopen = () => {
    if (socket !== current) return;
    connectionReady = true;
    history = [];
    lastSequence = -1;
    resetCount = -1;
    el<HTMLButtonElement>("run-toggle").disabled = false;
  };
  current.onmessage = (event: MessageEvent) => {
    if (socket !== current) return;
    try {
      if (typeof event.data === "string")
        updateState(JSON.parse(event.data) as State);
      else receivePacket(event.data as ArrayBuffer);
    } catch (error) {
      console.error(error);
      toast("数据读取失败，请刷新本地页面");
    }
  };
  current.onclose = () => {
    if (socket !== current) return;
    socket = null;
    connectionReady = false;
    el<HTMLButtonElement>("run-toggle").disabled = true;
    el("live-dot").classList.remove("live");
    el("connection-label").textContent = document.hidden
      ? "页面已挂起"
      : "连接模型";
    clearTimeout(reconnectTimer);
    if (!document.hidden && !unloading)
      reconnectTimer = window.setTimeout(connect, 1500);
  };
  current.onerror = () => {
    connectionReady = false;
  };
}
function disconnect(): void {
  clearTimeout(reconnectTimer);
  socket?.close();
}
async function refreshNeuron(index: number): Promise<void> {
  try {
    const response = await fetch("/api/neuron/" + index);
    if (!response.ok) return;
    const neuron = (await response.json()) as Neuron;
    if (selected !== index) return;
    el("neuron-type").textContent = neuron.type;
    el("neuron-id").textContent = neuron.id;
    el("neuron-spikes").textContent = String(neuron.spikes);
    el("neuron-voltage").textContent = neuron.voltage_mv?.toFixed(2) ?? "—";
    el("neuron-detail").textContent =
      `${meta.regions[neuron.region].name} · ${neuron.side === "L" ? "左侧" : neuron.side === "R" ? "右侧" : neuron.side} · ${neuron.location_kind}`;
  } catch {
    /* The next actual neural frame retries the selected cell. */
  }
}
async function selectNeuron(index: number): Promise<void> {
  const serial = ++selectionSerial;
  selected = index;
  selectPanel("cells");
  el("cell-empty").hidden = true;
  el("neuron-card").hidden = false;
  el("skeleton-status").textContent = "正在读取官方完整骨架…";
  try {
    const response = await fetch("/api/neuron/" + index);
    if (!response.ok) throw new Error("无法读取细胞信息");
    const neuron = (await response.json()) as Neuron;
    if (selected !== index || serial !== selectionSerial) return;
    view.showSelection(index, neuron.position);
    await refreshNeuron(index);
    const edges = await view.showSkeleton(index, neuron.region);
    if (selected === index && serial === selectionSerial)
      el("skeleton-status").textContent =
        `官方完整骨架 · ${format(edges)} 条分支段`;
  } catch (error) {
    if (selected === index && serial === selectionSerial)
      el("skeleton-status").textContent =
        error instanceof Error ? error.message : "骨架暂时不可用";
  }
}
function bindControls(): void {
  document
    .querySelectorAll<HTMLButtonElement>("[data-panel]")
    .forEach((button) =>
      button.addEventListener("click", () =>
        selectPanel(button.dataset.panel!),
      ),
    );
  document
    .querySelectorAll<HTMLButtonElement>("[data-mode]")
    .forEach((button) =>
      button.addEventListener("click", () => {
        void control("stimulus", { mode: button.dataset.mode });
      }),
    );
  el("run-toggle").addEventListener("click", () => {
    void control(state?.running ? "pause" : "start");
  });
  el("reset").addEventListener("click", () => {
    void control("reset");
    toast("已请求回到同一静息初态");
  });
  const slider = el<HTMLInputElement>("intensity");
  slider.addEventListener("input", () => {
    el("intensity-value").textContent = slider.value + "%";
    slider.style.setProperty("--range-fill", slider.value + "%");
  });
  slider.addEventListener("change", () => {
    void control("stimulus", { intensity: Number(slider.value) / 100 });
  });
  document
    .querySelectorAll<HTMLButtonElement>("[data-view]")
    .forEach((button) =>
      button.addEventListener("click", () => {
        document
          .querySelectorAll<HTMLButtonElement>("[data-view]")
          .forEach((other) => {
            other.classList.toggle("selected", other === button);
            other.setAttribute("aria-pressed", String(other === button));
          });
        view.setView(button.dataset.view!);
      }),
    );
  el("home-view").addEventListener("click", () =>
    view.setView(
      document.querySelector<HTMLElement>("[data-view].selected")?.dataset
        .view ?? "all",
    ),
  );
  el("rotate-toggle").addEventListener("click", () => {
    view.setRotation(!view.controls.autoRotate);
    el("rotate-toggle").classList.toggle("selected", view.controls.autoRotate);
    el("rotate-toggle").setAttribute(
      "aria-pressed",
      String(view.controls.autoRotate),
    );
  });
  el("anatomy-toggle").addEventListener("click", () => {
    view.setAnatomy(!view.anatomyVisible);
    el("anatomy-toggle").classList.toggle("selected", view.anatomyVisible);
    el("anatomy-toggle").setAttribute(
      "aria-pressed",
      String(view.anatomyVisible),
    );
  });
  el("active-only").addEventListener("click", () => {
    view.setOnlyActive(!view.onlyActive);
    el("active-only").classList.toggle("selected", view.onlyActive);
    el("active-only").setAttribute("aria-pressed", String(view.onlyActive));
  });
  el("fullscreen-toggle").addEventListener("click", () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else
      void document
        .querySelector<HTMLElement>(".brain-panel")!
        .requestFullscreen();
  });
  el("focus-neuron").addEventListener("click", () => view.focusSelection());
  el("neuron-close").addEventListener("click", () => {
    selected = null;
    view.clearSelection();
    el("neuron-card").hidden = true;
    el("cell-empty").hidden = false;
  });
  const dialog = el<HTMLDialogElement>("about-dialog");
  for (const id of ["about-open", "source-open"])
    el(id).addEventListener("click", () => dialog.showModal());
  el("about-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  document.addEventListener("keydown", (event) => {
    if (
      event.code === "Space" &&
      !(event.target instanceof HTMLInputElement) &&
      !(event.target instanceof HTMLButtonElement) &&
      !dialog.open
    ) {
      event.preventDefault();
      void control(state?.running ? "pause" : "start");
    }
  });
}
async function boot(): Promise<void> {
  try {
    const response = await fetch("/data/meta.json");
    if (!response.ok) throw new Error("缺少本地空间数据");
    meta = (await response.json()) as Meta;
    el("coverage").textContent =
      `${format(meta.located)} 个真实位置 · 本地连接组`;
    for (const region of meta.regions) {
      const legend = document.createElement("span"),
        legendDot = document.createElement("i");
      legendDot.style.background = region.color;
      legend.append(legendDot, document.createTextNode(region.name));
      el("canvas-legend").append(legend);
      const button = document.createElement("button");
      button.className = "region-row";
      button.style.setProperty("--region-color", region.color);
      button.setAttribute("aria-label", `切换${region.name}显示`);
      button.setAttribute("aria-pressed", "true");
      const dot = document.createElement("i"),
        label = document.createElement("span"),
        active = document.createElement("em"),
        eye = document.createElement("b");
      label.textContent = region.name;
      active.id = "region-active-" + region.id;
      active.textContent = "—";
      eye.textContent = "◉";
      button.append(dot, label, active, eye);
      button.addEventListener("click", () => {
        const enabled = !button.classList.toggle("off");
        view.setRegion(region.id, enabled);
        button.setAttribute("aria-pressed", String(enabled));
      });
      el("regions").append(button);
    }
    view = new BrainView(el("brain-canvas"), meta, (index) => {
      void selectNeuron(index);
    });
    await view.loadNodes();
    el("brain-loading").hidden = true;
    bindControls();
    connect();
    await view.loadAnatomy((done, total) => {
      el("anatomy-status").textContent = `官方重建 ${done} / ${total}`;
    });
    el("anatomy-status").textContent = "114 个官方脑区 · 100 ms 观察窗口";
  } catch (error) {
    const message = error instanceof Error ? error.message : "页面加载失败";
    if (!view) el("loading-detail").textContent = message;
    else el("anatomy-status").textContent = message;
    console.error(error);
    toast(message);
  }
}
window.setInterval(() => {
  if (!view || document.hidden) return;
  const metrics = view.readMetrics();
  el("resource-note").textContent =
    `${state ? Math.round(state.rss_mib) + " MiB · " : ""}${metrics.fps < 0.5 ? "按需绘制" : Math.round(metrics.fps) + " FPS"}`;
  if (socket?.readyState === WebSocket.OPEN)
    socket.send(JSON.stringify({ type: "view_metrics", ...metrics }));
}, 1500);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) disconnect();
  else {
    connect();
    view?.invalidate();
  }
});
window.addEventListener("beforeunload", () => {
  unloading = true;
  disconnect();
  view?.dispose();
});
void boot();
