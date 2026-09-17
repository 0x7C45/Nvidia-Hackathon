import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import type { Meta } from "./types";

const vertex = `
attribute vec3 aRegionColor;
attribute float aRegion;
attribute float aActivity;
uniform float uPixelRatio;
uniform float uOnlyActive;
uniform float uRegions[5];
varying vec3 vColor;
varying float vActivity;
varying float vVisible;
void main(){
  vColor=aRegionColor;vActivity=aActivity;vVisible=uRegions[int(aRegion+.1)];
  if(uOnlyActive>.5 && aActivity<.001)vVisible=0.;
  vec4 p=modelViewMatrix*vec4(position,1.);
  gl_Position=projectionMatrix*p;
  gl_PointSize=clamp((1.3+4.2*aActivity)*uPixelRatio*950./max(10.,-p.z),.7,10.);
}`;
const fragment = `
varying vec3 vColor;
varying float vActivity;
varying float vVisible;
void main(){
  if(vVisible<.5)discard;
  float d=length(gl_PointCoord-vec2(.5))*2.;
  if(d>1.)discard;
  float core=exp(-d*d*15.);
  float halo=pow(1.-d,1.5);
  vec3 color=mix(vColor,vec3(.9,1.,.94),vActivity*core*.5);
  gl_FragColor=vec4(color*(.7+vActivity*.3),(.16+vActivity*.75)*halo);
  #include <colorspace_fragment>
}`;
interface MeshInfo {
  label: string;
  region: number;
  file: string;
  vertices: number;
  triangles: number;
}
interface Transition {
  start: number;
  from: THREE.Vector3;
  to: THREE.Vector3;
  targetFrom: THREE.Vector3;
  targetTo: THREE.Vector3;
}
export interface BrainViewOptions {
  maxFps?: number;
  pixelRatio?: number;
  scaleBar?: HTMLElement | null;
}
export class BrainView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(39, 1, 1, 8000);
  controls: OrbitControls;
  geometry = new THREE.BufferGeometry();
  material: THREE.ShaderMaterial;
  points: THREE.Points;
  indices: Uint32Array = new Uint32Array();
  regionIndex: Uint8Array = new Uint8Array();
  activity: Float32Array = new Float32Array();
  counts: Uint16Array = new Uint16Array();
  visibility = [1, 1, 1, 1, 1];
  surfaces: THREE.Mesh[] = [];
  skeleton: THREE.LineSegments | null = null;
  marker: THREE.Mesh;
  selected: number | null = null;
  anatomyVisible = true;
  onlyActive = false;
  fps = 0;
  frameCount = 0;
  private vertexOf: Int32Array = new Int32Array();
  private previousActive: Uint32Array = new Uint32Array();
  private observer: ResizeObserver;
  private queued = false;
  private disposed = false;
  private visible = true;
  private frameRequest = 0;
  private redrawTimer = 0;
  private lastDraw = -Infinity;
  private lastFrame = performance.now();
  private lastMetricTime = performance.now();
  private lastMetricFrames = 0;
  private selectionSerial = 0;
  private currentView = "all";
  private transition: Transition | null = null;
  private reducedMotion = matchMedia("(prefers-reduced-motion: reduce)")
    .matches;
  private light = new THREE.DirectionalLight(0xd9ebe4, 1.4);
  private scaleBar: HTMLElement | null =
    document.querySelector(".canvas-scale>span");

  constructor(
    private container: HTMLElement,
    private meta: Meta,
    private selectCallback: (index: number) => void,
    private options: BrainViewOptions = {},
  ) {
    if (options.scaleBar !== undefined) this.scaleBar = options.scaleBar;
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, options.pixelRatio ?? 1.5));
    this.renderer.setClearColor(0x0b1016, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 90;
    this.controls.maxDistance = 4000;
    this.controls.autoRotateSpeed = 0.22;
    this.controls.addEventListener("change", this.invalidate);
    this.controls.addEventListener("start", () => {
      this.transition = null;
      this.invalidate();
    });
    this.material = new THREE.ShaderMaterial({
      vertexShader: vertex,
      fragmentShader: fragment,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      uniforms: {
        uPixelRatio: { value: Math.min(devicePixelRatio, options.pixelRatio ?? 1.5) },
        uOnlyActive: { value: 0 },
        uRegions: { value: this.visibility },
      },
    });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.visible = false;
    this.points.renderOrder = 2;
    this.scene.add(this.points);
    this.scene.add(new THREE.HemisphereLight(0xdde9e5, 0x182331, 1.2));
    this.light.position.set(250, 550, 900);
    this.scene.add(this.light);
    this.marker = new THREE.Mesh(
      new THREE.RingGeometry(7.5, 8.5, 40),
      new THREE.MeshBasicMaterial({
        color: 0xd1f4af,
        transparent: true,
        opacity: 0.9,
        side: THREE.DoubleSide,
        depthTest: false,
      }),
    );
    this.marker.visible = false;
    this.marker.renderOrder = 5;
    this.scene.add(this.marker);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(container);
    this.resize();
    let down: [number, number] = [0, 0];
    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      down = [event.clientX, event.clientY];
    });
    this.renderer.domElement.addEventListener("pointerup", (event) => {
      if (
        event.button === 0 &&
        Math.hypot(event.clientX - down[0], event.clientY - down[1]) < 4
      )
        this.pick(event.clientX, event.clientY);
    });
    document.addEventListener("visibilitychange", this.onVisibility);
  }
  async loadNodes(): Promise<void> {
    const response = await fetch("/data/nodes.bin");
    if (!response.ok) throw new Error("无法读取神经元坐标");
    const buffer = await response.arrayBuffer(),
      header = new DataView(buffer),
      total = header.getUint32(0, true),
      n = header.getUint32(4, true);
    if (
      total !== this.meta.neurons ||
      n !== this.meta.located ||
      buffer.byteLength !== 8 + n * 17
    )
      throw new Error("空间数据与模型不一致");
    const positions = new Float32Array(buffer, 8, n * 3);
    this.indices = new Uint32Array(buffer, 8 + n * 12, n);
    this.regionIndex = new Uint8Array(buffer, 8 + n * 16, n);
    this.activity = new Float32Array(n);
    this.counts = new Uint16Array(total);
    this.vertexOf = new Int32Array(total).fill(-1);
    const palette = this.meta.regions.map(
        (group) => new THREE.Color(group.color),
      ),
      colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      palette[this.regionIndex[i]].toArray(colors, i * 3);
      this.vertexOf[this.indices[i]] = i;
    }
    this.geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    this.geometry.setAttribute(
      "aRegionColor",
      new THREE.BufferAttribute(colors, 3),
    );
    this.geometry.setAttribute(
      "aRegion",
      new THREE.BufferAttribute(this.regionIndex, 1),
    );
    this.geometry.setAttribute(
      "aActivity",
      new THREE.BufferAttribute(this.activity, 1).setUsage(
        THREE.DynamicDrawUsage,
      ),
    );
    this.geometry.computeBoundingBox();
    this.geometry.computeBoundingSphere();
    this.points.visible = true;
    this.setView("all", false);
  }
  async loadAnatomy(
    progress: (done: number, total: number) => void,
  ): Promise<void> {
    const response = await fetch("/data/anatomy.json");
    if (!response.ok) throw new Error("官方脑区外形尚未准备");
    const manifest = (await response.json()) as { meshes: MeshInfo[] };
    const groups: THREE.BufferGeometry[][] = Array.from(
      { length: this.meta.regions.length },
      () => [],
    );
    let next = 0,
      done = 0;
    const work = async () => {
      while (next < manifest.meshes.length) {
        const info = manifest.meshes[next++],
          result = await fetch("/data/" + info.file);
        if (!result.ok) throw new Error("脑区外形读取失败");
        const buffer = await result.arrayBuffer(),
          header = new DataView(buffer),
          nv = header.getUint32(0, true),
          nt = header.getUint32(4, true);
        if (buffer.byteLength !== 8 + nv * 12 + nt * 12)
          throw new Error("脑区外形不完整");
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(new Float32Array(buffer, 8, nv * 3), 3),
        );
        geometry.setIndex(
          new THREE.BufferAttribute(
            new Uint32Array(buffer, 8 + nv * 12, nt * 3),
            1,
          ),
        );
        groups[info.region].push(geometry);
        progress(++done, manifest.meshes.length);
      }
    };
    await Promise.all(Array.from({ length: 6 }, work));
    for (let region = 0; region < groups.length; region++) {
      const source = groups[region];
      if (!source.length) continue;
      const geometry = mergeGeometries(source, false);
      if (!geometry) throw new Error("脑区批次合并失败");
      geometry.computeVertexNormals();
      geometry.computeBoundingSphere();
      for (const original of source) original.dispose();
      const material = new THREE.MeshLambertMaterial({
        color: this.meta.regions[region].color,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        forceSinglePass: true,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.region = region;
      mesh.visible = this.anatomyVisible && !!this.visibility[region];
      mesh.renderOrder = 0;
      this.surfaces.push(mesh);
      this.scene.add(mesh);
      this.invalidate();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }
  }
  updateCounts(counts: Uint16Array): void {
    const indices: number[] = [],
      values: number[] = [];
    for (let i = 0; i < counts.length; i++)
      if (counts[i]) {
        indices.push(i);
        values.push(counts[i]);
      }
    this.updateSparse(new Uint32Array(indices), new Uint16Array(values));
  }
  updateSparse(indices: Uint32Array, values: Uint16Array): void {
    let same = indices.length === this.previousActive.length;
    for (let i = 0; same && i < indices.length; i++)
      if (this.counts[indices[i]] !== values[i]) same = false;
    if (same) return;
    for (const id of this.previousActive) {
      this.counts[id] = 0;
      const vertex = this.vertexOf[id];
      if (vertex >= 0) this.activity[vertex] = 0;
    }
    for (let i = 0; i < indices.length; i++) {
      const id = indices[i];
      if (id >= this.meta.neurons) throw new Error("非法神经元索引");
      this.counts[id] = values[i];
      const vertex = this.vertexOf[id];
      if (vertex >= 0)
        this.activity[vertex] = Math.min(
          1,
          0.2 + Math.log2(1 + values[i]) * 0.18,
        );
    }
    this.previousActive = indices.slice();
    this.geometry.getAttribute("aActivity").needsUpdate = true;
    if (this.skeleton && this.selected !== null)
      (this.skeleton.material as THREE.LineBasicMaterial).opacity =
        this.counts[this.selected] > 0.0 ? 0.95 : 0.4;
    this.invalidate();
  }
  setVisible(value: boolean): void {
    if (this.visible === value) return;
    this.visible = value;
    this.controls.enabled = value;
    if (value) this.invalidate();
    else this.cancelDraw();
  }
  private cancelDraw(): void {
    cancelAnimationFrame(this.frameRequest);
    clearTimeout(this.redrawTimer);
    this.queued = false;
  }
  private onVisibility = (): void => {
    if (document.hidden) this.cancelDraw();
    else this.invalidate();
  };
  setRegion(id: number, visible: boolean): void {
    this.visibility[id] = visible ? 1 : 0;
    for (const mesh of this.surfaces)
      if (mesh.userData.region === id)
        mesh.visible = this.anatomyVisible && visible;
    this.invalidate();
  }
  setOnlyActive(value: boolean): void {
    this.onlyActive = value;
    this.material.uniforms.uOnlyActive.value = value ? 1 : 0;
    this.invalidate();
  }
  setAnatomy(value: boolean): void {
    this.anatomyVisible = value;
    for (const mesh of this.surfaces)
      mesh.visible = value && !!this.visibility[mesh.userData.region];
    this.invalidate();
  }
  setRotation(value: boolean): void {
    this.controls.autoRotate = value;
    this.invalidate();
  }
  setView(view: string, animate = true): void {
    this.currentView = view;
    let target = new THREE.Vector3(),
      size = new THREE.Vector3(760, 1060, 560);
    if (this.geometry.boundingBox) {
      this.geometry.boundingBox.getCenter(target);
      this.geometry.boundingBox.getSize(size);
    }
    if (view === "brain") {
      target.set(0, 335, 0);
      size.set(760, 380, 420);
    }
    if (view === "vnc") {
      target.set(0, -245, 60);
      size.set(390, 650, 330);
    }
    this.moveCamera(target, size, animate);
  }
  private moveCamera(
    target: THREE.Vector3,
    size: THREE.Vector3,
    animate: boolean,
  ): void {
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const distance =
      Math.max(
        size.y / (2 * tangent),
        size.x / (2 * tangent * this.camera.aspect),
      ) *
        1.05 +
      size.z * 0.35;
    const to = target.clone().add(new THREE.Vector3(30, 15, distance));
    if (!animate || this.reducedMotion) {
      this.camera.position.copy(to);
      this.controls.target.copy(target);
      this.camera.lookAt(target);
      this.transition = null;
    } else
      this.transition = {
        start: performance.now(),
        from: this.camera.position.clone(),
        to,
        targetFrom: this.controls.target.clone(),
        targetTo: target,
      };
    this.invalidate();
  }
  showSelection(index: number, position: number[] | null): void {
    this.clearSelection();
    this.selected = index;
    if (position) {
      this.marker.position.fromArray(position);
      this.marker.visible = true;
    }
    this.invalidate();
  }
  async showSkeleton(index: number, region: number): Promise<number> {
    const serial = this.selectionSerial,
      response = await fetch("/api/skeleton/" + index);
    if (!response.ok) throw new Error("官方骨架暂时不可用");
    const buffer = await response.arrayBuffer();
    if (this.selected !== index || serial !== this.selectionSerial) return 0;
    const header = new DataView(buffer),
      nv = header.getUint32(0, true),
      ne = header.getUint32(4, true);
    if (buffer.byteLength !== 8 + nv * 12 + ne * 8)
      throw new Error("骨架数据不完整");
    const positions = new Float32Array(buffer, 8, nv * 3),
      geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(
      new THREE.BufferAttribute(
        new Uint32Array(buffer, 8 + nv * 12, ne * 2),
        1,
      ),
    );
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const material = new THREE.LineBasicMaterial({
      color: new THREE.Color(this.meta.regions[region].color).lerp(
        new THREE.Color("#d2efa8"),
        0.35,
      ),
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    this.skeleton = new THREE.LineSegments(geometry, material);
    this.skeleton.renderOrder = 4;
    this.scene.add(this.skeleton);
    if (!this.marker.visible && nv) {
      this.marker.position.fromArray(positions);
      this.marker.visible = true;
    }
    this.invalidate();
    return ne;
  }
  focusSelection(): void {
    if (this.skeleton?.geometry.boundingBox) {
      const box = this.skeleton.geometry.boundingBox;
      const size = box
        .getSize(new THREE.Vector3())
        .max(new THREE.Vector3(130, 130, 130));
      this.moveCamera(box.getCenter(new THREE.Vector3()), size, true);
    } else if (this.marker.visible)
      this.moveCamera(
        this.marker.position,
        new THREE.Vector3(170, 170, 170),
        true,
      );
  }
  clearSelection(): void {
    this.selectionSerial++;
    this.selected = null;
    this.marker.visible = false;
    if (this.skeleton) {
      this.scene.remove(this.skeleton);
      this.skeleton.geometry.dispose();
      (this.skeleton.material as THREE.Material).dispose();
      this.skeleton = null;
    }
    this.invalidate();
  }
  readMetrics(): {
    frames: number;
    draw_calls: number;
    fps: number;
    visible: boolean;
  } {
    const now = performance.now(),
      dt = (now - this.lastMetricTime) / 1000;
    if (dt >= 0.5) {
      this.fps = (this.frameCount - this.lastMetricFrames) / dt;
      this.lastMetricTime = now;
      this.lastMetricFrames = this.frameCount;
    }
    return {
      frames: this.frameCount,
      draw_calls: this.renderer.info.render.calls,
      fps: this.fps,
      visible: this.visible && !document.hidden,
    };
  }
  private pick(x: number, y: number): void {
    const rect = this.renderer.domElement.getBoundingClientRect(),
      pointer = new THREE.Vector2(
        ((x - rect.left) / rect.width) * 2 - 1,
        (-(y - rect.top) / rect.height) * 2 + 1,
      );
    const ray = new THREE.Raycaster();
    ray.params.Points.threshold = 3;
    ray.setFromCamera(pointer, this.camera);
    for (const hit of ray.intersectObject(this.points)) {
      const i = hit.index;
      if (i === undefined) continue;
      const global = this.indices[i];
      if (
        this.visibility[this.regionIndex[i]] &&
        (!this.onlyActive || this.counts[global] > 0)
      ) {
        this.selectCallback(global);
        break;
      }
    }
  }
  private resize = (): void => {
    const { width, height } = this.container.getBoundingClientRect();
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.setView(this.currentView, false);
  };
  invalidate = (): void => {
    if (!this.queued && !this.disposed && this.visible && !document.hidden) {
      this.queued = true;
      this.frameRequest = requestAnimationFrame(this.render);
    }
  };
  private render = (now: number): void => {
    this.queued = false;
    if (this.disposed || !this.visible || document.hidden) return;
    const remaining = this.options.maxFps ? 1000 / this.options.maxFps - (now - this.lastDraw) : 0;
    if (remaining > 0.5) {
      this.queued = true;
      this.redrawTimer = window.setTimeout(() => {
        this.queued = false;
        this.invalidate();
      }, remaining);
      return;
    }
    this.lastDraw = now;
    const dt = Math.min(0.05, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    if (this.transition) {
      const t = Math.min(1, (now - this.transition.start) / 420),
        e = 1 - Math.pow(1 - t, 3);
      this.camera.position.lerpVectors(
        this.transition.from,
        this.transition.to,
        e,
      );
      this.controls.target.lerpVectors(
        this.transition.targetFrom,
        this.transition.targetTo,
        e,
      );
      if (t === 1) this.transition = null;
    }
    this.controls.dampingFactor = 1 - Math.exp(-8 * dt);
    this.controls.update(dt);
    this.marker.quaternion.copy(this.camera.quaternion);
    const distance = this.camera.position.distanceTo(this.marker.position),
      worldPerPixel =
        (2 *
          distance *
          Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) /
        Math.max(1, this.container.clientHeight);
    this.marker.scale.setScalar(worldPerPixel);
    this.renderer.render(this.scene, this.camera);
    this.frameCount++;
    const center = this.controls.target.clone(),
      right = new THREE.Vector3(100, 0, 0).applyQuaternion(
        this.camera.quaternion,
      ),
      a = center.clone().project(this.camera),
      b = center.add(right).project(this.camera);
    if (this.scaleBar)
      this.scaleBar.style.width =
        (Math.abs(b.x - a.x) * this.container.clientWidth) / 2 + "px";
    if (this.transition || this.controls.autoRotate) this.invalidate();
  };
  dispose(): void {
    this.disposed = true;
    this.cancelDraw();
    document.removeEventListener("visibilitychange", this.onVisibility);
    this.observer.disconnect();
    this.clearSelection();
    this.controls.dispose();
    this.geometry.dispose();
    this.material.dispose();
    for (const mesh of this.surfaces) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.marker.geometry.dispose();
    (this.marker.material as THREE.Material).dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
