import * as THREE from 'three';
import {type Car, type Race} from './simulation.ts';
import {clamp, random, ROAD_WIDTH, trackAt, TRACK, TRACK_LENGTH} from './track.ts';

const Y = .24;
export type CameraMode = 'overview' | 'chase' | 'cockpit';
const material = (color: number, roughness = .75, metalness = .1) => new THREE.MeshStandardMaterial({color, roughness, metalness});

export class RaceRenderer {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  mainCamera = new THREE.PerspectiveCamera(43, 1, .1, 750);
  eyeCamera = new THREE.PerspectiveCamera(90, 2, .15, 500);
  sensor = new THREE.WebGLRenderTarget(64, 32, {depthBuffer: true, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter});
  previewSensor = new THREE.WebGLRenderTarget(64, 32, {depthBuffer: true, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter});
  private sensorPixels = new Uint8Array(64 * 32 * 4);
  private previewPixels = new Uint8Array(64 * 32 * 4);
  private readingSensor = false;
  private readingPreview = false;
  readbacks = 0;
  lastReadbackMs = 0;
  models: THREE.Group[] = [];
  wheels: THREE.Object3D[][] = [];
  flames: THREE.Mesh[] = [];
  wings: THREE.Mesh[][] = [];
  private skid = new THREE.InstancedMesh(new THREE.PlaneGeometry(.12, .9), new THREE.MeshBasicMaterial({color: 0x18231e, transparent: true, opacity: .55, depthWrite: false}), 512);
  private skidCursor = 0;
  private skidTimes: number[] = [];
  private skidDummy = new THREE.Object3D();
  selected = new THREE.Mesh(new THREE.RingGeometry(2.7, 2.78, 48), new THREE.MeshBasicMaterial({color: 0xedff7a, transparent: true, opacity: .8, side: THREE.DoubleSide}));
  mode: CameraMode = 'overview';
  azimuth = .81;
  elevation = .92;
  distance = 295;
  calls = 0;
  triangles = 0;
  frames = 0;
  private target = new THREE.Vector3();
  private desired = new THREE.Vector3();
  private snap = true;
  private rectangles: {world: DOMRect; eye: DOMRect};
  private eyeClip: DOMRect | null = null;
  private viewport = {width: window.innerWidth, height: window.innerHeight};
  private lastEye = -1;
  constructor(private canvas: HTMLCanvasElement, private world: HTMLElement, private eye: HTMLElement, private invalidate: () => void) {
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true, powerPreference: 'high-performance'});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.info.autoReset = false;
    this.sensor.texture.colorSpace = THREE.SRGBColorSpace;
    this.previewSensor.texture.colorSpace = THREE.SRGBColorSpace;
    this.scene.background = new THREE.Color(0x354943);
    this.scene.fog = new THREE.Fog(0x354943, 235, 610);
    this.scene.add(new THREE.HemisphereLight(0xcfe7df, 0x33432c, 2.8));
    const sun = new THREE.DirectionalLight(0xffe5c2, 3.2); sun.position.set(-60, 110, 60); this.scene.add(sun);
    this.buildWorld();
    this.skid.count = 0; this.skid.frustumCulled = false; this.scene.add(this.skid);
    this.selected.rotation.x = -Math.PI / 2; this.selected.position.y = .32; this.scene.add(this.selected);
    this.rectangles = {world: world.getBoundingClientRect(), eye: eye.getBoundingClientRect()};
    const layout = new ResizeObserver(() => this.resize());
    layout.observe(document.body);
    this.eye.closest('.right-column')?.querySelectorAll('.panel').forEach(panel => layout.observe(panel));
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('scroll', () => this.resize(), {passive: true, capture: true});
  }
  resize() {
    this.viewport = {width: window.innerWidth, height: window.innerHeight};
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x !== this.viewport.width || size.y !== this.viewport.height) this.renderer.setSize(this.viewport.width, this.viewport.height);
    this.rectangles = {world: this.world.getBoundingClientRect(), eye: this.eye.getBoundingClientRect()};
    const aside = this.eye.closest<HTMLElement>('.right-column');
    this.eyeClip = aside && ['auto', 'scroll'].includes(getComputedStyle(aside).overflowY) ? aside.getBoundingClientRect() : null;
    this.invalidate();
  }
  private ribbon(inner: number, outer: number, height: number, color: number, every = 1, parity = 0) {
    const positions: number[] = [], indices: number[] = [];
    for (let i = 0; i < TRACK.length; i++) {
      if (every > 1 && Math.floor(i / 5) % every !== parity) continue;
      const start = positions.length / 3;
      for (const s of [i / TRACK.length * TRACK_LENGTH, (i + 1) / TRACK.length * TRACK_LENGTH]) for (const offset of [inner, outer]) {
        const p = trackAt(s, offset); positions.push(p.x, height, p.z);
      }
      indices.push(start, start + 2, start + 1, start + 1, start + 2, start + 3);
    }
    const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geo.setIndex(indices); geo.computeVertexNormals();
    const mat = material(color); mat.side = THREE.DoubleSide;
    const mesh = new THREE.Mesh(geo, mat); this.scene.add(mesh); return mesh;
  }
  private instances(geometry: THREE.BufferGeometry, mat: THREE.Material, entries: {x: number; y: number; z: number; sx: number; sy: number; sz: number; angle?: number}[]) {
    const mesh = new THREE.InstancedMesh(geometry, mat, entries.length), dummy = new THREE.Object3D();
    entries.forEach((p, i) => {dummy.position.set(p.x, p.y, p.z); dummy.scale.set(p.sx, p.sy, p.sz); dummy.rotation.set(0, p.angle ?? 0, 0); dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);});
    mesh.instanceMatrix.needsUpdate = true; this.scene.add(mesh); return mesh;
  }
  private sign(text: string, sub: string, x: number, y: number, z: number, width: number, angle = 0, color = '#e8efdf') {
    const canvas = document.createElement('canvas'); canvas.width = 1024; canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#192722'; ctx.fillRect(0, 0, 1024, 256);
    ctx.fillStyle = color; ctx.font = 'bold 103px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(text, 512, 135);
    ctx.font = '28px monospace'; ctx.fillText(sub, 512, 203);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, width / 4), new THREE.MeshBasicMaterial({map: texture, side: THREE.DoubleSide}));
    mesh.position.set(x, y, z); mesh.rotation.y = angle; this.scene.add(mesh);
  }
  private buildWorld() {
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(1500, 1500), material(0x354943)); ground.rotation.x = -Math.PI / 2; ground.position.y = -1.3; this.scene.add(ground);
    const island = new THREE.Mesh(new THREE.CylinderGeometry(139, 141, 3, 96), material(0x56614b)); island.scale.z = .73; island.position.y = -1.6; this.scene.add(island);
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(141, 143, 1.4, 96, 1, true), material(0x202e2a)); rim.scale.z = .73; rim.position.y = -2.6; this.scene.add(rim);
    this.ribbon(-ROAD_WIDTH / 2 - 1.8, ROAD_WIDTH / 2 + 1.8, .06, 0x78816a);
    this.ribbon(-ROAD_WIDTH / 2, ROAD_WIDTH / 2, Y, 0x303c39);
    for (const side of [-1, 1]) {
      this.ribbon(side * 6.9, side * 7.1, .27, 0xdde2cb);
      this.ribbon(side * 7.2, side * 8.0, .25, 0xef8f50, 2, 0);
      this.ribbon(side * 7.2, side * 8.0, .25, 0xe0ddc3, 2, 1);
      this.ribbon(side * 9.1, side * 9.4, .40, 0x24372f);
      this.ribbon(side * 9.15, side * 9.32, .74, 0xb5c0a0);
    }
    const stripes = [], posts = [], lamps = [];
    for (let s = 0; s < TRACK_LENGTH; s += 7) {
      const p = trackAt(s); stripes.push({x: p.x, y: .265, z: p.z, sx: .1, sy: .015, sz: 2.3, angle: p.angle});
      for (const side of [-1, 1]) {const e = trackAt(s, side * 9.3); posts.push({x: e.x, y: .47, z: e.z, sx: .2, sy: .8, sz: .2});}
    }
    this.instances(new THREE.BoxGeometry(1, 1, 1), material(0x90998a), stripes);
    this.instances(new THREE.BoxGeometry(1, 1, 1), material(0x9da78d), posts);
    for (let s = 12; s < TRACK_LENGTH; s += 29) {const p = trackAt(s, -11); lamps.push({x: p.x, y: 2.5, z: p.z, sx: .17, sy: 5, sz: .17});}
    this.instances(new THREE.BoxGeometry(1, 1, 1), material(0x283a33), lamps);
    this.instances(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({color: 0xeaf6cc}), lamps.map(p => ({...p, y: 5, sx: .7, sy: .2, sz: .7})));
    const rng = random(38), trunks = [], trees = [], rocks = [];
    for (let i = 0; i < 280; i++) {
      const x = (rng() - .5) * 254, z = (rng() - .5) * 178;
      if (x * x / 134 ** 2 + z * z / 94 ** 2 > .94 || TRACK.some(p => Math.hypot(x - p.x, z - p.z) < 13) || (Math.abs(x) < 31 && Math.abs(z) < 28)) continue;
      const h = 3 + rng() * 8;
      trunks.push({x, y: h * .3, z, sx: .35, sy: h * .65, sz: .35});
      trees.push({x, y: h * .68, z, sx: 2.1 + h * .13, sy: h * .52, sz: 2.1 + h * .13, angle: rng() * 7});
      if (i % 3 === 0) rocks.push({x: x + 2.3, y: .3, z: z + 1, sx: 1.3, sy: .9, sz: 1.1, angle: rng() * 7});
    }
    this.instances(new THREE.CylinderGeometry(1, 1, 1, 5), material(0x384138), trunks);
    this.instances(new THREE.ConeGeometry(1, 1, 6), material(0x2e5143), trees);
    this.instances(new THREE.DodecahedronGeometry(1), material(0x89927a), rocks);
    const box = new THREE.BoxGeometry(1, 1, 1);
    this.instances(box, material(0x819080), [
      {x: -11, y: 2, z: 0, sx: 20, sy: 4, sz: 13}, {x: -10, y: 4.5, z: 0, sx: 21, sy: 1, sz: 14},
      {x: 18, y: 2.4, z: 12, sx: 9, sy: 4.8, sz: 14}, {x: 17, y: 5.5, z: 11, sx: 10, sy: 1.2, sz: 15},
    ]);
    this.instances(box, material(0x233c35, .25, .6), [{x: -10, y: 3, z: 6.56, sx: 18, sy: 1.6, sz: .1}, {x: 18, y: 3.4, z: 19.1, sx: 7, sy: 1.6, sz: .1}]);
    this.sign('FLY CIRCUIT', 'NEURAL MOTORSPORT / LOCAL LAB', -8, 8, 7, 26);
    this.sign('BODY / 01', 'SENSE. ACT. LEARN.', 18, 7.5, 19, 13);
    const tower = new THREE.Mesh(new THREE.CylinderGeometry(2, 2.8, 24, 8), material(0x3a4e42)); tower.position.set(9, 12, -14); this.scene.add(tower);
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(5.5, 4.5, 3.4, 8), material(0xd9e6bd, .3, .35)); cap.position.set(9, 24, -14); this.scene.add(cap);
    const antenna = new THREE.Mesh(new THREE.CylinderGeometry(.1, .2, 10, 6), material(0xe4ead3)); antenna.position.set(9, 30, -14); this.scene.add(antenna);
    // Start/finish gantry and checkerboard, aligned to the actual zero-distance line.
    const start = trackAt(0);
    const gantry = new THREE.Group(); gantry.position.set(start.x, Y, start.z); gantry.rotation.y = start.angle;
    for (const x of [-9, 9]) {const pillar = new THREE.Mesh(new THREE.BoxGeometry(.7, 8.5, .7), material(0xb8c5a0)); pillar.position.set(x, 4.25, 0); gantry.add(pillar);}
    const top = new THREE.Mesh(new THREE.BoxGeometry(19, 1.8, .6), material(0xddec98)); top.position.y = 8; gantry.add(top); this.scene.add(gantry);
    this.sign('FLY / CIRCUIT', 'START — FINISH', start.x, 8.3, start.z, 16, start.angle);
    for (let x = -6; x <= 6; x++) for (let row = 0; row < 2; row++) {
      const p = trackAt(row * .8, x);
      const tile = new THREE.Mesh(new THREE.PlaneGeometry(1, .8), new THREE.MeshBasicMaterial({color: (x + row) % 2 ? 0x242e29 : 0xe5e9cf}));
      tile.rotation.set(-Math.PI / 2, 0, -p.angle); tile.position.set(p.x, .28, p.z); this.scene.add(tile);
    }
    // Outer billboard and striped pit pads.
    const b = trackAt(TRACK_LENGTH * .52, -15); this.sign('LEARN. DRIVE. REPEAT.', 'DROSOPHILA RACING RESEARCH', b.x, 4.2, b.z, 24, b.angle + Math.PI / 2, '#edff7a');
    const pads = [];
    for (let i = 0; i < 5; i++) pads.push({x: -22 + i * 4, y: .07, z: 15, sx: 2.6, sy: .1, sz: 4});
    this.instances(box, material(0x9ba68b), pads);
  }
  syncCars(cars: Car[]) {
    this.skid.count = 0; this.skidCursor = 0; this.skidTimes = cars.map(() => -1);
    for (const m of this.models) {this.scene.remove(m); m.traverse(o => {if (o instanceof THREE.Mesh) {o.geometry.dispose(); (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => m.dispose());}});}
    this.models = []; this.wheels = []; this.flames = []; this.wings = [];
    for (const car of cars) {
      const g = new THREE.Group(), paint = material(car.color, .4, .25), black = material(0x15221e), alloy = material(0xb1bcac, .25, .7);
      const add = (geo: THREE.BufferGeometry, mat: THREE.Material, x: number, y: number, z: number) => {const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); g.add(m); return m;};
      const shadow = add(new THREE.CircleGeometry(2.1, 24), new THREE.MeshBasicMaterial({color: 0x0a1512, transparent: true, opacity: .24, depthWrite: false}), 0, .01, 0); shadow.rotation.x = -Math.PI / 2; shadow.scale.x = .64;
      add(new THREE.BoxGeometry(1.65, .45, 3.1), paint, 0, .57, 0);
      add(new THREE.BoxGeometry(1.3, .22, 1.05), paint, 0, .86, 1.05);
      add(new THREE.BoxGeometry(2.05, .15, .45), black, 0, .37, 1.65);
      add(new THREE.BoxGeometry(2.1, .16, .52), paint, 0, 1.18, -1.52);
      add(new THREE.BoxGeometry(.13, .7, .15), alloy, -.65, .86, -1.5);
      add(new THREE.BoxGeometry(.13, .7, .15), alloy, .65, .86, -1.5);
      const wheels: THREE.Mesh[] = [];
      for (const x of [-1, 1]) for (const z of [-.95, 1.03]) {const w = add(new THREE.CylinderGeometry(.46, .46, .42, 12), black, x, .46, z); w.rotation.z = Math.PI / 2; wheels.push(w);}
      const fly = new THREE.Group(); g.add(fly); fly.position.set(0, 1, -.15);
      const abdomen = new THREE.Mesh(new THREE.SphereGeometry(.53, 12, 8), material(car.kind === 'mechanical' ? 0x758a82 : 0x474c32)); abdomen.scale.set(.7, .8, 1.2); fly.add(abdomen);
      const head = new THREE.Mesh(new THREE.SphereGeometry(.43, 12, 8), material(0xa2aa82)); head.position.set(0, .28, .39); fly.add(head);
      for (const side of [-1, 1]) {const e = new THREE.Mesh(new THREE.SphereGeometry(.245, 12, 8), material(car.kind === 'mechanical' ? 0x8dffe9 : 0xce5e48, .2, .3)); e.position.set(side * .28, .35, .58); fly.add(e);}
      const wings: THREE.Mesh[] = [];
      if (car.kind !== 'mechanical') for (const side of [-1, 1]) {
        const wing = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 6), new THREE.MeshStandardMaterial({color: 0xdfeddf, transparent: true, opacity: .58, roughness: .35, depthWrite: false}));
        wing.scale.set(.72, .035, .32); wing.position.set(side * .56, .15, -.18); wing.rotation.y = side * -.5; fly.add(wing); wings.push(wing);
      }
      for (const x of [-.5, .5]) add(new THREE.BoxGeometry(.35, .13, .06), new THREE.MeshBasicMaterial({color: 0xf4ffce}), x, .78, 1.58);
      const flame = add(new THREE.ConeGeometry(.38, 2, 8), new THREE.MeshBasicMaterial({color: 0x87ebf4, transparent: true, opacity: .8, depthWrite: false}), 0, .55, -2.3); flame.rotation.x = -Math.PI / 2; flame.visible = false;
      g.userData.carId = car.id; this.scene.add(g); this.models.push(g); this.wheels.push(wheels); this.flames.push(flame); this.wings.push(wings);
    }
    this.snap = true;
  }
  setMode(mode: CameraMode) {this.mode = mode; this.snap = true;}
  private eyeAt(car: Car) {
    const p = trackAt(car.s, car.offset), angle = p.angle + car.heading + car.slip;
    this.eyeCamera.position.set(p.x + Math.sin(angle) * .45, 1.85, p.z + Math.cos(angle) * .45);
    this.eyeCamera.up.set(0, 1, 0);
    this.eyeCamera.lookAt(p.x + Math.sin(angle) * 40, 1.0, p.z + Math.cos(angle) * 40);
    this.eyeCamera.rotateZ(car.action.drift ? -car.action.steer * .035 : 0);
  }
  private clipped(rect: DOMRect, clip: DOMRect | null = null) {
    const left = Math.max(0, rect.left, clip?.left ?? 0), right = Math.min(this.viewport.width, rect.right, clip?.right ?? Infinity);
    const top = Math.max(0, rect.top, clip?.top ?? 0), bottom = Math.min(this.viewport.height, rect.bottom, clip?.bottom ?? Infinity);
    return right > left && bottom > top ? {left, right, top, bottom} : null;
  }
  get eyeVisible() {return !!this.clipped(this.rectangles.eye, this.eyeClip);}
  private drawViewport(rect: DOMRect, camera: THREE.Camera, clip: DOMRect | null = null) {
    const visible = this.clipped(rect, clip); if (!visible) return;
    const {height} = this.viewport;
    this.renderer.setViewport(rect.x, height - rect.bottom, rect.width, rect.height);
    this.renderer.setScissor(visible.left, height - visible.bottom, visible.right - visible.left, visible.bottom - visible.top);
    this.renderer.render(this.scene, camera);
  }
  render(race: Race, selectedId: number, dt: number) {
    this.frames++;
    const car = race.cars.find(c => c.id === selectedId) ?? race.cars[0];
    race.cars.forEach((c, i) => {
      const p = trackAt(c.s, c.offset), g = this.models[i];
      g.position.set(p.x, Y + .02, p.z); g.rotation.y = p.angle + c.heading + c.slip;
      g.rotation.z = -c.action.steer * c.speed * .0017;
      this.wheels[i].forEach(w => w.rotation.y = race.time * c.speed * 1.8);
      this.flames[i].visible = c.action.nitro;
      this.flames[i].scale.y = 1 + Math.sin(race.time * 51) * .2;
      this.wings[i].forEach((w, k) => {w.rotation.z = Math.sin(race.time * 45) * .11 * (k ? 1 : -1);});
      if (c.action.drift && Math.abs(c.slip) > .12 && race.time - this.skidTimes[i] > .045) {
        this.skidTimes[i] = race.time;
        for (const side of [-1, 1]) {
          const yaw = p.angle + c.heading + c.slip;
          this.skidDummy.position.set(p.x + Math.cos(yaw) * side * .9 - Math.sin(yaw), Y + .035, p.z - Math.sin(yaw) * side * .9 - Math.cos(yaw));
          this.skidDummy.rotation.set(-Math.PI / 2, 0, -yaw); this.skidDummy.updateMatrix();
          this.skid.setMatrixAt(this.skidCursor++ % 512, this.skidDummy.matrix); this.skid.count = Math.min(512, this.skidCursor);
        }
        this.skid.instanceMatrix.needsUpdate = true;
      }
    });
    const p = trackAt(car.s, car.offset), heading = p.angle + car.heading;
    this.selected.position.set(p.x, .31, p.z); (this.selected.material as THREE.MeshBasicMaterial).color.setHex(car.color);
    if (this.lastEye !== car.id) {this.snap = true; this.lastEye = car.id;}
    if (this.mode === 'overview') {
      this.desired.set(Math.sin(this.azimuth) * Math.cos(this.elevation) * this.distance, Math.sin(this.elevation) * this.distance, Math.cos(this.azimuth) * Math.cos(this.elevation) * this.distance);
      this.target.set(0, 0, 0);
    } else if (this.mode === 'chase') {
      this.desired.set(p.x - Math.sin(heading) * 11.5, 6.5, p.z - Math.cos(heading) * 11.5);
      this.target.set(p.x + Math.sin(heading) * 10, 1, p.z + Math.cos(heading) * 10);
    } else {this.eyeAt(car); this.desired.copy(this.eyeCamera.position); this.eyeCamera.getWorldDirection(this.target); this.target.multiplyScalar(40).add(this.desired);}
    this.mainCamera.position.lerp(this.desired, this.snap || this.mode === 'cockpit' ? 1 : 1 - Math.exp(-8 * Math.min(dt, .05)));
    this.mainCamera.lookAt(this.target);
    const targetFov = this.mode === 'overview' ? 43 : this.mode === 'cockpit' ? 90 : car.action.nitro ? 67 : 59;
    this.mainCamera.fov += (targetFov - this.mainCamera.fov) * (this.snap ? 1 : 1 - Math.exp(-10 * dt)); this.snap = false;
    this.mainCamera.aspect = this.rectangles.world.width / this.rectangles.world.height; this.mainCamera.updateProjectionMatrix();
    this.renderer.setRenderTarget(null); this.renderer.setScissorTest(false); this.renderer.setClearColor(0, 0); this.renderer.clear(); this.renderer.setScissorTest(true); this.renderer.info.reset();
    const model = this.models[race.cars.indexOf(car)];
    if (this.mode === 'cockpit') {model.visible = false; this.selected.visible = false;}
    this.drawViewport(this.rectangles.world, this.mainCamera);
    model.visible = false; this.selected.visible = false; this.eyeAt(car); this.drawViewport(this.rectangles.eye, this.eyeCamera, this.eyeClip);
    model.visible = true; this.selected.visible = true;
    this.calls = this.renderer.info.render.calls; this.triangles = this.renderer.info.render.triangles;
    this.renderer.setScissorTest(false);
  }
  async capture(car: Car, cars: Car[], preview = false): Promise<Uint8Array> {
    if (preview ? this.readingPreview : this.readingSensor) throw new Error('A camera readback is already in flight');
    if (preview) this.readingPreview = true; else this.readingSensor = true;
    const target = preview ? this.previewSensor : this.sensor, pixels = preview ? this.previewPixels : this.sensorPixels;
    const started = performance.now();
    cars.forEach((c, i) => {const p = trackAt(c.s, c.offset); this.models[i].position.set(p.x, Y + .02, p.z); this.models[i].rotation.y = p.angle + c.heading + c.slip;});
    this.eyeAt(car);
    const model = this.models[cars.indexOf(car)]; model.visible = false; this.selected.visible = false;
    try {
      this.renderer.setScissorTest(false); this.renderer.setRenderTarget(target); this.renderer.render(this.scene, this.eyeCamera);
      // WebGL2 pixel-pack buffer + GPU fence; no synchronous gl.readPixels stall.
      const readback = this.renderer.readRenderTargetPixelsAsync(target, 0, 0, 64, 32, pixels);
      this.renderer.setRenderTarget(null); model.visible = true; this.selected.visible = true;
      await readback; this.readbacks++; this.lastReadbackMs = performance.now() - started;
      return pixels;
    } finally {
      model.visible = true; this.selected.visible = true;
      if (preview) this.readingPreview = false; else this.readingSensor = false;
    }
  }
  pick(x: number, y: number) {
    const r = this.rectangles.world;
    const ray = new THREE.Raycaster(); ray.setFromCamera(new THREE.Vector2((x - r.x) / r.width * 2 - 1, -(y - r.y) / r.height * 2 + 1), this.mainCamera);
    for (const hit of ray.intersectObjects(this.models, true)) {let o: THREE.Object3D | null = hit.object; while (o) {if (o.userData.carId !== undefined) return o.userData.carId as number; o = o.parent;}}
    return null;
  }
}
