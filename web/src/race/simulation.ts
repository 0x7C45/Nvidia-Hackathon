import {clamp, mod, ROAD_WIDTH, trackAt, TRACK_LENGTH, wrap} from './track.ts';

export type DriverKind = 'neural' | 'mechanical' | 'human';
export interface Action {steer: number; throttle: number; brake: number; nitro: boolean; drift: boolean}
export const IDLE: Action = {steer: 0, throttle: 0, brake: 0, nitro: false, drift: false};
export const REWARD_VALUES = {overtake: 8, nitro: 1.5, drift: 2, corner: 3, lap: 20, collision: -5, offtrack: -1.2};
export type RewardName = keyof typeof REWARD_VALUES | 'progress';
export const REWARD_LABELS: Record<RewardName, string> = {overtake: '超车', nitro: '氮气加速', drift: '有效漂移', corner: '干净过弯', lap: '完成一圈', collision: '碰撞', offtrack: '驶离赛道', progress: '有效前进'};
export interface RewardEvent {carId: number; type: RewardName; value: number; time: number}
export interface Car {
  id: number; name: string; color: number; kind: DriverKind;
  s: number; startS: number; offset: number; heading: number; slip: number; speed: number; nitro: number;
  action: Action; reward: number; lap: number; lapStarted: number; bestLap: number | null;
  finished: boolean; finishTime: number | null; offtrack: boolean; collisionCooldown: number;
  driftCharge: number; nitroDistance: number; nitroAwarded: boolean;
  inCorner: boolean; cornerClean: boolean; cornerDistance: number;
  lapRewards: number; distance: number; offtrackSeconds: number; collisions: number;
  components: Partial<Record<RewardName, number>>;
}
const COLORS = [0xedff7a, 0x80dce8, 0xdf9ff6, 0xff9858, 0xe4e9e5];
export function createCar(id: number, kind: DriverKind = 'neural'): Car {
  const s = -Math.floor(id / 2) * 6;
  return {id, name: kind === 'mechanical' ? 'MECH / 机械车手' : kind === 'human' ? 'YOU / 手动驾驶' : `果蝇 ${String(id + 1).padStart(2, '0')}`,
    color: COLORS[id % COLORS.length], kind, s, startS: s, offset: id % 2 ? 2.6 : -2.6, heading: 0, slip: 0, speed: 0, nitro: 100,
    action: {...IDLE}, reward: 0, lap: 0, lapStarted: 0, bestLap: null, finished: false, finishTime: null,
    offtrack: false, collisionCooldown: 0, driftCharge: 0, nitroDistance: 0, nitroAwarded: false,
    inCorner: false, cornerClean: true, cornerDistance: 0, lapRewards: 0, distance: 0, offtrackSeconds: 0, collisions: 0, components: {}};
}

/** Frenet bicycle model. s is unwrapped metres; heading/steering positive = right. */
export class Race {
  cars: Car[];
  time = 0;
  events: RewardEvent[] = [];
  laps = 3;
  private passState = new Map<string, number>();
  private passTime = new Map<string, number>();
  constructor(flies = 4, mechanical = true) {
    this.cars = Array.from({length: flies}, (_, i) => createCar(i));
    if (mechanical) this.cars.push(createCar(flies, 'mechanical'));
    for (const a of this.cars) for (const b of this.cars) if (a.id < b.id) this.passState.set(`${a.id}:${b.id}`, Math.sign(a.s - b.s));
  }
  get finished() {return this.cars.every(c => c.finished);}
  ranked() {return [...this.cars].sort((a, b) => a.finished && b.finished ? a.finishTime! - b.finishTime! : b.s - a.s);}
  give(c: Car, type: RewardName, value: number) {
    c.reward += value;
    c.components[type] = (c.components[type] ?? 0) + value;
    if (type !== 'progress') {
      this.events.push({carId: c.id, type, value, time: this.time});
      if (this.events.length > 2000) this.events.shift();
    }
  }
  step(dt: number, actions: Action[]) {
    if (!(dt > 0 && dt <= .1)) throw new Error('Physics step must be in (0, 0.1] seconds');
    // Substeps keep the same world dynamics for browser and Body API 100 ms lockstep.
    const steps = Math.ceil(dt / .02), h = dt / steps;
    for (let tick = 0; tick < steps; tick++) {
      this.time += h;
      this.cars.forEach((c, i) => this.advance(c, actions[i] ?? IDLE, h));
      this.collideAndPass();
    }
  }
  private advance(c: Car, input: Action, dt: number) {
    if (c.finished) return;
    const a: Action = {steer: clamp(input.steer), throttle: clamp(input.throttle, 0, 1), brake: clamp(input.brake, 0, 1), nitro: !!input.nitro, drift: !!input.drift};
    const oldS = c.s, p = trackAt(c.s);
    c.offtrack = Math.abs(c.offset) > ROAD_WIDTH / 2 - .7;
    const boosting = a.nitro && c.nitro > 0 && c.speed > 8 && !c.offtrack;
    const drifting = a.drift && c.speed > 14 && Math.abs(p.curvature) > .010 && Math.abs(a.steer) > .06 && !c.offtrack;
    c.action = {...a, nitro: boosting, drift: drifting};
    // Arcade tyre slip: heading is the travel direction, chassis points into the slide.
    c.slip += ((drifting ? Math.sign(a.steer) * .22 : 0) - c.slip) * (1 - Math.exp(-8 * dt));
    c.nitro = clamp(c.nitro + (boosting ? -24 : 5) * dt, 0, 100);
    c.speed = clamp(c.speed + (a.throttle * 17 + (boosting ? 15 : 0) - a.brake * 29 - 1.4 - .006 * c.speed ** 2 - (drifting ? 1.4 : 0) - (c.offtrack ? 18 : 0)) * dt, 0, boosting ? 62 : 51);
    const velocity = c.speed * Math.cos(c.heading);
    const ds = velocity / Math.max(.55, 1 - p.curvature * c.offset) * dt;
    c.s += ds;
    c.offset += Math.sin(c.heading) * c.speed * dt;
    c.heading = wrap(c.heading + Math.tan(a.steer * .42) * c.speed / 3.2 * (drifting ? .92 : 1) * dt - wrap(trackAt(c.s).angle - p.angle));
    c.collisionCooldown = Math.max(0, c.collisionCooldown - dt);
    if (Math.abs(c.offset) > ROAD_WIDTH / 2 + 2.3) {
      c.offset = Math.sign(c.offset) * (ROAD_WIDTH / 2 + 2.3);
      c.heading *= -.32;
      c.speed *= .72;
      this.collision(c);
    }
    const progress = c.s - oldS;
    c.distance += Math.max(0, progress);
    this.give(c, 'progress', (c.offtrack ? Math.min(0, progress) : progress) * .055);
    if (c.offtrack) {
      c.offtrackSeconds += dt;
      const v = REWARD_VALUES.offtrack * dt;
      c.reward += v; c.components.offtrack = (c.components.offtrack ?? 0) + v;
    }
    if (boosting) {
      c.nitroDistance += Math.max(0, progress);
      if (!c.nitroAwarded && c.nitroDistance > 18 && c.speed > 33 && Math.abs(p.curvature) < .018) {
        this.give(c, 'nitro', REWARD_VALUES.nitro); c.nitroAwarded = true;
      }
    } else if (!a.nitro) {c.nitroDistance = 0; c.nitroAwarded = false;}
    const curve = Math.abs(p.curvature) > .019;
    if (curve && !c.inCorner) {c.inCorner = true; c.cornerClean = true; c.cornerDistance = 0; c.driftCharge = 0;}
    if (c.inCorner) {
      c.cornerClean &&= !c.offtrack && c.collisionCooldown === 0 && progress > 0;
      c.cornerDistance += Math.max(0, progress);
      if (drifting && Math.abs(c.slip) > .12) c.driftCharge += Math.max(0, progress);
      if (!curve) {
        if (c.cornerClean && c.cornerDistance > 10 && c.speed > 12) {
          this.give(c, 'corner', REWARD_VALUES.corner);
          if (c.driftCharge > 9) this.give(c, 'drift', REWARD_VALUES.drift);
        }
        c.inCorner = false; c.driftCharge = 0;
      }
    }
    c.lap = Math.max(0, Math.floor(c.s / TRACK_LENGTH));
    if (c.lap > c.lapRewards) {
      c.lapRewards = c.lap;
      this.give(c, 'lap', REWARD_VALUES.lap);
      const lapTime = this.time - c.lapStarted;
      c.bestLap = c.bestLap === null ? lapTime : Math.min(c.bestLap, lapTime);
      c.lapStarted = this.time;
    }
    if (c.lap >= this.laps) {c.finished = true; c.finishTime = this.time; c.speed = 0; c.action = {...IDLE};}
  }
  private collision(c: Car) {
    if (!c.collisionCooldown) {this.give(c, 'collision', REWARD_VALUES.collision); c.collisions++;}
    c.collisionCooldown = .8; c.cornerClean = false;
  }
  private collideAndPass() {
    for (let i = 0; i < this.cars.length; i++) for (let j = i + 1; j < this.cars.length; j++) {
      const a = this.cars[i], b = this.cars[j];
      if (a.finished || b.finished) continue;
      const delta = a.s - b.s, local = mod(delta + TRACK_LENGTH / 2, TRACK_LENGTH) - TRACK_LENGTH / 2;
      if (Math.abs(local) < 3.1 && Math.abs(a.offset - b.offset) < 1.7) {
        const side = a.offset >= b.offset ? 1 : -1;
        const push = (1.7 - Math.abs(a.offset - b.offset)) * .51;
        a.offset += push * side; b.offset -= push * side;
        if (!a.collisionCooldown && !b.collisionCooldown) {a.speed *= .8; b.speed *= .8;}
        this.collision(a); this.collision(b);
      }
      const key = `${a.id}:${b.id}`, previous = this.passState.get(key) ?? 0;
      if (Math.abs(delta) > 4) {
        const side = Math.sign(delta);
        if (previous !== 0 && previous !== side && this.time > 2 && this.time - (this.passTime.get(key) ?? -20) > 8) {
          const overtaker = side > 0 ? a : b;
          if (!overtaker.offtrack && !overtaker.collisionCooldown && overtaker.speed > 10) {
            this.give(overtaker, 'overtake', REWARD_VALUES.overtake); this.passTime.set(key, this.time);
          }
        }
        this.passState.set(key, side);
      }
    }
  }
}

/** Observable road/body sensors, no future opponent action or reward in policy input. */
export function features(c: Car, cars: Car[]): number[] {
  const p = trackAt(c.s), ahead = cars.filter(b => b.id !== c.id).map(b => ({b, d: mod(b.s - c.s, TRACK_LENGTH)})).sort((a, b) => a.d - b.d)[0];
  const close = ahead && ahead.d < 28;
  return [c.offset / 7, Math.sin(c.heading), c.speed / 50, p.curvature * 30, trackAt(c.s + 12).curvature * 30,
    trackAt(c.s + 32).curvature * 30, c.nitro / 100, close ? 1 - ahead.d / 28 : 0,
    close ? (ahead.b.offset - c.offset) / 7 : 0, c.offtrack ? 1 : 0, c.id % 2 ? 1 : -1];
}
export const INPUTS = 11;
/** Conventional curvature feedforward + lateral/heading feedback + passing rule. */
export function mechanical(c: Car, cars: Car[]): Action {
  const f = features(c, cars);
  let target = c.id % 2 ? 1.65 : -1.65;
  if (f[7] > .12 && Math.abs(f[8]) < .38) target = c.offset > 0 ? -3.4 : 3.4;
  const curve = trackAt(c.s + Math.max(3, c.speed * .12)).curvature;
  const steer = Math.atan(curve * 3.2) / .42 - (c.offset - target) * .05 - c.heading * 1.55;
  const maxCurve = Math.max(Math.abs(f[3]), Math.abs(f[4]), Math.abs(f[5])) / 30;
  const targetSpeed = clamp(Math.sqrt(19 / Math.max(.009, maxCurve)), 24, 47);
  return {steer: clamp(steer), throttle: clamp((targetSpeed - c.speed) * .20 + .6, 0, 1), brake: clamp((c.speed - targetSpeed - 2) * .16, 0, 1),
    nitro: maxCurve < .015 && Math.abs(c.offset) < 4.5 && c.nitro > 8 && c.speed > 25,
    drift: Math.abs(curve) > .020 && c.speed > 16 && Math.abs(c.offset) < 5};
}
