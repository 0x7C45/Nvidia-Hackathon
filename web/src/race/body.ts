import {FlyBody, visionFromRGBA, type StepResult, type RewardReceipt} from '../../../sdk/flybody.mjs';
import {type Action, type Car, IDLE, Race, type RewardName} from './simulation.ts';
import {hasBehaviorReward, racingInput, racingSensoryChannels, type RacingInput} from './sensory.ts';

/** All game/brain integration is confined to the existing Body API v1 contract. */
export class BodyBridge {
  client = new FlyBody(location.origin);
  boundCar: number | null = null;
  mode: 'drive' | 'observe' = 'drive';
  lastInput: RacingInput | null = null;
  private pendingReward = false;
  private observedThrough: Partial<Record<RewardName, number>> | null = null;
  private observedRace: Race | null = null;
  private nextObservationAt = 0;
  private sensorSinceTime = 0;
  private heldNitro = 0;
  private heldTouch = 0;
  sampledWorldTime: number | null = null;
  sampleLatencyMs = 0;
  lastSampleWallAt = 0;
  lastResultWallAt = 0;
  feedbackThroughWorldTime: number | null = null;
  samples: {step: number; worldTime: number; neuralTime: number; feedbackThrough: number; latencyMs: number; reward: number; components: Partial<Record<RewardName, number>>; input: RacingInput}[] = [];
  busy = false;
  connecting = false;
  closing = false;
  last: StepResult | null = null;
  receipt: RewardReceipt | null = null;
  journal: string | null = null;
  error: string | null = null;
  get connected() {return this.boundCar !== null && !!this.client.episodeId;}
  get finished() {return !!(this.receipt?.terminated || this.receipt?.truncated);}
  get observationDue() {return this.connected && this.mode === 'observe' && !this.busy && !this.connecting && !this.closing && !this.error && !this.finished && performance.now() >= this.nextObservationAt;}
  get observationAgeMs() {return this.sampledWorldTime === null ? null : performance.now() - this.lastSampleWallAt;}
  async connect(carId: number, mode: 'drive' | 'observe' = 'drive') {
    if (this.busy || this.connecting || this.connected) return;
    this.connecting = true; this.error = null;
    try {
      const capabilities = await fetch('/api/v1/body/capabilities').then(r => {if (!r.ok) throw new Error(`服务状态 ${r.status}`); return r.json();});
      if (capabilities.protocol_version !== '1.0') throw new Error('Body API 版本不兼容');
      if (mode === 'observe' && !capabilities.observation_context) throw new Error('请重启 ./flybrain web，以启用带时间戳的异步观察');
      const sensory_channels = await racingSensoryChannels();
      const info = await this.client.connect({controller_name: `fly-circuit-${mode}-car-${carId + 1}`, preset: 'visual_bci', learning_mode: 'frozen', sensory_channels, record_observations: true});
      this.journal = info.journal;
      sessionStorage.setItem('fly-circuit-owned-body', info.session_id);
      // Keep ownership discoverable if reset fails, so release remains possible.
      await this.client.reset({seed: 71, reset_learning: false});
      this.boundCar = carId; this.mode = mode; this.clearEpisode();
    } catch (error) {this.error = String(error); if (this.client.sessionId) await this.client.release().catch(() => {}); throw error;}
    finally {this.connecting = false;}
  }
  async step(race: Race, frame: Uint8Array, actions: Action[]) {
    if (this.mode !== 'drive') throw new Error('Use observe() for sampled observation; it never advances the world');
    if (!this.connected || this.busy || this.closing) return;
    this.busy = true;
    const car = race.cars.find(c => c.id === this.boundCar)!;
    const before = {...car.components};
    try {
      const input = racingInput(car, this.pendingReward);
      const result = await this.client.step({vision: visionFromRGBA(frame, 64, 32, {flipY: true}), sensors: input.sensors, sugar: input.sugar}, {dt_ms: 100});
      this.lastInput = input; this.pendingReward = false;
      this.last = result;
      const forward = result.actions.forward ?? 0;
      const turn = result.actions.turn ?? 0;
      // Direct engineering mapping. No hidden autopilot or minimum throttle.
      if (this.mode === 'drive') actions[race.cars.indexOf(car)] = {...IDLE, steer: turn, throttle: Math.max(0, forward), brake: Math.max(0, -forward), drift: Math.abs(turn) > .35 && forward > .2, nitro: forward > .8 && Math.abs(turn) < .1};
      race.step(result.dt_ms / 1000, actions);
      const components: Record<string, number> = {};
      for (const key of Object.keys(car.components) as RewardName[]) {
        const value = (car.components[key] ?? 0) - (before[key] ?? 0);
        if (value !== 0) components[key] = value;
      }
      const feedback = hasBehaviorReward(components);
      const terminated = car.finished;
      // Flush a final positive reward as explicit input in the following 100 ms
      // before ending the episode. Every neural step still advances the world.
      this.receipt = await this.client.reward(Object.values(components).reduce((a, b) => a + b, 0), {components, terminated: terminated && !feedback, truncated: !terminated && race.time >= 180 && !feedback});
      this.pendingReward = feedback;
    } catch (error) {this.error = String(error); throw error;}
    finally {this.busy = false;}
  }
  /** Observe the latest source frame without owning or advancing the game clock.
   * One in-flight request; rewards accumulated while it runs are sent once, then
   * converted to an explicit pulse in the next observation. No backlog of images.
   */
  async observe(race: Race, capture: (car: Car) => Promise<Uint8Array>) {
    if (!this.observationDue) return;
    const car = race.cars.find(c => c.id === this.boundCar);
    if (!car) throw new Error('Observed car is no longer present');
    this.busy = true;
    const capturedAt = performance.now(), worldTime = race.time;
    this.trackWorld(race);
    const input = racingInput(car, this.pendingReward);
    input.sensors.nitro = Math.max(input.sensors.nitro, this.heldNitro);
    input.sensors.touch = Math.max(input.sensors.touch, this.heldTouch);
    const intervalStart = this.sensorSinceTime;
    this.heldNitro = this.heldTouch = 0; this.sensorSinceTime = worldTime;
    try {
      const frame = await capture(car);
      const result = await this.client.step({
        vision: visionFromRGBA(frame, 64, 32, {flipY: true}), sensors: input.sensors, sugar: input.sugar,
        context: {clock: 'sampled', source_time_ms: worldTime * 1000, source_interval_start_ms: intervalStart * 1000, source_id: `car_${car.id}`},
      }, {dt_ms: 100});
      this.last = result; this.lastInput = input; this.pendingReward = false;
      this.sampledWorldTime = worldTime; this.lastSampleWallAt = capturedAt;
      this.lastResultWallAt = performance.now(); this.sampleLatencyMs = this.lastResultWallAt - capturedAt;
      // Snapshot before awaiting reward so events during that request survive.
      const totals = {...car.components}, through = race.time;
      const components: Partial<Record<RewardName, number>> = {};
      for (const key of Object.keys(totals) as RewardName[]) {
        const value = (totals[key] ?? 0) - (this.observedThrough?.[key] ?? 0);
        if (value !== 0) components[key] = value;
      }
      const feedback = hasBehaviorReward(components);
      const value = Object.values(components).reduce((sum, v) => sum + v, 0);
      const receipt = await this.client.reward(value, {components, terminated: race.finished && !feedback, truncated: !race.finished && through >= 180 && !feedback});
      this.receipt = receipt; this.pendingReward = feedback; this.observedThrough = totals;
      this.feedbackThroughWorldTime = through;
      this.samples.push({step: result.step_index, worldTime, neuralTime: result.sim_time_ms / 1000, feedbackThrough: through, latencyMs: this.sampleLatencyMs, reward: value, components, input});
      if (this.samples.length > 2000) this.samples.shift();
    } catch (error) {this.error = String(error); throw error;}
    finally {this.busy = false; this.nextObservationAt = Math.max(capturedAt + 100, performance.now() + 16);}
  }
  trackWorld(race: Race) {
    if (!this.connected || this.mode !== 'observe' || this.finished || this.error) return;
    const car = race.cars.find(c => c.id === this.boundCar); if (!car) return;
    if (this.observedRace !== race) {
      this.observedRace = race; this.observedThrough = {...car.components}; this.sensorSinceTime = race.time;
      this.heldNitro = this.heldTouch = 0;
    }
    this.heldNitro = Math.max(this.heldNitro, car.action.nitro ? 1 : 0);
    this.heldTouch = Math.max(this.heldTouch, car.collisionCooldown > 0 ? 1 : car.offtrack ? .5 : 0);
  }
  private clearEpisode() {
    this.last = null; this.receipt = null; this.lastInput = null; this.pendingReward = false;
    this.observedRace = null; this.observedThrough = null; this.sampledWorldTime = null;
    this.sampleLatencyMs = 0; this.feedbackThroughWorldTime = null; this.nextObservationAt = 0;
    this.lastSampleWallAt = 0; this.lastResultWallAt = 0; this.samples = [];
    this.heldNitro = this.heldTouch = 0; this.sensorSinceTime = 0;
  }
  async release() {
    this.closing = true;
    // Finish the already-started step/reward pair before releasing ownership.
    while (this.busy || this.connecting) await new Promise(r => setTimeout(r, 25));
    try {await this.client.release(); this.boundCar = null; this.error = null; this.lastInput = null; this.pendingReward = false; sessionStorage.removeItem('fly-circuit-owned-body');}
    finally {this.closing = false;}
  }
  async reset() {
    if (!this.connected || this.busy) return;
    await this.client.reset({seed: 71, reset_learning: false}); this.clearEpisode(); this.error = null;
  }
  async follow(carId: number) {
    if (!this.connected || this.mode !== 'observe' || this.boundCar === carId) return;
    await this.release(); await this.connect(carId, 'observe');
  }
  async recoverOwnSession() {
    const saved = sessionStorage.getItem('fly-circuit-owned-body');
    if (!saved) return;
    const response = await fetch('/api/v1/body/session');
    if (!response.ok) return;
    const current = await response.json();
    if (current.session?.session_id === saved) {
      const released = await fetch(`/api/v1/body/sessions/${saved}/release`, {method: 'POST', headers: {'X-Flybrain-Local': '1'}});
      if (!released.ok) throw new Error('上一场赛车的原生会话尚未释放，请稍后重试');
    }
    sessionStorage.removeItem('fly-circuit-owned-body');
  }
  releaseOnExit() {
    if (!this.client.sessionId) return;
    // Unload cannot await JS continuations. Queue a best-effort release even if
    // the page is being destroyed during a step; never invent its missing reward.
    void fetch(`/api/v1/body/sessions/${this.client.sessionId}/release`, {method: 'POST', headers: {'X-Flybrain-Local': '1'}, keepalive: true});
  }
}
