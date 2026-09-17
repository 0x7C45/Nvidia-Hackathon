import {type Action, type Car, features, INPUTS, mechanical, Race} from './simulation.ts';
import {clamp, random, trackAt, TRACK_LENGTH} from './track.ts';

export const HIDDEN = 18;
export const OUTPUTS = 5;
const FIRST = HIDDEN * (INPUTS + 1);
export const WEIGHTS = FIRST + OUTPUTS * (HIDDEN + 1);
export interface Evaluation {reward: number; distance: number; offtrackSeconds: number; collisions: number; laps: number}
export interface Checkpoint {
  version: 1; architecture: string; source: 'mechanical-imitation' | 'reward-evolution';
  weights: number[]; generation: number; demonstrations: number; seed: number;
  evaluation?: Evaluation; history: {generation: number; reward: number}[];
}
/** Small actual MLP. Inference never calls the mechanical controller. */
export class Policy {
  weights: Float64Array;
  hidden = new Float64Array(HIDDEN);
  out = new Float64Array(OUTPUTS);
  constructor(weights?: number[], seed = 71) {
    const rng = random(seed);
    this.weights = weights ? Float64Array.from(weights) : Float64Array.from({length: WEIGHTS}, () => (rng() - .5) * .32);
  }
  forward(input: number[]) {
    for (let h = 0; h < HIDDEN; h++) {
      let z = this.weights[h * (INPUTS + 1) + INPUTS];
      for (let i = 0; i < INPUTS; i++) z += this.weights[h * (INPUTS + 1) + i] * input[i];
      this.hidden[h] = Math.tanh(z);
    }
    for (let o = 0; o < OUTPUTS; o++) {
      let z = this.weights[FIRST + o * (HIDDEN + 1) + HIDDEN];
      for (let h = 0; h < HIDDEN; h++) z += this.weights[FIRST + o * (HIDDEN + 1) + h] * this.hidden[h];
      this.out[o] = Math.tanh(z);
    }
    return this.out;
  }
  act(c: Car, cars: Car[]): Action {
    const o = this.forward(features(c, cars));
    return {steer: o[0], throttle: clamp((o[1] + 1) / 2, 0, 1), brake: clamp((o[2] + 1) / 2, 0, 1), nitro: o[3] > 0, drift: o[4] > 0};
  }
  learn(input: number[], target: number[], rate: number) {
    const o = this.forward(input), delta = new Float64Array(OUTPUTS), hd = new Float64Array(HIDDEN);
    for (let k = 0; k < OUTPUTS; k++) {
      delta[k] = (o[k] - target[k]) * (1 - o[k] ** 2) * (k === 0 ? 3 : 1);
      for (let h = 0; h < HIDDEN; h++) hd[h] += delta[k] * this.weights[FIRST + k * (HIDDEN + 1) + h];
    }
    for (let k = 0; k < OUTPUTS; k++) {
      for (let h = 0; h < HIDDEN; h++) this.weights[FIRST + k * (HIDDEN + 1) + h] -= rate * delta[k] * this.hidden[h];
      this.weights[FIRST + k * (HIDDEN + 1) + HIDDEN] -= rate * delta[k];
    }
    for (let h = 0; h < HIDDEN; h++) {
      const d = hd[h] * (1 - this.hidden[h] ** 2);
      for (let i = 0; i < INPUTS; i++) this.weights[h * (INPUTS + 1) + i] -= rate * d * input[i];
      this.weights[h * (INPUTS + 1) + INPUTS] -= rate * d;
    }
  }
  checkpoint(): Checkpoint {
    return {version: 1, architecture: `${INPUTS}-${HIDDEN}-${OUTPUTS}-tanh`, source: 'mechanical-imitation', weights: [...this.weights], generation: 0, demonstrations: 0, seed: 71, history: []};
  }
}
export function validateCheckpoint(value: unknown): Checkpoint {
  if (!value || typeof value !== 'object') throw new Error('训练文件格式无效');
  const c = value as Checkpoint;
  if (c.version !== 1 || c.architecture !== `${INPUTS}-${HIDDEN}-${OUTPUTS}-tanh` || !Array.isArray(c.weights) || c.weights.length !== WEIGHTS || c.weights.some(w => !Number.isFinite(w) || Math.abs(w) > 100) || !Number.isInteger(c.generation) || c.generation < 0 || !Array.isArray(c.history) || c.history.length > 10000 || c.history.some(h => !Number.isFinite(h.reward) || !Number.isInteger(h.generation))) throw new Error('训练文件的网络结构或权重不兼容');
  if (!['mechanical-imitation', 'reward-evolution'].includes(c.source) || !Number.isInteger(c.seed) || c.seed < 0 || c.seed > 4294967295 || !Number.isInteger(c.demonstrations) || c.demonstrations < 0) throw new Error('训练文件的来源或随机种子无效');
  if (c.evaluation && ['reward', 'distance', 'offtrackSeconds', 'collisions', 'laps'].some(k => !Number.isFinite(c.evaluation![k as keyof Evaluation]))) throw new Error('训练评估数据无效');
  return c;
}
/** Supervised driving school, explicitly sourced from the traditional controller. */
export function drivingSchool(count = 65000, seed = 71): Checkpoint {
  const p = new Policy(undefined, seed), rng = random(seed + 1), race = new Race(2, false);
  for (let i = 0; i < count; i++) {
    const c = race.cars[0], b = race.cars[1];
    c.id = i % 4; b.id = 99;
    c.s = rng() * TRACK_LENGTH;
    c.offset = (rng() - .5) * 12;
    c.heading = (rng() - .5) * .8;
    c.speed = rng() * 54;
    c.nitro = rng() * 100;
    c.offtrack = Math.abs(c.offset) > 6.3;
    b.s = c.s + (rng() < .4 ? rng() * 27 : 100); b.offset = (rng() - .5) * 9;
    const a = mechanical(c, race.cars);
    p.learn(features(c, race.cars), [a.steer, a.throttle * 1.8 - .9, a.brake * 1.8 - .9, a.nitro ? .85 : -.85, a.drift ? .85 : -.85], .018 * (1 - .6 * i / count));
  }
  const checkpoint = p.checkpoint(); checkpoint.demonstrations = count; checkpoint.seed = seed;
  return checkpoint;
}
export function evaluate(weights: number[], seeds = [17, 31, 57], seconds = 38): Evaluation {
  const result: Evaluation = {reward: 0, distance: 0, offtrackSeconds: 0, collisions: 0, laps: 0};
  for (const seed of seeds) {
    const rng = random(seed), race = new Race(1, true), p = new Policy(weights), c = race.cars[0];
    race.laps = 100;
    c.s = rng() * TRACK_LENGTH; c.startS = c.s; c.offset = (rng() - .5) * 7;
    // Compare all candidates at identical starts and the same world finish lines.
    c.lap = Math.floor(c.s / TRACK_LENGTH); c.lapRewards = c.lap;
    race.cars[1].s = c.s + 12; race.cars[1].startS = race.cars[1].s;
    for (let step = 0; step < seconds * 10; step++) race.step(.1, [p.act(c, race.cars), mechanical(race.cars[1], race.cars)]);
    result.reward += c.reward; result.distance += c.distance; result.offtrackSeconds += c.offtrackSeconds;
    result.collisions += c.collisions; result.laps += (c.s - c.startS) / TRACK_LENGTH;
  }
  for (const k of Object.keys(result) as (keyof Evaluation)[]) result[k] /= seeds.length;
  return result;
}
/** Reward-only (1+lambda) neuroevolution. Best policy retained; no teacher in rollouts. */
export function evolve(checkpoint: Checkpoint, generations: number, onGeneration?: (c: Checkpoint) => void): Checkpoint {
  let best = {...checkpoint, weights: [...checkpoint.weights], history: [...checkpoint.history]};
  let score = evaluate(best.weights);
  const rng = random(checkpoint.seed + checkpoint.generation * 101 + 9001);
  const gaussian = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, rng()))) * Math.cos(2 * Math.PI * rng());
  for (let gen = 0; gen < generations; gen++) {
    let next = best.weights, nextScore = score;
    const sigma = .018 / (1 + best.generation * .018);
    for (let candidate = 0; candidate < 12; candidate++) {
      const weights = best.weights.map((w, i) => clamp(w + gaussian() * sigma * (i < FIRST ? .3 : 1), -20, 20));
      const evaluation = evaluate(weights);
      if (evaluation.reward > nextScore.reward) {next = weights; nextScore = evaluation;}
    }
    score = nextScore;
    best = {...best, weights: [...next], source: 'reward-evolution', generation: best.generation + 1, evaluation: score,
      history: [...best.history, {generation: best.generation + 1, reward: score.reward}].slice(-2000)};
    onGeneration?.(best);
  }
  return best;
}
