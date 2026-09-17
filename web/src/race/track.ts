/** One metric track shared by physics, rendering, sensors and training. */
export const ROAD_WIDTH = 14;
export const TAU = Math.PI * 2;
export const clamp = (v: number, lo = -1, hi = 1) => Math.max(lo, Math.min(hi, v));
export const wrap = (v: number) => ((v + Math.PI) % TAU + TAU) % TAU - Math.PI;
export const mod = (v: number, n: number) => ((v % n) + n) % n;
export interface TrackPoint {x: number; z: number; angle: number; curvature: number}
const raw = Array.from({length: 2401}, (_, i) => {
  const a = i / 2400 * TAU;
  return {x: 89 * Math.cos(a) + 13 * Math.sin(2 * a), z: 59 * Math.sin(a) + 10 * Math.sin(3 * a)};
});
const distances = [0];
for (let i = 1; i < raw.length; i++) distances.push(distances[i - 1] + Math.hypot(raw[i].x - raw[i - 1].x, raw[i].z - raw[i - 1].z));
export const TRACK_LENGTH = distances[distances.length - 1];
const N = 800;
const STEP = TRACK_LENGTH / N;
let j = 1;
export const TRACK: TrackPoint[] = Array.from({length: N}, (_, i) => {
  const s = i * STEP;
  while (distances[j] < s) j++;
  const t = (s - distances[j - 1]) / (distances[j] - distances[j - 1]);
  return {x: raw[j - 1].x * (1 - t) + raw[j].x * t, z: raw[j - 1].z * (1 - t) + raw[j].z * t, angle: 0, curvature: 0};
});
for (let i = 0; i < N; i++) {
  const a = TRACK[mod(i - 1, N)], b = TRACK[(i + 1) % N];
  TRACK[i].angle = Math.atan2(b.x - a.x, b.z - a.z);
}
for (let i = 0; i < N; i++) TRACK[i].curvature = wrap(TRACK[(i + 1) % N].angle - TRACK[mod(i - 1, N)].angle) / (2 * STEP);
export function trackAt(s: number, offset = 0): TrackPoint {
  const f = mod(s, TRACK_LENGTH) / STEP, i = Math.floor(f), t = f - i;
  const a = TRACK[i], b = TRACK[(i + 1) % N], angle = a.angle + wrap(b.angle - a.angle) * t;
  return {x: a.x * (1 - t) + b.x * t + Math.cos(angle) * offset, z: a.z * (1 - t) + b.z * t - Math.sin(angle) * offset,
    angle, curvature: a.curvature * (1 - t) + b.curvature * t};
}
export function random(seed: number) {
  let s = seed >>> 0;
  return () => {s += 0x6D2B79F5; let t = Math.imul(s ^ s >>> 15, 1 | s); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296;};
}
