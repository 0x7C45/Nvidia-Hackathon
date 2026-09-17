import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Race, mechanical, IDLE} from '../web/src/race/simulation.ts';
import {Policy, validateCheckpoint, drivingSchool, evaluate, evolve} from '../web/src/race/policy.ts';
import {trackAt, TRACK_LENGTH} from '../web/src/race/track.ts';

const memory = validateCheckpoint(JSON.parse(fs.readFileSync(new URL('../web/public/race/baseline.json', import.meta.url))));

test('track is closed, distance-parametrized and has signed right-facing normals', () => {
  assert.deepEqual(trackAt(0), trackAt(TRACK_LENGTH));
  for (let s = 0; s < TRACK_LENGTH; s += 3) {
    const a = trackAt(s), b = trackAt(s + .01), r = trackAt(s, 2);
    assert.ok(Math.abs(Math.hypot(b.x - a.x, b.z - a.z) - .01) < .0002);
    assert.ok(Math.abs(Math.hypot(r.x - a.x, r.z - a.z) - 2) < 1e-8);
  }
});

for (const flies of [2, 3, 4]) test(`${flies} independent neural racers and one mechanical racer finish the race`, () => {
  const race = new Race(flies, true), policies = race.cars.map(c => new Policy(memory.weights));
  for (let i = 0; i < 1200 && !race.finished; i++) race.step(.1, race.cars.map((c, j) => c.kind === 'mechanical' ? mechanical(c, race.cars) : policies[j].act(c, race.cars)));
  assert.equal(race.cars.length, flies + 1); assert.ok(race.finished);
  assert.ok(race.cars.every(c => c.finished && Number.isFinite(c.reward) && c.s >= TRACK_LENGTH * 3));
  assert.ok(race.cars.every(c => c.offtrackSeconds < 1));
  const types = new Set(race.events.map(e => e.type));
  for (const type of ['corner', 'drift', 'nitro', 'lap']) assert.ok(types.has(type), type);
  assert.equal(race.ranked()[0].finishTime, Math.min(...race.cars.map(c => c.finishTime)));
});

test('nitro and drift buttons at rest cannot farm rewards', () => {
  const race = new Race(1, false);
  for (let i = 0; i < 400; i++) race.step(.1, [{...IDLE, nitro: true, drift: true}]);
  assert.equal(race.cars[0].reward, 0);
  assert.equal(race.cars[0].nitro, 100);
  assert.equal(race.events.length, 0);
});

test('laps require crossing the finish line and cannot be repaid by recrossing it', () => {
  const race = new Race(1, false), c = race.cars[0];
  c.s = TRACK_LENGTH - .1; c.speed = 20;
  race.step(.02, [mechanical(c, race.cars)]);
  assert.equal(c.lap, 1); assert.equal(c.components.lap, 20);
  c.s = TRACK_LENGTH - .1;
  race.step(.02, [mechanical(c, race.cars)]);
  assert.equal(c.components.lap, 20);
});

test('leaving the road produces a penalty and prevents progress farming', () => {
  const race = new Race(1, false), c = race.cars[0]; c.offset = 8; c.speed = 15;
  race.step(.1, [{...IDLE, throttle: 1}]);
  assert.ok(c.offtrack); assert.ok(c.reward < 0); assert.ok(c.components.offtrack < 0);
  assert.equal(c.components.progress, 0);
});

test('a clean pass rewards once with hysteresis, contact does not reward a pass', () => {
  const race = new Race(2, false), a = race.cars[0], b = race.cars[1];
  race.time = 3; a.s = -10; b.s = 0; a.offset = -3; b.offset = 3;
  race.step(.02, [IDLE, IDLE]); a.s = 10; a.speed = 20;
  race.step(.02, [IDLE, IDLE]); assert.equal(a.components.overtake, 8);
  a.s = -10; race.step(.02, [IDLE, IDLE]); a.s = 10; race.step(.02, [IDLE, IDLE]);
  assert.equal(a.components.overtake, 8);
  const contact = new Race(2, false); contact.cars.forEach(c => {c.s = 5; c.offset = 0; c.speed = 20;});
  contact.step(.1, [IDLE, IDLE]); assert.ok(contact.cars.every(c => c.components.collision < 0));
  assert.ok(contact.cars.every(c => !c.components.overtake));
});

test('physics is deterministic and rejects inconsistent step sizes', () => {
  const a = new Race(2, false), b = new Race(2, false);
  for (let i = 0; i < 100; i++) {a.step(.1, a.cars.map(c => mechanical(c, a.cars))); b.step(.1, b.cars.map(c => mechanical(c, b.cars)));}
  assert.deepEqual(a.cars, b.cars);
  assert.throws(() => a.step(.11, [IDLE])); assert.throws(() => a.step(NaN, [IDLE]));
});

test('reward optimization updates real weights and improves held-out driving', () => {
  const baseline = drivingSchool(); baseline.evaluation = evaluate(baseline.weights);
  const after = evolve(baseline, 12);
  assert.notDeepEqual(after.weights, baseline.weights);
  assert.ok(after.evaluation.reward > baseline.evaluation.reward);
  const beforeTest = evaluate(baseline.weights, [101, 211, 307]);
  const afterTest = evaluate(after.weights, [101, 211, 307]);
  assert.ok(afterTest.reward > beforeTest.reward);
  assert.ok(afterTest.distance > beforeTest.distance);
  assert.equal(afterTest.offtrackSeconds, 0);
  assert.equal(afterTest.collisions, 0);
});

test('exported and reloaded memory retains identical decisions', () => {
  const loaded = validateCheckpoint(JSON.parse(JSON.stringify(memory))), race = new Race(2, false);
  const before = new Policy(memory.weights), after = new Policy(loaded.weights);
  for (let i = 0; i < 100; i++) {const c = race.cars[0]; assert.deepEqual(before.act(c, race.cars), after.act(c, race.cars)); race.step(.1, race.cars.map(c => mechanical(c, race.cars)));}
});

test('malformed memory is rejected before it can poison saved policy or the renderer', () => {
  assert.throws(() => validateCheckpoint({}));
  assert.throws(() => validateCheckpoint({...memory, weights: [1, 2]}));
  assert.throws(() => validateCheckpoint({...memory, evaluation: {reward: 'bad'}}));
  assert.throws(() => validateCheckpoint({...memory, seed: -1}));
  const weights = [...memory.weights]; weights[4] = Infinity;
  assert.throws(() => validateCheckpoint({...memory, weights}));
});
