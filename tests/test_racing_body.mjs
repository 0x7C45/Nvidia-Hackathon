import test from 'node:test';
import assert from 'node:assert/strict';
import {BodyBridge} from '../web/src/race/body.ts';
import {Race, mechanical, IDLE} from '../web/src/race/simulation.ts';
import {TRACK_LENGTH} from '../web/src/race/track.ts';
import {hasBehaviorReward, racingInput} from '../web/src/race/sensory.ts';

test('Body bridge waits for neural action, advances exactly 100 ms, rewards then releases', async () => {
  const original = {fetch: globalThis.fetch, location: globalThis.location, sessionStorage: globalThis.sessionStorage};
  const requests = [], saved = new Map();
  let unblock;
  globalThis.location = {origin: 'http://127.0.0.1:8787'};
  globalThis.sessionStorage = {getItem: k => saved.get(k) ?? null, setItem: (k, v) => saved.set(k, v), removeItem: k => saved.delete(k)};
  const info = {protocol_version: '1.0', session_id: 'owned', episode_id: 'episode', step_index: 0, journal: '/events'};
  globalThis.fetch = async (url, options) => {
    const parsed = new URL(String(url), 'http://127.0.0.1:8787');
    const path = parsed.pathname.split('/').pop(), data = options?.body ? JSON.parse(options.body) : undefined;
    requests.push({path, data});
    if (path === 'step') {
      await new Promise(resolve => {unblock = resolve;});
      return Response.json({protocol_version: '1.0', dt_ms: 100, step_index: data.step_index, sim_time_ms: 100, actions: {turn: 0, forward: 0}, learning: {mode: 'frozen', enabled: false, weight_updates: 0}});
    }
    if (path === 'reward') return Response.json({step_index: data.step_index, accepted: true, applied_to_weights: false, cumulative_reward: data.value, learning: {mode: 'frozen', enabled: false, weight_updates: 0}});
    if (path === 'capabilities') return Response.json({protocol_version: '1.0'});
    if (path === 'neurons') return Response.json({total: 1, items: [{id: parsed.searchParams.get('side') === 'R' ? '801533' : '800049'}]});
    if (path === 'release') return Response.json({released: true});
    return Response.json(info);
  };
  try {
    const bridge = new BodyBridge(), race = new Race(2, false), car = race.cars[0];
    await bridge.connect(0);
    const rgba = new Uint8Array(64 * 32 * 4); rgba.fill(255);
    const advancing = bridge.step(race, rgba, [IDLE, mechanical(race.cars[1], race.cars)]);
    assert.equal(race.time, 0); assert.equal(bridge.busy, true);
    await bridge.step(race, rgba, [IDLE]); assert.equal(requests.filter(r => r.path === 'step').length, 1);
    const releasing = bridge.release();
    unblock(); await advancing; await releasing;
    assert.ok(Math.abs(race.time - .1) < 1e-10);
    assert.equal(car.speed, 0, 'Zero native output cannot secretly become a throttle command');
    const step = requests.find(r => r.path === 'step').data;
    assert.equal(step.dt_ms, 100); assert.equal(step.observation.vision.pixels.length, 2048);
    assert.ok(step.observation.vision.pixels.every(v => Math.abs(v - 1) < 1e-10));
    assert.deepEqual(requests.map(r => r.path), ['capabilities', 'neurons', 'neurons', 'neurons', 'neurons', 'sessions', 'reset', 'step', 'reward', 'release']);
    const configuration = requests.find(r => r.path === 'sessions').data;
    assert.deepEqual(configuration.sensory_channels.map(c => c.name), ['body_left', 'body_right', 'nitro', 'touch']);
    assert.deepEqual(step.observation.sensors, {body_left: 0, body_right: 0, nitro: 0, touch: 0});
    assert.equal(step.observation.sugar, false);
    const reward = requests.find(r => r.path === 'reward').data;
    assert.equal(reward.step_index, step.step_index);
    assert.equal(reward.value, Object.values(reward.components).reduce((a, b) => a + b, 0));
    assert.equal(bridge.receipt.applied_to_weights, false); assert.equal(bridge.connected, false);
  } finally {Object.assign(globalThis, original);}
});

test('RGBA retinal conversion flips WebGL rows and linearizes sRGB', async () => {
  const {visionFromRGBA} = await import('../sdk/flybody.mjs');
  const pixels = new Uint8Array([255, 255, 255, 255, 128, 128, 128, 255]);
  const image = visionFromRGBA(pixels, 1, 2, {flipY: true});
  assert.ok(image.pixels[0] > .21 && image.pixels[0] < .22);
  assert.ok(Math.abs(image.pixels[1] - 1) < 1e-10);
});

test('slow observation never advances or stalls the game and retains rewards and short sensory events', async () => {
  const original = {fetch: globalThis.fetch, location: globalThis.location, sessionStorage: globalThis.sessionStorage};
  const requests = [];
  let unblockStep;
  globalThis.location = {origin: 'http://127.0.0.1:8787'};
  globalThis.sessionStorage = {setItem() {}, removeItem() {}, getItem() {return null;}};
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), location.origin).pathname.split('/').pop();
    const data = options?.body ? JSON.parse(options.body) : undefined; requests.push({path, data});
    if (path === 'capabilities') return Response.json({protocol_version: '1.0', observation_context: {}});
    if (path === 'neurons') return Response.json({total: 1, items: [{id: '800049'}]});
    if (path === 'step') {
      await new Promise(resolve => {unblockStep = resolve;});
      return Response.json({dt_ms: 100, sim_time_ms: data.step_index * 100, step_index: data.step_index, actions: {forward: 0, turn: 0}, learning: {weight_updates: 0}});
    }
    if (path === 'reward') return Response.json({accepted: true, applied_to_weights: false, terminated: data.terminated, truncated: data.truncated, learning: {weight_updates: 0}});
    return Response.json({session_id: 'observer', episode_id: 'episode', step_index: 0, journal: '/events'});
  };
  try {
    const bridge = new BodyBridge(), race = new Race(1, false), car = race.cars[0];
    await bridge.connect(0, 'observe');
    const pixels = new Uint8Array(64 * 32 * 4);
    const pending = bridge.observe(race, async () => pixels);
    await new Promise(resolve => setImmediate(resolve));
    for (let i = 0; i < 60; i++) {race.step(1 / 60, [{...IDLE, throttle: 1}]); bridge.trackWorld(race);}
    assert.ok(Math.abs(race.time - 1) < 1e-9); assert.ok(car.speed > 10); assert.equal(bridge.last, null);
    // Finished behavior signals from the real-time world while native I/O is blocked.
    race.give(car, 'overtake', 8); race.give(car, 'corner', 3);
    car.action.nitro = true; car.collisionCooldown = .2; bridge.trackWorld(race);
    car.action.nitro = false; car.collisionCooldown = 0; bridge.trackWorld(race);
    await bridge.observe(race, async () => pixels);
    assert.equal(requests.filter(r => r.path === 'step').length, 1, 'No queue of stale frames');
    unblockStep(); await pending;
    assert.ok(Math.abs(race.time - 1) < 1e-9, 'Observer cannot call race.step()');
    assert.equal(bridge.samples[0].components.overtake, 8); assert.equal(bridge.samples[0].components.corner, 3);
    await new Promise(resolve => setTimeout(resolve, 120));
    const second = bridge.observe(race, async () => pixels);
    await new Promise(resolve => setImmediate(resolve)); unblockStep(); await second;
    const inputs = requests.filter(r => r.path === 'step').map(r => r.data.observation);
    assert.equal(inputs[0].sugar, false); assert.equal(inputs[1].sugar, true);
    assert.equal(inputs[1].sensors.nitro, 1); assert.equal(inputs[1].sensors.touch, 1);
    assert.equal(inputs[0].context.source_time_ms, 0); assert.ok(Math.abs(inputs[1].context.source_time_ms - 1000) < 1e-6);
    assert.equal(inputs[1].context.clock, 'sampled'); assert.equal(inputs[1].context.source_interval_start_ms, 0);
    assert.deepEqual(bridge.samples[1].components, {}, 'Acknowledged rewards must not be repeated');
    assert.equal(bridge.samples[1].neuralTime, .2); assert.ok(Math.abs(race.time - 1) < 1e-9);
    await bridge.release();
  } finally {Object.assign(globalThis, original);}
});

test('an observer flushes the finish reward without advancing the finished world', async () => {
  const original = {fetch: globalThis.fetch, location: globalThis.location, sessionStorage: globalThis.sessionStorage};
  let unblockStep;
  globalThis.location = {origin: 'http://127.0.0.1:8787'};
  globalThis.sessionStorage = {setItem() {}, removeItem() {}, getItem() {return null;}};
  globalThis.fetch = async (url, options) => {
    const path = new URL(String(url), location.origin).pathname.split('/').pop();
    const data = options?.body ? JSON.parse(options.body) : undefined;
    if (path === 'capabilities') return Response.json({protocol_version: '1.0', observation_context: {}});
    if (path === 'neurons') return Response.json({total: 1, items: [{id: '800049'}]});
    if (path === 'step') {
      await new Promise(resolve => {unblockStep = resolve;});
      return Response.json({dt_ms: 100, sim_time_ms: data.step_index * 100, step_index: data.step_index, actions: {forward: 0, turn: 0}});
    }
    if (path === 'reward') return Response.json({accepted: true, applied_to_weights: false, terminated: data.terminated, truncated: data.truncated});
    return Response.json({session_id: 'observer', episode_id: 'episode', step_index: 0, journal: '/events'});
  };
  try {
    const bridge = new BodyBridge(), race = new Race(1, false), car = race.cars[0];
    race.laps = 1; car.s = TRACK_LENGTH - 1; car.speed = 20;
    await bridge.connect(0, 'observe');
    const pixels = new Uint8Array(64 * 32 * 4);
    const first = bridge.observe(race, async () => pixels);
    await new Promise(resolve => setImmediate(resolve));
    race.step(.1, [{...IDLE, throttle: 1}]); bridge.trackWorld(race);
    unblockStep(); await first;
    assert.ok(car.finished); assert.equal(car.components.lap, 20); assert.equal(bridge.finished, false);
    await new Promise(resolve => setTimeout(resolve, 120));
    const last = bridge.observe(race, async () => pixels);
    await new Promise(resolve => setImmediate(resolve)); unblockStep(); await last;
    assert.equal(bridge.lastInput.rewardPulse, true);
    assert.equal(bridge.receipt.terminated, true); assert.equal(bridge.receipt.applied_to_weights, false);
    assert.ok(Math.abs(race.time - .1) < 1e-10); assert.equal(bridge.last.sim_time_ms, 200);
    assert.equal(bridge.observationDue, false);
    await bridge.release();
  } finally {Object.assign(globalThis, original);}
});

test('body sensors use actual movement and events; dense progress does not continually stimulate reward', () => {
  const car = new Race(1, false).cars[0];
  const idle = racingInput(car, false); assert.deepEqual(idle.sensors, {body_left: 0, body_right: 0, nitro: 0, touch: 0});
  car.speed = 40; car.action = {...IDLE, steer: .6, nitro: true}; car.collisionCooldown = .3;
  const input = racingInput(car, true);
  assert.ok(input.sensors.body_left > 0 && input.sensors.body_right > input.sensors.body_left);
  assert.equal(input.sensors.nitro, 1); assert.equal(input.sensors.touch, 1); assert.equal(input.sugar, true);
  assert.ok(Object.values(input.sensors).every(v => v >= 0 && v <= 1));
  assert.equal(hasBehaviorReward({progress: 1.5, collision: -5}), false);
  assert.equal(hasBehaviorReward({overtake: 8, progress: .1}), true);
  assert.equal(racingInput(car, false).sugar, false, 'Reward stimulation is one explicit input window, not sticky state');
});
