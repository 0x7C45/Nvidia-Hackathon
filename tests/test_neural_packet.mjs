import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeNeuralPacket} from '../web/src/neural-packet.ts';

function sparse(indices = [1, 7], values = [3, 12]) {
  const b = new ArrayBuffer(28 + indices.length * 6), h = new DataView(b);
  h.setUint32(0, 0x32594c46, true); h.setUint32(4, 11, true); h.setFloat64(8, 1.1, true); h.setFloat32(16, 100, true);
  h.setUint32(20, 8, true); h.setUint32(24, indices.length, true);
  new Uint32Array(b, 28, indices.length).set(indices); new Uint16Array(b, 28 + indices.length * 4).set(values);
  return b;
}
test('shared FLY2 sparse decoder preserves actual IDs, counts and neural clock', () => {
  const p = decodeNeuralPacket(sparse(), 8);
  assert.equal(p.sequence, 11); assert.equal(p.simSeconds, 1.1); assert.equal(p.windowMs, 100);
  assert.deepEqual([...p.indices], [1, 7]); assert.deepEqual([...p.values], [3, 12]);
  assert.equal(decodeNeuralPacket(sparse([], []), 8).indices.length, 0);
});
test('shared decoder preserves zero entries and uint16 spikes in dense and legacy frames', () => {
  for (const legacy of [false, true]) {
    const offset = legacy ? 24 : 28, b = new ArrayBuffer(offset + 16), h = new DataView(b);
    h.setUint32(0, legacy ? 0x31594c46 : 0x32594c46, true); h.setUint32(20, 8, true);
    if (!legacy) h.setUint32(24, 0xffffffff, true);
    new Uint16Array(b, offset).set([0, 2, 0, 65535, 1, 0, 0, 6]);
    assert.deepEqual([...decodeNeuralPacket(b, 8).dense], [0, 2, 0, 65535, 1, 0, 0, 6]);
  }
});
test('mismatched or damaged frames never become fictional neural activity', () => {
  assert.throws(() => decodeNeuralPacket(sparse(), 9));
  assert.throws(() => decodeNeuralPacket(sparse().slice(0, -1), 8));
  for (const ids of [[1, 1], [7, 1], [1, 8]]) assert.throws(() => decodeNeuralPacket(sparse(ids), 8));
  assert.throws(() => decodeNeuralPacket(sparse([1, 2], [3, 0]), 8));
});
