/** Lossless viewer transport shared by the laboratory and the racing brain panel. */
export interface NeuralPacket {
  sequence: number;
  simSeconds: number;
  windowMs: number;
  neurons: number;
  dense?: Uint16Array;
  indices?: Uint32Array;
  values?: Uint16Array;
}
export function decodeNeuralPacket(buffer: ArrayBuffer, expected: number): NeuralPacket {
  if (buffer.byteLength < 24) throw new Error('Incomplete neural packet');
  const h = new DataView(buffer), magic = h.getUint32(0, true), neurons = h.getUint32(20, true);
  if (neurons !== expected) throw new Error('Neural graph size mismatch');
  const packet: NeuralPacket = {sequence: h.getUint32(4, true), simSeconds: h.getFloat64(8, true), windowMs: h.getFloat32(16, true), neurons};
  if (!Number.isFinite(packet.simSeconds) || !Number.isFinite(packet.windowMs) || packet.windowMs < 0) throw new Error('Invalid neural clock');
  if (magic === 0x31594c46) {
    if (buffer.byteLength !== 24 + neurons * 2) throw new Error('Invalid FLY1 length');
    return {...packet, dense: new Uint16Array(buffer, 24, neurons)};
  }
  if (magic !== 0x32594c46 || buffer.byteLength < 28) throw new Error('Unknown neural packet');
  const code = h.getUint32(24, true);
  if (code === 0xffffffff) {
    if (buffer.byteLength !== 28 + neurons * 2) throw new Error('Invalid dense count payload');
    return {...packet, dense: new Uint16Array(buffer, 28, neurons)};
  }
  if (code > neurons || buffer.byteLength !== 28 + code * 6) throw new Error('Invalid sparse count payload');
  const indices = new Uint32Array(buffer, 28, code), values = new Uint16Array(buffer, 28 + code * 4, code);
  for (let i = 0; i < code; i++) if (indices[i] >= neurons || (i > 0 && indices[i] <= indices[i - 1]) || values[i] === 0) throw new Error('Invalid neuron event index');
  return {...packet, indices, values};
}
