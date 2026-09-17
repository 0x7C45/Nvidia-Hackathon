import type {SensoryChannel} from '../../../sdk/flybody.mjs';
import type {Car, RewardName} from './simulation.ts';
import {clamp} from './track.ts';

/** Declared engineering inputs, resolved against official retained annotations.
 * These are not claims that a fruit fly has a biological "nitro" receptor.
 */
export const RACING_SENSORY = [
  {name: 'body_left', cellType: 'SNpp30', side: 'L', gain: 30},
  {name: 'body_right', cellType: 'SNpp30', side: 'R', gain: 30},
  {name: 'nitro', cellType: 'SApp10', side: '', gain: 30},
  {name: 'touch', cellType: 'SNta13', side: '', gain: 30},
] as const;
export async function racingSensoryChannels(): Promise<SensoryChannel[]> {
  return Promise.all(RACING_SENSORY.map(async spec => {
    const query = new URLSearchParams({cell_type: spec.cellType, limit: '1000'});
    if (spec.side) query.set('side', spec.side);
    const response = await fetch(`/api/v1/body/neurons?${query}`);
    if (!response.ok) throw new Error(`无法读取 ${spec.cellType} 感觉映射`);
    const result = await response.json();
    if (!Number.isInteger(result.total) || result.total < 1 || result.total !== result.items?.length) throw new Error(`${spec.cellType} 感觉映射不完整`);
    const ids = result.items.map((n: {id: string}) => n.id) as string[];
    if (ids.some(id => !/^\d{1,19}$/.test(id)) || new Set(ids).size !== ids.length) throw new Error('感觉神经元 ID 无效');
    return {name: spec.name, neuron_ids: ids, gain_mv: spec.gain};
  }));
}
export interface RacingInput {
  sensors: Record<string, number>;
  sugar: boolean;
  vision: boolean;
  rewardPulse: boolean;
}
export function racingInput(car: Car, rewardPulse: boolean): RacingInput {
  const motion = clamp(car.speed / 50, 0, 1), turn = car.action.steer;
  return {
    sensors: {
      body_left: clamp(motion * (1 - Math.max(0, turn) * .6), 0, 1),
      body_right: clamp(motion * (1 + Math.min(0, turn) * .6), 0, 1),
      nitro: car.action.nitro ? 1 : 0,
      touch: car.collisionCooldown > 0 ? 1 : car.offtrack ? .5 : 0,
    },
    sugar: rewardPulse, vision: true, rewardPulse,
  };
}
const BEHAVIOR_REWARDS: RewardName[] = ['overtake', 'nitro', 'drift', 'corner', 'lap'];
export function hasBehaviorReward(components: Partial<Record<RewardName, number>>): boolean {
  return BEHAVIOR_REWARDS.some(name => (components[name] ?? 0) > 0);
}
