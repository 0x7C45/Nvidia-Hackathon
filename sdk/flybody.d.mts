export interface SensoryChannel {name:string; neuron_ids:string[]; gain_mv?:number}
export interface ReadoutChannel {name:string; positive_ids?:string[]; negative_ids?:string[]; scale_hz?:number; smoothing_ms?:number}
export interface Configuration {controller_name?:string; preset?:'descending'|'visual_bci'|'custom'; sensory_channels?:SensoryChannel[]; readout_channels?:ReadoutChannel[]; learning_mode?:string; record_observations?:boolean; request_id?:string}
export interface VisionFrame {width:number; height:number; pixels:number[]}
export interface ObservationContext {clock:'lockstep'|'sampled'; source_time_ms:number; source_id:string; source_interval_start_ms?:number}
export interface Observation {vision?:VisionFrame; retina?:number[]; sensors?:Record<string,number>; sugar?:boolean; lamina_bias_mv?:number; context?:ObservationContext}
export interface LearningStatus {mode:string; enabled:boolean; weight_updates:number}
export interface SessionInfo {protocol_version:'1.0'; session_id:string; controller_name:string; episode_id:string|null; step_index:number; terminated:boolean; truncated:boolean; failed:boolean; sensory_channels:SensoryChannel[]; readout_channels:ReadoutChannel[]; learning:LearningStatus; journal:string}
export interface StepResult {protocol_version:'1.0'; session_id:string; episode_id:string; step_index:number; dt_ms:number; sim_time_ms:number; wall_ms:number; real_time_factor:number; actions:Record<string,number>; readouts:Record<string,{positive_hz:number; negative_hz:number; smoothed_difference_hz:number; value:number}>; neurons:{id:string; index:number; spikes:number; rate_hz:number}[]; active_neurons:number; total_window_spikes:number; stream_sequence:number; learning:LearningStatus}
export interface RewardReceipt {session_id:string; episode_id:string; step_index:number; accepted:boolean; applied_to_weights:boolean; cumulative_reward:number; terminated:boolean; truncated:boolean; learning:LearningStatus}
export class BodyAPIError extends Error {status:number; response:unknown}
export class FlyBody {
  constructor(baseUrl?:string);
  readonly sessionId:string|null;
  readonly episodeId:string|null;
  readonly stepIndex:number;
  connect(config?:Configuration):Promise<SessionInfo>;
  reset(options?:{seed?:number; reset_learning?:boolean}):Promise<SessionInfo>;
  step(observation:Observation, options?:{dt_ms?:number}):Promise<StepResult>;
  reward(value:number, options?:{components?:Record<string,number>; terminated?:boolean; truncated?:boolean}):Promise<RewardReceipt>;
  release():Promise<{session_id:string; released:boolean; viewer_paused:boolean}|undefined>;
}
export function visionFromRGBA(rgba:Uint8Array|Uint8ClampedArray, width:number, height:number, options?:{flipY?:boolean}):VisionFrame;
