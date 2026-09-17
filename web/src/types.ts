export interface Region {
  id: number;
  name: string;
  en: string;
  color: string;
  count: number;
  located: number;
}
export interface Meta {
  neurons: number;
  located: number;
  unlocated: number;
  edges: number;
  contacts: number;
  regions: Region[];
}
export interface Neuron {
  index: number;
  id: string;
  type: string;
  class: string;
  superclass: string;
  side: string;
  region: number;
  located: boolean;
  position: number[] | null;
  location_kind: string;
  spikes: number;
  rate_hz: number;
  voltage_mv?: number;
}
export interface State {
  controller?: "viewer" | "body";
  body?: { session_id: string; episode_id: string | null; controller_name: string; busy: boolean; awaiting_input: boolean } | null;
  type: string;
  sequence: number;
  revision: number;
  reset_count: number;
  running: boolean;
  mode: string;
  intensity: number;
  phase: string;
  sim_seconds: number;
  window_ms: number;
  real_time_factor: number;
  average_real_time_factor: number;
  active_neurons: number;
  active_located: number;
  total_spikes: number;
  spikes: number;
  rss_mib: number;
  error: string | null;
  regions: { id: number; active: number; spikes: number }[];
  top: Neuron[];
}
