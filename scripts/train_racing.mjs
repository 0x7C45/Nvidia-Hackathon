/** Rebuild the shipped driving-school baseline and its genuine reward-training evidence. */
import fs from 'node:fs';
import {drivingSchool, evaluate, evolve} from '../web/src/race/policy.ts';
const baseline = drivingSchool(65000, 71);
baseline.evaluation = evaluate(baseline.weights);
baseline.history = [{generation: 0, reward: baseline.evaluation.reward}];
const trained = evolve(baseline, 12);
const evidence = {
  protocol: 'docs/body-api.md', architecture: trained.architecture, source: 'external-racing-control-network',
  malecns_learning_mode: 'frozen', malecns_weights_updated: false,
  demonstration_count: baseline.demonstrations, seed: 71, generations: 12,
  rollout_seconds: 38, training_seeds: [17, 31, 57], heldout_seeds: [101, 211, 307],
  baseline: baseline.evaluation, trained: trained.evaluation,
  heldoutBefore: evaluate(baseline.weights, [101, 211, 307]), heldoutAfter: evaluate(trained.weights, [101, 211, 307]),
  memory_roundtrip: evaluate(JSON.parse(JSON.stringify(trained)).weights, [101, 211, 307]), history: trained.history,
};
fs.mkdirSync(new URL('../web/public/race/', import.meta.url), {recursive: true});
fs.writeFileSync(new URL('../web/public/race/baseline.json', import.meta.url), JSON.stringify(trained));
fs.writeFileSync(new URL('../reports/racing-training.json', import.meta.url), JSON.stringify(evidence, null, 2) + '\n');
console.log(JSON.stringify(evidence, null, 2));
