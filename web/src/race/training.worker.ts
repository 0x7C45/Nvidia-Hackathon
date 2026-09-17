import {evolve, validateCheckpoint} from './policy.ts';
self.onmessage = (event: MessageEvent) => {
  try {
    const checkpoint = validateCheckpoint(event.data.checkpoint);
    evolve(checkpoint, 24, state => self.postMessage({type: 'generation', checkpoint: state}));
    self.postMessage({type: 'complete'});
  } catch (error) {self.postMessage({type: 'error', message: String(error)});}
};
