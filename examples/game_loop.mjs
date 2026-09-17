/** Import from your own 3D scene; the body API knows nothing about your game engine. */
import {FlyBody, visionFromRGBA} from '../sdk/flybody.mjs';

/** world must implement reset(seed), cameraRGBA(), advance(actions, seconds), stopped().
 * cameraRGBA returns {rgba: Uint8Array, width, height, flipY?: boolean}.
 * advance returns {reward:number, components?:object, terminated?:boolean, truncated?:boolean}.
 * Render independently with requestAnimationFrame; advance physics only here.
 */
export async function runBodyEpisode(world, {baseUrl, seed = 0, dt_ms = 20} = {}) {
  const body = new FlyBody(baseUrl);
  try {
    await body.connect({controller_name: 'external-3d-scene', preset: 'visual_bci'});
    await body.reset({seed});
    await world.reset(seed);
    while (!world.stopped()) {
      const frame = await world.cameraRGBA();
      const vision = visionFromRGBA(frame.rgba, frame.width, frame.height, {flipY: !!frame.flipY});
      const neural = await body.step({vision}, {dt_ms});
      // turn is right-positive [-1,1]; forward is signed [-1,1].
      // Choose car steering angle, acceleration, braking and units in the world adapter.
      const feedback = await world.advance(neural.actions, dt_ms / 1000);
      await body.reward(feedback.reward, feedback);
      if (feedback.terminated || feedback.truncated) break;
    }
  } finally { await body.release(); }
}
