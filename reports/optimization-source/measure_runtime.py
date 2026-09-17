"""Paired web-engine measurements; each trial runs the full graph from a fresh state."""
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np
import web_server as current

spec = importlib.util.spec_from_file_location('web_server_before', Path(__file__).with_name('web_server_before.py'))
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
catalog = current.Catalog.load()
trials = []
reference = None
for name, module in [('before', previous), ('after', current), ('after', current), ('before', previous)]:
    class MeasuredEngine(module.Engine):
        def _publish(self, counts, window_ms, phase, elapsed):
            start = time.perf_counter()
            super()._publish(counts, window_ms, phase, elapsed)
            self.publish_times.append(time.perf_counter() - start)
            if self.brain.sim_ms >= 5000:
                self.running = False
                self.finished.set()
    engine = MeasuredEngine.__new__(MeasuredEngine)
    engine.finished = threading.Event()
    engine.publish_times = []
    engine.__init__(catalog)
    start = time.perf_counter()
    with engine.cv:
        engine.mode = 'bright'
        engine.clients = 1
        engine.cv.notify_all()
    assert engine.finished.wait(60), engine.error
    elapsed = time.perf_counter() - start
    engine.close()
    if reference is None:
        reference = (engine.counts.copy(), engine.brain.v.copy(), engine.brain.g.copy(), engine.metadata['regions'])
    for a, b in zip(reference[:3], [engine.counts, engine.brain.v, engine.brain.g]):
        np.testing.assert_array_equal(a, b)
    assert reference[3] == engine.metadata['regions']
    if name == 'after':
        np.testing.assert_array_equal(current.decode_counts(engine.binary), engine.counts)
    row = {'version': name, 'sim_seconds': 5, 'wall_seconds': elapsed,
           'real_time_factor': 5 / elapsed, 'mean_publication_ms': 1000 * statistics.mean(engine.publish_times[1:]),
           'packet_bytes': len(engine.binary), 'active_neurons': int(np.count_nonzero(engine.counts)),
           'rss_mib': engine.process.memory_info().rss / 2**20}
    trials.append(row)
    print(json.dumps(row), flush=True)
    del engine
    gc.collect()
means = {name: {key: statistics.mean(t[key] for t in trials if t['version'] == name)
               for key in ['wall_seconds', 'real_time_factor', 'mean_publication_ms', 'packet_bytes']}
         for name in ['before', 'after']}
result = {'scope': 'Four sequential bright-field full-graph trials, 5 simulated seconds each; no browser connected to measured worker; existing viewer paused; one computing native worker',
          'trials': trials, 'means': means,
          'wall_speedup': means['before']['wall_seconds'] / means['after']['wall_seconds'],
          'publication_speedup': means['before']['mean_publication_ms'] / means['after']['mean_publication_ms'],
          'packet_reduction_percent': 100 * (1 - means['after']['packet_bytes'] / means['before']['packet_bytes']),
          'all_final_spikes_voltage_conductance_and_region_statistics_identical': True,
          'native_kernel_unchanged': True}
Path('reports/runtime-optimization.json').write_text(json.dumps(result, indent=2) + '\n')
