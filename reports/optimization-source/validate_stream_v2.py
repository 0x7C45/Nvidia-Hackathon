"""Compare lossless local WebSocket packets with a fresh native full graph."""
import json
from pathlib import Path
import struct
import time
import httpx
import numpy as np
from websockets.sync.client import connect
from common import GRAPH
from doom.native import NativeBrain
from web_server import decode_counts

base='http://127.0.0.1:8787'
with httpx.Client(trust_env=False, timeout=10, headers={'Origin':base,'X-Flybrain-Local':'1'}) as client:
    original=client.get(base+'/api/state').json()
    def control(command, **extra):
        response=client.post(base+'/api/control',json={'command':command,**extra})
        response.raise_for_status()
        return response.json()
    control('pause')
    with connect('ws://127.0.0.1:8787/stream',origin=base,proxy=None,compression=None,max_size=2000000) as stream:
        epoch=control('reset',mode='bright',intensity=1)['reset_count']+1
        meta=None
        while True:
            message=stream.recv(timeout=10)
            if isinstance(message,str):
                meta=json.loads(message)
            elif meta['reset_count']==epoch and meta['sim_seconds']==0:
                assert not np.any(decode_counts(message))
                break
        control('start')
        windows=[]
        while len(windows)<5:
            message=stream.recv(timeout=10)
            if isinstance(message,str):
                meta=json.loads(message)
            elif meta['reset_count']==epoch and meta['sim_seconds']>0:
                header=struct.unpack_from('<4sIdfII',message)
                assert header[1]==meta['sequence'] and header[2]==meta['sim_seconds']
                windows.append((meta,decode_counts(message),len(message)))
        control('pause')
        time.sleep(.15)
        paused=client.get(base+'/api/state').json()
        time.sleep(.2)
        assert client.get(base+'/api/state').json()['sim_seconds']==paused['sim_seconds']
    reference=NativeBrain(GRAPH)
    light=np.ones(len(reference.retina),dtype=np.float32)
    results=[]
    for meta, actual, size in windows:
        target=round(meta['sim_seconds']*1000)
        expected=np.zeros(len(reference.ids),dtype=np.int32)
        while reference.sim_ms<target:
            counts,_=reference.step(light,10,lamina_bias=12)
            if reference.sim_ms>target-100:
                expected+=counts
        np.testing.assert_array_equal(actual,expected)
        assert int(expected.sum())==meta['spikes']
        assert np.count_nonzero(expected)==meta['active_neurons']
        results.append({'sequence':meta['sequence'],'sim_seconds':meta['sim_seconds'],
                        'all_166700_counts_identical':True,'active_neurons':meta['active_neurons'],
                        'spikes':meta['spikes'],'packet_bytes':size})
    control('stimulus',mode=original['mode'],intensity=original['intensity'])
    if original['running']:
        control('start')
report={'passed':True,'protocol':'FLY2','full_graph_neurons':166700,'neural_step_ms':.1,
        'window_ms':100,'reset_to_zero':True,'pause_freezes_simulated_time':True,
        'frames':results,'native_kernel_unchanged':True}
Path('reports/web-signal-validation-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
