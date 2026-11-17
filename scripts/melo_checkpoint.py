"""Versioned checkpoints; a manifest written last commits a complete save."""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from run_korean_melo_spec import save,sha256


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def rng_state():
    state=np.random.get_state()
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],'python':random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    random.setstate(state['python'])


def save_checkpoint(directory,cfg,nets,optimizers,step,history,probe):
    started=time.perf_counter()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    generator,discriminator,duration=nets
    names={'generator':f'G_{step}.pth','state':'training_state.pth'}
    # Unique directories preserve all older checkpoints. Incomplete directories
    # have no manifest and are never accepted for recovery.
    torch.save({'model':{k:v.detach().cpu() for k,v in generator.state_dict().items()},
        'iteration':step,'learning_rate':cfg['train']['learning_rate'],'scope':cfg['experiment']['scope']},
        directory/names['generator'])
    torch.save({'schema_version':2,'discriminator':discriminator.state_dict(),
        'duration_discriminator':duration.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
        'step':step,'history':history,'initial_probe':probe.detach().cpu(),'rng':rng_state()},directory/names['state'])
    payload={'schema_version':2,'step':step,'config_sha256':config_digest(cfg),
        'dataset_manifest_sha256':sha256(cfg['experiment']['dataset_manifest']),
        'files':{name:{'path':path,'sha256':sha256(directory/path)} for name,path in names.items()},
        'save_seconds':time.perf_counter()-started}
    save(directory/'checkpoint.json',payload)
    return payload


def verify_checkpoint(directory,cfg):
    directory=Path(directory)
    manifest=json.loads((directory/'checkpoint.json').read_text())
    if manifest.get('schema_version')!=2 or manifest['config_sha256']!=config_digest(cfg):
        raise ValueError('Checkpoint/config mismatch')
    if manifest['dataset_manifest_sha256']!=sha256(cfg['experiment']['dataset_manifest']):
        raise ValueError('Checkpoint dataset manifest changed')
    for name in ('generator','state'):
        item=manifest['files'][name]
        if Path(item['path']).name!=item['path'] or sha256(directory/item['path'])!=item['sha256']:
            raise ValueError('Checkpoint integrity failure')
    return manifest
