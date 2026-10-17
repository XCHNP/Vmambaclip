"""Epoch-boundary resume only; incomplete epochs are replayed from last boundary."""
import os, random, tempfile
from pathlib import Path
import numpy as np
import torch
from .common import fingerprint,tensor_digest,sha,atomic_json

def load(path): return torch.load(path,map_location='cpu',weights_only=True,mmap=os.name!='nt')

def atomic_save(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        check=load(tmp)
        if set(check)!=set(payload): raise RuntimeError('Checkpoint verification failed')
        del check
        os.replace(tmp,path)
        if os.name=='posix':
            d=os.open(path.parent,os.O_DIRECTORY)
            try: os.fsync(d)
            finally: os.close(d)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(s):
    random.setstate(s['python']); n=s['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(s['cpu'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

def restore(payload,identity,model,optimizer,scheduler,scaler):
    if payload['kind']!='resume' or payload['identity']!=identity: raise ValueError('Strict resume identity mismatch')
    if payload['next_batch']!=0: raise ValueError('V2 guarantees epoch-boundary resume only')
    model.load_state_dict(payload['model'],strict=True); optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler']); scaler.load_state_dict(payload['scaler']); restore_rng(payload['rng'])
    if payload['global_update']!=scheduler.updates: raise ValueError('Scheduler cursor mismatch')

def transfer(*args,**kwargs):
    raise ValueError('Transfer is disabled in the core matrix; cannot disguise transfer as resume/init')

def record_stage_transition(out,model,optimizer,scheduler,boundary):
    """Persist the exact S1 last parent before the first S2 update; never select S1 best."""
    out=Path(out); previous=load(out/'last.pt')
    if previous['kind']!='resume' or previous['next_epoch']!=boundary or previous['next_batch']!=0 or previous['global_update']!=scheduler.updates:
        raise ValueError('S1 last boundary/cursor mismatch')
    if tensor_digest(model.state_dict())!=tensor_digest(previous['model']): raise ValueError('S2 did not continue from S1 last model')
    actual=optimizer.state_dict()
    if actual['param_groups']!=previous['optimizer']['param_groups'] or set(actual['state'])!=set(previous['optimizer']['state']):
        raise ValueError('S1 optimizer state or groups were reset')
    for key,state in actual['state'].items():
        for field,value in state.items():
            other=previous['optimizer']['state'][key][field]
            if torch.is_tensor(value):
                if not torch.equal(value.detach().cpu(),other): raise ValueError('S1 Adam state changed at transition')
            elif value!=other: raise ValueError('S1 optimizer state changed at transition')
    atomic_save(out/'s1_last.pt',previous)
    parent=sha(out/'s1_last.pt')
    atomic_json(out/'stage_transition.json',{'source':'s1_last','checkpoint_sha256':parent,'global_update':scheduler.updates,
        'next_epoch':boundary,'model_digest':tensor_digest(model.state_dict()),'optimizer_state_entries':len(actual['state']),
        'optimizer_reset':False,'scheduler_reset':False})
    return parent
