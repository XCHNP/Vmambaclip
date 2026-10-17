from __future__ import annotations
import hashlib, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch

CLASSES = ['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
MODEL = 'vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2'
PROJECT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def check_asset(spec):
    if not isinstance(spec,dict) or not spec.get('path') or not spec.get('sha256'):
        raise ValueError('Asset must have explicit path and SHA256: '+str(spec))
    p=Path(spec['path'])
    if not p.is_absolute() or not p.is_file(): raise FileNotFoundError(p)
    if sha(p)!=spec['sha256']: raise ValueError('Asset SHA256 mismatch: '+str(p))
    return p

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def tensor_digest(state):
    h=hashlib.sha256()
    for k,t in sorted(state.items()):
        t=t.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str((t.dtype,tuple(t.shape))).encode()); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def environment():
    import sys, importlib.metadata as md
    versions={}
    for name in ['torch','torchvision','timm','mamba-ssm','causal-conv1d','triton','fvcore','numpy','Pillow']:
        try: versions[name]=md.version(name)
        except md.PackageNotFoundError: versions[name]=None
    return {'python':sys.version,'versions':versions,'cuda_build':torch.version.cuda,
            'cuda_available':torch.cuda.is_available(),'devices':torch.cuda.device_count()}

def code_identity():
    files=list((PROJECT/'vim_v2').glob('*.py'))
    files += [PROJECT/'src/open_clip/vision_mamba_clip'/x for x in ['official_vim.py','official_vmamba.py','model.py','adapters.py','data.py']]
    files += [PROJECT/'third_party/Vim/vim/models_mamba.py',PROJECT/'third_party/Vim/official_source.json']
    files += [PROJECT/'third_party/VMamba/vmamba.py',PROJECT/'third_party/VMamba/official_source.json']
    files += [PROJECT/'configs/vision_mamba_clip'/name for name in ['text_simple.json','text_distinctive.json']]
    return {str(p.relative_to(PROJECT)):sha(p) for p in files}
