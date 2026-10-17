import random
from collections import Counter
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from .common import CLASSES,check_asset,read,fingerprint,sha

IM_MEAN=(.485,.456,.406); IM_STD=(.229,.224,.225)
CLIP_MEAN=(.48145466,.4578275,.40821073); CLIP_STD=(.26862954,.26130258,.27577711)

def manifest(spec):
    m=read(check_asset(spec))
    if m['class_order']!=CLASSES or set(m['samples'])!={'train','val','test'}: raise ValueError('Manifest schema/class/split mismatch')
    payload={'class_order':m['class_order'],'validation_seed':m['validation_seed'],'validation_ratio':m['validation_ratio'],
             'metadata':m['metadata'],'splits':m['samples']}
    # Legacy canonical encoding uses ensure_ascii=True.
    import hashlib,json
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if h!=m['manifest_hash'] or h!=spec['manifest_hash']: raise ValueError('Frozen protocol hash mismatch')
    root=Path(m['root']).resolve(); seen=set()
    if not root.is_dir(): raise FileNotFoundError(root)
    for split,rows in m['samples'].items():
        if set(r['label'] for r in rows)!=set(range(7)): raise ValueError('Every split must cover all 7 classes')
        for r in rows:
            p=Path(r['path'])
            if p.is_absolute() or '..' in p.parts or r['path'] in seen: raise ValueError('Duplicate/unsafe sample path')
            if r['split']!=split or r['class_name']!=CLASSES[r['label']]: raise ValueError('Sample label/split inconsistent')
            seen.add(r['path'])
    return m

class PairedDataset(Dataset):
    def __init__(self,m,split,seed=42,epoch=0,augment=True,indices=None):
        self.root=Path(m['root']); self.rows=m['samples'][split]
        if indices is not None: self.rows=[self.rows[i] for i in indices]
        self.seed=seed; self.epoch=epoch; self.augment=augment and split=='train'
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(self.root/row['path']) as source: image=source.convert('RGB')
        # Isolate per-sample geometry RNG; workers and resume cannot change views.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((self.seed+self.epoch*1000003+index*97)%2**32)
            if self.augment:
                i,j,h,w=T.RandomResizedCrop.get_params(image,(.9,1.),(.75,4/3))
                image=TF.resized_crop(image,i,j,h,w,[256,256],T.InterpolationMode.BICUBIC,antialias=True)
                if torch.rand(())<.5: image=TF.hflip(image)
            else: image=TF.center_crop(TF.resize(image,256,T.InterpolationMode.BICUBIC,antialias=True),[256,256])
        raw=TF.to_tensor(image)
        return {'image_student':TF.normalize(raw,IM_MEAN,IM_STD),'image_teacher':TF.normalize(raw,CLIP_MEAN,CLIP_STD),
                'label':row['label'],'sample_id':row['path']}

def class_weights(rows):
    counts=torch.bincount(torch.tensor([row['label'] for row in rows]),minlength=7).float()
    if (counts==0).any(): raise ValueError('Class balancing requires all seven training classes')
    weights=counts.reciprocal()
    return weights/weights.mean()


def loader(cfg,split,epoch=0,augment=True,indices=None):
    m=manifest(cfg['data']); ds=PairedDataset(m,split,cfg['seed'],epoch,augment,indices)
    sampler=None
    if split=='train' and cfg.get('sampling')=='balanced':
        weights=class_weights(ds.rows)
        sample_weights=weights[torch.tensor([row['label'] for row in ds.rows])]
        sampler=WeightedRandomSampler(sample_weights,len(ds),replacement=True,
                                      generator=torch.Generator().manual_seed(cfg['seed']+epoch+17))
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=split=='train' and sampler is None,sampler=sampler,drop_last=split=='train',
        generator=torch.Generator().manual_seed(cfg['seed']+epoch),num_workers=cfg['workers'],pin_memory=True,persistent_workers=False)

def audit(cfg,decode=True):
    m=manifest(cfg['data']); root=Path(m['root']); seen={}; duplicate=[]; bad=[]; counts={}; paths=set()
    for split,rows in m['samples'].items():
        counts[split]=dict(Counter(r['label'] for r in rows))
        for row in rows:
            p=(root/row['path']).resolve()
            if root.resolve() not in p.parents: raise ValueError('Symlink escapes dataset root')
            if str(p) in paths: raise ValueError('Resolved path duplicate')
            paths.add(str(p))
            try:
                if not p.is_file(): raise FileNotFoundError(p)
                if decode:
                    with Image.open(p) as im: im.convert('RGB').load()
                    h=sha(p)
                    if h in seen: duplicate.append([seen[h],{'split':split,'path':row['path'],'label':row['label']}])
                    else: seen[h]={'split':split,'path':row['path'],'label':row['label']}
            except Exception as e: bad.append({'path':row['path'],'error':str(e)})
    cross=[d for d in duplicate if d[0]['split']!=d[1]['split']]
    return {'dataset':cfg['dataset'],'manifest_sha256':cfg['data']['sha256'],'counts':counts,'bad':bad,
            'duplicates':duplicate,'cross_split_duplicates':cross,'decoded_all':decode,'metadata':m['metadata'],
            'identity_leakage':'not established without person/video identifiers','accepted':decode and not bad and not cross}
