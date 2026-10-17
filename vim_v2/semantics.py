"""Frozen seven-class prototype supervision and teacher/text compatibility audits."""

from pathlib import Path

import torch
import torch.nn.functional as F

from .common import CLASSES,PROJECT,atomic_json,check_asset,read,sha
from .checkpoint import atomic_save,load

PROMPTS={'simple':PROJECT/'configs/vision_mamba_clip/text_simple.json',
         'distinctive':PROJECT/'configs/vision_mamba_clip/text_distinctive.json'}


def encode_prototypes(encoder,tokenizer,schema,device):
    if schema.get('class_order')!=CLASSES or set(schema.get('prompts',{}))!=set(CLASSES) or not schema.get('source'):
        raise ValueError('Prototype prompts must have explicit ordered classes and source')
    prompts=[]; counts=[]
    for name in CLASSES:
        texts=schema['prompts'][name]
        if not isinstance(texts,list) or not texts or any(not isinstance(text,str) or not text.strip() for text in texts):
            raise ValueError('Every prototype class needs a nonempty list of prompts')
        prompts.extend(texts); counts.append(len(texts))
    tokens=tokenizer(prompts).to(device)
    with torch.no_grad():
        raw=encoder.encode_text(tokens,normalize=False).float()
        if raw.shape!=(len(prompts),768) or not torch.isfinite(raw).all(): raise ValueError('CLIP text encoder must return finite 768-D features')
        result=[]; offset=0
        for count in counts:
            average=raw[offset:offset+count].mean(0)
            if average.norm()<1e-8: raise ValueError('Zero text prototype')
            result.append(F.normalize(average,dim=0)); offset+=count
    return torch.stack(result).detach().cpu()


def prototype_statistics(bank):
    normalized=F.normalize(bank.float(),dim=-1); similarity=normalized@normalized.T
    pairs=[{'left':CLASSES[i],'right':CLASSES[j],'cosine':float(similarity[i,j])}
           for i in range(7) for j in range(i+1,7)]
    return {'class_order':CLASSES,'cosine_matrix':similarity.tolist(),
            'most_similar_pairs':sorted(pairs,key=lambda item:item['cosine'],reverse=True)}


def prepare_prototypes(paths_path,out,device='cpu'):
    from open_clip import create_model,get_tokenizer
    paths=read(paths_path); checkpoint=check_asset(paths['clip_base']); out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use a new prototype output directory')
    encoder=create_model('ViT-L-14',pretrained=str(checkpoint),force_quick_gelu=True,force_image_size=256,
                         device=device,precision='fp32')
    encoder.requires_grad_(False); encoder.eval(); tokenizer=get_tokenizer('ViT-L-14'); result={}
    for mode,path in PROMPTS.items():
        schema=read(path); bank=encode_prototypes(encoder,tokenizer,schema,device)
        metadata={'class_order':CLASSES,'mode':mode,'clip_sha256':paths['clip_base']['sha256'],
                  'prompt_sha256':sha(path),'prompt_source':schema['source'],'prompts':schema['prompts'],
                  'aggregation':'normalize(mean(raw projected text embeddings per class))'}
        target=out/(mode+'.pt'); atomic_save(target,{'prototypes':bank,'metadata':metadata})
        result[mode]={'path':str(target.resolve()),'sha256':sha(target)}
        atomic_json(out/(mode+'_similarity.json'),prototype_statistics(bank))
    atomic_json(out/'text_prototypes.json',result)
    return {'text_prototypes':result,'text_encoder_frozen':True,'training_started':False}


def load_prototypes(c,device='cpu'):
    mode=c['text']['mode']
    if mode not in PROMPTS: raise ValueError('Text-disabled experiment must not load prototypes')
    payload=load(check_asset(c['text']['prototypes'])); bank=payload['prototypes']; metadata=payload['metadata']
    if not isinstance(c.get('clip_base'),dict) or not c['clip_base'].get('sha256'):
        raise ValueError('Prototype supervision requires the declared CLIP source hash')
    if (metadata['class_order']!=CLASSES or metadata['mode']!=mode or
            metadata['clip_sha256']!=c['clip_base']['sha256'] or metadata['prompt_sha256']!=sha(PROMPTS[mode])):
        raise ValueError('Text prototype class/source/prompt identity mismatch')
    if bank.shape!=(7,768) or not torch.isfinite(bank).all() or (bank.norm(dim=-1)<1e-8).any():
        raise ValueError('Invalid text prototype bank')
    return F.normalize(bank.detach().float(),dim=-1).to(device)


@torch.no_grad()
def audit_semantic_batches(teacher,batches,bank,device):
    from .runtime import metrics
    modes={module:module.training for module in teacher.modules()}; teacher.eval()
    head=torch.zeros(7,7,dtype=torch.long); prototype=head.clone(); margins=[]; labels=[]
    try:
        for batch in batches:
            y=batch['label'].to(device); feature=teacher.encode_image(batch['image_teacher'].to(device)).float()
            similarities=F.normalize(feature,dim=-1)@bank.to(device).T
            classifier=teacher.classifier(feature)
            head+=torch.bincount((y*7+classifier.argmax(-1)).cpu(),minlength=49).reshape(7,7)
            prototype+=torch.bincount((y*7+similarities.argmax(-1)).cpu(),minlength=49).reshape(7,7)
            correct=similarities.gather(1,y[:,None]).squeeze(1)
            wrong=similarities.clone().scatter_(1,y[:,None],float('-inf')).max(-1).values
            margins.append((correct-wrong).cpu()); labels.append(y.cpu())
        if not margins: raise ValueError('Empty semantic audit split')
        margin=torch.cat(margins); y=torch.cat(labels)
        return {'teacher_classifier':metrics(head),'teacher_text_prototypes':metrics(prototype),
                'margin_mean':float(margin.mean()),'positive_margin_fraction':float((margin>0).float().mean()),
                'per_class_margin':[{'class':name,'samples':int((y==i).sum()),'mean':float(margin[y==i].mean())} for i,name in enumerate(CLASSES)],
                'prototype_statistics':prototype_statistics(bank),'weights_updated':False,
                'interpretation':'Report both classifiers and margins; no automatic claim that text and feature KD are compatible.'}
    finally:
        for module,mode in modes.items(): module.training=mode
