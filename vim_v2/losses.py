import torch
import torch.nn.functional as F

def objectives(logits, feature, labels, weights, target=None):
    """FP32 only; zero-weight branches never dereference their input/teacher."""
    raw={}; weighted={}
    if weights['ce']:
        raw['ce']=F.cross_entropy(logits.float(),labels)
    if weights['feature']:
        if target is None or feature.shape!=target['feature'].shape: raise ValueError('Feature target shape mismatch')
        raw['feature']=(1-F.cosine_similarity(feature.float(),target['feature'].detach().float(),dim=-1,eps=1e-6)).mean()
    if weights['logits']:
        t=float(weights['temperature'])
        raw['logits']=F.kl_div(F.log_softmax(logits.float()/t,dim=-1),
            F.softmax(target['logits'].detach().float()/t,dim=-1),reduction='batchmean')*t*t
    if not raw: raise ValueError('No active objective')
    for k,v in raw.items(): weighted[k]=v*weights[k]
    return sum(weighted.values()),raw,weighted

def coefficients(method,stage):
    return {'ce':0. if stage=='s1' else 1.,'feature':1. if stage=='s1' else (.2 if method in ['full_v2','hybrid_feature_kd'] else 0.),
            'logits':.2 if method=='hybrid_logits_kd' else 0.,'temperature':2.,
            'au':0.,'weight_align':0.,'semantic':0.,'kl_direction':'teacher||student','feature_target':'CLIP_projected_global'}
