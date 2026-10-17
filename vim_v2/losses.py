import torch
import torch.nn.functional as F

def objectives(logits, feature, labels, weights, target=None, prototypes=None, cnn_aux_logits=None):
    """FP32 only; zero-weight branches never dereference their input/teacher."""
    raw={}; weighted={}
    if weights.get('cnn_aux',0):
        if cnn_aux_logits is None or cnn_aux_logits.shape!=logits.shape:
            raise ValueError('CNN auxiliary CE requires independent [B,C] logits')
        raw['cnn_aux']=F.cross_entropy(cnn_aux_logits.float(),labels)
    if weights['ce']:
        class_weights=weights.get('class_weights')
        class_weights=torch.as_tensor(class_weights,device=logits.device,dtype=torch.float32) if class_weights is not None else None
        raw['ce']=F.cross_entropy(logits.float(),labels,weight=class_weights)
    if weights['feature']:
        if target is None or feature.shape!=target['feature'].shape: raise ValueError('Feature target shape mismatch')
        student_f=feature.float(); teacher_f=target['feature'].detach().float()
        cosine=(1-F.cosine_similarity(student_f,teacher_f,dim=-1,eps=1e-6)).mean()
        if 'feature_cosine' not in weights:
            raw['feature']=cosine
        else:
            if weights['feature_cosine']: raw['feature_cosine']=cosine
            if weights['feature_mse']:
                if weights['feature_mse_space']=='normalized':
                    student_f=F.normalize(student_f,dim=-1); teacher_f=F.normalize(teacher_f,dim=-1)
                raw['feature_mse']=F.mse_loss(student_f,teacher_f)
    if weights['logits']:
        t=float(weights['temperature'])
        raw['logits']=F.kl_div(F.log_softmax(logits.float()/t,dim=-1),
            F.softmax(target['logits'].detach().float()/t,dim=-1),reduction='batchmean')*t*t
    if weights.get('text',0):
        if prototypes is None or prototypes.shape!=(logits.shape[-1],feature.shape[-1]):
            raise ValueError('Text prototypes must match class order/count and feature dimension')
        text_logits=weights['text_scale']*F.normalize(feature.float(),dim=-1)@F.normalize(prototypes.detach().float(),dim=-1).T
        raw['text']=F.cross_entropy(text_logits,labels)
    if weights.get('affinity',0):
        if target is None or feature.shape!=target['feature'].shape: raise ValueError('Affinity target shape mismatch')
        a=F.normalize(feature.float(),dim=-1); b=F.normalize(target['feature'].detach().float(),dim=-1)
        mask=~torch.eye(len(a),device=a.device,dtype=torch.bool)
        raw['affinity']=(a@a.T-b@b.T)[mask].square().mean() if len(a)>1 else a.sum()*0.
    if not raw: raise ValueError('No active objective')
    for k,v in raw.items():
        coefficient=weights[k]*weights['feature'] if k in ['feature_cosine','feature_mse'] else weights[k]
        weighted[k]=v*coefficient
    return sum(weighted.values()),raw,weighted


def gradient_report(weighted,parameter):
    """Compare objective gradients on one declared shared parameter, without updates."""
    gradients={}
    for name,loss in weighted.items():
        value=torch.autograd.grad(loss,parameter,retain_graph=True,allow_unused=True)[0]
        gradients[name]=None if value is None else value.detach().float().flatten()
    result={'parameter_scope':'fusion_projection.weight','norms':{},'cosines':{}}
    for name,value in gradients.items(): result['norms'][name]=0. if value is None else float(value.norm())
    names=list(gradients)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            a,b=gradients[left],gradients[right]
            result['cosines'][left+'__'+right]=(float(F.cosine_similarity(a,b,dim=0))
                if a is not None and b is not None and a.norm()>0 and b.norm()>0 else None)
    return result

def coefficients(method,stage):
    return {'ce':0. if stage=='s1' else 1.,'feature':1. if stage=='s1' else (.2 if method in ['full_v2','hybrid_feature_kd'] else 0.),
            'logits':.2 if method=='hybrid_logits_kd' else 0.,'temperature':2.,
            'au':0.,'weight_align':0.,'semantic':0.,'kl_direction':'teacher||student','feature_target':'CLIP_projected_global'}
