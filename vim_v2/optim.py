import math
import torch

def configure_stage(model,stage):
    model.train()
    for name,p in model.named_parameters():
        permanent=name in getattr(model,'permanently_frozen_names',set()) or name.startswith('text_to_classifier.') or (name.startswith('vim.') and '.norm_f.' in name)
        classification_only=name.startswith('classifier.') or name.startswith('cnn_residual_aux.cnn_aux_head.')
        p.requires_grad_(not permanent and not (stage=='s1' and classification_only))
        if not p.requires_grad: p.grad=None
    if hasattr(model,'compute_logits'): model.compute_logits=stage!='s1'
    if hasattr(model,'compute_aux_logits'):
        block=getattr(model,'cnn_residual_aux',None)
        model.compute_aux_logits=stage!='s1' and block is not None and block.cnn_aux_head is not None
    if stage=='s1': model.classifier.eval()


def branch_gradient_stats(model):
    """Unclipped gradient scale together with trainability and missing-gradient counts."""
    families={'cnn':('cnn_backbone.','cnn_projection.','cnn_lateral.','cnn_residual_aux.'),
              'vmamba':('vim.','backbone.'),'sasf':('sasf.',),'cnn_aux_head':('cnn_residual_aux.cnn_aux_head.',)}
    result={}
    named=list(model.named_parameters())
    for family,prefixes in families.items():
        parameters=[p for name,p in named if name.startswith(prefixes)]
        trainable=[p for p in parameters if p.requires_grad]
        gradients=[p.grad.detach().float().norm() for p in trainable if p.grad is not None]
        norm=float(torch.stack(gradients).norm()) if gradients else 0.
        result[family+'_grad_norm']=norm
        result[family]={'parameters':sum(p.numel() for p in parameters),
                        'trainable_parameters':sum(p.numel() for p in trainable),
                        'trainable_tensors':len(trainable),'gradient_none_tensors':sum(p.grad is None for p in trainable)}
    return result

def groups(model,cfg):
    declared=set()
    for prefix,module in model.named_modules():
        if hasattr(module,'no_weight_decay'):
            declared.update((prefix+'.' if prefix else '')+x for x in module.no_weight_decay())
    result={}
    for name,p in model.named_parameters():
        if name in getattr(model,'permanently_frozen_names',set()) or name.startswith('text_to_classifier.') or (name.startswith('vim.') and '.norm_f.' in name): continue
        backbone=name.startswith(('backbone.','vim.backbone.','cnn_backbone.'))
        no_decay=getattr(p,'_no_weight_decay',False) or name in declared or p.ndim<2 or any(x in name for x in ['pos_embed','cls_token','global_query'])
        key=('backbone' if backbone else 'new')+('_no_decay' if no_decay else '_decay')
        if key not in result:
            lr=cfg['backbone_lr'] if backbone else cfg['new_lr']
            result[key]={'params':[],'names':[],'name':key,'lr':lr,'peak_lr':lr,'weight_decay':0. if no_decay else cfg['weight_decay']}
        result[key]['params'].append(p); result[key]['names'].append(name)
    if len({id(p) for g in result.values() for p in g['params']})!=sum(len(g['params']) for g in result.values()): raise ValueError('Duplicate optimizer parameter')
    return list(result.values())

class UpdateCosine:
    def __init__(self,optimizer,total,warmup,min_ratio=.01):
        if total<2 or not 0<warmup<total: raise ValueError('Invalid cosine horizon')
        self.optimizer=optimizer; self.total=total; self.warmup=warmup; self.min_ratio=min_ratio; self.updates=0
    def factor(self,u):
        if u<self.warmup: return (u+1)/self.warmup
        progress=(u-self.warmup)/max(1,self.total-self.warmup-1)
        return self.min_ratio+(1-self.min_ratio)*(1+math.cos(math.pi*progress))/2
    def prepare(self):
        if self.updates>=self.total: raise ValueError('Update budget exhausted')
        for g in self.optimizer.param_groups: g['lr']=g['peak_lr']*self.factor(self.updates)
    def step(self): self.updates+=1
    def state_dict(self): return {k:getattr(self,k) for k in ['total','warmup','min_ratio','updates']}
    def load_state_dict(self,s):
        if any(s[k]!=getattr(self,k) for k in ['total','warmup','min_ratio']): raise ValueError('Resume horizon mismatch')
        self.updates=s['updates']

def probes(model):
    # One full representative tensor per actual backbone block / participating module.
    # Prefer in_proj weights to tiny A/dt gradients that can round to zero on a single step.
    selected={}; scores={}
    for name,p in model.named_parameters():
        family=name.split('.')[0]
        score=1
        if '.backbone.layers.' in name and '.blocks.' in name:
            parts=name.split('.backbone.layers.')[1].split('.')
            family='vmamba.stage_'+parts[0]+'.block_'+parts[2]
            score=100 if name.endswith('op.in_proj.weight') else 1
        elif '.core.layers.' in name:
            parts=name.split('.core.layers.')[1].split('.')
            family='vim.layer_'+parts[0]+'.'+parts[1]
            score=100 if name.endswith(('in_proj.weight','norm.weight')) else 1
        elif family in ['vim','backbone'] and '.backbone.patch_embed.' in name:
            family='vmamba.patch_embed'; score=100 if name.endswith('0.weight') else 1
        elif family in ['vim','backbone'] and '.backbone.layers.' in name:
            family='vmamba.stage_'+name.split('.backbone.layers.')[1].split('.')[0]+'.downsample'
            score=100 if name.endswith('1.weight') else 1
        elif family in ['vim','backbone']:
            family='vim.'+('patch_embed' if '.patch_embed.' in name else 'final_norm' if '.norm_f.' in name else 'positions' if 'pos_embed' in name else 'cls')
            score=100 if name.endswith('proj.weight') else 1
        elif family in ['state_projections','classifier','cnn_backbone']:
            family='.'.join(name.split('.')[:2]); score=100 if name.endswith(('weight','conv1.weight')) else 1
        if p.requires_grad and (family not in selected or score>scores[family]):
            selected[family]=(name,p); scores[family]=score
    return selected
