"""Read-only selection-set branch interventions; no retraining or final-set tuning."""

from pathlib import Path

import torch

from .checkpoint import load
from .common import fingerprint,sha

INTERVENTIONS=('full','no_cnn','no_mamba','no_sasf','shuffle_cnn','shuffle_cnn_values')


@torch.no_grad()
def branch_batches(model,batches,device):
    from .runtime import metrics
    modes={module:module.training for module in model.modules()}; model.eval()
    matrices={name:torch.zeros(7,7,dtype=torch.long) for name in INTERVENTIONS}
    differences={name:0. for name in INTERVENTIONS}; norms={name:{} for name in INTERVENTIONS}; count=0
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(42)
            for batch in batches:
                image=batch['image_student'].to(device); y=batch['label'].cpu(); count+=len(y)
                reference=None
                for name in INTERVENTIONS:
                    output=model(image,intervention=name)
                    probabilities=output.logits.float().softmax(-1).cpu()
                    if reference is None: reference=probabilities
                    matrices[name]+=torch.bincount(y*7+probabilities.argmax(-1),minlength=49).reshape(7,7)
                    differences[name]+=float((probabilities-reference).abs().sum())
                    for key,value in (output.diagnostics or {}).items():
                        norms[name][key]=norms[name].get(key,0.)+float(value)*len(y)
        if not count: raise ValueError('Empty branch audit split')
        result={name:metrics(matrix) for name,matrix in matrices.items()}
        for name in INTERVENTIONS:
            result[name]['probability_l1_per_sample']=differences[name]/count
            result[name]['branch_statistics']={key:value/count for key,value in norms[name].items()}
            result[name]['delta_vs_full']={key:result[name][key]-result['full'][key] for key in ['accuracy','uar','macro_f1']}
        return {'interventions':result,'samples':count,'weights_updated':False,'split':'selection_val',
                'shuffle_cnn_interpretation':'Joint K/V permutation is an order-invariance control without positional encoding; invariance does not show CNN is unused.',
                'shuffle_cnn_values_interpretation':'Permutes V relative to K to test content correspondence, not anatomical landmark alignment.',
                'masking_interpretation':'Zero contributions at fusion and classifier bypasses; distribution shift prevents interpreting score deltas as causal effect sizes.'}
    finally:
        for module,mode in modes.items(): module.training=mode


def diagnose_branches(c,device='cpu'):
    from .config import validate
    from .data import loader
    from .models import make_student
    validate(c); path=Path(c['output'])/'best.pt'; payload=load(path)
    if payload['identity']['config_sha256']!=fingerprint(c): raise ValueError('Diagnostic checkpoint/configuration mismatch')
    model,_=make_student(c,initialize=False); model.load_state_dict(payload['model'],strict=True); model.to(device)
    if hasattr(model,'vim'):
        encoder=model.vim
        if hasattr(encoder,'set_grad_checkpointing'): encoder.set_grad_checkpointing(False)
        elif hasattr(encoder,'backbone'): encoder.backbone.set_grad_checkpointing(False)
    result=branch_batches(model,loader(c,'val'),torch.device(device))
    result.update(config_hash=fingerprint(c),checkpoint_sha256=sha(path))
    return result


def audit_semantics(c,teacher_stage='t2',device='cpu'):
    from .models import make_teacher
    from .semantics import audit_semantic_batches,load_prototypes
    from .data import loader
    teacher=make_teacher(c,teacher_stage,torch.device(device)); bank=load_prototypes(c,device)
    result=audit_semantic_batches(teacher,loader(c,'val'),bank,torch.device(device))
    result.update(accepted=True,acceptance_meaning='Audit executed with verified assets; no automatic semantic compatibility threshold.',
                  config_hash=fingerprint(c),teacher_sha256=c['teachers'][teacher_stage]['sha256'],
                  prototypes_sha256=c['text']['prototypes']['sha256'],split='selection_val')
    return result
