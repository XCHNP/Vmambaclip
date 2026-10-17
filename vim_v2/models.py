"""Official native CLS readout and the existing cloud Hybrid, without replacing legacy code."""
import argparse, contextlib
import torch
from torch import nn
from .common import MODEL, check_asset, tensor_digest

VIM_SHA='aae4583e2def6389b66cfaf292cfade47a873182a39adbec19ec55b730ea9fe3'
CNN_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'

@contextlib.contextmanager
def official_safe_globals():
    # PyTorch 2.4 exposes add/get/clear, but not the 2.5 context manager.
    previous=torch.serialization.get_safe_globals()
    torch.serialization.add_safe_globals([argparse.Namespace])
    try: yield
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)

def interpolate_middle(pos,target_patches):
    if pos.ndim!=3 or pos.shape[0]!=1 or pos.shape[2]!=384: raise ValueError('Not Vim Small positions')
    n=pos.shape[1]-1; side=int(n**.5); dest=int(target_patches**.5)
    if n!=196 or side*side!=n or dest*dest!=target_patches: raise ValueError('Expected stride16 224 pretrained grid')
    c=n//2; special=pos[:,c:c+1]
    patches=torch.cat([pos[:,:c],pos[:,c+1:]],1)
    if n==target_patches: return pos.clone()
    patches=patches.reshape(1,side,side,384).permute(0,3,1,2)
    patches=torch.nn.functional.interpolate(patches.float(),size=(dest,dest),mode='bicubic',align_corners=False)
    patches=patches.permute(0,2,3,1).reshape(1,target_patches,384).to(pos.dtype)
    c=target_patches//2
    return torch.cat([patches[:,:c],special,patches[:,c:]],1)

def strict_vim_load(core,spec):
    path=check_asset(spec)
    # Provenance is pinned in addition to shape checks (stride is not a tensor shape).
    if spec['sha256']!=VIM_SHA or spec.get('model_name')!=MODEL:
        raise ValueError('Not the approved official small/middle-CLS/bimambav2/stride16 checkpoint')
    with official_safe_globals():
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    containers=[k for k in ['model','state_dict'] if k in payload]
    if len(containers)>1: raise ValueError('Ambiguous checkpoint containers')
    state=dict(payload[containers[0]] if containers else payload)
    if state and all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    dropped=[]
    for k in ['head.weight','head.bias']:
        if k in state:
            expected=(1000,384) if k.endswith('weight') else (1000,)
            if tuple(state[k].shape)!=expected: raise ValueError('Unexpected ImageNet head shape')
            dropped.append(k); del state[k]
    expected=core.state_dict()
    if set(state)!=set(expected): raise ValueError('Backbone keys mismatch: '+str({'missing':sorted(set(expected)-set(state)),'unexpected':sorted(set(state)-set(expected))}))
    old=list(state['pos_embed'].shape)
    state['pos_embed']=interpolate_middle(state['pos_embed'],core.patch_embed.num_patches)
    for k,v in state.items():
        if v.shape!=expected[k].shape or not torch.isfinite(v).all(): raise ValueError('Invalid backbone tensor '+k)
    core.load_state_dict(state,strict=True)
    return {'success':True,'source':spec,'dropped_keys':dropped,'key_count':len(state),
            'matched_elements':sum(v.numel() for v in state.values()),'coverage':1.,
            'position_conversion':{'before':old,'after':list(state['pos_embed'].shape),'source_cls':98,
                                   'target_cls':core.patch_embed.num_patches//2,'method':'bicubic align_corners=False patches only'}}

def make_student(cfg,initialize=True):
    from open_clip.vision_mamba_clip.official_vim import create_official_vim,OfficialPatchAdapter,MambaCE
    from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent,ResNet18ThroughLayer3
    from torchvision.models import resnet18
    student=cfg['student']
    # Same official random constructor and independent new-module RNG for all comparisons.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'])
        backbone,meta=create_official_vim(img_size=student['input_size'],drop_path_rate=student['drop_path_rate'])
    if meta['commit']!=student['source_revision']: raise ValueError('Official source revision differs from configuration')
    core=backbone.core
    if (core.embed_dim!=384 or len(core.layers)!=24 or tuple(core.patch_embed.proj.stride)!=(16,16)
        or tuple(core.patch_embed.proj.kernel_size)!=(16,16) or not core.use_middle_cls_token):
        raise ValueError('Official factory variant mismatch')
    report={'initialization':student['initialization'],'native_readout':'official norm_f then middle CLS (384)',
            'layer_indices':student['layer_indices'],'hybrid_readout':'post-block mixer_output + residual, CLS excluded; unnormalized; not final norm',
            'official':meta}
    if initialize and student['initialization']=='pretrained': report['vim']=strict_vim_load(core,student['pretrained'])
    meta=dict(meta)
    meta['pretrained_loaded']=bool(initialize and student['initialization']=='pretrained')
    meta['initialization_source']=student['pretrained']['path'] if meta['pretrained_loaded'] else ('strict student reconstruction; weights supplied by checkpoint' if not initialize else 'official random initialization, seed42')
    meta['backbone_initialization_sha256']=tensor_digest(core.state_dict())
    meta['digest_method']='sorted key + dtype/shape + tensor bytes (V2)'
    report['official']=meta
    backbone.set_grad_checkpointing(student['gradient_checkpointing'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['head_seed'])
        if student['architecture']=='vim_native_ce': model=MambaCE(backbone)
        else:
            cnn=ResNet18ThroughLayer3()
            if initialize:
                if student['cnn_pretrained']['sha256']!=CNN_SHA: raise ValueError('Expected torchvision ImageNet ResNet18 V1 checkpoint')
                path=check_asset(student['cnn_pretrained']); source=resnet18(weights=None)
                source.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                cnn.stem=nn.Sequential(source.conv1,source.bn1,source.relu,source.maxpool)
                cnn.layer1=source.layer1; cnn.layer2=source.layer2; cnn.layer3=source.layer3
                del source
                report['cnn']={'source':student['cnn_pretrained'],'strict_full_resnet18_load':True,'retained':'stem, layer1, layer2, layer3','excluded':'layer4, ImageNet fc'}
            # Reset separately: CNN weight loading must not shift fusion/classifier RNG.
            torch.random.default_generator.manual_seed(cfg['head_seed'])
            adapter=OfficialPatchAdapter(backbone)
            if list(adapter.layer_indices)!=student['layer_indices']: raise ValueError('Hook layers changed')
            model=HybridVisionMambaStudent(adapter,{'e':384,'m':384,'l':384},common_dim=768,num_classes=7,attention_heads=8,cnn_backbone=cnn)
            model.text_to_classifier.requires_grad_(False)
            core.norm_f.requires_grad_(False)  # unused by legacy Hybrid hook readout
    report['classifier_sha256']=tensor_digest(model.classifier.state_dict())
    report['initial_state_sha256']=tensor_digest(model.state_dict())
    return model,report

def validate_teacher(cfg,stage):
    from .checkpoint import load
    from .common import read
    spec=cfg['teachers'][stage]
    status=read(check_asset(spec['completion']))
    if status.get('state')!='completed': raise ValueError('Teacher training has not completed; existing best is provisional')
    expected_updates=cfg['schedule']['steps_per_epoch']*{'t1':5,'t2':10}[stage]
    if status.get('epochs')!={'t1':5,'t2':10}[stage] or status.get('updates')!=expected_updates: raise ValueError('Teacher completed budget mismatch')
    path=check_asset(spec); payload=load(path); identity=payload['identity']
    if (identity['dataset']!=cfg['dataset'] or identity['variant']!='teacher_'+stage
        or identity['manifest_hash']!=cfg['data']['manifest_hash'] or identity['class_order']!=cfg['class_order']
        or identity.get('smoke_only_original_teacher') or identity['architecture']['checkpoint_sha256']!=cfg['clip_base']['sha256']
        or identity['architecture']['identity']!='original_CLIP_ViT_L14_QuickGELU_projected768_img256'):
        raise ValueError('Teacher identity incompatible')
    if identity.get('epochs')!={'t1':5,'t2':10}[stage]: raise ValueError('Unexpected teacher training budget')
    if stage=='t2' and (identity.get('teacher_source')!=cfg['dataset'] or identity.get('teacher_best_sha256')!=cfg['teachers']['t1']['sha256']):
        raise ValueError('T2 parent SHA/domain does not match registered T1')
    check_asset(cfg['clip_base'])
    return payload

def make_teacher(cfg,stage,device):
    from open_clip import create_model
    from open_clip.vision_mamba_clip.adapters import TeacherAdapter
    payload=validate_teacher(cfg,stage)
    core=create_model('ViT-L-14',pretrained=cfg['clip_base']['path'],force_quick_gelu=True,force_image_size=256,device='cpu',precision='fp32')
    teacher=TeacherAdapter(core,7)
    required={k for k in teacher.state_dict() if k.startswith(('clip_model.visual.','classifier.')) or k=='clip_model.logit_scale'}
    if set(payload['model'])!=required: raise ValueError('Teacher visual/head keys mismatch')
    result=teacher.load_state_dict(payload['model'],strict=False)
    if result.unexpected_keys or required.intersection(result.missing_keys): raise ValueError('Teacher strict load failed')
    if teacher.output_dim!=768: raise ValueError('Actual teacher projection dimension differs from 768')
    teacher.requires_grad_(False); teacher.eval()
    return teacher.to(device)
