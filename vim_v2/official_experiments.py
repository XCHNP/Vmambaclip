"""Build all experiment variants from the same official backbones and student code."""

import copy
import hashlib

import torch
from torch import nn

from .common import MODEL, check_asset, tensor_digest


def student_spec(variation,paths):
    from open_clip.vision_mamba_clip.official_vmamba import VMAMBA_COMMIT,VMAMBA_MODEL
    family=variation['backbone_family']; cnn=variation['variant']!='mamba_only'
    return dict(copy.deepcopy(variation),architecture='experimental_student',common_dim=768,input_size=256,
                initialization='pretrained',drop_path_rate=.1,gradient_checkpointing=True,
                official_model_name={'vim':MODEL,'vmamba':VMAMBA_MODEL,'cnn':None}[family],
                source_revision={'vim':'dd0358ad1e42701f22afbefa0717cc8825cf9f45','vmamba':VMAMBA_COMMIT,'cnn':None}[family],
                pretrained=copy.deepcopy(paths.get(family)) if family!='cnn' else None,
                cnn_pretrained=copy.deepcopy(paths.get('cnn')) if cnn else None,
                patch_dropout_scope='cnn_kv_only')


def validate_student_spec(s):
    from open_clip.vision_mamba_clip.official_vmamba import VMAMBA_COMMIT,VMAMBA_MODEL
    expected={'backbone_family','variant','use_sasf','sasf_stage','cnn_multiscale','query_mode','patch_dropout','fusion',
              'architecture','common_dim','input_size','initialization','drop_path_rate','gradient_checkpointing',
              'official_model_name','source_revision','pretrained','cnn_pretrained','patch_dropout_scope'}
    if set(s)!=expected: raise ValueError('Unknown/missing experimental student fields')
    if s['architecture']!='experimental_student' or s['common_dim']!=768 or s['input_size']!=256 or s['initialization']!='pretrained': raise ValueError('Unsupported student dimensions/initialization')
    if s['backbone_family'] not in ['cnn','vim','vmamba'] or s['variant'] not in ['hybrid','cnn_only','mamba_only']: raise ValueError('Unknown experiment backbone/variant')
    if (s['backbone_family']=='cnn')!=(s['variant']=='cnn_only'): raise ValueError('CNN-only must not instantiate an SSM')
    if s['variant']=='cnn_only' and s['use_sasf']: raise ValueError('CNN-only cannot use SASF')
    expected_fusion={'hybrid':'mamba_cnn_cross_attention','cnn_only':'none','mamba_only':'concat_projection'}[s['variant']]
    if s['fusion']!=expected_fusion: raise ValueError('Fusion does not match the declared branch variant')
    if s['variant']=='mamba_only' and (s['cnn_multiscale'] or s['cnn_pretrained'] is not None): raise ValueError('Mamba-only must not declare CNN modules or weights')
    if s['variant']=='cnn_only' and s['pretrained'] is not None: raise ValueError('CNN-only must not declare SSM weights')
    if s['sasf_stage'] not in ['e','m','l'] or s['query_mode'] not in ['single','multi']: raise ValueError('Invalid stage/query mode')
    if type(s['use_sasf']) is not bool or type(s['cnn_multiscale']) is not bool: raise ValueError('Module switches must be boolean')
    if not 0<=s['patch_dropout']<1 or s['patch_dropout_scope']!='cnn_kv_only': raise ValueError('Unsupported patch dropout')
    if s['drop_path_rate']!=.1 or s['gradient_checkpointing'] is not True: raise ValueError('Unreviewed backbone training settings')
    model={'vim':MODEL,'vmamba':VMAMBA_MODEL,'cnn':None}[s['backbone_family']]
    revision={'vim':'dd0358ad1e42701f22afbefa0717cc8825cf9f45','vmamba':VMAMBA_COMMIT,'cnn':None}[s['backbone_family']]
    if s['official_model_name']!=model or s['source_revision']!=revision: raise ValueError('Official backbone identity mismatch')
    return s


def stable_head_initialization(model,seed):
    """Common head tensors stay identical when optional modules are added/removed."""
    for name,module in model.named_modules():
        if name.startswith(('vim.','cnn_backbone.')): continue
        reset=getattr(module,'reset_parameters',None)
        if isinstance(module,nn.MultiheadAttention): reset=module._reset_parameters
        if reset is None: continue
        module_seed=int.from_bytes(hashlib.sha256((str(seed)+'/'+name).encode()).digest()[:8],'little')%(2**63)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(module_seed); reset()
    if hasattr(model,'global_query'):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            nn.init.normal_(model.global_query,std=model.common_dim**-.5)
    if hasattr(model,'text_to_classifier'):
        with torch.no_grad(): model.text_to_classifier.weight.copy_(torch.eye(model.common_dim))


def build_student(c,initialize=True):
    from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent,CNNOnlyStudent,ResNet18ThroughLayer3
    from .models import CNN_SHA,strict_vim_load
    s=validate_student_spec(c['student']); family=s['backbone_family']
    adapter=None; meta={'identity':'torchvision_resnet18_layer3','pretrained_loaded':False}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(c['seed'])
        if family=='vmamba':
            from open_clip.vision_mamba_clip.official_vmamba import create_official_vmamba
            spec=s['pretrained']
            if initialize and (not isinstance(spec,dict) or spec.get('model_name')!=s['official_model_name']): raise ValueError('Missing/mismatched VMamba pretrained asset')
            adapter,meta=create_official_vmamba(
                checkpoint_path=check_asset(spec) if initialize else None,
                checkpoint_sha256=spec['sha256'] if initialize else None,drop_path_rate=s['drop_path_rate'],
                gradient_checkpointing=s['gradient_checkpointing'])
            dims=adapter.state_dims
        elif family=='vim':
            from open_clip.vision_mamba_clip.official_vim import create_official_vim,OfficialPatchAdapter
            backbone,meta=create_official_vim(img_size=s['input_size'],drop_path_rate=s['drop_path_rate'])
            if initialize: meta['load_report']=strict_vim_load(backbone.core,s['pretrained'])
            meta['pretrained_loaded']=bool(initialize)
            backbone.set_grad_checkpointing(s['gradient_checkpointing'])
            adapter=OfficialPatchAdapter(backbone); dims={'e':384,'m':384,'l':384}
        if family!='cnn' and meta.get('commit')!=s['source_revision']:
            raise ValueError('Official backbone source revision differs from experiment')
        cnn=None
        if s['variant']!='mamba_only':
            cnn=ResNet18ThroughLayer3(return_multiscale=s['cnn_multiscale'])
            if initialize:
                from torchvision.models import resnet18
                spec=s['cnn_pretrained']
                if not isinstance(spec,dict) or spec.get('sha256')!=CNN_SHA: raise ValueError('Missing/mismatched ResNet18 V1 checkpoint')
                source=resnet18(weights=None)
                source.load_state_dict(torch.load(check_asset(spec),map_location='cpu',weights_only=True),strict=True)
                cnn.stem=nn.Sequential(source.conv1,source.bn1,source.relu,source.maxpool)
                cnn.layer1=source.layer1; cnn.layer2=source.layer2; cnn.layer3=source.layer3
                del source
        torch.random.default_generator.manual_seed(c['head_seed'])
        if family=='cnn':
            model=CNNOnlyStudent(cnn,multiscale=s['cnn_multiscale'])
        else:
            model=HybridVisionMambaStudent(adapter,dims,768,7,cnn_backbone=cnn,use_cnn=s['variant']!='mamba_only',
                use_sasf=s['use_sasf'],sasf_stage=s['sasf_stage'],cnn_multiscale=s['cnn_multiscale'],
                query_mode=s['query_mode'],patch_dropout=s['patch_dropout'],fusion=s['fusion'])
        stable_head_initialization(model,c['head_seed'])
    frozen=set()
    for name,p in model.named_parameters():
        if name.startswith('text_to_classifier.') or (family=='vim' and '.norm_f.' in name):
            p.requires_grad_(False); frozen.add(name)
    model.permanently_frozen_names=frozen
    report={'official':meta,'initialization':s['initialization'] if initialize else 'checkpoint_reconstruction_only',
            'cnn_pretrained_loaded':bool(initialize and cnn is not None),'student':s,
            'classifier_sha256':tensor_digest(model.classifier.state_dict()),
            'initial_state_sha256':tensor_digest(model.state_dict()),
            'head_rng_policy':'stable per module name; optional modules do not shift common head initialization'}
    return model,report
