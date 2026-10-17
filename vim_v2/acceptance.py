import copy, json, time
from pathlib import Path
import torch
from .common import atomic_json,check_asset,fingerprint,sha,seed_all,tensor_digest
from .config import validate
from .models import make_student,make_teacher,strict_vim_load
from .data import loader,manifest,PairedDataset
from .runtime import gpu_lease,forward_loss,evaluate_model,student_from_init
from .optim import configure_stage,groups
from .checkpoint import atomic_save,load,rng_state,restore_rng

def compare_gradient_states(reference,actual):
    """Normwise FP32 equivalence, including near-zero gradients and exact coverage.

    Official CUDA backward reductions are not bitwise reproducible. Elementwise
    relative errors at cancellation zeros do not measure a tensor's gradient error.
    A hard 0.05% tensor-L2 bound still rejects altered checkpoint gradients.
    """
    if set(reference)!=set(actual): raise ValueError('Checkpointed gradient coverage differs')
    rows=[]
    for name,a in reference.items():
        b=actual[name]
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('Invalid gradient '+name)
        delta=(a.float()-b.float()); error=float(delta.norm()); scale=float(a.float().norm())
        if error>1e-7+5e-4*scale: raise ValueError('Checkpoint gradient L2 mismatch: '+name+' error='+str(error)+' reference='+str(scale))
        rows.append({'name':name,'difference_l2':error,'reference_l2':scale,'relative_l2':error/max(scale,1e-12),'max_abs':float(delta.abs().max())})
    return {'absolute_l2_tolerance':1e-7,'relative_l2_tolerance':5e-4,'max_relative_l2':max(r['relative_l2'] for r in rows),'tensors':rows}

def assert_exact_tree(actual,expected,path='state'):
    """Check recovered state before another nondeterministic GPU operation."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype!=expected.dtype or actual.shape!=expected.shape or not torch.equal(actual.detach().cpu(),expected.detach().cpu()):
            raise ValueError('Restored tensor differs: '+path)
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected): raise ValueError('Restored mapping differs: '+path)
        for key in expected: assert_exact_tree(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected,(tuple,list)):
        if type(actual)!=type(expected) or len(actual)!=len(expected): raise ValueError('Restored sequence differs: '+path)
        for index,(a,b) in enumerate(zip(actual,expected)): assert_exact_tree(a,b,path+'/'+str(index))
    elif actual!=expected: raise ValueError('Restored value differs: '+path)

def audit_init(c,gpu=False):
    # Does not require teachers: CE initialization auditing remains independent.
    check_asset(c['student']['pretrained']) if c['student']['initialization']=='pretrained' else None
    torch.set_num_threads(2); seed_all(42)
    model,report,init_sha=student_from_init(c); report=dict(report)
    report['init_snapshot_sha256']=init_sha; report['gpu_accepted']=False; report['config_hash']=fingerprint(c)
    pg=groups(model,c['optimizer']); report['parameter_groups']=[{k:v for k,v in g.items() if k!='params'} for g in pg]
    report['parameters']=sum(p.numel() for p in model.parameters())
    if gpu:
        with gpu_lease(Path(c['output']).parents[1]):
            vmamba=c['student']['architecture'].startswith('vmamba_')
            for size in [224,256]:
                if vmamba:
                    from open_clip.vision_mamba_clip.official_vmamba import create_official_vmamba
                    spec=c['student']['pretrained']; pretrained=c['student']['initialization']=='pretrained'
                    backbone,_=create_official_vmamba(checkpoint_path=spec['path'] if pretrained else None,
                        checkpoint_sha256=spec['sha256'] if pretrained else None,keep_final_norm=True)
                    backbone.cuda().eval(); x=torch.randn(2,3,size,size,device='cuda')
                    with torch.no_grad():
                        direct=backbone.backbone.patch_embed(x)
                        for layer in backbone.backbone.layers: direct=layer(direct)
                        expected=backbone.final_norm(direct).mean((2,3)); actual=backbone.forward_features(x)
                    report['native_'+str(size)]={'shape':list(actual.shape),'max_abs_error':float((actual-expected).abs().max()),'readout':'official final norm and spatial mean, no CLS'}
                else:
                    from open_clip.vision_mamba_clip.official_vim import create_official_vim
                    backbone,_=create_official_vim(img_size=size)
                    if c['student']['initialization']=='pretrained': strict_vim_load(backbone.core,c['student']['pretrained'])
                    backbone.cuda().eval(); x=torch.randn(2,3,size,size,device='cuda')
                    with torch.no_grad(): expected=backbone.core.forward_features(x); actual=backbone(x)
                    report['native_'+str(size)]={'shape':list(actual.shape),'max_abs_error':float((actual-expected).abs().max()),'cls_index':backbone.cls_index}
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
                del backbone,x
            # Test actual Hybrid hooks under activation checkpointing, including gradients.
            model.cuda().train(); x=next(iter(loader(c,'train')))['image_student'][:2].cuda()
            backbone=model.backbone if hasattr(model,'backbone') else (model.vim if vmamba else model.vim.backbone)
            saved_rng=rng_state(); states=[]; outputs=[]
            initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            for enabled in [False,False,True]:
                model.load_state_dict(initial); model.zero_grad(set_to_none=True); restore_rng(saved_rng); backbone.set_grad_checkpointing(enabled)
                r=model(x); y=r if torch.is_tensor(r) else r.logits; outputs.append(y.detach())
                y.float().square().mean().backward()
                states.append({k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None})
            torch.testing.assert_close(outputs[0],outputs[1],atol=1e-5,rtol=1e-4)
            torch.testing.assert_close(outputs[0],outputs[2],atol=1e-5,rtol=1e-4)
            report['uncheckpointed_repeat_gradient_error']=compare_gradient_states(states[0],states[1])
            report['checkpoint_gradient_error']=compare_gradient_states(states[0],states[2])
            report['gradient_equivalence_contract']='same gradient coverage and finite tensors; tensor L2 error <= 1e-7 + 0.0005 * reference L2; not bitwise CUDA backward equality'
            report['checkpoint_gradient_equivalence']=True; report['gpu_accepted']=True
    return report

def fit(c,max_updates=1000):
    if not 1<=max_updates<=1000: raise ValueError('Fit diagnostic capped at 1000 updates')
    root=Path(c['output']).parents[1]
    with gpu_lease(root):
        seed_all(42); m=manifest(c['data']); indices=[]
        for label in range(7): indices.extend([i for i,r in enumerate(m['samples']['train']) if r['label']==label][:8])
        if len(indices)!=56: raise ValueError('Need 8 training examples per class')
        model,report=make_student(c); model.cuda(); configure_stage(model,'s2')
        # Explicit diagnostic override only: no augmentation, KD, weight decay, or stochastic depth/dropout.
        for module in model.modules():
            if hasattr(module,'drop_prob'): module.drop_prob=0.
            if isinstance(module,torch.nn.Dropout): module.p=0.
        pg=groups(model,c['optimizer'])
        for g in pg: g['weight_decay']=0.
        opt=torch.optim.AdamW(pg,betas=(.9,.98),eps=1e-6)
        weights={'ce':1.,'feature':0.,'logits':0.,'temperature':2.}; amp=torch.cuda.is_bf16_supported(); device=torch.device('cuda')
        from torch.utils.data import DataLoader
        data=PairedDataset(m,'train',augment=False,indices=indices)
        batches=DataLoader(data,batch_size=32,shuffle=True,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(42))
        report={'config_hash':fingerprint(c),'samples':56,'sample_ids':[m['samples']['train'][i]['path'] for i in indices],
                'overrides':['no augmentation','no KD','no weight decay','no dropout/stochastic depth'],'curves':[],'accepted':False}
        update=0; begin=time.perf_counter()
        while update<max_updates:
            model.train()
            for b in batches:
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): loss,_,_,_=forward_loss(model,None,b,weights,device)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite CE fit loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); update+=1
                if update>=max_updates: break
            result=evaluate_model(model,DataLoader(data,batch_size=32,shuffle=False),device,amp)
            report['curves'].append({'update':update,**result})
            if result['accuracy']>=.95: report['accepted']=True; break
        report['updates']=update; report['seconds']=time.perf_counter()-begin
        report['final_train_accuracy']=result['accuracy']; return report

def audit_teacher(c,stage):
    from open_clip import get_tokenizer
    from .common import read,PROJECT,CLASSES
    from torch.utils.data import DataLoader
    root=Path(c['output']).parents[1]
    with gpu_lease(root):
        model=make_teacher(c,stage,torch.device('cuda')); model.eval(); amp=torch.cuda.is_bf16_supported()
        m=manifest(c['data']); batches=loader(c,'val'); matrix=torch.zeros(7,7,dtype=torch.long); total=0.; count=0
        anchors=None
        if stage=='t1':
            schema=read(PROJECT/'configs/rafdb_au_prompts.json')
            if schema['class_order']!=CLASSES: raise ValueError('Teacher prototype classes mismatch')
            tokens=get_tokenizer('ViT-L-14')([schema['prompts'][name] for name in CLASSES]).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): anchors=torch.nn.functional.normalize(model.encode_text(tokens).float(),dim=-1)
        payload=load(c['teachers'][stage]['path']); identity=payload['identity']; expected=payload['selection_metrics']; cw=identity.get('class_weights')
        if stage=='t1' and sha(PROJECT/'configs/rafdb_au_prompts.json')!=identity['prompt_sha256']:
            raise ValueError('Teacher T1 prototype source SHA mismatch')
        weights=torch.tensor(cw,device='cuda') if cw else None; denominator=0.
        with torch.no_grad():
            for b in batches:
                y=b['label'].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                    feature=model.encode_image(b['image_teacher'].cuda())
                    logits=model.classifier(feature) if stage=='t2' else model.logit_scale.clamp(max=__import__('math').log(100)).exp()*torch.nn.functional.normalize(feature.float(),dim=-1)@anchors.T
                loss=torch.nn.functional.cross_entropy(logits.float(),y,weight=weights)
                den=len(y) if weights is None else float(weights[y].sum()); total+=float(loss)*den; denominator+=den; count+=len(y)
                matrix+=torch.bincount((y*7+logits.argmax(-1)).cpu(),minlength=49).reshape(7,7)
        from .runtime import metrics
        result=metrics(matrix); result['loss']=total/denominator
        # Hardware/AMP kernels can alter boundary predictions; expose differences and require review if >0.5 pp.
        delta={k:result[k]-expected[k] for k in ['accuracy','uar','macro_f1','loss']}
        return {'accepted':all(abs(delta[k])<=.005 for k in ['accuracy','uar','macro_f1']) and abs(delta['loss'])<=max(.001,.005*abs(expected['loss'])),
                'loss_tolerance':'max(0.001, 0.005 * abs(saved_loss))','teacher_sha256':c['teachers'][stage]['sha256'],
                'manifest_sha256':c['data']['sha256'],'stage':stage,'samples':count,'metrics':result,'saved_metrics':expected,'delta':delta,'output_dim':model.output_dim}

def audit_resume(c):
    """Two real training batches: uninterrupted vs disk save/reconstruct/replay."""
    from .optim import UpdateCosine
    from .checkpoint import restore
    root=Path(c['output']).parents[1]
    with gpu_lease(root):
        seed_all(42); model,_=make_student(c); model.cuda(); configure_stage(model,'s2')
        opt=torch.optim.AdamW(groups(model,c['optimizer']),betas=(.9,.98),eps=1e-6)
        schedule=UpdateCosine(opt,10,1); scaler=torch.amp.GradScaler('cuda',enabled=False)
        identity={'config_hash':fingerprint(c),'test':'GPU_real_two_step_state_restore'}; device=torch.device('cuda')
        data=loader(c,'train'); iterator=iter(data); first=next(iterator); second=next(iterator)
        amp=torch.cuda.is_bf16_supported(); weights={'ce':1.,'feature':0.,'logits':0.,'temperature':2.}
        def step(m,o,s,b):
            s.prepare(); o.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): loss,_,_,_=forward_loss(m,None,b,weights,device)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); o.step(); s.step()
            return float(loss)
        step(model,opt,schedule,first)
        p=root/'acceptance'/'resume_scratch.pt'
        atomic_save(p,{'kind':'resume','identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),
                      'scheduler':schedule.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'next_batch':0,'global_update':1})
        expected_loss=step(model,opt,schedule,second)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        del model,opt,schedule; torch.cuda.empty_cache()
        fresh,_=make_student(c,initialize=False); fresh.cuda(); configure_stage(fresh,'s2')
        opt=torch.optim.AdamW(groups(fresh,c['optimizer']),betas=(.9,.98),eps=1e-6); schedule=UpdateCosine(opt,10,1)
        saved=load(p); restore(saved,identity,fresh,opt,schedule,scaler)
        assert_exact_tree(fresh.state_dict(),saved['model'],'model')
        assert_exact_tree(opt.state_dict(),saved['optimizer'],'optimizer')
        assert_exact_tree(schedule.state_dict(),saved['scheduler'],'scheduler')
        assert_exact_tree(scaler.state_dict(),saved['scaler'],'scaler')
        assert_exact_tree(rng_state(),saved['rng'],'rng')
        actual_loss=step(fresh,opt,schedule,second)
        if abs(actual_loss-expected_loss)>1e-4*(1+abs(expected_loss)): raise ValueError('Resumed loss differs beyond numerical tolerance')
        max_error=0.; error_squared=0.; update_squared=0.
        for k,v in fresh.state_dict().items():
            actual=v.cpu(); error=float((actual-expected[k]).abs().max()); max_error=max(max_error,error)
            if v.is_floating_point():
                error_squared+=float((actual.double()-expected[k].double()).square().sum())
                update_squared+=float((expected[k].double()-saved['model'][k].double()).square().sum())
            torch.testing.assert_close(actual,expected[k],atol=1e-4,rtol=1e-5)
        p.unlink()
        return {'accepted':True,'config_hash':fingerprint(c),'max_parameter_error':max_error,'loss_before':expected_loss,'loss_resumed':actual_loss,
                'exact_restored_state_fields':['model','optimizer (including groups and all Adam moments)','scheduler','scaler','Python/NumPy/CPU/CUDA RNG'],
                'continuation_tolerance':{'parameter_atol':1e-4,'parameter_rtol':1e-5,'loss_atol':1e-4,'loss_rtol':1e-4},
                'relative_global_update_l2_error':(error_squared/max(update_squared,1e-30))**.5,
                'contract':'exact saved-state restoration before continuation; epoch-boundary production only; subsequent CUDA/BF16 operations are numerically checked, not bitwise reproducible even on the same GPU; no mid-epoch exact guarantee'}
