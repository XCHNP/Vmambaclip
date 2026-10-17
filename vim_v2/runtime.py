from __future__ import annotations
import contextlib, csv, json, os, shutil, subprocess, time, traceback
from pathlib import Path
import torch
from .common import atomic_json,check_asset,code_identity,environment,fingerprint,read,seed_all,sha,tensor_digest
from .checkpoint import atomic_save,load,rng_state,restore,record_stage_transition
from .config import validate
from .data import loader,manifest
from .losses import objectives
from .models import make_student,make_teacher
from .optim import configure_stage,groups,probes,UpdateCosine

@contextlib.contextmanager
def gpu_lease(root):
    if not torch.cuda.is_available(): raise RuntimeError('BLOCKED: no CUDA device; GPU acceptance/training has not run')
    import fcntl
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    with (root/'gpu0.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Detect training outside this runtime too. Never kill or preempt it.
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        others=[int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
        if others: raise RuntimeError('GPU occupied; retry later. PIDs='+str(others))
        f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

def init_identity(c):
    return {'init_id':c['init_id'],'student':c['student'],'seed':c['seed'],'head_seed':c['head_seed'],'code':c['code']}

def student_from_init(c):
    identity=init_identity(c); p=Path(c['init_path'])
    model,report=make_student(c,initialize=not p.exists())
    if p.exists():
        payload=load(p)
        if payload['kind']!='init' or payload['identity']!=identity: raise ValueError('Shared initial snapshot identity mismatch')
        model.load_state_dict(payload['model'],strict=True); report=payload['report']
        if tensor_digest(model.state_dict())!=report['initial_state_sha256']: raise ValueError('Shared init corrupted')
    else: atomic_save(p,{'kind':'init','identity':identity,'model':model.state_dict(),'report':report})
    return model,report,sha(p)

def split_stage(c,epoch):
    n=0
    for s in c['stages']:
        n+=s['epochs']
        if epoch<n: return s
    raise ValueError('Epoch beyond schedule')

def forward_loss(model,teacher,batch,weights,device):
    image=batch['image_student'].to(device,non_blocking=True); labels=batch['label'].to(device,non_blocking=True)
    output=model(image)
    logits=output if isinstance(output,torch.Tensor) else output.logits
    feature=None if isinstance(output,torch.Tensor) else output.fused_raw
    target=None
    if weights['feature'] or weights['logits'] or weights.get('affinity',0):
        if teacher is None: raise ValueError('Active distillation requires verified teacher')
        teacher.eval()
        with torch.no_grad():
            feature_t=teacher.encode_image(batch['image_teacher'].to(device,non_blocking=True))
            target={'feature':feature_t}
            if weights['logits']: target['logits']=teacher.classifier(feature_t)
    # Autocast explicitly disabled for numerically sensitive FP32 loss arithmetic.
    prototypes=getattr(model,'_text_prototypes',None) if weights.get('text',0) else None
    with torch.autocast(device.type,enabled=False): loss,raw,weighted=objectives(logits,feature,labels,weights,target,prototypes)
    return loss,logits,raw,weighted

def metrics(matrix):
    x=matrix.double(); tp=x.diag(); support=x.sum(1)
    if (support==0).any(): raise ValueError('Evaluation missing classes; fixed 7-class UAR cannot silently change denominator')
    precision=tp/x.sum(0).clamp_min(1); recall=tp/support; f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    return {'accuracy':float(tp.sum()/x.sum()),'uar':float(recall.mean()),'macro_f1':float(f1.mean()),
            'per_class':[{'class_index':i,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(7)],
            'confusion_matrix':matrix.tolist(),'predicted_counts':x.sum(0).long().tolist(),'unit':'fraction'}

@torch.no_grad()
def evaluate_model(model,batches,device,amp=False,teacher=None,s1_loss=None,predictions=None,max_batches=None):
    modes={m:m.training for m in model.modules()}; model.eval(); matrix=torch.zeros(7,7,dtype=torch.long); loss_sum=0.; seen=0
    f=Path(predictions).open('w',newline='') if predictions else None
    writer=csv.writer(f) if f else None
    if writer: writer.writerow(['sample_id','true_label','predicted_label']+[f'p{i}' for i in range(7)])
    started=time.perf_counter()
    try:
        for index,b in enumerate(batches):
            if max_batches is not None and index>=max_batches: break
            weights=s1_loss or {'ce':1.,'feature':0.,'logits':0.,'temperature':2.}
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                loss,logits,_,_=forward_loss(model,teacher,b,weights,device)
            n=len(b['label']); seen+=n; loss_sum+=float(loss)*n
            if logits is not None and s1_loss is None:
                probs=logits.float().softmax(-1).cpu(); predicted=probs.argmax(-1)
                matrix+=torch.bincount(b['label']*7+predicted,minlength=49).reshape(7,7)
                if writer: writer.writerows([sid,int(y),int(p),*pr] for sid,y,p,pr in zip(b['sample_id'],b['label'],predicted,probs.tolist()))
        if not seen: raise ValueError('Empty evaluation')
        result={'loss':loss_sum/seen,'samples':seen,'seconds':time.perf_counter()-started}
        if s1_loss is None:
            if max_batches is None: result.update(metrics(matrix))
            else: result['accuracy']=float(matrix.diag().sum()/matrix.sum())
        return result
    finally:
        if f: f.close()
        for m,mode in modes.items(): m.training=mode

def require_lock(c,plan_path):
    p=Path(plan_path); plan=read(p); lock=read(p.with_name('plan.lock'))
    if lock['plan_sha256']!=sha(p) or lock['code']!=code_identity(): raise ValueError('Locked plan/code changed')
    row=next(r for r in plan['runs'] if r['run_id']==c['run_id'])
    if row['config_hash']!=fingerprint(c): raise ValueError('Config differs from locked plan')
    if c['run_id'] not in lock['qualified_runs']: raise ValueError('This run has not passed acceptance')
    # Acceptance reports are part of the lock; deletion/mutation invalidates it.
    for asset in lock['evidence']: check_asset(asset)
    return lock

def run(c,plan_path=None,resume=False,smoke=False,max_updates=4):
    validate(c,assets=True)
    if smoke and not 2<=max_updates<=20: raise ValueError('Smoke requires 2..20 updates per stage')
    if not smoke:
        require_lock(c,plan_path)
        smoke_report=read(Path(c['output']).parents[1]/'acceptance'/c['run_id'].replace('/','__')/'smoke.json')
        estimate=smoke_report['parameters']*4*(11 if len(c['stages'])==2 else 7)
        if shutil.disk_usage(Path(c['output']).parents[1]).free<estimate+4*1024**3: raise RuntimeError('Insufficient disk for atomic checkpoint reserve')
    root=Path(c['output']).parents[1]
    with gpu_lease(root):
        return _run(c,root,resume,smoke,max_updates)

def _run(c,root,resume,smoke,max_updates):
    torch.set_num_threads(8); seed_all(c['seed']); torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0'); amp=torch.cuda.is_bf16_supported()
    out=root/'acceptance'/c['run_id'].replace('/','__') if smoke else Path(c['output'])
    out.mkdir(parents=True,exist_ok=True)
    if not smoke and not resume and any((out/x).exists() for x in ['last.pt','best.pt','effective_config.json']): raise FileExistsError('Use resume; run directory already exists')
    if resume and not (out/'last.pt').exists(): raise FileNotFoundError('Resume requires complete last.pt')
    model,init_report,init_sha=student_from_init(c); model.to(device)
    if c.get('schema_version')==3 and c['text']['mode']!='none':
        from .semantics import load_prototypes
        model._text_prototypes=load_prototypes(c,device)
    pg=groups(model,c['optimizer']); optimizer=torch.optim.AdamW(pg,betas=tuple(c['optimizer']['betas']),eps=c['optimizer']['eps'])
    # Smoke tests exercise representable updates at the recipe's peak LR. The
    # full-horizon first warmup step can be smaller than FP32 ULP for pretrained
    # norm weights. Production always uses the unchanged locked whole-run cosine.
    scheduler=UpdateCosine(optimizer,max_updates*len(c['stages'])+1 if smoke else c['schedule']['total_updates'],
                           1 if smoke else c['schedule']['warmup_updates'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)  # BF16 or FP32; no unsupported FP16 fallback.
    identity={'config_sha256':fingerprint(c),'config':c,'init_sha256':init_sha,'precision':'bf16' if amp else 'fp32',
              'environment':environment(),'parameter_groups':[{k:v for k,v in g.items() if k!='params'} for g in pg]}
    atomic_json(out/'initialization_report.json',init_report)
    start_epoch=0; best=None; best_s1=None; elapsed=0.; stage_parent=None
    if resume:
        saved=load(out/'last.pt'); restore(saved,identity,model,optimizer,scheduler,scaler)
        start_epoch=saved['next_epoch']; best=saved['best_score']; best_s1=saved['best_s1']; elapsed=saved['elapsed_seconds']
        stage_parent=saved.get('stage_parent_sha256')
        if stage_parent and sha(out/'s1_last.pt')!=stage_parent: raise ValueError('S1 last parent changed or missing')
        # Keep abandoned partial-epoch logs as evidence; valid curve restored from last.
        if (out/'curve.jsonl').exists():
            stamp=str(time.time_ns()); (out/'curve.jsonl').rename(out/('curve.before_resume.'+stamp+'.jsonl'))
        with (out/'curve.jsonl').open('w') as f:
            for row in saved['curves']: f.write(json.dumps(row)+'\n')
        curves=saved['curves']; del saved
    else: curves=[]
    atomic_json(out/'effective_config.json',identity)
    atomic_json(out/'status.json',{'state':'running','pid':os.getpid(),'smoke':smoke})
    torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter(); teacher=None; teacher_stage=None; smoke_reports=[]
    try:
        total_epochs=sum(s['epochs'] for s in c['stages'])
        epochs=range(start_epoch,total_epochs)
        if smoke: epochs=[0]+([c['stages'][0]['epochs']] if len(c['stages'])==2 else [])
        for epoch in epochs:
            stage=split_stage(c,epoch); configure_stage(model,stage['name'])
            loss_weights=dict(stage['loss'])
            if c.get('sampling')=='weighted_ce':
                from .data import class_weights
                loss_weights['class_weights']=class_weights(manifest(c['data'])['samples']['train']).tolist()
            if not smoke and len(c['stages'])==2 and stage['name']=='s2' and stage_parent is None:
                stage_parent=record_stage_transition(out,model,optimizer,scheduler,c['stages'][0]['epochs'])
            stage_config=out/'stages'/stage['name']/'effective_config.json'
            if not stage_config.exists():
                atomic_json(stage_config,{'run_identity':identity,'stage':stage,
                    'effective_loss_weights':loss_weights,
                    'entry_global_update':scheduler.updates,'s1_last_parent_sha256':stage_parent})
            frozen_head_digest=tensor_digest(model.classifier.state_dict()) if stage['name']=='s1' else None
            if stage['teacher']!=teacher_stage:
                del teacher; teacher=None; torch.cuda.empty_cache()
                if stage['teacher']: teacher=make_teacher(c,stage['teacher'],device)
                teacher_stage=stage['teacher']
            train_matrix=torch.zeros(7,7,dtype=torch.long); sums={}; nseen=0; updates=0; step_times=[]; probes_report=[]; stage_updated={}
            for batch_idx,b in enumerate(loader(c,'train',epoch)):
                scheduler.prepare(); optimizer.zero_grad(set_to_none=True)
                deep_probe=smoke or batch_idx==0 or batch_idx==c['schedule']['steps_per_epoch']-1
                selected=probes(model) if deep_probe else {}
                before={k:p.detach().clone() for k,(name,p) in selected.items()}
                torch.cuda.synchronize(); t=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): loss,logits,raw,weighted=forward_loss(model,teacher,b,loss_weights,device)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                objective_gradients=None
                interval=c.get('diagnostics',{}).get('gradient_interval',0)
                if interval and (scheduler.updates%interval==0) and hasattr(model,'fusion_projection'):
                    from .losses import gradient_report
                    objective_gradients=gradient_report(weighted,model.fusion_projection.weight)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['optimizer']['grad_clip'])
                if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
                if teacher and any(p.grad is not None for p in teacher.parameters()): raise RuntimeError('Teacher has gradients')
                grad={k:float(p.grad.float().norm()) if p.grad is not None else 0. for k,(name,p) in selected.items()}
                scaler.step(optimizer); scaler.update(); scheduler.step(); torch.cuda.synchronize()
                sec=time.perf_counter()-t; step_times.append(sec); updates+=1
                delta={k:float((p.detach()-before[k]).float().norm()) for k,(name,p) in selected.items()}
                record={'update':scheduler.updates,'epoch':epoch+1,'batch':batch_idx+1,'stage':stage['name'],'loss':float(loss),
                        'raw':{k:float(v) for k,v in raw.items()},'weighted':{k:float(v) for k,v in weighted.items()},
                        'grad_norm':float(norm),'probe_names':{k:name for k,(name,p) in selected.items()},'probe_gradient_norms':grad,
                        'probe_update_norms':delta,'lr':[g['lr'] for g in optimizer.param_groups],'seconds':sec}
                if objective_gradients is not None: record['objective_gradients']=objective_gradients
                if deep_probe and hasattr(model,'last_diagnostics'):
                    record['branch_diagnostics']={name:float(value) for name,value in model.last_diagnostics.items()}
                with (out/'updates.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                probes_report.append(record) if smoke else None
                if smoke:
                    for k in grad:
                        stage_updated[k]=stage_updated.get(k,False) or (grad[k]>0. and delta[k]>0.)
                elif batch_idx==0 and (not any(grad.values()) or not any(delta.values())):
                    raise RuntimeError('Student has no representable update or no gradients')
                n=len(b['label']); nseen+=n
                for k,v in raw.items(): sums[k]=sums.get(k,0.)+float(v)*n
                if logits is not None:
                    train_matrix+=torch.bincount(b['label']*7+logits.detach().argmax(-1).cpu(),minlength=49).reshape(7,7)
                if smoke and updates>=max_updates: break
            if smoke and not all(stage_updated.values()):
                raise RuntimeError('Target module had no actual gradient/update in the stage: '+str([k for k,v in stage_updated.items() if not v]))
            validation=evaluate_model(model,loader(c,'val'),device,amp,teacher,stage['loss'] if stage['name']=='s1' else None,max_batches=2 if smoke else None)
            if not model.training or (stage['name']=='s1' and model.classifier.training): raise RuntimeError('Validation mode restoration failed')
            if frozen_head_digest and tensor_digest(model.classifier.state_dict())!=frozen_head_digest: raise RuntimeError('S1 frozen head/BN changed')
            if smoke:
                # Strict reconstruction without a teacher or pretrained files at inference.
                configure_stage(model,'s2'); model.eval(); image=b['image_student'].to(device)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                    r=model(image); expected=(r if torch.is_tensor(r) else r.logits).detach().cpu()
                temp=out/'roundtrip.pt'; atomic_save(temp,{'model':model.state_dict()})
                fresh,_=make_student(c,initialize=False); fresh.load_state_dict(load(temp)['model'],strict=True); fresh.to(device).eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                    r=fresh(image); actual=(r if torch.is_tensor(r) else r.logits).detach().cpu()
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
                del fresh; temp.unlink()
                smoke_reports.append({'stage':stage['name'],'updates':updates,'validation':validation,'parameter_updates':probes_report,
                     'seconds_per_update':sum(step_times[1:] or step_times)/len(step_times[1:] or step_times),'inference_roundtrip':True,'stage_update_coverage':stage_updated})
                continue
            elapsed_now=elapsed+time.perf_counter()-begin
            row={'epoch':epoch+1,'stage':stage['name'],'updates':scheduler.updates,'train_losses':{k:v/nseen for k,v in sums.items()},
                 'train_accuracy':float(train_matrix.diag().sum()/train_matrix.sum()) if train_matrix.sum() else None,
                 'train_confusion_matrix':train_matrix.tolist(),'selection':validation,'elapsed_seconds':elapsed_now,
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'last_lr':[g['lr'] for g in optimizer.param_groups]}
            if train_matrix.sum(): row['train_metrics']=metrics(train_matrix)
            best_payload={'kind':'inference','identity':identity,'model':model.state_dict(),'epoch':epoch+1,'global_update':scheduler.updates,'selection':validation,'stage_parent_sha256':stage_parent}
            if stage['name']=='s1':
                if best_s1 is None or validation['loss']<best_s1: best_s1=validation['loss']; atomic_save(out/'s1_best.pt',best_payload)
            else:
                score=[validation['uar'],validation['accuracy']]
                if best is None or score>best: best=score; atomic_save(out/'best.pt',best_payload)
            curves.append(row)
            atomic_save(out/'last.pt',{'kind':'resume','identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'next_epoch':epoch+1,'next_batch':0,
                'global_update':scheduler.updates,'best_score':best,'best_s1':best_s1,'elapsed_seconds':elapsed_now,'curves':curves,
                'stage_parent_sha256':stage_parent,
                'sampler':{'next_epoch':epoch+1,'seed':c['seed'],'contract':'replay incomplete epoch from previous boundary'}})
            with (out/'curve.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            atomic_json(out/'status.json',{'state':'running','epoch':epoch+1,'updates':scheduler.updates})
        if smoke:
            result={'accepted':True,'config_hash':fingerprint(c),'stages':smoke_reports,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                    'parameters':sum(p.numel() for p in model.parameters()),'precision':identity['precision'],'learning_verified':False,
                    'diagnostic_scheduler':{'total_updates':scheduler.total,'warmup_updates':scheduler.warmup,
                        'purpose':'peak-LR representable parameter update acceptance; not a formal training run',
                        'production_schedule':c['schedule']}}
            atomic_json(out/'smoke.json',result); return result
        if scheduler.updates!=c['schedule']['total_updates']: raise RuntimeError('Final update count mismatch')
        atomic_json(out/'status.json',{'state':'trained_pending_final_evaluation','updates':scheduler.updates,'elapsed_seconds':elapsed+time.perf_counter()-begin})
    except Exception:
        atomic_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc(),'last_resume_contract':'epoch boundary only'})
        raise

def evaluate_run(c,plan_path,split):
    validate(c)
    require_lock(c,plan_path)
    out=Path(c['output']); payload=load(out/'best.pt')
    if payload['identity']['config_sha256']!=fingerprint(c): raise ValueError('Inference identity mismatch')
    # No teacher construction, no teacher/pretrained asset reads.
    with gpu_lease(out.parents[1]):
        model,_=make_student(c,initialize=False); model.load_state_dict(payload['model'],strict=True); model.cuda(); model.eval()
        name='final' if split=='test' else 'selection'
        result=evaluate_model(model,loader(c,split),torch.device('cuda'),amp=torch.cuda.is_bf16_supported(),predictions=out/(name+'_predictions.csv'))
        atomic_json(out/(name+'_metrics.json'),{'metrics':result,'best_sha256':sha(out/'best.pt'),'best_epoch':payload['epoch'],'config_hash':fingerprint(c)})
        if split=='test':
            last=load(out/'last.pt')
            if last['global_update']!=c['schedule']['total_updates']: raise ValueError('Run budget incomplete')
            atomic_json(out/'status.json',{'state':'completed','updates':last['global_update']})
        return result
