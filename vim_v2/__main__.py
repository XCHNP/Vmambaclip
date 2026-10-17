import argparse, csv, json, shutil, sys, traceback
from pathlib import Path
from .common import PROJECT,read,atomic_json,sha,fingerprint,code_identity,environment,check_asset

def fit_key(c): return fingerprint({k:c[k] for k in ['student','data','optimizer','seed','head_seed','code']})
def evidence_root(plan): return Path(plan['output_root'])/'acceptance'

def preflight(plan_path):
    from .config import validate
    p=read(plan_path); results=[]
    for r in p['runs']:
        try:
            c=read(r['config']); validate(c,assets=True)
            if fingerprint(c)!=r['config_hash']: raise ValueError('Plan config hash mismatch')
            results.append({'run_id':r['run_id'],'state':'assets_verified','gpu_available':environment()['cuda_available']})
        except Exception as e: results.append({'run_id':r['run_id'],'state':'blocked','reason':str(e)})
    return {'runs':results,'formal_training_started':False}

def budget(plan_path):
    plan=read(plan_path); out=[]; disk=0
    for r in plan['runs']:
        c=read(r['config']); path=evidence_root(plan)/r['run_id'].replace('/','__')/'smoke.json'
        if path.exists():
            s=read(path); measured=s['config_hash']==r['config_hash'] and s['accepted']
        else: s={}; measured=False
        row={'run_id':r['run_id'],'updates':r['updates'],'epochs':sum(x['epochs'] for x in c['stages']),
             'estimated_gpu_hours':None,'basis':'pending actual GPU measurement','estimated_disk_bytes':None}
        if measured:
            sec=sum(stage['seconds_per_update']*st['epochs']*c['schedule']['steps_per_epoch'] for stage,st in zip(s['stages'],c['stages']))
            # Validation smoke sees 64 examples; extrapolation is explicitly an estimate.
            count=sum(len(x) for k,x in __import__('vim_v2.data',fromlist=['manifest']).manifest(c['data'])['samples'].items() if k=='val')
            sec+=sum(st['epochs']*stage['validation']['seconds']*count/stage['validation']['samples'] for stage,st in zip(s['stages'],c['stages']))
            size=s['parameters']*4; required=size*(11 if len(c['stages'])==2 else 7)  # best, last(+Adam), atomic temp; Full also retains S1 best/last
            row.update(estimated_gpu_hours=sec/3600,estimated_disk_bytes=required,peak_gpu_bytes=s['peak_gpu_bytes'],basis='measured warm GPU steps + extrapolated selection + atomic-save disk reserve')
            disk+=required
        out.append(row)
    root=Path(plan['output_root']); root.mkdir(parents=True,exist_ok=True)
    return {'runs':out,'estimated_gpu_hours':sum(r['estimated_gpu_hours'] for r in out) if all(r['estimated_gpu_hours'] is not None for r in out) else None,
            'disk_estimate_partial_bytes':disk,'free_bytes':shutil.disk_usage(root).free,'includes_teacher_preparation':False,
            'extra_disk_reserve_bytes':4*1024**3,'ready_for_all_runs':all(r['estimated_gpu_hours'] is not None for r in out) and disk+4*1024**3<shutil.disk_usage(root).free}

def lock_plan(plan_path):
    from .config import validate
    plan=read(plan_path); root=evidence_root(plan); accepted=[]; blocked=[]; evidence=[]
    for row in plan['runs']:
        c=read(row['config']); rid=row['run_id']; name=rid.replace('/','__')
        files=[root/(c['dataset']+'_data.json'),root/(name+'_init_gpu.json'),root/name/'smoke.json',root/('fit_'+fit_key(c)+'.json'),root/(name+'_resume_gpu.json')]
        files += [root/(c['dataset']+'_'+s['teacher']+'_teacher.json') for s in c['stages'] if s['teacher']]
        if c.get('schema_version')==3 and c['text']['mode']!='none': files.append(root/(name+'_semantics.json'))
        try:
            validate(c,assets=True)
            if fingerprint(c)!=row['config_hash']: raise ValueError('Plan config identity changed')
            for path in files:
                item=read(path)
                if not (item.get('accepted') or item.get('gpu_accepted')): raise ValueError('Acceptance not passed: '+str(path))
                if item.get('config_hash') and item['config_hash']!=row['config_hash'] and not path.name.startswith('fit_'): raise ValueError('Stale acceptance: '+str(path))
                if path.name.endswith('_data.json') and item['manifest_sha256']!=c['data']['sha256']: raise ValueError('Stale data audit')
                if path.name.startswith('fit_') and item['fit_key']!=fit_key(c): raise ValueError('Stale fit audit')
                if path.name.endswith('_teacher.json') and item['teacher_sha256']!=c['teachers'][item['stage']]['sha256']: raise ValueError('Stale teacher audit')
                if path.name.endswith('_semantics.json') and item['prototypes_sha256']!=c['text']['prototypes']['sha256']: raise ValueError('Stale prototype audit')
            smoke=read(root/name/'smoke.json'); disk=smoke['parameters']*4*(11 if len(c['stages'])==2 else 7)
            if shutil.disk_usage(Path(plan['output_root'])).free<disk+4*1024**3: raise ValueError('Insufficient disk reserve')
            accepted.append(rid); evidence.extend({'path':str(f.resolve()),'sha256':sha(f)} for f in files)
        except Exception as e: blocked.append({'run_id':rid,'reason':str(e)})
    result={'plan_sha256':sha(plan_path),'code':code_identity(),'qualified_runs':accepted,'blocked':blocked,'evidence':evidence,'final_eval_used_for_selection':False}
    atomic_json(Path(plan_path).with_name('lock_readiness.json'),result)
    if not accepted: raise RuntimeError('No runs passed acceptance; plan.lock deliberately not written')
    lock=Path(plan_path).with_name('plan.lock')
    if lock.exists() and read(lock)!=result: raise FileExistsError('Existing lock immutable; create a new plan version')
    atomic_json(lock,result); return result

def aggregate(plan_path):
    p=read(plan_path); rows=[]; class_rows=[]
    for run in p['runs']:
        c=read(run['config']); out=Path(c['output']); f=out/'final_metrics.json'; state=read(out/'status.json') if (out/'status.json').exists() else {'state':'not_started'}
        row={'run_id':run['run_id'],'state':state['state'],'accuracy':None,'uar':None,'macro_f1':None,'updates':run['updates']}
        if f.exists():
            data=read(f)
            if data['config_hash']!=run['config_hash'] or sha(out/'best.pt')!=data['best_sha256']: raise ValueError('Result identity mismatch')
            import torch
            from .runtime import metrics
            matrix=torch.zeros(7,7,dtype=torch.long)
            from .data import manifest
            expected_rows=manifest(c['data'])['samples']['test']; seen=[]
            with (out/'final_predictions.csv').open() as stream:
                for pred in csv.DictReader(stream):
                    probabilities=torch.tensor([float(pred['p'+str(i)]) for i in range(7)])
                    if not torch.isfinite(probabilities).all() or probabilities.min()<0 or abs(float(probabilities.sum())-1)>1e-5: raise ValueError('Invalid prediction probabilities')
                    if int(probabilities.argmax())!=int(pred['predicted_label']): raise ValueError('Prediction/probability mismatch')
                    matrix[int(pred['true_label']),int(pred['predicted_label'])]+=1; seen.append((pred['sample_id'],int(pred['true_label'])))
            if seen!=[(r['path'],r['label']) for r in expected_rows]: raise ValueError('Prediction coverage/order differs from frozen final manifest')
            recalculated=metrics(matrix)
            if any(abs(recalculated[k]-data['metrics'][k])>1e-10 for k in ['accuracy','uar','macro_f1']): raise ValueError('Prediction/metric mismatch')
            row.update({k:recalculated[k] for k in ['accuracy','uar','macro_f1']})
            class_rows.extend(dict(run_id=run['run_id'],class_name=c['class_order'][item['class_index']],**item) for item in recalculated['per_class'])
        rows.append(row)
    conclusions={'pretraining':'pending A2-A1 and A4-A3','budget':'pending A3-A1 and A4-A2','KD':'pending matched Hybrid results',
                 'two_stage':'pending; teacher switch, S1 without CE, and loss schedule remain different',
                 'sfew':'pending train/selection curves and predicted class distribution','uncertainty':'seed42 only; no significance or seed variance claims'}
    by_id={r['run_id']:r for r in rows}
    comparisons={}
    pairs=[('pretraining_30','rafdb/native_pretrained_30','rafdb/native_random_30'),('pretraining_60','rafdb/native_ce','rafdb/native_random_60'),
           ('budget_random','rafdb/native_random_60','rafdb/native_random_30'),('budget_pretrained','rafdb/native_ce','rafdb/native_pretrained_30')]
    for d in ['rafdb','sfew','affectnet']:
        pairs.extend([(d+'_logits_vs_ce',d+'/hybrid_logits_kd',d+'/hybrid_ce'),(d+'_feature_vs_ce',d+'/hybrid_feature_kd',d+'/hybrid_ce'),(d+'_two_stage_vs_feature',d+'/full_v2',d+'/hybrid_feature_kd')])
    if p.get('schema_version')==3:
        pairs=[]
        for run in p['runs']:
            cfg=read(run['config']); parent=cfg['comparison']['parent']
            if parent:
                parent=p.get('aliases',{}).get(parent,parent)
                from .experiments import experiment_run_name
                pairs.append((run['run_id']+'_vs_'+parent,run['run_id'],cfg['dataset']+'/'+experiment_run_name(parent,cfg['seed'])))
            name=cfg['comparison']['experiment']
            if name.startswith('T3_w'):
                simple=name.replace('T3_w','T2_w',1)
                pairs.append((run['run_id']+'_vs_'+simple,run['run_id'],cfg['dataset']+'/'+experiment_run_name(simple,cfg['seed'])))
        conclusions={'scope':'Catalog comparisons; combined variants are labeled, not single-factor evidence.',
                     'text':'Use semantic audits and per-class results; FER head accuracy alone does not establish compatibility.',
                     'uncertainty':'Read planned/completed seeds in seed_summary; seed dispersion and paired image bootstrap describe different uncertainty sources.'}
    for label,left,right in pairs:
        a,b=by_id.get(left),by_id.get(right)
        if a is None or b is None:
            comparisons[label]={'left':left,'right':right,'state':'reference_not_in_selected_plan'}; continue
        comparisons[label]={'left':left,'right':right,'delta_percentage_points':{k:100*(a[k]-b[k]) if a[k] is not None and b[k] is not None else None for k in ['accuracy','uar','macro_f1']}}
    result={'results':rows,'comparisons':comparisons,'analysis':conclusions,'protocol':p['final_protocol']}
    if p.get('schema_version')==3:
        import statistics
        buckets={}
        for run in p['runs']:
            cfg=read(run['config']); key=cfg['dataset']+'/'+cfg['comparison']['experiment']
            buckets.setdefault(key,[]).append((cfg['seed'],by_id[run['run_id']]))
        result['seed_summary']={}
        for key,items in buckets.items():
            complete=[(seed,row) for seed,row in items if row['accuracy'] is not None]
            result['seed_summary'][key]={'planned_seeds':[seed for seed,_ in items],'completed_seeds':[seed for seed,_ in complete],
                'metrics':{metric:{'mean':statistics.mean([row[metric] for _,row in complete]) if complete else None,
                                  'sample_std':statistics.stdev([row[metric] for _,row in complete]) if len(complete)>1 else None}
                           for metric in ['accuracy','uar','macro_f1']}}
    root=Path(p['output_root'])/'reports'; atomic_json(root/'results.json',result)
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (root/'per_class.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['run_id','class_name','class_index','precision','recall','f1','support']); writer.writeheader(); writer.writerows(class_rows)
    return result

def main():
    parser=argparse.ArgumentParser(description='Vim V2 additive training and acceptance runtime')
    sub=parser.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('inventory'); q.add_argument('--output',required=True)
    q=sub.add_parser('prepare-v2'); q.add_argument('--paths',required=True); q.add_argument('--output',required=True)
    q.add_argument('--use-sasf',action='store_true'); q.add_argument('--cross-branch-fusion',action='store_true')
    q=sub.add_parser('prepare-vmamba'); q.add_argument('--paths',required=True); q.add_argument('--output',required=True)
    q.add_argument('--no-sasf',action='store_true'); q.add_argument('--legacy-fusion',action='store_true')
    sub.add_parser('list-experiments')
    q=sub.add_parser('prepare-experiments'); q.add_argument('--paths',required=True); q.add_argument('--output',required=True)
    q.add_argument('--datasets',nargs='+',default=['rafdb']); q.add_argument('--experiments',nargs='+')
    q.add_argument('--refinement-base',choices=['R0','R1','R2','R3'],default='R3')
    q.add_argument('--seeds',type=int,nargs='+',default=[42])
    q=sub.add_parser('paired-bootstrap'); q.add_argument('--baseline',required=True); q.add_argument('--candidate',required=True)
    q.add_argument('--output',required=True); q.add_argument('--resamples',type=int,default=2000)
    q.add_argument('--seed',type=int,default=42); q.add_argument('--split',choices=['selection_val','final_eval'],default='selection_val')
    q=sub.add_parser('prepare-text'); q.add_argument('--paths',required=True); q.add_argument('--output',required=True); q.add_argument('--device',default='cpu')
    for command in ['validate-plan','budget','lock-plan','aggregate','run-plan','verify']:
        q=sub.add_parser(command); q.add_argument('--plan',required=True)
        if command=='validate-plan': q.add_argument('--dry-run',action='store_true')
        if command=='run-plan':
            q.add_argument('--resume-incomplete',action='store_true')
            q.add_argument('--group',choices=['diagnostic','core','architecture','branches','combined','supervision','text','augmentation','imbalance','refinement','refinement_followup'],default='core')
        if command=='verify': q.add_argument('--gpu',action='store_true')
    for command in ['audit-data','audit-init','smoke','fit','run','resume','evaluate','audit-teacher','audit-resume','benchmark','audit-semantics','diagnose-branches']:
        q=sub.add_parser(command); q.add_argument('--config',required=True)
        if command in ['run','resume','evaluate','benchmark']: q.add_argument('--plan',required=True)
        if command=='evaluate': q.add_argument('--split',choices=['selection_val','final_eval'],required=True); q.add_argument('--require-locked-plan',action='store_true')
        if command=='audit-init': q.add_argument('--gpu',action='store_true')
        if command in ['fit','smoke']: q.add_argument('--max-updates',type=int,default=1000 if command=='fit' else 4)
        if command=='audit-teacher': q.add_argument('--stage',choices=['t1','t2'],required=True)
        if command in ['audit-semantics','diagnose-branches']: q.add_argument('--device',default='cpu')
        if command=='audit-semantics': q.add_argument('--teacher-stage',choices=['t1','t2'],default='t2')
    a=parser.parse_args(); result=None
    from .config import prepare
    from .runtime import run,evaluate_run
    from .acceptance import audit_init,fit,audit_teacher,audit_resume
    if hasattr(a,'config'):
        c=read(a.config); root=Path(c['output']).parents[1]/'acceptance'; root.mkdir(parents=True,exist_ok=True); name=c['run_id'].replace('/','__')
    if a.cmd=='inventory': result={'environment':environment(),'code':code_identity()}; atomic_json(a.output,result)
    elif a.cmd=='prepare-v2': result=prepare(a.paths,a.output,use_sasf=a.use_sasf,cross_branch_fusion=a.cross_branch_fusion)
    elif a.cmd=='prepare-vmamba': result=prepare(a.paths,a.output,backbone='vmamba',use_sasf=not a.no_sasf,cross_branch_fusion=not a.legacy_fusion)
    elif a.cmd=='list-experiments':
        from .experiments import catalog,ALIASES
        result={'experiments':catalog(),'aliases':ALIASES}
    elif a.cmd=='prepare-experiments':
        from .experiments import prepare_experiments
        result=prepare_experiments(a.paths,a.output,datasets=a.datasets,experiments=a.experiments,refinement_base=a.refinement_base,seeds=a.seeds)
    elif a.cmd=='paired-bootstrap':
        from .statistics import paired_bootstrap
        result=paired_bootstrap(a.baseline,a.candidate,resamples=a.resamples,seed=a.seed,split=a.split); atomic_json(a.output,result)
    elif a.cmd=='prepare-text':
        from .semantics import prepare_prototypes
        result=prepare_prototypes(a.paths,a.output,a.device)
    elif a.cmd=='validate-plan': result=preflight(a.plan); atomic_json(Path(a.plan).with_name('preflight.json'),result)
    elif a.cmd=='budget': result=budget(a.plan); atomic_json(Path(a.plan).with_name('budget.json'),result)
    elif a.cmd=='lock-plan': result=lock_plan(a.plan)
    elif a.cmd=='aggregate': result=aggregate(a.plan)
    elif a.cmd=='audit-data':
        from .data import audit
        result=audit(c); atomic_json(root/(c['dataset']+'_data.json'),result)
    elif a.cmd=='audit-init': result=audit_init(c,a.gpu); atomic_json(root/(name+('_init_gpu.json' if a.gpu else '_init_cpu.json')),result)
    elif a.cmd=='fit': result=fit(c,a.max_updates); result['fit_key']=fit_key(c); atomic_json(root/('fit_'+fit_key(c)+'.json'),result)
    elif a.cmd=='audit-teacher': result=audit_teacher(c,a.stage); atomic_json(root/(c['dataset']+'_'+a.stage+'_teacher.json'),result)
    elif a.cmd=='audit-resume': result=audit_resume(c); atomic_json(root/(name+'_resume_gpu.json'),result)
    elif a.cmd=='audit-semantics':
        from .diagnostics import audit_semantics
        result=audit_semantics(c,a.teacher_stage,a.device); atomic_json(root/(name+'_semantics.json'),result)
    elif a.cmd=='diagnose-branches':
        from .diagnostics import diagnose_branches
        result=diagnose_branches(c,a.device); atomic_json(Path(c['output'])/'branch_diagnostics.json',result)
    elif a.cmd=='smoke': result=run(c,smoke=True,max_updates=a.max_updates)
    elif a.cmd in ['run','resume']: result=run(c,a.plan,resume=a.cmd=='resume')
    elif a.cmd=='evaluate': result=evaluate_run(c,a.plan,'test' if a.split=='final_eval' else 'val')
    elif a.cmd=='benchmark':
        import time,statistics,torch
        from .runtime import gpu_lease,require_lock
        from .models import make_student
        from .checkpoint import load
        from .data import loader
        require_lock(c,a.plan); out=Path(c['output']); saved=load(out/'best.pt')
        if saved['identity']['config_sha256']!=fingerprint(c): raise ValueError('Benchmark identity mismatch')
        with gpu_lease(out.parents[1]):
            model,_=make_student(c,initialize=False); model.load_state_dict(saved['model'],strict=True); model.cuda().eval()
            image=next(iter(loader(c,'val')))['image_student'].cuda(); amp=torch.cuda.is_bf16_supported(); timings=[]
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                for index in range(60):
                    torch.cuda.synchronize(); begin=time.perf_counter(); model(image); torch.cuda.synchronize()
                    if index>=10: timings.append(time.perf_counter()-begin)
            result={'best_sha256':sha(out/'best.pt'),'parameters':sum(p.numel() for p in model.parameters()),'batch_size':len(image),
                    'median_batch_seconds':statistics.median(timings),'images_per_second':len(image)/statistics.median(timings),
                    'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'warmup_iterations':10,'timed_iterations':50,'precision':'bf16' if amp else 'fp32',
                    'scope':'student inference only; excludes image decode, host-to-device copy, model loading and teacher'}
            atomic_json(out/'efficiency.json',result)
    elif a.cmd=='run-plan':
        plan=read(a.plan); qualified=read(Path(a.plan).with_name('plan.lock'))['qualified_runs']; result=[]
        for row in plan['runs']:
            if a.group=='diagnostic' and not (row['run_id'].startswith('rafdb/native')): continue
            if a.group not in ['core','diagnostic'] and row.get('group')!=a.group: continue
            if row['run_id'] not in qualified: result.append({'run_id':row['run_id'],'state':'blocked'}); continue
            cfg=read(row['config']); out=Path(cfg['output']); status=read(out/'status.json') if (out/'status.json').exists() else {}
            if status.get('state') in ['completed','trained_pending_final_evaluation']:
                saved=__import__('vim_v2.checkpoint',fromlist=['load']).load(out/'last.pt')
                if saved['identity']['config_sha256']!=row['config_hash'] or saved['global_update']!=cfg['schedule']['total_updates']: raise ValueError('Completion marker/checkpoint mismatch')
                continue
            run(cfg,a.plan,resume=a.resume_incomplete and (out/'last.pt').exists()); result.append({'run_id':row['run_id'],'state':'trained_pending_final_evaluation'})
    elif a.cmd=='verify':
        import subprocess
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(PROJECT/'tests_vim_v2'),'-v'],check=True)
        plan=read(a.plan); done=set(); result=[]
        if plan.get('schema_version')==3:
            subprocess.run([sys.executable,'-m','pytest',str(PROJECT/'tests/test_experiment_suite.py'),
                            str(PROJECT/'tests/test_cnn_refinement.py'),
                            str(PROJECT/'tests/test_vmamba.py'),str(PROJECT/'tests/test_sasf.py'),
                            str(PROJECT/'tests/test_vision_mamba_clip.py'),'-q'],check=True,cwd=PROJECT)
        for row in plan['runs']:
            cfg=read(row['config']); root=evidence_root(plan); name=cfg['run_id'].replace('/','__')
            try:
                if cfg['dataset'] not in done:
                    from .data import audit
                    atomic_json(root/(cfg['dataset']+'_data.json'),audit(cfg)); done.add(cfg['dataset'])
                atomic_json(root/(name+'_init_cpu.json'),audit_init(cfg))
                if a.gpu:
                    atomic_json(root/(name+'_init_gpu.json'),audit_init(cfg,True)); run(cfg,smoke=True)
                    atomic_json(root/(name+'_resume_gpu.json'),audit_resume(cfg))
                    key=fit_key(cfg)
                    if key not in done:
                        fr=fit(cfg); fr['fit_key']=key; atomic_json(root/('fit_'+key+'.json'),fr); done.add(key)
                    for st in cfg['stages']:
                        key=cfg['dataset']+'_'+str(st['teacher'])
                        if st['teacher'] and key not in done:
                            atomic_json(root/(key+'_teacher.json'),audit_teacher(cfg,st['teacher'])); done.add(key)
                    if cfg.get('schema_version')==3 and cfg['text']['mode']!='none':
                        from .diagnostics import audit_semantics
                        atomic_json(root/(name+'_semantics.json'),audit_semantics(cfg,device='cuda'))
                result.append({'run_id':row['run_id'],'state':'gpu_checks_executed' if a.gpu else 'cpu_checks_only'})
            except Exception as e: result.append({'run_id':row['run_id'],'state':'blocked','reason':str(e)})
        atomic_json(evidence_root(plan)/'verification.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if isinstance(result,dict) and result.get('accepted') is False: sys.exit(2)

if __name__=='__main__': main()
