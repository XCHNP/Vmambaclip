import copy, math
from pathlib import Path
from .common import CLASSES,MODEL,check_asset,read,atomic_json,fingerprint,code_identity,environment
from .losses import coefficients

METHODS=['native_ce','hybrid_ce','hybrid_logits_kd','hybrid_feature_kd','full_v2']
PREPROCESS={'geometry_version':'shared_rgb256_rrc09_flip05_v2','student_normalize':'ImageNet','teacher_normalize':'CLIP',
    'train_crop_scale':[.9,1.],'train_crop_ratio':[.75,4/3],'flip_probability':.5,'evaluation':'Resize256_CenterCrop256_bicubic',
    'student_mean':[.485,.456,.406],'student_std':[.229,.224,.225],
    'teacher_mean':[.48145466,.4578275,.40821073],'teacher_std':[.26862954,.26130258,.27577711],
    'resize_size':256,'eval_crop_size':256,'rgb':True,'interpolation':'bicubic','antialias':True,
    'train_drop_last':True,'eval_drop_last':False,'ce_weighting':'unweighted'}
def validate(c,assets=False):
    required={'schema_version','run_id','seed','head_seed','dataset','class_order','student','data','teachers','clip_base','batch_size','workers','optimizer','schedule','stages','preprocess','output','init_id','init_path','checkpoint','code','environment'}
    if set(c)!=required: raise ValueError('Unknown/missing config fields '+str(set(c)^required))
    if c['schema_version']!=2 or c['seed']!=42 or c['class_order']!=CLASSES or c['dataset'] not in ['rafdb','sfew','affectnet']: raise ValueError('Invalid core schema')
    s=c['student']; schedule=c['schedule']
    if c['head_seed']!=1042 or c['preprocess']!=PREPROCESS: raise ValueError('Unreviewed initialization/preprocessing protocol')
    expected_student={'architecture','official_model_name','source_revision','initialization','pretrained','cnn_pretrained','cnn_initialization','input_size','layer_indices','feature_readout','cls_position_policy','drop_path_rate','gradient_checkpointing'}
    if set(s)!=expected_student or set(schedule)!={'total_updates','steps_per_epoch','warmup_updates','min_lr_ratio','transition_source','optimizer_reset','scheduler_reset'}: raise ValueError('Unknown nested configuration fields')
    if s['architecture'] not in ['vim_native_ce','hybrid_v2'] or s['official_model_name']!=MODEL or s['input_size']!=256 or s['layer_indices']!=[21,22,23]: raise ValueError('Unsupported model configuration')
    if s['initialization'] not in ['random','pretrained'] or (s['architecture']=='hybrid_v2' and s['initialization']!='pretrained'): raise ValueError('Invalid initialization')
    if s['source_revision']!='dd0358ad1e42701f22afbefa0717cc8825cf9f45' or s['cnn_initialization']!=('imagenet_resnet18_v1' if s['architecture']=='hybrid_v2' else None): raise ValueError('Invalid official/CNN provenance')
    if c['batch_size']!=32 or c['workers']<0: raise ValueError('Invalid batch/workers')
    if schedule['total_updates']!=schedule['steps_per_epoch']*sum(x['epochs'] for x in c['stages']): raise ValueError('Budget mismatch')
    if schedule['warmup_updates']!=max(1,math.ceil(schedule['total_updates']*.05)) or schedule['min_lr_ratio']!=.01: raise ValueError('Invalid cosine')
    if schedule['optimizer_reset'] or schedule['scheduler_reset'] or schedule['transition_source']!='last': raise ValueError('Legacy transitions forbidden in V2')
    method=c['run_id'].split('/')[-1]
    if c['run_id']!=c['dataset']+'/'+method or method not in METHODS+(['native_random_30','native_pretrained_30','native_random_60'] if c['dataset']=='rafdb' else []): raise ValueError('Unsupported core run ID')
    expected_epochs={'rafdb':60,'sfew':50,'affectnet':30}[c['dataset']]
    if method in ['native_random_30','native_pretrained_30']: expected_epochs=30
    if sum(st['epochs'] for st in c['stages'])!=expected_epochs: raise ValueError('Core epoch budget changed')
    expected_stages=['s1','s2'] if method=='full_v2' else ['s2']
    if [s['name'] for s in c['stages']]!=expected_stages: raise ValueError('Core stages changed')
    if method=='full_v2' and c['stages'][0]['epochs']!=(10 if c['dataset']=='rafdb' else 5): raise ValueError('Core S1 budget changed')
    expected_init='random' if method.startswith('native_random_') else 'pretrained'
    if s['initialization']!=expected_init or s['architecture']!=('vim_native_ce' if method.startswith('native') else 'hybrid_v2'): raise ValueError('Run/model initialization mismatch')
    if s['cls_position_policy']!='middle' or s['drop_path_rate']!=.1 or s['gradient_checkpointing'] is not True: raise ValueError('Core Vim settings changed')
    if s['feature_readout']!=('official_final_norm_middle_cls' if method.startswith('native') else 'legacy_post_block_sum_patch_pool'): raise ValueError('Feature semantics changed')
    out=Path(c['output']); init_path=Path(c['init_path'])
    if not out.is_absolute() or out.parts[-2:]!=(c['dataset'],method) or 'fer_suite_simplified_20261013' in out.parts: raise ValueError('Unsafe/legacy output path')
    if init_path.parent!=out.parents[1]/'initializations' or init_path.name!=c['init_id']+'.pt': raise ValueError('Invalid init snapshot path')
    if method.startswith('native_'): method='native_ce'
    for st in c['stages']:
        if st['loss']!=coefficients(method,st['name']): raise ValueError('Loss coefficients do not match locked core design')
        if st['name'] not in ['s1','s2'] or st['epochs']<=0: raise ValueError('Invalid stage')
        if (st['loss']['feature'] or st['loss']['logits']) and st['teacher'] not in ['t1','t2']: raise ValueError('Missing teacher dependency')
        expected_teacher='t1' if st['name']=='s1' else ('t2' if method in ['full_v2','hybrid_feature_kd','hybrid_logits_kd'] else None)
        if st['teacher']!=expected_teacher: raise ValueError('Unexpected teacher dependency')
    if c['optimizer']!={'backbone_lr':1e-5 if c['dataset']=='sfew' else 3e-5,'new_lr':1e-4 if c['dataset']=='sfew' else 3e-4,'weight_decay':.01,'betas':[.9,.98],'eps':1e-6,'grad_clip':1.,'amp':'bf16_if_supported_else_fp32'}: raise ValueError('Unreviewed optimizer recipe')
    if assets:
        from .data import manifest
        m=manifest(c['data'])
        if len(m['samples']['train'])//32!=schedule['steps_per_epoch']: raise ValueError('Manifest update count mismatch')
        if s['initialization']=='pretrained': check_asset(s['pretrained'])
        if s['architecture']=='hybrid_v2': check_asset(s['cnn_pretrained'])
        for st in c['stages']:
            if st['teacher']:
                from .models import validate_teacher
                validate_teacher(c,st['teacher'])
        if c['code']!=code_identity(): raise ValueError('Source changed; regenerate plan before acceptance')
    return c

def prepare(paths_path,out):
    paths=read(paths_path); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    runs=[]; code=code_identity(); env=environment()
    for domain in ['rafdb','sfew','affectnet']:
        info=paths['datasets'][domain]; total={'rafdb':60,'sfew':50,'affectnet':30}[domain]; s1=10 if domain=='rafdb' else 5
        recipes=[(x,'pretrained',total) for x in METHODS]
        if domain=='rafdb': recipes=[('native_random_30','random',30),('native_pretrained_30','pretrained',30),('native_random_60','random',60)]+recipes
        for method,init,epochs in recipes:
            native=method.startswith('native'); name='native_ce' if native else method
            architecture='vim_native_ce' if native else 'hybrid_v2'
            init_id=architecture+'_'+init+'_seed42'
            stages=[{'name':'s2','epochs':epochs,'teacher':'t2' if name in ['hybrid_feature_kd','hybrid_logits_kd'] else None,'loss':coefficients(name,'s2')}]
            if name=='full_v2': stages=[{'name':'s1','epochs':s1,'teacher':'t1','loss':coefficients(name,'s1')},{'name':'s2','epochs':epochs-s1,'teacher':'t2','loss':coefficients(name,'s2')}]
            steps=info['counts']['train']//32; updates=steps*epochs; run_id=domain+'/'+method
            c={'schema_version':2,'run_id':run_id,'seed':42,'head_seed':1042,'dataset':domain,'class_order':CLASSES,
               'student':{'architecture':architecture,'official_model_name':MODEL,'source_revision':'dd0358ad1e42701f22afbefa0717cc8825cf9f45','initialization':init,'pretrained':paths['vim'],
                  'cnn_pretrained':paths['cnn'] if not native else None,'cnn_initialization':'imagenet_resnet18_v1' if not native else None,'input_size':256,'layer_indices':[21,22,23],
                  'feature_readout':'official_final_norm_middle_cls' if native else 'legacy_post_block_sum_patch_pool',
                  'cls_position_policy':'middle','drop_path_rate':.1,'gradient_checkpointing':True},
               'data':info['manifest'],'teachers':info['teachers'] if any(st['teacher'] for st in stages) else {},'clip_base':paths['clip_base'],
               'batch_size':32,'workers':4,'optimizer':{'backbone_lr':1e-5 if domain=='sfew' else 3e-5,'new_lr':1e-4 if domain=='sfew' else 3e-4,
                   'weight_decay':.01,'betas':[.9,.98],'eps':1e-6,'grad_clip':1.,'amp':'bf16_if_supported_else_fp32'},
               'schedule':{'total_updates':updates,'steps_per_epoch':steps,'warmup_updates':max(1,math.ceil(updates*.05)),'min_lr_ratio':.01,
                   'transition_source':'last','optimizer_reset':False,'scheduler_reset':False},'stages':stages,
               'preprocess':copy.deepcopy(PREPROCESS),
               'output':str(Path(paths['output_root'])/run_id),'init_id':init_id,'init_path':str(Path(paths['output_root'])/'initializations'/f'{init_id}.pt'),
               'checkpoint':{'resume_contract':'epoch_boundary_only','selection':['max_uar','max_accuracy','earlier_epoch'],'s1_selection':'min_feature_loss'},
               'code':code,'environment':env}
            validate(c); path=out/(run_id.replace('/','__')+'.json'); atomic_json(path,c)
            runs.append({'run_id':run_id,'config':str(path.resolve()),'config_hash':fingerprint(c),'stages':len(stages),'updates':updates,
                         'dependencies':[domain+'/'+s['teacher'] for s in stages if s['teacher']], 'state':'implemented'})
    plan={'schema_version':2,'runs':runs,'unique_student_paths':18,'student_stages':21,'reuse':{'A4_raf_pretrained_60':'rafdb/native_ce'},
          'output_root':paths['output_root'],'core_only':True,'final_protocol':'iterative development on previously observed benchmark; not a new blind test'}
    if len(runs)!=18 or sum(r['stages'] for r in runs)!=21: raise AssertionError('DAG count')
    atomic_json(out/'plan.json',plan)
    extensions=[{'name':n,'enabled':False,'budget':b,'status':'blocked_pending_explicit_protocol_and_assets'} for n,b in [
        ('feature_plus_logits','1 x RAF 60 epochs'),('backbone_only_feature','matched CE and feature pair, 2 x RAF 60'),
        ('AU_W_multilayer','at most 3 x RAF 60'),('T0_feature','1 x RAF 60'),('student_transfer','4 x SFEW 30 + source costs'),
        ('shared_affectnet_teacher','RAF 60 + SFEW 50'),('cross_domain_eval','18 evaluations'),('multiseed','selected runs x 2'),('input224','separate matched protocol required')]]
    atomic_json(out/'extensions.disabled.json',extensions)
    atomic_json(out/'legacy_full.disabled.json',{'enabled':False,'entry':'artifacts/training_prepare_20261013/fer_expanded.py','name':'legacy_full',
                'kl':'student||teacher','transition':'S1 best, rebuild Adam','warning':'Not V2 or a single-loss-factor comparison'})
    return plan
