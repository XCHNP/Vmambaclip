"""Schema 3: one runtime, explicit controlled FER experiments, no implicit training."""

import copy
import math
from pathlib import Path

from .common import CLASSES, MODEL, atomic_json, check_asset, code_identity, environment, fingerprint, read
from .config import PREPROCESS

ALIASES={'D_ce':'F_recommended','B_hybrid':'E3','T2':'T2_w0p1','T3':'T3_w0p1','R0':'E3'}

def recipe(ce=1., feature=0., logits=0., text=0., affinity=0.):
    return {'ce':ce,'feature':feature,'feature_cosine':1.,'feature_mse':1.,'feature_mse_space':'raw',
            'logits':logits,'text':text,'text_scale':10.,'affinity':affinity,'temperature':2.}


def catalog(refinement_base='R3'):
    """Each row declares its comparator; variants never silently form a Cartesian grid."""
    base={'backbone_family':'vmamba','variant':'hybrid','use_sasf':True,'sasf_stage':'l',
          'cnn_multiscale':False,'query_mode':'single','patch_dropout':0.,'fusion':'mamba_cnn_cross_attention'}
    rows={}
    def add(name,parent=None,student=None,loss=None,text_mode='none',two_stage=False,sampling='uniform',group='architecture'):
        rows[name]={'parent':parent,'student':dict(base,**(student or {})),
                    'loss':loss or recipe(),'text_mode':text_mode,'two_stage':two_stage,'sampling':sampling,'group':group}
    add('E1',student={'backbone_family':'vim','use_sasf':False})
    add('E2','E1',student={'backbone_family':'vim'})
    add('E3','E1',student={'use_sasf':False})
    add('E4','E3')
    add('B_cnn','E3',student={'backbone_family':'cnn','variant':'cnn_only','use_sasf':False,'fusion':'none'},group='branches')
    add('B_vmamba','E3',student={'variant':'mamba_only','use_sasf':False,'fusion':'concat_projection'},group='branches')
    add('S_early','E4',student={'sasf_stage':'e'})
    add('S_middle','E4',student={'sasf_stage':'m'})
    add('C_multiscale','E4',student={'cnn_multiscale':True})
    add('Q_multi','E4',student={'query_mode':'multi'})
    add('F_recommended','S_middle',student={'sasf_stage':'m','cnn_multiscale':True},group='combined')
    architecture={'sasf_stage':'m','cnn_multiscale':True}
    add('T1','D_ce',student=architecture,loss=recipe(logits=.2),group='supervision')
    for mode,label in [('simple','T2'),('distinctive','T3')]:
        for weight in [.05,.1,.2]:
            add(label+'_w'+str(weight).replace('.','p'),'T1',student=architecture,
                loss=recipe(logits=.2,text=weight),text_mode=mode,group='text')
    add('D_feature','D_ce',student=architecture,loss=recipe(feature=.2),group='supervision')
    add('D_two_stage','T1',student=architecture,loss=recipe(logits=.2),two_stage=True,group='supervision')
    add('D_affinity','T1',student=architecture,loss=recipe(logits=.2,affinity=.1),group='supervision')
    add('A_patchdrop','D_ce',student=dict(architecture,patch_dropout=.3),group='augmentation')
    add('A_balanced_sampler','D_ce',student=architecture,sampling='balanced',group='imbalance')
    add('A_weighted_ce','D_ce',student=architecture,sampling='weighted_ce',group='imbalance')
    refinement={'use_sasf':False,'cnn_auxiliary':False,'cnn_residual':False}
    add('R1','R0',student=dict(refinement,cnn_auxiliary=True),loss=dict(recipe(),cnn_aux=.2),group='refinement')
    add('R2','R0',student=dict(refinement,cnn_residual=True),group='refinement')
    add('R3','R1',student=dict(refinement,cnn_auxiliary=True,cnn_residual=True),loss=dict(recipe(),cnn_aux=.2),group='refinement')
    if refinement_base not in ['R0','R1','R2','R3']: raise ValueError('refinement_base must be R0/R1/R2/R3')
    parent=rows['E3'] if refinement_base=='R0' else rows[refinement_base]
    selected=copy.deepcopy(parent['student'])
    loss=copy.deepcopy(parent['loss'])
    add('R4',refinement_base,student=selected,loss=loss,sampling='class_balanced_ce',group='refinement_followup')
    add('R5',refinement_base,student=selected,loss=dict(loss,text=.05),text_mode='simple',group='refinement_followup')
    add('R6',refinement_base,student=dict(selected,use_sasf=True,sasf_stage='m'),loss=loss,group='refinement_followup')
    return rows


def model_identity(student,seed=42,head_seed=1042):
    suffix='' if (seed,head_seed)==(42,1042) else f'_seed{seed}_head{head_seed}'
    return 'experiment_'+fingerprint(student)[:20]+suffix


def experiment_run_name(name,seed):
    return name if seed==42 else f'{name}_seed{seed}'


def prepare_experiments(paths_path,out,*,datasets=('rafdb',),experiments=None,refinement_base='R3',seeds=(42,)):
    from .official_experiments import student_spec
    paths=read(paths_path); out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Use an empty new plan directory')
    rows=catalog(refinement_base); requested=list(rows) if experiments is None else list(experiments)
    if not seeds or len(seeds)!=len(set(seeds)) or any(type(seed) is not int or seed<0 or seed>2**32-1001 for seed in seeds): raise ValueError('Seeds must be unique nonnegative 32-bit integers')
    if not requested or len(requested)!=len(set(requested)) or set(requested)-(set(rows)|set(ALIASES)): raise ValueError('Unknown/duplicate experiment selection')
    chosen=list(dict.fromkeys(ALIASES.get(name,name) for name in requested))
    if not datasets or len(datasets)!=len(set(datasets)) or set(datasets)-{'rafdb','sfew','affectnet'}: raise ValueError('Unknown/duplicate datasets')
    output_root=Path(paths['output_root'])
    if not output_root.is_absolute(): raise ValueError('output_root must be absolute')
    if output_root.exists() and any(output_root.iterdir()): raise FileExistsError('Use a new output_root for this experiment suite')
    out.mkdir(parents=True,exist_ok=True); runs=[]; code=code_identity(); env=environment()
    for domain in datasets:
        info=paths['datasets'][domain]; steps=info['counts']['train']//32
        if steps<1: raise ValueError('Training split must contain at least one complete batch32')
        epochs={'rafdb':60,'sfew':50,'affectnet':30}[domain]
        for name,seed in [(name,seed) for name in chosen for seed in seeds]:
            row=rows[name]; student=student_spec(row['student'],paths)
            head_seed=1000+seed; init_id=model_identity(student,seed,head_seed)
            run_name=experiment_run_name(name,seed); run_id=domain+'/'+run_name
            stages=[{'name':'s2','epochs':epochs,'teacher':'t2' if row['loss']['feature'] or row['loss']['logits'] or row['loss']['affinity'] else None,'loss':copy.deepcopy(row['loss'])}]
            if row['two_stage']:
                s1=10 if domain=='rafdb' else 5
                stages[0]['epochs']-=s1
                stages.insert(0,{'name':'s1','epochs':s1,'teacher':'t1','loss':recipe(ce=0.,feature=1.)})
            dependencies={stage['teacher'] for stage in stages if stage['teacher']}
            registry=dependencies|({'t1'} if 't2' in dependencies else set())
            teachers={key:copy.deepcopy(info.get('teachers',{}).get(key)) for key in sorted(registry)}
            text={'mode':row['text_mode'],'prototypes':copy.deepcopy(paths.get('text_prototypes',{}).get(row['text_mode']))}
            c={'schema_version':3,'run_id':run_id,'seed':seed,'head_seed':head_seed,'dataset':domain,'class_order':CLASSES,
               'student':student,'data':copy.deepcopy(info['manifest']),'teachers':teachers,'clip_base':copy.deepcopy(paths.get('clip_base')),
               'batch_size':32,'workers':paths.get('workers',4),
               'optimizer':{'backbone_lr':1e-5 if domain=='sfew' else 3e-5,'new_lr':1e-4 if domain=='sfew' else 3e-4,'weight_decay':.01,'betas':[.9,.98],'eps':1e-6,'grad_clip':1.,'amp':'bf16_if_supported_else_fp32'},
               'schedule':{'total_updates':steps*epochs,'steps_per_epoch':steps,'warmup_updates':max(1,math.ceil(steps*epochs*.05)),
                           'min_lr_ratio':.01,'transition_source':'last','optimizer_reset':False,'scheduler_reset':False},
               'stages':stages,'preprocess':copy.deepcopy(PREPROCESS),'text':text,
               'sampling':row['sampling'],'diagnostics':{'gradient_interval':1000},
               'comparison':{'experiment':name,'parent':row['parent'],'group':row['group'],
                             'note':'F_recommended combines two changes; E1/E3 change backbone capacity and initialization source, not just scan type'},
               'output':str(output_root/domain/run_name),'init_id':init_id,'init_path':str(output_root/'initializations'/(init_id+'.pt')),
               'checkpoint':{'resume_contract':'epoch_boundary_only','selection':['max_uar','max_accuracy','earlier_epoch'],'s1_selection':'min_feature_loss'},
               'code':code,'environment':env}
            if name in ['R4','R5','R6']: c['comparison']['refinement_base']=refinement_base
            validate_experiment(c)
            path=out/(domain+'__'+run_name+'.json'); atomic_json(path,c)
            runs.append({'run_id':run_id,'config':str(path),'config_hash':fingerprint(c),'stages':len(stages),
                         'updates':steps*epochs,'dependencies':[domain+'/'+key for key in sorted(dependencies)],'state':'implemented','group':row['group']})
    plan={'schema_version':3,'runs':runs,'output_root':str(output_root),'core_only':False,
          'final_protocol':'select on selection_val only; final_eval reserved for reporting',
          'catalog':rows,'aliases':ALIASES,'seeds':list(seeds),'refinement_base':refinement_base,
          'unique_student_paths':len(runs),'student_stages':sum(row['stages'] for row in runs)}
    atomic_json(out/'plan.json',plan)
    return plan


def validate_experiment(c,assets=False):
    from .official_experiments import validate_student_spec
    required={'schema_version','run_id','seed','head_seed','dataset','class_order','student','data','teachers','clip_base','batch_size','workers','optimizer','schedule','stages','preprocess','text','sampling','diagnostics','comparison','output','init_id','init_path','checkpoint','code','environment'}
    if set(c)!=required or c['schema_version']!=3: raise ValueError('Unknown/missing experiment schema fields')
    if type(c['seed']) is not int or not 0<=c['seed']<=2**32-1001 or c['head_seed']!=1000+c['seed'] or c['class_order']!=CLASSES or c['dataset'] not in ['rafdb','sfew','affectnet']: raise ValueError('Experiment class/seed/dataset mismatch')
    validate_student_spec(c['student'])
    name=c['comparison']['experiment']; rows=catalog(c['comparison'].get('refinement_base','R3'))
    run_name=experiment_run_name(name,c['seed'])
    if name not in rows or c['run_id']!=c['dataset']+'/'+run_name: raise ValueError('Unknown experiment ID')
    row=rows[name]
    for key,value in row['student'].items():
        if c['student'][key]!=value: raise ValueError('Experiment architecture differs from declared catalog: '+key)
    for key in ['cnn_auxiliary','cnn_residual']:
        if c['student'].get(key,False)!=row['student'].get(key,False): raise ValueError('Undeclared CNN refinement switch: '+key)
    if c['sampling']!=row['sampling'] or c['text']['mode']!=row['text_mode']: raise ValueError('Experiment sampling/text differs from catalog')
    if set(c['text'])!={'mode','prototypes'} or (c['text']['mode']=='none' and c['text']['prototypes'] is not None): raise ValueError('Unexpected text assets in a text-disabled experiment')
    if c['batch_size']!=32 or type(c['workers']) is not int or c['workers']<0 or c['preprocess']!=PREPROCESS: raise ValueError('Unsupported batch/preprocessing')
    if c['init_id']!=model_identity(c['student'],c['seed'],c['head_seed']): raise ValueError('Model initialization identity mismatch')
    out=Path(c['output']); init=Path(c['init_path'])
    if not out.is_absolute() or out.parts[-2:]!=(c['dataset'],run_name) or init!=out.parents[1]/'initializations'/(c['init_id']+'.pt'): raise ValueError('Invalid experiment output/initialization path')
    expected_epochs={'rafdb':60,'sfew':50,'affectnet':30}[c['dataset']]
    expected_stages=['s1','s2'] if row['two_stage'] else ['s2']
    if [stage['name'] for stage in c['stages']]!=expected_stages or sum(stage['epochs'] for stage in c['stages'])!=expected_epochs: raise ValueError('Experiment stage budget changed')
    for stage in c['stages']:
        expected=recipe(ce=0.,feature=1.) if stage['name']=='s1' else row['loss']
        if stage['loss']!=expected or stage['epochs']<1: raise ValueError('Experiment loss recipe changed')
        expected_teacher='t1' if stage['name']=='s1' else ('t2' if expected['feature'] or expected['logits'] or expected['affinity'] else None)
        if stage['teacher']!=expected_teacher: raise ValueError('Teacher dependency differs from objectives')
    dependencies={stage['teacher'] for stage in c['stages'] if stage['teacher']}
    registry=dependencies|({'t1'} if 't2' in dependencies else set())
    if set(c['teachers'])!=registry: raise ValueError('Teacher registry differs from active dependencies and T2 lineage')
    if row['two_stage'] and c['stages'][0]['epochs']!=(10 if c['dataset']=='rafdb' else 5): raise ValueError('S1 budget changed')
    schedule=c['schedule']; total=schedule['steps_per_epoch']*expected_epochs
    if schedule!={'total_updates':total,'steps_per_epoch':schedule['steps_per_epoch'],'warmup_updates':max(1,math.ceil(total*.05)),
                  'min_lr_ratio':.01,'transition_source':'last','optimizer_reset':False,'scheduler_reset':False} or schedule['steps_per_epoch']<1: raise ValueError('Invalid experiment update schedule')
    if c['diagnostics']!={'gradient_interval':1000}: raise ValueError('Unknown diagnostics configuration')
    expected_optimizer={'backbone_lr':1e-5 if c['dataset']=='sfew' else 3e-5,'new_lr':1e-4 if c['dataset']=='sfew' else 3e-4,
                        'weight_decay':.01,'betas':[.9,.98],'eps':1e-6,'grad_clip':1.,'amp':'bf16_if_supported_else_fp32'}
    if c['optimizer']!=expected_optimizer: raise ValueError('Experiment optimizer recipe changed')
    if c['comparison']['parent']!=row['parent'] or c['comparison']['group']!=row['group']: raise ValueError('Comparator identity changed')
    if c['checkpoint']!={'resume_contract':'epoch_boundary_only','selection':['max_uar','max_accuracy','earlier_epoch'],'s1_selection':'min_feature_loss'}: raise ValueError('Checkpoint policy changed')
    if assets:
        from .data import manifest
        from .models import validate_teacher
        m=manifest(c['data'])
        if len(m['samples']['train'])//32!=schedule['steps_per_epoch']: raise ValueError('Train count/update mismatch')
        student=c['student']
        if student['backbone_family']!='cnn':
            spec=student['pretrained']; check_asset(spec)
            if spec.get('model_name')!=student['official_model_name']: raise ValueError('Backbone checkpoint model identity mismatch')
        if student['variant']!='mamba_only': check_asset(student['cnn_pretrained'])
        for key in registry:
            spec=c['teachers'][key]; check_asset(spec)
        for stage in {item['teacher'] for item in c['stages'] if item['teacher']}: validate_teacher(c,stage)
        if c['text']['mode']!='none':
            from .semantics import load_prototypes
            load_prototypes(c)
        if code_identity()!=c['code']: raise ValueError('Code changed; regenerate experiment plan')
    return c
