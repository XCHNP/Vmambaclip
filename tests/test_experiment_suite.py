"""Unified experiment contracts; real SS2D on CPU, explicit Vim interface stubs."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent
from vim_v2.common import CLASSES,read,fingerprint
from vim_v2.config import validate
from vim_v2.experiments import catalog,prepare_experiments
from vim_v2.models import make_student
from vim_v2.losses import objectives,gradient_report
from vim_v2.optim import configure_stage,groups
from vim_v2.runtime import forward_loss
from vim_v2.diagnostics import branch_batches
from vim_v2.semantics import encode_prototypes,audit_semantic_batches

torch.set_num_threads(2)


@pytest.fixture
def suite(tmp_path):
    paths={'output_root':str(tmp_path/'runs'),'vim':{},'vmamba':{},'cnn':{},'clip_base':{'sha256':'clip-test-only'},
           'datasets':{name:{'counts':{'train':320},'manifest':{},'teachers':{}}
                       for name in ['rafdb','sfew','affectnet']},'workers':0}
    path=tmp_path/'paths.json'; path.write_text(json.dumps(paths),encoding='utf-8')
    plan=prepare_experiments(path,tmp_path/'cfg',datasets=['rafdb','sfew','affectnet'])
    configs={row['run_id']:read(row['config']) for row in plan['runs']}
    return path,plan,configs


def test_whole_catalog_has_explicit_budgets_and_deduplicated_aliases(suite,tmp_path):
    path,plan,configs=suite
    assert len(plan['runs'])==3*len(catalog())==90
    for config in configs.values():
        validate(config)
        assert sum(stage['epochs'] for stage in config['stages'])=={'rafdb':60,'sfew':50,'affectnet':30}[config['dataset']]
    assert configs['rafdb/F_recommended']['init_id']==configs['rafdb/T1']['init_id']==configs['rafdb/T3_w0p1']['init_id']
    assert configs['rafdb/E3']['init_id']!=configs['rafdb/E4']['init_id']
    small=prepare_experiments(path,tmp_path/'small',experiments=['D_ce','F_recommended','T2','T2_w0p1'])
    assert [row['run_id'] for row in small['runs']]==['rafdb/F_recommended','rafdb/T2_w0p1']
    with pytest.raises(FileExistsError): prepare_experiments(path,tmp_path/'small')
    changed=copy.deepcopy(configs['rafdb/T1']); changed['stages'][0]['loss']['text']=.1
    with pytest.raises(ValueError,match='recipe'): validate(changed)
    changed=copy.deepcopy(configs['rafdb/E4']); changed['optimizer']['new_lr']=1.
    with pytest.raises(ValueError,match='optimizer'): validate(changed)


@pytest.mark.parametrize('name',[name for name in catalog() if name not in ['E1','E2']])
def test_every_vmamba_cnn_preset_runs_through_the_shared_builder(name,suite):
    config=suite[2]['rafdb/'+name]
    model,report=make_student(config,initialize=False)
    configure_stage(model,'s2'); model.eval()
    with torch.no_grad():
        output=model(torch.randn(2,3,32,32))
    assert output.fused_raw.shape==(2,768) and output.logits.shape==(2,7)
    assert torch.isfinite(output.logits).all()
    if name=='B_cnn': assert not any('vim' in key or 'mamba' in key for key in model.state_dict())
    if name=='B_vmamba': assert model.cnn_backbone is None and not any(key.startswith('cnn_') for key in model.state_dict())
    if model.__class__.__name__!='CNNOnlyStudent' and config['student']['use_sasf']:
        assert model.sasf.in_dim=={'e':192,'m':384,'l':768}[config['student']['sasf_stage']]
    if config['student']['cnn_multiscale']:
        assert output.patch_norm.shape[1]==16  # 32px input -> layer2 4x4 K/V
    assert report['head_rng_policy'].startswith('stable per module')


class VimStub(nn.Module):
    """Token interface only; never reported as official Vim kernel validation."""
    def __init__(self):
        super().__init__(); self.proj=nn.Conv2d(3,384,16,stride=16)
    def forward(self,images):
        tokens=self.proj(images).flatten(2).transpose(1,2)
        return {'e':tokens,'m':tokens*2,'l':tokens*3}
    def patch_grid(self,images): return images.shape[-2]//16,images.shape[-1]//16


@pytest.mark.parametrize('name',['E1','E2'])
def test_vim_e1_e2_share_the_same_student_path_with_explicit_interface_stub(name,suite):
    fake=nn.Module(); fake.set_grad_checkpointing=lambda enabled:None
    with patch('open_clip.vision_mamba_clip.official_vim.create_official_vim',return_value=(fake,{'pretrained_loaded':False,'commit':'dd0358ad1e42701f22afbefa0717cc8825cf9f45'})),\
         patch('open_clip.vision_mamba_clip.official_vim.OfficialPatchAdapter',return_value=VimStub()):
        model,_=make_student(suite[2]['rafdb/'+name],initialize=False)
    model.eval()
    with torch.no_grad(): output=model(torch.randn(2,3,32,32))
    assert output.logits.shape==(2,7)
    assert model.use_sasf==(name=='E2')


def test_common_parameters_do_not_shift_when_sasf_is_added(suite):
    model,_=make_student(suite[2]['rafdb/E3'],initialize=False)
    baseline={name:value.clone() for name,value in model.state_dict().items() if not name.startswith(('vim.','cnn_backbone.'))}
    del model
    enhanced,_=make_student(suite[2]['rafdb/E4'],initialize=False)
    for name,value in baseline.items(): torch.testing.assert_close(value,enhanced.state_dict()[name],rtol=0,atol=0)


def test_mid_sasf_multiscale_multiquery_and_all_new_parameters_have_gradients(suite):
    config=suite[2]['rafdb/F_recommended']; model,_=make_student(config,initialize=False)
    model.query_mode='multi'; configure_stage(model,'s2')
    with torch.no_grad(): model.sasf_alpha.fill_(.2)
    seen=[]
    handle=model.local_cross_attention.register_forward_pre_hook(lambda module,args:seen.append([tuple(value.shape) for value in args[:3]]))
    output=model(torch.randn(2,3,32,32)); loss=F.cross_entropy(output.logits,torch.tensor([0,4])); loss.backward()
    handle.remove()
    assert seen==[[(2,3,768),(2,16,768),(2,16,768)]]
    names={name for group in groups(model,config['optimizer']) for name in group['names']}
    for prefix in ['sasf.','cnn_lateral.','local_cross_attention.']:
        for name,p in model.named_parameters():
            if name.startswith(prefix):
                assert name in names and p.grad is not None and torch.isfinite(p.grad).all()
    assert model.cnn_lateral.weight.grad.norm()>0
    assert model.sasf.branches[0][0].weight.grad.norm()>0


def test_dropout_only_changes_training_kv_and_keeps_spatial_sasf_grid(suite):
    model,_=make_student(suite[2]['rafdb/A_patchdrop'],initialize=False)
    images=torch.randn(2,3,32,32); model.train(); output=model(images)
    assert int(output.diagnostics['cnn_kv_tokens'])==11  # floor(16 * .7)
    model.eval()
    with torch.no_grad(): output=model(images)
    assert int(output.diagnostics['cnn_kv_tokens'])==16


def test_cosine_mse_affinity_and_frozen_prototype_math():
    feature=torch.randn(3,8,requires_grad=True); target=torch.randn(3,8,requires_grad=True)
    logits=torch.randn(3,7,requires_grad=True); labels=torch.tensor([2,2,5]); bank=torch.randn(7,8,requires_grad=True)
    weights={'ce':1.,'feature':.2,'feature_cosine':1.,'feature_mse':.5,'feature_mse_space':'raw',
             'logits':.2,'text':.1,'text_scale':10.,'affinity':.1,'temperature':2.}
    teacher_logits=torch.randn(3,7,requires_grad=True)
    total,raw,weighted=objectives(logits,feature,labels,weights,{'feature':target,'logits':teacher_logits},bank)
    torch.testing.assert_close(raw['feature_mse'],F.mse_loss(feature,target.detach()))
    torch.testing.assert_close(weighted['feature_mse'],raw['feature_mse']*.1)
    expected_text=F.cross_entropy(10*F.normalize(feature,dim=-1)@F.normalize(bank.detach(),dim=-1).T,labels)
    torch.testing.assert_close(raw['text'],expected_text)
    total.backward()
    assert target.grad is None and teacher_logits.grad is None and bank.grad is None
    assert feature.grad.norm()>0 and logits.grad.norm()>0


def test_duplicate_class_prompts_do_not_create_in_batch_false_negatives():
    weights={'ce':0.,'feature':0.,'logits':0.,'text':1.,'text_scale':10.}
    feature=torch.randn(1,8,requires_grad=True); bank=torch.randn(7,8); logits=torch.zeros(1,7)
    first=objectives(logits,feature,torch.tensor([3]),weights,prototypes=bank)[0]
    repeated=objectives(logits.expand(4,-1),feature.expand(4,-1),torch.tensor([3,3,3,3]),weights,prototypes=bank)[0]
    torch.testing.assert_close(first,repeated)


def test_gradient_conflicts_are_measured_without_parameter_updates():
    shared=nn.Parameter(torch.tensor([1.,2.]))
    report=gradient_report({'left':shared.sum(),'opposite':-shared.sum(),'zero':shared.sum()*0},shared)
    assert report['cosines']['left__opposite']==pytest.approx(-1.)
    assert report['cosines']['left__zero'] is None
    assert shared.grad is None


def test_prototypes_average_raw_embeddings_then_normalize():
    class TextEncoder:
        def encode_text(self,tokens,normalize=False):
            result=torch.zeros(len(tokens),768); result[:,0]=tokens[:,0].float()+1; result[:,1]=1
            return result
    schema={'class_order':CLASSES,'source':'unit-test','prompts':{name:[name+'0',name+'1'] for name in CLASSES}}
    bank=encode_prototypes(TextEncoder(),lambda texts:torch.arange(len(texts)).reshape(-1,1),schema,'cpu')
    expected=torch.zeros(768); expected[0]=1.5; expected[1]=1.
    torch.testing.assert_close(bank[0],F.normalize(expected,dim=0))


def test_semantic_audit_reports_head_and_text_separately_without_updates():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__(); self.classifier=nn.Linear(768,7)
            nn.init.zeros_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)
        def encode_image(self,images): return images
    teacher=Teacher(); teacher.train(); before=copy.deepcopy(teacher.state_dict())
    bank=torch.eye(7,768); batch={'image_teacher':bank,'label':torch.arange(7)}
    report=audit_semantic_batches(teacher,[batch],bank,torch.device('cpu'))
    assert report['teacher_text_prototypes']['accuracy']==1.
    assert report['teacher_classifier']['accuracy']==pytest.approx(1/7)
    assert report['margin_mean']==1. and teacher.training
    for name,value in before.items(): torch.testing.assert_close(teacher.state_dict()[name],value,rtol=0,atol=0)


def test_branch_interventions_cover_classifier_bypasses_and_restore_modes():
    from test_sasf import student
    model=student(use_sasf=True,fusion='mamba_cnn_cross_attention'); model.train(); model.classifier.batch_norm.eval()
    with torch.no_grad(): model.sasf_alpha.fill_(.3)
    images=torch.randn(7,3,16,24)
    before=copy.deepcopy(model.state_dict())
    report=branch_batches(model,[{'image_student':images,'label':torch.arange(7)}],torch.device('cpu'))
    values=report['interventions']
    assert values['shuffle_cnn']['probability_l1_per_sample']<1e-6
    assert values['no_cnn']['branch_statistics']['cnn_rms']==0.
    assert values['no_mamba']['branch_statistics']['mamba_rms']==0.
    assert values['no_sasf']['branch_statistics']['sasf_delta_rms']==0.
    assert model.training and not model.classifier.batch_norm.training
    for name,value in before.items(): torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)


def test_aggregate_supports_selected_experiments_without_hardcoded_18_run_matrix(suite):
    from vim_v2.__main__ import aggregate
    _,plan,_=suite
    path=Path(plan['runs'][0]['config']).parent/'plan.json'
    result=aggregate(path)
    assert len(result['results'])==90 and result['comparisons']
    assert (Path(plan['output_root'])/'reports/per_class.csv').exists()


def test_real_forward_loss_connects_kd_text_and_gradient_diagnostics(suite):
    config=suite[2]['rafdb/T3_w0p1']; model,_=make_student(config,initialize=False)
    configure_stage(model,'s2')
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=nn.Linear(3,768); self.classifier=nn.Linear(768,7)
        def encode_image(self,image): return self.encoder(image.mean((2,3)))
    teacher=Teacher().requires_grad_(False)
    bank=torch.randn(7,768,requires_grad=True); model._text_prototypes=bank
    batch={'image_student':torch.randn(2,3,32,32),'image_teacher':torch.randn(2,3,32,32),'label':torch.tensor([1,1])}
    loss,logits,raw,weighted=forward_loss(model,teacher,batch,config['stages'][0]['loss'],torch.device('cpu'))
    assert set(raw)=={'ce','logits','text'}
    report=gradient_report(weighted,model.fusion_projection.weight)
    assert all(report['norms'][key]>0 for key in ['ce','logits','text'])
    loss.backward()
    assert bank.grad is None and all(p.grad is None for p in teacher.parameters())
    assert model.sasf_alpha.grad is not None


def test_t2_configs_preserve_t1_lineage_and_missing_assets_fail_preflight(suite):
    from vim_v2.__main__ import preflight
    _,plan,configs=suite
    assert set(configs['rafdb/T1']['teachers'])=={'t1','t2'}
    assert configs['rafdb/E4']['teachers']=={}
    config=configs['rafdb/E4']
    with patch('vim_v2.data.manifest',return_value={'samples':{'train':[{}]*320}}):
        with pytest.raises(ValueError,match='Asset must have explicit path'):
            validate(config,assets=True)


def test_text_student_checkpoint_inference_needs_no_text_or_teacher_assets(suite,tmp_path):
    from open_clip.vision_mamba_clip.inference import load_student_for_inference
    config=suite[2]['rafdb/T2_w0p1']; model,_=make_student(config,initialize=False)
    model.vim.set_grad_checkpointing(False); model.eval()
    model._text_prototypes=torch.randn(7,768)
    image=torch.randn(2,3,32,32)
    with torch.no_grad(): expected=model(image).logits
    state=model.state_dict()
    assert not any('_text_prototypes' in key for key in state)
    path=tmp_path/'text_student.pt'
    torch.save({'identity':{'config':config,'config_sha256':fingerprint(config)},'model':state},path)
    with patch('vim_v2.official_experiments.check_asset',side_effect=AssertionError('Inference must not read pretrained assets')),\
         patch('vim_v2.semantics.check_asset',side_effect=AssertionError('Inference must not read text assets')):
        restored,order,_,_=load_student_for_inference(str(path))
    with torch.no_grad(): torch.testing.assert_close(restored(image).logits,expected,rtol=0,atol=0)
    assert order==CLASSES


def test_prototype_loader_rejects_wrong_source_or_class_order(suite,tmp_path):
    from vim_v2.semantics import load_prototypes,PROMPTS
    from vim_v2.common import sha
    config=copy.deepcopy(suite[2]['rafdb/T2_w0p1'])
    metadata={'class_order':CLASSES,'mode':'simple','clip_sha256':'clip-test-only','prompt_sha256':sha(PROMPTS['simple'])}
    path=tmp_path/'bank.pt'; torch.save({'prototypes':torch.randn(7,768),'metadata':metadata},path)
    config['text']['prototypes']={'path':str(path),'sha256':sha(path)}
    assert load_prototypes(config).shape==(7,768)
    metadata['class_order']=list(reversed(CLASSES))
    torch.save({'prototypes':torch.randn(7,768),'metadata':metadata},path)
    config['text']['prototypes']['sha256']=sha(path)
    with pytest.raises(ValueError,match='identity'): load_prototypes(config)


def test_balanced_sampler_is_epoch_deterministic_and_uses_train_counts_only(tmp_path):
    from PIL import Image
    from vim_v2.data import loader,class_weights
    rows=[]
    for index,label in enumerate([0]*10+list(range(1,7))):
        path=str(index)+'.png'; Image.new('RGB',(32,32),(index,0,0)).save(tmp_path/path)
        rows.append({'path':path,'label':label})
    manifest={'root':str(tmp_path),'samples':{'train':rows,'val':rows}}
    config={'data':{},'seed':42,'batch_size':4,'workers':0,'sampling':'balanced'}
    with patch('vim_v2.data.manifest',return_value=manifest):
        a=loader(config,'train',epoch=0); b=loader(config,'train',epoch=0); c=loader(config,'train',epoch=1)
        assert list(a.sampler)==list(b.sampler) and list(a.sampler)!=list(c.sampler)
        assert loader(config,'val').sampler.__class__.__name__=='SequentialSampler'
    weights=class_weights(rows)
    assert weights[0]*10==pytest.approx(float(weights[1]))


def test_new_multiquery_checkpoint_optimizer_and_rng_resume(suite,tmp_path):
    from vim_v2.checkpoint import atomic_save,load,rng_state,restore
    from vim_v2.optim import UpdateCosine
    config=suite[2]['rafdb/Q_multi']; model,_=make_student(config,initialize=False)
    configure_stage(model,'s2'); optimizer=torch.optim.AdamW(groups(model,config['optimizer']))
    scheduler=UpdateCosine(optimizer,10,1); scaler=torch.amp.GradScaler('cpu',enabled=False)
    image=torch.randn(2,3,32,32); labels=torch.tensor([0,4])
    def step():
        scheduler.prepare(); optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(image).logits,labels).backward(); optimizer.step(); scheduler.step()
    step(); path=tmp_path/'resume.pt'
    identity={'config_sha256':fingerprint(config)}
    atomic_save(path,{'kind':'resume','identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'next_batch':0,'global_update':1})
    step(); expected={name:value.clone() for name,value in model.state_dict().items()}
    restore(load(path),identity,model,optimizer,scheduler,scaler); step()
    for name,value in expected.items(): torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
