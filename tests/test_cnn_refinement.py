"""R0–R6 factors, independent CNN supervision, diagnostics and seed-safe plans."""

import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_sasf import student
from vim_v2.common import read,fingerprint
from vim_v2.config import validate
from vim_v2.data import effective_number_weights
from vim_v2.diagnostics import branch_batches
from vim_v2.experiments import catalog,prepare_experiments
from vim_v2.losses import objectives
from vim_v2.models import make_student
from vim_v2.optim import branch_gradient_stats,configure_stage,groups
from vim_v2.runtime import forward_loss,evaluate_model
from vim_v2.statistics import paired_bootstrap

torch.set_num_threads(2)


def paths_file(tmp_path):
    paths={'output_root':str(tmp_path/'runs'),'vmamba':{},'cnn':{},'clip_base':{},'workers':0,
           'datasets':{'rafdb':{'counts':{'train':320},'manifest':{},'teachers':{}}}}
    path=tmp_path/'paths.json'; path.write_text(json.dumps(paths),encoding='utf-8')
    return path


@pytest.mark.parametrize('base',['R0','R1','R2','R3'])
def test_refinements_are_independent_and_followup_base_is_explicit(tmp_path,base):
    plan=prepare_experiments(paths_file(tmp_path),tmp_path/'cfg',experiments=['R0','R1','R2','R3','R4','R5','R6'],refinement_base=base,seeds=[42,43,44])
    configs={row['run_id']:read(row['config']) for row in plan['runs']}
    assert len(configs)==21
    for c in configs.values(): validate(c)
    assert configs['rafdb/R1']['student']['cnn_auxiliary'] and not configs['rafdb/R1']['student']['cnn_residual']
    assert not configs['rafdb/R2']['student']['cnn_auxiliary'] and configs['rafdb/R2']['student']['cnn_residual']
    assert configs['rafdb/R3']['stages'][0]['loss']['cnn_aux']==.2
    for name in ['R4','R5','R6']:
        c=configs['rafdb/'+name]
        assert c['comparison']['refinement_base']==base
        assert c['student']['cnn_multiscale'] is False and c['student']['query_mode']=='single'
    baseline=configs['rafdb/'+('E3' if base=='R0' else base)]
    assert configs['rafdb/R4']['init_id']==baseline['init_id']==configs['rafdb/R5']['init_id']
    assert configs['rafdb/R4']['sampling']=='class_balanced_ce'
    assert configs['rafdb/R5']['stages'][0]['loss']['text']==.05
    assert configs['rafdb/R6']['student']['use_sasf'] and configs['rafdb/R6']['student']['sasf_stage']=='m'
    assert len({configs['rafdb/R1'+suffix]['init_id'] for suffix in ['', '_seed43','_seed44']})==3
    changed=copy.deepcopy(configs['rafdb/R1_seed43']); changed['head_seed']=1042
    with pytest.raises(ValueError): validate(changed)
    # The old E3 remains a genuine no-SASF/no-added-head control.
    assert not configs['rafdb/E3']['student'].get('cnn_auxiliary',False)
    invalid=copy.deepcopy(configs['rafdb/E3']); invalid['student']['cnn_auxiliary']=True
    with pytest.raises(ValueError,match='Undeclared CNN'): validate(invalid)


def test_auxiliary_only_does_not_modify_e3_main_outputs_or_old_state_dict():
    torch.manual_seed(42); baseline=student(fusion='mamba_cnn_cross_attention').eval()
    torch.manual_seed(42); aux=student(fusion='mamba_cnn_cross_attention',cnn_auxiliary=True).eval()
    for name,value in baseline.state_dict().items():
        torch.testing.assert_close(value,aux.state_dict()[name],rtol=0,atol=0)
    image=torch.randn(3,3,16,24)
    with torch.no_grad():
        first=baseline(image); second=aux(image)
        torch.testing.assert_close(first.logits,second.logits,rtol=0,atol=0)
        assert second.cnn_aux_logits is None
        assert aux(image,return_aux=True).cnn_aux_logits.shape==(3,7)


def test_auxiliary_loss_is_independent_of_mamba_and_spatial_attention():
    model=student(fusion='mamba_cnn_cross_attention',cnn_auxiliary=True)
    output=model(torch.randn(3,3,16,24))
    F.cross_entropy(output.cnn_aux_logits,torch.tensor([0,3,5])).backward()
    assert model.cnn_backbone.conv.weight.grad.norm()>0
    assert model.cnn_projection.weight.grad.norm()>0
    assert all(p.grad is None for p in model.vim.parameters())
    assert all(p.grad is None for p in model.spatial_attention.parameters())
    assert all(p.grad is None for p in model.local_cross_attention.parameters())
    assert model.fusion_projection.weight.grad is None
    stats=branch_gradient_stats(model)
    assert stats['cnn_grad_norm']>0 and stats['vmamba_grad_norm']==0
    assert stats['vmamba']['gradient_none_tensors']==stats['vmamba']['trainable_tensors']


def test_gate_starts_at_point_two_and_residual_formula_is_exact():
    model=student(fusion='mamba_cnn_cross_attention',cnn_residual=True).eval()
    old=torch.randn(3,8); tokens=torch.randn(3,12,8)
    fused,aux,gate,global_cnn=model.cnn_residual_aux(old,tokens)
    assert float(gate)==pytest.approx(.2,abs=1e-7) and aux is None
    expected=model.cnn_residual_aux.fusion_norm(old+gate*model.cnn_residual_aux.cnn_norm(tokens.mean(1)))
    torch.testing.assert_close(fused,expected)
    global_cnn.sum().backward()
    assert model.cnn_residual_aux.cnn_norm.weight.grad is not None


def test_no_cnn_intervention_removes_new_residual_bypass():
    model=student(fusion='mamba_cnn_cross_attention',cnn_auxiliary=True,cnn_residual=True).eval()
    image=torch.randn(2,3,16,24)
    with torch.no_grad():
        reference=model(image,intervention='no_cnn').logits
        for parameter in model.cnn_backbone.parameters(): parameter.add_(torch.randn_like(parameter)*10)
        actual=model(image,intervention='no_cnn')
    torch.testing.assert_close(reference,actual.logits,rtol=0,atol=0)
    assert actual.diagnostics['cnn_independent_rms']==0


def test_auxiliary_ce_is_weighted_once_and_disabled_input_is_not_read():
    main=torch.randn(3,7,requires_grad=True); auxiliary=torch.randn(3,7,requires_grad=True); y=torch.tensor([1,1,5])
    weights={'ce':1.,'feature':0.,'logits':0.,'cnn_aux':.2}
    total,raw,weighted=objectives(main,None,y,weights,cnn_aux_logits=auxiliary)
    torch.testing.assert_close(total,F.cross_entropy(main,y)+.2*F.cross_entropy(auxiliary,y))
    assert set(raw)=={'ce','cnn_aux'}
    total.backward(); assert main.grad.norm()>0 and auxiliary.grad.norm()>0
    class Poison:
        def __getattribute__(self,name): raise AssertionError('Disabled auxiliary head accessed')
    objectives(main,None,y,dict(weights,cnn_aux=0.),cnn_aux_logits=Poison())
    with pytest.raises(ValueError,match='auxiliary'): objectives(main,None,y,weights)


def test_effective_sample_weights_match_formula_and_train_labels_only():
    counts=[100,3,8,1000,30,50,7]
    rows=[{'label':label} for label,count in enumerate(counts) for _ in range(count)]
    actual=effective_number_weights(rows,.999)
    expected=(1-.999)/(1-torch.pow(torch.tensor(.999,dtype=torch.float64),torch.tensor(counts)))
    torch.testing.assert_close(actual,(expected/expected.mean()).float())
    torch.testing.assert_close(effective_number_weights(rows,0.),torch.ones(7))
    with pytest.raises(ValueError): effective_number_weights([{'label':0}])


def test_auxiliary_metrics_available_in_runtime_but_not_normal_inference():
    model=student(fusion='mamba_cnn_cross_attention',cnn_auxiliary=True)
    batch={'image_student':torch.randn(7,3,16,24),'label':torch.arange(7)}
    model.train()
    loss,*_=forward_loss(model,None,batch,{'ce':1.,'feature':0.,'logits':0.,'cnn_aux':.2},torch.device('cpu'))
    loss.backward()
    result=evaluate_model(model,[batch],torch.device('cpu'))
    assert 'uar' in result['cnn_aux_metrics'] and len(result['cnn_aux_metrics']['per_class'])==7
    assert model.training
    configure_stage(model,'s1')
    assert not model.compute_aux_logits
    assert not model.cnn_residual_aux.cnn_aux_head.weight.requires_grad
    configure_stage(model,'s2'); assert model.compute_aux_logits
    model.eval()
    with torch.no_grad(): assert model(batch['image_student']).cnn_aux_logits is None


def test_per_sample_diagnostics_distinguish_logit_probability_and_prediction_changes():
    model=student(fusion='mamba_cnn_cross_attention',cnn_residual=True,use_sasf=True)
    images=torch.randn(7,3,16,24); samples=[str(i) for i in range(7)]
    result=branch_batches(model,[{'image_student':images,'label':torch.arange(7),'sample_id':samples}],torch.device('cpu'))
    assert len(result['per_sample'])==42
    shuffled=result['interventions']['shuffle_cnn_values']
    assert all(key in shuffled for key in ['probability_delta_max','logits_delta_mean','changed_predictions'])
    assert result['interventions']['shuffle_cnn']['probability_delta_max']<1e-6


def write_predictions(path,predictions,reverse=False):
    rows=[]
    for index,prediction in enumerate(predictions):
        probs=[.01]*7; probs[prediction]=.94
        rows.append([str(index),index%7,prediction,*probs])
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['sample_id','true_label','predicted_label']+['p'+str(i) for i in range(7)])
        writer.writerows(list(reversed(rows)) if reverse else rows)


def test_paired_bootstrap_pairs_by_id_and_detects_mismatched_samples(tmp_path):
    a=tmp_path/'a.csv'; b=tmp_path/'b.csv'
    labels=[i%7 for i in range(70)]
    write_predictions(a,labels); write_predictions(b,labels,reverse=True)
    result=paired_bootstrap(a,b,resamples=20)
    assert all(value['ci95']==[0.,0.] for value in result['delta_percentage_points'].values())
    wrong=labels.copy(); wrong[0]=1; write_predictions(a,wrong)
    result=paired_bootstrap(a,b,resamples=20)
    assert result['delta_percentage_points']['accuracy']['candidate_minus_baseline']>0
    write_predictions(b,labels[:14])
    with pytest.raises(ValueError,match='sample IDs'): paired_bootstrap(a,b,resamples=20)


def test_seed_aggregate_uses_comparators_from_the_same_training_seed(tmp_path):
    from vim_v2.__main__ import aggregate
    prepare_experiments(paths_file(tmp_path),tmp_path/'cfg',experiments=['R0','R1'],seeds=[42,43,44])
    result=aggregate(tmp_path/'cfg/plan.json')
    assert result['comparisons']['rafdb/R1_seed43_vs_E3']['right']=='rafdb/E3_seed43'
    assert result['seed_summary']['rafdb/R1']['planned_seeds']==[42,43,44]
    assert result['seed_summary']['rafdb/R1']['completed_seeds']==[]


def test_real_vmamba_refinement_roundtrip_preserves_inference_without_aux(tmp_path):
    from open_clip.vision_mamba_clip.inference import load_student_for_inference
    plan=prepare_experiments(paths_file(tmp_path),tmp_path/'cfg',experiments=['R3'])
    config=read(plan['runs'][0]['config']); model,_=make_student(config,initialize=False)
    configure_stage(model,'s2'); image=torch.randn(2,3,32,32); y=torch.tensor([0,4])
    output=model(image)
    loss=F.cross_entropy(output.logits,y)+.2*F.cross_entropy(output.cnn_aux_logits,y); loss.backward()
    assert model.cnn_residual_aux.cnn_gate_logit.grad.abs()>0
    assert model.cnn_residual_aux.cnn_aux_head.weight.grad.norm()>0
    model.vim.set_grad_checkpointing(False); model.eval()
    with torch.no_grad(): expected=model(image).logits
    path=tmp_path/'student.pt'
    torch.save({'model':model.state_dict(),'identity':{'config':config,'config_sha256':fingerprint(config)}},path)
    restored,*_=load_student_for_inference(str(path))
    with torch.no_grad():
        actual=restored(image)
        torch.testing.assert_close(expected,actual.logits,rtol=0,atol=0)
        assert actual.cnn_aux_logits is None
