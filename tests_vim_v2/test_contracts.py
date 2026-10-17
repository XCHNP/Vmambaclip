import copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
from vim_v2.losses import objectives,coefficients
from vim_v2.models import interpolate_middle,strict_vim_load
from vim_v2.optim import UpdateCosine,configure_stage,groups
from vim_v2.checkpoint import atomic_save,load,rng_state,restore_rng,restore,transfer
from vim_v2.common import sha,seed_all,tensor_digest

class Contracts(unittest.TestCase):
    def test_exact_recovered_state_gate(self):
        from vim_v2.acceptance import assert_exact_tree
        state={'optimizer':{'moment':torch.tensor([1.,2.]),'step':3},'rng':[torch.tensor([42],dtype=torch.uint8)]}
        assert_exact_tree(copy.deepcopy(state),state)
        wrong=copy.deepcopy(state); wrong['optimizer']['moment'][0]+=1e-6
        with self.assertRaises(ValueError): assert_exact_tree(wrong,state)
        wrong=copy.deepcopy(state); wrong['rng'][0][0]=43
        with self.assertRaises(ValueError): assert_exact_tree(wrong,state)
        wrong=copy.deepcopy(state); wrong['optimizer']['step']=4
        with self.assertRaises(ValueError): assert_exact_tree(wrong,state)
    def test_gradient_equivalence_gate(self):
        from vim_v2.acceptance import compare_gradient_states
        a={'grad':torch.tensor([1.,-1.,0.])}
        result=compare_gradient_states(a,{'grad':torch.tensor([1.0001,-1.,1e-6])})
        self.assertLess(result['max_relative_l2'],5e-4)
        with self.assertRaises(ValueError): compare_gradient_states(a,{'grad':torch.tensor([1.01,-1.,0.])})
        with self.assertRaises(ValueError): compare_gradient_states({'grad':torch.zeros(3)},{'grad':torch.ones(3)*1e-6})
        with self.assertRaises(ValueError): compare_gradient_states(a,{})
        with self.assertRaises(ValueError): compare_gradient_states(a,{'grad':torch.tensor([float('nan'),0.,0.])})
    def test_forward_kl_math_and_teacher_detach(self):
        torch.manual_seed(2); s=torch.randn(5,7,requires_grad=True); t=torch.randn(5,7,requires_grad=True)
        w={'ce':0.,'feature':0.,'logits':.2,'temperature':2.}
        loss,raw,_=objectives(s,None,None,w,{'logits':t})
        pt=(t.detach()/2).softmax(-1)
        expected=(pt*((t.detach()/2).log_softmax(-1)-(s/2).log_softmax(-1))).sum(-1).mean()*4
        torch.testing.assert_close(raw['logits'],expected); loss.backward()
        self.assertIsNone(t.grad); self.assertGreater(float(s.grad.norm()),0.)
    def test_zero_terms_never_read(self):
        class Poison:
            def __getattribute__(self,name): raise AssertionError('Disabled branch accessed')
        logits=torch.randn(3,7,requires_grad=True); labels=torch.tensor([0,1,2])
        loss,raw,_=objectives(logits,Poison(),labels,{'ce':1.,'feature':0.,'logits':0.,'temperature':2.},Poison())
        torch.testing.assert_close(loss,F.cross_entropy(logits,labels)); self.assertEqual(list(raw),['ce'])
    def test_feature_no_ce_head(self):
        f=torch.randn(4,768,requires_grad=True); t=f.detach().clone().requires_grad_()
        loss,_,_=objectives(None,f,None,coefficients('full_v2','s1'),{'feature':t})
        self.assertLess(abs(float(loss)),1e-6); loss.backward(); self.assertIsNone(t.grad)
    def test_middle_cls_grid(self):
        p=torch.arange(197.).reshape(1,197,1).expand(1,197,384).clone(); p[:,98]=10000
        out=interpolate_middle(p,256)
        torch.testing.assert_close(out[:,128],p[:,98]); self.assertEqual(out.shape,(1,257,384))
        patches=torch.cat([p[:,:98],p[:,99:]],1).reshape(1,14,14,384).permute(0,3,1,2)
        expected=F.interpolate(patches,size=(16,16),mode='bicubic',align_corners=False).permute(0,2,3,1).reshape(1,256,384)
        torch.testing.assert_close(torch.cat([out[:,:128],out[:,129:]],1),expected)
        torch.testing.assert_close(interpolate_middle(p,196),p)
        with self.assertRaises(ValueError): interpolate_middle(torch.zeros(1,730,384),256)
    def test_unrelated_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.pt'; torch.save({'model':{'unrelated':torch.ones(1)}},p)
            with self.assertRaises(ValueError): strict_vim_load(None,{'path':str(p),'sha256':sha(p),'model_name':'wrong'})
    def test_no_decay_and_fixed_head_groups(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.backbone=torch.nn.Linear(3,3); self.classifier=torch.nn.Linear(3,7); self.backbone.weight._no_weight_decay=True
        m=Toy(); cfg={'backbone_lr':3e-5,'new_lr':3e-4,'weight_decay':.01}; pg=groups(m,cfg)
        configure_stage(m,'s1'); self.assertFalse(m.classifier.weight.requires_grad); self.assertFalse(m.classifier.training)
        self.assertTrue(any(m.classifier.weight is p for g in pg for p in g['params']))
        self.assertEqual(next(g for g in pg if 'backbone.weight' in g['names'])['weight_decay'],0.)
        configure_stage(m,'s2'); self.assertTrue(m.classifier.weight.requires_grad); self.assertTrue(m.classifier.training)
    def test_full_continuous_schedule(self):
        m=torch.nn.Linear(2,2); opt=torch.optim.AdamW([{'params':m.parameters(),'lr':.1,'peak_lr':.1}]); s=UpdateCosine(opt,100,5)
        self.assertAlmostEqual(s.factor(4),1.); self.assertAlmostEqual(s.factor(99),.01)
        for _ in range(50): s.prepare(); s.step()
        state=s.state_dict(); fresh=UpdateCosine(opt,100,5); fresh.load_state_dict(state); self.assertEqual(fresh.updates,50)
        with self.assertRaises(ValueError): UpdateCosine(opt,200,10).load_state_dict(state)
    def test_atomic_failure_preserves_previous(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'; atomic_save(p,{'v':torch.tensor([1])}); old=sha(p)
            with patch('vim_v2.checkpoint.os.replace',side_effect=OSError('injected failure')):
                with self.assertRaises(OSError): atomic_save(p,{'v':torch.tensor([2])})
            self.assertEqual(old,sha(p)); self.assertEqual(list(Path(d).iterdir()),[p])
    def test_resume_cpu_actual_adam_rng(self):
        seed_all(42); m=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.3),torch.nn.Linear(5,2)); x=torch.randn(4,3)
        opt=torch.optim.AdamW([{'params':m.parameters(),'lr':.001,'peak_lr':.001}]); s=UpdateCosine(opt,10,1); scaler=torch.amp.GradScaler('cpu',enabled=False)
        def step():
            s.prepare(); opt.zero_grad(); m(x).square().mean().backward(); opt.step(); s.step()
        step()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'; atomic_save(p,{'kind':'resume','identity':{'id':1},'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':s.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'next_batch':0,'global_update':1})
            step(); expected=tensor_digest(m.state_dict()); saved=load(p)
            restore(saved,{'id':1},m,opt,s,scaler); step(); self.assertEqual(expected,tensor_digest(m.state_dict()))
            with self.assertRaises(ValueError): restore(saved,{'id':2},m,opt,s,scaler)
            saved['next_batch']=1
            with self.assertRaises(ValueError): restore(saved,{'id':1},m,opt,s,scaler)
    def test_shared_geometry_normalize(self):
        from PIL import Image
        from vim_v2.data import PairedDataset,IM_MEAN,IM_STD,CLIP_MEAN,CLIP_STD
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(300,270),(50,100,150)).save(Path(d)/'a.png')
            ds=PairedDataset({'root':d,'samples':{'train':[{'path':'a.png','label':2}]}},'train')
            b=ds[0]; s=b['image_student']*torch.tensor(IM_STD)[:,None,None]+torch.tensor(IM_MEAN)[:,None,None]
            t=b['image_teacher']*torch.tensor(CLIP_STD)[:,None,None]+torch.tensor(CLIP_MEAN)[:,None,None]
            torch.testing.assert_close(s,t); torch.testing.assert_close(ds[0]['image_student'],b['image_student'])
    def test_validation_restores_modes(self):
        from vim_v2.runtime import evaluate_model
        m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,7)); m.train(); m[1].eval()
        b={'image_student':torch.randn(7,3,2,2),'label':torch.arange(7)}
        evaluate_model(m,[b],torch.device('cpu'))
        self.assertTrue(m.training); self.assertFalse(m[1].training)
    def test_transfer_is_not_resume(self):
        with self.assertRaises(ValueError): transfer()
    def test_finite_core_dag_and_schema_rejects_changes(self):
        from vim_v2.config import prepare,validate
        from vim_v2.common import atomic_json,read,MODEL
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            paths={'vim':{'path':'missing','sha256':'not-real','model_name':MODEL},'cnn':None,'clip_base':None,'output_root':str(root/'runs'),
                   'datasets':{domain:{'counts':{'train':count},'manifest':{},'teachers':{'t1':{'path':'missing'},'t2':{'path':'missing'}}} for domain,count in [('rafdb',11040),('sfew',763),('affectnet',255507)]}}
            atomic_json(root/'paths.json',paths)
            with patch('vim_v2.config.code_identity',return_value={'test_fixture':'not production source'}): plan=prepare(root/'paths.json',root/'cfg')
            self.assertEqual(len(plan['runs']),18); self.assertEqual(sum(r['stages'] for r in plan['runs']),21)
            configs={r['run_id']:read(r['config']) for r in plan['runs']}
            self.assertEqual(configs['rafdb/native_ce']['schedule']['total_updates'],20700)
            self.assertEqual(configs['sfew/full_v2']['schedule']['total_updates'],1150)
            self.assertEqual(configs['affectnet/full_v2']['schedule']['total_updates'],239520)
            self.assertEqual(configs['rafdb/hybrid_ce']['init_path'],configs['rafdb/full_v2']['init_path'])
            self.assertEqual(configs['rafdb/native_ce']['teachers'],{})
            self.assertEqual(configs['sfew/hybrid_ce']['teachers'],{})
            self.assertEqual(set(configs['sfew/full_v2']['teachers']),{'t1','t2'})
            c=configs['rafdb/full_v2']; c['schedule']['optimizer_reset']=True
            with self.assertRaises(ValueError): validate(c)
            c=copy.deepcopy(configs['sfew/native_ce']); c['preprocess']['student_normalize']='CLIP'
            with self.assertRaises(ValueError): validate(c)
            c=copy.deepcopy(configs['sfew/hybrid_feature_kd']); c['stages'][0]['teacher']='t1'
            with self.assertRaises(ValueError): validate(c)
    def test_missing_gpu_fails_before_model_construction(self):
        from vim_v2.runtime import gpu_lease
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no CUDA'):
                with gpu_lease(d): self.fail('must not enter')
    def test_s1_last_transition_preserves_adam(self):
        from vim_v2.checkpoint import record_stage_transition
        m=torch.nn.Linear(3,7); opt=torch.optim.AdamW([{'params':m.parameters(),'lr':.001,'peak_lr':.001}]); s=UpdateCosine(opt,10,1)
        s.prepare(); m(torch.randn(4,3)).square().mean().backward(); opt.step(); s.step()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); payload={'kind':'resume','next_epoch':1,'next_batch':0,'global_update':1,'model':m.state_dict(),'optimizer':opt.state_dict()}
            atomic_save(out/'last.pt',payload)
            parent=record_stage_transition(out,m,opt,s,1)
            self.assertEqual(parent,sha(out/'s1_last.pt')); self.assertTrue((out/'stage_transition.json').exists())
            replacement=torch.optim.AdamW([{'params':m.parameters(),'lr':.001,'peak_lr':.001}])
            with self.assertRaises(ValueError): record_stage_transition(out,m,replacement,s,1)

if __name__=='__main__': unittest.main()
