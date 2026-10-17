"""Single-GPU baseline runner; immutable splits, official Vim, bounded artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from open_clip import create_model, get_tokenizer
from open_clip.transform import image_transform
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.vision_mamba_clip.data import FERProtocol, FERSample, FERDataset
from open_clip.vision_mamba_clip.model import HybridVisionMambaStudent
from open_clip.vision_mamba_clip.adapters import TeacherAdapter
from open_clip.vision_mamba_clip.stages import configure_teacher_stage, freeze_teacher_for_student
from open_clip.vision_mamba_clip.official_vim import create_official_vim, OfficialPatchAdapter, MambaCE

PROJECT = Path(__file__).resolve().parents[2]
EXPECTED = {
    'rafdb': '7e8bbb8985416b83d439f8a25aae74f1dacbd1021d4a94485a846d38ace6758b',
    'affectnet': '11aab95941ff78fa4baf04f5a25a17a97fadf7eff99454c00dc91e4f403b8a95',
    'sfew': '82e5d7cf52bb3471bbe1f2d4a5cee7f0e8e0f1c5b155889d6c89af4a1aa0c045',
}
WEIGHTS = PROJECT / 'src/pretrained_models/ViT-L-14.pt'
PROMPTS = PROJECT / 'configs/rafdb_au_prompts.json'


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def atomic_save(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def frozen_protocol(dataset):
    raw = json.loads((PROJECT / 'artifacts/preprocessed' / dataset / 'manifest.json').read_text())
    splits = {name: [FERSample(**row) for row in rows] for name, rows in raw['samples'].items()}
    protocol = FERProtocol._build(Path(raw['root']).resolve(), raw['class_order'], splits,
                                  raw['validation_seed'], raw['validation_ratio'], raw['metadata'])
    if protocol.manifest_hash != raw['manifest_hash'] or protocol.manifest_hash != EXPECTED[dataset]:
        raise ValueError('Frozen manifest identity changed')
    return protocol


class EpochDataset(FERDataset):
    """Augmentation depends on (epoch, sample), independent of worker scheduling."""
    epoch = 0

    def __getitem__(self, index):
        py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        sample_seed = (42 + self.epoch * 1000003 + index * 97) % (2**32)
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.random.default_generator.manual_seed(sample_seed)
        try:
            return super().__getitem__(index)
        finally:
            random.setstate(py)
            np.random.set_state(npstate)
            torch.set_rng_state(cpu)


def transforms(training):
    return image_transform(256, is_train=training, mean=OPENAI_DATASET_MEAN,
                           std=OPENAI_DATASET_STD, interpolation='bicubic', resize_mode='shortest',
                           aug_cfg={'scale': (0.9, 1.0)} if training else None)


def transform_description(training):
    # Function repr includes a process-specific address; it must not invalidate resume.
    return re.sub(r' at 0x[0-9a-fA-F]+', '', str(transforms(training)))


def loader(protocol, split, batch, epoch=0, workers=4):
    data = EpochDataset(protocol, split, transforms(split == 'train'))
    data.epoch = epoch
    generator = torch.Generator().manual_seed(42 + epoch)
    return DataLoader(data, batch_size=batch, shuffle=split == 'train', drop_last=split == 'train',
                      num_workers=workers, pin_memory=True, generator=generator,
                      persistent_workers=False, prefetch_factor=2 if workers else None)


def student(variant):
    seed_all()
    backbone, metadata = create_official_vim(img_size=256)
    backbone.set_grad_checkpointing(True)
    if variant == 'mamba_ce':
        model = MambaCE(backbone)
    else:
        model = HybridVisionMambaStudent(OfficialPatchAdapter(backbone), {'e': 384, 'm': 384, 'l': 384},
                                        common_dim=768, num_classes=7, attention_heads=8)
        model.text_to_classifier.requires_grad_(False)
    return model.cuda(), metadata


def teacher(stage, parent=None):
    seed_all()
    core = create_model('ViT-L-14', pretrained=str(WEIGHTS), force_quick_gelu=True,
                        force_image_size=256, device='cpu', precision='fp32')
    core.visual.set_grad_checkpointing(True)
    model = TeacherAdapter(core, 7).cuda()
    if parent is not None:
        load_weights(model, parent, teacher_model=True)
    if stage in ('teacher_t1', 'teacher_t2'):
        configure_teacher_stage(model, stage)
    else:
        freeze_teacher_for_student(model)
    return model


def model_state(model, teacher_model=False):
    state = model.state_dict()
    if teacher_model:
        state = {k: v for k, v in state.items() if k.startswith(('clip_model.visual.', 'classifier.'))
                 or k == 'clip_model.logit_scale'}
    # Avoid holding a second complete CPU copy during serialization.
    return state


def load_weights(model, payload, teacher_model=False):
    state = payload['model']
    if not teacher_model:
        model.load_state_dict(state, strict=True)
        return
    required = set(model_state(model, True))
    if set(state) != required:
        raise ValueError('Teacher visual/head checkpoint keys do not strictly match')
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(k in required for k in result.missing_keys):
        raise ValueError('Teacher checkpoint loading failed')


def class_weights(protocol, dataset):
    counts = torch.bincount(torch.tensor([s.label for s in protocol.splits['train']]), minlength=7).float()
    if dataset != 'affectnet':
        return None
    inverse = counts.reciprocal()
    return (inverse / inverse.mean()).cuda()


def score_matrix(matrix):
    value = matrix.double()
    tp = value.diag()
    support = value.sum(1)
    precision = tp / value.sum(0).clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {'accuracy': float(tp.sum() / value.sum().clamp_min(1)), 'uar': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'f1': f1.tolist(), 'support': support.tolist(), 'confusion_matrix': matrix.tolist()}


def batch_loss(model, images, labels, variant, weights, anchors=None, kd_teacher=None):
    if variant == 'teacher_t1':
        features = model.encode_image(images)
        logits = model.logit_scale.clamp(max=math.log(100)).exp() * F.normalize(features.float(), dim=-1) @ anchors.T
    elif variant == 'teacher_t2':
        logits = model.forward_classifier(images)['logits']
    else:
        result = model(images)
        logits = result if isinstance(result, torch.Tensor) else result.logits
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, weight=weights)
    kd = logits.new_zeros(())
    if variant == 'logits_kd':
        with torch.no_grad():
            targets = kd_teacher.forward_classifier(images)['logits'].float()
        kd = F.kl_div(F.log_softmax(logits / 2., dim=-1), F.softmax(targets / 2., dim=-1),
                      reduction='batchmean') * 4.
    return ce + 0.2 * kd, logits, ce, kd


@torch.no_grad()
def evaluate(model, batches, variant, weights, anchors=None, predictions=None):
    model.eval()
    matrix = torch.zeros((7, 7), dtype=torch.long)
    total_loss, seen, loss_denominator = 0., 0, 0.
    writer = csv.writer(predictions) if predictions else None
    if writer:
        writer.writerow(['relative_path', 'true_label', 'predicted_label', 'confidence'])
    for images, labels, paths in batches:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            # Selection of KD is student classification only, no teacher selection signal.
            loss, logits, _, _ = batch_loss(model, images, labels, 'hybrid_ce' if variant == 'logits_kd' else variant, weights, anchors)
        predicted = logits.argmax(-1)
        matrix += torch.bincount((labels * 7 + predicted).cpu(), minlength=49).reshape(7, 7)
        denominator = len(labels) if weights is None else float(weights[labels].sum())
        total_loss += float(loss) * denominator
        loss_denominator += denominator
        seen += len(labels)
        if writer:
            confidence = logits.softmax(-1).amax(-1).cpu().tolist()
            writer.writerows(zip(paths, labels.cpu().tolist(), predicted.cpu().tolist(), confidence))
    return dict(score_matrix(matrix), loss=total_loss / loss_denominator)


def rng_state():
    npstate = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (npstate[0], npstate[1].tolist(), *npstate[2:]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def lr_factor(step, steps_per_epoch, epochs, teacher_stage=False):
    if teacher_stage:
        start, length = 0, epochs * steps_per_epoch
    elif step < 10 * steps_per_epoch:
        start, length = 0, 10 * steps_per_epoch
    else:
        start, length = 10 * steps_per_epoch, 20 * steps_per_epoch
    progress = step - start
    warmup = max(1, round(length * .05))
    if progress < warmup:
        return (progress + 1) / warmup
    return .5 * (1 + math.cos(math.pi * min(1., (progress - warmup) / max(1, length - warmup))))


def run(args):
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    seed_all()
    protocol = frozen_protocol(args.dataset)
    output = Path(args.output_root) / args.dataset / args.variant
    output.mkdir(parents=True, exist_ok=True)
    is_teacher = args.variant.startswith('teacher_')
    epochs = {'teacher_t1': 5, 'teacher_t2': 10}.get(args.variant, 30)
    steps_per_epoch = len(protocol.splits['train']) // args.batch_size
    parent_path = None
    parent = None
    if args.variant in ('teacher_t2', 'logits_kd'):
        parent_stage = 'teacher_t1' if args.variant == 'teacher_t2' else 'teacher_t2'
        parent_path = Path(args.output_root) / args.dataset / parent_stage / 'best.pt'
        if parent_path.exists():
            parent = torch.load(parent_path, map_location='cpu', weights_only=True)
            if parent['identity']['manifest_hash'] != protocol.manifest_hash or parent['identity']['variant'] != parent_stage:
                raise ValueError('Teacher source/domain mismatch')
            if parent['identity']['architecture']['checkpoint_sha256'] != digest_file(WEIGHTS):
                raise ValueError('Frozen teacher text/base initialization identity mismatch')
        elif args.smoke:
            # Only a gradient/memory acceptance probe, never a trained KD result.
            parent_path = None
        else:
            raise FileNotFoundError(parent_path)
    if is_teacher:
        model = teacher(args.variant, parent)
        architecture = {'identity': 'original_CLIP_ViT_L14_QuickGELU_projected768_img256',
                        'checkpoint_sha256': digest_file(WEIGHTS), 'patch_grid': [18, 18], 'uncovered_border': [4, 4]}
    else:
        model, architecture = student(args.variant)
    kd_teacher = teacher('frozen', parent) if args.variant == 'logits_kd' else None
    del parent
    anchors = None
    if args.variant == 'teacher_t1':
        schema = json.loads(PROMPTS.read_text())
        if schema['class_order'] != protocol.class_order:
            raise ValueError('AU class order mismatch')
        tokens = get_tokenizer('ViT-L-14')([schema['prompts'][name] for name in protocol.class_order]).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            anchors = F.normalize(model.encode_text(tokens).float(), dim=-1).detach()
    identity = {'dataset': args.dataset, 'variant': args.variant, 'manifest_hash': protocol.manifest_hash,
                'class_order': protocol.class_order, 'seed': 42, 'architecture': architecture,
                'batch_size': args.batch_size, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch,
                'teacher_source': args.dataset if parent_path else None,
                'teacher_best_sha256': digest_file(parent_path) if parent_path else None,
                'smoke_only_original_teacher': bool(args.smoke and args.variant in ('teacher_t2', 'logits_kd') and parent_path is None),
                'prompt_sha256': digest_file(PROMPTS), 'precision': 'bf16_amp_fp32_parameters',
                'runner_sha256': digest_file(__file__),
                'official_adapter_sha256': digest_file(PROJECT / 'src/open_clip/vision_mamba_clip/official_vim.py'),
                'augmentation': transform_description(True), 'evaluation_transform': transform_description(False),
                'ce_weighting': 'normalized_inverse_training_frequency' if args.dataset == 'affectnet' else 'none',
                'objective': 'prototype_ce' if args.variant == 'teacher_t1' else ('ce+0.2*KL(teacher||student)*T^2,T=2' if args.variant == 'logits_kd' else 'ce'),
                'lr': 1e-5 if is_teacher else 1e-4, 'head_lr': 1e-4, 'weight_decay': .01,
                'lr_segments_epochs': [epochs] if is_teacher else [10, 20], 'warmup_ratio': .05,
                'optimizer_stage_reset': False, 'checkpoint_rule': 'minimum_selection_loss' if args.variant == 'teacher_t1' else 'maximum_selection_UAR_then_accuracy'}
    weights = class_weights(protocol, args.dataset)
    identity['class_weights'] = weights.cpu().tolist() if weights is not None else None
    identity['split_sizes'] = {key: len(rows) for key, rows in protocol.splits.items()}
    if is_teacher:
        visual = [p for p in model.clip_model.visual.parameters() if p.requires_grad]
        other = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith('clip_model.visual.')]
        groups = [{'params': visual, 'lr': 1e-5, 'base_lr': 1e-5}, {'params': other, 'lr': 1e-4, 'base_lr': 1e-4}]
    else:
        groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-4, 'base_lr': 1e-4}]
    optimizer = torch.optim.AdamW(groups, betas=(.9, .98), eps=1e-6, weight_decay=.01)
    start_epoch, skip_batches, updates, best_score = 0, 0, 0, None
    last_path, best_path = output / 'last.pt', output / 'best.pt'
    if last_path.exists() and not args.smoke:
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume identity mismatch')
        load_weights(model, saved, is_teacher)
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, skip_batches, updates = saved['next_epoch'], saved['next_batch'], saved['updates']
        best_score = saved['best_score']
        restore_rng(saved['rng'])
        del saved
    atomic_json(output / ('smoke_config.json' if args.smoke else 'effective_config.json'), identity)
    validation = loader(protocol, 'val', args.batch_size, workers=args.workers)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    def save_last(next_epoch, next_batch):
        atomic_save(last_path, {'identity': identity, 'model': model_state(model, is_teacher),
                               'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'next_epoch': next_epoch,
                               'next_batch': next_batch, 'updates': updates, 'best_score': best_score,
                               'scheduler': {'rule': 'segmented_cosine', 'next_step': updates}})

    for epoch in range(start_epoch, epochs):
        batches = loader(protocol, 'train', args.batch_size, epoch, args.workers)
        if is_teacher:
            configure_teacher_stage(model, args.variant)
        else:
            model.train()
        accumulated, actual = 0., 0
        for index, (images, labels, _) in enumerate(batches):
            if index < skip_batches:
                continue
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            factor = lr_factor(updates, steps_per_epoch, epochs, is_teacher)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, logits, ce, kd = batch_loss(model, images, labels, args.variant, weights, anchors, kd_teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient norm')
            if args.smoke:
                if kd_teacher and any(p.grad is not None for p in kd_teacher.parameters()):
                    raise RuntimeError('Frozen KD teacher has gradients')
                if is_teacher and any(p.grad is not None for p in model.text_parameters()):
                    raise RuntimeError('Frozen text tower has gradients')
            optimizer.step()
            if args.variant == 'teacher_t1':
                with torch.no_grad():
                    model.logit_scale.clamp_(max=math.log(100))
            updates += 1
            accumulated += float(loss)
            actual += 1
            if index % 50 == 0 or args.smoke:
                print(json.dumps({'epoch': epoch + 1, 'batch': index + 1, 'updates': updates,
                                  'loss': float(loss), 'ce': float(ce), 'kd': float(kd), 'grad_norm': float(norm),
                                  'peak_gpu_GiB': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
            if args.smoke and actual >= 2:
                torch.cuda.synchronize()
                report = {'identity': identity, 'accepted': True, 'real_batches': actual,
                          'seconds': time.time() - started, 'peak_gpu_GiB': torch.cuda.max_memory_allocated() / 2**30,
                          'parameters': sum(p.numel() for p in model.parameters()), 'logits_shape': list(logits.shape)}
                # Exercise serialization and strict model+optimizer+RNG restoration without leaving weights.
                # Acceptance scratch is transient: local data disk avoids a multi-GB network-storage roundtrip.
                temporary = PROJECT / 'artifacts/training_prepare_20261013' / ('smoke_roundtrip_' + args.variant + '.pt')
                atomic_save(temporary, {'model': model_state(model, is_teacher), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
                roundtrip = torch.load(temporary, map_location='cpu', weights_only=True)
                load_weights(model, roundtrip, is_teacher)
                optimizer.load_state_dict(roundtrip['optimizer'])
                restore_rng(roundtrip['rng'])
                temporary.unlink()
                atomic_json(output / 'smoke_acceptance.json', report)
                print(json.dumps(report), flush=True)
                return
            if updates % 1000 == 0:
                save_last(epoch, index + 1)
        skip_batches = 0
        metrics = evaluate(model, validation, args.variant, weights, anchors)
        current = [-metrics['loss']] if args.variant == 'teacher_t1' else [metrics['uar'], metrics['accuracy']]
        if best_score is None or current > best_score:
            best_score = current
            atomic_save(best_path, {'identity': identity, 'model': model_state(model, is_teacher),
                                   'epoch': epoch + 1, 'updates': updates, 'selection_metrics': metrics})
        save_last(epoch + 1, 0)
        record = {'epoch': epoch + 1, 'updates': updates, 'train_loss': accumulated / max(1, actual),
                  'selection': metrics, 'elapsed_seconds': time.time() - started,
                  'peak_gpu_GiB': torch.cuda.max_memory_allocated() / 2**30}
        with (output / 'curve.jsonl').open('a') as curve:
            curve.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        atomic_json(output / 'status.json', dict(state='training', epoch=epoch + 1, epochs=epochs, updates=updates))
    if updates != epochs * steps_per_epoch:
        raise ValueError('Final optimizer update budget mismatch')
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    load_weights(model, best, is_teacher)
    with (output / 'final_predictions.csv').open('w', newline='') as predictions:
        final = evaluate(model, loader(protocol, 'test', args.batch_size, workers=args.workers),
                         args.variant, weights, anchors, predictions=predictions)
    atomic_json(output / 'final_metrics.json', {'identity': identity, 'best_epoch': best['epoch'], 'metrics': final,
                'training_seconds': time.time() - started, 'updates': updates,
                'peak_gpu_GiB': torch.cuda.max_memory_allocated() / 2**30})
    atomic_json(output / 'status.json', dict(state='completed', epochs=epochs, updates=updates))
    print(json.dumps({'completed': args.variant, 'dataset': args.dataset, 'final': final}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(EXPECTED))
    parser.add_argument('--variant', required=True, choices=['mamba_ce', 'hybrid_ce', 'logits_kd', 'teacher_t1', 'teacher_t2'])
    parser.add_argument('--output-root', default='/root/autodl-fs/fer_suite_simplified_20261013')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
