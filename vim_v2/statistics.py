"""Paired, class-stratified bootstrap of saved per-image FER predictions."""

import csv
from pathlib import Path

import numpy as np

from .common import sha


def prediction_rows(path):
    rows={}
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            sample=row['sample_id']; label=int(row['true_label']); prediction=int(row['predicted_label'])
            probabilities=np.array([float(row['p'+str(i)]) for i in range(7)])
            if (not sample or sample in rows or not 0<=label<7 or not 0<=prediction<7 or
                    not np.isfinite(probabilities).all() or probabilities.min()<0 or
                    abs(probabilities.sum()-1)>1e-5 or int(probabilities.argmax())!=prediction):
                raise ValueError('Invalid/duplicate saved prediction: '+sample)
            rows[sample]=(label,prediction)
    if not rows: raise ValueError('Empty prediction file')
    return rows


def scores(labels,predictions):
    matrix=np.bincount(labels*7+predictions,minlength=49).reshape(7,7).astype(float)
    tp=np.diag(matrix); support=matrix.sum(1)
    if (support==0).any(): raise ValueError('Paired UAR bootstrap requires all seven classes')
    recall=tp/support; precision=tp/np.maximum(matrix.sum(0),1)
    f1=2*recall*precision/np.maximum(recall+precision,1e-12)
    return np.array([tp.sum()/matrix.sum(),recall.mean(),f1.mean()])


def paired_bootstrap(baseline,candidate,*,resamples=2000,seed=42,split='selection_val'):
    if type(resamples) is not int or not 2<=resamples<=100000: raise ValueError('resamples must be 2..100000')
    if split not in ['selection_val','final_eval']: raise ValueError('Unknown evaluation split')
    if split=='selection_val' and any(Path(path).name.startswith('final_') for path in [baseline,candidate]):
        raise ValueError('Final-evaluation predictions require explicit split=final_eval')
    left,right=prediction_rows(baseline),prediction_rows(candidate)
    if set(left)!=set(right) or any(left[key][0]!=right[key][0] for key in left):
        raise ValueError('Paired predictions must have exactly the same sample IDs and labels')
    labels=np.array([value[0] for value in left.values()]); a=np.array([value[1] for value in left.values()])
    b=np.array([right[key][1] for key in left]); groups=[np.flatnonzero(labels==i) for i in range(7)]
    observed=(scores(labels,b)-scores(labels,a))*100
    rng=np.random.default_rng(seed); deltas=np.empty((resamples,3))
    for index in range(resamples):
        ids=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in groups])
        deltas[index]=(scores(labels[ids],b[ids])-scores(labels[ids],a[ids]))*100
    low,high=np.percentile(deltas,[2.5,97.5],axis=0)
    return {'baseline_sha256':sha(baseline),'candidate_sha256':sha(candidate),'samples':len(labels),
            'class_counts':[len(group) for group in groups],'resamples':resamples,'seed':seed,'split':split,
            'method':'paired class-stratified percentile bootstrap; same sampled image IDs in both models',
            'delta_percentage_points':{name:{'candidate_minus_baseline':float(observed[i]),'ci95':[float(low[i]),float(high[i])]}
                                       for i,name in enumerate(['accuracy','uar','macro_f1'])},
            'limits':'Conditional on saved models and observed class counts; not a training-seed confidence interval, no correction for multiple comparisons.'}
