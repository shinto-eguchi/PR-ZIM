#!/usr/bin/env python3
"""Apply newly computed PR_MLE score-sandwich SE to NHANES coefficient output.

Only the 18 PR_MLE_corrected primary_no_usualcare SE cells are changed.
Other rows, point estimates and the other columns are preserved unchanged.

Example:
 python apply_PR_sandwich_csv.py --input results/nhanes/all_coefficients.csv \
   --reference results/pr_sandwich/PR_MLE_weighted_sandwich_SE.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['mode', 'design', 'method', 'block', 'term']

def update(input_csv, reference_csv):
    target = pd.read_csv(input_csv)
    ref = pd.read_csv(reference_csv)
    needed_t = set(KEY + ['estimate', 'se'])
    needed_r = set(KEY + ['estimate', 'curvature_se_original', 'sandwich_se'])
    assert needed_t.issubset(target.columns), f'Missing target columns: {needed_t - set(target.columns)}'
    assert needed_r.issubset(ref.columns), f'Missing reference columns: {needed_r - set(ref.columns)}'
    assert len(ref)==18, f'Expected 18 PR coefficient rows, got {len(ref)}'
    assert not ref.duplicated(KEY).any(), 'Duplicate sandwich key in reference'
    assert not target.duplicated(KEY).any(), 'Duplicate coefficient key in target'
    assert set(ref['mode']) == {'A1C','FPG'}
    assert (ref['method']=='PR_MLE_corrected').all()
    assert (ref['design']=='primary_no_usualcare').all()
    assert np.isfinite(ref['sandwich_se']).all() and (ref['sandwich_se']>0).all()
    indexed = target.set_index(KEY, drop=False)
    errors=[]
    for r in ref.itertuples(index=False):
        key=tuple(getattr(r,k) for k in KEY)
        if key not in indexed.index:
            errors.append(f'Missing target key {key}')
            continue
        i = target.index[(target[KEY] == pd.Series(dict(zip(KEY,key)))).all(axis=1)]
        if len(i)!=1: errors.append(f'Unmatched target index {key}'); continue
        i=int(i[0])
        # Protect against altering coefficients from different fitting runs.
        if not np.isclose(float(target.at[i,'estimate']),float(r.estimate),atol=0.003,rtol=0.002):
            errors.append(f'Estimate mismatch for {key}: {target.at[i,"estimate"]} vs {r.estimate}')
        # Existing target can already have sandwich SE on a second run.
        old=float(target.at[i,'se'])
        if not (np.isclose(old,float(r.curvature_se_original),atol=0.003,rtol=0.002)
                or np.isclose(old,float(r.sandwich_se),atol=0.003,rtol=0.002)):
            errors.append(f'SE mismatch for {key}: {old}')
        if not errors:
            target.at[i,'se']=float(r.sandwich_se)
    if errors:
        raise ValueError('Refusing to update due to mismatches:\n'+'\n'.join(errors))
    target.to_csv(input_csv,index=False)
    print(f'Updated 18 PR_MLE SE values: {input_csv}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args()
    update(a.input,a.reference)
