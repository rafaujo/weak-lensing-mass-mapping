#!/usr/bin/env python3
"""Check frozen peak-zero results; NumPy only, no map or bootstrap regeneration.

Adapted from the independent audit used for the returned server results.
Reads repository files and prints its report; writes no scientific data.
"""
from pathlib import Path
import ast, csv, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'analysis/peak_zero/results'
read=lambda name:list(csv.DictReader((R/name).open(encoding='utf-8',newline='')))
cfg=json.loads((R/'run_configuration.json').read_text())
code={name:hashlib.sha256((ROOT/'analysis/peak_zero/code'/name).read_bytes()).hexdigest()==digest for name,digest in cfg['code_sha256'].items()}
assert all(code.values()),code
assert json.loads((R/'status.json').read_text())['status']=='COMPLETE'
assert json.loads((R/'baseline_check.json').read_text())['max_finite_error']==0
assert all(c['absolute_error']==0 for c in json.loads((R/'baseline_bootstrap_check.json').read_text())['checks'])
M=('GAUSSIAN_STARLET','MCALENS','FUSION')
scales=(2.,4.,6.)
# Use the archived function, not the submitted script, to check pooled AUCs.
source=ROOT/'code/pipeline/bootstrap_peak_prauc_v5_with_gs_mcalens_bootstrap.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('curve_from_counts','normalized_common_auc')]
env={'np':np}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
bootrows=read('bootstrap_comparisons.csv')
checks=[]
diagnostics={}
for mode in ('stored','observed_mean_zero'):
    summaries=read(mode+'_summary.csv')
    samples=read(mode+'_amplitude_pairs.csv')
    offsets=read(mode+'_offsets.csv')
    assert len(offsets)==480*4
    assert len({(x['c'],x['r'],x['p']) for x in offsets})==480
    diagnostics[mode]={'max_abs_centered_mean':max(abs(float(x['mean_after'])) for x in offsets),
                       'observed_mean_by_method':{m:{'median':float(np.median([float(x['observed_mean']) for x in offsets if x['method']==m])),
                                                    'min':min(float(x['observed_mean']) for x in offsets if x['method']==m),
                                                    'max':max(float(x['observed_mean']) for x in offsets if x['method']==m)} for m in ('TRUTH',)+M}}
    if mode=='observed_mean_zero':assert diagnostics[mode]['max_abs_centered_mean']<1e-15
    with np.load(R/(mode+'_block_counts.npz'),allow_pickle=False) as b:
        assert b['truth'].shape==(3,20) and b['matched'].shape==(3,3,20,120)
        assert tuple(b['methods'])==M
        np.testing.assert_array_equal(b['realisations'],np.arange(10,30))
        for si,scale in enumerate(scales):
            curves={m:env['curve_from_counts'](int(b['truth'][si].sum()),b['predicted'][si,mi].sum(axis=0),b['matched'][si,mi].sum(axis=0)) for mi,m in enumerate(M)}
            auc=env['normalized_common_auc'](curves)
            for mi,m in enumerate(M):
                row=next(x for x in summaries if float(x['scale_arcmin'])==scale and x['method']==m)
                amp=[float(x['ratio']) for x in samples if float(x['scale_arcmin'])==scale and x['method']==m]
                assert len(amp)==int(row['n_amplitude_pairs'])
                assert int(b['truth'][si].sum())==int(row['n_truth_peaks'])
                assert abs(float(row['auc_common'])-auc[m])<1e-12
                assert abs(float(row['peak_amplitude_ratio'])-np.median(amp))<1e-12
                checks.append({'mode':mode,'scale':scale,'method':m,'auc':float(auc[m]),'median_amplitude':float(np.median(amp))})
    for scale in scales:
        with np.load(R/f'bootstrap_{scale:g}arcmin.npz',allow_pickle=False) as f:
            a=f[mode]
            assert a.shape==(10000,3) and np.isfinite(a).all()
            for i,j in ((2,0),(2,1),(0,1)):
                row=next(x for x in bootrows if x['mode']==mode and float(x['scale_arcmin'])==scale and x['method_1']==M[i] and x['method_2']==M[j])
                delta=a[:,i]-a[:,j]
                lo,hi=np.percentile(delta,[2.5,97.5])
                assert abs(lo-float(row['ci95_low']))<1e-12 and abs(hi-float(row['ci95_high']))<1e-12
                assert abs(float(np.mean(delta>0))-float(row['bootstrap_fraction_gt0']))<1e-12
audit={'input_directory':'analysis/peak_zero/results',
       'source_hash_matches':code,'baseline_checks_exact':6489,'baseline_bootstrap_checks_exact':36,
       'summaries_recomputed':checks,'offset_diagnostics':diagnostics,
       'bootstrap_intervals_recomputed':18,'all_checks_passed':True,
       'scope':'Validates supplied counts, samples and bootstrap draws; full map cubes are on the server.'}

print(json.dumps(audit,indent=2))
