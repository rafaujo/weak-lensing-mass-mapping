#!/usr/bin/env python3
"""Check archived CSV values against the seven tables in supplied manuscript (8).

Standard library only. Does not import or execute reconstruction scripts, alter
scientific results, run bootstrap resampling, or certify figure regeneration.
Expected printed values are transcribed from the supplied main.tex, whose SHA256
is recorded in provenance/manuscript_reference.json.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def read(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def check(table,quantity,actual,expected,fmt):
    observed=format(actual,fmt)
    checks.append({'table':table,'quantity':quantity,'value':actual,
                   'printed':observed,'expected':expected,'format':fmt,'pass':observed==expected})
def state(table,quantity,actual,expected):
    checks.append({'table':table,'quantity':quantity,'value':actual,'expected':expected,'pass':actual==expected})
def group(rows,key):
    out=defaultdict(list)
    for row in rows:out[row[key]].append(row)
    return out

def verify():
    # Table 1: use independently stored per-case tables, not hard-coded figure values.
    sources=[('results/test/square/square_ours_metrics.csv',None),
             ('results/test/glimpse_test_metrics.csv',None),
             ('results/test/mcalens/mcalens_test_metrics.csv',None),
             ('results/test/starlet_gaussian_prior/gaussian_starlet_test_metrics.csv',None),
             ('results/test/fusion/field_metrics_all.csv',None),
             ('results/wiener/wiener_test_cases.csv','WIENER')]
    all_cases={}
    for path,fixed in sources:
        gs=defaultdict(list)
        for row in read(path):
            name=fixed or row['method']
            if name=='W_A1_075_A2_075':name='FUSION'
            gs[name].append(row)
        for name,rows in gs.items():
            if name in all_cases:continue
            all_cases[name]=rows
    expected_fields={
        'KS':('0.724','0.531','0.607'), 'GLIMPSE':('0.824','0.437','0.565'),
        'WIENER':('0.687','0.569','0.472'),'STARLET_NOMASK':('0.693','0.555','0.487'),
        'STARLET_MASK':('0.679','0.565','0.531'),'MCALENS':('0.651','0.596','0.686'),
        'GAUSSIAN_STARLET':('0.643','0.597','0.583'),'FUSION':('0.634','0.604','0.631')}
    grid={(str(c),str(r),str(p)) for c in range(3) for r in range(10,30) for p in range(8)}
    means={}
    for method,expected in expected_fields.items():
        rows=all_cases[method]
        ids=[(r['c'],r['r'],r['p']) for r in rows]
        state('tab:test_field',method+' complete 480-case grid',len(ids)==480 and set(ids)==grid,True)
        means[method]={col:mean(float(r[col]) for r in rows) for col in ['nmse','pcc','std_ratio']}
        for col,want in zip(['nmse','pcc','std_ratio'],expected):
            check('tab:test_field',method+' '+col,means[method][col],want,'.3f')
    # Table 2 and the unrounded gains described alongside it.
    for method,want,gain in [('STARLET_MASK','0.679','-0.045'),('WIENER','0.687','-0.037'),('GAUSSIAN_STARLET','0.643','-0.081')]:
        check('tab:prior_factorial',method,means[method]['nmse'],want,'.3f')
        check('tab:prior_factorial',method+' minus KS',means[method]['nmse']-means['KS']['nmse'],gain,'.3f')
    check('tab:prior_factorial','KS',means['KS']['nmse'],'0.724','.3f')
    # Table 3: final V12 per-block spectral curves, not earlier diagnostic scores.
    expected_spectral={'GAUSSIAN_STARLET':('0.555','0.574','0.655'),
                       'MCALENS':('0.469','0.395','0.656'),'FUSION':('0.513','0.499','0.662')}
    spectra=group(read('results/paper/spectral_curves.csv'),'method')
    for method,wants in expected_spectral.items():
        rows=spectra[method]
        state('tab:spectral_summary',method+' rows',len(rows),320)
        for col,want in zip(['T','A','r'],wants):
            value=mean(abs(1-float(r[col])) if col!='r' else float(r[col]) for r in rows)
            check('tab:spectral_summary',method+' '+col,value,want,'.3f')
    # Table 4: read common-completeness AUC stored by final V12 generator.
    expected_peaks={'GAUSSIAN_STARLET':(['0.722','0.740','0.729'],['0.500','0.640','0.708']),
                    'MCALENS':(['0.676','0.729','0.747'],['0.573','0.979','0.957']),
                    'FUSION':(['0.711','0.754','0.763'],['0.561','0.798','0.817'])}
    curves=read('results/paper/peak_completeness_purity.csv')
    amplitudes=read('results/paper/peak_amplitude.csv')
    for method,(aucs,amps) in expected_peaks.items():
        for i,scale in enumerate([2.,4.,6.]):
            values={float(r['auc_common']) for r in curves if r['method']==method and float(r['scale_arcmin'])==scale}
            state('tab:peak_summary',f'{method} {scale} constant AUC column',len(values),1)
            check('tab:peak_summary',f'{method} {scale} PR-AUC',next(iter(values)),aucs[i],'.3f')
            amp=next(float(r['peak_amplitude_ratio']) for r in amplitudes if r['method']==method and float(r['scale_arcmin'])==scale)
            check('tab:peak_summary',f'{method} {scale} amplitude',amp,amps[i],'.3f')
    # Table 5: all nine stored bootstrap intervals; no resampling is rerun.
    expected_boot={
        (2.,'FUSION','GAUSSIAN_STARLET'):('-0.0109','-0.0138','-0.0062'),
        (2.,'FUSION','MCALENS'):('0.0348','0.0314','0.0390'),
        (2.,'GAUSSIAN_STARLET','MCALENS'):('0.0457','0.0393','0.0509'),
        (4.,'FUSION','GAUSSIAN_STARLET'):('0.0132','0.0086','0.0187'),
        (4.,'FUSION','MCALENS'):('0.0246','0.0183','0.0293'),
        (4.,'GAUSSIAN_STARLET','MCALENS'):('0.0114','0.0031','0.0184'),
        (6.,'FUSION','GAUSSIAN_STARLET'):('0.0339','0.0224','0.0453'),
        (6.,'FUSION','MCALENS'):('0.0163','0.0068','0.0251'),
        (6.,'GAUSSIAN_STARLET','MCALENS'):('-0.0176','-0.0333','-0.0015')}
    for row in read('results/test/peak_bootstrap_v5/peak_prauc_block_bootstrap.csv'):
        key=(float(row['scale_arcmin']),row['method_1'],row['method_2'])
        for col,want in zip(['delta','ci95_low','ci95_high'],expected_boot[key]):
            check('tab:peak_bootstrap',str(key)+' '+col,float(row[col]),want,'.4f')
    # Table 6.
    expected_photo={'GAUSSIAN_STARLET':('0.161','-0.235','0.474','0.56'),
                    'MCALENS':('0.209','-0.227','0.557','0.71'),
                    'FUSION':('0.174','-0.259','0.509','0.61')}
    for row in read('results/test/photoz_audit/photoz_regression_summary.csv'):
        for col,want,fmt in zip(['slope_nmse_per_delta_z','slope_ci95_low','slope_ci95_high','relative_effect_0p022_percent'],expected_photo[row['method']],['.3f','.3f','.3f','.2f']):
            check('tab:photoz',row['method']+' '+col,float(row[col]),want,fmt)
    # Table A.1: recompute aggregate values from all 30 ADMM case rows.
    audit=read('results/validation/admm_solver_audit/admm_vs_pdhg_cases.csv')
    state('tab:solver_audit','case count',len(audit),30)
    state('tab:solver_audit','both stopping criteria satisfied',sum(float(r['primal_ratio_to_eps'])<=1 and float(r['dual_ratio_to_eps'])<=1 for r in audit),30)
    check('tab:solver_audit','mean iterations',mean(float(r['admm_iterations']) for r in audit),'421.7','.1f')
    for col,agg,want in [('obj_rel_diff',median,'1.69e-05'),('obj_rel_diff',max,'2.05e-05'),
                         ('map_rel_diff_observed_centered',median,'2.60e-03'),('map_rel_diff_observed_centered',max,'3.15e-03')]:
        check('tab:solver_audit',agg.__name__+' '+col,agg(float(r[col]) for r in audit),want,'.2e')
    for col,want in [('nmse_admm','0.6365907'),('nmse_pdhg','0.6365985'),
                     ('pcc_admm','0.6015129'),('pcc_pdhg','0.6015069'),
                     ('std_admm','0.5892284'),('std_pdhg','0.5892523')]:
        check('tab:solver_audit','mean '+col,mean(float(r[col]) for r in audit),want,'.7f')
    # Additional field-comparison intervals in the prose.
    for row in read('results/test/wiener_vs_gaussian_starlet_bootstrap/wiener_vs_gs_block_bootstrap.csv'):
        if row['quantity']=='relative_nmse_reduction_gs':
            for col,want in zip(['point','ci95_low','ci95_high'],['6.4','5.7','7.1']):
                check('sec:results_field','GS vs Wiener percent '+col,100*float(row[col]),want,'.1f')
    rows=read('results/validation/glimpse_validation_metrics.csv')
    for lam,want in [(2.5,'0.998'),(3.,'0.828'),(3.5,'0.833')]:
        selected=[r for r in rows if float(r['lambda'])==lam]
        state('sec:comparators',f'GLIMPSE lambda {lam} validation cases',len(selected),240)
        check('sec:comparators',f'GLIMPSE lambda {lam} mean NMSE',mean(float(r['nmse']) for r in selected),want,'.3f')
    # Section 4.8: independently aggregate the final paired 480-case GS stress test.
    label='sec:results_systematics reduced shear'
    cases=read('results/test/gaussian_starlet_reduced_shear/gaussian_starlet_reduced_shear_cases.csv')
    ids=[(r['c'],r['r'],r['p']) for r in cases]
    state(label,'complete 480-case grid',len(ids)==480 and set(ids)==grid,True)
    state(label,'realization blocks',len({r['r'] for r in cases}),20)
    mismatch=[float(r['reduced_shear_mismatch_relative']) for r in cases]
    check(label,'mean forward mismatch percent',100*mean(mismatch),'2.02','.2f')
    check(label,'median forward mismatch percent',100*median(mismatch),'1.76','.2f')
    nominal=mean(float(r['nominal_nmse']) for r in cases)
    reduced=mean(float(r['rs_nmse']) for r in cases)
    check(label,'nominal mean NMSE',nominal,'0.643','.3f')
    check(label,'reduced-shear mean NMSE',reduced,'0.640','.3f')
    check(label,'relative change in mean NMSE percent',100*(reduced/nominal-1),'-0.458','.3f')
    for col,want in [('pcc','0.00223'),('std_ratio','0.00415')]:
        check(label,'paired mean delta '+col,mean(float(r['rs_'+col])-float(r['nominal_'+col]) for r in cases),want,'.5f')
    stored=read('results/test/gaussian_starlet_reduced_shear/gaussian_starlet_reduced_shear_summary.csv')[0]
    for col,actual in [('nominal_nmse',nominal),('reduced_shear_nmse',reduced),
                       ('mean_reduced_shear_mismatch_relative',mean(mismatch)),
                       ('median_reduced_shear_mismatch_relative',median(mismatch))]:
        state(label,'stored summary agrees with cases: '+col,abs(actual-float(stored[col]))<1e-12,True)
    expected_rs={'relative_nmse_change_pct':(['-0.458','-0.501','-0.417'],'.3f'),
                 'delta_pcc':(['0.00223','0.00208','0.00238'],'.5f'),
                 'delta_std_ratio':(['0.00415','0.00369','0.00462'],'.5f')}
    boots=read('results/test/gaussian_starlet_reduced_shear/gaussian_starlet_reduced_shear_bootstrap.csv')
    state(label,'bootstrap metadata',all((r['n_blocks'],r['n_cases'],r['n_boot'])==('20','480','10000') for r in boots),True)
    for row in boots:
        if row['quantity'] not in expected_rs:continue
        expected,fmt=expected_rs[row['quantity']]
        for col,want in zip(['point','ci95_low','ci95_high'],expected):
            check(label,row['quantity']+' '+col,float(row[col]),want,fmt)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,help='Optional JSON report; default only prints the summary')
    args=ap.parse_args()
    verify()
    failures=[c for c in checks if not c['pass']]
    report={'scope':'Consistency of archived CSV values with printed manuscript tables; no scientific pipeline rerun.',
            'tables_checked':sorted({c['table'] for c in checks if c['table'].startswith('tab:')}),
            'checks':checks,'failed':len(failures),
            'not_verified':['Generation of maps and figures from the raw data.',
                            'Regeneration of the frozen P_kappa and noise-calibration arrays.',
                            'Bootstrap resampling; archived confidence intervals were compared only.']}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'tables':len(report['tables_checked']),'checks':len(checks),'failed':failures,'not_verified':report['not_verified']},indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
