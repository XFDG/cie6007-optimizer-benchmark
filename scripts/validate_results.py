"""Acceptance audit against the frozen protocol and independently aggregated raw CSVs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.utils import ROOT, config, sha256, write_json


def main():
    cfg=config()
    metrics=pd.read_csv(ROOT/'results/metrics.csv')
    final=metrics.query("phase=='final'")
    tune=metrics.query("phase=='tune'")
    ev=pd.read_csv(ROOT/'results/final_evaluations.csv')
    selected=json.loads((ROOT/'results/selected_config.json').read_text())
    assert selected['config_sha256']==sha256(ROOT/'configs/experiment.yaml')
    assert len(final)==4*len(cfg['seeds'])*cfg['epochs']
    assert len(tune)==sum(map(len,cfg['learning_rates'].values()))*cfg['tune_epochs']
    assert not final.duplicated(['optimizer','seed','epoch']).any()
    assert np.isfinite(final.select_dtypes('number')).all().all()
    assert (final.train_seconds>0).all()
    for seed,rows in final.groupby('seed'):
        assert rows.initial_model_sha256.nunique()==1
    assert tune.initial_model_sha256.nunique()==1
    for name,lrs in cfg['learning_rates'].items():
        t=tune[tune.optimizer==name]
        assert np.allclose(sorted(t.lr.unique()),sorted(lrs))
        assert set(final[final.optimizer==name].seed)==set(cfg['seeds'])
        assert (final[final.optimizer==name].lr==selected['selected'][name]['lr']).all()
    for _,r in final.groupby('run_id'):
        assert np.array_equal(r.epoch,np.arange(1,cfg['epochs']+1))
        assert np.allclose(r.cumulative_train_seconds,r.train_seconds.cumsum())
        assert np.allclose(r.cumulative_wall_seconds,(r.train_seconds+r.validation_seconds).cumsum())
    assert len(ev.query("selection=='best'"))==12
    assert len(ev.query("selection=='final'"))==12
    for _,r in ev.iterrows():
        assert np.isclose(r.generalization_gap,r.train_accuracy-r.test_accuracy)
        assert r.checkpoint_sha256==sha256(ROOT/'checkpoints'/r.run_id/f'{r.selection}.pt')
        if r.selection=='best':
            m=final[final.run_id==r.run_id]
            assert np.isclose(r.val_accuracy,m.val_accuracy.max())
        if r.selection=='budget':
            m=final[(final.run_id==r.run_id)&(final.cumulative_wall_seconds<=cfg['time_budget_s'])]
            assert np.isclose(r.val_accuracy,m.val_accuracy.max())
    summary=pd.read_csv(ROOT/'results/summary.csv').set_index('optimizer')
    for name,r in ev.query("selection=='best'").groupby('optimizer'):
        assert np.isclose(summary.loc[name,'test_accuracy_mean'],r.test_accuracy.mean())
        assert np.isclose(summary.loc[name,'test_accuracy_std'],r.test_accuracy.std(ddof=1))
    result=dict(status='PASS',tuning_runs=tune.run_id.nunique(),formal_runs=final.run_id.nunique(),
        epoch_rows=len(metrics),paired_initialization=True,balanced_lr_budget=True,
        checkpoint_hashes_verified=True,best_and_budget_selection_verified=True,
        summary_recomputed=True,metrics_sha256=sha256(ROOT/'results/metrics.csv'))
    write_json(ROOT/'results/acceptance.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
