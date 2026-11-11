"""Only this script reads official test images. Frozen checkpoints only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import torch
from src.utils import ROOT, config, sha256, write_json, seed_all
from src.data import load_development, load_test
from src.model import MLP
from src.evaluate import evaluate


def main():
    frozen = json.loads((ROOT/'results/training_frozen.json').read_text())
    assert frozen['selected_config_sha256'] == sha256(ROOT/'results/selected_config.json')
    cfg = config()
    assert len(frozen['runs']) == 4*len(cfg['seeds'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_all(0)
    x,y,vx,vy = load_development(device,cfg['split_seed'])
    tx,ty = load_test(device)
    results = []
    for run_id in frozen['runs']:
        meta = json.loads((ROOT/'results/runs'/run_id/'completed.json').read_text())
        for selection in ['best','final','budget']:
            path = ROOT/'checkpoints'/run_id/f'{selection}.pt'
            if not path.exists():
                if selection != 'budget' or meta['budget_available']:
                    raise FileNotFoundError(f'Reproduce training first: {path}')
                continue
            model = MLP(cfg['hidden']).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            trl,tra = evaluate(model,x,y)
            vl,va = evaluate(model,vx,vy)
            tl,ta = evaluate(model,tx,ty)
            results.append(dict(run_id=run_id,optimizer=meta['optimizer'],seed=meta['seed'],
                lr=meta['lr'],selection=selection,epoch=(meta['best_epoch'] if selection=='best'
                    else meta['epochs'] if selection=='final' else meta['budget_epoch']),
                train_loss=trl,train_accuracy=tra,val_loss=vl,val_accuracy=va,
                test_loss=tl,test_accuracy=ta,generalization_gap=tra-ta,
                checkpoint_sha256=sha256(path)))
    pd.DataFrame(results).to_csv(ROOT/'results/final_evaluations.csv',index=False)
    write_json(ROOT/'results/evaluation_manifest.json', {
        'training_frozen_sha256':sha256(ROOT/'results/training_frozen.json'),
        'evaluations_sha256':sha256(ROOT/'results/final_evaluations.csv'),
        'note':'Test metrics never used for selection or tuning; final/best/budget predeclared.'})
    print(pd.DataFrame(results).query("selection == 'best'").to_string(index=False))

if __name__=='__main__':
    main()
