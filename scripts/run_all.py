"""Sequential jobs; freeze selection before opening test data. Safe run-level resume."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import random
import torch
from src.utils import ROOT, config, write_json, sha256
from src.data import load_development
from src.train import run_training


def run(phase):
    cfg = config()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('CPU fallback: same protocol; use --phase smoke first. No GPU timing claims.')
    data = load_development(device, cfg['split_seed'])
    if phase == 'smoke':
        small = (data[0][:512], data[1][:512], data[2][:256], data[3][:256])
        for name, rates in cfg['learning_rates'].items():
            run_training(cfg, name, rates[1], 11, 'smoke', small, device, 2)
        return
    if phase in ('all', 'tune'):
        candidates = [(n, lr) for n, rates in cfg['learning_rates'].items() for lr in rates]
        random.Random(2026).shuffle(candidates)
        records = [run_training(cfg, n, lr, cfg['tune_seed'], 'tune', data, device,
                                cfg['tune_epochs']) for n, lr in candidates]
        chosen = {}
        for name in cfg['learning_rates']:
            best = sorted([r for r in records if r['optimizer'] == name],
                key=lambda r: (-r['best_val_accuracy'], r['best_val_loss'], r['lr']))[0]
            chosen[name] = dict(lr=best['lr'], tuning_run=best['run_id'],
                best_val_accuracy=best['best_val_accuracy'], best_val_loss=best['best_val_loss'])
        selection = dict(config_sha256=sha256(ROOT/'configs/experiment.yaml'),
                         criterion='max validation accuracy; tie min loss then min lr', selected=chosen)
        write_json(ROOT/'results/selected_config.json', selection)
        print('FROZEN SELECTION', selection, flush=True)
    if phase in ('all', 'final'):
        selection = json.loads((ROOT/'results/selected_config.json').read_text())
        assert selection['config_sha256'] == sha256(ROOT/'configs/experiment.yaml')
        for seed in cfg['seeds']:
            names = list(selection['selected'])
            random.Random(seed).shuffle(names)
            for name in names:
                run_training(cfg, name, selection['selected'][name]['lr'], seed,
                             'final', data, device)
        write_json(ROOT/'results/training_frozen.json', {
            'selected_config_sha256': sha256(ROOT/'results/selected_config.json'),
            'runs': [json.loads(p.read_text())['run_id']
                     for p in sorted((ROOT/'results/runs').glob('final_*/completed.json'))]})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke','tune','final','all'], default='all')
    run(p.parse_args().phase)
