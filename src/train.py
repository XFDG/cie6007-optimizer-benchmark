"""Measured training; test data cannot be accessed from this module."""
import argparse
import copy
import time
import torch
import torch.nn.functional as F
from .model import MLP
from .optimizers import make_optimizer
from .evaluate import evaluate
from .utils import ROOT, seed_all, sync, model_hash, source_manifest, write_json, append_csv, git_commit


def run_training(cfg, name, lr, seed, phase, data, device, epochs=None):
    epochs = epochs or cfg['epochs']
    run_id = f'{phase}_{name}_lr{lr:g}_seed{seed}'
    folder = ROOT / 'results/runs' / run_id
    if (folder / 'completed.json').exists():
        import json
        cached = json.loads((folder / 'completed.json').read_text())
        required = ['best', 'final'] + (['budget'] if cached['budget_available'] else [])
        if all((ROOT / 'checkpoints' / run_id / f'{s}.pt').exists() for s in required):
            if cached['config'] != cfg or cached['epochs'] != epochs:
                raise ValueError('Configuration changed: use a fresh output directory.')
            return cached
        # Public clones contain metrics but no large checkpoint files: regenerate them.
    folder.mkdir(parents=True, exist_ok=True)
    # Never append onto an interrupted partial run.
    (folder / 'metrics.csv').unlink(missing_ok=True)
    x, y, vx, vy = data
    seed_all(seed)
    warm = MLP(cfg['hidden']).to(device)
    warm_opt = make_optimizer(warm, name, lr, cfg['weight_decay'], cfg['aux_lr'])
    sync(device)
    t = time.perf_counter()
    for _ in range(cfg['warmup_steps']):
        warm_opt.zero_grad()
        F.cross_entropy(warm(x[:cfg['batch_size']]), y[:cfg['batch_size']]).backward()
        warm_opt.step()
    sync(device)
    warmup_s = time.perf_counter() - t
    del warm, warm_opt
    seed_all(seed)
    model = MLP(cfg['hidden']).to(device)
    initial_hash = model_hash(model)
    opt = make_optimizer(model, name, lr, cfg['weight_decay'], cfg['aux_lr'])
    generator = torch.Generator(device='cpu').manual_seed(seed + 10000)
    checkpoint_dir = ROOT / 'checkpoints' / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_key = (-1., float('-inf'))
    best_epoch = 0
    cumulative_train = cumulative_wall = 0.
    budget_state = None
    budget_key = (-1., float('-inf'))
    budget_epoch = 0
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for epoch in range(1, epochs + 1):
        # CPU permutation is paired across all treatments, independently of optimizer RNG.
        indices = torch.randperm(len(y), generator=generator).to(device)
        model.train()
        sync(device)
        t = time.perf_counter()
        total_loss = torch.zeros((), device=device)
        for ix in indices.split(cfg['batch_size']):
            opt.zero_grad()
            loss = F.cross_entropy(model(x[ix]), y[ix])
            loss.backward()
            opt.step()
            total_loss += loss.detach() * len(ix)
        sync(device)
        train_s = time.perf_counter() - t
        online_loss = float(total_loss / len(y))
        t = time.perf_counter()
        val_loss, val_acc = evaluate(model, vx, vy)
        sync(device)
        val_s = time.perf_counter() - t
        cumulative_train += train_s
        cumulative_wall += train_s + val_s
        # Extra reporting evaluations/checkpoint I/O are outside the train+validation budget.
        train_loss, train_acc = evaluate(model, x, y)
        key = (val_acc, -val_loss)
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(model.state_dict(), checkpoint_dir / 'best.pt')
        if cumulative_wall <= cfg['time_budget_s'] and key > budget_key:
            budget_key, budget_epoch = key, epoch
            budget_state = copy.deepcopy(model.state_dict())
        row = dict(run_id=run_id, phase=phase, optimizer=name, lr=lr, seed=seed,
            epoch=epoch, step=epoch*((len(y)+cfg['batch_size']-1)//cfg['batch_size']),
            online_train_loss=online_loss, train_loss=train_loss, train_accuracy=train_acc,
            val_loss=val_loss, val_accuracy=val_acc, train_seconds=train_s,
            validation_seconds=val_s, cumulative_train_seconds=cumulative_train,
            cumulative_wall_seconds=cumulative_wall, samples_per_second=len(y)/train_s,
            peak_memory_mb=(torch.cuda.max_memory_allocated(device)/2**20
                            if str(device).startswith('cuda') else 0.),
            initial_model_sha256=initial_hash)
        append_csv(folder / 'metrics.csv', row)
        rows.append(row)
        print(f'{run_id} epoch={epoch}/{epochs} loss={train_loss:.4f} val={val_acc:.4f} train_s={train_s:.3f}', flush=True)
    torch.save(model.state_dict(), checkpoint_dir / 'final.pt')
    if budget_state is not None and cumulative_wall >= cfg['time_budget_s']:
        torch.save(budget_state, checkpoint_dir / 'budget.pt')
    metadata = dict(run_id=run_id, optimizer=name, lr=lr, seed=seed, phase=phase,
        epochs=epochs, best_val_accuracy=best_key[0], best_val_loss=-best_key[1],
        best_epoch=best_epoch, budget_epoch=budget_epoch,
        budget_available=budget_state is not None and cumulative_wall >= cfg['time_budget_s'],
        initial_model_sha256=initial_hash, warmup_seconds=warmup_s,
        train_seconds=cumulative_train, wall_seconds=cumulative_wall,
        config=cfg, git_commit=git_commit(), source_sha256=source_manifest())
    write_json(folder / 'completed.json', metadata)
    return metadata


if __name__ == '__main__':
    from .utils import config
    from .data import load_development
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', choices=['sgd','adam','adamw','muon'], required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--epochs', type=int, default=15)
    a = p.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg = config()
    run_training(cfg, a.optimizer, a.lr, a.seed, 'single', load_development(device), device, a.epochs)
