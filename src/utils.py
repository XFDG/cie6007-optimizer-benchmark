import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def config():
    return yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def source_manifest():
    files = sorted([*ROOT.glob('src/*.py'), *ROOT.glob('scripts/*.py'),
                    ROOT / 'configs/experiment.yaml'])
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def git_commit():
    p = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else 'uncommitted'
