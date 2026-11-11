"""Test split is loaded only by the final evaluation entry point."""
import gzip
import hashlib
from pathlib import Path
import urllib.request
import numpy as np
import torch
from .utils import ROOT, write_json

FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
BASE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'


def download(root=None):
    root = Path(root or ROOT / 'data')
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        if not path.exists():
            urllib.request.urlretrieve(BASE + name, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Invalid dataset checksum: {name}')


def _read_split(prefix, root=None):
    root = Path(root or ROOT / 'data')
    paths = [root / f'{prefix}-{kind}-idx{idx}-ubyte.gz'
             for kind, idx in [('images', 3), ('labels', 1)]]
    for path in paths:
        if hashlib.md5(path.read_bytes()).hexdigest() != FILES[path.name]:
            raise ValueError(f'Invalid dataset checksum: {path.name}')
    x = np.frombuffer(gzip.decompress(paths[0].read_bytes()), np.uint8, offset=16).copy()
    y = np.frombuffer(gzip.decompress(paths[1].read_bytes()), np.uint8, offset=8).copy()
    return torch.from_numpy(x.reshape(-1, 784)).float().div_(255), torch.from_numpy(y).long()


def stratified_indices(labels, split_seed=2026):
    rng = np.random.default_rng(split_seed)
    train, val = [], []
    for cls in range(10):
        indices = np.flatnonzero(np.asarray(labels) == cls)
        rng.shuffle(indices)
        val.extend(indices[:1000])
        train.extend(indices[1000:])
    return np.sort(train), np.sort(val)


def load_development(device, split_seed=2026):
    x, y = _read_split('train')
    train, val = stratified_indices(y.numpy(), split_seed)
    write_json(ROOT / 'results/split_manifest.json', {
        'source': 'official training split only', 'split_seed': split_seed,
        'train_n': len(train), 'val_n': len(val),
        'train_index_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
        'val_index_sha256': hashlib.sha256(val.tobytes()).hexdigest(),
        'normalization': 'fixed pixel/255; no fitted statistics',
        'dataset_md5': FILES})
    return tuple(t.to(device) for t in (x[train], y[train], x[val], y[val]))


def load_test(device):
    return tuple(t.to(device) for t in _read_split('t10k'))
