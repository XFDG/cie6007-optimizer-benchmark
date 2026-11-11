import csv
import inspect
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from src.model import MLP
from src.optimizers import make_optimizer, parameter_routes
from src.utils import seed_all, append_csv
from src.data import stratified_indices
from src import train

DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('name', ['sgd','adam','adamw','muon'])
def test_updates_and_repeatability(name,device):
    def run():
        seed_all(42)
        m=MLP(16).to(device)
        original={n:p.detach().clone() for n,p in m.named_parameters()}
        x=torch.rand(16,784,device=device)
        y=torch.arange(16,device=device)%10
        opt=make_optimizer(m,name,.01)
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(m(x),y).backward()
            opt.step()
        for n,p in m.named_parameters():
            assert torch.isfinite(p).all()
            assert not torch.equal(p,original[n])
            assert p.dtype==torch.float32 and p.device==x.device
        for o in opt.optimizers:
            for p,state in o.state.items():
                for key,v in state.items():
                    if torch.is_tensor(v) and key!='step':
                        assert v.device==p.device
        return torch.cat([p.detach().flatten() for p in m.parameters()])
    assert torch.equal(run(),run())


def test_muon_routes():
    m=MLP()
    hidden,aux=parameter_routes(m)
    assert [n for n,_ in hidden]==['body.0.weight','body.2.weight']
    ids=[id(p) for _,p in hidden+aux]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert all(p.ndim==2 for _,p in hidden)
    opt=make_optimizer(m,'muon',.02)
    actual=[id(p) for o in opt.optimizers for g in o.param_groups for p in g['params']]
    assert sorted(actual)==sorted(ids)


def test_no_split_leakage():
    labels=np.repeat(np.arange(10),6000)
    train_idx,val=stratified_indices(labels)
    assert len(train_idx)==50000 and len(val)==10000
    assert not np.intersect1d(train_idx,val).size
    assert len(np.union1d(train_idx,val))==60000
    for cls in range(10):
        assert np.sum(labels[train_idx]==cls)==5000
        assert np.sum(labels[val]==cls)==1000
    assert 'load_test' not in inspect.getsource(train)


def test_csv_schema(tmp_path):
    p=tmp_path/'metrics.csv'
    row={'optimizer':'adam','seed':101,'epoch':1,'val_accuracy':.8,'train_seconds':1.}
    append_csv(p,row)
    append_csv(p,row)
    with p.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and set(rows[0])==set(row)


def test_adam_decay_is_not_adamw():
    seed_all(42)
    a,b=MLP(16),MLP(16)
    b.load_state_dict(a.state_dict())
    for m in (a,b):
        for p in m.parameters(): p.grad=torch.ones_like(p)*.01
    make_optimizer(a,'adam',.01,weight_decay=.1).step()
    make_optimizer(b,'adamw',.01,weight_decay=.1).step()
    assert not torch.equal(a.body[0].weight,b.body[0].weight)


def test_resume_regenerates_missing_checkpoints(tmp_path, monkeypatch):
    # A public clone includes metrics but excludes checkpoints. Reproduction must rerun.
    from src.utils import config
    monkeypatch.setattr(train, 'ROOT', tmp_path)
    cfg=config()
    cfg.update(hidden=16, warmup_steps=1, batch_size=8)
    seed_all(3)
    x=torch.rand(16,784); y=torch.arange(16)%10
    data=(x,y,x[:8],y[:8])
    metadata=train.run_training(cfg,'adam',.001,3,'smoke',data,'cpu',1)
    path=tmp_path/'checkpoints'/metadata['run_id']/'final.pt'
    assert path.exists()
    path.unlink()
    repeated=train.run_training(cfg,'adam',.001,3,'smoke',data,'cpu',1)
    assert path.exists()
    assert repeated['initial_model_sha256']==metadata['initial_model_sha256']
    with pytest.raises(ValueError,match='Configuration changed'):
        train.run_training(dict(cfg,hidden=32),'adam',.001,3,'smoke',data,'cpu',1)
