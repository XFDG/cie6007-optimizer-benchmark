"""Use unmodified PyTorch 2.11 optimizers. Muon is a hybrid treatment."""
import torch


def parameter_routes(model):
    hidden = [(n, p) for n, p in model.named_parameters()
              if n.startswith('body.') and p.ndim == 2]
    hidden_ids = {id(p) for _, p in hidden}
    auxiliary = [(n, p) for n, p in model.named_parameters() if id(p) not in hidden_ids]
    all_ids = [id(p) for _, p in hidden + auxiliary]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    return hidden, auxiliary


def decay_groups(named, decay):
    return [dict(params=[p for _, p in named if p.ndim >= 2], weight_decay=decay),
            dict(params=[p for _, p in named if p.ndim < 2], weight_decay=0.)]


class OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


def make_optimizer(model, name, lr, weight_decay=1e-4, aux_lr=1e-3):
    if name == 'muon':
        hidden, auxiliary = parameter_routes(model)
        muon = torch.optim.Muon([p for _, p in hidden], lr=lr,
            weight_decay=weight_decay, momentum=.95, nesterov=True,
            ns_steps=5, adjust_lr_fn='original')
        adamw = torch.optim.AdamW(decay_groups(auxiliary, weight_decay), lr=aux_lr,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
        return OptimizerBundle([muon, adamw])
    groups = decay_groups(list(model.named_parameters()), weight_decay)
    if name == 'sgd':
        opt = torch.optim.SGD(groups, lr=lr, momentum=.9, foreach=False)
    elif name == 'adam':
        opt = torch.optim.Adam(groups, lr=lr, betas=(.9, .999), eps=1e-8,
                               foreach=False, decoupled_weight_decay=False)
    elif name == 'adamw':
        opt = torch.optim.AdamW(groups, lr=lr, betas=(.9, .999), eps=1e-8, foreach=False)
    else:
        raise ValueError(name)
    return OptimizerBundle([opt])
