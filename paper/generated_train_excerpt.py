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
