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
